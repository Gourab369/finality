// Serves VS Code's language features to the Finality agent as a local MCP server (streamable HTTP, JSON replies).
// Finality finds it through the lock file `<finality data dir>/ide/<port>.json`, which holds a random bearer token.

import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';

const PROTOCOL_VERSIONS = ['2025-06-18', '2025-03-26', '2024-11-05'];
const MAX_BODY_BYTES = 1024 * 1024;
const MAX_DIAGNOSTICS = 200;
const MAX_LOCATIONS = 100;
const MAX_SYMBOLS = 300;
const MAX_WORKSPACE_SYMBOLS = 50;
const MAX_HOVER_CHARS = 4000;
const MAX_LINE_CHARS = 200;
const MAX_PREVIEW_FILE_BYTES = 2 * 1024 * 1024;
const QUIET_MS = 700;
const DEFAULT_WAIT_MS = 3000;
const RECHECK_WAIT_MS = 20000;
const MAX_WAIT_MS = 120000;

let server: http.Server | undefined;
let lockFile: string | undefined;
let token = '';
let output: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
    output = vscode.window.createOutputChannel('Finality Bridge');
    token = crypto.randomBytes(32).toString('hex');
    server = http.createServer((req, res) => {
        handle(req, res).catch((e) => {
            output.appendLine(`request failed: ${e}`);
            if (!res.headersSent) {
                res.writeHead(500);
            }
            res.end();
        });
    });
    server.on('error', (e) => output.appendLine(`server error: ${e}`));
    server.listen(0, '127.0.0.1', () => {
        writeLock();
        output.appendLine(`listening on 127.0.0.1:${port()}`);
    });
    context.subscriptions.push(
        output,
        vscode.workspace.onDidChangeWorkspaceFolders(() => writeLock()),
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('finality.dataDir')) {
                removeLock();
                writeLock();
            }
        }),
        vscode.commands.registerCommand('finality.bridgeStatus', () => {
            const text = server?.listening
                ? `Finality bridge on 127.0.0.1:${port()} for ${folders().join(', ') || 'no folder'} (lock file ${lockFile ?? 'not written'})`
                : 'The Finality bridge is not running.';
            void vscode.window.showInformationMessage(text);
        }),
        { dispose: stop },
    );
}

export function deactivate(): void {
    stop();
}

// ---------------------------------------------------------------------------------------------- lock file

function port(): number {
    const address = server?.address();
    return typeof address === 'object' && address ? address.port : 0;
}

function folders(): string[] {
    return (vscode.workspace.workspaceFolders ?? [])
        .filter((f) => f.uri.scheme === 'file')
        .map((f) => f.uri.fsPath);
}

function dataDir(): string {
    const configured = vscode.workspace.getConfiguration('finality').get<string>('dataDir')?.trim();
    if (configured) {
        return configured;
    }
    const home = process.env.FINALITY_HOME?.trim();
    return home || path.join(os.homedir(), '.finality');
}

function writeLock(): void {
    if (!server?.listening) {
        return;
    }
    const dir = path.join(dataDir(), 'ide');
    try {
        fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
        const file = path.join(dir, `${port()}.json`);
        const body = {
            port: port(),
            token,
            pid: process.pid,
            ideName: vscode.env.appName,
            workspaceFolders: folders(),
        };
        fs.writeFileSync(file, JSON.stringify(body, null, 2), { mode: 0o600 });
        lockFile = file;
    } catch (e) {
        output.appendLine(`cannot write the lock file in ${dir}: ${e}`);
    }
}

function removeLock(): void {
    if (lockFile) {
        try {
            fs.unlinkSync(lockFile);
        } catch {
            // Already gone.
        }
        lockFile = undefined;
    }
}

function stop(): void {
    removeLock();
    server?.close();
    server = undefined;
}

// ---------------------------------------------------------------------------------------------- HTTP and JSON-RPC

interface JsonRpc {
    jsonrpc?: string;
    id?: string | number | null;
    method?: string;
    params?: any;
}

function reply(res: http.ServerResponse, status: number, body?: unknown): void {
    if (body === undefined) {
        res.writeHead(status);
        res.end();
        return;
    }
    const text = JSON.stringify(body);
    res.writeHead(status, {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(text),
    });
    res.end(text);
}

function authorized(header: string | undefined): boolean {
    const expected = Buffer.from(`Bearer ${token}`);
    const got = Buffer.from(header ?? '');
    return got.length === expected.length && crypto.timingSafeEqual(got, expected);
}

function readBody(req: http.IncomingMessage): Promise<string | undefined> {
    return new Promise((resolve, reject) => {
        const chunks: Buffer[] = [];
        let size = 0;
        req.on('data', (chunk: Buffer) => {
            size += chunk.length;
            if (size <= MAX_BODY_BYTES) {
                chunks.push(chunk);
            }
        });
        req.on('end', () => resolve(size > MAX_BODY_BYTES ? undefined : Buffer.concat(chunks).toString('utf8')));
        req.on('error', reject);
    });
}

async function handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    // Web pages send Origin and may reach 127.0.0.1 through DNS rebinding (a foreign Host); refuse both.
    const host = req.headers.host ?? '';
    if (req.headers.origin !== undefined || (host !== `127.0.0.1:${port()}` && host !== `localhost:${port()}`)) {
        return reply(res, 403);
    }
    if (!authorized(req.headers.authorization)) {
        return reply(res, 401);
    }
    if ((req.url ?? '').split('?')[0] !== '/mcp') {
        return reply(res, 404);
    }
    if (req.method !== 'POST') {
        // No server-initiated stream.
        res.writeHead(405, { Allow: 'POST' });
        res.end();
        return;
    }
    const body = await readBody(req);
    if (body === undefined) {
        return reply(res, 413);
    }
    let message: unknown;
    try {
        message = JSON.parse(body);
    } catch {
        return reply(res, 400, { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } });
    }
    const root = workdirRoot(req.headers['x-finality-workdir']);
    const batch = Array.isArray(message);
    const items = (batch ? message : [message]) as JsonRpc[];
    const replies: unknown[] = [];
    for (const m of items) {
        if (m && typeof m === 'object' && m.id !== undefined && m.id !== null && typeof m.method === 'string') {
            replies.push(await dispatch(m, root));
        }
    }
    if (replies.length === 0) {
        return reply(res, 202);
    }
    reply(res, 200, batch ? replies : replies[0]);
}

async function dispatch(m: JsonRpc, root: Root): Promise<unknown> {
    const ok = (result: unknown) => ({ jsonrpc: '2.0', id: m.id, result });
    const fail = (code: number, message: string) => ({ jsonrpc: '2.0', id: m.id, error: { code, message } });
    switch (m.method) {
        case 'initialize': {
            const requested = m.params?.protocolVersion;
            return ok({
                protocolVersion: PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0],
                capabilities: { tools: {} },
                serverInfo: { name: 'finality-vscode', version: version() },
                instructions:
                    "Language features of the user's VS Code window: compiler and linter diagnostics, definitions, " +
                    'references, implementations, symbols and hover text. Paths are relative to the working directory; ' +
                    'lines are 1-based; name the symbol on the line to point at it.',
            });
        }
        case 'ping':
            return ok({});
        case 'tools/list':
            return ok({ tools: TOOLS });
        case 'tools/call': {
            const name = String(m.params?.name ?? '');
            const run = HANDLERS[name];
            if (!run) {
                return fail(-32602, `unknown tool ${name}`);
            }
            try {
                const text = await run(m.params?.arguments ?? {}, root);
                return ok({ content: [{ type: 'text', text }] });
            } catch (e) {
                const text = e instanceof Error ? e.message : String(e);
                return ok({ content: [{ type: 'text', text }], isError: true });
            }
        }
        default:
            return fail(-32601, `method not found: ${m.method}`);
    }
}

function version(): string {
    return vscode.extensions.getExtension('finality.finality-bridge')?.packageJSON?.version ?? '0';
}

// ---------------------------------------------------------------------------------------------- paths

/** The agent's working directory, which must be inside one of this window's folders. */
interface Root {
    dir?: string;
    error?: string;
}

function normalize(p: string): string {
    const resolved = path.resolve(p);
    return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
}

function contains(folder: string, p: string): boolean {
    const f = normalize(folder);
    const q = normalize(p);
    return q === f || q.startsWith(f.endsWith(path.sep) ? f : f + path.sep);
}

function workdirRoot(header: string | string[] | undefined): Root {
    const open = folders();
    if (open.length === 0) {
        return { error: 'no folder is open in this VS Code window' };
    }
    if (typeof header !== 'string' || header === '') {
        return { dir: open[0] };
    }
    const dir = vscode.Uri.parse(header).fsPath;
    if (!open.some((f) => contains(f, dir))) {
        return { error: `this VS Code window has ${open.join(', ')} open, not ${dir}` };
    }
    return { dir };
}

function need(root: Root): string {
    if (!root.dir) {
        throw new Error(root.error ?? 'no working directory');
    }
    return root.dir;
}

function resolveFile(root: Root, raw: unknown): vscode.Uri {
    const dir = need(root);
    if (typeof raw !== 'string' || raw.trim() === '') {
        throw new Error('`path` is required');
    }
    const full = path.resolve(dir, raw);
    if (!contains(dir, full)) {
        throw new Error(`${raw} is outside the working directory`);
    }
    return vscode.Uri.file(full);
}

function display(root: Root, uri: vscode.Uri): string {
    if (uri.scheme === 'file' && root.dir && contains(root.dir, uri.fsPath)) {
        return path.relative(root.dir, uri.fsPath).split(path.sep).join('/') || '.';
    }
    return uri.scheme === 'file' ? uri.fsPath : uri.toString();
}

async function fileLines(uri: vscode.Uri, cache: Map<string, string[] | undefined>): Promise<string[] | undefined> {
    const key = uri.toString();
    if (cache.has(key)) {
        return cache.get(key);
    }
    let lines: string[] | undefined;
    const open = vscode.workspace.textDocuments.find((d) => d.uri.toString() === key);
    if (open) {
        lines = open.getText().split(/\r?\n/);
    } else if (uri.scheme === 'file') {
        try {
            const stat = await vscode.workspace.fs.stat(uri);
            if (stat.size <= MAX_PREVIEW_FILE_BYTES) {
                lines = Buffer.from(await vscode.workspace.fs.readFile(uri)).toString('utf8').split(/\r?\n/);
            }
        } catch {
            lines = undefined;
        }
    }
    cache.set(key, lines);
    return lines;
}

function clip(text: string, max: number): string {
    return text.length > max ? `${text.slice(0, max)}…` : text;
}

function int(value: unknown, name: string): number {
    const n = typeof value === 'string' ? Number(value) : value;
    if (typeof n !== 'number' || !Number.isInteger(n)) {
        throw new Error(`\`${name}\` must be a whole number`);
    }
    return n;
}

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Opens the document (so its language server analyzes it) and finds the position: `symbol` on `line`, else `column`. */
async function position(uri: vscode.Uri, args: any): Promise<vscode.Position> {
    const doc = await vscode.workspace.openTextDocument(uri);
    const line = int(args.line, 'line');
    if (line < 1 || line > doc.lineCount) {
        throw new Error(`line ${line} is out of range 1..${doc.lineCount}`);
    }
    const text = doc.lineAt(line - 1).text;
    if (typeof args.symbol === 'string' && args.symbol !== '') {
        const word = new RegExp(`(?<![\\w$])${escapeRegExp(args.symbol)}(?![\\w$])`).exec(text);
        const at = word ? word.index : text.indexOf(args.symbol);
        if (at < 0) {
            throw new Error(`\`${args.symbol}\` is not on line ${line}: ${clip(text.trim(), MAX_LINE_CHARS)}`);
        }
        return new vscode.Position(line - 1, at);
    }
    if (args.column !== undefined) {
        return new vscode.Position(line - 1, Math.max(0, int(args.column, 'column') - 1));
    }
    return new vscode.Position(line - 1, text.length - text.trimStart().length);
}

// ---------------------------------------------------------------------------------------------- tools

const POSITION_PROPS = {
    path: { type: 'string', description: 'File path, relative to the working directory' },
    line: { type: 'integer', minimum: 1 },
    symbol: { type: 'string', description: 'The identifier on that line to point at' },
    column: { type: 'integer', minimum: 1, description: 'Used when symbol is not given' },
};

const READ_ONLY = { readOnlyHint: true, destructiveHint: false, openWorldHint: false };

const TOOLS = [
    {
        name: 'diagnostics',
        description:
            "Errors and warnings from the editor's language servers and linters, for one file or the whole working " +
            'directory. Files changed on disk are re-read by the editor; give it a moment (wait_ms). recheck: true ' +
            "also re-runs rust-analyzer's cargo check (slower).",
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'File to check; omit for every file with diagnostics' },
                severity: { type: 'string', enum: ['error', 'warning', 'all'], default: 'warning' },
                wait_ms: { type: 'integer', minimum: 0, maximum: MAX_WAIT_MS },
                recheck: { type: 'boolean' },
            },
        },
        annotations: READ_ONLY,
    },
    {
        name: 'definition',
        description: 'Where the symbol at a position is defined.',
        inputSchema: { type: 'object', properties: POSITION_PROPS, required: ['path', 'line'] },
        annotations: READ_ONLY,
    },
    {
        name: 'references',
        description: 'Every reference to the symbol at a position, including its declaration.',
        inputSchema: { type: 'object', properties: POSITION_PROPS, required: ['path', 'line'] },
        annotations: READ_ONLY,
    },
    {
        name: 'implementations',
        description: 'Implementations of the interface, trait or abstract method at a position.',
        inputSchema: { type: 'object', properties: POSITION_PROPS, required: ['path', 'line'] },
        annotations: READ_ONLY,
    },
    {
        name: 'hover',
        description: 'Type and documentation of the symbol at a position.',
        inputSchema: { type: 'object', properties: POSITION_PROPS, required: ['path', 'line'] },
        annotations: READ_ONLY,
    },
    {
        name: 'symbols',
        description: "Outline of a file: its classes, functions, fields and so on, with line ranges.",
        inputSchema: {
            type: 'object',
            properties: { path: POSITION_PROPS.path },
            required: ['path'],
        },
        annotations: READ_ONLY,
    },
    {
        name: 'workspace_symbols',
        description: 'Find symbols by name across the working directory.',
        inputSchema: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
        },
        annotations: READ_ONLY,
    },
];

type Handler = (args: any, root: Root) => Promise<string>;

const HANDLERS: Record<string, Handler> = {
    diagnostics,
    definition: (a, r) => locations('vscode.executeDefinitionProvider', a, r, 'definition'),
    references: (a, r) => locations('vscode.executeReferenceProvider', a, r, 'references'),
    implementations: (a, r) => locations('vscode.executeImplementationProvider', a, r, 'implementations'),
    hover,
    symbols,
    workspace_symbols: workspaceSymbols,
};

/** Resolves after diagnostics stop changing for QUIET_MS, or at maxMs. With `expectChange`, waits for a first change. */
function settle(maxMs: number, expectChange: boolean): Promise<void> {
    return new Promise((resolve) => {
        let quiet: NodeJS.Timeout | undefined;
        const done = () => {
            sub.dispose();
            clearTimeout(quiet);
            clearTimeout(cap);
            resolve();
        };
        const arm = (ms: number) => {
            clearTimeout(quiet);
            quiet = setTimeout(done, ms);
        };
        const sub = vscode.languages.onDidChangeDiagnostics(() => arm(QUIET_MS));
        const cap = setTimeout(done, maxMs);
        if (!expectChange) {
            arm(Math.min(QUIET_MS * 2, maxMs));
        }
    });
}

const SEVERITY = ['error', 'warning', 'info', 'hint'];

async function diagnostics(args: any, root: Root): Promise<string> {
    const dir = need(root);
    const notes: string[] = [];
    const target = args.path ? resolveFile(root, args.path) : undefined;
    if (target) {
        const doc = await vscode.workspace.openTextDocument(target);
        if (doc.isDirty) {
            notes.push(`${display(root, target)} has unsaved changes in the editor; these diagnostics are for that version`);
        }
    }
    let rechecked = false;
    if (args.recheck === true && vscode.extensions.getExtension('rust-lang.rust-analyzer')?.isActive) {
        try {
            await vscode.commands.executeCommand('rust-analyzer.runFlycheck');
            rechecked = true;
        } catch (e) {
            notes.push(`could not re-run rust-analyzer's check: ${e}`);
        }
    }
    const wait = Math.min(
        Math.max(args.wait_ms === undefined ? (rechecked ? RECHECK_WAIT_MS : DEFAULT_WAIT_MS) : int(args.wait_ms, 'wait_ms'), 0),
        MAX_WAIT_MS,
    );
    await settle(wait, rechecked);

    const worst = args.severity === 'error' ? 0 : args.severity === 'all' ? 3 : 1;
    const entries: [vscode.Uri, readonly vscode.Diagnostic[]][] = target
        ? [[target, vscode.languages.getDiagnostics(target)]]
        : vscode.languages.getDiagnostics();
    const found: { severity: number; where: string; line: number; text: string }[] = [];
    for (const [uri, list] of entries) {
        if (uri.scheme !== 'file' || !contains(dir, uri.fsPath)) {
            continue;
        }
        const where = display(root, uri);
        for (const d of list) {
            if (d.severity > worst) {
                continue;
            }
            const code = typeof d.code === 'object' ? d.code.value : d.code;
            const source = [d.source, code].filter((x) => x !== undefined && x !== '').join(' ');
            const message = d.message.split(/\r?\n/).slice(0, 4).join('\n    ');
            found.push({
                severity: d.severity,
                where,
                line: d.range.start.line,
                text: `${where}:${d.range.start.line + 1}:${d.range.start.character + 1}: ${SEVERITY[d.severity]}${source ? ` [${source}]` : ''}: ${message}`,
            });
        }
    }
    found.sort((a, b) => a.severity - b.severity || a.where.localeCompare(b.where) || a.line - b.line);
    const errors = found.filter((f) => f.severity === 0).length;
    const scope = target ? display(root, target) : 'the working directory';
    let text =
        found.length === 0
            ? `no ${worst === 0 ? 'errors' : worst === 1 ? 'errors or warnings' : 'diagnostics'} in ${scope}`
            : `${errors} error(s), ${found.length - errors} other(s) in ${scope}:\n` +
              found
                  .slice(0, MAX_DIAGNOSTICS)
                  .map((f) => f.text)
                  .join('\n');
    if (found.length > MAX_DIAGNOSTICS) {
        text += `\n… ${found.length - MAX_DIAGNOSTICS} more`;
    }
    if (notes.length > 0) {
        text += `\n(${notes.join('; ')})`;
    }
    return text;
}

async function locations(command: string, args: any, root: Root, what: string): Promise<string> {
    const uri = resolveFile(root, args.path);
    const pos = await position(uri, args);
    const result = (await vscode.commands.executeCommand<(vscode.Location | vscode.LocationLink)[]>(command, uri, pos)) ?? [];
    const found = result.map((l) =>
        'targetUri' in l ? { uri: l.targetUri, range: l.targetSelectionRange ?? l.targetRange } : { uri: l.uri, range: l.range },
    );
    if (found.length === 0) {
        return `no ${what} found`;
    }
    const dir = need(root);
    const cache = new Map<string, string[] | undefined>();
    const lines: string[] = [];
    for (const f of found.slice(0, MAX_LOCATIONS)) {
        // Line text only from inside the working directory, as Finality's own read tools would ask first.
        const inside = f.uri.scheme === 'file' && contains(dir, f.uri.fsPath);
        const text = inside ? ((await fileLines(f.uri, cache))?.[f.range.start.line]?.trim() ?? '') : '';
        lines.push(`${display(root, f.uri)}:${f.range.start.line + 1}: ${clip(text, MAX_LINE_CHARS)}`.trimEnd());
    }
    if (found.length > MAX_LOCATIONS) {
        lines.push(`… ${found.length - MAX_LOCATIONS} more`);
    }
    return lines.join('\n');
}

async function hover(args: any, root: Root): Promise<string> {
    const uri = resolveFile(root, args.path);
    const pos = await position(uri, args);
    const hovers = (await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider', uri, pos)) ?? [];
    const text = hovers
        .flatMap((h) => h.contents.map((c) => (typeof c === 'string' ? c : c.value)))
        .filter((t) => t.trim() !== '')
        .join('\n---\n');
    return text === '' ? 'no hover information' : clip(text, MAX_HOVER_CHARS);
}

function kindName(kind: vscode.SymbolKind): string {
    return (vscode.SymbolKind[kind] ?? 'symbol').toLowerCase();
}

async function symbols(args: any, root: Root): Promise<string> {
    const uri = resolveFile(root, args.path);
    await vscode.workspace.openTextDocument(uri);
    const result =
        (await vscode.commands.executeCommand<(vscode.DocumentSymbol | vscode.SymbolInformation)[]>(
            'vscode.executeDocumentSymbolProvider',
            uri,
        )) ?? [];
    if (result.length === 0) {
        return `no symbols in ${display(root, uri)} (is a language extension for this file type installed?)`;
    }
    const lines: string[] = [];
    const walk = (s: vscode.DocumentSymbol | vscode.SymbolInformation, depth: number) => {
        if (lines.length >= MAX_SYMBOLS) {
            return;
        }
        if ('children' in s) {
            const detail = s.detail ? ` — ${clip(s.detail, 80)}` : '';
            lines.push(
                `${'  '.repeat(depth)}${kindName(s.kind)} ${s.name}${detail}  (lines ${s.range.start.line + 1}-${s.range.end.line + 1})`,
            );
            s.children.forEach((c) => walk(c, depth + 1));
        } else {
            const container = s.containerName ? ` in ${s.containerName}` : '';
            lines.push(`${kindName(s.kind)} ${s.name}${container}  (line ${s.location.range.start.line + 1})`);
        }
    };
    result.forEach((s) => walk(s, 0));
    if (lines.length >= MAX_SYMBOLS) {
        lines.push('… (cut)');
    }
    return lines.join('\n');
}

async function workspaceSymbols(args: any, root: Root): Promise<string> {
    const dir = need(root);
    if (typeof args.query !== 'string' || args.query.trim() === '') {
        throw new Error('`query` is required');
    }
    const result =
        (await vscode.commands.executeCommand<vscode.SymbolInformation[]>('vscode.executeWorkspaceSymbolProvider', args.query)) ??
        [];
    const inside = result.filter((s) => s.location.uri.scheme === 'file' && contains(dir, s.location.uri.fsPath));
    if (inside.length === 0) {
        return `no symbols matching \`${args.query}\` in the working directory`;
    }
    const lines = inside.slice(0, MAX_WORKSPACE_SYMBOLS).map((s) => {
        const container = s.containerName ? ` in ${s.containerName}` : '';
        return `${kindName(s.kind)} ${s.name}${container}  ${display(root, s.location.uri)}:${s.location.range.start.line + 1}`;
    });
    if (inside.length > MAX_WORKSPACE_SYMBOLS) {
        lines.push(`… ${inside.length - MAX_WORKSPACE_SYMBOLS} more`);
    }
    return lines.join('\n');
}
