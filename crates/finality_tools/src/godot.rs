//! `godot`: GDScript diagnostics, symbols, definitions and hover text from the Godot editor's built-in language
//! server (TCP on 127.0.0.1, port 6005 by default). Each call opens its own connection while the editor runs.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use finality_core::RiskClass;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, display_path, parse_args,
    resolve_path, truncate_middle,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_WAIT_MS: u64 = 3000;
const MAX_WAIT_MS: u64 = 30_000;
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_LINE_CHARS: usize = 200;
const MAX_SYMBOLS: usize = 300;
const SYMBOL_KINDS: [&str; 26] = [
    "file",
    "module",
    "namespace",
    "package",
    "class",
    "method",
    "property",
    "field",
    "constructor",
    "enum",
    "interface",
    "function",
    "variable",
    "constant",
    "string",
    "number",
    "boolean",
    "array",
    "object",
    "key",
    "null",
    "enum member",
    "struct",
    "event",
    "operator",
    "type parameter",
];

/// The folder with `project.godot`: the working directory, one of its parents, or one of its subfolders.
pub fn find_project(workdir: &Path) -> Option<PathBuf> {
    if let Some(dir) = workdir
        .ancestors()
        .find(|d| d.join("project.godot").is_file())
    {
        return Some(dir.to_owned());
    }
    std::fs::read_dir(workdir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|d| d.join("project.godot").is_file())
}

fn file_uri(path: &Path) -> Result<String, ToolError> {
    url::Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|()| ToolError::Failed(format!("{} is not an absolute path", path.display())))
}

/// Comparable form of a `file:` URI, whatever escaping and drive-letter case the server uses.
fn uri_key(uri: &str) -> String {
    let path = url::Url::parse(uri)
        .ok()
        .and_then(|u| u.to_file_path().ok())
        .map_or_else(
            || uri.to_owned(),
            |p| p.to_string_lossy().replace('\\', "/"),
        );
    if cfg!(windows) {
        path.to_lowercase()
    } else {
        path
    }
}

fn uri_path(uri: &str) -> Option<PathBuf> {
    url::Url::parse(uri).ok()?.to_file_path().ok()
}

/// One connection to the language server.
struct Session {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
    next_id: i64,
    /// `publishDiagnostics` received so far, by `uri_key`.
    diagnostics: Vec<(String, Value)>,
}

impl Session {
    async fn open(port: u16, root: &Path) -> Result<Self, ToolError> {
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(("127.0.0.1", port)))
            .await
            .ok()
            .and_then(Result::ok)
            .ok_or_else(|| {
                ToolError::Failed(format!(
                    "Godot's language server is not reachable on 127.0.0.1:{port}. Open the project in the Godot \
                     editor; the port is under Editor Settings > Network > Language Server (godot.lsp_port here)."
                ))
            })?;
        let (read, writer) = stream.into_split();
        let mut s = Self {
            reader: BufReader::new(read),
            writer,
            next_id: 1,
            diagnostics: Vec::new(),
        };
        let root_uri = file_uri(root)?;
        s.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "rootPath": root.to_string_lossy(),
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {},
                        "documentSymbol": { "hierarchicalDocumentSymbolSupport": true },
                        "definition": {},
                        "hover": { "contentFormat": ["plaintext", "markdown"] }
                    }
                },
                "clientInfo": { "name": finality_core::APP_NAME }
            }),
        )
        .await?;
        s.notify("initialized", json!({})).await?;
        Ok(s)
    }

    async fn send(&mut self, msg: &Value) -> Result<(), ToolError> {
        let body = msg.to_string();
        let frame = format!("Content-Length: {}\r\n\r\n{body}", body.len());
        self.writer.write_all(frame.as_bytes()).await?;
        Ok(())
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), ToolError> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await
    }

    async fn read(&mut self) -> Result<Value, ToolError> {
        let mut length: Option<usize> = None;
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line).await? == 0 {
                return Err(ToolError::Failed(
                    "Godot closed the language server connection".into(),
                ));
            }
            let line = line.trim_end();
            if line.is_empty() {
                if length.is_some() {
                    break;
                }
                continue;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().ok();
            }
        }
        let length = length.unwrap_or(0);
        if length > MAX_MESSAGE_BYTES {
            return Err(ToolError::Failed(
                "oversized language server message".into(),
            ));
        }
        let mut body = vec![0; length];
        self.reader.read_exact(&mut body).await?;
        serde_json::from_slice(&body).map_err(|e| ToolError::Failed(format!("bad message: {e}")))
    }

    /// Keeps diagnostics and answers requests from the server (which this client does not serve).
    async fn absorb(&mut self, msg: Value) -> Result<(), ToolError> {
        match msg.get("method").and_then(Value::as_str) {
            Some("textDocument/publishDiagnostics") => {
                let uri = msg["params"]["uri"].as_str().unwrap_or("");
                self.diagnostics
                    .push((uri_key(uri), msg["params"]["diagnostics"].clone()));
            }
            Some(_) if msg.get("id").is_some() => {
                let id = msg["id"].clone();
                self.send(&json!({ "jsonrpc": "2.0", "id": id, "result": null }))
                    .await?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, ToolError> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .await?;
        let deadline = tokio::time::Instant::now() + REQUEST_TIMEOUT;
        loop {
            let msg = tokio::time::timeout_at(deadline, self.read())
                .await
                .map_err(|_| ToolError::Failed(format!("Godot did not answer {method}")))??;
            if msg.get("id") == Some(&json!(id)) && msg.get("method").is_none() {
                if let Some(e) = msg.get("error") {
                    return Err(ToolError::Failed(format!(
                        "Godot: {}",
                        e["message"].as_str().unwrap_or("request failed")
                    )));
                }
                return Ok(msg["result"].clone());
            }
            self.absorb(msg).await?;
        }
    }

    async fn wait_diagnostics(&mut self, key: &str, wait: Duration) -> Option<Value> {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            if let Some(i) = self.diagnostics.iter().rposition(|(k, _)| k == key) {
                return Some(self.diagnostics.swap_remove(i).1);
            }
            let msg = tokio::time::timeout_at(deadline, self.read())
                .await
                .ok()?
                .ok()?;
            self.absorb(msg).await.ok()?;
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    op: String,
    path: String,
    line: Option<u32>,
    symbol: Option<String>,
    column: Option<u32>,
    wait_ms: Option<u64>,
}

/// LSP position (0-based line, UTF-16 column): `symbol` on `line`, else `column`, else the first non-blank.
fn position(text: &str, a: &Args) -> Result<Value, ToolError> {
    let line_no = a
        .line
        .ok_or_else(|| ToolError::InvalidArgs(format!("{} needs `line`", a.op)))?;
    let line = text
        .lines()
        .nth(line_no.saturating_sub(1) as usize)
        .filter(|_| line_no >= 1)
        .ok_or_else(|| ToolError::InvalidArgs(format!("line {line_no} is out of range")))?;
    let byte = match (&a.symbol, a.column) {
        (Some(word), _) if !word.is_empty() => find_word(line, word).ok_or_else(|| {
            ToolError::InvalidArgs(format!(
                "`{word}` is not on line {line_no}: {}",
                truncate_middle(line.trim(), MAX_LINE_CHARS).0
            ))
        })?,
        (_, Some(col)) => line
            .char_indices()
            .nth(col.saturating_sub(1) as usize)
            .map_or(line.len(), |(i, _)| i),
        _ => line.len() - line.trim_start().len(),
    };
    Ok(json!({ "line": line_no - 1, "character": line[..byte].encode_utf16().count() }))
}

/// Byte index of `word` in `line` as a whole identifier, else anywhere.
fn find_word(line: &str, word: &str) -> Option<usize> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    line.match_indices(word)
        .find(|(i, _)| {
            !line[..*i].chars().next_back().is_some_and(ident)
                && !line[i + word.len()..].chars().next().is_some_and(ident)
        })
        .or_else(|| line.match_indices(word).next())
        .map(|(i, _)| i)
}

fn kind_name(kind: &Value) -> &'static str {
    kind.as_u64()
        .and_then(|k| SYMBOL_KINDS.get((k as usize).wrapping_sub(1)))
        .copied()
        .unwrap_or("symbol")
}

fn format_diagnostics(shown: &str, list: &Value) -> String {
    let items = list.as_array().map(Vec::as_slice).unwrap_or_default();
    if items.is_empty() {
        return format!("no errors or warnings in {shown}");
    }
    let mut out: Vec<String> = items
        .iter()
        .map(|d| {
            let severity = match d["severity"].as_u64() {
                Some(1) => "error",
                Some(2) => "warning",
                Some(3) => "info",
                _ => "hint",
            };
            format!(
                "{shown}:{}:{}: {severity}: {}",
                d["range"]["start"]["line"].as_u64().unwrap_or(0) + 1,
                d["range"]["start"]["character"].as_u64().unwrap_or(0) + 1,
                d["message"].as_str().unwrap_or("").trim()
            )
        })
        .collect();
    out.sort_by_key(|l| !l.contains(": error: "));
    out.join("\n")
}

fn format_symbols(result: &Value) -> String {
    fn walk(s: &Value, depth: usize, out: &mut Vec<String>) {
        if out.len() >= MAX_SYMBOLS {
            return;
        }
        let range = if s.get("location").is_some() {
            &s["location"]["range"]
        } else {
            &s["range"]
        };
        let detail = s["detail"]
            .as_str()
            .filter(|d| !d.is_empty())
            .map(|d| format!(" — {}", truncate_middle(d, 80).0))
            .unwrap_or_default();
        out.push(format!(
            "{}{} {}{detail}  (lines {}-{})",
            "  ".repeat(depth),
            kind_name(&s["kind"]),
            s["name"].as_str().unwrap_or("?"),
            range["start"]["line"].as_u64().unwrap_or(0) + 1,
            range["end"]["line"].as_u64().unwrap_or(0) + 1
        ));
        for c in s["children"].as_array().into_iter().flatten() {
            walk(c, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    for s in result.as_array().into_iter().flatten() {
        walk(s, 0, &mut out);
    }
    if out.is_empty() {
        "no symbols".into()
    } else {
        out.join("\n")
    }
}

fn hover_text(result: &Value) -> String {
    let part = |v: &Value| -> String {
        v.as_str()
            .map(str::to_owned)
            .or_else(|| v["value"].as_str().map(str::to_owned))
            .unwrap_or_default()
    };
    let text = match &result["contents"] {
        Value::Array(items) => items.iter().map(part).collect::<Vec<_>>().join("\n---\n"),
        other => part(other),
    };
    if text.trim().is_empty() {
        "no hover information".into()
    } else {
        truncate_middle(&text, 4000).0
    }
}

pub struct GodotLsp {
    project: PathBuf,
    port: u16,
    /// One connection at a time.
    busy: tokio::sync::Mutex<()>,
}

impl GodotLsp {
    pub fn new(project: PathBuf, port: u16) -> Self {
        Self {
            project,
            port,
            busy: tokio::sync::Mutex::new(()),
        }
    }

    async fn locations(&self, result: &Value, ctx: &ToolCtx) -> String {
        let items: Vec<&Value> = match result {
            Value::Array(a) => a.iter().collect(),
            Value::Null => Vec::new(),
            one => vec![one],
        };
        let mut out = Vec::new();
        for l in items {
            let uri = l["uri"].as_str().or(l["targetUri"].as_str()).unwrap_or("");
            let range = if l.get("targetSelectionRange").is_some() {
                &l["targetSelectionRange"]
            } else {
                &l["range"]
            };
            let line = range["start"]["line"].as_u64().unwrap_or(0) as usize;
            let Some(path) = uri_path(uri) else {
                out.push(format!("{uri}:{}", line + 1));
                continue;
            };
            // Line text only from inside the working directory, like read_file without approval.
            let text = if ctx.guard().contains(&path) {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| {
                        t.lines()
                            .nth(line)
                            .map(|l| truncate_middle(l.trim(), MAX_LINE_CHARS).0)
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            };
            out.push(
                format!("{}:{}: {text}", display_path(&ctx.env, &path), line + 1)
                    .trim_end()
                    .to_owned(),
            );
        }
        if out.is_empty() {
            "no definition found".into()
        } else {
            out.join("\n")
        }
    }
}

#[async_trait]
impl Tool for GodotLsp {
    fn name(&self) -> &str {
        "godot"
    }

    fn description(&self) -> &str {
        "GDScript language features from the Godot editor, which must have this project open. op: diagnostics \
         {path, wait_ms?} (parse and type errors), symbols {path}, definition {path, line, symbol?|column?}, \
         hover {path, line, symbol?|column?}. Lines are 1-based; name the identifier on the line."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["diagnostics", "symbols", "definition", "hover"] },
                "path": { "type": "string", "description": "A .gd file, relative to the working directory" },
                "line": { "type": "integer", "minimum": 1 },
                "symbol": { "type": "string" },
                "column": { "type": "integer", "minimum": 1 },
                "wait_ms": { "type": "integer", "minimum": 0, "maximum": MAX_WAIT_MS }
            },
            "required": ["op", "path"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Read
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        if !matches!(
            a.op.as_str(),
            "diagnostics" | "symbols" | "definition" | "hover"
        ) {
            return Err(ToolError::InvalidArgs(format!("unknown op `{}`", a.op)));
        }
        let (location, path, reasons) = resolve_path(ctx, &a.path);
        let mut p = ActionPreview::new(
            format!("godot {} {}", a.op, display_path(&ctx.env, &path)),
            location,
        );
        p.reasons = reasons;
        p.paths = vec![path];
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = preview.paths[0].clone();
        let shown = display_path(&ctx.env, &path);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ToolError::Failed(format!("cannot read {shown}: {e}")))?;
        let _busy = self.busy.lock().await;
        let work = async {
            let mut s = Session::open(self.port, &self.project).await?;
            let uri = file_uri(&path)?;
            s.notify(
                "textDocument/didOpen",
                json!({ "textDocument": { "uri": uri, "languageId": "gdscript", "version": 1, "text": text } }),
            )
            .await?;
            let doc = json!({ "uri": uri });
            let out = match a.op.as_str() {
                "diagnostics" => {
                    let wait = Duration::from_millis(
                        a.wait_ms.unwrap_or(DEFAULT_WAIT_MS).min(MAX_WAIT_MS),
                    );
                    match s.wait_diagnostics(&uri_key(&uri), wait).await {
                        Some(list) => format_diagnostics(&shown, &list),
                        None => format!(
                            "Godot sent no diagnostics for {shown} within {} ms; is the file inside the project open in \
                             the editor ({})?",
                            wait.as_millis(),
                            self.project.display()
                        ),
                    }
                }
                "symbols" => format_symbols(
                    &s.request(
                        "textDocument/documentSymbol",
                        json!({ "textDocument": doc }),
                    )
                    .await?,
                ),
                op => {
                    let method = if op == "hover" {
                        "textDocument/hover"
                    } else {
                        "textDocument/definition"
                    };
                    let result = s
                        .request(
                            method,
                            json!({ "textDocument": doc, "position": position(&text, &a)? }),
                        )
                        .await?;
                    if op == "hover" {
                        hover_text(&result)
                    } else {
                        self.locations(&result, ctx).await
                    }
                }
            };
            let _ = s
                .notify("textDocument/didClose", json!({ "textDocument": doc }))
                .await;
            Ok::<String, ToolError>(out)
        };
        let out = tokio::select! {
            r = work => r?,
            () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        let (content, truncated) = truncate_middle(&out, ctx.limits.max_output_chars);
        Ok(ToolOutput {
            content,
            is_error: false,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_point_at_the_named_identifier() {
        let text = "extends Node\n\tvar speed := max_speed * 2\n";
        let a = |symbol: Option<&str>, column: Option<u32>| Args {
            op: "definition".into(),
            path: "x.gd".into(),
            line: Some(2),
            symbol: symbol.map(str::to_owned),
            column,
            wait_ms: None,
        };
        assert_eq!(
            position(text, &a(Some("speed"), None)).unwrap(),
            json!({ "line": 1, "character": 5 })
        );
        assert_eq!(
            position(text, &a(Some("max_speed"), None)).unwrap()["character"],
            14
        );
        assert_eq!(position(text, &a(None, None)).unwrap()["character"], 1);
        assert!(position(text, &a(Some("nope"), None)).is_err());
    }

    #[test]
    fn finds_the_project_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let game = tmp.path().join("game");
        std::fs::create_dir_all(game.join("scenes")).unwrap();
        assert_eq!(find_project(tmp.path()), None);
        std::fs::write(game.join("project.godot"), "").unwrap();
        assert_eq!(find_project(tmp.path()).as_deref(), Some(game.as_path()));
        assert_eq!(
            find_project(&game.join("scenes")).as_deref(),
            Some(game.as_path())
        );
    }

    #[test]
    fn formats_server_results() {
        let diags = json!([
            { "severity": 2, "message": "Unused variable", "range": { "start": { "line": 4, "character": 1 } } },
            { "severity": 1, "message": "Identifier not declared", "range": { "start": { "line": 9, "character": 0 } } }
        ]);
        assert_eq!(
            format_diagnostics("player.gd", &diags),
            "player.gd:10:1: error: Identifier not declared\nplayer.gd:5:2: warning: Unused variable"
        );
        let symbols = json!([{ "name": "Player", "kind": 5, "range": { "start": { "line": 0 }, "end": { "line": 20 } },
            "children": [{ "name": "jump", "kind": 12, "range": { "start": { "line": 3 }, "end": { "line": 6 } } }] }]);
        assert_eq!(
            format_symbols(&symbols),
            "class Player  (lines 1-21)\n  function jump  (lines 4-7)"
        );
        assert_eq!(
            hover_text(
                &json!({ "contents": { "kind": "markdown", "value": "func jump() -> void" } })
            ),
            "func jump() -> void"
        );
    }

    /// A scripted language server: answers initialize, then publishes diagnostics for the opened file.
    #[tokio::test]
    async fn talks_to_a_language_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, writer) = stream.into_split();
            let mut s = Session {
                reader: BufReader::new(read),
                writer,
                next_id: 1,
                diagnostics: Vec::new(),
            };
            loop {
                let Ok(msg) = s.read().await else { break };
                match msg["method"].as_str() {
                    Some("initialize") => {
                        s.send(&json!({ "jsonrpc": "2.0", "id": msg["id"], "result": { "capabilities": {} } }))
                            .await
                            .unwrap();
                        s.notify("gdscript_client/changeWorkspace", json!({}))
                            .await
                            .unwrap();
                    }
                    Some("textDocument/didOpen") => {
                        let uri = msg["params"]["textDocument"]["uri"].clone();
                        s.notify(
                            "textDocument/publishDiagnostics",
                            json!({ "uri": uri, "diagnostics": [
                                { "severity": 1, "message": "Expected end of statement", "range": { "start": { "line": 0, "character": 7 } } }
                            ] }),
                        )
                        .await
                        .unwrap();
                    }
                    _ => {}
                }
            }
        });
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let file = root.join("main.gd");
        std::fs::write(&file, "extends\n").unwrap();
        let mut s = Session::open(port, &root).await.unwrap();
        let uri = file_uri(&file).unwrap();
        s.notify(
            "textDocument/didOpen",
            json!({ "textDocument": { "uri": uri, "languageId": "gdscript", "version": 1, "text": "extends\n" } }),
        )
        .await
        .unwrap();
        let list = s
            .wait_diagnostics(&uri_key(&uri), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(
            format_diagnostics("main.gd", &list),
            "main.gd:1:8: error: Expected end of statement"
        );
        drop(s);
        server.await.unwrap();
    }
}
