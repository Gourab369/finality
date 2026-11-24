# Finality

Finality is a terminal (TUI) harness for coding and research with several agents. An Orchestrator plans the work and
delegates tasks to Worker and Research sub-agents. Background agents keep memory, score plans, learn skills and maintain a
small knowledge base. It runs local models through [Ollama](https://ollama.com) by default, and can also use OpenAI-compatible
endpoints, Anthropic and local `.gguf` models through llama.cpp.

## Requirements

- One of:
  - Ollama running locally (`ollama serve`) with a model that supports tool calls, for example `ollama pull qwen3:14b`.
  - An OpenAI-compatible endpoint, or an Anthropic API key.
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) and downloaded `.gguf` model files.
- Optional: Chrome or Chromium for the browser tools, and MCP servers.
- Windows 10+ (pre-built binaries and CI). The code also targets Linux and macOS, but there are no binaries for them yet
  and they are not tested in CI; see [Roadmap](#roadmap). Use a terminal with keyboard enhancement (Windows Terminal, kitty,
  WezTerm, iTerm2) for Shift+Enter; Alt+Enter works everywhere.
- Building from source only: Rust 1.89 or newer (the repository pins a toolchain in `rust-toolchain.toml`).

## Install

### Pre-built binaries (recommended, Windows)

Download `finality-<version>-windows-x86_64.zip` from the [Releases](https://github.com/Gourab369/finality/releases) page.

1. Check the download against `SHA256SUMS.txt` from the same release: in PowerShell,
   `(Get-FileHash .\finality-<version>-windows-x86_64.zip).Hash` must match the hash in that file (the case of the letters
   does not matter).
2. Unpack it and put `finality.exe` in a folder on your `PATH`, for example `%USERPROFILE%\.local\bin`.
3. Run `finality --version`.

The binary is not code-signed yet, so Windows SmartScreen may warn on first start (**More info → Run anyway**).

On Linux and macOS, install from source.

### From source

```sh
cargo install --git https://github.com/Gourab369/finality --tag v0.4.0 --locked finality
# or, in a clone of the repository:
cargo install --path crates/finality --locked
```

You can also build it with `cargo build --release` and use `target/release/finality`. To update an existing install, run the
same command again (add `--force` if Cargo says it is already installed), after closing any running `finality`.

### Releasing (maintainers)

Bump `version` in the workspace `Cargo.toml`, run `cargo update --workspace`, commit, then push a tag:

```sh
git tag v0.4.0
git push origin v0.4.0
```

The `Release` workflow builds the Windows archive (the binary, `LICENSE` and this README), writes `SHA256SUMS.txt`, and
publishes both as a GitHub Release with generated notes.

## Quick start

```sh
cd path/to/your/project
finality                                   # opens the TUI; the session starts with your first prompt
finality "explain how the config is loaded"  # starts with a prompt
finality --provider anthropic --model claude-sonnet-4-5 --mode auto
```

The directory where you start `finality` is the **working directory**. File tools and shell commands are checked against it
(see [Safety](#safety)).

## Configuration

Settings are layered. Higher layers override lower ones:

1. CLI flags.
2. Environment variables: `FINALITY_<SECTION>__<KEY>`, for example `FINALITY_GENERAL__DEFAULT_MODE=auto` or
   `FINALITY_PROVIDER__OLLAMA__MODEL=qwen3:8b`.
3. `~/.finality/config.toml` (or the file passed with `--config`).
4. Built-in defaults.

Every key is optional; all keys and their defaults are defined in
[crates/finality_core/src/config.rs](crates/finality_core/src/config.rs). A small example:

```toml
[general]
default_mode = "cowork"

[provider]
default = "ollama"

[provider.ollama]
base_url = "http://127.0.0.1:11434"
model = "qwen3:14b"
num_ctx = 32768

[models]                 # per-role overrides, "<provider>:<model>"
memory = "ollama:qwen3:4b"
# triage = "ollama:qwen3:4b"  # default: the memory model if set (not on llama.cpp), else the main model

[safety]
os_sandbox = false       # Linux/macOS only, see below
```

Set `FINALITY_HOME` to move the data directory (default `~/.finality`). `FINALITY_LOG` (or `RUST_LOG`) sets the log level.

### CLI flags

```
finality [PROMPT]
  --mode <plan|cowork|auto>              (auto-edit is accepted as cowork)
  --provider <ollama|openai|anthropic|llamacpp>   --model <name>
  --think <off|low|medium|high>
  --subagents <N>                        maximum concurrent sub-agents
  --resume <SESSION_ID>
  --no-mcp                               do not use MCP servers
  --decide                               enable the decision model
  --direct                               turn direct mode on (stays on until /direct off)
  --config <PATH>
```

### Providers and API keys

| Provider | Setup |
|---|---|
| `ollama` | `provider.ollama.base_url` and `model`. `num_ctx` is always sent because Ollama's default context is small. |
| `openai` | Any OpenAI-compatible API: set `provider.openai.base_url`, `model` and `context_window` (the API doesn't report it). Set `api_key_env = ""` for local servers that need no key. |
| `anthropic` | Set `provider.anthropic.model`. Thinking levels map to thinking budgets. |
| `llamacpp` | Local `.gguf` models run by [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`; see below. No API key. |

API keys are read from the environment variable named in `api_key_env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` by default).
If that variable is not set, they are read from the OS keyring: service `finality`, user `api-key:<provider>`. Keys are never
written to the config, the session logs or the log file.

Switch providers at runtime with `/provider <name> [model]` or `/model <name>`.

Unless you pass `--model` (or `--resume`), Finality starts by listing the provider's installed models so you can choose one
with ↑/↓ and Enter; the configured model is preselected. Anything you type before choosing is sent once a model is picked.
`/model` without a name opens the same list.

### llama.cpp

Install llama.cpp so that `llama-server` is on your `PATH` (or set `server_path`), and point Finality at the folder with your
downloaded `.gguf` files:

```toml
[provider]
default = "llamacpp"

[provider.llamacpp]
models_dir = 'D:\models'          # searched three folders deep for .gguf files
# model = "Qwen3-14B-Q4_K_M.gguf"  # path inside models_dir, or an absolute path; omit to choose at start
# server_path = 'C:\llama.cpp\llama-server.exe'
context_window = 16384           # --ctx-size
# gpu_layers = 99                 # --n-gpu-layers; unset lets llama.cpp decide
# extra_args = ["--flash-attn", "on"]
```

The model list (at start, `/model`, or `/provider llamacpp`) shows the `.gguf` files in `models_dir`, skipping vision
projectors (`mmproj`) and the extra parts of split models. Choosing one starts `llama-server` on a free local port
(`127.0.0.1`) with `--jinja` for tool calling, and waits until the model is loaded (`startup_timeout_secs`, default 300).
Choosing another model restarts it. Finality stops the server when it exits; on Windows the server also ends if Finality
crashes. The server's output goes to Finality's log file.

- One model is loaded at a time. Roles pinned to a different llama.cpp model in `[models]` make the server switch back and
  forth; pin them to the same model or to another provider.
- Use a model whose chat template supports tool calls (Qwen 3, Llama 3.1+, Mistral, Gemma 3 and similar instruct models).
- To use a server you start yourself, set `base_url = "http://127.0.0.1:8080"`; Finality then lists and uses that server's
  models and starts nothing. Set `context_window` to its `--ctx-size`.

### MCP servers

MCP servers use the Claude Code `mcpServers` format, in `~/.finality/mcp.json` (global) and `<project>/.mcp.json` (project):

```json
{ "mcpServers": {
    "github": { "type": "http", "url": "https://api.githubcopilot.com/mcp/" },
    "files":  { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
} }
```

Project servers must be trusted once per project; you are asked about this at startup and shown the command, its
environment overrides and working directory. A project file can only reference environment variables that pass
`safety.env_denylist`, so it cannot hand your API keys to a server. Servers start lazily, when one of
their tools is first needed, and each tool call is bounded by `mcp.call_timeout_secs`. Use `/mcp login <server>` for OAuth
servers; tokens go in the OS keyring. MCP tools are named
`mcp__<server>__<tool>`. They need approval in `cowork`, and are denied in `plan`.

Add `"trustReadOnlyHints": true` to a server in the **global** `mcp.json` to run its tools that declare
`readOnlyHint` like `read_file`: no approval in `cowork`, allowed in `plan`. Only do this for servers you trust to label
their tools honestly. A project `.mcp.json` cannot set it.

Two more Finality additions to the format:

- `"tools": ["name", ...]` offers only those tools of a server (`-` and `_` count as the same).
- In the global file, `${keyring:<user>}` reads a secret from the OS keyring (service `finality`), so keys need not sit in
  the file or the environment. A project file cannot use it.

For example, Tavily's server for page extraction and crawling (search already uses Tavily directly):

```json
{ "mcpServers": {
    "tavily": { "command": "npx", "args": ["-y", "tavily-mcp@latest"],
                "env": { "TAVILY_API_KEY": "${keyring:api-key:tavily}" },
                "tools": ["tavily_extract", "tavily_crawl", "tavily_map"] }
} }
```

### Editor language features (VS Code and Godot)

The agent can use the language servers your editor already runs, so it sees compiler errors after an edit and can find
definitions and references without searching text.

**VS Code** (rust-analyzer, Pylance/Pyright, TypeScript, GDScript through godot-tools, and any other language extension):
install the bridge extension from `editors/vscode`:

```sh
cd editors/vscode
npm install
npm run compile
npm run package                      # writes finality-bridge-0.4.0.vsix
code --install-extension finality-bridge-0.4.0.vsix
```

The extension serves VS Code's language features as a local MCP server on `127.0.0.1` with a random token, and writes
`~/.finality/ide/<port>.json` so Finality can find it (set `finality.dataDir` in VS Code if you moved Finality's data
directory). Finality then offers the `vscode` server's tools when VS Code has a folder open that contains the working
directory: `diagnostics`, `definition`, `references`, `implementations`, `hover`, `symbols` and `workspace_symbols`, all
read-only, so they run without approval. Paths are relative to the working directory and limited to it.

- Errors update a moment after the agent writes a file; `diagnostics` waits for them to settle (`wait_ms`, default 3 s).
- rust-analyzer's `cargo check` results normally refresh only when you save in the editor. `diagnostics` with
  `recheck: true` runs that check (and with it build scripts and proc macros, as saving would).
- If VS Code isn't open, the tools report that; opening it later is enough.

**Godot**: when the working directory (or a parent, or a direct subfolder) holds `project.godot`, the `godot` tool talks to
the Godot editor's built-in GDScript language server on `127.0.0.1:6005` (`[godot] lsp_port`; in Godot under Editor
Settings → Network → Language Server). It offers `diagnostics`, `symbols`, `definition` and `hover` for `.gd` files while
the project is open in the editor. Set `godot.enabled = false` to hide it.

### Web and browser

`web_search` picks its backend with `web.search_backend`:

| Value | Behaviour |
|---|---|
| `auto` (default) | [Tavily](https://tavily.com) when a key is available, else DuckDuckGo. If Tavily fails, the results come from DuckDuckGo with a note saying why. |
| `tavily` | Tavily only; an error without a key. |
| `duckduckgo` | DuckDuckGo only. |
| `searxng` | Your SearXNG instance at `web.searxng_url`. |

The Tavily key comes from `TAVILY_API_KEY` (or the variable named in `web.tavily_api_key_env`), else the OS keyring
(service `finality`, user `api-key:tavily`). On Windows that is a generic credential named `api-key:tavily.finality`
(keyring entries are named `<user>.<service>`); add it with `cmdkey /generic:api-key:tavily.finality /user:api-key:tavily
/pass`, which asks for the key. When DuckDuckGo answers with its bot check instead of results, Finality tries
DuckDuckGo Lite (twice, two seconds apart) and otherwise reports that searching is blocked, rather than "no results".

`web_fetch` returns a page's main content as Markdown: headings, lists, tables, code blocks and links (made absolute) are
kept; menus, headers, footers, sidebars, forms and media are left out, and when the page marks its content with `main` or
`article`, only that is used. PDFs are converted to text, page by page (up to 5 MB; scanned PDFs have no text). Long pages
come in parts: the end of each part says which `offset` to pass to read on.

Searches and fetched pages are kept in memory for 15 minutes, so repeating a search or reading on costs no request;
`refresh: true` fetches a page again.

The browser tools drive Chrome or Chromium; set `browser.chrome_path` if it isn't detected. Requests to
loopback, private and cloud metadata addresses are blocked unless `web.allow_private_network = true`. In the browser this
applies to every request the page makes (scripts, images, frames, redirects), not only the address you opened.

### Decision model

With `--decide` or `/decide on`, small classification calls (triage, template fit, retry advice) go to a fast decision model
(`decision.model`, served next to Ollama). Its answers are used only when they are confident; otherwise the main LLM decides.

### Direct mode

`/direct` (or `/direct on`, or `--direct` at start) adds a short section to the Orchestrator's and sub-agents' system prompts:
assume technical requests have a legitimate purpose, answer fully without moralizing or unrequested warnings, treat security
work on your own or authorized targets as in scope, and when declining part of a request say which part and why. The text is
in `crates/finality_agents/prompts/direct.md`. It reduces needless refusals; it does not override what a model was trained
never to do, and it does not change modes, approvals or the handling of untrusted content. The setting is saved in
`~/.finality/state.json` and stays on across sessions until `/direct off`; the status bar shows `direct` while it is on.

### Instruction files

Put project conventions in `AGENTS.md` or `FINALITY.md` at the root of the working directory, and personal ones in
`~/.finality/FINALITY.md`. All three are added to the Orchestrator's and sub-agents' system prompts (global first, then
`AGENTS.md`, then `FINALITY.md`; together at most 16,000 characters). They are read again for every turn, so edits apply
right away. They cannot change the mode, approvals or safety rules. You write these files; `/init` drafts the global one
for you from the current project (environment, working rules, style) when it does not exist yet.

### Cost and speed

- Short, simple messages (greetings, one-line questions and commands without "and", "then", "plan", ...) skip the
  triage request. Set `llm.quick_triage = false` to triage every message.
- Anthropic requests mark the system prompt, the tools and the conversation so far for
  [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), so a turn's later requests reuse
  them. llama.cpp requests set `cache_prompt`, and Ollama and OpenAI cache matching prompt prefixes on their own.
- Ollama keeps one prompt cache per loaded model by default, and any other request to that model replaces it. Triage
  therefore runs on `models.triage`, or on the memory model when `models.memory` is set, so it does not throw away the
  main model's cache before every turn (except on llama.cpp, which holds one model at a time). Pin a small model, e.g.
  `memory = "ollama:qwen3:4b"`; both models must fit in
  memory at once (`OLLAMA_MAX_LOADED_MODELS` ≥ 2), or Ollama swaps them on every request. Setting
  `OLLAMA_NUM_PARALLEL=2` instead gives the main model a second cache slot, at the cost of twice the context memory.
- The list of tools that can be loaded is sent as a note in the conversation, not in the system prompt, so loading or
  unloading a tool doesn't change the start of the prompt.

## Modes

The mode decides which actions run without asking. Change it with `--mode` or `/mode`. Only you can change it: nothing a model
reads or writes can.

| Action (inside the working directory) | plan | cowork | auto |
|---|---|---|---|
| Read | allow | allow | allow |
| Write (edit, create) | deny | allow | allow |
| Delete files (`edit_file` delete, restoring a checkpoint that removes files) | deny | ask | allow |
| Shell, `git_commit` | deny | ask | allow |
| Network / browser | deny | ask | allow |
| MCP tools | deny | ask | allow |
| **Anything touching outside the working directory, or that can't be verified** | deny | ask | **ask** |
| Matches `safety.always_ask_patterns` | deny | ask | **ask** |

In `cowork`, "allow for session" remembers a tool (or a shell command prefix) for the rest of the session. It is never offered
for deletions or for outside or unverifiable actions. The former `auto-edit` mode is now part of `cowork`; `auto-edit` is still
accepted in the config, on the command line and in `/mode`, and means `cowork`.

### Checkpoints and undo

Before a turn's first change to a file, `edit_file` saves the file's current contents (or notes that it did not exist) in a
checkpoint for that turn, under `~/.finality/checkpoints/<session>/`. `/checkpoints` lists them; `/undo` puts every file back
as it was before the latest checkpoint, and `/undo <n>` goes back to before checkpoint `n` (undoing it and every later one;
files created since then are deleted). The Orchestrator is told on its next turn. The agent can do the same with the
`checkpoint_list` and `checkpoint_restore` tools. Changes made by shell commands are not tracked.

### Git

The header shows the working directory's branch. The `git` tool gives agents read-only status, diff, log, show, blame and
branch output, limited to the working directory, with repository-configured programs (fsmonitor, external diff, textconv)
turned off so a read cannot run code. `git_commit` (loaded on demand) stages the given paths, or all tracked changes, and
commits; it runs your commit hooks, so it is treated like a shell command. `/diff` shows `git status` and the uncommitted diff.

## Using the TUI

| Key | Action |
|---|---|
| Enter | Send |
| Shift+Enter / Alt+Enter | New line |
| Ctrl+C | Cancel the current turn; press twice within 2 s while idle to quit |
| Tab | Toggle the agents panel (or complete a slash command or `@` path) |
| PgUp / PgDn, mouse wheel | Scroll |
| Up / Down (empty input) | Prompt history |
| Ctrl+O | Expand or collapse thinking and tool output |
| Alt+V (or Ctrl+V when the terminal passes it on) | Paste an image from the clipboard |
| Esc | Close the completion popup or a list; deny an approval |
| Up / Down, Enter (in a list) | Move and choose (`/resume`, `/model`, the start-up model list) |

**`@` mentions.** Type `@` and part of a path to pick a file or folder from the working directory (Tab inserts it). A
mentioned file is attached to your message (up to 50,000 characters each, 100,000 in total); a folder is attached as a
listing two levels deep. Paths outside the working directory are not attached. The session log keeps your text without
the attachments. The list skips `.gitignore`d paths and dependency folders (Python virtual environments, `node_modules`,
`__pycache__` and similar caches); typing the full path of a file inside them still attaches it.

**Images.** Alt+V saves the clipboard image as a PNG in `~/.finality/attachments/` and inserts `[image: <path>]`; pasting
or typing the path of a PNG, JPEG, GIF or WebP file works too. Images are sent with the message to models that accept them
(Anthropic, vision models behind OpenAI-compatible APIs, Ollama vision models), at most 8 per message and 5 MiB each.
For Ollama, a notice appears if the model doesn't report vision support. The llama.cpp server that Finality starts loads no
vision projector, so its models can't read images.

### Slash commands

| Command | Effect |
|---|---|
| `/help` | List commands and keys |
| `/new` | Close this session and start a new one |
| `/sessions [--all\|prune <days>]` | List past sessions (this project, or all), or delete closed sessions older than N days |
| `/resume [id]` | Resume a past session; without an id, choose from this directory's sessions |
| `/mode <plan\|cowork\|auto>` | Change the permission mode |
| `/think <off\|low\|medium\|high>` | Set the thinking level |
| `/provider <ollama\|openai\|anthropic\|llamacpp> [model]` | Switch provider (llamacpp without a model lists your `.gguf` files) |
| `/model [name]` | Switch model; without a name, choose from the provider's models |
| `/context` | Show token usage |
| `/compact` | Have the memory agent summarize the conversation into events, then reload the context with only the last 2 exchanges plus those summaries |
| `/checkpoints` | List the files saved before each turn's edits |
| `/undo [n]` | Restore files to before the latest checkpoint, or before checkpoint `n` |
| `/diff` | Show `git status` and the uncommitted diff |
| `/init` | Have the agent study this project and write your global instructions, `~/.finality/FINALITY.md` (only if the file does not exist yet) |
| `/agents`, `/tasks` | Show running agents and the current plan |
| `/subagents <n>` | Set how many sub-agents may run at once |
| `/decide <on\|off>` | Use the decision model where it is confident |
| `/direct [on\|off]` | Toggle direct mode (fewer refusals and disclaimers); remembered until switched off |
| `/memory [recall <query>\|review]` | Show the memory digest, search past sessions, or review notes the learner made about you |
| `/skills [review]` | List skills, or review skills the learner proposed |
| `/skill <name> [args]`, `/<name> [args]` | Run a skill |
| `/tools` | Show loaded and loadable tools |
| `/knowledge [topic]` | List knowledgepedia topics or show one |
| `/mcp [list\|login <server>\|enable\|disable]` | Manage MCP servers |
| `/agent [list\|show <name>\|delete <name>]` | List, show or delete custom agents |
| `/quit`, `/exit` | Save the session (the memory agent summarizes it into events) and exit; Ctrl+C skips the wait |

When you quit (`/quit`, `/exit` or Ctrl+C twice), Finality stays open with a "Quitting Finality" dialog while the memory agent
summarizes the session into events, the learner extracts skills, knowledgepedia facts and profile notes from it, and the scorer
scores finished plans (together up to `memory.quit_wait_secs`, default 300 s). Resuming the session later loads only
the last two exchanges word for word and uses those events for everything earlier, so it starts fast with a small context.
Press Ctrl+C in the dialog to leave at once; whatever was not finished is picked up the next time Finality starts.

Skills are Claude Code compatible `SKILL.md` folders in `~/.finality/skills/<name>/`. Finality ships two built-in skills:

- `stubborn` (`/stubborn <request>`): retry failed work with a different approach each time until it is done or blocked by
  something only you can resolve.
- `self-improve` (`/self-improve <request>`): a structured way of working for non-trivial tasks. It states the goal and how
  to verify it, recalls lessons from earlier sessions, reasons from evidence, follows a playbook for the task type (bug,
  feature, refactor, review, performance, research, analysis, planning, docs, shell), verifies the result, and ends with a
  one-line `Lesson:` that the memory agent and learner can turn into knowledge and skills for next time.

A skill folder with the same name replaces the built-in one.

### Custom agents

Besides the built-in Worker and Research sub-agents, the Orchestrator can create **custom agents**: reusable specialists
with their own instructions on a worker or research base, and optionally a narrower tool list. Ask for one ("create a
reviewer agent that checks for panics and missing tests"), or let the Orchestrator create one when a kind of task keeps
coming back. Creating an agent always needs your approval, whatever the mode, and is not possible in `plan` mode.

Custom agents live in `~/.finality/agents/<name>/AGENT.md`, where you can edit them. `/agent` lists them,
`/agent show <name>` prints one, and `/agent delete <name>` removes it. A custom agent can only use worker, research and
MCP tools, and the mode and approvals still apply to everything it does.

## Data directory

```
~/.finality/                  # private to your user on Linux and macOS (mode 0700)
├── config.toml, mcp.json
├── FINALITY.md              # optional instructions for every project
├── projects.json            # known projects, sessions, MCP trust
├── state.json               # settings toggled in the UI that persist (direct mode)
├── sessions/                # session logs (*.hlog) and index.jsonl; tool output is redacted
├── checkpoints/             # file contents saved before edits, per session (for /undo)
├── attachments/             # pasted images
├── ide/                     # lock files of the VS Code bridge extension
├── memory/                  # memory event trees, user_profile.md (editable) and user_profile.pending.md
├── skills/                  # skills; skills/_pending holds learner proposals
├── agents/                  # custom agents (AGENT.md)
├── knowledgepedia/          # learned topic notes
├── lookup/, plans/          # plan scoring data
└── logs/finality.log         # rotated daily, 7 files kept; secrets are redacted
```

Several `finality` processes can run at the same time; shared files are updated under locks.

## Safety

- Every path argument is resolved (including symlinks and junctions) and checked against the working directory.
- Shell commands are analyzed before they run. Directory changes, absolute or escaping paths, variable expansion, nested
  interpreters, redirections outside the project, link creation and privilege or remote commands are classified as outside or
  unverifiable, and **always ask**, even in `auto` mode.
- `safety.always_ask_patterns` (defaults: `git push --force`, `rm -rf`, `Remove-Item -Recurse -Force`, `format `,
  `shutdown`, `mkfs`) ask in every mode. Patterns match whatever the option order, clustering or quoting: `rm -rf` also
  catches `rm -fr`, `rm -r -f` and `rm -r'f'`, and the PowerShell pattern catches `rm -r -fo`.
- `search_files` never reads through a symlink that points outside the project, and `edit_file` reads nothing from a path
  outside the project until you approve the edit.
- The shell environment drops variables matching `safety.env_denylist` (API keys, tokens).
- Tool output, web pages, MCP results and sub-agent reports are marked as untrusted data, and so is text a model wrote
  earlier (memory digests, skills, custom-agent instructions). If content looks like a prompt injection
  ("ignore previous instructions", "switch to auto mode", fake tool calls), a warning notice appears and the model is reminded
  not to follow them.
- Notes the learner makes about you wait in `memory/user_profile.pending.md` until you keep or drop them with
  `/memory review` (set `learner.auto_approve_profile = true` to skip the review).
- Secrets are redacted from the log file and from tool calls and results in session logs; text typed into web pages is
  never recorded.

### OS sandbox (Linux and macOS)

With `safety.os_sandbox = true`, shell commands classified as inside the working directory run in an OS sandbox that allows
writes only to the working directory, the temp directory, `/tmp` and a few device nodes (`/dev/null`, `/dev/tty`, ...), plus
any directories in `safety.sandbox_writable`:

- **Linux** uses [Landlock](https://landlock.io) (kernel 5.13 or newer with Landlock enabled). If the kernel can't enforce it,
  the command is not run and the error says why.
- **macOS** uses `sandbox-exec` with a generated profile.
- **Windows** has no equivalent; the setting shows a warning and has no effect.

Commands that you approve as touching outside the working directory run without the sandbox, because you allowed them
explicitly. Reads are not limited. Network access is limited only if you ask for it: `safety.sandbox_network = "deny"` stops
TCP connections and listeners except outbound connections to `safety.sandbox_tcp_ports` (default `[443, 80]`); on Linux this
needs kernel 6.7 or newer, and an older kernel refuses to run the command instead of allowing the network. UDP (DNS) is not
filtered.

## Known limitations

- Static command analysis can't see what scripts or binaries do internally. `./build.sh` inside the project counts as inside,
  even if the script writes elsewhere. On Linux and macOS, `safety.os_sandbox` closes this gap for writes and
  `safety.sandbox_network = "deny"` for TCP; on Windows you rely on the analyzer and approvals.
- MCP servers run with your privileges, and their side effects can't be analyzed. That's why their tools are risk class
  External: denied in `plan`, and they need approval in `cowork`.
- Prompt-injection detection is a heuristic that only produces warnings. The real protection is that untrusted content can't
  change the mode, grants or config, and that risky actions need approval.
- Local models vary widely in how well they call tools. Small models may loop or ignore instructions; use a model with good
  tool support.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --all
cargo deny check            # advisories, licences, duplicate/wildcard bans (cargo install cargo-deny)
```

CI runs the same checks on Windows, plus `cargo deny` and a build with the minimum supported Rust (1.89). The OS sandbox tests
(Linux and macOS only) are not run in CI. Pushing a `v*` tag builds the release archive and its checksum
(see [Releasing](#releasing-maintainers)).

Session logs grow without limit by default; set `general.session_retention_days` or run `/sessions prune <days>` to delete
closed sessions older than a cutoff.

## Roadmap
  - To be added.
## License

Licensed under the [Apache License, Version 2.0](LICENSE).
