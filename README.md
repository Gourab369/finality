# Finality

Finality is a terminal (TUI) harness for coding and research with several agents. An Orchestrator plans the work and
delegates tasks to Worker and Research sub-agents. Background agents keep memory, score plans, learn skills and maintain a
small knowledge base. It runs local models through [Ollama](https://ollama.com) by default, and can also use OpenAI-compatible
endpoints, Anthropic and local `.gguf` models through llama.cpp.

The library crates, environment variables (`H_AGENT_*`) and the data directory (`~/.h_agent`) still use the working name
`h_agent`.

## Requirements

- One of:
  - Ollama running locally (`ollama serve`) with a model that supports tool calls, for example `ollama pull qwen3:14b`.
  - An OpenAI-compatible endpoint, or an Anthropic API key.
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) and downloaded `.gguf` model files.
- Optional: Chrome or Chromium for the browser tools, and MCP servers.
- Windows 10+, Linux or macOS. Use a terminal with keyboard enhancement (Windows Terminal, kitty, WezTerm, iTerm2) for
  Shift+Enter; Alt+Enter works everywhere.
- Building from source only: Rust 1.89 or newer (the repository pins a toolchain in `rust-toolchain.toml`).

## Install

### Pre-built binaries (recommended)

Download the archive for your system from the [Releases](https://github.com/<owner>/finality/releases) page:

| System | Archive |
|---|---|
| Windows x86_64 | `finality-<version>-windows-x86_64.zip` |
| Linux x86_64 | `finality-<version>-linux-x86_64.tar.gz` |
| macOS Apple Silicon | `finality-<version>-macos-arm64.tar.gz` |

1. Check the download against `SHA256SUMS.txt` from the same release:
   - Windows (PowerShell): `(Get-FileHash .\finality-<version>-windows-x86_64.zip).Hash` and compare it with the line for
     that file (the case of the letters does not matter).
   - Linux / macOS: `sha256sum --ignore-missing -c SHA256SUMS.txt` (macOS: `shasum -a 256 --ignore-missing -c SHA256SUMS.txt`).
2. Unpack it and put `finality` (`finality.exe` on Windows) in a folder on your `PATH`, for example
   `%USERPROFILE%\.local\bin` or `~/.local/bin`.
3. Run `finality --version`.

The binaries are not code-signed yet. Windows SmartScreen may warn on first start (**More info → Run anyway**), and macOS
may block it until you run `xattr -d com.apple.quarantine finality`.

### From source

```sh
cargo install --git https://github.com/<owner>/finality --tag v0.2.0 --locked finality
# or, in a clone of the repository:
cargo install --path crates/finality --locked
```

You can also build it with `cargo build --release` and use `target/release/finality`. To update an existing install, run the
same command again (add `--force` if Cargo says it is already installed), after closing any running `finality`.

### Releasing (maintainers)

Bump `version` in the workspace `Cargo.toml`, run `cargo update --workspace`, commit, then push a tag:

```sh
git tag v0.2.0
git push origin v0.2.0
```

The `Release` workflow builds the three archives (each with the binary, `LICENSE` and this README), writes
`SHA256SUMS.txt`, and publishes them as a GitHub Release with generated notes.

## Quick start

```sh
cd path/to/your/project
finality                                   # opens the TUI; the session starts with your first prompt
finality "explain how the config is loaded"  # starts with a prompt
finality --provider anthropic --model claude-sonnet-4-5 --mode auto-edit
```

The directory where you start `finality` is the **working directory**. File tools and shell commands are checked against it
(see [Safety](#safety)).

## Configuration

Settings are layered. Higher layers override lower ones:

1. CLI flags.
2. Environment variables: `H_AGENT_<SECTION>__<KEY>`, for example `H_AGENT_GENERAL__DEFAULT_MODE=auto-edit` or
   `H_AGENT_PROVIDER__OLLAMA__MODEL=qwen3:8b`.
3. `~/.h_agent/config.toml` (or the file passed with `--config`).
4. Built-in defaults.

Every key is optional; all keys and their defaults are defined in
[crates/h_agent_core/src/config.rs](crates/h_agent_core/src/config.rs). A small example:

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

[safety]
os_sandbox = false       # Linux/macOS only, see below
```

Set `H_AGENT_HOME` to move the data directory (default `~/.h_agent`). `H_AGENT_LOG` (or `RUST_LOG`) sets the log level.

### CLI flags

```
finality [PROMPT]
  --mode <plan|cowork|auto-edit|auto>
  --provider <ollama|openai|anthropic|llamacpp>   --model <name>
  --think <off|low|medium|high>
  --subagents <N>                        maximum concurrent sub-agents
  --resume <SESSION_ID>
  --no-mcp                               do not use MCP servers
  --decide                               enable the decision model
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
If that variable is not set, they are read from the OS keyring: service `h_agent`, user `api-key:<provider>`. Keys are never
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

MCP servers use the Claude Code `mcpServers` format, in `~/.h_agent/mcp.json` (global) and `<project>/.mcp.json` (project):

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
`mcp__<server>__<tool>`. They need approval in `cowork` and `auto-edit`, and are denied in `plan`.

### Web and browser

`web_search` uses DuckDuckGo by default (`web.search_backend = "searxng"` with `web.searxng_url` for SearXNG). `web_fetch`
returns readable text. The browser tools drive Chrome or Chromium; set `browser.chrome_path` if it isn't detected. Requests to
loopback, private and cloud metadata addresses are blocked unless `web.allow_private_network = true`. In the browser this
applies to every request the page makes (scripts, images, frames, redirects), not only the address you opened.

### Decision model

With `--decide` or `/decide on`, small classification calls (triage, template fit, retry advice) go to a fast decision model
(`decision.model`, served next to Ollama). Its answers are used only when they are confident; otherwise the main LLM decides.

## Modes

The mode decides which actions run without asking. Change it with `--mode` or `/mode`. Only you can change it: nothing a model
reads or writes can.

| Action (inside the working directory) | plan | cowork | auto-edit | auto |
|---|---|---|---|---|
| Read | allow | ask | allow | allow |
| Write | deny | ask | allow | allow |
| Shell | deny | ask | ask | allow |
| Network / browser | deny | ask | ask | allow |
| MCP tools | deny | ask | ask | allow |
| **Anything touching outside the working directory, or that can't be verified** | deny | ask | ask | **ask** |
| Matches `safety.always_ask_patterns` | deny | ask | ask | **ask** |

In `cowork`, "allow for session" remembers a tool (or a shell command prefix) for the rest of the session. It is never offered
for outside or unverifiable actions.

## Using the TUI

| Key | Action |
|---|---|
| Enter | Send |
| Shift+Enter / Alt+Enter | New line |
| Ctrl+C | Cancel the current turn; press twice within 2 s while idle to quit |
| Tab | Toggle the agents panel (or complete a slash command) |
| PgUp / PgDn, mouse wheel | Scroll |
| Up / Down (empty input) | Prompt history |
| Ctrl+O | Expand or collapse thinking and tool output |
| Esc | Close the completion popup or a list; deny an approval |
| Up / Down, Enter (in a list) | Move and choose (`/resume`, `/model`, the start-up model list) |

### Slash commands

| Command | Effect |
|---|---|
| `/help` | List commands and keys |
| `/new` | Close this session and start a new one |
| `/sessions [--all\|prune <days>]` | List past sessions (this project, or all), or delete closed sessions older than N days |
| `/resume [id]` | Resume a past session; without an id, choose from this directory's sessions |
| `/mode <plan\|cowork\|auto-edit\|auto>` | Change the permission mode |
| `/think <off\|low\|medium\|high>` | Set the thinking level |
| `/provider <ollama\|openai\|anthropic\|llamacpp> [model]` | Switch provider (llamacpp without a model lists your `.gguf` files) |
| `/model [name]` | Switch model; without a name, choose from the provider's models |
| `/context` | Show token usage |
| `/agents`, `/tasks` | Show running agents and the current plan |
| `/subagents <n>` | Set how many sub-agents may run at once |
| `/decide <on\|off>` | Use the decision model where it is confident |
| `/memory [recall <query>\|review]` | Show the memory digest, search past sessions, or review notes the learner made about you |
| `/skills [review]` | List skills, or review skills the learner proposed |
| `/skill <name> [args]`, `/<name> [args]` | Run a skill |
| `/tools` | Show loaded and loadable tools |
| `/knowledge [topic]` | List knowledgepedia topics or show one |
| `/mcp [list\|login <server>\|enable\|disable]` | Manage MCP servers |
| `/agent [list\|show <name>\|delete <name>]` | List, show or delete custom agents |
| `/quit`, `/exit` | Finish the session and exit |

Skills are Claude Code compatible `SKILL.md` folders in `~/.h_agent/skills/<name>/`. Finality ships one built-in skill,
`stubborn` (`/stubborn <request>`): retry failed work with a different approach each time until it is done or blocked by
something only you can resolve. A skill folder with the same name replaces the built-in one.

### Custom agents

Besides the built-in Worker and Research sub-agents, the Orchestrator can create **custom agents**: reusable specialists
with their own instructions on a worker or research base, and optionally a narrower tool list. Ask for one ("create a
reviewer agent that checks for panics and missing tests"), or let the Orchestrator create one when a kind of task keeps
coming back. Creating an agent always needs your approval, whatever the mode, and is not possible in `plan` mode.

Custom agents live in `~/.h_agent/agents/<name>/AGENT.md`, where you can edit them. `/agent` lists them,
`/agent show <name>` prints one, and `/agent delete <name>` removes it. A custom agent can only use worker, research and
MCP tools, and the mode and approvals still apply to everything it does.

## Data directory

```
~/.h_agent/                  # private to your user on Linux and macOS (mode 0700)
├── config.toml, mcp.json
├── projects.json            # known projects, sessions, MCP trust
├── sessions/                # session logs (*.hlog) and index.jsonl; tool output is redacted
├── memory/                  # memory event trees, user_profile.md (editable) and user_profile.pending.md
├── skills/                  # skills; skills/_pending holds learner proposals
├── agents/                  # custom agents (AGENT.md)
├── knowledgepedia/          # learned topic notes
├── lookup/, plans/          # plan scoring data
└── logs/h_agent.log         # rotated daily, 7 files kept; secrets are redacted
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
  External: denied in `plan`, and they need approval in `cowork` and `auto-edit`.
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

CI runs the same checks on Windows, Linux and macOS, plus `cargo deny` and a build with the minimum supported Rust (1.89).
The Linux and macOS jobs also run the OS sandbox tests. Pushing a `v*` tag builds the release archives and their checksums
(see [Releasing](#releasing-maintainers)).

Session logs grow without limit by default; set `general.session_retention_days` or run `/sessions prune <days>` to delete
closed sessions older than a cutoff.

## Roadmap

Platforms:
- [ ] macOS on Intel (`x86_64-apple-darwin`): no pre-built binary yet; install from source.
- [ ] Linux on ARM (`aarch64-unknown-linux-gnu`): no pre-built binary yet; install from source.
- [ ] Older Linux distributions: the Linux binary is linked against a recent glibc; a static `musl` build is planned.
- [ ] Linux and macOS are built and tested in CI but have had little hands-on use so far; reports are welcome.

Distribution:
- [ ] Code-signed Windows binaries and notarized macOS binaries.
- [ ] Package managers: winget, Scoop and Homebrew.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
