//! Shell tool: one fresh, non-interactive process per call with cwd = workdir (DESIGN §9.2).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use h_agent_core::RiskClass;
use h_agent_safety::{CommandAnalyzer, Location, ShellKind};
use process_wrap::tokio::{CommandWrap, KillOnDrop};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::tool::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput, parse_args};

const KILL_GRACE: Duration = Duration::from_secs(5);
const READER_GRACE: Duration = Duration::from_secs(2);
const PS_PRELUDE: &str = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; ";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    command: String,
    shell: Option<String>,
    timeout_secs: Option<u64>,
}

/// Resolved interpreter for a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellProgram {
    pub kind: ShellKind,
    pub path: PathBuf,
    pub name: String,
}

fn find_in_path(name: &str, path_var: Option<&str>) -> Option<PathBuf> {
    let candidates: Vec<String> = if cfg!(windows) && !name.contains('.') {
        vec![format!("{name}.exe"), name.to_owned()]
    } else {
        vec![name.to_owned()]
    };
    let path_var = path_var?;
    std::env::split_paths(path_var).find_map(|dir| {
        candidates.iter().map(|c| dir.join(c)).find(|p| {
            // System32\bash.exe launches WSL, a different OS; never pick it implicitly.
            let wsl = cfg!(windows)
                && p.parent()
                    .and_then(Path::file_name)
                    .is_some_and(|d| d.eq_ignore_ascii_case("system32"))
                && name.eq_ignore_ascii_case("bash");
            p.is_file() && !wsl
        })
    })
}

/// `requested`: `auto`, `bash`, `sh`, `zsh`, `pwsh`, `powershell` or `cmd`.
pub fn resolve_shell(requested: Option<&str>, ctx: &ToolCtx) -> Result<ShellProgram, ToolError> {
    let path_var = ctx.env.snapshot.get("PATH", true);
    let found = |name: &str, kind| {
        find_in_path(name, path_var).map(|path| ShellProgram {
            kind,
            path,
            name: name.to_owned(),
        })
    };
    let wanted = requested.unwrap_or("auto").to_ascii_lowercase();
    let program = match wanted.as_str() {
        "auto" if cfg!(windows) => found("pwsh", ShellKind::PowerShell)
            .or_else(|| found("powershell", ShellKind::PowerShell))
            .or_else(|| found("cmd", ShellKind::Cmd)),
        "auto" => {
            let login = ctx.env.snapshot.get("SHELL", false).map(PathBuf::from);
            login
                .filter(|p| {
                    p.is_file()
                        && p.file_name().is_some_and(|n| {
                            matches!(n.to_string_lossy().as_ref(), "bash" | "zsh" | "sh" | "dash")
                        })
                })
                .map(|path| ShellProgram {
                    kind: ShellKind::Posix,
                    name: path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    path,
                })
                .or_else(|| {
                    Some(PathBuf::from("/bin/sh"))
                        .filter(|p| p.is_file())
                        .map(|path| ShellProgram {
                            kind: ShellKind::Posix,
                            path,
                            name: "sh".into(),
                        })
                })
        }
        "bash" | "sh" | "zsh" | "dash" => found(&wanted, ShellKind::Posix),
        "pwsh" | "powershell" => found(&wanted, ShellKind::PowerShell),
        "cmd" if cfg!(windows) => found("cmd", ShellKind::Cmd),
        "cmd" => {
            return Err(ToolError::InvalidArgs(
                "cmd is only available on Windows".into(),
            ));
        }
        other => return Err(ToolError::InvalidArgs(format!("unknown shell `{other}`"))),
    };
    program.ok_or_else(|| ToolError::Failed(format!("shell `{wanted}` was not found on PATH")))
}

/// Keeps the first and last `cap` bytes of a stream.
struct Capped {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    cap: usize,
    total: usize,
}

impl Capped {
    fn new(cap: usize) -> Self {
        Self {
            head: Vec::new(),
            tail: VecDeque::new(),
            cap: cap.max(1),
            total: 0,
        }
    }

    fn push(&mut self, mut bytes: &[u8]) {
        self.total += bytes.len();
        let room = self.cap.saturating_sub(self.head.len());
        let take = room.min(bytes.len());
        self.head.extend_from_slice(&bytes[..take]);
        bytes = &bytes[take..];
        self.tail.extend(bytes);
        while self.tail.len() > self.cap {
            self.tail.pop_front();
        }
    }

    fn finish(self) -> String {
        let kept = self.head.len() + self.tail.len();
        let tail: Vec<u8> = self.tail.into_iter().collect();
        if kept >= self.total {
            let mut all = self.head;
            all.extend(tail);
            return String::from_utf8_lossy(&all).into_owned();
        }
        format!(
            "{}\n… [{} bytes omitted] …\n{}",
            String::from_utf8_lossy(&self.head),
            self.total - kept,
            String::from_utf8_lossy(&tail)
        )
    }
}

async fn drain(mut r: impl AsyncRead + Unpin, cap: usize) -> String {
    let mut buf = Capped::new(cap);
    let mut chunk = [0u8; 8192];
    while let Ok(n) = r.read(&mut chunk).await {
        if n == 0 {
            break;
        }
        buf.push(&chunk[..n]);
    }
    buf.finish()
}

fn command_for(
    program: &ShellProgram,
    command: &str,
    ctx: &ToolCtx,
    sandboxed: bool,
) -> Result<CommandWrap, ToolError> {
    let workdir = ctx.guard().workdir().to_path_buf();
    let vars = ctx.env.vars.clone();
    let kind = program.kind;
    let command = command.to_owned();
    let (exe, prefix) = match &ctx.env.sandbox {
        Some(s) if sandboxed => s.wrap(&program.path),
        _ => (program.path.clone(), Vec::new()),
    };
    let mut cmd = CommandWrap::with_new(&exe, move |c| {
        c.args(&prefix);
        match kind {
            ShellKind::Posix => {
                c.arg("-c").arg(&command);
            }
            ShellKind::PowerShell => {
                let script: Vec<u8> = format!("{PS_PRELUDE}{command}")
                    .encode_utf16()
                    .flat_map(u16::to_le_bytes)
                    .collect();
                let encoded = base64::engine::general_purpose::STANDARD.encode(script);
                c.args([
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-EncodedCommand",
                    &encoded,
                ]);
            }
            ShellKind::Cmd => {
                #[cfg(windows)]
                c.raw_arg(format!("/D /S /C \"{command}\""));
            }
        }
        c.current_dir(&workdir)
            .env_clear()
            .envs(vars)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    });
    #[cfg(unix)]
    cmd.wrap(process_wrap::tokio::ProcessGroup::leader());
    #[cfg(windows)]
    cmd.wrap(process_wrap::tokio::JobObject);
    cmd.wrap(KillOnDrop);
    Ok(cmd)
}

enum Ending {
    Exited(Option<i32>),
    TimedOut,
    Cancelled,
}

pub struct Shell;

#[async_trait]
impl Tool for Shell {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Run a non-interactive command in a fresh shell process whose working directory is the project root. \
         stdin is closed; the whole process tree is killed on timeout. shell: auto (default: pwsh/powershell \
         on Windows, $SHELL or sh elsewhere), bash, sh, zsh, pwsh, powershell, cmd."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "shell": { "type": "string", "enum": ["auto", "bash", "sh", "zsh", "pwsh", "powershell", "cmd"], "default": "auto" },
                "timeout_secs": { "type": "integer", "minimum": 1 }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Exec
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        if a.command.trim().is_empty() {
            return Err(ToolError::InvalidArgs("command is empty".into()));
        }
        let program = resolve_shell(a.shell.as_deref(), ctx)?;
        let analysis =
            CommandAnalyzer::new(ctx.guard(), &ctx.env.snapshot).analyze(program.kind, &a.command);
        let short: String = a.command.chars().take(80).collect();
        let mut p = ActionPreview::new(
            format!("run `{short}` in {}", program.name),
            analysis.location,
        );
        p.reasons = analysis.reasons();
        p.grant_keys = analysis.grant_keys;
        p.shapes = analysis.shapes;
        p.command = Some(a.command.clone());
        p.detail = Some(format!(
            "{} ({})\n{}",
            program.name,
            program.path.display(),
            a.command
        ));
        p.paths = vec![program.path.clone()];
        p.state = json!({ "kind": format!("{:?}", program.kind), "name": program.name });
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let kind = match preview.state["kind"].as_str() {
            Some("PowerShell") => ShellKind::PowerShell,
            Some("Cmd") => ShellKind::Cmd,
            _ => ShellKind::Posix,
        };
        let program = ShellProgram {
            kind,
            path: preview.paths[0].clone(),
            name: preview.state["name"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        };
        let timeout = a
            .timeout_secs
            .map_or(ctx.limits.shell_timeout, Duration::from_secs)
            .min(ctx.limits.shell_timeout_max);

        // Commands the user approved as reaching outside the working directory run unsandboxed.
        let sandboxed = ctx.env.sandbox.is_some() && preview.location == Location::Inside;
        let mut child = command_for(&program, &a.command, ctx, sandboxed)?.spawn()?;
        let cap = ctx.limits.max_output_chars;
        let stdout = child.stdout().take().map(|s| tokio::spawn(drain(s, cap)));
        let stderr = child.stderr().take().map(|s| tokio::spawn(drain(s, cap)));

        let ending = tokio::select! {
            status = child.wait() => Ending::Exited(status?.code()),
            () = tokio::time::sleep(timeout) => Ending::TimedOut,
            () = ctx.cancel.cancelled() => Ending::Cancelled,
        };
        if !matches!(ending, Ending::Exited(_)) {
            if let Err(e) = child.start_kill() {
                tracing::warn!(error = %e, "failed to kill shell process tree");
            }
            let _ = tokio::time::timeout(KILL_GRACE, child.wait()).await;
        }

        let collect = |h: Option<tokio::task::JoinHandle<String>>| async move {
            match h {
                Some(h) => {
                    let abort = h.abort_handle();
                    match tokio::time::timeout(READER_GRACE, h).await {
                        Ok(Ok(s)) => s,
                        _ => {
                            abort.abort();
                            String::new()
                        }
                    }
                }
                None => String::new(),
            }
        };
        let (out, err) = (collect(stdout).await, collect(stderr).await);

        let (header, failed) = match ending {
            Ending::Exited(Some(0)) => ("exit code: 0".to_owned(), false),
            Ending::Exited(Some(c)) => (format!("exit code: {c}"), true),
            Ending::Exited(None) => ("terminated by a signal".to_owned(), true),
            Ending::TimedOut => (
                format!(
                    "timed out after {}s; process tree killed",
                    timeout.as_secs()
                ),
                true,
            ),
            Ending::Cancelled => ("cancelled; process tree killed".to_owned(), true),
        };
        let mut content = header;
        for (label, text) in [("stdout", out), ("stderr", err)] {
            if !text.trim().is_empty() {
                content.push_str(&format!("\n--- {label} ---\n{}", text.trim_end()));
            }
        }
        if sandboxed && failed {
            content.push_str(
                "\n--- note ---\nThis command ran in the OS sandbox, which only allows writes inside the \
                 working directory and temp folders. If it failed because it wrote elsewhere, say so; the \
                 user must approve such writes explicitly.",
            );
        }
        Ok(ToolOutput {
            content,
            is_error: failed,
            truncated: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_keeps_head_and_tail() {
        let mut c = Capped::new(4);
        c.push(b"abcdefghij");
        c.push(b"klm");
        let s = c.finish();
        assert!(
            s.starts_with("abcd") && s.ends_with("jklm") && s.contains("5 bytes omitted"),
            "{s}"
        );
        let mut small = Capped::new(10);
        small.push(b"abc");
        small.push(b"def");
        assert_eq!(small.finish(), "abcdef");
    }
}
