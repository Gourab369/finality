//! Git: the read-only `git` tool, the deferred `git_commit` tool, and helpers for the header branch and
//! `/diff`. Every call disables repository-configured programs (fsmonitor, external diff, textconv) so a
//! read cannot run code from the repository's config, and is limited to the working directory by pathspec.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_safety::Location;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, parse_args, resolve_path,
    truncate_middle,
};

const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Commit hooks may run tests.
const COMMIT_TIMEOUT: Duration = Duration::from_secs(300);
const SAFE_OPTS: &[&str] = &[
    "--no-pager",
    "-c",
    "color.ui=never",
    "-c",
    "core.quotepath=off",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.pager=cat",
];
const DEFAULT_LOG: u32 = 20;
const MAX_LOG: u32 = 200;
const DIFF_CHARS: usize = 8000;

/// Output of a finished git process.
pub struct GitOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `git -C workdir <SAFE_OPTS> args`. `vars` replaces the environment when given.
pub async fn run_git(
    workdir: &Path,
    args: &[String],
    vars: Option<&[(String, String)]>,
    timeout: Duration,
    cancel: &CancellationToken,
) -> Result<GitOutput, String> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("-C")
        .arg(workdir)
        .args(SAFE_OPTS)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(vars) = vars {
        cmd.env_clear()
            .envs(vars.iter().cloned())
            .env("GIT_TERMINAL_PROMPT", "0");
    }
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let child = cmd
        .spawn()
        .map_err(|e| format!("cannot run git (is it installed and on PATH?): {e}"))?;
    let out = tokio::select! {
        out = tokio::time::timeout(timeout, child.wait_with_output()) => match out {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => return Err(format!("git failed: {e}")),
            Err(_) => return Err(format!("git timed out after {}s", timeout.as_secs())),
        },
        () = cancel.cancelled() => return Err("cancelled".into()),
    };
    Ok(GitOutput {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    })
}

/// The checked-out branch (or short commit when detached) of the repository containing `dir`, read from
/// `.git/HEAD` without running git.
pub fn current_branch(dir: &Path) -> Option<String> {
    let mut at = Some(dir);
    while let Some(d) = at {
        let dot_git = d.join(".git");
        let git_dir = if dot_git.is_dir() {
            Some(dot_git)
        } else if dot_git.is_file() {
            // Worktrees and submodules: "gitdir: <path>".
            let text = std::fs::read_to_string(&dot_git).ok()?;
            let target = PathBuf::from(text.trim().strip_prefix("gitdir:")?.trim());
            Some(if target.is_absolute() {
                target
            } else {
                d.join(target)
            })
        } else {
            None
        };
        if let Some(git_dir) = git_dir {
            let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
            let head = head.trim();
            return Some(match head.strip_prefix("ref:") {
                Some(r) => {
                    let r = r.trim();
                    r.strip_prefix("refs/heads/").unwrap_or(r).to_owned()
                }
                None => head.chars().take(7).collect(),
            });
        }
        at = d.parent();
    }
    None
}

/// `/diff`: status plus the uncommitted diff against HEAD, limited to `workdir`.
pub async fn diff_summary(workdir: &Path) -> Result<String, String> {
    let cancel = CancellationToken::new();
    let run = |args: &[&str]| {
        let args: Vec<String> = args.iter().map(|a| (*a).to_owned()).collect();
        let cancel = cancel.clone();
        async move { run_git(workdir, &args, None, READ_TIMEOUT, &cancel).await }
    };
    let status = run(&["status", "--short", "--branch", "--", "."]).await?;
    if !status.success {
        return Err(status.stderr.trim().to_owned());
    }
    let mut text = status.stdout.trim_end().to_owned();
    let diff = run(&["diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", "."]).await?;
    // A repository without commits has no HEAD; the status alone is still useful.
    if diff.success && !diff.stdout.trim().is_empty() {
        text.push_str("\n\n");
        text.push_str(&truncate_middle(diff.stdout.trim_end(), DIFF_CHARS).0);
    } else if diff.success {
        text.push_str("\n(no uncommitted changes)");
    }
    Ok(text)
}

/// A revision argument: no options, no `rev:path` (which could read files outside the working directory).
fn check_rev(rev: &str) -> Result<(), ToolError> {
    let ok = !rev.is_empty()
        && rev.len() <= 200
        && !rev.starts_with('-')
        && rev
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._/~^@{}-+".contains(c));
    if ok {
        Ok(())
    } else {
        Err(ToolError::InvalidArgs(format!(
            "`{rev}` is not a valid revision (use a commit, branch, tag, or forms like HEAD~2)"
        )))
    }
}

fn output(out: GitOutput, max_chars: usize) -> ToolOutput {
    if !out.success {
        let msg = if out.stderr.trim().is_empty() {
            out.stdout
        } else {
            out.stderr
        };
        return ToolOutput::error(truncate_middle(msg.trim(), max_chars).0);
    }
    let text = out.stdout.trim_end();
    if text.is_empty() {
        return ToolOutput::ok("(no output)");
    }
    let (text, truncated) = truncate_middle(text, max_chars);
    ToolOutput {
        content: text,
        is_error: false,
        truncated,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GitArgs {
    op: String,
    rev: Option<String>,
    path: Option<String>,
    #[serde(default)]
    staged: bool,
    limit: Option<u32>,
    start_line: Option<u32>,
    end_line: Option<u32>,
}

/// `git`: read-only repository queries.
pub struct Git;

#[async_trait]
impl Tool for Git {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Read-only git queries in the working directory. op: status; diff {staged?, rev?, path?} (uncommitted \
         changes, or against rev); log {rev?, path?, limit?}; show {rev, path?}; blame {path, start_line?, \
         end_line?, rev?}; branch (all branches)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["status", "diff", "log", "show", "blame", "branch"] },
                "rev": { "type": "string" },
                "path": { "type": "string" },
                "staged": { "type": "boolean" },
                "limit": { "type": "integer", "minimum": 1, "maximum": MAX_LOG },
                "start_line": { "type": "integer", "minimum": 1 },
                "end_line": { "type": "integer", "minimum": 1 }
            },
            "required": ["op"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Read
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: GitArgs = parse_args(args)?;
        if let Some(rev) = &a.rev {
            check_rev(rev)?;
        }
        let (location, path, reasons) = match &a.path {
            Some(p) => {
                let (l, path, r) = resolve_path(ctx, p);
                (l, Some(path), r)
            }
            None => (Location::Inside, None, Vec::new()),
        };
        let mut git: Vec<String> = Vec::new();
        let push = |git: &mut Vec<String>, items: &[&str]| {
            git.extend(items.iter().map(|s| (*s).to_owned()));
        };
        match a.op.as_str() {
            "status" => push(&mut git, &["status", "--short", "--branch"]),
            "diff" => {
                push(&mut git, &["diff", "--no-ext-diff", "--no-textconv"]);
                if a.staged {
                    git.push("--staged".into());
                }
            }
            "log" => {
                let n = a.limit.unwrap_or(DEFAULT_LOG).clamp(1, MAX_LOG);
                push(
                    &mut git,
                    &["log", "--date=short", "--format=%h %ad %an  %s", "-n"],
                );
                git.push(n.to_string());
            }
            "show" => {
                if a.rev.is_none() {
                    return Err(ToolError::InvalidArgs("show needs `rev`".into()));
                }
                push(
                    &mut git,
                    &[
                        "show",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--stat",
                        "--patch",
                    ],
                );
            }
            "blame" => {
                if path.is_none() {
                    return Err(ToolError::InvalidArgs("blame needs `path`".into()));
                }
                push(&mut git, &["blame", "--no-textconv"]);
                if let Some(start) = a.start_line {
                    git.push("-L".into());
                    git.push(match a.end_line {
                        Some(end) => format!("{start},{end}"),
                        None => format!("{start},"),
                    });
                }
            }
            "branch" => push(&mut git, &["branch", "--all", "-vv"]),
            other => return Err(ToolError::InvalidArgs(format!("unknown op `{other}`"))),
        }
        if let Some(rev) = &a.rev
            && a.op != "status"
            && a.op != "branch"
        {
            git.push(rev.clone());
        }
        if a.op != "branch" {
            git.push("--".into());
            git.push(match &path {
                Some(p) => p.to_string_lossy().into_owned(),
                // Only what is under the working directory, even when the repository is larger.
                None => ".".into(),
            });
        }
        let mut p = ActionPreview::new(format!("git {}", git[0]), location);
        p.reasons = reasons;
        p.paths = path.into_iter().collect();
        p.detail = Some(format!("git {}", git.join(" ")));
        p.state = json!(git);
        Ok(p)
    }

    async fn call(
        &self,
        _args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let git: Vec<String> = serde_json::from_value(preview.state.clone())
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        let out = run_git(
            ctx.guard().workdir(),
            &git,
            Some(&ctx.env.vars),
            READ_TIMEOUT,
            &ctx.cancel,
        )
        .await
        .map_err(ToolError::Failed)?;
        Ok(output(out, ctx.limits.max_output_chars))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommitArgs {
    message: String,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    all: bool,
}

/// `git_commit`: stage the given paths (or all tracked changes) and commit. Runs the repository's hooks.
pub struct GitCommit;

#[async_trait]
impl Tool for GitCommit {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Commit in the working directory's git repository. paths: files to stage first; all: stage every \
         tracked change (git commit -a); neither: commit what is already staged. Commit hooks run. Only \
         commit when the user asked for it."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" },
                "paths": { "type": "array", "items": { "type": "string" } },
                "all": { "type": "boolean" }
            },
            "required": ["message"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Exec
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: CommitArgs = parse_args(args)?;
        if a.message.trim().is_empty() {
            return Err(ToolError::InvalidArgs("message is empty".into()));
        }
        if a.all && !a.paths.is_empty() {
            return Err(ToolError::InvalidArgs(
                "pass either paths or all, not both".into(),
            ));
        }
        let mut location = Location::Inside;
        let mut reasons = Vec::new();
        let mut paths = Vec::new();
        for raw in &a.paths {
            let (l, p, r) = resolve_path(ctx, raw);
            if l != Location::Inside {
                location = l;
            }
            reasons.extend(r);
            paths.push(p);
        }
        let first: String = a
            .message
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(60)
            .collect();
        let staged = if a.all {
            "all tracked changes".to_owned()
        } else if paths.is_empty() {
            "what is already staged".to_owned()
        } else {
            a.paths.join(", ")
        };
        let mut p = ActionPreview::new(format!("git commit \"{first}\""), location);
        p.reasons = reasons;
        p.paths = paths;
        p.detail = Some(format!("commit {staged}\n\n{}", a.message));
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: CommitArgs = parse_args(&args)?;
        let workdir = ctx.guard().workdir();
        let run = |git: Vec<String>, timeout| async move {
            run_git(workdir, &git, Some(&ctx.env.vars), timeout, &ctx.cancel)
                .await
                .map_err(ToolError::Failed)
        };
        if !preview.paths.is_empty() {
            let mut add = vec!["add".to_owned(), "--".to_owned()];
            add.extend(
                preview
                    .paths
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned()),
            );
            let out = run(add, READ_TIMEOUT).await?;
            if !out.success {
                return Ok(output(out, ctx.limits.max_output_chars));
            }
        }
        let mut commit = vec!["commit".to_owned()];
        if a.all {
            commit.push("-a".into());
        }
        commit.push("-m".into());
        commit.push(a.message);
        let out = run(commit, COMMIT_TIMEOUT).await?;
        let mut result = output(
            GitOutput {
                stdout: format!("{}{}", out.stdout, out.stderr),
                ..out
            },
            ctx.limits.max_output_chars,
        );
        if result.is_error && result.content.is_empty() {
            result.content = "git commit failed".into();
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisions_cannot_be_options_or_paths() {
        assert!(check_rev("HEAD~2").is_ok());
        assert!(check_rev("origin/main").is_ok());
        assert!(check_rev("v1.2.3").is_ok());
        assert!(check_rev("--output=x").is_err());
        assert!(check_rev("HEAD:../secret").is_err());
        assert!(check_rev("a b").is_err());
    }

    #[test]
    fn branch_comes_from_head() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("a/b");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(current_branch(&sub), None);
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join(".git/HEAD"), "ref: refs/heads/feature/x\n").unwrap();
        assert_eq!(current_branch(&sub).as_deref(), Some("feature/x"));
        std::fs::write(tmp.path().join(".git/HEAD"), "0123456789abcdef\n").unwrap();
        assert_eq!(current_branch(&sub).as_deref(), Some("0123456"));
    }
}
