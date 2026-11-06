use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use finality_core::config::ToolsConfig;
use finality_core::{AgentId, RiskClass};
use finality_safety::{CommandShape, EnvSnapshot, Location, PathGuard};
use finality_sandbox::Sandbox;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Always in the agent's context.
    Core,
    /// Loaded on demand (DESIGN §9.3).
    Deferred,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("{0}")]
    Failed(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub truncated: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            truncated: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            truncated: false,
        }
    }
}

/// What a call will do, computed before authorization. All path arguments are resolved here.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPreview {
    pub summary: String,
    pub location: Location,
    pub reasons: Vec<String>,
    /// Resolved paths the call will touch, in tool-defined order.
    pub paths: Vec<PathBuf>,
    /// Diff or command analysis shown in the approval prompt.
    pub detail: Option<String>,
    /// Shell command text for always-ask matching.
    pub command: Option<String>,
    /// The command's simple commands, for option-order-independent always-ask matching.
    pub shapes: Vec<CommandShape>,
    /// Set when the action must be confirmed in every mode but plan; the reason shown to the user.
    pub always_ask: Option<String>,
    pub grant_keys: Vec<String>,
    /// Tool-private data carried from preview to call.
    pub state: Value,
}

impl ActionPreview {
    pub fn new(summary: impl Into<String>, location: Location) -> Self {
        Self {
            summary: summary.into(),
            location,
            reasons: Vec::new(),
            paths: Vec::new(),
            detail: None,
            command: None,
            shapes: Vec::new(),
            always_ask: None,
            grant_keys: Vec::new(),
            state: Value::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> Value;
    fn risk(&self) -> RiskClass;
    fn tier(&self) -> Tier {
        Tier::Core
    }
    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError>;
    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError>;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            parameters: self.schema(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolLimits {
    pub shell_timeout: Duration,
    pub shell_timeout_max: Duration,
    pub max_output_chars: usize,
    pub read_max_lines: usize,
}

impl ToolLimits {
    pub fn from_config(cfg: &ToolsConfig) -> Self {
        Self {
            shell_timeout: Duration::from_secs(cfg.shell_timeout_secs),
            shell_timeout_max: Duration::from_secs(cfg.shell_timeout_max_secs),
            max_output_chars: cfg.max_output_chars,
            read_max_lines: cfg.read_max_lines,
        }
    }
}

impl Default for ToolLimits {
    fn default() -> Self {
        Self::from_config(&ToolsConfig::default())
    }
}

/// Per-workspace state shared by every tool call.
#[derive(Debug)]
pub struct ToolEnv {
    pub guard: PathGuard,
    /// Environment passed to child processes, with `safety.env_denylist` removed.
    pub vars: Vec<(String, String)>,
    pub snapshot: EnvSnapshot,
    /// OS sandbox for shell commands classified as inside the working directory (`safety.os_sandbox`).
    pub sandbox: Option<Sandbox>,
}

impl ToolEnv {
    pub fn new(
        guard: PathGuard,
        vars: impl IntoIterator<Item = (String, String)>,
        denylist: &[String],
    ) -> Self {
        let vars = filter_env(vars, denylist);
        let snapshot = EnvSnapshot::from_pairs(vars.clone());
        Self {
            guard,
            vars,
            snapshot,
            sandbox: None,
        }
    }

    pub fn with_sandbox(mut self, sandbox: Sandbox) -> Self {
        self.sandbox = Some(sandbox);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ToolCtx {
    pub env: Arc<ToolEnv>,
    pub limits: ToolLimits,
    pub cancel: CancellationToken,
    pub agent: AgentId,
}

impl ToolCtx {
    pub fn guard(&self) -> &PathGuard {
        &self.env.guard
    }
}

/// Removes variables matching any `*`-wildcard pattern (case-insensitive).
pub fn filter_env(
    vars: impl IntoIterator<Item = (String, String)>,
    denylist: &[String],
) -> Vec<(String, String)> {
    vars.into_iter()
        .filter(|(k, _)| {
            !denylist
                .iter()
                .any(|p| wildcard_match(&p.to_ascii_uppercase(), &k.to_ascii_uppercase()))
        })
        .collect()
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !text.starts_with(first) || !text.ends_with(last) || text.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for mid in &parts[1..parts.len() - 1] {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

pub fn parse_args<T: DeserializeOwned>(args: &Value) -> Result<T, ToolError> {
    serde_json::from_value(args.clone()).map_err(|e| ToolError::InvalidArgs(e.to_string()))
}

/// Keeps the head and tail of `text` within `max_chars`.
pub fn truncate_middle(text: &str, max_chars: usize) -> (String, bool) {
    let total = text.chars().count();
    if total <= max_chars {
        return (text.to_owned(), false);
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let start: String = text.chars().take(head).collect();
    let end: String = text.chars().skip(total - tail).collect();
    (
        format!(
            "{start}\n… [{} characters omitted] …\n{end}",
            total - head - tail
        ),
        true,
    )
}

/// Workdir-relative display with `/` separators.
pub fn display_path(env: &ToolEnv, path: &std::path::Path) -> String {
    path.strip_prefix(env.guard.workdir())
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.display().to_string())
}

/// Resolves a path argument through the path guard for a preview.
pub fn resolve_path(ctx: &ToolCtx, raw: &str) -> (Location, PathBuf, Vec<String>) {
    match ctx.guard().check(raw) {
        finality_safety::PathVerdict::Inside { resolved } => {
            (Location::Inside, resolved, Vec::new())
        }
        finality_safety::PathVerdict::Outside { resolved, reason } => (
            Location::Outside,
            resolved.unwrap_or_else(|| PathBuf::from(raw)),
            vec![format!("`{raw}` {reason}")],
        ),
    }
}

/// Decodes text, honoring UTF-8/UTF-16 BOMs. `None` means binary.
pub fn decode_text(bytes: &[u8]) -> Option<String> {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return Some(String::from_utf8_lossy(rest).into_owned());
    }
    let utf16 = |rest: &[u8], le: bool| {
        let units: Vec<u16> = rest
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| {
                if le {
                    u16::from_le_bytes(c)
                } else {
                    u16::from_be_bytes(c)
                }
            })
            .collect();
        String::from_utf16_lossy(&units)
    };
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return Some(utf16(rest, true));
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return Some(utf16(rest, false));
    }
    if bytes[..bytes.len().min(8192)].contains(&0) {
        return None;
    }
    Some(String::from_utf8_lossy(bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_denylist_wildcards() {
        let vars = [
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
            "aws_region",
            "MY_SECRET",
            "PATH",
            "TOKENIZER",
            "HOME",
        ]
        .map(|k| (k.to_owned(), "v".to_owned()));
        let deny = ["*_API_KEY", "*_TOKEN", "*_SECRET", "AWS_*"].map(String::from);
        let kept: Vec<String> = filter_env(vars, &deny)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(kept, vec!["PATH", "TOKENIZER", "HOME"]);
        assert!(wildcard_match("A*B*C", "AXXBYYC"));
        assert!(!wildcard_match("A*B*C", "AXXC"));
        assert!(wildcard_match("*", ""));
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let (t, cut) = truncate_middle("abcdefghij", 4);
        assert!(cut);
        assert!(t.starts_with("ab") && t.ends_with("ij") && t.contains("6 characters omitted"));
        assert_eq!(truncate_middle("abc", 4), ("abc".to_owned(), false));
    }
}
