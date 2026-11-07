//! Configuration model (DESIGN §4.6). Precedence: CLI > env (`FINALITY_<SECTION>__<KEY>`) > file > defaults.
//! The CLI layer is applied by the binary as typed mutations after [`ConfigLoader::load`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use toml::{Table, Value};

use crate::{Mode, ProviderKind, ThinkLevel};

pub const ENV_PREFIX: &str = "FINALITY_";

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid config {origin}: {message}")]
    Parse { origin: String, message: String },
    #[error("invalid config value `{key}`: {message}")]
    Invalid { key: &'static str, message: String },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: GeneralConfig,
    pub provider: ProviderConfig,
    pub models: ModelsConfig,
    pub llm: LlmConfig,
    pub thinking: ThinkingConfig,
    pub agents: AgentsConfig,
    pub retry: RetryConfig,
    pub memory: MemoryConfig,
    pub tools: ToolsConfig,
    pub safety: SafetyConfig,
    pub web: WebConfig,
    pub browser: BrowserConfig,
    pub godot: GodotConfig,
    pub mcp: McpConfig,
    pub decision: DecisionConfig,
    pub scorer: ScorerConfig,
    pub learner: LearnerConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralConfig {
    pub default_mode: Mode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<PathBuf>,
    /// Delete closed sessions older than this many days at startup; 0 keeps everything.
    pub session_retention_days: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderConfig {
    pub default: ProviderKind,
    pub ollama: OllamaConfig,
    pub openai: OpenAiConfig,
    pub anthropic: AnthropicConfig,
    pub llamacpp: LlamaCppConfig,
}

impl ProviderConfig {
    pub fn model_for(&self, kind: ProviderKind) -> &str {
        match kind {
            ProviderKind::Ollama => &self.ollama.model,
            ProviderKind::OpenAi => &self.openai.model,
            ProviderKind::Anthropic => &self.anthropic.model,
            ProviderKind::LlamaCpp => &self.llamacpp.model,
        }
    }

    pub fn set_model(&mut self, kind: ProviderKind, model: String) {
        match kind {
            ProviderKind::Ollama => self.ollama.model = model,
            ProviderKind::OpenAi => self.openai.model = model,
            ProviderKind::Anthropic => self.anthropic.model = model,
            ProviderKind::LlamaCpp => self.llamacpp.model = model,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
    pub num_ctx: u32,
    pub keep_alive: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:11434".into(),
            model: "qwen3:14b".into(),
            num_ctx: 32768,
            keep_alive: "10m".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiConfig {
    pub base_url: String,
    /// Empty for local OpenAI-compatible servers that need no key.
    pub api_key_env: String,
    pub model: String,
    /// The API does not report context length.
    pub context_window: u32,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            model: String::new(),
            context_window: 128_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicConfig {
    pub base_url: String,
    pub api_key_env: String,
    pub model: String,
    pub context_window: u32,
    /// `max_tokens` per request; raised above the thinking budget when needed.
    pub max_tokens: u32,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.anthropic.com".into(),
            api_key_env: "ANTHROPIC_API_KEY".into(),
            model: String::new(),
            context_window: 200_000,
            max_tokens: 8192,
        }
    }
}

/// llama.cpp: Finality starts `llama-server` with a `.gguf` file from `models_dir`, or uses a server already
/// running at `base_url`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCppConfig {
    /// Folder searched (three levels deep) for `.gguf` files; models are named by their path inside it.
    pub models_dir: String,
    /// A `.gguf` path relative to `models_dir`, or an absolute path.
    pub model: String,
    /// `llama-server` executable; a bare name is looked up on PATH.
    pub server_path: String,
    /// When set, use this running server (e.g. `http://127.0.0.1:8080`) instead of starting one.
    pub base_url: String,
    /// Port for the server Finality starts; 0 picks a free one.
    pub port: u16,
    /// `--ctx-size`.
    pub context_window: u32,
    /// `--n-gpu-layers`; unset leaves it to llama.cpp.
    pub gpu_layers: Option<i32>,
    /// More `llama-server` arguments, added last (so they win over Finality's).
    pub extra_args: Vec<String>,
    /// How long loading a model may take.
    pub startup_timeout_secs: u64,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            models_dir: String::new(),
            model: String::new(),
            server_path: "llama-server".into(),
            base_url: String::new(),
            port: 0,
            context_window: 16_384,
            gpu_layers: None,
            extra_args: Vec::new(),
            startup_timeout_secs: 300,
        }
    }
}

/// Per-role model overrides in `provider:model` form; empty means the provider default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    pub orchestrator: String,
    pub worker: String,
    pub research: String,
    pub memory: String,
    pub scorer: String,
    pub learner: String,
    /// Empty: `memory` when pinned, else the orchestrator's model.
    pub triage: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    pub max_concurrent_requests: u32,
    pub request_timeout_secs: u64,
    /// Classify obviously simple messages (greetings, short questions and commands) without a triage request.
    pub quick_triage: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 2,
            request_timeout_secs: 300,
            quick_triage: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkingConfig {
    pub level: ThinkLevel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentsConfig {
    pub max_subagents: u32,
    pub max_workers: u32,
    pub max_research: u32,
    pub heartbeat_interval_secs: u64,
    pub stall_after_missed_beats: u32,
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            max_subagents: 4,
            max_workers: 2,
            max_research: 3,
            heartbeat_interval_secs: 5,
            stall_after_missed_beats: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub task_token_budget: u64,
    pub task_time_budget_secs: u64,
    pub plan_time_budget_secs: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 7,
            task_token_budget: 200_000,
            task_time_budget_secs: 900,
            plan_time_budget_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub user_exchange_trigger: u32,
    pub subagent_exchange_trigger: u32,
    pub compaction_threshold: f64,
    pub digest_token_budget: u32,
    /// How long quitting waits for the Memory agent to summarize the session into events.
    pub quit_wait_secs: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            user_exchange_trigger: 7,
            subagent_exchange_trigger: 10,
            compaction_threshold: 0.80,
            digest_token_budget: 1500,
            quit_wait_secs: 300,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    pub shell_timeout_secs: u64,
    pub shell_timeout_max_secs: u64,
    pub max_output_chars: usize,
    pub read_max_lines: usize,
    pub offload_after_turns: u32,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            shell_timeout_secs: 120,
            shell_timeout_max_secs: 600,
            max_output_chars: 30_000,
            read_max_lines: 2000,
            offload_after_turns: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SafetyConfig {
    pub always_ask_patterns: Vec<String>,
    pub env_denylist: Vec<String>,
    /// Run shell commands classified as inside the working directory in an OS sandbox that only
    /// allows writes there (Linux Landlock, macOS sandbox-exec; no effect on Windows).
    pub os_sandbox: bool,
    /// Extra writable directories for the sandbox (relative paths are under the working directory).
    pub sandbox_writable: Vec<String>,
    /// `allow` (default) or `deny`: with `deny`, sandboxed commands cannot bind or connect over TCP
    /// except to `sandbox_tcp_ports`. Needs Linux 6.7+ (Landlock ABI 4) or macOS.
    pub sandbox_network: String,
    /// Outbound TCP ports still allowed when `sandbox_network = "deny"`.
    pub sandbox_tcp_ports: Vec<u16>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            always_ask_patterns: [
                "git push --force",
                "rm -rf",
                "Remove-Item -Recurse -Force",
                "format ",
                "shutdown",
                "mkfs",
            ]
            .map(String::from)
            .to_vec(),
            env_denylist: [
                "*_API_KEY",
                "*_TOKEN",
                "*SECRET*",
                "*PASSWORD*",
                "*CREDENTIAL*",
                "*PRIVATE_KEY*",
                "DATABASE_URL",
                "AWS_*",
            ]
            .map(String::from)
            .to_vec(),
            os_sandbox: false,
            sandbox_writable: Vec::new(),
            sandbox_network: "allow".into(),
            sandbox_tcp_ports: vec![443, 80],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchBackend {
    /// Tavily when a key is available, else DuckDuckGo; Tavily errors fall back to DuckDuckGo.
    #[default]
    Auto,
    Tavily,
    DuckDuckGo,
    SearXng,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    pub search_backend: SearchBackend,
    pub searxng_url: String,
    /// Environment variable with the Tavily API key; else the OS keyring (service `finality`, user `api-key:tavily`).
    pub tavily_api_key_env: String,
    pub allow_private_network: bool,
    pub user_agent: String,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            search_backend: SearchBackend::Auto,
            searxng_url: String::new(),
            tavily_api_key_env: "TAVILY_API_KEY".into(),
            allow_private_network: false,
            user_agent: format!("{}/{}", crate::APP_NAME, env!("CARGO_PKG_VERSION")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BrowserConfig {
    pub chrome_path: String,
    pub headless: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            chrome_path: String::new(),
            headless: true,
        }
    }
}

/// The `godot` tool, offered when the working directory holds a Godot project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GodotConfig {
    pub enabled: bool,
    /// The editor's GDScript language server (Editor Settings → Network → Language Server).
    pub lsp_port: u16,
}

impl Default for GodotConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lsp_port: 6005,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    pub enabled: bool,
    /// How long a server may take to start and list its tools.
    pub startup_timeout_secs: u64,
    /// How long one tool call may take before the agent gets an error.
    pub call_timeout_secs: u64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            startup_timeout_secs: 30,
            call_timeout_secs: 120,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecisionConfig {
    pub enabled: bool,
    pub model: String,
    pub path: String,
    pub min_confidence: f64,
    /// A slower answer is dropped and the LLM decides instead.
    pub timeout_secs: u64,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "d1-omni".into(),
            path: "/v1/systemone".into(),
            min_confidence: 0.75,
            timeout_secs: 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScorerWeights {
    pub success: f64,
    pub speed: f64,
    pub retries: f64,
}

impl Default for ScorerWeights {
    fn default() -> Self {
        Self {
            success: 0.60,
            speed: 0.25,
            retries: 0.15,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScorerConfig {
    pub weights: ScorerWeights,
    pub reuse_min_score: f64,
    pub reuse_min_runs: u32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            weights: ScorerWeights::default(),
            reuse_min_score: 0.70,
            reuse_min_runs: 2,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearnerConfig {
    pub auto_approve_skills: bool,
    /// Write learned user-profile notes straight into the profile instead of holding them for `/memory review`.
    pub auto_approve_profile: bool,
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        fn check(ok: bool, key: &'static str, message: &str) -> Result<(), ConfigError> {
            if ok {
                Ok(())
            } else {
                Err(ConfigError::Invalid {
                    key,
                    message: message.into(),
                })
            }
        }
        let unit = |v: f64| (0.0..=1.0).contains(&v);
        check(
            self.provider.ollama.num_ctx >= 512,
            "provider.ollama.num_ctx",
            "must be >= 512",
        )?;
        check(
            self.provider.llamacpp.context_window >= 512,
            "provider.llamacpp.context_window",
            "must be >= 512",
        )?;
        check(
            self.provider.llamacpp.startup_timeout_secs >= 1,
            "provider.llamacpp.startup_timeout_secs",
            "must be >= 1",
        )?;
        check(
            self.llm.max_concurrent_requests >= 1,
            "llm.max_concurrent_requests",
            "must be >= 1",
        )?;
        check(
            self.llm.request_timeout_secs >= 1,
            "llm.request_timeout_secs",
            "must be >= 1",
        )?;
        check(
            self.mcp.startup_timeout_secs >= 1 && self.mcp.call_timeout_secs >= 1,
            "mcp.call_timeout_secs",
            "timeouts must be >= 1",
        )?;
        check(
            self.decision.timeout_secs >= 1,
            "decision.timeout_secs",
            "must be >= 1",
        )?;
        check(
            matches!(self.safety.sandbox_network.as_str(), "allow" | "deny"),
            "safety.sandbox_network",
            "must be `allow` or `deny`",
        )?;
        check(
            self.agents.max_subagents >= 1,
            "agents.max_subagents",
            "must be >= 1",
        )?;
        check(
            self.agents.heartbeat_interval_secs >= 1,
            "agents.heartbeat_interval_secs",
            "must be >= 1",
        )?;
        check(
            self.agents.stall_after_missed_beats >= 1,
            "agents.stall_after_missed_beats",
            "must be >= 1",
        )?;
        check(
            self.memory.compaction_threshold > 0.0 && self.memory.compaction_threshold <= 1.0,
            "memory.compaction_threshold",
            "must be in (0, 1]",
        )?;
        check(
            self.memory.user_exchange_trigger >= 1,
            "memory.user_exchange_trigger",
            "must be >= 1",
        )?;
        check(
            self.memory.subagent_exchange_trigger >= 1,
            "memory.subagent_exchange_trigger",
            "must be >= 1",
        )?;
        check(
            self.tools.shell_timeout_secs <= self.tools.shell_timeout_max_secs,
            "tools.shell_timeout_secs",
            "must be <= tools.shell_timeout_max_secs",
        )?;
        check(
            unit(self.decision.min_confidence),
            "decision.min_confidence",
            "must be in [0, 1]",
        )?;
        check(
            unit(self.scorer.reuse_min_score),
            "scorer.reuse_min_score",
            "must be in [0, 1]",
        )?;
        let w = &self.scorer.weights;
        check(
            unit(w.success) && unit(w.speed) && unit(w.retries),
            "scorer.weights",
            "each weight must be in [0, 1]",
        )?;
        check(
            self.web.search_backend != SearchBackend::SearXng || !self.web.searxng_url.is_empty(),
            "web.searxng_url",
            "required when web.search_backend = \"searxng\"",
        )?;
        Ok(())
    }
}

/// Builds a [`Config`] from an optional TOML file overlaid with `FINALITY_<SECTION>__<KEY>` env vars.
#[derive(Debug, Default)]
pub struct ConfigLoader {
    file: Option<PathBuf>,
    env: Vec<(String, String)>,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self::default()
    }

    /// A missing file is treated as empty.
    pub fn file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file = Some(path.into());
        self
    }

    pub fn env_vars(mut self, vars: impl IntoIterator<Item = (String, String)>) -> Self {
        self.env = vars.into_iter().collect();
        self
    }

    pub fn load(self) -> Result<Config, ConfigError> {
        let mut table = match &self.file {
            Some(path) => read_table(path)?,
            None => Table::new(),
        };
        for (key, raw) in &self.env {
            if let Some(path) = env_key_path(key) {
                set_path(&mut table, &path, parse_env_value(raw), key)?;
            }
        }
        let config: Config = Value::Table(table)
            .try_into()
            .map_err(|e: toml::de::Error| ConfigError::Parse {
                origin: self
                    .file
                    .as_ref()
                    .map_or_else(|| "(env)".into(), |p| p.display().to_string()),
                message: e.message().to_owned(),
            })?;
        config.validate()?;
        Ok(config)
    }
}

fn read_table(path: &Path) -> Result<Table, ConfigError> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Table::new()),
        Err(source) => {
            return Err(ConfigError::Read {
                path: path.to_owned(),
                source,
            });
        }
    };
    toml::from_str(&text).map_err(|e| ConfigError::Parse {
        origin: path.display().to_string(),
        message: e.to_string(),
    })
}

/// `FINALITY_PROVIDER__OLLAMA__MODEL` → `["provider", "ollama", "model"]`. Vars without `__` are not config keys.
fn env_key_path(key: &str) -> Option<Vec<String>> {
    let rest = key.strip_prefix(ENV_PREFIX)?;
    if !rest.contains("__") {
        return None;
    }
    let parts: Vec<String> = rest.split("__").map(str::to_ascii_lowercase).collect();
    parts.iter().all(|p| !p.is_empty()).then_some(parts)
}

/// Values that parse as TOML literals (numbers, bools, arrays) keep their type; anything else is a string.
fn parse_env_value(raw: &str) -> Value {
    toml::from_str::<Table>(&format!("v = {raw}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| Value::String(raw.to_owned()))
}

fn set_path(
    table: &mut Table,
    path: &[String],
    value: Value,
    origin: &str,
) -> Result<(), ConfigError> {
    let (last, parents) = path.split_last().ok_or_else(|| ConfigError::Parse {
        origin: origin.into(),
        message: "empty key".into(),
    })?;
    let mut cur = table;
    for part in parents {
        let entry = cur
            .entry(part.clone())
            .or_insert_with(|| Value::Table(Table::new()));
        cur = entry.as_table_mut().ok_or_else(|| ConfigError::Parse {
            origin: origin.into(),
            message: format!("`{part}` is not a table"),
        })?;
    }
    cur.insert(last.clone(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let cfg = Config::default();
        let text = toml::to_string(&cfg).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back, cfg);
        cfg.validate().unwrap();
    }

    #[test]
    fn missing_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = ConfigLoader::new()
            .file(dir.path().join("nope.toml"))
            .load()
            .unwrap();
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn partial_file_keeps_other_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[general]\ndefault_mode = \"auto-edit\"\n[provider.ollama]\nmodel = \"llama3\"\n",
        )
        .unwrap();
        let cfg = ConfigLoader::new().file(&path).load().unwrap();
        assert_eq!(cfg.general.default_mode, Mode::AutoEdit);
        assert_eq!(cfg.provider.ollama.model, "llama3");
        assert_eq!(cfg.provider.ollama.num_ctx, 32768);
        assert_eq!(cfg.retry.max_retries, 7);
    }

    #[test]
    fn env_overrides_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[agents]\nmax_subagents = 2\n").unwrap();
        let env = [
            ("FINALITY_AGENTS__MAX_SUBAGENTS", "6"),
            ("FINALITY_PROVIDER__OLLAMA__MODEL", "qwen3:8b"),
            ("FINALITY_DECISION__ENABLED", "true"),
            ("FINALITY_HOME", "/ignored"),
            ("PATH", "/ignored"),
        ]
        .map(|(k, v)| (k.to_owned(), v.to_owned()));
        let cfg = ConfigLoader::new()
            .file(&path)
            .env_vars(env)
            .load()
            .unwrap();
        assert_eq!(cfg.agents.max_subagents, 6);
        assert_eq!(cfg.provider.ollama.model, "qwen3:8b");
        assert!(cfg.decision.enabled);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[agents]\nmax_subagent = 2\n").unwrap();
        let err = ConfigLoader::new().file(&path).load().unwrap_err();
        assert!(err.to_string().contains("max_subagent"), "{err}");
    }

    #[test]
    fn validation_catches_bad_values() {
        let env = [(
            "FINALITY_MEMORY__COMPACTION_THRESHOLD".to_owned(),
            "1.5".to_owned(),
        )];
        let err = ConfigLoader::new().env_vars(env).load().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::Invalid {
                key: "memory.compaction_threshold",
                ..
            }
        ));
    }

    #[test]
    fn searxng_requires_url() {
        let env = [(
            "FINALITY_WEB__SEARCH_BACKEND".to_owned(),
            "searxng".to_owned(),
        )];
        let err = ConfigLoader::new().env_vars(env).load().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::Invalid {
                key: "web.searxng_url",
                ..
            }
        ));
    }
}
