use std::path::PathBuf;

use clap::Parser;
use finality_core::config::Config;
use finality_core::{Mode, ProviderKind, SessionId, ThinkLevel};

/// Multi-agent coding and research harness.
#[derive(Debug, Parser)]
#[command(name = "finality", version, about)]
pub struct Cli {
    /// First prompt; starts a session immediately.
    #[arg(trailing_var_arg = true)]
    pub prompt: Vec<String>,

    /// plan | cowork | auto-edit | auto
    #[arg(long, value_parser = parse::<Mode>)]
    pub mode: Option<Mode>,

    /// ollama | openai | anthropic | llamacpp
    #[arg(long, value_parser = parse::<ProviderKind>)]
    pub provider: Option<ProviderKind>,

    /// Model name for the selected provider.
    #[arg(long)]
    pub model: Option<String>,

    /// off | low | medium | high
    #[arg(long, value_parser = parse::<ThinkLevel>)]
    pub think: Option<ThinkLevel>,

    /// Maximum concurrent worker/research sub-agents.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub subagents: Option<u32>,

    /// Resume a past session by id (NNNNNN_YYYYMMDD-HHMMSS).
    #[arg(long, value_parser = parse::<SessionId>)]
    pub resume: Option<SessionId>,

    /// Disable MCP servers.
    #[arg(long)]
    pub no_mcp: bool,

    /// Enable the decision model.
    #[arg(long)]
    pub decide: bool,

    /// Turn direct mode on (fewer refusals and disclaimers); it stays on until `/direct off`.
    #[arg(long)]
    pub direct: bool,

    /// Path to config.toml (default: <data dir>/config.toml).
    #[arg(long)]
    pub config: Option<PathBuf>,
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    s.parse().map_err(|e: T::Err| e.to_string())
}

impl Cli {
    pub fn initial_prompt(&self) -> Option<String> {
        let p = self.prompt.join(" ");
        (!p.trim().is_empty()).then_some(p)
    }

    /// Applies the CLI layer, the highest-precedence config source.
    pub fn apply_to(&self, cfg: &mut Config) {
        if let Some(mode) = self.mode {
            cfg.general.default_mode = mode;
        }
        if let Some(provider) = self.provider {
            cfg.provider.default = provider;
        }
        if let Some(model) = &self.model {
            cfg.provider.set_model(cfg.provider.default, model.clone());
        }
        if let Some(think) = self.think {
            cfg.thinking.level = think;
        }
        if let Some(n) = self.subagents {
            cfg.agents.max_subagents = n;
        }
        if self.no_mcp {
            cfg.mcp.enabled = false;
        }
        if self.decide {
            cfg.decision.enabled = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn overrides_apply_to_selected_provider() {
        let cli = Cli::parse_from([
            "finality",
            "--provider",
            "anthropic",
            "--model",
            "m1",
            "--mode",
            "auto-edit",
            "--no-mcp",
            "--subagents",
            "6",
            "fix",
            "the",
            "bug",
        ]);
        let mut cfg = Config::default();
        cli.apply_to(&mut cfg);
        assert_eq!(cfg.provider.default, ProviderKind::Anthropic);
        assert_eq!(cfg.provider.anthropic.model, "m1");
        assert_eq!(
            cfg.provider.ollama.model,
            Config::default().provider.ollama.model
        );
        assert_eq!(cfg.general.default_mode, Mode::AutoEdit);
        assert_eq!(cfg.agents.max_subagents, 6);
        assert!(!cfg.mcp.enabled);
        assert_eq!(cli.initial_prompt().as_deref(), Some("fix the bug"));
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(Cli::try_parse_from(["finality", "--mode", "yolo"]).is_err());
        assert!(Cli::try_parse_from(["finality", "--subagents", "0"]).is_err());
        assert!(Cli::try_parse_from(["finality", "--resume", "42"]).is_err());
    }
}
