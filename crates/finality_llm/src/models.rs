use finality_core::config::Config;
use finality_core::{ProviderKind, Role};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub provider: ProviderKind,
    pub model: String,
}

/// `models.<role>` (`provider:model`, or a bare model on the default provider), else the default provider's model.
pub fn resolve_model(cfg: &Config, role: Role) -> ModelRef {
    let default = cfg.provider.default;
    let configured = match role {
        Role::Orchestrator => &cfg.models.orchestrator,
        Role::Worker => &cfg.models.worker,
        Role::Research => &cfg.models.research,
        Role::Memory => &cfg.models.memory,
        Role::Scorer => &cfg.models.scorer,
        Role::Learner => &cfg.models.learner,
    }
    .trim();
    if configured.is_empty() {
        return ModelRef {
            provider: default,
            model: cfg.provider.model_for(default).to_owned(),
        };
    }
    parse_ref(cfg, configured)
}

/// The triage model: `models.triage`, else a pinned `models.memory`, else the orchestrator's. A model other than the
/// orchestrator's keeps triage from evicting the orchestrator's prompt cache on local servers. The memory model is
/// skipped on llama.cpp, which loads one model at a time and would switch models every turn.
pub fn resolve_triage(cfg: &Config) -> ModelRef {
    let triage = cfg.models.triage.trim();
    if !triage.is_empty() {
        return parse_ref(cfg, triage);
    }
    let memory = cfg.models.memory.trim();
    if !memory.is_empty() {
        let r = parse_ref(cfg, memory);
        if r.provider != ProviderKind::LlamaCpp {
            return r;
        }
    }
    resolve_model(cfg, Role::Orchestrator)
}

fn parse_ref(cfg: &Config, configured: &str) -> ModelRef {
    let default = cfg.provider.default;
    // Ollama tags contain ':' too ("qwen3:8b"), so only a known provider prefix is split off.
    if let Some((prefix, model)) = configured.split_once(':')
        && let Ok(provider) = prefix.parse::<ProviderKind>()
        && !model.is_empty()
    {
        return ModelRef {
            provider,
            model: model.to_owned(),
        };
    }
    ModelRef {
        provider: default,
        model: configured.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triage_prefers_its_own_then_the_memory_model() {
        let mut cfg = Config::default();
        assert_eq!(
            resolve_triage(&cfg),
            resolve_model(&cfg, Role::Orchestrator)
        );
        cfg.models.memory = "llamacpp:small.gguf".into();
        assert_eq!(
            resolve_triage(&cfg),
            resolve_model(&cfg, Role::Orchestrator),
            "llama.cpp would swap models every turn"
        );
        cfg.models.memory = "qwen3:4b".into();
        assert_eq!(resolve_triage(&cfg).model, "qwen3:4b");
        cfg.models.triage = "anthropic:claude-haiku-4-5".into();
        let t = resolve_triage(&cfg);
        assert_eq!(
            (t.provider, t.model.as_str()),
            (ProviderKind::Anthropic, "claude-haiku-4-5")
        );
    }

    #[test]
    fn falls_back_to_default_provider_model() {
        let cfg = Config::default();
        let r = resolve_model(&cfg, Role::Worker);
        assert_eq!(r.provider, ProviderKind::Ollama);
        assert_eq!(r.model, cfg.provider.ollama.model);
    }

    #[test]
    fn parses_provider_prefixed_ollama_tag() {
        let mut cfg = Config::default();
        cfg.models.memory = "ollama:qwen3:8b".into();
        let r = resolve_model(&cfg, Role::Memory);
        assert_eq!(
            r,
            ModelRef {
                provider: ProviderKind::Ollama,
                model: "qwen3:8b".into()
            }
        );
    }

    #[test]
    fn bare_tag_uses_default_provider() {
        let mut cfg = Config::default();
        cfg.models.scorer = "qwen3:8b".into();
        let r = resolve_model(&cfg, Role::Scorer);
        assert_eq!(
            r,
            ModelRef {
                provider: ProviderKind::Ollama,
                model: "qwen3:8b".into()
            }
        );
    }

    #[test]
    fn other_provider_override() {
        let mut cfg = Config::default();
        cfg.models.research = "anthropic:some-model".into();
        let r = resolve_model(&cfg, Role::Research);
        assert_eq!(r.provider, ProviderKind::Anthropic);
        assert_eq!(r.model, "some-model");
        assert_eq!(
            resolve_model(&cfg, Role::Orchestrator).provider,
            ProviderKind::Ollama
        );
    }

    #[test]
    fn default_change_follows_unpinned_roles() {
        let mut cfg = Config::default();
        cfg.models.memory = "ollama:small".into();
        cfg.provider.default = ProviderKind::OpenAi;
        cfg.provider.openai.model = "gpt-x".into();
        assert_eq!(resolve_model(&cfg, Role::Worker).model, "gpt-x");
        assert_eq!(resolve_model(&cfg, Role::Memory).model, "small");
    }
}
