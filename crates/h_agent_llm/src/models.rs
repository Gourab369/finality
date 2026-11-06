use h_agent_core::config::Config;
use h_agent_core::{ProviderKind, Role};

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
