//! Routes requests to Ollama, OpenAI-compatible, Anthropic or llama.cpp providers, building them on first use.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use finality_core::config::Config;
use finality_core::{ProviderKind, Role};
use tokio_util::sync::CancellationToken;

use crate::keys::{ApiKeys, missing_key_message};
use crate::{
    AnthropicProvider, ChatRequest, ChatStream, LlamaCppProvider, LlmError, LlmProvider, ModelInfo,
    ModelRef, OllamaProvider, OpenAiProvider, resolve_model,
};

pub struct ProviderRouter {
    config: Config,
    keys: Arc<dyn ApiKeys>,
    providers: RwLock<HashMap<ProviderKind, Arc<dyn LlmProvider>>>,
    default: RwLock<ProviderKind>,
}

impl ProviderRouter {
    pub fn new(config: &Config, keys: Arc<dyn ApiKeys>) -> Self {
        Self {
            config: config.clone(),
            keys,
            providers: RwLock::new(HashMap::new()),
            default: RwLock::new(config.provider.default),
        }
    }

    /// Uses `provider` for `kind` instead of building one from the config.
    pub fn with_provider(self, kind: ProviderKind, provider: Arc<dyn LlmProvider>) -> Self {
        if let Ok(mut p) = self.providers.write() {
            p.insert(kind, provider);
        }
        self
    }

    pub fn default_kind(&self) -> ProviderKind {
        self.default
            .read()
            .map_or(self.config.provider.default, |d| *d)
    }

    pub fn set_default(&self, kind: ProviderKind) {
        if let Ok(mut d) = self.default.write() {
            *d = kind;
        }
    }

    /// The provider for `kind`, built on first use. Fails when a hosted provider has no API key.
    pub fn get(&self, kind: ProviderKind) -> Result<Arc<dyn LlmProvider>, LlmError> {
        if let Some(p) = self
            .providers
            .read()
            .ok()
            .and_then(|p| p.get(&kind).cloned())
        {
            return Ok(p);
        }
        let cfg = &self.config.provider;
        let key = |env: &str| self.keys.get(kind, env);
        let built: Arc<dyn LlmProvider> = match kind {
            ProviderKind::Ollama => Arc::new(OllamaProvider::new(&cfg.ollama, &self.config.llm)?),
            ProviderKind::OpenAi => {
                let env = &cfg.openai.api_key_env;
                let api_key = key(env);
                if api_key.is_none() && !env.is_empty() {
                    return Err(LlmError::Other(missing_key_message(kind, env)));
                }
                Arc::new(OpenAiProvider::new(&cfg.openai, &self.config.llm, api_key)?)
            }
            ProviderKind::Anthropic => {
                let env = &cfg.anthropic.api_key_env;
                let api_key =
                    key(env).ok_or_else(|| LlmError::Other(missing_key_message(kind, env)))?;
                Arc::new(AnthropicProvider::new(
                    &cfg.anthropic,
                    &self.config.llm,
                    api_key,
                )?)
            }
            ProviderKind::LlamaCpp => {
                Arc::new(LlamaCppProvider::new(&cfg.llamacpp, &self.config.llm)?)
            }
        };
        let mut providers = self
            .providers
            .write()
            .map_err(|_| LlmError::Other("provider table poisoned".into()))?;
        Ok(providers.entry(kind).or_insert(built).clone())
    }

    /// Stops servers that providers started (llama.cpp).
    pub async fn shutdown(&self) {
        let built: Vec<Arc<dyn LlmProvider>> = self
            .providers
            .read()
            .map(|p| p.values().cloned().collect())
            .unwrap_or_default();
        for p in built {
            p.shutdown().await;
        }
    }
}

#[async_trait]
impl LlmProvider for ProviderRouter {
    fn id(&self) -> &'static str {
        "router"
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.get(self.default_kind())?.list_models().await
    }

    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        self.get(self.default_kind())?.model_info(model).await
    }

    async fn model_info_for(
        &self,
        kind: Option<ProviderKind>,
        model: &str,
    ) -> Result<ModelInfo, LlmError> {
        self.get(kind.unwrap_or_else(|| self.default_kind()))?
            .model_info(model)
            .await
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let kind = req.provider.unwrap_or_else(|| self.default_kind());
        self.get(kind)?.chat(req, cancel).await
    }
}

/// Per-role models that follow `/provider` and `/model` unless `models.<role>` pins them.
pub struct ModelSelector {
    config: RwLock<Config>,
}

impl ModelSelector {
    pub fn new(config: &Config) -> Arc<Self> {
        Arc::new(Self {
            config: RwLock::new(config.clone()),
        })
    }

    pub fn resolve(&self, role: Role) -> ModelRef {
        match self.config.read() {
            Ok(cfg) => resolve_model(&cfg, role),
            Err(e) => resolve_model(&e.into_inner(), role),
        }
    }

    /// The default provider and its model, for roles without `models.<role>`.
    pub fn set_default(&self, provider: ProviderKind, model: &str) {
        if let Ok(mut cfg) = self.config.write() {
            cfg.provider.default = provider;
            cfg.provider.set_model(provider, model.to_owned());
        }
    }

    /// The context window for `provider`: Ollama's `num_ctx`, else the configured `context_window`.
    pub fn window(&self, provider: ProviderKind) -> u32 {
        let read = |cfg: &Config| match provider {
            ProviderKind::Ollama => cfg.provider.ollama.num_ctx,
            ProviderKind::OpenAi => cfg.provider.openai.context_window,
            ProviderKind::Anthropic => cfg.provider.anthropic.context_window,
            ProviderKind::LlamaCpp => cfg.provider.llamacpp.context_window,
        };
        match self.config.read() {
            Ok(cfg) => read(&cfg),
            Err(e) => read(&e.into_inner()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::StaticKeys;

    #[test]
    fn hosted_providers_need_keys() {
        let mut cfg = Config::default();
        cfg.provider.openai.api_key_env = "FINALITY_TEST_NO_SUCH_KEY".into();
        let router = ProviderRouter::new(&cfg, Arc::new(StaticKeys::default()));
        let err = router.get(ProviderKind::OpenAi).err().unwrap().to_string();
        assert!(err.contains("set FINALITY_TEST_NO_SUCH_KEY"), "{err}");
        assert!(router.get(ProviderKind::Anthropic).is_err());
        assert_eq!(router.get(ProviderKind::Ollama).unwrap().id(), "ollama");

        cfg.provider.openai.api_key_env = String::new();
        let local = ProviderRouter::new(&cfg, Arc::new(StaticKeys::default()));
        assert_eq!(
            local.get(ProviderKind::OpenAi).unwrap().id(),
            "openai",
            "local servers need no key"
        );

        let keyed = ProviderRouter::new(
            &Config::default(),
            Arc::new(StaticKeys::default().with(ProviderKind::Anthropic, "k")),
        );
        assert_eq!(
            keyed.get(ProviderKind::Anthropic).unwrap().id(),
            "anthropic"
        );
    }

    #[test]
    fn selector_follows_the_default_unless_pinned() {
        let mut cfg = Config::default();
        cfg.models.memory = "ollama:qwen3:8b".into();
        let s = ModelSelector::new(&cfg);
        s.set_default(ProviderKind::Anthropic, "claude-sonnet-4-5");
        let worker = s.resolve(Role::Worker);
        assert_eq!(
            (worker.provider, worker.model.as_str()),
            (ProviderKind::Anthropic, "claude-sonnet-4-5")
        );
        assert_eq!(s.resolve(Role::Memory).model, "qwen3:8b");
        assert_eq!(s.window(ProviderKind::Anthropic), 200_000);
        assert_eq!(s.window(ProviderKind::Ollama), cfg.provider.ollama.num_ctx);
    }
}
