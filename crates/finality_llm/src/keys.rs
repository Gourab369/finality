//! API keys for hosted providers: the env var named in `api_key_env`, else the OS keyring.

use std::collections::HashMap;

use finality_core::ProviderKind;

pub const KEYRING_SERVICE: &str = "finality";

pub trait ApiKeys: Send + Sync {
    fn get(&self, provider: ProviderKind, env_var: &str) -> Option<String>;
}

/// Keyring user name for a provider's key.
pub fn keyring_user(provider: ProviderKind) -> String {
    format!("api-key:{provider}")
}

pub fn missing_key_message(provider: ProviderKind, env_var: &str) -> String {
    format!(
        "no API key for {provider}: set {env_var}, or store the key in the OS keyring (service `{KEYRING_SERVICE}`, user `{}`)",
        keyring_user(provider)
    )
}

/// The production lookup.
pub struct EnvOrKeyring;

impl ApiKeys for EnvOrKeyring {
    fn get(&self, provider: ProviderKind, env_var: &str) -> Option<String> {
        if !env_var.is_empty()
            && let Ok(v) = std::env::var(env_var)
            && !v.trim().is_empty()
        {
            return Some(v.trim().to_owned());
        }
        keyring::Entry::new(KEYRING_SERVICE, &keyring_user(provider))
            .ok()?
            .get_password()
            .ok()
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
    }
}

/// Fixed keys, for tests.
#[derive(Default)]
pub struct StaticKeys(HashMap<ProviderKind, String>);

impl StaticKeys {
    pub fn with(mut self, provider: ProviderKind, key: impl Into<String>) -> Self {
        self.0.insert(provider, key.into());
        self
    }
}

impl ApiKeys for StaticKeys {
    fn get(&self, provider: ProviderKind, _env_var: &str) -> Option<String> {
        self.0.get(&provider).cloned()
    }
}
