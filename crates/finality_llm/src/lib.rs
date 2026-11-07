//! LLM provider abstraction and implementations (DESIGN §5.3).

pub mod anthropic;
pub mod decision;
pub mod error;
mod http;
pub mod keys;
pub mod llamacpp;
pub mod mock;
pub mod models;
pub mod ollama;
pub mod openai;
pub mod provider;
pub mod retry;
pub mod router;
mod sse;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use decision::{Answer, Decision, DecisionClient, Question};
pub use error::LlmError;
pub use keys::{ApiKeys, EnvOrKeyring, StaticKeys};
pub use llamacpp::LlamaCppProvider;
pub use mock::MockProvider;
pub use models::{ModelRef, resolve_model, resolve_triage};
pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;
pub use provider::{ChatStream, CollectedResponse, LlmProvider, cancellable, collect};
pub use retry::RetryPolicy;
pub use router::{ModelSelector, ProviderRouter};
pub use types::{
    ChatChunk, ChatRequest, ImageData, Message, ModelInfo, StopReason, ToolCall, ToolSchema, Usage,
};
