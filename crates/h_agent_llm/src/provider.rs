use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use h_agent_core::ProviderKind;
use tokio_util::sync::CancellationToken;

use crate::{ChatChunk, ChatRequest, LlmError, ModelInfo, StopReason, ToolCall, Usage};

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &'static str;
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError>;
    /// `model_info` from the provider that serves `kind`; only routers look at `kind`.
    async fn model_info_for(
        &self,
        kind: Option<ProviderKind>,
        model: &str,
    ) -> Result<ModelInfo, LlmError> {
        let _ = kind;
        self.model_info(model).await
    }
    /// The stream yields `Err(LlmError::Cancelled)` and ends promptly once `cancel` fires.
    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError>;
    /// Stops anything the provider started (e.g. a local inference server).
    async fn shutdown(&self) {}
}

/// Ends `inner` with `Err(Cancelled)` as soon as `cancel` fires, even while it is waiting for data.
pub fn cancellable(inner: ChatStream, cancel: CancellationToken) -> ChatStream {
    Box::pin(futures::stream::unfold(
        Some((inner, cancel)),
        |state| async move {
            let (mut inner, cancel) = state?;
            tokio::select! {
                biased;
                () = cancel.cancelled() => Some((Err(LlmError::Cancelled), None)),
                item = inner.next() => item.map(|item| (item, Some((inner, cancel)))),
            }
        },
    ))
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CollectedResponse {
    pub text: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub stop_reason: StopReason,
}

/// Drains a stream into a single response. Errors if the stream ends without `Done`.
pub async fn collect(mut stream: ChatStream) -> Result<CollectedResponse, LlmError> {
    let mut out = CollectedResponse::default();
    while let Some(chunk) = stream.next().await {
        match chunk? {
            ChatChunk::Text(t) => out.text.push_str(&t),
            ChatChunk::Thinking(t) => out.thinking.push_str(&t),
            ChatChunk::ToolCall(c) => out.tool_calls.push(c),
            ChatChunk::Done { usage, stop_reason } => {
                out.usage = usage;
                out.stop_reason = stop_reason;
                return Ok(out);
            }
        }
    }
    Err(LlmError::Decode("stream ended without completion".into()))
}
