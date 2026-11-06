//! Scripted provider for deterministic tests. Each `chat` call consumes the next expectation in order.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, ModelInfo, StopReason, ToolCall,
    Usage, cancellable,
};

type Matcher = Box<dyn Fn(&ChatRequest) -> bool + Send + Sync>;

enum Reply {
    Chunks(Vec<ChatChunk>),
    Error(LlmError),
}

struct Expectation {
    matcher: Option<Matcher>,
    reply: Reply,
}

pub struct MockProvider {
    script: Mutex<VecDeque<Expectation>>,
    requests: Mutex<Vec<ChatRequest>>,
    info: ModelInfo,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProvider {
    pub fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            info: ModelInfo {
                name: "mock".into(),
                context_length: Some(32_768),
                supports_tools: true,
                supports_thinking: true,
                supports_vision: false,
            },
        }
    }

    pub fn with_model_info(mut self, info: ModelInfo) -> Self {
        self.info = info;
        self
    }

    /// Replies with raw chunks. A `Done` chunk is appended if missing.
    pub fn then_chunks(self, chunks: Vec<ChatChunk>) -> Self {
        self.push(None, Reply::Chunks(chunks))
    }

    pub fn then_text(self, text: &str) -> Self {
        self.then_chunks(vec![ChatChunk::Text(text.to_owned())])
    }

    pub fn then_tool_call(self, name: &str, arguments: Value) -> Self {
        let n = self.script_len();
        self.then_chunks(vec![ChatChunk::ToolCall(ToolCall {
            id: format!("mock_call_{n}"),
            name: name.to_owned(),
            arguments,
        })])
    }

    pub fn then_error(self, error: LlmError) -> Self {
        self.push(None, Reply::Error(error))
    }

    /// Like `then_chunks`, but the request must satisfy `matcher` or the call fails.
    pub fn then_when(
        self,
        matcher: impl Fn(&ChatRequest) -> bool + Send + Sync + 'static,
        chunks: Vec<ChatChunk>,
    ) -> Self {
        self.push(Some(Box::new(matcher)), Reply::Chunks(chunks))
    }

    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }

    pub fn remaining(&self) -> usize {
        self.script_len()
    }

    fn script_len(&self) -> usize {
        self.script.lock().map(|s| s.len()).unwrap_or(0)
    }

    fn push(self, matcher: Option<Matcher>, reply: Reply) -> Self {
        if let Ok(mut s) = self.script.lock() {
            s.push_back(Expectation { matcher, reply });
        }
        self
    }
}

fn estimate_tokens(text: &str) -> u32 {
    u32::try_from(text.chars().count().div_ceil(4)).unwrap_or(u32::MAX)
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn id(&self) -> &'static str {
        "mock"
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![self.info.name.clone()])
    }

    async fn model_info(&self, _model: &str) -> Result<ModelInfo, LlmError> {
        Ok(self.info.clone())
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let index = self.requests.lock().map(|r| r.len()).unwrap_or(0);
        if let Ok(mut r) = self.requests.lock() {
            r.push(req.clone());
        }
        let next = self
            .script
            .lock()
            .map_err(|_| LlmError::Other("mock: script poisoned".into()))?
            .pop_front()
            .ok_or_else(|| LlmError::Other(format!("mock: unexpected request #{index}")))?;
        if let Some(matcher) = &next.matcher
            && !matcher(&req)
        {
            return Err(LlmError::Other(format!(
                "mock: request #{index} did not match expectation"
            )));
        }
        let mut chunks = match next.reply {
            Reply::Error(e) => return Err(e),
            Reply::Chunks(c) => c,
        };
        if !chunks.iter().any(|c| matches!(c, ChatChunk::Done { .. })) {
            let output: u32 = chunks
                .iter()
                .map(|c| match c {
                    ChatChunk::Text(t) | ChatChunk::Thinking(t) => estimate_tokens(t),
                    ChatChunk::ToolCall(call) => estimate_tokens(&call.arguments.to_string()),
                    ChatChunk::Done { .. } => 0,
                })
                .sum();
            let input = req
                .messages
                .iter()
                .map(|m| estimate_tokens(m.content()))
                .sum();
            let stop_reason = if chunks.iter().any(|c| matches!(c, ChatChunk::ToolCall(_))) {
                StopReason::ToolUse
            } else {
                StopReason::Stop
            };
            chunks.push(ChatChunk::Done {
                usage: Usage {
                    input_tokens: input,
                    output_tokens: output,
                },
                stop_reason,
            });
        }
        let stream = futures::stream::iter(chunks.into_iter().map(Ok));
        Ok(cancellable(Box::pin(stream), cancel))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Message, collect};

    #[tokio::test]
    async fn replays_script_in_order_and_records_requests() {
        let mock = MockProvider::new()
            .then_tool_call("read_file", json!({"path": "Cargo.toml"}))
            .then_text("done");
        let c = CancellationToken::new();

        let r1 = collect(
            mock.chat(
                ChatRequest::new("m", vec![Message::user("read it")]),
                c.clone(),
            )
            .await
            .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(r1.tool_calls[0].name, "read_file");
        assert_eq!(r1.stop_reason, StopReason::ToolUse);

        let r2 = collect(mock.chat(ChatRequest::new("m", vec![]), c).await.unwrap())
            .await
            .unwrap();
        assert_eq!(r2.text, "done");
        assert_eq!(mock.requests().len(), 2);
        assert_eq!(mock.requests()[0].messages[0].content(), "read it");
        assert_eq!(mock.remaining(), 0);
    }

    #[tokio::test]
    async fn unexpected_and_mismatched_requests_fail() {
        let mock = MockProvider::new().then_when(|r| r.model == "want", vec![]);
        let c = CancellationToken::new();
        assert!(
            mock.chat(ChatRequest::new("other", vec![]), c.clone())
                .await
                .is_err()
        );
        assert!(
            mock.chat(ChatRequest::new("want", vec![]), c)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn scripted_error_is_returned() {
        let mock = MockProvider::new().then_error(LlmError::Timeout);
        let r = mock
            .chat(ChatRequest::default(), CancellationToken::new())
            .await;
        assert!(matches!(r, Err(LlmError::Timeout)));
    }
}
