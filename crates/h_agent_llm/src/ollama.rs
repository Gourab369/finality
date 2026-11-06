//! Ollama provider: `/api/chat` (NDJSON streaming), `/api/show`, `/api/tags`.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use h_agent_core::ThinkLevel;
use h_agent_core::config::{LlmConfig, OllamaConfig};
use reqwest::Method;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{check_status, read_json};
use crate::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, ModelInfo, RetryPolicy,
    StopReason, ToolCall, Usage, cancellable,
};

pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    num_ctx: u32,
    keep_alive: String,
    request_timeout: Duration,
    retry: RetryPolicy,
    info_cache: Mutex<HashMap<String, ModelInfo>>,
}

impl OllamaProvider {
    pub fn new(cfg: &OllamaConfig, llm: &LlmConfig) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| LlmError::Other(format!("http client: {e}")))?;
        Ok(Self {
            client,
            base_url: cfg.base_url.trim_end_matches('/').to_owned(),
            num_ctx: cfg.num_ctx,
            keep_alive: cfg.keep_alive.clone(),
            request_timeout: Duration::from_secs(llm.request_timeout_secs),
            retry: RetryPolicy::default(),
            info_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        cancel: &CancellationToken,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}{path}", self.base_url);
        self.retry
            .run(cancel, || {
                let (this, url, method) = (self, &url, method.clone());
                async move {
                    let mut builder = this.client.request(method, url);
                    if let Some(body) = body {
                        builder = builder.json(body);
                    }
                    let resp = tokio::time::timeout(this.request_timeout, builder.send())
                        .await
                        .map_err(|_| LlmError::Timeout)?
                        .map_err(|e| LlmError::from_reqwest(url, e))?;
                    check_status(resp).await
                }
            })
            .await
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        #[derive(Deserialize)]
        struct Tags {
            #[serde(default)]
            models: Vec<Tag>,
        }
        #[derive(Deserialize)]
        struct Tag {
            name: String,
        }
        let resp = self
            .request(Method::GET, "/api/tags", None, &CancellationToken::new())
            .await?;
        let tags: Tags = read_json(resp, self.request_timeout).await?;
        Ok(tags.models.into_iter().map(|t| t.name).collect())
    }

    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        if let Some(info) = self
            .info_cache
            .lock()
            .ok()
            .and_then(|c| c.get(model).cloned())
        {
            return Ok(info);
        }
        let body = json!({ "model": model });
        let resp = self
            .request(
                Method::POST,
                "/api/show",
                Some(&body),
                &CancellationToken::new(),
            )
            .await?;
        let show: ShowResponse = read_json(resp, self.request_timeout).await?;
        let info = show.into_model_info(model);
        if let Ok(mut c) = self.info_cache.lock() {
            c.insert(model.to_owned(), info.clone());
        }
        Ok(info)
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let info = self.model_info(&req.model).await?;
        let body = build_chat_body(&req, &info, self.num_ctx, &self.keep_alive);
        let resp = self
            .request(Method::POST, "/api/chat", Some(&body), &cancel)
            .await?;
        let bytes: ByteStream = Box::pin(resp.bytes_stream().map(|r| {
            r.map(|b| b.to_vec())
                .map_err(|e| LlmError::Decode(e.to_string()))
        }));
        Ok(cancellable(
            ndjson_chunks(bytes, self.request_timeout),
            cancel,
        ))
    }
}

#[derive(Deserialize)]
struct ShowResponse {
    #[serde(default)]
    capabilities: Option<Vec<String>>,
    #[serde(default)]
    model_info: Map<String, Value>,
}

impl ShowResponse {
    fn into_model_info(self, model: &str) -> ModelInfo {
        let context_length = self
            .model_info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, v)| v.as_u64())
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX));
        let (tools, thinking, vision) = match &self.capabilities {
            Some(caps) => {
                let has = |c: &str| caps.iter().any(|x| x == c);
                (has("tools"), has("thinking"), has("vision"))
            }
            // Older servers omit capabilities; let the server reject tools if unsupported.
            None => (true, false, false),
        };
        ModelInfo {
            name: model.to_owned(),
            context_length,
            supports_tools: tools,
            supports_thinking: thinking,
            supports_vision: vision,
        }
    }
}

/// Model families that accept `"low" | "medium" | "high"` instead of a boolean.
fn supports_graded_thinking(model: &str) -> bool {
    model.starts_with("gpt-oss")
}

fn think_value(level: ThinkLevel, info: &ModelInfo) -> Option<Value> {
    if !info.supports_thinking {
        return None;
    }
    Some(match level {
        ThinkLevel::Off => Value::Bool(false),
        level if supports_graded_thinking(&info.name) => Value::String(level.as_str().to_owned()),
        _ => Value::Bool(true),
    })
}

fn message_json(m: &Message) -> Value {
    match m {
        Message::System { content } => json!({ "role": "system", "content": content }),
        Message::User { content } => json!({ "role": "user", "content": content }),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            // Past thinking is not sent back; it would only fill the context.
            let mut v = json!({ "role": "assistant", "content": content });
            if !tool_calls.is_empty() {
                v["tool_calls"] = tool_calls
                    .iter()
                    .map(|c| json!({ "function": { "name": c.name, "arguments": c.arguments } }))
                    .collect();
            }
            v
        }
        Message::Tool { name, content, .. } => {
            json!({ "role": "tool", "content": content, "tool_name": name })
        }
    }
}

pub(crate) fn build_chat_body(
    req: &ChatRequest,
    info: &ModelInfo,
    default_ctx: u32,
    keep_alive: &str,
) -> Value {
    let mut num_ctx = req.num_ctx.unwrap_or(default_ctx);
    if let Some(max) = info.context_length {
        num_ctx = num_ctx.min(max);
    }
    let mut options = json!({ "num_ctx": num_ctx });
    if let Some(t) = req.temperature {
        options["temperature"] = json!(t);
    }
    let mut body = json!({
        "model": req.model,
        "messages": req.messages.iter().map(message_json).collect::<Vec<_>>(),
        "stream": true,
        "options": options,
        "keep_alive": keep_alive,
    });
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| {
                json!({ "type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters
                }})
            })
            .collect();
    }
    if let Some(think) = think_value(req.think, info) {
        body["think"] = think;
    }
    if let Some(format) = &req.response_format {
        body["format"] = format.clone();
    }
    body
}

#[derive(Deserialize)]
struct ChatLine {
    #[serde(default)]
    message: Option<LineMessage>,
    #[serde(default)]
    done: bool,
    done_reason: Option<String>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct LineMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<LineToolCall>,
}

#[derive(Deserialize)]
struct LineToolCall {
    id: Option<String>,
    function: LineFunction,
}

#[derive(Deserialize)]
struct LineFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, LlmError>> + Send>>;

struct NdjsonState {
    bytes: ByteStream,
    idle: Duration,
    buf: Vec<u8>,
    pending: VecDeque<Result<ChatChunk, LlmError>>,
    finished: bool,
    call_seq: u32,
    saw_tool_call: bool,
}

impl NdjsonState {
    fn drain_lines(&mut self) {
        while !self.finished {
            let Some(pos) = self.buf.iter().position(|&b| b == b'\n') else {
                return;
            };
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.handle_line(&line);
        }
    }

    fn handle_line(&mut self, raw: &[u8]) {
        let Ok(text) = std::str::from_utf8(raw) else {
            self.fail(LlmError::Decode("non-UTF-8 line".into()));
            return;
        };
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let line: ChatLine = match serde_json::from_str(text) {
            Ok(l) => l,
            Err(e) => return self.fail(LlmError::Decode(format!("{e}: {text}"))),
        };
        if let Some(error) = line.error {
            return self.fail(LlmError::Other(error));
        }
        if let Some(msg) = line.message {
            if let Some(t) = msg.thinking.filter(|t| !t.is_empty()) {
                self.pending.push_back(Ok(ChatChunk::Thinking(t)));
            }
            if !msg.content.is_empty() {
                self.pending.push_back(Ok(ChatChunk::Text(msg.content)));
            }
            for call in msg.tool_calls {
                self.call_seq += 1;
                self.saw_tool_call = true;
                let arguments = match call.function.arguments {
                    Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
                    Value::Null => json!({}),
                    v => v,
                };
                self.pending.push_back(Ok(ChatChunk::ToolCall(ToolCall {
                    id: call.id.unwrap_or_else(|| format!("call_{}", self.call_seq)),
                    name: call.function.name,
                    arguments,
                })));
            }
        }
        if line.done {
            let stop_reason = match line.done_reason.as_deref() {
                None | Some("stop") if self.saw_tool_call => StopReason::ToolUse,
                None | Some("stop") => StopReason::Stop,
                Some("length") => StopReason::Length,
                Some(other) => StopReason::Other(other.to_owned()),
            };
            self.pending.push_back(Ok(ChatChunk::Done {
                usage: Usage {
                    input_tokens: line.prompt_eval_count.unwrap_or(0),
                    output_tokens: line.eval_count.unwrap_or(0),
                },
                stop_reason,
            }));
            self.finished = true;
        }
    }

    fn fail(&mut self, e: LlmError) {
        self.pending.push_back(Err(e));
        self.finished = true;
    }
}

fn ndjson_chunks(bytes: ByteStream, idle: Duration) -> ChatStream {
    let state = NdjsonState {
        bytes,
        idle,
        buf: Vec::new(),
        pending: VecDeque::new(),
        finished: false,
        call_seq: 0,
        saw_tool_call: false,
    };
    Box::pin(futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(item) = st.pending.pop_front() {
                return Some((item, st));
            }
            if st.finished {
                return None;
            }
            match tokio::time::timeout(st.idle, st.bytes.next()).await {
                Err(_) => st.fail(LlmError::Timeout),
                Ok(Some(Ok(b))) => {
                    st.buf.extend_from_slice(&b);
                    if st.buf.len() > crate::sse::MAX_EVENT_BYTES {
                        st.fail(LlmError::Decode("stream line larger than 8 MiB".into()));
                    } else {
                        st.drain_lines();
                    }
                }
                Ok(Some(Err(e))) => st.fail(e),
                Ok(None) => {
                    let rest = std::mem::take(&mut st.buf);
                    st.handle_line(&rest);
                    if !st.finished {
                        st.fail(LlmError::Decode("stream ended before completion".into()));
                    }
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolSchema, collect};

    fn info(name: &str, thinking: bool, ctx: Option<u32>) -> ModelInfo {
        ModelInfo {
            name: name.into(),
            context_length: ctx,
            supports_tools: true,
            supports_thinking: thinking,
            supports_vision: false,
        }
    }

    fn byte_stream(parts: Vec<&'static str>) -> ByteStream {
        Box::pin(futures::stream::iter(
            parts.into_iter().map(|p| Ok(p.as_bytes().to_vec())),
        ))
    }

    #[test]
    fn body_clamps_num_ctx_and_maps_messages() {
        let mut req = ChatRequest::new(
            "qwen3:14b",
            vec![
                Message::system("sys"),
                Message::user("hi"),
                Message::Assistant {
                    content: String::new(),
                    thinking: None,
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "read_file".into(),
                        arguments: json!({"path": "a"}),
                    }],
                },
                Message::tool("c1", "read_file", "contents"),
            ],
        );
        req.tools = vec![ToolSchema {
            name: "read_file".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
        }];
        let body = build_chat_body(&req, &info("qwen3:14b", true, Some(8192)), 32768, "10m");
        assert_eq!(body["options"]["num_ctx"], 8192);
        assert_eq!(body["think"], true);
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn think_mapping() {
        let plain = info("qwen3", true, None);
        let graded = info("gpt-oss:20b", true, None);
        let none = info("llama3", false, None);
        assert_eq!(think_value(ThinkLevel::High, &plain), Some(json!(true)));
        assert_eq!(think_value(ThinkLevel::Off, &plain), Some(json!(false)));
        assert_eq!(think_value(ThinkLevel::High, &graded), Some(json!("high")));
        assert_eq!(think_value(ThinkLevel::Off, &graded), Some(json!(false)));
        assert_eq!(think_value(ThinkLevel::High, &none), None);
    }

    #[test]
    fn show_response_extracts_context_and_caps() {
        let show: ShowResponse = serde_json::from_value(json!({
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": { "general.architecture": "qwen3", "qwen3.context_length": 40960 }
        }))
        .unwrap();
        let i = show.into_model_info("qwen3:14b");
        assert_eq!(i.context_length, Some(40960));
        assert!(i.supports_tools && i.supports_thinking && !i.supports_vision);
    }

    #[tokio::test]
    async fn parses_lines_split_across_reads() {
        let stream = ndjson_chunks(
            byte_stream(vec![
                "{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"hmm\"},\"done\":false}\n{\"message\":{\"con",
                "tent\":\"Hel\"},\"done\":false}\n{\"message\":{\"content\":\"lo\"},\"done\":false}\n",
                "{\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":26,\"eval_count\":12}",
            ]),
            Duration::from_secs(5),
        );
        let r = collect(stream).await.unwrap();
        assert_eq!(r.thinking, "hmm");
        assert_eq!(r.text, "Hello");
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 26,
                output_tokens: 12
            }
        );
        assert_eq!(r.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn string_arguments_are_parsed_and_ids_generated() {
        let stream = ndjson_chunks(
            byte_stream(vec![
                "{\"message\":{\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":1}\"}}]},\"done\":false}\n",
                "{\"done\":true,\"done_reason\":\"stop\"}\n",
            ]),
            Duration::from_secs(5),
        );
        let r = collect(stream).await.unwrap();
        assert_eq!(r.tool_calls[0].id, "call_1");
        assert_eq!(r.tool_calls[0].arguments, json!({"x": 1}));
        assert_eq!(r.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn error_line_and_truncated_stream_fail() {
        let err = collect(ndjson_chunks(
            byte_stream(vec!["{\"error\":\"boom\"}\n"]),
            Duration::from_secs(5),
        ))
        .await
        .unwrap_err();
        assert!(matches!(err, LlmError::Other(m) if m == "boom"));
        let err = collect(ndjson_chunks(
            byte_stream(vec!["{\"message\":{\"content\":\"x\"},\"done\":false}\n"]),
            Duration::from_secs(5),
        ))
        .await
        .unwrap_err();
        assert!(matches!(err, LlmError::Decode(_)));
    }

    #[tokio::test]
    async fn idle_stream_times_out() {
        let pending: ByteStream = Box::pin(futures::stream::pending());
        let err = collect(ndjson_chunks(pending, Duration::from_millis(20)))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Timeout));
    }
}
