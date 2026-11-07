//! Anthropic Messages API (DESIGN §5.3.4): `POST {base}/v1/messages` over SSE.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_core::ThinkLevel;
use finality_core::config::{AnthropicConfig, LlmConfig};
use reqwest::Method;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{check_status, error_message, f32_json, read_json};
use crate::sse::{SseEvent, SseParser, body_stream, sse_chunks};
use crate::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, ModelInfo, RetryPolicy,
    StopReason, ToolCall, Usage, cancellable,
};
use zeroize::Zeroizing;

pub const API_VERSION: &str = "2023-06-01";
const SIGNATURES_KEPT: usize = 64;

pub fn thinking_budget(level: ThinkLevel) -> Option<u32> {
    match level {
        ThinkLevel::Off => None,
        ThinkLevel::Low => Some(2048),
        ThinkLevel::Medium => Some(8192),
        ThinkLevel::High => Some(24576),
    }
}

/// Claude 3.x models other than 3.7 have no extended thinking.
pub fn supports_thinking(model: &str) -> bool {
    !(model.starts_with("claude-3-") && !model.starts_with("claude-3-7"))
}

/// One streamed thinking block with the signature the API needs to accept it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ThinkingBlock {
    text: String,
    signature: String,
}

/// Responses' thinking blocks, keyed by their joined text (what the conversation stores).
#[derive(Default)]
pub(crate) struct Signatures(Mutex<VecDeque<(String, Vec<ThinkingBlock>)>>);

impl Signatures {
    fn insert(&self, blocks: Vec<ThinkingBlock>) {
        if blocks.is_empty() || blocks.iter().any(|b| b.signature.is_empty()) {
            return;
        }
        let key: String = blocks.iter().map(|b| b.text.as_str()).collect();
        if let Ok(mut s) = self.0.lock() {
            s.retain(|(k, _)| *k != key);
            s.push_back((key, blocks));
            while s.len() > SIGNATURES_KEPT {
                s.pop_front();
            }
        }
    }

    fn get(&self, thinking: &str) -> Option<Vec<ThinkingBlock>> {
        self.0
            .lock()
            .ok()?
            .iter()
            .find(|(k, _)| k == thinking)
            .map(|(_, b)| b.clone())
    }
}

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Zeroizing<String>,
    context_window: u32,
    max_tokens: u32,
    request_timeout: Duration,
    retry: RetryPolicy,
    signatures: Arc<Signatures>,
}

impl AnthropicProvider {
    pub fn new(cfg: &AnthropicConfig, llm: &LlmConfig, api_key: String) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| LlmError::Other(format!("http client: {e}")))?;
        Ok(Self {
            client,
            base_url: cfg.base_url.trim_end_matches('/').to_owned(),
            api_key: Zeroizing::new(api_key),
            context_window: cfg.context_window,
            max_tokens: cfg.max_tokens.max(1),
            request_timeout: Duration::from_secs(llm.request_timeout_secs),
            retry: RetryPolicy::default(),
            signatures: Arc::new(Signatures::default()),
        })
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
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
                let (url, method) = (&url, method.clone());
                async move {
                    let mut builder = self
                        .client
                        .request(method, url)
                        .header("x-api-key", self.api_key.as_str())
                        .header("anthropic-version", API_VERSION);
                    if let Some(body) = body {
                        builder = builder.json(body);
                    }
                    let resp = tokio::time::timeout(self.request_timeout, builder.send())
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
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        #[derive(Deserialize)]
        struct Models {
            #[serde(default)]
            data: Vec<Model>,
        }
        #[derive(Deserialize)]
        struct Model {
            id: String,
        }
        let resp = self
            .request(Method::GET, "/v1/models", None, &CancellationToken::new())
            .await?;
        let models: Models = read_json(resp, self.request_timeout).await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    /// Context length comes from `provider.anthropic.context_window`.
    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        Ok(ModelInfo {
            name: model.to_owned(),
            context_length: Some(self.context_window),
            supports_tools: true,
            supports_thinking: supports_thinking(model),
            supports_vision: true,
        })
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let body = build_body(&req, self.max_tokens, &self.signatures);
        let resp = self
            .request(Method::POST, "/v1/messages", Some(&body), &cancel)
            .await?;
        let parser = AnthropicParser::new(self.signatures.clone());
        Ok(cancellable(
            sse_chunks(body_stream(resp), self.request_timeout, parser),
            cancel,
        ))
    }
}

/// Adds content blocks to the conversation, merging consecutive turns of the same role.
fn push_blocks(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last["role"] == role
        && let Some(content) = last["content"].as_array_mut()
    {
        content.extend(blocks);
        return;
    }
    messages.push(json!({ "role": role, "content": blocks }));
}

pub(crate) fn build_body(req: &ChatRequest, max_tokens: u32, signatures: &Signatures) -> Value {
    let lead = req
        .messages
        .iter()
        .take_while(|m| matches!(m, Message::System { .. }))
        .count();
    let mut system: Vec<String> = req.messages[..lead]
        .iter()
        .map(|m| m.content().to_owned())
        .collect();
    if let Some(schema) = &req.response_format {
        system.push(format!(
            "Reply with only a JSON object that matches this JSON Schema, with no other text:\n{schema}"
        ));
    }
    // Thinking blocks are sent back only within the current turn (after the last user message).
    let turn_start = req
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }))
        .unwrap_or(0);
    let mut budget = thinking_budget(req.think).filter(|_| supports_thinking(&req.model));
    let mut messages: Vec<Value> = Vec::new();
    for (i, m) in req.messages.iter().enumerate().skip(lead) {
        match m {
            Message::System { content } => push_blocks(
                &mut messages,
                "user",
                vec![json!({ "type": "text", "text": format!("System note:\n{content}") })],
            ),
            Message::User { content, images } => push_blocks(
                &mut messages,
                "user",
                images
                    .iter()
                    .map(|img| {
                        json!({ "type": "image", "source": {
                            "type": "base64", "media_type": img.media_type, "data": img.base64
                        }})
                    })
                    .chain([json!({ "type": "text", "text": content })])
                    .collect(),
            ),
            Message::Assistant {
                content,
                thinking,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                let signed = thinking.as_deref().and_then(|t| signatures.get(t));
                if i > turn_start
                    && budget.is_some()
                    && let Some(signed) = &signed
                {
                    blocks.extend(signed.iter().map(|b| {
                        json!({ "type": "thinking", "thinking": b.text, "signature": b.signature })
                    }));
                } else if i > turn_start && !tool_calls.is_empty() {
                    // The API rejects a tool-use turn without its signed thinking while thinking is on.
                    budget = None;
                }
                if !content.is_empty() {
                    blocks.push(json!({ "type": "text", "text": content }));
                }
                blocks.extend(tool_calls.iter().map(|c| {
                    let input = if c.arguments.is_object() {
                        c.arguments.clone()
                    } else {
                        json!({})
                    };
                    json!({ "type": "tool_use", "id": c.id, "name": c.name, "input": input })
                }));
                push_blocks(&mut messages, "assistant", blocks);
            }
            Message::Tool {
                call_id, content, ..
            } => push_blocks(
                &mut messages,
                "user",
                vec![json!({ "type": "tool_result", "tool_use_id": call_id, "content": content })],
            ),
        }
    }
    if messages.first().is_none_or(|m| m["role"] != "user") {
        messages.insert(
            0,
            json!({ "role": "user", "content": [{ "type": "text", "text": "(continue)" }] }),
        );
    }
    let mut body = json!({
        "model": req.model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": true,
    });
    if !system.is_empty() {
        // Prompt caching: tools, the system prompt and the conversation so far are reused by the next request.
        body["system"] = json!([{
            "type": "text",
            "text": system.join("\n\n"),
            "cache_control": { "type": "ephemeral" }
        }]);
    }
    if !req.tools.is_empty() {
        let mut tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.parameters }))
            .collect();
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({ "type": "ephemeral" });
        }
        body["tools"] = Value::Array(tools);
    }
    if let Some(block) = body["messages"]
        .as_array_mut()
        .and_then(|m| m.last_mut())
        .and_then(|m| m["content"].as_array_mut())
        .and_then(|c| c.last_mut())
        .filter(|b| b["type"] != "thinking")
    {
        block["cache_control"] = json!({ "type": "ephemeral" });
    }
    match budget {
        Some(budget) => {
            body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
            body["max_tokens"] = json!(max_tokens.max(budget + 4096));
        }
        None => {
            if let Some(t) = req.temperature {
                body["temperature"] = f32_json(t);
            }
        }
    }
    body
}

enum Block {
    Text,
    Thinking(ThinkingBlock),
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    Other,
}

pub(crate) struct AnthropicParser {
    blocks: Vec<(u64, Block)>,
    thinking: Vec<ThinkingBlock>,
    signatures: Arc<Signatures>,
    usage: Usage,
    stop: Option<String>,
    saw_tool: bool,
}

impl AnthropicParser {
    pub(crate) fn new(signatures: Arc<Signatures>) -> Self {
        Self {
            blocks: Vec::new(),
            thinking: Vec::new(),
            signatures,
            usage: Usage::default(),
            stop: None,
            saw_tool: false,
        }
    }

    fn block(&mut self, index: u64) -> Option<&mut Block> {
        self.blocks
            .iter_mut()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| b)
    }

    fn done(&mut self, out: &mut VecDeque<Result<ChatChunk, LlmError>>) {
        self.signatures.insert(std::mem::take(&mut self.thinking));
        let stop_reason = match self.stop.as_deref() {
            Some("tool_use") => StopReason::ToolUse,
            _ if self.saw_tool => StopReason::ToolUse,
            None | Some("end_turn" | "stop_sequence") => StopReason::Stop,
            Some("max_tokens") => StopReason::Length,
            Some(other) => StopReason::Other(other.to_owned()),
        };
        out.push_back(Ok(ChatChunk::Done {
            usage: self.usage,
            stop_reason,
        }));
    }
}

fn count(v: &Value, key: &str) -> u32 {
    v.get(key)
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
}

impl SseParser for AnthropicParser {
    fn event(&mut self, ev: SseEvent, out: &mut VecDeque<Result<ChatChunk, LlmError>>) -> bool {
        let v: Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(e) => {
                out.push_back(Err(LlmError::Decode(format!("{e}: {}", ev.data))));
                return true;
            }
        };
        let kind = v["type"].as_str().or(ev.event.as_deref()).unwrap_or("");
        let index = v["index"].as_u64().unwrap_or(0);
        match kind {
            "message_start" => {
                let u = &v["message"]["usage"];
                self.usage.input_tokens = count(u, "input_tokens")
                    .saturating_add(count(u, "cache_creation_input_tokens"))
                    .saturating_add(count(u, "cache_read_input_tokens"));
                self.usage.output_tokens = count(u, "output_tokens");
            }
            "content_block_start" => {
                let b = &v["content_block"];
                let block = match b["type"].as_str().unwrap_or("") {
                    "text" => {
                        if let Some(t) = b["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push_back(Ok(ChatChunk::Text(t.to_owned())));
                        }
                        Block::Text
                    }
                    "thinking" => Block::Thinking(ThinkingBlock {
                        text: String::new(),
                        signature: String::new(),
                    }),
                    "tool_use" => Block::ToolUse {
                        id: b["id"].as_str().unwrap_or("").to_owned(),
                        name: b["name"].as_str().unwrap_or("").to_owned(),
                        json: String::new(),
                    },
                    _ => Block::Other,
                };
                self.blocks.push((index, block));
            }
            "content_block_delta" => {
                let d = &v["delta"];
                let text = |k: &str| d[k].as_str().unwrap_or("").to_owned();
                match (d["type"].as_str().unwrap_or(""), self.block(index)) {
                    ("text_delta", _) => {
                        let t = text("text");
                        if !t.is_empty() {
                            out.push_back(Ok(ChatChunk::Text(t)));
                        }
                    }
                    ("thinking_delta", Some(Block::Thinking(b))) => {
                        let t = text("thinking");
                        b.text.push_str(&t);
                        if !t.is_empty() {
                            out.push_back(Ok(ChatChunk::Thinking(t)));
                        }
                    }
                    ("signature_delta", Some(Block::Thinking(b))) => {
                        b.signature.push_str(&text("signature"));
                    }
                    ("input_json_delta", Some(Block::ToolUse { json, .. })) => {
                        json.push_str(&text("partial_json"));
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(pos) = self.blocks.iter().position(|(i, _)| *i == index) else {
                    return false;
                };
                match self.blocks.remove(pos).1 {
                    Block::ToolUse { id, name, json } => {
                        self.saw_tool = true;
                        let arguments = if json.trim().is_empty() {
                            json!({})
                        } else {
                            serde_json::from_str(&json).unwrap_or(Value::String(json))
                        };
                        out.push_back(Ok(ChatChunk::ToolCall(ToolCall {
                            id,
                            name,
                            arguments,
                        })));
                    }
                    Block::Thinking(b) => self.thinking.push(b),
                    Block::Text | Block::Other => {}
                }
            }
            "message_delta" => {
                if let Some(reason) = v["delta"]["stop_reason"].as_str() {
                    self.stop = Some(reason.to_owned());
                }
                if v["usage"].get("output_tokens").is_some() {
                    self.usage.output_tokens = count(&v["usage"], "output_tokens");
                }
            }
            "message_stop" => {
                self.done(out);
                return true;
            }
            "error" => {
                let kind = v["error"]["type"].as_str().unwrap_or("error");
                out.push_back(Err(LlmError::Other(format!(
                    "{kind}: {}",
                    error_message(&ev.data)
                ))));
                return true;
            }
            _ => {}
        }
        false
    }

    fn end(&mut self, out: &mut VecDeque<Result<ChatChunk, LlmError>>) {
        if self.stop.is_some() {
            self.done(out);
        } else {
            out.push_back(Err(LlmError::Decode(
                "stream ended before completion".into(),
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::byte_stream;
    use crate::{ToolSchema, collect};

    fn sse(events: &[(&str, &str)]) -> Vec<String> {
        events
            .iter()
            .map(|(e, d)| format!("event: {e}\ndata: {d}\n\n"))
            .collect()
    }

    fn tool_turn(thinking: Option<&str>) -> Vec<Message> {
        vec![
            Message::system("You are finality."),
            Message::user("read it"),
            Message::system("Triage marked this request as multi-step."),
            Message::Assistant {
                content: "Reading.".into(),
                thinking: thinking.map(str::to_owned),
                tool_calls: vec![ToolCall {
                    id: "toolu_1".into(),
                    name: "read_file".into(),
                    arguments: json!({ "path": "a" }),
                }],
            },
            Message::tool("toolu_1", "read_file", "contents"),
        ]
    }

    #[test]
    fn body_uses_system_blocks_and_tool_results() {
        let mut req = ChatRequest::new("claude-sonnet-4-5", tool_turn(None));
        req.tools = vec![ToolSchema {
            name: "read_file".into(),
            description: "d".into(),
            parameters: json!({ "type": "object" }),
        }];
        req.temperature = Some(0.2);
        let body = build_body(&req, 8192, &Signatures::default());
        assert_eq!(body["system"][0]["text"], "You are finality.");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["max_tokens"], 8192);
        assert_eq!(body["temperature"], 0.2);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "user, assistant, user(tool result)");
        assert_eq!(
            msgs[0]["content"][1]["text"],
            "System note:\nTriage marked this request as multi-step."
        );
        assert_eq!(
            msgs[1]["content"][0],
            json!({ "type": "text", "text": "Reading." })
        );
        assert_eq!(
            msgs[1]["content"][1],
            json!({ "type": "tool_use", "id": "toolu_1", "name": "read_file", "input": { "path": "a" } })
        );
        assert_eq!(
            msgs[2]["content"][0],
            json!({ "type": "tool_result", "tool_use_id": "toolu_1", "content": "contents",
                    "cache_control": { "type": "ephemeral" } }),
            "the conversation so far is cached"
        );
        assert_eq!(
            body["tools"][0]["input_schema"],
            json!({ "type": "object" })
        );
        assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn thinking_needs_signed_blocks_in_tool_loops() {
        let sigs = Signatures::default();
        let mut req = ChatRequest::new("claude-sonnet-4-5", vec![Message::user("hi")]);
        req.think = ThinkLevel::High;
        let body = build_body(&req, 8192, &sigs);
        assert_eq!(
            body["thinking"],
            json!({ "type": "enabled", "budget_tokens": 24576 })
        );
        assert_eq!(body["max_tokens"], 24576 + 4096);
        assert!(body.get("temperature").is_none());

        // Continuing a tool loop without the signature: thinking is turned off for this request.
        req.messages = tool_turn(Some("let me read"));
        assert!(build_body(&req, 8192, &sigs).get("thinking").is_none());

        sigs.insert(vec![ThinkingBlock {
            text: "let me read".into(),
            signature: "sig==".into(),
        }]);
        let body = build_body(&req, 8192, &sigs);
        assert_eq!(body["thinking"]["budget_tokens"], 24576);
        assert_eq!(
            body["messages"][1]["content"][0],
            json!({ "type": "thinking", "thinking": "let me read", "signature": "sig==" })
        );

        req.model = "claude-3-5-sonnet-latest".into();
        assert!(build_body(&req, 8192, &sigs).get("thinking").is_none());
    }

    #[tokio::test]
    async fn parses_text_thinking_tools_and_usage() {
        let sigs = Arc::new(Signatures::default());
        let stream = sse_chunks(
            byte_stream(sse(&[
                (
                    "message_start",
                    r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":410,"cache_read_input_tokens":90,"output_tokens":1}}}"#,
                ),
                (
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
                ),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need the file."}}"#,
                ),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQB"}}"#,
                ),
                (
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                (
                    "content_block_start",
                    r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                ),
                ("ping", r#"{"type":"ping"}"#),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Reading."}}"#,
                ),
                (
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":1}"#,
                ),
                (
                    "content_block_start",
                    r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"read_file","input":{}}}"#,
                ),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"Car"}}"#,
                ),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"go.toml\"}"}}"#,
                ),
                (
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":2}"#,
                ),
                (
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":57}}"#,
                ),
                ("message_stop", r#"{"type":"message_stop"}"#),
            ])),
            Duration::from_secs(5),
            AnthropicParser::new(sigs.clone()),
        );
        let r = collect(stream).await.unwrap();
        assert_eq!(r.thinking, "Need the file.");
        assert_eq!(r.text, "Reading.");
        assert_eq!(r.tool_calls[0].id, "toolu_9");
        assert_eq!(r.tool_calls[0].arguments, json!({ "path": "Cargo.toml" }));
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 500,
                output_tokens: 57
            }
        );
        assert_eq!(sigs.get("Need the file.").unwrap()[0].signature, "EqQB");
    }

    #[tokio::test]
    async fn error_events_fail_the_stream() {
        let err = collect(sse_chunks(
            byte_stream(sse(&[(
                "error",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            )])),
            Duration::from_secs(5),
            AnthropicParser::new(Arc::default()),
        ))
        .await
        .unwrap_err();
        assert!(matches!(err, LlmError::Other(m) if m == "overloaded_error: Overloaded"));
    }
}
