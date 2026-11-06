//! OpenAI-compatible chat completions (DESIGN §5.3.3): `POST {base}/chat/completions` over SSE.

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use async_trait::async_trait;
use finality_core::ThinkLevel;
use finality_core::config::{LlmConfig, OpenAiConfig};
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

pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<Zeroizing<String>>,
    context_window: u32,
    request_timeout: Duration,
    retry: RetryPolicy,
    /// llama.cpp: `chat_template_kwargs.enable_thinking = false` when thinking is off.
    template_thinking: bool,
}

/// Models that take `reasoning_effort` (and reject `temperature`).
pub fn is_reasoning_model(model: &str) -> bool {
    let m = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    ["o1", "o3", "o4", "gpt-5"].iter().any(|p| m.starts_with(p))
}

impl OpenAiProvider {
    pub fn new(
        cfg: &OpenAiConfig,
        llm: &LlmConfig,
        api_key: Option<String>,
    ) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| LlmError::Other(format!("http client: {e}")))?;
        Ok(Self {
            client,
            base_url: cfg.base_url.trim_end_matches('/').to_owned(),
            api_key: api_key.map(Zeroizing::new),
            context_window: cfg.context_window,
            request_timeout: Duration::from_secs(llm.request_timeout_secs),
            retry: RetryPolicy::default(),
            template_thinking: false,
        })
    }

    pub fn with_template_thinking(mut self) -> Self {
        self.template_thinking = true;
        self
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
                    let mut builder = self.client.request(method, url);
                    if let Some(key) = &self.api_key {
                        builder = builder.bearer_auth(key.as_str());
                    }
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
impl LlmProvider for OpenAiProvider {
    fn id(&self) -> &'static str {
        "openai"
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
            .request(Method::GET, "/models", None, &CancellationToken::new())
            .await?;
        let models: Models = read_json(resp, self.request_timeout).await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    /// The API reports no context length; it comes from `provider.openai.context_window`.
    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        Ok(ModelInfo {
            name: model.to_owned(),
            context_length: Some(self.context_window),
            supports_tools: true,
            supports_thinking: is_reasoning_model(model),
            supports_vision: false,
        })
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let mut body = build_body(&req);
        if self.template_thinking && req.think == ThinkLevel::Off {
            body["chat_template_kwargs"] = json!({ "enable_thinking": false });
        }
        let resp = self
            .request(Method::POST, "/chat/completions", Some(&body), &cancel)
            .await?;
        Ok(cancellable(
            sse_chunks(
                body_stream(resp),
                self.request_timeout,
                OpenAiParser::default(),
            ),
            cancel,
        ))
    }
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
            let mut v = json!({ "role": "assistant", "content": content });
            if !tool_calls.is_empty() {
                if content.is_empty() {
                    v["content"] = Value::Null;
                }
                v["tool_calls"] = tool_calls
                    .iter()
                    .map(|c| {
                        json!({ "id": c.id, "type": "function", "function": {
                            "name": c.name, "arguments": c.arguments.to_string()
                        }})
                    })
                    .collect();
            }
            v
        }
        Message::Tool {
            call_id, content, ..
        } => json!({ "role": "tool", "tool_call_id": call_id, "content": content }),
    }
}

pub(crate) fn build_body(req: &ChatRequest) -> Value {
    let reasoning = is_reasoning_model(&req.model);
    let mut body = json!({
        "model": req.model,
        "messages": req.messages.iter().map(message_json).collect::<Vec<_>>(),
        "stream": true,
        "stream_options": { "include_usage": true },
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
    if let Some(schema) = &req.response_format {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": schema }
        });
    }
    if reasoning {
        if req.think != ThinkLevel::Off {
            body["reasoning_effort"] = json!(req.think.as_str());
        }
    } else if let Some(t) = req.temperature {
        body["temperature"] = f32_json(t);
    }
    body
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
pub(crate) struct OpenAiParser {
    calls: BTreeMap<u64, PartialCall>,
    finish: Option<String>,
    usage: Usage,
}

fn tokens(v: &Value, key: &str) -> u32 {
    v.get(key)
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
}

impl OpenAiParser {
    fn complete(&mut self, out: &mut VecDeque<Result<ChatChunk, LlmError>>) {
        let calls = std::mem::take(&mut self.calls);
        let had_calls = !calls.is_empty();
        for (i, c) in calls {
            let arguments = if c.arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&c.arguments).unwrap_or(Value::String(c.arguments))
            };
            out.push_back(Ok(ChatChunk::ToolCall(ToolCall {
                id: if c.id.is_empty() {
                    format!("call_{i}")
                } else {
                    c.id
                },
                name: c.name,
                arguments,
            })));
        }
        let stop_reason = match self.finish.as_deref() {
            Some("tool_calls" | "function_call") => StopReason::ToolUse,
            None | Some("stop") if had_calls => StopReason::ToolUse,
            None | Some("stop") => StopReason::Stop,
            Some("length") => StopReason::Length,
            Some(other) => StopReason::Other(other.to_owned()),
        };
        out.push_back(Ok(ChatChunk::Done {
            usage: self.usage,
            stop_reason,
        }));
    }
}

impl SseParser for OpenAiParser {
    fn event(&mut self, ev: SseEvent, out: &mut VecDeque<Result<ChatChunk, LlmError>>) -> bool {
        let data = ev.data.trim();
        if data == "[DONE]" {
            self.complete(out);
            return true;
        }
        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                out.push_back(Err(LlmError::Decode(format!("{e}: {data}"))));
                return true;
            }
        };
        if v.get("error").is_some() {
            out.push_back(Err(LlmError::Other(error_message(data))));
            return true;
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Usage {
                input_tokens: tokens(u, "prompt_tokens"),
                output_tokens: tokens(u, "completion_tokens"),
            };
        }
        let Some(choice) = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            return false;
        };
        let delta = &choice["delta"];
        for key in ["reasoning_content", "reasoning"] {
            if let Some(t) = delta
                .get(key)
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
            {
                out.push_back(Ok(ChatChunk::Thinking(t.to_owned())));
                break;
            }
        }
        if let Some(t) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
        {
            out.push_back(Ok(ChatChunk::Text(t.to_owned())));
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let entry = self.calls.entry(index).or_default();
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                entry.id = id.to_owned();
            }
            let function = &call["function"];
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                entry.name.push_str(name);
            }
            if let Some(args) = function.get("arguments").and_then(Value::as_str) {
                entry.arguments.push_str(args);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(reason.to_owned());
        }
        false
    }

    fn end(&mut self, out: &mut VecDeque<Result<ChatChunk, LlmError>>) {
        if self.finish.is_some() {
            self.complete(out);
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

    #[test]
    fn body_maps_messages_tools_and_reasoning() {
        let mut req = ChatRequest::new(
            "gpt-4.1",
            vec![
                Message::system("sys"),
                Message::user("hi"),
                Message::Assistant {
                    content: String::new(),
                    thinking: Some("hidden".into()),
                    tool_calls: vec![ToolCall {
                        id: "call_9".into(),
                        name: "read_file".into(),
                        arguments: json!({ "path": "a" }),
                    }],
                },
                Message::tool("call_9", "read_file", "contents"),
            ],
        );
        req.tools = vec![ToolSchema {
            name: "read_file".into(),
            description: "d".into(),
            parameters: json!({ "type": "object" }),
        }];
        req.temperature = Some(0.0);
        req.think = ThinkLevel::High;
        req.response_format = Some(json!({ "type": "object" }));
        let body = build_body(&req);
        assert_eq!(body["stream_options"]["include_usage"], true);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[2]["content"], Value::Null);
        assert!(msgs[2].get("thinking").is_none());
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_9");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a\"}"
        );
        assert_eq!(
            msgs[3],
            json!({ "role": "tool", "tool_call_id": "call_9", "content": "contents" })
        );
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            json!({ "type": "object" })
        );
        assert_eq!(body["temperature"], 0.0);
        assert!(body.get("reasoning_effort").is_none());

        req.model = "o4-mini".into();
        let body = build_body(&req);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("temperature").is_none());
        req.think = ThinkLevel::Off;
        assert!(build_body(&req).get("reasoning_effort").is_none());
    }

    fn events(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|l| format!("data: {l}\n\n")).collect()
    }

    #[tokio::test]
    async fn assembles_tool_calls_from_fragments() {
        let stream = sse_chunks(
            byte_stream(events(&[
                r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}},{"index":1,"id":"call_b","function":{"name":"shell","arguments":"{}"}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                r#"{"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150}}"#,
                "[DONE]",
            ])),
            Duration::from_secs(5),
            OpenAiParser::default(),
        );
        let r = collect(stream).await.unwrap();
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].id, "call_a");
        assert_eq!(r.tool_calls[0].arguments, json!({ "path": "x" }));
        assert_eq!(r.tool_calls[1].name, "shell");
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 120,
                output_tokens: 30
            }
        );
    }

    #[tokio::test]
    async fn text_reasoning_and_errors() {
        let ok = collect(sse_chunks(
            byte_stream(events(&[
                r#"{"choices":[{"delta":{"reasoning_content":"think "}}]}"#,
                r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
                r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}"#,
                "[DONE]",
            ])),
            Duration::from_secs(5),
            OpenAiParser::default(),
        ))
        .await
        .unwrap();
        assert_eq!(
            (ok.thinking.as_str(), ok.text.as_str()),
            ("think ", "Hello")
        );
        assert_eq!(ok.stop_reason, StopReason::Stop);

        let err = collect(sse_chunks(
            byte_stream(events(&[
                r#"{"error":{"message":"Rate limit reached","type":"requests"}}"#,
            ])),
            Duration::from_secs(5),
            OpenAiParser::default(),
        ))
        .await
        .unwrap_err();
        assert!(matches!(err, LlmError::Other(m) if m == "Rate limit reached"));

        let cut = collect(sse_chunks(
            byte_stream(events(&[r#"{"choices":[{"delta":{"content":"x"}}]}"#])),
            Duration::from_secs(5),
            OpenAiParser::default(),
        ))
        .await
        .unwrap_err();
        assert!(matches!(cut, LlmError::Decode(_)));
    }
}
