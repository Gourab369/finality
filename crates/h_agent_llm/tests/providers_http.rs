#![allow(clippy::unwrap_used)]

use std::time::Duration;

use h_agent_core::ThinkLevel;
use h_agent_core::config::{AnthropicConfig, LlamaCppConfig, LlmConfig, OpenAiConfig};
use h_agent_llm::{
    AnthropicProvider, ChatRequest, LlamaCppProvider, LlmError, LlmProvider, Message,
    OpenAiProvider, RetryPolicy, StopReason, ToolSchema, Usage, collect,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const OPENAI_TOOL: &str = include_str!("fixtures/openai/tool_call.sse");
const ANTHROPIC_TOOL: &str = include_str!("fixtures/anthropic/tool_call.sse");

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

fn fast() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_delay: Duration::from_millis(1),
    }
}

fn tool() -> ToolSchema {
    ToolSchema {
        name: "read_file".into(),
        description: "Read a file".into(),
        parameters: json!({ "type": "object", "properties": { "path": { "type": "string" } } }),
    }
}

fn openai(server: &MockServer, key: Option<&str>) -> OpenAiProvider {
    let cfg = OpenAiConfig {
        base_url: format!("{}/v1", server.uri()),
        ..OpenAiConfig::default()
    };
    OpenAiProvider::new(&cfg, &LlmConfig::default(), key.map(str::to_owned))
        .unwrap()
        .with_retry(fast())
}

fn anthropic(server: &MockServer) -> AnthropicProvider {
    let cfg = AnthropicConfig {
        base_url: server.uri(),
        ..AnthropicConfig::default()
    };
    AnthropicProvider::new(&cfg, &LlmConfig::default(), "sk-ant-test".into())
        .unwrap()
        .with_retry(fast())
}

async fn bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}

#[tokio::test]
async fn openai_streams_tool_calls_with_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(sse(OPENAI_TOOL))
        .expect(1)
        .mount(&server)
        .await;
    let mut req = ChatRequest::new("gpt-4.1", vec![Message::user("read Cargo.toml")]);
    req.tools = vec![tool()];
    let r = collect(
        openai(&server, Some("sk-test"))
            .chat(req, CancellationToken::new())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].id, "call_Qx7Rk2");
    assert_eq!(r.tool_calls[0].arguments, json!({ "path": "Cargo.toml" }));
    assert_eq!(r.stop_reason, StopReason::ToolUse);
    assert_eq!(
        r.usage,
        Usage {
            input_tokens: 512,
            output_tokens: 18
        }
    );
    let body = &bodies(&server).await[0];
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
}

#[tokio::test]
async fn openai_errors_and_retries() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(
            json!({ "error": { "message": "Rate limit reached", "type": "requests" } }),
        ))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(sse(include_str!("fixtures/openai/answer.sse")))
        .mount(&server)
        .await;
    let p = openai(&server, None);
    let r = collect(
        p.chat(
            ChatRequest::new("gpt-4.1", vec![Message::user("hi")]),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(r.text, "The crate is named `x`.");
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        2,
        "retried the 429"
    );
    assert!(
        server.received_requests().await.unwrap()[0]
            .headers
            .get("authorization")
            .is_none(),
        "no key, no header"
    );

    let denied = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({ "error": { "message": "Incorrect API key provided", "code": "invalid_api_key" } }),
        ))
        .mount(&denied)
        .await;
    let err = openai(&denied, Some("bad"))
        .chat(
            ChatRequest::new("gpt-4.1", vec![Message::user("hi")]),
            CancellationToken::new(),
        )
        .await
        .err()
        .unwrap();
    assert!(
        matches!(&err, LlmError::Status { status: 401, body, .. } if body == "Incorrect API key provided"),
        "{err:?}"
    );
}

#[tokio::test]
async fn model_lists_and_info() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "object": "list", "data": [{ "id": "gpt-4.1", "object": "model" }, { "id": "o4-mini", "object": "model" }] }),
        ))
        .mount(&server)
        .await;
    let o = openai(&server, Some("k"));
    assert_eq!(o.list_models().await.unwrap(), ["gpt-4.1", "o4-mini"]);
    let info = o.model_info("o4-mini").await.unwrap();
    assert_eq!(info.context_length, Some(128_000));
    assert!(info.supports_thinking);

    let a = anthropic(&server);
    assert_eq!(
        a.list_models().await.unwrap(),
        ["gpt-4.1", "o4-mini"],
        "same /v1/models shape"
    );
    assert_eq!(
        a.model_info("claude-sonnet-4-5")
            .await
            .unwrap()
            .context_length,
        Some(200_000)
    );
    assert!(
        !a.model_info("claude-3-5-haiku-latest")
            .await
            .unwrap()
            .supports_thinking
    );
}

#[tokio::test]
async fn llamacpp_uses_a_running_server_and_turns_thinking_off() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "object": "list", "data": [{ "id": "qwen3-8b-q4_k_m.gguf", "object": "model" }] }),
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse(include_str!("fixtures/openai/answer.sse")))
        .mount(&server)
        .await;
    let cfg = LlamaCppConfig {
        base_url: server.uri(),
        context_window: 8192,
        ..LlamaCppConfig::default()
    };
    let p = LlamaCppProvider::new(&cfg, &LlmConfig::default()).unwrap();
    assert_eq!(p.list_models().await.unwrap(), ["qwen3-8b-q4_k_m.gguf"]);
    let info = p.model_info("qwen3-8b-q4_k_m.gguf").await.unwrap();
    assert_eq!(info.context_length, Some(8192));
    assert!(info.supports_tools);

    let mut req = ChatRequest::new("qwen3-8b-q4_k_m.gguf", vec![Message::user("hi")]);
    req.think = ThinkLevel::Off;
    let r = collect(p.chat(req, CancellationToken::new()).await.unwrap())
        .await
        .unwrap();
    assert_eq!(r.text, "The crate is named `x`.");
    let mut req = ChatRequest::new("qwen3-8b-q4_k_m.gguf", vec![Message::user("think")]);
    req.think = ThinkLevel::High;
    collect(p.chat(req, CancellationToken::new()).await.unwrap())
        .await
        .unwrap();
    let sent: Vec<Value> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.method.as_str() == "POST")
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect();
    assert_eq!(sent[0]["chat_template_kwargs"]["enable_thinking"], false);
    assert!(sent[1].get("chat_template_kwargs").is_none());
}

#[tokio::test]
async fn anthropic_streams_tool_calls_with_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(sse(ANTHROPIC_TOOL))
        .expect(1)
        .mount(&server)
        .await;
    let mut req = ChatRequest::new(
        "claude-sonnet-4-5",
        vec![Message::system("sys"), Message::user("read Cargo.toml")],
    );
    req.tools = vec![tool()];
    let r = collect(
        anthropic(&server)
            .chat(req, CancellationToken::new())
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(r.text, "I'll read the manifest.");
    assert_eq!(r.tool_calls[0].id, "toolu_01T1x1fJ34qAmk2tNTrN7Up6");
    assert_eq!(r.tool_calls[0].arguments, json!({ "path": "Cargo.toml" }));
    assert_eq!(r.stop_reason, StopReason::ToolUse);
    assert_eq!(
        r.usage,
        Usage {
            input_tokens: 498,
            output_tokens: 61
        }
    );
    let body = &bodies(&server).await[0];
    assert_eq!(body["system"], "sys");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
}

#[tokio::test]
async fn anthropic_sends_signed_thinking_back_in_tool_loops() {
    let server = MockServer::start().await;
    let thinking = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n\
        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n\
        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"I should read it.\"}}\n\n\
        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"EuYBCkQYAiJA\"}}\n\n\
        event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_7\",\"name\":\"read_file\",\"input\":{}}}\n\n\
        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a\\\"}\"}}\n\n\
        event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
        event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":40}}\n\n\
        event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    Mock::given(path("/v1/messages"))
        .respond_with(sse(thinking))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(path("/v1/messages"))
        .respond_with(sse(include_str!("fixtures/anthropic/answer.sse")))
        .mount(&server)
        .await;
    let p = anthropic(&server);
    let mut req = ChatRequest::new("claude-sonnet-4-5", vec![Message::user("read a")]);
    req.think = ThinkLevel::Medium;
    req.tools = vec![tool()];
    let first = collect(p.chat(req.clone(), CancellationToken::new()).await.unwrap())
        .await
        .unwrap();
    assert_eq!(first.thinking, "I should read it.");
    req.messages.push(Message::Assistant {
        content: first.text.clone(),
        thinking: Some(first.thinking.clone()),
        tool_calls: first.tool_calls.clone(),
    });
    req.messages
        .push(Message::tool("toolu_7", "read_file", "contents of a"));
    let second = collect(p.chat(req, CancellationToken::new()).await.unwrap())
        .await
        .unwrap();
    assert_eq!(second.text, "The crate is named `x`.");
    assert_eq!(second.usage.input_tokens, 620, "cache reads count as input");

    let sent = &bodies(&server).await[1];
    assert_eq!(
        sent["thinking"],
        json!({ "type": "enabled", "budget_tokens": 8192 })
    );
    assert_eq!(
        sent["messages"][1]["content"][0],
        json!({ "type": "thinking", "thinking": "I should read it.", "signature": "EuYBCkQYAiJA" })
    );
    assert_eq!(sent["messages"][1]["content"][1]["type"], "tool_use");
    assert_eq!(
        sent["messages"][2]["content"][0],
        json!({ "type": "tool_result", "tool_use_id": "toolu_7", "content": "contents of a" })
    );
}
