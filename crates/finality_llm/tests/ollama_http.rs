//! Ollama provider against a mock HTTP server.
#![allow(clippy::unwrap_used)]

use std::time::{Duration, Instant};

use finality_core::ThinkLevel;
use finality_core::config::{LlmConfig, OllamaConfig};
use finality_llm::{
    ChatChunk, ChatRequest, LlmError, LlmProvider, Message, OllamaProvider, RetryPolicy,
    StopReason, ToolSchema, Usage, collect,
};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(base_url: &str) -> OllamaProvider {
    let cfg = OllamaConfig {
        base_url: base_url.into(),
        ..OllamaConfig::default()
    };
    OllamaProvider::new(&cfg, &LlmConfig::default())
        .unwrap()
        .with_retry(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
        })
}

async fn mount_show(server: &MockServer, ctx: u64, caps: &[&str]) {
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "capabilities": caps,
            "model_info": { "general.architecture": "qwen3", "qwen3.context_length": ctx }
        })))
        .mount(server)
        .await;
}

fn ndjson(lines: &[Value]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

async fn chat_body(server: &MockServer) -> Value {
    let reqs = server.received_requests().await.unwrap();
    let chat = reqs
        .iter()
        .rev()
        .find(|r| r.url.path() == "/api/chat")
        .unwrap();
    serde_json::from_slice(&chat.body).unwrap()
}

#[tokio::test]
async fn streams_text_tool_call_and_usage() {
    let server = MockServer::start().await;
    mount_show(&server, 40960, &["completion", "tools", "thinking"]).await;
    let body = ndjson(&[
        json!({"message": {"role": "assistant", "content": "Hello "}, "done": false}),
        json!({"message": {"role": "assistant", "content": "world"}, "done": false}),
        json!({"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "Cargo.toml"}}}
        ]}, "done": false}),
        json!({"done": true, "done_reason": "stop", "prompt_eval_count": 26, "eval_count": 12}),
    ]);
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson"))
        .mount(&server)
        .await;

    let p = provider(&server.uri());
    let mut req = ChatRequest::new("qwen3:14b", vec![Message::user("read Cargo.toml")]);
    req.think = ThinkLevel::Medium;
    req.tools = vec![ToolSchema {
        name: "read_file".into(),
        description: "Read a file".into(),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
    }];
    let r = collect(p.chat(req, CancellationToken::new()).await.unwrap())
        .await
        .unwrap();

    assert_eq!(r.text, "Hello world");
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].name, "read_file");
    assert_eq!(r.tool_calls[0].arguments, json!({"path": "Cargo.toml"}));
    assert_eq!(
        r.usage,
        Usage {
            input_tokens: 26,
            output_tokens: 12
        }
    );
    assert_eq!(r.stop_reason, StopReason::ToolUse);

    let sent = chat_body(&server).await;
    assert_eq!(sent["options"]["num_ctx"], 32768);
    assert_eq!(sent["think"], true);
    assert_eq!(sent["tools"][0]["function"]["name"], "read_file");
}

#[tokio::test]
async fn streams_thinking_chunks_in_order() {
    let server = MockServer::start().await;
    mount_show(&server, 40960, &["completion", "thinking"]).await;
    let body = ndjson(&[
        json!({"message": {"content": "", "thinking": "Let me "}, "done": false}),
        json!({"message": {"content": "", "thinking": "think."}, "done": false}),
        json!({"message": {"content": "Answer"}, "done": false}),
        json!({"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3}),
    ]);
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson"))
        .mount(&server)
        .await;

    let p = provider(&server.uri());
    let stream = p
        .chat(
            ChatRequest::new("qwen3:14b", vec![Message::user("q")]),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let chunks: Vec<ChatChunk> = stream.map(Result::unwrap).collect().await;
    assert_eq!(chunks[0], ChatChunk::Thinking("Let me ".into()));
    assert_eq!(chunks[1], ChatChunk::Thinking("think.".into()));
    assert_eq!(chunks[2], ChatChunk::Text("Answer".into()));
    assert!(matches!(chunks[3], ChatChunk::Done { .. }));
}

#[tokio::test]
async fn context_length_is_extracted_and_clamps_num_ctx() {
    let server = MockServer::start().await;
    mount_show(&server, 8192, &["completion"]).await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            ndjson(&[json!({"done": true, "done_reason": "stop"})]),
            "application/x-ndjson",
        ))
        .mount(&server)
        .await;

    let p = provider(&server.uri());
    let info = p.model_info("qwen3:14b").await.unwrap();
    assert_eq!(info.context_length, Some(8192));
    assert!(!info.supports_thinking);

    collect(
        p.chat(
            ChatRequest::new("qwen3:14b", vec![]),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    )
    .await
    .unwrap();
    let sent = chat_body(&server).await;
    assert_eq!(sent["options"]["num_ctx"], 8192);
    assert!(
        sent.get("think").is_none(),
        "no think for non-thinking model"
    );

    let shows = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() == "/api/show")
        .count();
    assert_eq!(shows, 1, "model info is cached");
}

#[tokio::test]
async fn lists_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"name": "qwen3:14b"}, {"name": "llama3:8b"}]
        })))
        .mount(&server)
        .await;
    let models = provider(&server.uri()).list_models().await.unwrap();
    assert_eq!(models, vec!["qwen3:14b", "llama3:8b"]);
}

#[tokio::test]
async fn retries_transient_status_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": []})))
        .with_priority(2)
        .mount(&server)
        .await;
    assert!(
        provider(&server.uri())
            .list_models()
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn missing_model_reports_server_error_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/show"))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(json!({"error": "model 'nope' not found"})),
        )
        .mount(&server)
        .await;
    let err = provider(&server.uri())
        .model_info("nope")
        .await
        .unwrap_err();
    match err {
        LlmError::Status { status, body, .. } => {
            assert_eq!(status, 404);
            assert_eq!(body, "model 'nope' not found");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn unreachable_server_is_reported() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    // Windows takes ~2s to refuse a connection, so don't retry here.
    let p = provider(&format!("http://{addr}")).with_retry(RetryPolicy {
        max_attempts: 1,
        base_delay: Duration::ZERO,
    });
    let err = p.list_models().await.unwrap_err();
    assert!(matches!(err, LlmError::Unreachable { .. }), "{err:?}");
}

/// Serves `/api/show`, then streams one chat line and keeps the connection open.
async fn hanging_chat_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                loop {
                    let mut buf = vec![0u8; 16 * 1024];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.starts_with("POST /api/show") {
                        let body =
                            json!({"capabilities": ["completion"], "model_info": {}}).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        );
                        sock.write_all(resp.as_bytes()).await.unwrap();
                    } else if req.starts_with("POST /api/chat") {
                        let line = format!(
                            "{}\n",
                            json!({"message": {"content": "Hel"}, "done": false})
                        );
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{line}\r\n",
                            line.len()
                        );
                        sock.write_all(resp.as_bytes()).await.unwrap();
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn cancel_mid_stream_returns_quickly() {
    let p = provider(&hanging_chat_server().await);
    let cancel = CancellationToken::new();
    let mut stream = p
        .chat(
            ChatRequest::new("m", vec![Message::user("hi")]),
            cancel.clone(),
        )
        .await
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatChunk::Text("Hel".into())
    );

    let started = Instant::now();
    cancel.cancel();
    let next = stream.next().await;
    let elapsed = started.elapsed();
    assert!(matches!(next, Some(Err(LlmError::Cancelled))), "{next:?}");
    assert!(elapsed < Duration::from_millis(100), "took {elapsed:?}");
    assert!(stream.next().await.is_none());
}
