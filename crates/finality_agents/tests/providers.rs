//! Phase 15 acceptance: the same agent turn through each hosted provider, fed recorded SSE streams,
//! and `/provider` switching.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_agents::{Core, CoreDeps};
use finality_core::config::Config;
use finality_core::events::{CoreCommand, NoticeLevel, SlashCommand, ToolStatus, UiEvent};
use finality_core::{ProviderKind, Workspace};
use finality_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, MockProvider, ModelInfo, StaticKeys,
    cancellable,
};
use finality_store::DataDir;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::path;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const WAIT: Duration = Duration::from_secs(20);

/// Replays recorded streams: triage JSON, then a `read_file` call, then the answer.
struct Recorded(ProviderKind);

fn fixture(kind: ProviderKind, name: &str) -> &'static str {
    match (kind, name) {
        (ProviderKind::OpenAi, "triage") => {
            include_str!("../../finality_llm/tests/fixtures/openai/triage.sse")
        }
        (ProviderKind::OpenAi, "tool_call") => {
            include_str!("../../finality_llm/tests/fixtures/openai/tool_call.sse")
        }
        (ProviderKind::OpenAi, _) => {
            include_str!("../../finality_llm/tests/fixtures/openai/answer.sse")
        }
        (_, "triage") => include_str!("../../finality_llm/tests/fixtures/anthropic/triage.sse"),
        (_, "tool_call") => {
            include_str!("../../finality_llm/tests/fixtures/anthropic/tool_call.sse")
        }
        _ => include_str!("../../finality_llm/tests/fixtures/anthropic/answer.sse"),
    }
}

impl Respond for Recorded {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let last = body["messages"].as_array().unwrap().last().unwrap().clone();
        let name = match self.0 {
            ProviderKind::OpenAi if body.get("response_format").is_some() => "triage",
            ProviderKind::OpenAi if last["role"] == "tool" => "answer",
            ProviderKind::OpenAi => "tool_call",
            _ if body["system"]
                .as_str()
                .unwrap_or("")
                .contains("Reply with only a JSON object") =>
            {
                "triage"
            }
            _ if last["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|b| b["type"] == "tool_result") =>
            {
                "answer"
            }
            _ => "tool_call",
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(fixture(self.0, name))
    }
}

/// Ollama stand-in: triage JSON, otherwise a short answer.
#[derive(Default)]
struct Local {
    requests: Mutex<Vec<ChatRequest>>,
}

#[async_trait]
impl LlmProvider for Local {
    fn id(&self) -> &'static str {
        "local"
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn model_info(&self, _: &str) -> Result<ModelInfo, LlmError> {
        Ok(ModelInfo {
            context_length: Some(32_768),
            ..ModelInfo::default()
        })
    }
    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let text = if req.response_format.is_some() {
            json!({ "query_type": "question", "topic": "general", "complexity": "simple", "needs_tools": false })
                .to_string()
        } else {
            "local answer".to_owned()
        };
        self.requests.lock().unwrap().push(req);
        let chunks = vec![
            Ok(ChatChunk::Text(text)),
            Ok(ChatChunk::Done {
                usage: Default::default(),
                stop_reason: Default::default(),
            }),
        ];
        Ok(cancellable(Box::pin(futures::stream::iter(chunks)), cancel))
    }
}

struct Run {
    _tmp: tempfile::TempDir,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
}

fn hosted_config(server: &MockServer, default: ProviderKind) -> Config {
    let mut config = Config::default();
    config.provider.default = default;
    config.provider.openai.base_url = format!("{}/v1", server.uri());
    config.provider.openai.model = "gpt-4.1".into();
    config.provider.openai.api_key_env = "FINALITY_TEST_OPENAI_KEY".into();
    config.provider.anthropic.base_url = server.uri();
    config.provider.anthropic.model = "claude-sonnet-4-5".into();
    config.provider.anthropic.api_key_env = "FINALITY_TEST_ANTHROPIC_KEY".into();
    config
}

fn start(config: Config, local: Arc<Local>, keys: StaticKeys) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data,
        provider: local,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: Some(Arc::new(keys)),
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    Run {
        _tmp: tmp,
        ui,
        cmd,
        core,
        seen: Vec::new(),
    }
}

impl Run {
    async fn until(&mut self, mut pred: impl FnMut(&UiEvent) -> bool) -> UiEvent {
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            let Ok(ev) = tokio::time::timeout_at(deadline, self.ui.recv()).await else {
                let tail = &self.seen[self.seen.len().saturating_sub(8)..];
                panic!("timed out; last events: {tail:#?}");
            };
            let ev = ev.expect("core ended");
            if let UiEvent::Approval(p) = &ev {
                self.cmd
                    .send(CoreCommand::ApprovalResponse {
                        id: p.id,
                        choice: finality_core::events::ApprovalChoice::AllowOnce,
                    })
                    .unwrap();
            }
            self.seen.push(ev.clone());
            if pred(&ev) {
                return ev;
            }
        }
    }

    async fn turn(&mut self, prompt: &str) -> Vec<UiEvent> {
        let from = self.seen.len();
        self.cmd
            .send(CoreCommand::UserPrompt(prompt.into()))
            .unwrap();
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
            .await;
        self.seen[from..].to_vec()
    }

    async fn notice(&mut self, slash: SlashCommand) -> (NoticeLevel, String) {
        self.cmd.send(CoreCommand::Slash(slash)).unwrap();
        match self.until(|e| matches!(e, UiEvent::Notice { .. })).await {
            UiEvent::Notice { level, text } => (level, text),
            _ => unreachable!(),
        }
    }

    async fn quit(self) -> Vec<UiEvent> {
        self.cmd.send(CoreCommand::Quit).unwrap();
        tokio::time::timeout(WAIT, self.core)
            .await
            .unwrap()
            .unwrap();
        self.seen
    }
}

fn answer(events: &[UiEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            UiEvent::AssistantDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
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

/// The agent turn every provider must support: triage, a tool call, and the answer with usage.
async fn tool_turn_through(kind: ProviderKind) -> (Vec<UiEvent>, Vec<Value>) {
    let server = MockServer::start().await;
    let route = if kind == ProviderKind::OpenAi {
        "/v1/chat/completions"
    } else {
        "/v1/messages"
    };
    Mock::given(path(route))
        .respond_with(Recorded(kind))
        .mount(&server)
        .await;
    let local = Arc::new(Local::default());
    let mut run = start(
        hosted_config(&server, kind),
        local.clone(),
        StaticKeys::default().with(kind, "test-key"),
    );
    let status = run
        .until(|e| matches!(e, UiEvent::Status(s) if s.provider == kind))
        .await;
    let UiEvent::Status(s) = status else {
        unreachable!()
    };
    assert_eq!(
        s.model,
        if kind == ProviderKind::OpenAi {
            "gpt-4.1"
        } else {
            "claude-sonnet-4-5"
        }
    );
    let events = run.turn("what is the crate name in Cargo.toml?").await;
    run.quit().await;
    assert!(local.requests.lock().unwrap().is_empty(), "Ollama unused");
    (events, bodies(&server).await)
}

fn check_turn(events: &[UiEvent], expected_session_in: u64, window: u32) {
    assert!(
        answer(events).ends_with("The crate is named `x`."),
        "{}",
        answer(events)
    );
    assert!(events.iter().any(|e| matches!(
        e,
        UiEvent::ToolFinished { status: ToolStatus::Succeeded, output, .. } if output.contains("name = \"x\"")
    )));
    let usage = events
        .iter()
        .rev()
        .find_map(|e| match e {
            UiEvent::Usage(u) => Some(*u),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        usage.session_in, expected_session_in,
        "triage + tool call + answer"
    );
    assert_eq!(usage.ctx_total, window);
}

#[tokio::test]
async fn openai_runs_the_agent_turn() {
    let (events, bodies) = tool_turn_through(ProviderKind::OpenAi).await;
    check_turn(&events, 231 + 512 + 640, 128_000);
    assert_eq!(bodies.len(), 3);
    assert!(bodies[0]["response_format"]["json_schema"]["schema"].is_object());
    assert!(
        bodies[1]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["function"]["name"] == "read_file")
    );
    let tool_result = bodies[2]["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(tool_result["tool_call_id"], "call_Qx7Rk2");
    assert!(
        tool_result["content"]
            .as_str()
            .unwrap()
            .contains("name = \"x\"")
    );
}

#[tokio::test]
async fn anthropic_runs_the_agent_turn() {
    let (events, bodies) = tool_turn_through(ProviderKind::Anthropic).await;
    check_turn(&events, 266 + 498 + 620, 200_000);
    assert_eq!(bodies.len(), 3);
    assert!(
        bodies[1]["system"]
            .as_str()
            .unwrap()
            .starts_with("You are Finality")
    );
    let msgs = bodies[2]["messages"].as_array().unwrap();
    let assistant = &msgs[msgs.len() - 2];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["content"][0]["text"], "I'll read the manifest.");
    assert_eq!(
        assistant["content"][1]["id"],
        "toolu_01T1x1fJ34qAmk2tNTrN7Up6"
    );
    let result = &msgs[msgs.len() - 1]["content"][0];
    assert_eq!(result["type"], "tool_result");
    assert_eq!(result["tool_use_id"], "toolu_01T1x1fJ34qAmk2tNTrN7Up6");
}

#[tokio::test]
async fn provider_switch_routes_the_next_turn() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/messages"))
        .respond_with(Recorded(ProviderKind::Anthropic))
        .mount(&server)
        .await;
    let local = Arc::new(Local::default());
    let mut run = start(
        hosted_config(&server, ProviderKind::Ollama),
        local.clone(),
        StaticKeys::default().with(ProviderKind::Anthropic, "test-key"),
    );
    run.turn("hello").await;
    let before = local.requests.lock().unwrap().len();
    assert!(before >= 2, "triage and answer went to Ollama");

    let (level, text) = run
        .notice(SlashCommand::Provider {
            provider: ProviderKind::OpenAi,
            model: None,
        })
        .await;
    assert_eq!(level, NoticeLevel::Error);
    assert!(text.contains("set FINALITY_TEST_OPENAI_KEY"), "{text}");

    let (level, text) = run
        .notice(SlashCommand::Provider {
            provider: ProviderKind::Anthropic,
            model: None,
        })
        .await;
    assert_eq!(level, NoticeLevel::Info, "{text}");
    assert_eq!(
        text,
        "provider: anthropic, model: claude-sonnet-4-5 (context window 200000)"
    );

    let events = run.turn("what is the crate name in Cargo.toml?").await;
    assert_eq!(
        answer(&events),
        "I'll read the manifest.The crate is named `x`."
    );
    assert_eq!(
        local.requests.lock().unwrap().len(),
        before,
        "nothing more went to Ollama"
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
    let seen = run.quit().await;
    assert!(seen.iter().any(|e| matches!(e, UiEvent::Status(s) if s.provider == ProviderKind::Anthropic && s.model == "claude-sonnet-4-5")));
}

#[tokio::test]
async fn missing_model_and_key_are_refused() {
    let server = MockServer::start().await;
    let mut config = hosted_config(&server, ProviderKind::OpenAi);
    config.provider.anthropic.model.clear();
    let local = Arc::new(Local::default());
    let mut run = start(config, local.clone(), StaticKeys::default());

    let (level, text) = notice_parts(run.until(|e| matches!(e, UiEvent::Notice { .. })).await);
    assert_eq!(level, NoticeLevel::Error);
    assert!(
        text.starts_with(
            "cannot use provider `openai`: no API key for openai: set FINALITY_TEST_OPENAI_KEY"
        ) && text.ends_with("; using ollama"),
        "{text}"
    );
    let (_, text) = run
        .notice(SlashCommand::Provider {
            provider: ProviderKind::Anthropic,
            model: None,
        })
        .await;
    assert_eq!(
        text,
        "no model for anthropic: run /provider anthropic <model>, or set provider.anthropic.model in config.toml"
    );
    run.turn("hello").await;
    assert!(
        !local.requests.lock().unwrap().is_empty(),
        "fell back to Ollama"
    );
    run.quit().await;
}

fn notice_parts(event: UiEvent) -> (NoticeLevel, String) {
    match event {
        UiEvent::Notice { level, text } => (level, text),
        other => panic!("not a notice: {other:?}"),
    }
}

#[tokio::test]
async fn llamacpp_provider_picks_a_model_and_runs_the_turn() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({ "object": "list", "data": [{ "id": "qwen3-8b-q4_k_m.gguf", "object": "model" }] }),
        ))
        .mount(&server)
        .await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(Recorded(ProviderKind::OpenAi))
        .mount(&server)
        .await;
    let mut config = Config::default();
    config.provider.llamacpp.base_url = server.uri();
    let local = Arc::new(Local::default());
    let mut run = start(config, local.clone(), StaticKeys::default());

    // Without a model, `/provider llamacpp` lists what the server offers.
    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Provider {
            provider: ProviderKind::LlamaCpp,
            model: None,
        }))
        .unwrap();
    let UiEvent::Picker(picker) = run.until(|e| matches!(e, UiEvent::Picker(_))).await else {
        unreachable!()
    };
    assert_eq!(picker.items[0].value, "qwen3-8b-q4_k_m.gguf");
    run.cmd
        .send(CoreCommand::Picked {
            kind: picker.kind,
            value: Some(picker.items[0].value.clone()),
        })
        .unwrap();
    run.until(|e| matches!(e, UiEvent::Status(s) if s.provider == ProviderKind::LlamaCpp && s.model == "qwen3-8b-q4_k_m.gguf"))
        .await;

    let events = run.turn("read Cargo.toml").await;
    assert_eq!(answer(&events), "The crate is named `x`.");
    assert!(
        local.requests.lock().unwrap().is_empty(),
        "nothing went to Ollama"
    );
    let sent: Vec<Value> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.method.as_str() == "POST")
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect();
    assert!(!sent.is_empty() && sent.iter().all(|b| b["model"] == "qwen3-8b-q4_k_m.gguf"));
    run.quit().await;
}
