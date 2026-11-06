//! Custom agents acceptance (DESIGN §6.5): the Orchestrator creates one, plan steps run with it, `/agent delete`
//! removes it, and plan mode cannot create one.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::config::Config;
use h_agent_core::events::{
    AgentAction, ApprovalChoice, CoreCommand, NoticeLevel, SlashCommand, UiEvent,
};
use h_agent_core::{Mode, Role, Workspace};
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    ToolCall, cancellable,
};
use h_agent_memory::agents::{AgentDraft, AgentLibrary};
use h_agent_store::DataDir;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(30);
const CHECKLIST: &str = "REVIEW-CHECKLIST: look for unwrap() and missing tests.";

enum Reply {
    Text(&'static str),
    Call(&'static str, Value),
}

fn is_sub(req: &ChatRequest) -> bool {
    req.messages.first().is_some_and(|m| {
        let c = m.content();
        c.starts_with("You are a worker sub-agent") || c.starts_with("You are a research sub-agent")
    })
}

/// Tool results after the last user message (this turn's).
fn turn_results(req: &ChatRequest) -> Vec<&str> {
    let start = req
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }))
        .unwrap_or(0);
    req.messages[start..]
        .iter()
        .filter_map(|m| match m {
            Message::Tool { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

fn last_user(req: &ChatRequest) -> &str {
    req.messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .unwrap_or("")
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

struct Scripted {
    orchestrator: Box<Policy>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl Scripted {
    fn new(orchestrator: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            orchestrator: Box::new(orchestrator),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self, sub: bool) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.response_format.is_none() && is_sub(r) == sub)
            .cloned()
            .collect()
    }
}

#[async_trait]
impl LlmProvider for Scripted {
    fn id(&self) -> &'static str {
        "scripted"
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
        self.requests.lock().unwrap().push(req.clone());
        let reply = if req.response_format.is_some() {
            return Ok(stream(
                ChatChunk::Text(
                    json!({ "query_type": "task", "topic": "review", "complexity": "multi_step",
                            "needs_tools": true, "tool_hints": [] })
                    .to_string(),
                ),
                cancel,
            ));
        } else if is_sub(&req) {
            if turn_results(&req).is_empty() {
                Reply::Call(
                    "report_result",
                    json!({ "outcome": "success", "summary": "no issues", "evidence": [] }),
                )
            } else {
                Reply::Text("reported")
            }
        } else {
            (self.orchestrator)(&req)
        };
        let chunk = match reply {
            Reply::Text(t) => ChatChunk::Text(t.into()),
            Reply::Call(name, arguments) => ChatChunk::ToolCall(ToolCall {
                id: format!("call_{}_{name}", req.messages.len()),
                name: name.to_owned(),
                arguments,
            }),
        };
        Ok(stream(chunk, cancel))
    }
}

fn stream(chunk: ChatChunk, cancel: CancellationToken) -> ChatStream {
    let chunks = vec![
        Ok(chunk),
        Ok(ChatChunk::Done {
            usage: Default::default(),
            stop_reason: Default::default(),
        }),
    ];
    cancellable(Box::pin(futures::stream::iter(chunks)), cancel)
}

struct Run {
    _tmp: tempfile::TempDir,
    data: DataDir,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
}

fn start(provider: Arc<dyn LlmProvider>, mode: Mode, setup: impl FnOnce(&DataDir)) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work: PathBuf = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("lib.rs"), "pub fn f() {}\n").unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    setup(&data);
    let mut config = Config::default();
    config.general.default_mode = mode;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data.clone(),
        provider,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: None,
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    Run {
        _tmp: tmp,
        data,
        ui,
        cmd,
        core,
        seen: Vec::new(),
    }
}

impl Run {
    /// Waits for an event, approving every request; one overall deadline so heartbeats can't hide a hang.
    async fn until(&mut self, mut pred: impl FnMut(&UiEvent) -> bool) {
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            let ev = tokio::time::timeout_at(deadline, self.ui.recv())
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "timed out; last events: {:?}",
                        self.seen.iter().rev().take(5).collect::<Vec<_>>()
                    )
                })
                .expect("core ended");
            if let UiEvent::Approval(p) = &ev {
                self.cmd
                    .send(CoreCommand::ApprovalResponse {
                        id: p.id,
                        choice: ApprovalChoice::AllowOnce,
                    })
                    .unwrap();
            }
            let done = pred(&ev);
            self.seen.push(ev);
            if done {
                return;
            }
        }
    }

    async fn turn(&mut self, prompt: &str) {
        self.cmd
            .send(CoreCommand::UserPrompt(prompt.into()))
            .unwrap();
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
            .await;
    }

    async fn delete(&mut self, name: &str) -> String {
        self.cmd
            .send(CoreCommand::Slash(SlashCommand::Agent(
                AgentAction::Delete(name.into()),
            )))
            .unwrap();
        let mut text = String::new();
        self.until(|e| match e {
            UiEvent::Notice { text: t, .. } if t.contains("custom agent") => {
                text.clone_from(t);
                true
            }
            _ => false,
        })
        .await;
        text
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

fn notices(seen: &[UiEvent]) -> Vec<(NoticeLevel, String)> {
    seen.iter()
        .filter_map(|e| match e {
            UiEvent::Notice { level, text } => Some((*level, text.clone())),
            _ => None,
        })
        .collect()
}

fn create_args() -> Value {
    json!({
        "name": "reviewer",
        "description": "Reviews Rust code for panics and missing tests",
        "base": "research",
        "instructions": CHECKLIST,
        "tools": ["read_file"]
    })
}

fn review_step(agent: &str) -> Value {
    json!({ "goal": "review", "steps": [{ "title": "Review", "instructions": "review lib.rs", "role": "worker", "agent": agent }] })
}

#[tokio::test]
async fn orchestrator_creates_uses_and_user_deletes_a_custom_agent() {
    let orchestrator = |req: &ChatRequest| {
        let r = turn_results(req);
        let user = last_user(req);
        if user.contains("create") {
            return match r.len() {
                0 => Reply::Call("agent_create", create_args()),
                _ => Reply::Text("created"),
            };
        }
        if user.contains("again") {
            return match r.len() {
                0 => Reply::Call("plan_create", review_step("reviewer")),
                _ => Reply::Text("cannot"),
            };
        }
        match r.len() {
            0 => Reply::Call("plan_create", review_step("ghost")),
            1 => Reply::Call("plan_create", review_step("reviewer")),
            2 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            _ if r.iter().any(|x| x.contains("success")) => Reply::Text("reviewed"),
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 15 })),
        }
    };
    let provider = Scripted::new(orchestrator);
    let mut run = start(provider.clone(), Mode::Auto, |_| {});

    run.turn("please create a reviewer agent").await;
    let file = run
        .data
        .custom_agents_dir()
        .join("reviewer")
        .join("AGENT.md");
    assert!(file.exists(), "agent_create wrote the definition");
    let approvals: Vec<&UiEvent> = run
        .seen
        .iter()
        .filter(|e| matches!(e, UiEvent::Approval(_)))
        .collect();
    match approvals.as_slice() {
        [UiEvent::Approval(p)] => {
            assert_eq!(p.tool, "agent_create");
            assert!(p.inside && !p.allow_session && !p.default_deny, "{p:?}");
            assert!(
                p.reasons
                    .iter()
                    .any(|r| r.contains("always needs approval")),
                "{p:?}"
            );
            assert!(
                p.detail.as_deref().is_some_and(|d| d.contains(CHECKLIST)),
                "{p:?}"
            );
        }
        other => panic!("agent_create must ask even in auto mode: {other:?}"),
    }

    run.turn("review the code").await;
    let deleted = run.delete("reviewer").await;
    assert!(
        deleted.starts_with("deleted custom agent `reviewer`"),
        "{deleted}"
    );
    assert!(!file.exists());
    let missing = run.delete("reviewer").await;
    assert_eq!(missing, "no custom agent named `reviewer`");
    run.turn("review it again").await;
    let seen = run.quit().await;

    let notes = notices(&seen);
    assert!(
        notes.iter().any(|(l, t)| *l == NoticeLevel::Info
            && t.starts_with("created custom agent `reviewer` (research)")
            && t.ends_with("/agent delete reviewer")),
        "{notes:?}"
    );

    let orch = provider.requests(false);
    let review: Vec<&ChatRequest> = orch
        .iter()
        .filter(|r| last_user(r).contains("review the code"))
        .collect();
    assert!(
        review[0].messages[0]
            .content()
            .contains("- reviewer (research): Reviews Rust code for panics and missing tests"),
        "the catalog is in the orchestrator prompt"
    );
    let unknown = review.iter().find(|r| turn_results(r).len() == 1).unwrap();
    assert!(
        turn_results(unknown)[0].contains("unknown custom agent `ghost`; available: reviewer"),
        "{}",
        turn_results(unknown)[0]
    );

    let subs = provider.requests(true);
    let first = &subs[0];
    let system = first.messages[0].content();
    assert!(
        system.starts_with("You are a research sub-agent"),
        "the base role replaces the step's role"
    );
    assert!(system.contains("You are the custom agent \"reviewer\""));
    assert!(system.contains(CHECKLIST));
    let mut tools: Vec<&str> = first.tools.iter().map(|t| t.name.as_str()).collect();
    tools.sort_unstable();
    assert_eq!(tools, ["ask_orchestrator", "read_file", "report_result"]);
    assert!(
        review
            .iter()
            .flat_map(|r| turn_results(r))
            .any(|t| t.contains("task 1: running as reviewer1")),
        "labelled with the agent name"
    );

    let again: Vec<&ChatRequest> = orch
        .iter()
        .filter(|r| last_user(r).contains("again"))
        .collect();
    assert!(
        !again[0].messages[0]
            .content()
            .contains("- reviewer (research)")
    );
    let rejected = again.iter().find(|r| turn_results(r).len() == 1).unwrap();
    assert!(turn_results(rejected)[0].contains("unknown custom agent `reviewer`"));
}

#[tokio::test]
async fn deleted_agent_fails_the_attempt_and_plan_mode_cannot_create() {
    let orchestrator = |req: &ChatRequest| {
        let r = turn_results(req);
        let user = last_user(req);
        if user.contains("plan it") {
            return match r.len() {
                0 => Reply::Call("plan_create", review_step("checker")),
                _ => Reply::Text("planned"),
            };
        }
        if user.contains("create") {
            return match r.len() {
                0 => Reply::Call("agent_create", create_args()),
                _ => Reply::Text("could not"),
            };
        }
        match r.len() {
            0 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            _ if r.iter().any(|x| x.contains("no longer exists")) => Reply::Text("noted"),
            n if n > 6 => Reply::Text("gave up"),
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 10 })),
        }
    };
    let provider = Scripted::new(orchestrator);
    let mut run = start(provider.clone(), Mode::Plan, |data| {
        AgentLibrary::new(data)
            .create(&AgentDraft {
                name: "checker".into(),
                description: "Checks things".into(),
                base: Role::Research,
                tools: vec![],
                instructions: "Check carefully.".into(),
                created_by: "user".into(),
            })
            .unwrap();
    });

    run.turn("plan it").await;
    run.delete("checker").await;
    run.turn("run the review").await;
    run.turn("create a reviewer agent").await;
    assert!(
        !run.data.custom_agents_dir().join("reviewer").exists(),
        "plan mode denies agent_create"
    );
    run.quit().await;

    let orch = provider.requests(false);
    let dispatched = orch
        .iter()
        .filter(|r| last_user(r).contains("run the review"))
        .flat_map(turn_results)
        .any(|t| t.contains("custom agent `checker` no longer exists"));
    assert!(dispatched, "the attempt reports the deleted agent");
    let denied = orch
        .iter()
        .find(|r| last_user(r).contains("create") && turn_results(r).len() == 1)
        .unwrap();
    assert!(
        !turn_results(denied)[0].contains("created custom agent"),
        "{}",
        turn_results(denied)[0]
    );
}
