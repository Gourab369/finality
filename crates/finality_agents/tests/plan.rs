//! Phase 7 acceptance: plans, sub-agents, retries and the sub-agent pool.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_agents::{Core, CoreDeps};
use finality_core::Role;
use finality_core::Workspace;
use finality_core::config::Config;
use finality_core::events::{ApprovalChoice, CoreCommand, UiEvent};
use finality_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, ModelInfo, ToolCall,
    cancellable,
};
use finality_store::{DataDir, Record, read_log};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(20);

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

/// Answers each request from its content, so concurrent agents get deterministic replies.
struct Scripted {
    orchestrator: Box<Policy>,
    worker: Box<Policy>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl Scripted {
    fn new(
        orchestrator: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
        worker: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            orchestrator: Box::new(orchestrator),
            worker: Box::new(worker),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn worker_requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| is_worker(r))
            .cloned()
            .collect()
    }
}

fn is_worker(req: &ChatRequest) -> bool {
    req.messages
        .first()
        .is_some_and(|m| m.content().starts_with("You are a worker sub-agent"))
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
            Reply::Text(
                json!({ "query_type": "task", "topic": "coding/rust", "complexity": "multi_step", "needs_tools": true })
                    .to_string(),
            )
        } else if is_worker(&req) {
            (self.worker)(&req)
        } else {
            (self.orchestrator)(&req)
        };
        let n = req.messages.len();
        let chunk = match reply {
            Reply::Text(t) => ChatChunk::Text(t),
            Reply::Call(name, arguments) => ChatChunk::ToolCall(ToolCall {
                id: format!("call_{n}_{name}"),
                name: name.to_owned(),
                arguments,
            }),
        };
        let chunks = vec![
            Ok(chunk),
            Ok(ChatChunk::Done {
                usage: Default::default(),
                stop_reason: Default::default(),
            }),
        ];
        Ok(cancellable(Box::pin(futures::stream::iter(chunks)), cancel))
    }
}

/// Names of the tools the orchestrator has called so far, with their arguments.
fn calls(req: &ChatRequest) -> Vec<(String, Value)> {
    req.messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { tool_calls, .. } => Some(tool_calls),
            _ => None,
        })
        .flatten()
        .map(|c| (c.name.clone(), c.arguments.clone()))
        .collect()
}

fn results(req: &ChatRequest) -> String {
    req.messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn count(calls: &[(String, Value)], name: &str) -> usize {
    calls.iter().filter(|(n, _)| n == name).count()
}

fn step(title: &str, depends_on: &[u32]) -> Value {
    json!({ "title": title, "instructions": format!("do {title}"), "role": "worker", "depends_on": depends_on })
}

/// (task, attempt) from the assignment message.
fn assignment(req: &ChatRequest) -> (u32, u32) {
    let text = req.messages.get(1).map(Message::content).unwrap_or("");
    let num = |key: &str| -> u32 {
        text.split(key)
            .nth(1)
            .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|d| d.parse().ok())
            .unwrap_or(0)
    };
    (num("Task "), num("Attempt "))
}

fn report(outcome: &str, summary: &str) -> Reply {
    Reply::Call(
        "report_result",
        json!({ "outcome": outcome, "summary": summary, "evidence": ["src/lib.rs"] }),
    )
}

fn setup(
    provider: Arc<dyn LlmProvider>,
    config: Config,
) -> (
    tempfile::TempDir,
    DataDir,
    mpsc::UnboundedReceiver<UiEvent>,
    mpsc::UnboundedSender<CoreCommand>,
    tokio::task::JoinHandle<()>,
) {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    // Scripted providers expect a triage request for every prompt.
    let mut config = config;
    config.llm.quick_triage = false;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data.clone(),
        provider,
        memory_provider: Some(Arc::new(finality_llm::MockProvider::new())),
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: None,
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    (tmp, data, ui, cmd, core)
}

/// Runs one prompt to completion, answering approvals, and returns every UI event.
async fn run_prompt(
    ui: &mut mpsc::UnboundedReceiver<UiEvent>,
    cmd: &mpsc::UnboundedSender<CoreCommand>,
    prompt: &str,
) -> Vec<UiEvent> {
    cmd.send(CoreCommand::UserPrompt(prompt.into())).unwrap();
    let mut seen = Vec::new();
    loop {
        let ev = tokio::time::timeout(WAIT, ui.recv())
            .await
            .expect("timed out")
            .expect("core ended");
        if let UiEvent::Approval(p) = &ev {
            cmd.send(CoreCommand::ApprovalResponse {
                id: p.id,
                choice: ApprovalChoice::AllowOnce,
            })
            .unwrap();
        }
        let done = matches!(ev, UiEvent::TurnFinished { .. });
        seen.push(ev);
        if done {
            return seen;
        }
    }
}

#[tokio::test]
async fn plan_with_retries_reaches_success() {
    let orchestrator = |req: &ChatRequest| {
        let c = calls(req);
        let r = results(req);
        let retries = count(&c, "task_retry");
        if count(&c, "plan_create") == 0 {
            return Reply::Call(
                "plan_create",
                json!({ "goal": "ship it", "steps": [step("First", &[]), step("Second", &[]), step("Third", &[1, 2])] }),
            );
        }
        if count(&c, "tasks_dispatch") == 0 {
            return Reply::Call(
                "tasks_dispatch",
                json!({ "tasks": [1, 2], "strategy": "direct edit" }),
            );
        }
        if r.contains("task 2 (Second) failed on attempt 1/") && retries == 0 {
            // Same strategy in different case: must be rejected.
            return Reply::Call(
                "task_retry",
                json!({ "task": 2, "strategy": "Direct  EDIT" }),
            );
        }
        if retries == 1 {
            return Reply::Call(
                "task_retry",
                json!({ "task": 2, "strategy": "rewrite the module" }),
            );
        }
        if r.contains("task 2 (Second) failed on attempt 2/") && retries == 2 {
            return Reply::Call(
                "task_retry",
                json!({ "task": 2, "strategy": "add a compatibility shim" }),
            );
        }
        let done = |n: &str| r.contains(&format!("task {n} success on attempt"));
        if done("1 (First)") && done("2 (Second)") && count(&c, "tasks_dispatch") == 1 {
            return Reply::Call("tasks_dispatch", json!({ "tasks": [3] }));
        }
        if done("3 (Third)") {
            return Reply::Text("All three tasks are done.".into());
        }
        Reply::Call("tasks_await", json!({ "timeout_secs": 15 }))
    };
    let worker = |req: &ChatRequest| {
        if calls(req).iter().any(|(n, _)| n == "report_result") {
            return Reply::Text("reported".into());
        }
        match assignment(req) {
            (2, 1) if calls(req).is_empty() => {
                Reply::Call("plan_create", json!({ "goal": "x", "steps": [] }))
            }
            (2, 1) => report("failed", "compile error"),
            (2, 2) => report("failed", "tests still fail"),
            _ => report("success", "done"),
        }
    };
    let provider = Scripted::new(orchestrator, worker);
    let (_tmp, data, mut ui, cmd, core) = setup(provider.clone(), Config::default());
    let seen = run_prompt(&mut ui, &cmd, "ship it").await;
    let id = seen
        .iter()
        .find_map(|e| match e {
            UiEvent::SessionStarted { id } => Some(*id),
            _ => None,
        })
        .unwrap();
    cmd.send(CoreCommand::Quit).unwrap();
    tokio::time::timeout(WAIT, core).await.unwrap().unwrap();

    assert!(seen.iter().any(
        |e| matches!(e, UiEvent::AssistantDelta { text, .. } if text == "All three tasks are done.")
    ));
    // The final plan shows every task succeeded, task 2 on its third attempt.
    let plan = seen
        .iter()
        .rev()
        .find_map(|e| match e {
            UiEvent::PlanUpdated(Some(p)) => Some(p.clone()),
            _ => None,
        })
        .unwrap();
    let states: Vec<(String, u32)> = plan
        .tasks
        .iter()
        .map(|t| (t.status.to_string(), t.attempt))
        .collect();
    assert_eq!(
        states,
        [
            ("success".into(), 1),
            ("success".into(), 3),
            ("success".into(), 1)
        ]
    );

    // Duplicate strategy rejected; the orchestrator saw why.
    let last_orchestrator = provider
        .requests
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|r| !is_worker(r) && r.response_format.is_none())
        .cloned()
        .unwrap();
    assert!(results(&last_orchestrator).contains("strategy already tried in attempt 1"));

    // The worker that called an orchestrator tool got ToolNotAllowed, and was never offered it.
    let workers = provider.worker_requests();
    assert!(
        workers
            .iter()
            .any(|r| results(r).contains("ToolNotAllowed"))
    );
    assert!(workers.iter().all(|r| {
        let names: Vec<&str> = r.tools.iter().map(|t| t.name.as_str()).collect();
        names.contains(&"report_result") && !names.contains(&"plan_create")
    }));
    // Workers see earlier attempts in their assignment.
    assert!(workers.iter().any(|r| {
        let (task, attempt) = assignment(r);
        task == 2
            && attempt == 3
            && r.messages[1]
                .content()
                .contains("attempt 2 (rewrite the module)")
    }));

    // The log has the plan, spawns and messages; sub-agent records do not reach resume.
    let records = read_log(&data.sessions_dir().join(id.log_file_name()))
        .unwrap()
        .records;
    let spawns = records
        .iter()
        .filter(|r| {
            matches!(
                r.body,
                Record::AgentSpawn {
                    role: Role::Worker,
                    ..
                }
            )
        })
        .count();
    assert_eq!(spawns, 5, "tasks 1 and 3 once, task 2 three times");
    assert!(
        records
            .iter()
            .any(|r| matches!(r.body, Record::PlanUpdate { .. }))
    );
    assert!(
        records
            .iter()
            .any(|r| matches!(r.body, Record::AgentMessage { .. }))
    );
}

#[tokio::test]
async fn pool_limit_queues_extra_tasks() {
    let orchestrator = |req: &ChatRequest| {
        let c = calls(req);
        let r = results(req);
        if count(&c, "plan_create") == 0 {
            return Reply::Call(
                "plan_create",
                json!({ "goal": "two things", "steps": [step("A", &[]), step("B", &[])] }),
            );
        }
        if count(&c, "tasks_dispatch") == 0 {
            return Reply::Call("tasks_dispatch", json!({ "tasks": [1, 2] }));
        }
        if r.contains("task 1 (A) success") && r.contains("task 2 (B) success") {
            return Reply::Text("both done".into());
        }
        Reply::Call("tasks_await", json!({ "timeout_secs": 15 }))
    };
    let worker = |req: &ChatRequest| {
        if calls(req).iter().any(|(n, _)| n == "report_result") {
            Reply::Text("reported".into())
        } else {
            report("success", "ok")
        }
    };
    let provider = Scripted::new(orchestrator, worker);
    let mut config = Config::default();
    config.agents.max_subagents = 1;
    let (_tmp, _data, mut ui, cmd, core) = setup(provider.clone(), config);
    let seen = run_prompt(&mut ui, &cmd, "do two things").await;
    cmd.send(CoreCommand::Quit).unwrap();
    tokio::time::timeout(WAIT, core).await.unwrap().unwrap();

    let dispatch = provider
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|r| !is_worker(r))
        .map(results)
        .find(|r| r.contains("task 2: queued"));
    assert!(dispatch.is_some(), "task 2 waits for a free slot");
    let most_workers = seen
        .iter()
        .filter_map(|e| match e {
            UiEvent::AgentStatus(lines) => {
                Some(lines.iter().filter(|l| l.role == Role::Worker).count())
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);
    assert!(most_workers <= 1);
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::AssistantDelta { text, .. } if text == "both done"))
    );
}

#[tokio::test]
async fn interjection_reaches_the_running_turn() {
    // The orchestrator waits on a task whose worker blocks until cancelled, so the turn stays open.
    let orchestrator = |req: &ChatRequest| {
        let c = calls(req);
        let has_interjection = req
            .messages
            .iter()
            .any(|m| matches!(m, Message::User { content, .. } if content.starts_with("[User interjection] stop")));
        if has_interjection {
            return Reply::Text("Stopping as asked.".into());
        }
        if count(&c, "plan_create") == 0 {
            return Reply::Call(
                "plan_create",
                json!({ "goal": "slow", "steps": [step("Slow", &[])] }),
            );
        }
        if count(&c, "tasks_dispatch") == 0 {
            return Reply::Call("tasks_dispatch", json!({ "tasks": [1] }));
        }
        Reply::Call("tasks_await", json!({ "timeout_secs": 1 }))
    };
    let worker =
        |_: &ChatRequest| Reply::Call("ask_orchestrator", json!({ "question": "which file?" }));
    let provider = Scripted::new(orchestrator, worker);
    let (_tmp, _data, mut ui, cmd, core) = setup(provider, Config::default());
    cmd.send(CoreCommand::UserPrompt("go slow".into())).unwrap();
    loop {
        let ev = tokio::time::timeout(WAIT, ui.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(ev, UiEvent::PlanUpdated(Some(_))) {
            break;
        }
    }
    cmd.send(CoreCommand::UserPrompt("stop".into())).unwrap();
    let mut answered = false;
    loop {
        let ev = tokio::time::timeout(WAIT, ui.recv())
            .await
            .unwrap()
            .unwrap();
        if matches!(&ev, UiEvent::AssistantDelta { text, .. } if text == "Stopping as asked.") {
            answered = true;
        }
        if matches!(ev, UiEvent::TurnFinished { .. }) {
            break;
        }
    }
    assert!(answered);
    cmd.send(CoreCommand::Quit).unwrap();
    tokio::time::timeout(WAIT, core).await.unwrap().unwrap();
}
