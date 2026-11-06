//! Phase 10 acceptance: finished plans are scored into the lookup table, and proven templates are reused.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use h_agent_agents::scorer::PlanFile;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::config::Config;
use h_agent_core::events::{ApprovalChoice, CoreCommand, UiEvent};
use h_agent_core::{Mode, Workspace};
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    ToolCall, cancellable,
};
use h_agent_memory::lookup::{LookupEntry, LookupStep, LookupStore, PlanRun, RunOutcome};
use h_agent_store::DataDir;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(20);

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

fn is_worker(req: &ChatRequest) -> bool {
    req.messages
        .first()
        .is_some_and(|m| m.content().contains("sub-agent of Finality"))
}

/// Tool results after the last user message.
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

fn called(req: &ChatRequest, name: &str) -> usize {
    req.messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { tool_calls, .. } => Some(tool_calls),
            _ => None,
        })
        .flatten()
        .filter(|c| c.name == name)
        .count()
}

/// (task, attempt) from a worker's assignment.
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

fn report(outcome: &str) -> Reply {
    Reply::Call(
        "report_result",
        json!({ "outcome": outcome, "summary": outcome, "evidence": [] }),
    )
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

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

    fn orchestrator_requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.response_format.is_none() && !is_worker(r))
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
            Reply::Text(
                json!({ "query_type": "command", "topic": "coding/rust", "complexity": "multi_step", "needs_tools": true })
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

struct Env {
    _tmp: tempfile::TempDir,
    work: PathBuf,
    data: DataDir,
}

fn env() -> Env {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    Env {
        _tmp: tmp,
        work,
        data,
    }
}

/// Runs one prompt in a fresh core on `env`, then quits (which waits for the Scorer).
async fn session(env: &Env, provider: Arc<dyn LlmProvider>, prompt: &str) {
    let mut config = Config::default();
    config.general.default_mode = Mode::Auto;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&env.work).unwrap(),
        data_dir: env.data.clone(),
        provider,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: None,
    };
    let (ui_tx, mut ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    cmd.send(CoreCommand::UserPrompt(prompt.into())).unwrap();
    loop {
        let ev = tokio::time::timeout(WAIT, ui.recv())
            .await
            .unwrap()
            .unwrap();
        if let UiEvent::Approval(p) = &ev {
            cmd.send(CoreCommand::ApprovalResponse {
                id: p.id,
                choice: ApprovalChoice::AllowOnce,
            })
            .unwrap();
        }
        if matches!(ev, UiEvent::TurnFinished { .. }) {
            break;
        }
    }
    cmd.send(CoreCommand::Quit).unwrap();
    tokio::time::timeout(WAIT, core).await.unwrap().unwrap();
}

fn plan_files(data: &DataDir) -> Vec<PlanFile> {
    let mut out = Vec::new();
    for dir in std::fs::read_dir(data.plans_dir()).unwrap().flatten() {
        for f in std::fs::read_dir(dir.path()).unwrap().flatten() {
            out.push(serde_json::from_str(&std::fs::read_to_string(f.path()).unwrap()).unwrap());
        }
    }
    out
}

const GOAL: &str = "Add a verbose flag to the CLI";

/// One worker step; `flaky` makes the first attempt fail so the Orchestrator retries.
fn one_step_orchestrator(req: &ChatRequest) -> Reply {
    let r = turn_results(req).join("\n");
    if called(req, "plan_create") == 0 {
        return Reply::Call(
            "plan_create",
            json!({ "goal": GOAL, "steps": [{ "title": "Add flag", "instructions": "add --verbose", "role": "worker" }] }),
        );
    }
    if called(req, "tasks_dispatch") == 0 {
        return Reply::Call("tasks_dispatch", json!({ "tasks": [1] }));
    }
    if r.contains("task 1 (Add flag) failed on attempt 1/") && called(req, "task_retry") == 0 {
        return Reply::Call(
            "task_retry",
            json!({ "task": 1, "strategy": "edit main.rs directly" }),
        );
    }
    if r.contains("task 1 (Add flag) success") {
        return Reply::Text("done".into());
    }
    Reply::Call("tasks_await", json!({ "timeout_secs": 15 }))
}

fn reading_worker(flaky: bool) -> impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static {
    move |req: &ChatRequest| {
        let (_, attempt) = assignment(req);
        match turn_results(req).len() {
            _ if flaky && attempt == 1 && turn_results(req).is_empty() => report("failed"),
            0 => Reply::Call("read_file", json!({ "path": "Cargo.toml" })),
            1 if !(flaky && attempt == 1) => report("success"),
            _ => Reply::Text("reported".into()),
        }
    }
}

#[tokio::test]
async fn same_signature_and_strategy_aggregate_into_one_entry() {
    let env = env();
    session(
        &env,
        Scripted::new(one_step_orchestrator, reading_worker(false)),
        "add a verbose flag",
    )
    .await;
    session(
        &env,
        Scripted::new(one_step_orchestrator, reading_worker(true)),
        "add a verbose flag",
    )
    .await;

    let files = plan_files(&env.data);
    assert_eq!(files.len(), 2);
    assert!(files.iter().all(|f| f.scored && f.outcome == "success"));
    assert_eq!(files.iter().map(|f| f.retries).sum::<u32>(), 1);
    let plan = &files[0].plan;
    assert_eq!(
        (plan.query_type.as_str(), plan.topic.as_str()),
        ("command", "coding/rust")
    );

    let table = LookupStore::new(&env.data).load().unwrap();
    assert_eq!(table.entries.len(), 1);
    let e = &table.entries[0];
    assert_eq!(e.event_signature, GOAL);
    assert_eq!((e.stats.runs, e.stats.success), (2, 2));
    assert!((e.stats.avg_retries - 0.5).abs() < 1e-9);
    assert_eq!(
        e.steps[0].tools,
        ["read_file"],
        "tools of the successful attempt"
    );
    let expected = 0.60 + 0.25 + 0.15 * (1.0 - 0.5 / 7.0);
    assert!((e.score - expected).abs() < 1e-9, "score {}", e.score);
    assert_eq!(e.source_plans.len(), 2);

    // A different strategy for the same request is kept as its own entry.
    let two_steps = |req: &ChatRequest| {
        let r = turn_results(req).join("\n");
        if called(req, "plan_create") == 0 {
            return Reply::Call(
                "plan_create",
                json!({ "goal": GOAL, "steps": [
                    { "title": "Find args", "instructions": "find", "role": "research" },
                    { "title": "Add flag", "instructions": "add", "role": "worker", "depends_on": [1] }] }),
            );
        }
        match called(req, "tasks_dispatch") {
            0 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            1 if r.contains("task 1 (Find args) success") => {
                Reply::Call("tasks_dispatch", json!({ "tasks": [2] }))
            }
            _ if r.contains("task 2 (Add flag) success") => Reply::Text("done".into()),
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 15 })),
        }
    };
    session(
        &env,
        Scripted::new(two_steps, reading_worker(false)),
        "add a verbose flag",
    )
    .await;
    let table = LookupStore::new(&env.data).load().unwrap();
    assert_eq!(table.entries.len(), 2);
    assert!(table.entries.iter().all(|e| e.event_signature == GOAL));
}

fn seed(env: &Env) -> (String, String) {
    let step = |title: &str, role: &str, tool: &str, deps: &[u32]| LookupStep {
        title: title.into(),
        role: role.into(),
        tools: vec![tool.into()],
        skills: vec![],
        depends_on: deps.to_vec(),
    };
    let cfg = Config::default();
    LookupStore::new(&env.data)
        .update(|t| {
            let now = Utc::now();
            let mut good = LookupEntry::new(
                "Add a CLI flag to a Rust clap app",
                "command",
                "coding/rust",
                vec![
                    step("Locate args", "research", "search_files", &[]),
                    step("Add flag", "worker", "read_file", &[1]),
                ],
                now,
            );
            let mut weak = LookupEntry::new(
                "Add a CLI flag with argh",
                "command",
                "coding/rust",
                vec![step("Patch", "worker", "edit_file", &[])],
                now,
            );
            for i in 0..3 {
                good.record(
                    &PlanRun {
                        plan_id: format!("g{i}"),
                        outcome: RunOutcome::Success,
                        duration_s: 60,
                        retries: 0,
                    },
                    now,
                );
            }
            for i in 0..2 {
                weak.record(
                    &PlanRun {
                        plan_id: format!("w{i}"),
                        outcome: RunOutcome::Failed,
                        duration_s: 300,
                        retries: 3,
                    },
                    now,
                );
            }
            t.entries = vec![good, weak];
            t.rescore(&cfg.scorer.weights, cfg.retry.max_retries);
            (t.entries[0].id.clone(), t.entries[1].id.clone())
        })
        .unwrap()
}

#[tokio::test]
async fn proven_template_is_instantiated_and_credited() {
    let env = env();
    let (good, weak) = seed(&env);
    let (g, w) = (good.clone(), weak.clone());
    let orchestrator = move |req: &ChatRequest| {
        let r = turn_results(req).join("\n");
        match turn_results(req).len() {
            0 => {
                return Reply::Call(
                    "lookup_table_query",
                    json!({ "description": "add a cli flag" }),
                );
            }
            1 => {
                return Reply::Call(
                    "plan_create",
                    json!({ "goal": "Add --verbose", "template_id": w }),
                );
            }
            2 => {
                return Reply::Call(
                    "plan_create",
                    json!({ "goal": "Add --verbose", "template_id": g }),
                );
            }
            _ => {}
        }
        match called(req, "tasks_dispatch") {
            0 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            1 if r.contains("task 1 (Locate args) success") => {
                Reply::Call("tasks_dispatch", json!({ "tasks": [2] }))
            }
            _ if r.contains("task 2 (Add flag) success") => Reply::Text("done".into()),
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 15 })),
        }
    };
    let worker = |req: &ChatRequest| {
        let (task, _) = assignment(req);
        match (task, turn_results(req).len()) {
            (1, 0) => Reply::Call("search_files", json!({ "pattern": "name" })),
            (2, 0) => Reply::Call("read_file", json!({ "path": "Cargo.toml" })),
            (_, 1) => report("success"),
            _ => Reply::Text("reported".into()),
        }
    };
    let provider = Scripted::new(orchestrator, worker);
    session(&env, provider.clone(), "add a CLI flag to the clap app").await;

    let first = &provider.orchestrator_requests()[0];
    let note = first
        .messages
        .iter()
        .find(|m| m.content().starts_with("A proven plan template"))
        .expect("template suggested")
        .content()
        .to_owned();
    assert!(note.contains(&good) && !note.contains(&weak), "{note}");

    let last = provider.orchestrator_requests().pop().unwrap();
    let results = turn_results(&last);
    assert!(results[0].contains(&good) && results[0].contains("reusable"));
    assert!(results[0].contains("not proven enough to reuse"));
    assert!(
        results[1].contains("not proven enough to reuse")
            && results[1].contains("plan from scratch")
    );
    assert!(
        results[2].contains("#1 [pending] Locate args (research)"),
        "{}",
        results[2]
    );
    assert!(results[2].contains("#2 [pending] Add flag (worker after #1)"));

    let files = plan_files(&env.data);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].plan.template_id.as_deref(), Some(good.as_str()));
    let table = LookupStore::new(&env.data).load().unwrap();
    assert_eq!(table.entries.len(), 2, "credited, not duplicated");
    let credited = table.entry(&good).unwrap();
    assert_eq!(credited.stats.runs, 4);
    assert_eq!(table.entry(&weak).unwrap().stats.runs, 2);
}
