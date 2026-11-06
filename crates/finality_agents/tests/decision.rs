//! Phase 14 acceptance: the decision model at the Orchestrator's decision points.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use finality_agents::{Core, CoreDeps};
use finality_core::config::Config;
use finality_core::events::{ApprovalChoice, CoreCommand, NoticeLevel, SlashCommand, UiEvent};
use finality_core::{SessionId, Workspace};
use finality_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    ToolCall, cancellable,
};
use finality_memory::EventStore;
use finality_memory::knowledge::{Contribution, Knowledgepedia};
use finality_memory::lookup::{LookupEntry, LookupStep, LookupStore, PlanRun, RunOutcome};
use finality_memory::tree::RefInfo;
use finality_store::DataDir;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::path;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const WAIT: Duration = Duration::from_secs(20);

/// Answers of the fake decision model, adjustable between turns.
#[derive(Clone)]
struct Answers {
    complexity: f64,
    needs_tools: f64,
    query_type: (&'static str, f64),
    topic: (&'static str, f64),
    retry: f64,
    template: f64,
}

impl Default for Answers {
    fn default() -> Self {
        Self {
            complexity: 0.03,
            needs_tools: 0.98,
            query_type: ("command", 0.95),
            topic: ("coding/rust", 0.9),
            retry: 0.5,
            template: 0.5,
        }
    }
}

struct DecisionMock(Arc<Mutex<Answers>>);

impl Respond for DecisionMock {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let a = self.0.lock().unwrap().clone();
        let body: Value = serde_json::from_slice(&req.body).unwrap();
        let noul = |p: f64| json!({ "type": "noul", "noul": p });
        let choice = |(c, conf): (&str, f64)| json!({ "type": "choice", "choice": c, "probabilities": { c: conf }, "confidence": conf });
        let mut answers = Map::new();
        for key in body["questions"].as_object().unwrap().keys() {
            let answer = match key.as_str() {
                "probe" => noul(0.99),
                "complexity" => noul(a.complexity),
                "needs_tools" => noul(a.needs_tools),
                "query_type" => choice(a.query_type),
                "topic" => choice(a.topic),
                "retry" => noul(a.retry),
                "template" => noul(a.template),
                _ => continue,
            };
            answers.insert(key.clone(), answer);
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "model": "d1-omni",
            "answers": answers,
            "usage": { "input_tokens": 175, "output_tokens": 0 }
        }))
    }
}

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

/// Replies by request content. LLM triage always says "simple".
struct Scripted {
    orchestrator: Box<Policy>,
    worker: Box<Policy>,
    requests: Mutex<Vec<ChatRequest>>,
}

fn is_worker(req: &ChatRequest) -> bool {
    req.messages
        .first()
        .is_some_and(|m| m.content().starts_with("You are a worker sub-agent"))
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

    fn triage_requests(&self) -> usize {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.response_format.is_some())
            .count()
    }

    /// The first orchestrator request of each turn.
    fn turn_starts(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.response_format.is_none() && !is_worker(r) && turn_results(r).is_empty())
            .cloned()
            .collect()
    }

    fn last_orchestrator(&self) -> ChatRequest {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|r| r.response_format.is_none() && !is_worker(r))
            .cloned()
            .unwrap()
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
                json!({ "query_type": "question", "topic": "general", "complexity": "simple", "needs_tools": true })
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

/// Tool results after the last user message (this turn's).
fn turn_results(req: &ChatRequest) -> Vec<String> {
    let start = req
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }))
        .unwrap_or(0);
    req.messages[start..]
        .iter()
        .filter_map(|m| match m {
            Message::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// A system note added in this turn (after the last user message) starting with `prefix`.
fn has_system(req: &ChatRequest, prefix: &str) -> Option<String> {
    let start = req
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }))
        .unwrap_or(0);
    req.messages[start..]
        .iter()
        .find(|m| matches!(m, Message::System { .. }) && m.content().starts_with(prefix))
        .map(|m| m.content().to_owned())
}

fn answer_text(_: &ChatRequest) -> Reply {
    Reply::Text("ok".into())
}

struct Run {
    _tmp: tempfile::TempDir,
    server: MockServer,
    answers: Arc<Mutex<Answers>>,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
}

/// A Core whose Ollama base URL is a wiremock server answering the decision endpoint.
/// `probe_failures` makes that many first requests fail with 404.
async fn start(
    provider: Arc<dyn LlmProvider>,
    enabled: bool,
    probe_failures: u64,
    seed: impl FnOnce(&DataDir),
) -> Run {
    let server = MockServer::start().await;
    let answers = Arc::new(Mutex::new(Answers::default()));
    if probe_failures > 0 {
        Mock::given(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(404).set_body_string("404 page not found"))
            .up_to_n_times(probe_failures)
            .with_priority(1)
            .mount(&server)
            .await;
    }
    Mock::given(path("/v1/systemone"))
        .respond_with(DecisionMock(answers.clone()))
        .mount(&server)
        .await;
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    seed(&data);
    let mut config = Config::default();
    config.provider.ollama.base_url = server.uri();
    config.decision.enabled = enabled;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data,
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
        server,
        answers,
        ui,
        cmd,
        core,
        seen: Vec::new(),
    }
}

impl Run {
    async fn until(&mut self, mut pred: impl FnMut(&UiEvent) -> bool) -> UiEvent {
        loop {
            let ev = tokio::time::timeout(WAIT, self.ui.recv())
                .await
                .expect("timed out")
                .expect("core ended");
            if let UiEvent::Approval(p) = &ev {
                self.cmd
                    .send(CoreCommand::ApprovalResponse {
                        id: p.id,
                        choice: ApprovalChoice::AllowOnce,
                    })
                    .unwrap();
            }
            self.seen.push(ev.clone());
            if pred(&ev) {
                return ev;
            }
        }
    }

    async fn decide(&mut self, on: bool) -> bool {
        self.cmd
            .send(CoreCommand::Slash(SlashCommand::Decide(on)))
            .unwrap();
        match self
            .until(|e| matches!(e, UiEvent::DecideChanged { .. }))
            .await
        {
            UiEvent::DecideChanged { on } => on,
            _ => unreachable!(),
        }
    }

    async fn turn(&mut self, prompt: &str) {
        self.cmd
            .send(CoreCommand::UserPrompt(prompt.into()))
            .unwrap();
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
            .await;
    }

    /// Request bodies sent to the decision endpoint whose questions include `key`.
    async fn decision_requests(&self, key: &str) -> Vec<Value> {
        self.server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter_map(|r| serde_json::from_slice::<Value>(&r.body).ok())
            .filter(|b| b["questions"].get(key).is_some())
            .collect()
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

/// A past session under command / coding/rust, so the topic question has a known topic.
fn seed_topic(data: &DataDir) {
    let started = NaiveDate::from_ymd_opt(2026, 11, 1)
        .unwrap()
        .and_hms_opt(9, 0, 0)
        .unwrap();
    let id = SessionId::new(1, started);
    let info = RefInfo {
        session_id: id,
        log_file: id.log_file_name(),
        project_id: "other".into(),
        date: started.date(),
    };
    EventStore::main(data)
        .update(|f| {
            f.ensure_ref("command", "coding/rust", &info);
        })
        .unwrap();
}

#[tokio::test]
async fn confident_answers_replace_llm_triage() {
    let orchestrator = |req: &ChatRequest| match turn_results(req).len() {
        0 => Reply::Call(
            "edit_file",
            json!({ "path": "notes.txt", "operations": [{ "op": "write", "content": "x\n" }] }),
        ),
        _ => Reply::Text("done".into()),
    };
    let provider = Scripted::new(orchestrator, answer_text);
    let seed = |data: &DataDir| {
        seed_topic(data);
        Knowledgepedia::new(data)
            .merge(
                &Contribution {
                    topic: "Zig build system".into(),
                    aliases: vec!["build.zig".into()],
                    facts: vec!["zig build runs build.zig".into()],
                    sources: vec!["https://ziglang.org/learn/build-system/".into()],
                    confidence: "high".into(),
                },
                NaiveDate::from_ymd_opt(2026, 11, 2).unwrap(),
            )
            .unwrap();
    };
    let mut run = start(provider.clone(), true, 0, seed).await;
    assert!(
        run.until(|e| matches!(e, UiEvent::DecideChanged { .. }))
            .await
            == UiEvent::DecideChanged { on: true }
    );
    run.answers.lock().unwrap().complexity = 0.97;

    run.turn("write a note about the release of build.zig")
        .await;
    assert_eq!(provider.triage_requests(), 0, "no LLM triage");
    let first = &provider.turn_starts()[0];
    assert!(has_system(first, "Triage marked this request as multi-step").is_some());
    assert!(first.tools.iter().any(|t| t.name == "edit_file"));
    let knowledge = has_system(first, "Knowledgepedia entries for this request").unwrap();
    assert!(
        knowledge.contains("zig build runs build.zig"),
        "{knowledge}"
    );

    let bodies = run.decision_requests("complexity").await;
    assert_eq!(bodies.len(), 1);
    let q = &bodies[0]["questions"];
    assert_eq!(q["complexity"]["type"], "noul");
    assert_eq!(q["needs_tools"]["type"], "noul");
    assert_eq!(
        q["query_type"]["criteria"]["command"],
        "Do something: run, change, fix or build"
    );
    assert_eq!(q["query_type"]["criteria"]["other"], Value::Null);
    assert_eq!(q["topic"]["criteria"]["coding/rust"], "coding/rust");
    assert_eq!(q["topic"]["criteria"]["other"], Value::Null);
    assert!(
        bodies[0]["state"]
            .as_str()
            .unwrap()
            .ends_with("Latest request:\nwrite a note about the release of build.zig")
    );
    assert_eq!(bodies[0]["model"], "d1-omni");

    let seen = run.quit().await;
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Approval(p) if p.tool == "edit_file")),
        "approvals are unaffected by the decision model"
    );
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Usage(u) if u.session_in >= 175)),
        "decision usage counts toward the session"
    );
}

#[tokio::test]
async fn unsure_answers_fall_back_to_the_llm() {
    let provider = Scripted::new(answer_text, answer_text);
    let mut run = start(provider.clone(), true, 0, seed_topic).await;
    run.until(|e| matches!(e, UiEvent::DecideChanged { on: true }))
        .await;
    run.answers.lock().unwrap().complexity = 0.55;

    run.turn("refactor the parser").await;
    assert_eq!(provider.triage_requests(), 1, "complexity went to the LLM");
    assert!(
        has_system(&provider.turn_starts()[0], "Triage marked").is_none(),
        "the LLM's `simple` wins over the unsure answer"
    );

    // `other` also goes to the LLM.
    {
        let mut a = run.answers.lock().unwrap();
        a.complexity = 0.97;
        a.topic = ("other", 0.99);
    }
    run.turn("tune the database").await;
    assert_eq!(provider.triage_requests(), 2);
    assert!(
        has_system(
            &provider.turn_starts()[1],
            "Triage marked this request as multi-step"
        )
        .is_some(),
        "confident answers still override the LLM's"
    );
    run.quit().await;
}

#[tokio::test]
async fn failed_probe_keeps_it_off_and_slash_toggles() {
    let provider = Scripted::new(answer_text, answer_text);
    let mut run = start(provider.clone(), false, 1, seed_topic).await;

    assert!(!run.decide(true).await);
    let error = run
        .until(|e| {
            matches!(
                e,
                UiEvent::Notice {
                    level: NoticeLevel::Error,
                    ..
                }
            )
        })
        .await;
    let UiEvent::Notice { text, .. } = error else {
        unreachable!()
    };
    assert!(
        text.contains("HTTP 404") && text.ends_with("the decision model stays off"),
        "{text}"
    );
    run.turn("first").await;
    assert_eq!(provider.triage_requests(), 1);
    assert_eq!(
        run.server.received_requests().await.unwrap().len(),
        1,
        "only the probe"
    );

    assert!(run.decide(true).await);
    run.turn("second").await;
    assert_eq!(provider.triage_requests(), 1, "decided without the LLM");
    assert_eq!(run.decision_requests("complexity").await.len(), 1);

    assert!(!run.decide(false).await);
    run.turn("third").await;
    assert_eq!(provider.triage_requests(), 2);
    assert_eq!(run.decision_requests("complexity").await.len(), 1);
    run.quit().await;
}

#[tokio::test]
async fn retry_advice_reaches_the_orchestrator() {
    let orchestrator = |req: &ChatRequest| {
        let r = turn_results(req);
        match r.len() {
            0 => Reply::Call(
                "plan_create",
                json!({ "goal": "fix", "steps": [{ "title": "Fix build", "instructions": "make the build pass", "role": "worker" }] }),
            ),
            1 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            _ if r.iter().any(|x| x.contains("task 1 (Fix build) failed")) => {
                Reply::Text("giving up".into())
            }
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 15 })),
        }
    };
    let worker = |req: &ChatRequest| {
        if turn_results(req).is_empty() {
            Reply::Call(
                "report_result",
                json!({ "outcome": "failed", "summary": "compile error in parser.rs", "evidence": [] }),
            )
        } else {
            Reply::Text("reported".into())
        }
    };
    let provider = Scripted::new(orchestrator, worker);
    let mut run = start(provider.clone(), true, 0, seed_topic).await;
    run.until(|e| matches!(e, UiEvent::DecideChanged { on: true }))
        .await;
    run.answers.lock().unwrap().retry = 0.08;

    run.turn("fix the build").await;
    let results = turn_results(&provider.last_orchestrator()).join("\n");
    assert!(
        results.contains("task 1 (Fix build) failed on attempt 1/")
            && results.contains("the decision model judged another strategy unlikely to succeed"),
        "{results}"
    );
    let asked = run.decision_requests("retry").await;
    assert_eq!(asked.len(), 1);
    let state = asked[0]["state"].as_str().unwrap();
    assert!(
        state.starts_with("Task:\nFix build\nmake the build pass"),
        "{state}"
    );
    assert!(
        state.contains("1. strategy: ") && state.contains("Failed: compile error in parser.rs"),
        "{state}"
    );
    run.quit().await;
}

fn seed_template(data: &DataDir) {
    seed_topic(data);
    let cfg = Config::default();
    LookupStore::new(data)
        .update(|t| {
            let now = Utc::now();
            let mut e = LookupEntry::new(
                "Add a CLI flag to a Rust clap app",
                "command",
                "coding/rust",
                vec![LookupStep {
                    title: "Add flag".into(),
                    role: "worker".into(),
                    tools: vec!["edit_file".into()],
                    skills: vec![],
                    depends_on: vec![],
                }],
                now,
            );
            for i in 0..3 {
                e.record(
                    &PlanRun {
                        plan_id: format!("p{i}"),
                        outcome: RunOutcome::Success,
                        duration_s: 60,
                        retries: 0,
                    },
                    now,
                );
            }
            t.entries = vec![e];
            t.rescore(&cfg.scorer.weights, cfg.retry.max_retries);
        })
        .unwrap();
}

#[tokio::test]
async fn template_fit_filters_the_suggestion() {
    let provider = Scripted::new(answer_text, answer_text);
    let mut run = start(provider.clone(), true, 0, seed_template).await;
    run.until(|e| matches!(e, UiEvent::DecideChanged { on: true }))
        .await;
    for p in [0.05, 0.95, 0.6] {
        {
            let mut a = run.answers.lock().unwrap();
            a.complexity = 0.97;
            a.template = p;
        }
        run.turn("add a CLI flag to the clap app").await;
    }
    let notes: Vec<Option<String>> = provider
        .turn_starts()
        .iter()
        .map(|r| has_system(r, "A proven plan template"))
        .collect();
    assert!(notes[0].is_none(), "confidently rejected");
    assert!(
        notes[1]
            .as_deref()
            .unwrap()
            .ends_with("The decision model judged that it fits.")
    );
    let unsure = notes[2].as_deref().unwrap();
    assert!(
        !unsure.contains("decision model"),
        "unsure: the LLM judges the fit"
    );
    assert_eq!(run.decision_requests("template").await.len(), 3);
    assert_eq!(provider.triage_requests(), 0);
    run.quit().await;
}
