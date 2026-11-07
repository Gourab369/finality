//! Phase 8 acceptance: the Memory agent files exchanges into events, links them, summarizes sessions and answers recalls.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_agents::{Core, CoreDeps};
use finality_core::config::Config;
use finality_core::events::{ApprovalChoice, CoreCommand, NoticeLevel, SlashCommand, UiEvent};
use finality_core::{Mode, Role, SessionId, Workspace};
use finality_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, ModelInfo, ToolCall,
    cancellable,
};
use finality_memory::{EventStatus, EventStore, EventsFile};
use finality_store::{DataDir, SessionIndex};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(20);

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Triage,
    Memory,
    Worker,
    Orchestrator,
}

fn kind(req: &ChatRequest) -> Kind {
    let system = req.messages.first().map_or("", Message::content);
    if system.starts_with("You are the memory agent") {
        Kind::Memory
    } else if system.starts_with("You are a worker sub-agent") {
        Kind::Worker
    } else if req.response_format.is_some() {
        Kind::Triage
    } else {
        Kind::Orchestrator
    }
}

fn last_user(req: &ChatRequest) -> &str {
    req.messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .unwrap_or("")
}

fn tool_results(req: &ChatRequest) -> Vec<&str> {
    req.messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

/// The memory agent's side: answers by task, keyed on words in the transcript.
fn memory_reply(req: &ChatRequest) -> Reply {
    let task = last_user(req);
    let text = if task.starts_with("Task: file a new session") {
        let lower = task.to_lowercase();
        if lower.contains("tunnelling") {
            json!({ "query_type": "question", "topic": "Physics", "session_summary": "Quantum tunnelling", "event_summary": "Explained quantum tunnelling" })
        } else if lower.contains("crc") {
            json!({ "query_type": "command", "topic": "coding/rust", "session_summary": "CRC work", "event_summary": "CRC mismatch on reopen" })
        } else {
            json!({ "query_type": "command", "topic": "coding/rust", "session_summary": "Bug fixing", "event_summary": "Fixing the parser bug" })
        }
    } else if task.starts_with("Task: decide") {
        let new_part = task
            .split("New exchanges:")
            .nth(1)
            .unwrap_or("")
            .to_lowercase();
        if new_part.contains("deploy") && !task.contains("Open event: Deployment") {
            json!({ "action": "close_and_new", "summary": "Deployment questions", "outcome": "bug fixed" })
        } else {
            json!({ "action": "extend" })
        }
    } else if task.starts_with("Task: find earlier events") {
        let ids: Vec<&str> = task
            .split("Candidates:\n")
            .nth(1)
            .unwrap_or("")
            .lines()
            .take_while(|l| !l.trim().is_empty())
            .filter_map(|l| l.split(" · ").next())
            .collect();
        json!({ "related": ids })
    } else if task.starts_with("Task: summarize a finished session") {
        json!({ "summary": "Session summary from memory", "outcome": "done" })
    } else if task.starts_with("Task: pick the topics") {
        json!({ "topics": [1] })
    } else {
        json!({ "events": [1] })
    };
    Reply::Text(text.to_string())
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

    fn of(&self, k: Kind) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| kind(r) == k)
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
        let reply = match kind(&req) {
            Kind::Triage => Reply::Text(
                json!({ "query_type": "question", "topic": "physics", "complexity": "simple", "needs_tools": true })
                    .to_string(),
            ),
            Kind::Memory => memory_reply(&req),
            Kind::Worker => (self.worker)(&req),
            Kind::Orchestrator => (self.orchestrator)(&req),
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

fn answer_text(_: &ChatRequest) -> Reply {
    Reply::Text("Here is the answer.".into())
}

fn no_worker(_: &ChatRequest) -> Reply {
    Reply::Text("unused".into())
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

struct Run {
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
}

fn start(env: &Env, provider: Arc<dyn LlmProvider>, config: Config) -> Run {
    // Scripted providers expect a triage request for every prompt.
    let mut config = config;
    config.llm.quick_triage = false;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&env.work).unwrap(),
        data_dir: env.data.clone(),
        provider,
        memory_provider: None,
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: None,
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    Run {
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

    async fn turn(&mut self, prompt: &str) {
        self.cmd
            .send(CoreCommand::UserPrompt(prompt.into()))
            .unwrap();
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
            .await;
    }

    fn session(&self) -> SessionId {
        self.seen
            .iter()
            .rev()
            .find_map(|e| match e {
                UiEvent::SessionStarted { id } => Some(*id),
                _ => None,
            })
            .unwrap()
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

/// Polls an events file until `check` passes.
async fn eventually(path: &EventStore, mut check: impl FnMut(&EventsFile) -> bool) -> EventsFile {
    let deadline = tokio::time::Instant::now() + WAIT;
    loop {
        let file = path.load().unwrap();
        if check(&file) {
            return file;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "memory never reached the expected state"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn scratch_files(data: &DataDir) -> usize {
    std::fs::read_dir(data.memory_tmp_dir())
        .map(|d| {
            d.flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("memjob_"))
                .count()
        })
        .unwrap_or(0)
}

fn main_events(data: &DataDir) -> EventStore {
    EventStore::main(data)
}

#[tokio::test]
async fn first_exchange_then_every_seventh_then_session_end() {
    let env = env();
    let provider = Scripted::new(answer_text, no_worker);
    let mut config = Config::default();
    // Small window: every request is compacted to the last 4 exchanges.
    config.provider.ollama.num_ctx = 2000;
    let mut run = start(&env, provider.clone(), config);
    let store = main_events(&env.data);

    run.turn("explain quantum tunnelling").await;
    let id = run.session();
    let file = eventually(&store, |f| !f.refs(&id).is_empty()).await;
    let refs = file.refs(&id);
    assert_eq!(
        (refs[0].0, refs[0].1),
        ("question", "physics"),
        "names are normalized"
    );
    assert_eq!(refs[0].2.events.len(), 1);
    assert_eq!(refs[0].2.events[0].summary, "Explained quantum tunnelling");
    assert_eq!(refs[0].2.events[0].exchanges, Some((0, 0)));

    for i in 2..=7 {
        run.turn(&format!("more about tunnelling, part {i}")).await;
    }
    let file = eventually(&store, |f| {
        f.refs(&id).first().is_some_and(|(_, _, r)| {
            r.events
                .first()
                .is_some_and(|e| e.exchanges == Some((0, 6)))
        })
    })
    .await;
    assert_eq!(
        file.refs(&id)[0].2.events.len(),
        1,
        "same subject extends the open event"
    );

    run.turn("and in semiconductors?").await;
    let last = provider.of(Kind::Orchestrator).pop().unwrap();
    let earlier = last.messages[1].content();
    assert!(
        earlier.starts_with("[Earlier in this conversation]"),
        "{earlier}"
    );
    assert!(earlier.contains("- Explained quantum tunnelling"));

    let memory_calls = provider.of(Kind::Memory).len();
    assert!(memory_calls >= 2, "classify + decide");
    run.quit().await;

    let file = store.load().unwrap();
    let r = file.refs(&id)[0].2;
    assert!(r.finalized);
    assert_eq!(r.summary, "Session summary from memory");
    assert!(r.events.iter().all(|e| e.status == EventStatus::Closed));
    assert_eq!(
        r.events.last().unwrap().exchanges.map(|x| x.1),
        Some(7),
        "session end files the rest"
    );
    let index = SessionIndex::new(env.data.session_index_file(), env.data.locks_dir());
    assert_eq!(
        index.get(&id).unwrap().unwrap().title,
        "Session summary from memory"
    );
    assert_eq!(scratch_files(&env.data), 0);
}

#[tokio::test]
async fn subject_change_closes_the_event_and_opens_a_new_one() {
    let env = env();
    let provider = Scripted::new(answer_text, no_worker);
    let mut config = Config::default();
    config.memory.user_exchange_trigger = 2;
    let mut run = start(&env, provider, config);
    for prompt in [
        "fix the parser bug",
        "still failing",
        "how do I deploy it",
        "deploy to which server",
    ] {
        run.turn(prompt).await;
    }
    let id = run.session();
    let file = eventually(&main_events(&env.data), |f| {
        f.refs(&id)
            .iter()
            .map(|(_, _, r)| r.events.len())
            .sum::<usize>()
            == 2
    })
    .await;
    let events: Vec<_> = file
        .refs(&id)
        .iter()
        .flat_map(|(_, _, r)| r.events.clone())
        .collect();
    assert_eq!(events[0].summary, "Fixing the parser bug");
    assert_eq!(events[0].status, EventStatus::Closed);
    assert_eq!(events[0].outcome.as_deref(), Some("bug fixed"));
    assert_eq!(events[0].exchanges, Some((0, 1)));
    assert_eq!(events[1].summary, "Deployment questions");
    assert_eq!(events[1].status, EventStatus::Open);
    assert_eq!(events[1].exchanges, Some((2, 3)));
    run.quit().await;
}

#[tokio::test]
async fn recall_links_and_expand_across_sessions() {
    let env = env();
    for prompt in [
        "the CRC mismatch on reopen is back",
        "CRC mismatch again after truncation",
    ] {
        let mut run = start(
            &env,
            Scripted::new(answer_text, no_worker),
            Config::default(),
        );
        run.turn(prompt).await;
        run.quit().await;
    }
    let file = main_events(&env.data).load().unwrap();
    let events = file.events();
    assert_eq!(events.len(), 2);
    let (a, b) = (events[0].event, events[1].event);
    assert_eq!(a.related, vec![b.id], "similar events link both ways");
    assert_eq!(b.related, vec![a.id]);
    let first_session = events[0].session.session_id;

    // Third session: the digest is in the prompt, and the Orchestrator recalls and expands.
    let orchestrator = |req: &ChatRequest| {
        let results = tool_results(req);
        match results.len() {
            0 => Reply::Call("memory_recall", json!({ "query": "crc mismatch" })),
            1 => {
                let id = results[0]
                    .split('[')
                    .nth(1)
                    .and_then(|s| s.split(']').next())
                    .unwrap_or("");
                Reply::Call("memory_expand", json!({ "event_id": id }))
            }
            _ => Reply::Text("Found it.".into()),
        }
    };
    let provider = Scripted::new(orchestrator, no_worker);
    let mut run = start(&env, provider.clone(), Config::default());
    run.turn("do you remember the crc issue?").await;
    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Memory {
            recall: Some("crc".into()),
        }))
        .unwrap();
    let notice = run
        .until(|e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("memories for")))
        .await;
    run.quit().await;

    let req = provider.of(Kind::Orchestrator).pop().unwrap();
    let system = req.messages[0].content();
    assert!(
        system.contains("Memory digest") && system.contains("coding/rust (2)"),
        "{system}"
    );
    let results = tool_results(&req);
    assert!(
        results[0].contains(&first_session.to_string()),
        "{}",
        results[0]
    );
    assert!(
        results[1].contains("user: the CRC mismatch on reopen is back"),
        "{}",
        results[1]
    );
    match notice {
        UiEvent::Notice { level, text } => {
            assert_eq!(level, NoticeLevel::Info);
            assert!(text.contains(&first_session.to_string()));
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn scratch_file_is_removed_when_a_job_fails() {
    let env = env();
    let store = main_events(&env.data);
    std::fs::write(store.path(), "{ not json").unwrap();
    let mut run = start(
        &env,
        Scripted::new(answer_text, no_worker),
        Config::default(),
    );
    run.turn("hello").await;
    run.quit().await;
    assert_eq!(
        std::fs::read_to_string(store.path()).unwrap(),
        "{ not json",
        "nothing was written"
    );
    assert_eq!(scratch_files(&env.data), 0);
}

#[tokio::test]
async fn sub_agent_thread_goes_to_the_role_file() {
    let env = env();
    let orchestrator = |req: &ChatRequest| {
        let r = tool_results(req).join("\n");
        if r.is_empty() {
            return Reply::Call(
                "plan_create",
                json!({ "goal": "inspect", "steps": [{ "title": "Read", "instructions": "read Cargo.toml twice", "role": "worker" }] }),
            );
        }
        if !r.contains("running as") {
            return Reply::Call("tasks_dispatch", json!({ "tasks": [1] }));
        }
        if r.contains("task 1 (Read) success") {
            return Reply::Text("done".into());
        }
        Reply::Call("tasks_await", json!({ "timeout_secs": 15 }))
    };
    let worker = |req: &ChatRequest| match tool_results(req).len() {
        0 | 1 => Reply::Call("read_file", json!({ "path": "Cargo.toml" })),
        2 => Reply::Call(
            "report_result",
            json!({ "outcome": "success", "summary": "read it", "evidence": ["Cargo.toml"] }),
        ),
        _ => Reply::Text("reported".into()),
    };
    let mut config = Config::default();
    config.general.default_mode = Mode::Auto;
    config.memory.subagent_exchange_trigger = 2;
    let mut run = start(&env, Scripted::new(orchestrator, worker), config);
    run.turn("inspect the manifest").await;
    let id = run.session();
    run.quit().await;

    let role = EventStore::for_role(&env.data, Role::Worker)
        .load()
        .unwrap();
    let refs = role.refs(&id);
    assert_eq!(refs.len(), 1);
    let events = &refs[0].2.events;
    assert!(!events.is_empty());
    assert!(
        events
            .iter()
            .all(|e| e.thread.is_some() && e.status == EventStatus::Closed)
    );
    assert!(events.iter().all(|e| e.exchanges.is_none()));
    let main = main_events(&env.data).load().unwrap();
    assert!(
        main.events().iter().all(|v| v.event.thread.is_none()),
        "sub-agent steps stay out of main.events"
    );
    assert!(
        !EventStore::for_role(&env.data, Role::Research)
            .path()
            .exists()
    );
}
