//! Phase 11 acceptance: the Learner writes skills, knowledgepedia facts and profile notes; knowledge reaches the Orchestrator.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::NaiveDate;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::Workspace;
use h_agent_core::config::Config;
use h_agent_core::events::{CoreCommand, SlashCommand, UiEvent};
use h_agent_core::secrets::contains_secret;
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, ModelInfo, ToolCall,
    cancellable,
};
use h_agent_memory::EventStore;
use h_agent_memory::knowledge::{Contribution, Knowledgepedia};
use h_agent_memory::profile;
use h_agent_memory::skills::SkillLibrary;
use h_agent_store::DataDir;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(20);
const KEY: &str = "sk-abcdefghijklmnopqrstuvwx1234";

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Triage,
    Memory,
    Learner,
    Orchestrator,
}

fn kind(req: &ChatRequest) -> Kind {
    let system = req.messages.first().map_or("", Message::content);
    if system.starts_with("You are the memory agent") {
        Kind::Memory
    } else if system.starts_with("You are the learner agent") {
        Kind::Learner
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
            Message::User { content } => Some(content.as_str()),
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

fn memory_reply(req: &ChatRequest) -> Value {
    let task = last_user(req);
    if task.starts_with("Task: file a new session") {
        json!({ "query_type": "creation", "topic": "coding/zig", "session_summary": "Zig hello world",
                "event_summary": "Wrote and built a Zig hello world after fixing build.zig" })
    } else if task.starts_with("Task: summarize a finished session") {
        json!({ "summary": "Built a Zig hello world", "outcome": "success" })
    } else if task.starts_with("Task: decide") {
        json!({ "action": "extend" })
    } else {
        json!({ "related": [] })
    }
}

fn lesson() -> Value {
    json!({
        "outcome": "success",
        "skill": {
            "name": "zig-build",
            "description": "Create and build a Zig project with build.zig",
            "allowed_tools": ["shell", "read_file", "edit_file"],
            "body": "1. Run `zig init`.\n2. Declare the executable in build.zig.\n3. Run `zig build run`."
        },
        "facts": [
            { "topic": "Zig build system", "aliases": ["build.zig", "zig build"],
              "fact": "zig build compiles the project described by build.zig",
              "sources": ["https://ziglang.org/learn/build-system/"], "confidence": "high" },
            { "topic": "Credentials", "fact": format!("the deploy token is {KEY}") }
        ],
        "profile": ["Prefers short answers", format!("Their OpenAI key is {KEY}")]
    })
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

struct Scripted {
    knowledge_topics: Vec<String>,
    orchestrator: Box<Policy>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl Scripted {
    fn new(
        knowledge_topics: &[&str],
        orchestrator: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            knowledge_topics: knowledge_topics.iter().map(|t| (*t).to_owned()).collect(),
            orchestrator: Box::new(orchestrator),
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
                json!({ "query_type": "creation", "topic": "coding/zig", "complexity": "simple",
                        "needs_tools": true, "knowledge_topics": self.knowledge_topics })
                .to_string(),
            ),
            Kind::Memory => Reply::Text(memory_reply(&req).to_string()),
            Kind::Learner => Reply::Text(lesson().to_string()),
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

    /// Quits; this waits for the Memory agent and the Learner.
    async fn quit(self) {
        self.cmd.send(CoreCommand::Quit).unwrap();
        tokio::time::timeout(WAIT, self.core)
            .await
            .unwrap()
            .unwrap();
    }
}

fn answer(_: &ChatRequest) -> Reply {
    Reply::Text("Built it.".into())
}

#[tokio::test]
async fn closed_event_yields_pending_skill_knowledge_and_safe_profile() {
    let env = env();
    let mut run = start(&env, Scripted::new(&[], answer), Config::default());
    run.turn("write a zig hello world and build it").await;
    run.quit().await;

    let main = EventStore::main(&env.data).load().unwrap();
    let event = main.events()[0].event.id.to_string();

    let catalog = SkillLibrary::new(&env.data).load();
    assert!(
        catalog.get("zig-build").is_none(),
        "learner skills wait for review"
    );
    let skill = catalog
        .pending
        .iter()
        .find(|s| s.name == "zig-build")
        .expect("pending skill");
    assert_eq!(skill.source_events, std::slice::from_ref(&event));
    assert_eq!(skill.created_by.as_deref(), Some("learner"));
    assert!(skill.body.contains("zig build run"));

    let kb = Knowledgepedia::new(&env.data);
    let index = kb.index().unwrap();
    assert_eq!(
        index.topics.len(),
        1,
        "the fact with a credential is dropped"
    );
    assert_eq!(index.topics[0].file, "zig-build-system.txt");
    let text = std::fs::read_to_string(env.data.knowledgepedia_dir().join("zig-build-system.txt"))
        .unwrap();
    assert!(
        text.starts_with("Topic: Zig build system\nAliases: build.zig, zig build\n"),
        "{text}"
    );
    assert!(text.contains("Sources: https://ziglang.org/learn/build-system/ ; session "));
    assert!(text.contains("Confidence: high"));
    assert!(text.contains("- zig build compiles the project described by build.zig"));

    let profile = std::fs::read_to_string(env.data.user_profile_file()).unwrap_or_default();
    assert!(
        !profile.contains("Prefers short answers"),
        "learned notes wait for review: {profile}"
    );
    let pending = profile::pending(&env.data);
    assert_eq!(
        pending.len(),
        1,
        "the note with a credential is dropped: {pending:?}"
    );
    assert!(
        pending[0].starts_with("Prefers short answers ("),
        "{pending:?}"
    );
    assert!(pending[0].contains(&format!("event {event})")));
    assert!(!pending[0].contains(KEY));

    // A later session points at the queue; /memory review keeps the note.
    let mut run = start(&env, Scripted::new(&[], answer), Config::default());
    run.until(|e| matches!(e, UiEvent::Notice { text, .. } if text.contains("1 learned note(s) about you await review")))
        .await;
    run.cmd
        .send(CoreCommand::Slash(SlashCommand::ProfileReview))
        .unwrap();
    let question = run.until(|e| matches!(e, UiEvent::Question(_))).await;
    let UiEvent::Question(q) = question else {
        unreachable!()
    };
    assert!(q.text.contains("Prefers short answers"), "{q:?}");
    assert_eq!(q.options, ["keep", "drop", "skip"]);
    run.cmd
        .send(CoreCommand::QuestionAnswer {
            id: q.id,
            answer: "keep".into(),
        })
        .unwrap();
    run.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("profile review:\nkept:")),
    )
    .await;
    run.quit().await;
    let profile = std::fs::read_to_string(env.data.user_profile_file()).unwrap();
    assert!(profile.contains("- Prefers short answers ("), "{profile}");
    assert!(profile::pending(&env.data).is_empty());
    for line in profile.lines().filter(|l| l.starts_with("- ")) {
        assert!(!contains_secret(line), "{line}");
    }
}

#[tokio::test]
async fn auto_approve_profile_writes_straight_to_the_profile() {
    let env = env();
    let mut config = Config::default();
    config.learner.auto_approve_profile = true;
    let mut run = start(&env, Scripted::new(&[], answer), config);
    run.turn("write a zig hello world and build it").await;
    run.quit().await;
    let profile = std::fs::read_to_string(env.data.user_profile_file()).unwrap();
    assert!(profile.contains("- Prefers short answers ("), "{profile}");
    assert!(!profile.contains(KEY));
    assert!(profile::pending(&env.data).is_empty());
}

#[tokio::test]
async fn auto_approve_writes_active_skills() {
    let env = env();
    let mut config = Config::default();
    config.learner.auto_approve_skills = true;
    let mut run = start(&env, Scripted::new(&[], answer), config);
    run.turn("write a zig hello world and build it").await;
    run.quit().await;
    let catalog = SkillLibrary::new(&env.data).load();
    assert!(catalog.pending.is_empty());
    assert_eq!(
        catalog.get("zig-build").unwrap().status.as_deref(),
        Some("active")
    );
}

#[tokio::test]
async fn knowledge_reaches_the_orchestrator() {
    let env = env();
    Knowledgepedia::new(&env.data)
        .merge(
            &Contribution {
                topic: "Zig build system".into(),
                aliases: vec!["build.zig".into()],
                facts: vec!["Use b.addExecutable in build.zig".into()],
                sources: vec!["https://ziglang.org/learn/build-system/".into()],
                confidence: "high".into(),
            },
            NaiveDate::from_ymd_opt(2026, 11, 5).unwrap(),
        )
        .unwrap();
    let orchestrator = |req: &ChatRequest| {
        if last_user(req).contains("look it up") && tool_results(req).is_empty() {
            Reply::Call("knowledge_lookup", json!({ "query": "build.zig" }))
        } else {
            Reply::Text("ok".into())
        }
    };
    let provider = Scripted::new(&["zig build system"], orchestrator);
    let mut run = start(&env, provider.clone(), Config::default());
    run.turn("how do I add an executable in zig?").await;
    run.turn("look it up").await;
    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Knowledge {
            topic: Some("zig build".into()),
        }))
        .unwrap();
    let notice = run
        .until(|e| matches!(e, UiEvent::Notice { text, .. } if text.contains("Topic: Zig build system")))
        .await;
    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Knowledge { topic: None }))
        .unwrap();
    run.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("knowledgepedia topics:")),
    )
    .await;
    run.quit().await;

    assert!(matches!(notice, UiEvent::Notice { text, .. } if text.contains("b.addExecutable")));
    let reqs = provider.of(Kind::Orchestrator);
    let first = &reqs[0];
    assert!(
        first
            .messages
            .iter()
            .any(|m| m.content().starts_with("Knowledgepedia entries")
                && m.content().contains("Use b.addExecutable")),
        "triage knowledge topics are put in context"
    );
    let looked_up = reqs.iter().find(|r| !tool_results(r).is_empty()).unwrap();
    assert!(
        tool_results(looked_up)[0].contains("Sources: https://ziglang.org/learn/build-system/")
    );
    assert!(reqs[0].tools.iter().any(|t| t.name == "knowledge_lookup"));
}
