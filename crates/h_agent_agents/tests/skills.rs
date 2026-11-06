//! Phase 9 acceptance: skills and dynamic tool loading.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::config::Config;
use h_agent_core::events::{ApprovalChoice, CoreCommand, NoticeLevel, SlashCommand, UiEvent};
use h_agent_core::{Mode, RiskClass, Workspace};
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    ToolCall, cancellable,
};
use h_agent_safety::Location;
use h_agent_store::DataDir;
use h_agent_tools::{ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(20);

/// A deferred test tool, standing in for web/browser/MCP tools.
struct Weather;

#[async_trait]
impl Tool for Weather {
    fn name(&self) -> &str {
        "weather_lookup"
    }
    fn description(&self) -> &str {
        "Look up the weather forecast for a city"
    }
    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": { "city": { "type": "string" } } })
    }
    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }
    fn tier(&self) -> Tier {
        Tier::Deferred
    }
    async fn preview(&self, _: &Value, _: &ToolCtx) -> Result<ActionPreview, ToolError> {
        Ok(ActionPreview::new("weather", Location::Inside))
    }
    async fn call(
        &self,
        _: Value,
        _: &ToolCtx,
        _: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::ok("sunny, 18 C"))
    }
}

enum Reply {
    Text(String),
    Call(&'static str, Value),
}

fn is_worker(req: &ChatRequest) -> bool {
    req.messages
        .first()
        .is_some_and(|m| m.content().starts_with("You are a worker sub-agent"))
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

fn tool_names(req: &ChatRequest) -> Vec<&str> {
    req.tools.iter().map(|t| t.name.as_str()).collect()
}

type Policy = dyn Fn(&ChatRequest) -> Reply + Send + Sync;

struct Scripted {
    hints: Vec<String>,
    orchestrator: Box<Policy>,
    worker: Box<Policy>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl Scripted {
    fn new(
        hints: &[&str],
        orchestrator: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
        worker: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            hints: hints.iter().map(|h| (*h).to_owned()).collect(),
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
                json!({ "query_type": "question", "topic": "weather", "complexity": "simple",
                        "needs_tools": true, "tool_hints": self.hints })
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

struct Run {
    _tmp: tempfile::TempDir,
    data: DataDir,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
}

fn data_dir() -> (tempfile::TempDir, PathBuf, DataDir) {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    (tmp, work, data)
}

fn start(
    (tmp, work, data): (tempfile::TempDir, PathBuf, DataDir),
    provider: Arc<dyn LlmProvider>,
    config: Config,
) -> Run {
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data.clone(),
        provider,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![Arc::new(Weather)],
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

    async fn quit(self) -> Vec<UiEvent> {
        self.cmd.send(CoreCommand::Quit).unwrap();
        tokio::time::timeout(WAIT, self.core)
            .await
            .unwrap()
            .unwrap();
        self.seen
    }
}

fn weather_policy(req: &ChatRequest) -> Reply {
    if !last_user(req).contains("weather") {
        return Reply::Text("You're welcome.".into());
    }
    match turn_results(req).len() {
        0 => Reply::Call("weather_lookup", json!({ "city": "Oslo" })),
        1 => Reply::Call("tool_search", json!({ "query": "weather forecast" })),
        2 => Reply::Call("weather_lookup", json!({ "city": "Oslo" })),
        _ => Reply::Text("Sunny.".into()),
    }
}

fn no_worker(_: &ChatRequest) -> Reply {
    Reply::Text("unused".into())
}

#[tokio::test]
async fn deferred_tool_loads_on_search_and_unloads_after_idle_turns() {
    let provider = Scripted::new(&[], weather_policy, no_worker);
    let mut config = Config::default();
    config.tools.offload_after_turns = 2;
    let mut run = start(data_dir(), provider.clone(), config);
    run.turn("what is the weather in Oslo").await;
    for i in 2..=4 {
        run.turn(&format!("thanks {i}")).await;
    }
    let seen = run.quit().await;

    let reqs = provider.orchestrator_requests();
    let has = |r: &ChatRequest| tool_names(r).contains(&"weather_lookup");
    let first: Vec<&ChatRequest> = reqs
        .iter()
        .filter(|r| last_user(r).contains("weather"))
        .collect();
    assert_eq!(first.len(), 4);
    assert!(!has(first[0]), "absent until loaded");
    assert!(tool_names(first[0]).contains(&"tool_search"));
    let loadable = first[0].messages[0]
        .content()
        .lines()
        .find(|l| l.starts_with("More tools can be loaded with tool_search: "))
        .unwrap()
        .to_owned();
    assert!(
        loadable.contains("weather_lookup") && loadable.contains("web_fetch"),
        "{loadable}"
    );
    let direct = &turn_results(first[1])[0];
    assert!(
        !direct.contains("ToolNotAllowed") && direct.contains("sunny"),
        "calling a loadable tool by name loads it: {direct}"
    );
    assert!(has(first[1]), "offered once loaded");
    assert!(turn_results(first[2])[1].contains("weather_lookup"));
    assert!(has(first[2]) && has(first[3]));
    assert!(turn_results(first[3])[2].contains("sunny"));

    let turn = |n: u32| {
        reqs.iter()
            .find(|r| last_user(r) == format!("thanks {n}"))
            .unwrap()
    };
    assert!(has(turn(2)) && has(turn(3)), "kept for 2 idle turns");
    assert!(!has(turn(4)), "unloaded after the idle turns");
    assert!(
        turn(4).messages[0].content().lines().any(|l| l
            .starts_with("More tools can be loaded with tool_search: ")
            && l.contains("weather_lookup")),
        "listed as loadable again"
    );

    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Usage(u) if u.tools_loaded == 1))
    );
}

#[tokio::test]
async fn triage_hints_preload_tools() {
    let provider = Scripted::new(&["weather"], |_| Reply::Text("ok".into()), no_worker);
    let mut run = start(data_dir(), provider.clone(), Config::default());
    run.turn("is it raining").await;
    run.quit().await;
    let req = &provider.orchestrator_requests()[0];
    assert!(tool_names(req).contains(&"weather_lookup"));
}

const CLAP: &str = "---\nname: rust-clap\ndescription: Add CLI flags with clap\nallowed-tools: Read, Bash(cargo:*)\n---\n\
                    Use clap derive; add the flag to the Args struct.\n";

fn write_skill(dir: PathBuf, text: &str) {
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("SKILL.md"), text).unwrap();
}

fn with_skills() -> (tempfile::TempDir, PathBuf, DataDir) {
    let (tmp, work, data) = data_dir();
    write_skill(data.skills_dir().join("rust-clap"), CLAP);
    write_skill(
        data.skills_dir().join("broken"),
        "---\nname: broken\n---\nno description\n",
    );
    write_skill(
        data.pending_skills_dir().join("zig-build"),
        "---\nname: zig-build\ndescription: Build zig projects\nstatus: pending\n---\nRun zig build.\n",
    );
    (tmp, work, data)
}

#[tokio::test]
async fn skills_catalog_slash_invocation_load_and_review() {
    let orchestrator = |req: &ChatRequest| {
        let user = last_user(req);
        if user.contains("use the skill tool") && turn_results(req).is_empty() {
            return Reply::Call("skill_load", json!({ "name": "rust-clap" }));
        }
        Reply::Text("done".into())
    };
    let provider = Scripted::new(&[], orchestrator, no_worker);
    let mut run = start(with_skills(), provider.clone(), Config::default());
    let warning = run
        .until(|e| matches!(e, UiEvent::Notice { level: NoticeLevel::Warning, text } if text.contains("broken")))
        .await;
    assert!(matches!(warning, UiEvent::Notice { text, .. } if text.contains("description")));
    let skills = run.until(|e| matches!(e, UiEvent::Skills(_))).await;
    match skills {
        UiEvent::Skills(list) => assert_eq!(
            list.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["rust-clap", "stubborn"]
        ),
        _ => unreachable!(),
    }

    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Skill {
            name: "rust-clap".into(),
            args: "add --verbose".into(),
        }))
        .unwrap();
    run.until(|e| matches!(e, UiEvent::UserMessage { text } if text == "/rust-clap add --verbose"))
        .await;
    run.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
        .await;
    run.turn("use the skill tool").await;

    run.cmd
        .send(CoreCommand::Slash(SlashCommand::Skills { review: true }))
        .unwrap();
    let q = run.until(|e| matches!(e, UiEvent::Question(_))).await;
    let UiEvent::Question(q) = q else {
        unreachable!()
    };
    assert!(q.text.contains("zig-build") && q.options.contains(&"approve".to_owned()));
    run.cmd
        .send(CoreCommand::QuestionAnswer {
            id: q.id,
            answer: "approve".into(),
        })
        .unwrap();
    let after = run
        .until(|e| matches!(e, UiEvent::Skills(list) if list.len() == 3))
        .await;
    assert!(matches!(after, UiEvent::Skills(list) if list.iter().any(|s| s.name == "zig-build")));
    let data = run.data.clone();
    assert!(
        data.skills_dir()
            .join("zig-build")
            .join("SKILL.md")
            .exists()
    );
    assert!(!data.pending_skills_dir().join("zig-build").exists());
    run.quit().await;

    let reqs = provider.orchestrator_requests();
    let slash = reqs
        .iter()
        .find(|r| last_user(r).contains("invoked the skill"))
        .unwrap();
    assert!(
        slash.messages[0]
            .content()
            .contains("- rust-clap: Add CLI flags with clap"),
        "catalog only"
    );
    assert!(
        !slash.messages[0].content().contains("Use clap derive"),
        "bodies load on demand"
    );
    let user = last_user(slash);
    assert!(user.contains("<skill name=\"rust-clap\">") && user.contains("Use clap derive"));
    assert!(user.ends_with("add --verbose"));
    let loaded = reqs.iter().find(|r| !turn_results(r).is_empty()).unwrap();
    assert!(turn_results(loaded)[0].contains("Use clap derive"));
}

#[tokio::test]
async fn task_skill_goes_to_the_worker_and_cannot_widen_tools() {
    let orchestrator = |req: &ChatRequest| {
        let r = turn_results(req);
        let steps = |skill: &str| json!({ "goal": "add flag", "steps": [{ "title": "Flag", "instructions": "add --verbose", "role": "worker", "skills": [skill] }] });
        match r.len() {
            0 => Reply::Call("plan_create", steps("no-such-skill")),
            1 => Reply::Call("plan_create", steps("rust-clap")),
            2 => Reply::Call("tasks_dispatch", json!({ "tasks": [1] })),
            _ if r.iter().any(|x| x.contains("task 1 (Flag) success")) => {
                Reply::Text("done".into())
            }
            _ => Reply::Call("tasks_await", json!({ "timeout_secs": 15 })),
        }
    };
    let worker = |req: &ChatRequest| {
        if turn_results(req).is_empty() {
            Reply::Call(
                "report_result",
                json!({ "outcome": "success", "summary": "added", "evidence": [] }),
            )
        } else {
            Reply::Text("reported".into())
        }
    };
    let provider = Scripted::new(&[], orchestrator, worker);
    let mut config = Config::default();
    config.general.default_mode = Mode::Plan;
    let mut run = start(with_skills(), provider.clone(), config);
    run.turn("add a verbose flag").await;
    run.quit().await;

    let reqs = provider.orchestrator_requests();
    let unknown = reqs.iter().find(|r| turn_results(r).len() == 1).unwrap();
    assert!(
        turn_results(unknown)[0].contains("unknown skill `no-such-skill`; available: rust-clap")
    );
    let worker = &provider.worker_requests()[0];
    assert!(
        worker.messages[0]
            .content()
            .contains("<skill name=\"rust-clap\">\nUse clap derive")
    );
    let mut tools = tool_names(worker);
    tools.sort_unstable();
    assert_eq!(
        tools,
        ["ask_orchestrator", "read_file", "report_result"],
        "the skill narrows to read_file + shell, and plan mode removes shell"
    );
}
