#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::config::Config;
use h_agent_core::events::{
    ApprovalChoice, CoreCommand, NoticeLevel, PickerKind, SlashCommand, ToolStatus, UiEvent,
};
use h_agent_core::{Mode, SessionId, Workspace};
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    cancellable,
};
use h_agent_store::{DataDir, Record, SessionStart, SessionStore, read_log};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const WAIT: Duration = Duration::from_secs(10);

fn triage(needs_tools: bool) -> String {
    json!({ "query_type": "question", "topic": "coding/rust", "complexity": "simple", "needs_tools": needs_tools })
        .to_string()
}

struct Harness {
    _tmp: tempfile::TempDir,
    work: PathBuf,
    data: DataDir,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: JoinHandle<()>,
    seen: Vec<UiEvent>,
}

fn dirs() -> (tempfile::TempDir, PathBuf, DataDir) {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(
        work.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n",
    )
    .unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    (tmp, work, data)
}

fn start(
    provider: Arc<dyn LlmProvider>,
    mode: Mode,
    existing: Option<(tempfile::TempDir, PathBuf, DataDir)>,
) -> Harness {
    start_with(provider, mode, existing, false)
}

fn start_with(
    provider: Arc<dyn LlmProvider>,
    mode: Mode,
    existing: Option<(tempfile::TempDir, PathBuf, DataDir)>,
    choose_model: bool,
) -> Harness {
    let (tmp, work, data) = existing.unwrap_or_else(dirs);
    let mut config = Config::default();
    config.general.default_mode = mode;
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data.clone(),
        provider,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![],
        env_vars: std::env::vars().collect(),
        api_keys: None,
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let mut core = Core::new(deps, ui_tx).unwrap();
    if choose_model {
        core.choose_model_at_start();
    }
    let core = tokio::spawn(core.run(cmd_rx));
    Harness {
        _tmp: tmp,
        work,
        data,
        ui,
        cmd,
        core,
        seen: Vec::new(),
    }
}

impl Harness {
    fn send(&self, c: CoreCommand) {
        self.cmd.send(c).unwrap();
    }

    /// Receives events until one matches; approvals are answered with `answer` when given.
    async fn until(
        &mut self,
        mut pred: impl FnMut(&UiEvent) -> bool,
        answer: Option<ApprovalChoice>,
    ) -> UiEvent {
        loop {
            let ev = tokio::time::timeout(WAIT, self.ui.recv())
                .await
                .expect("timed out")
                .expect("core ended");
            self.seen.push(ev.clone());
            if let (UiEvent::Approval(p), Some(choice)) = (&ev, &answer) {
                self.send(CoreCommand::ApprovalResponse {
                    id: p.id,
                    choice: choice.clone(),
                });
            }
            if pred(&ev) {
                return ev;
            }
        }
    }

    async fn turn(&mut self, prompt: &str, answer: Option<ApprovalChoice>) {
        self.send(CoreCommand::UserPrompt(prompt.into()));
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }), answer)
            .await;
    }

    fn session_id(&self) -> SessionId {
        self.seen
            .iter()
            .rev()
            .find_map(|e| match e {
                UiEvent::SessionStarted { id } => Some(*id),
                _ => None,
            })
            .unwrap()
    }

    async fn quit(self) -> (tempfile::TempDir, PathBuf, DataDir, Vec<UiEvent>) {
        self.send(CoreCommand::Quit);
        tokio::time::timeout(WAIT, self.core)
            .await
            .unwrap()
            .unwrap();
        (self._tmp, self.work, self.data, self.seen)
    }
}

fn kinds(data: &DataDir, id: &SessionId) -> Vec<&'static str> {
    let path = data.sessions_dir().join(id.log_file_name());
    read_log(&path)
        .unwrap()
        .records
        .iter()
        .map(|r| r.body.kind())
        .collect()
}

#[tokio::test]
async fn read_file_turn_with_approval_is_logged_end_to_end() {
    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(true))
            .then_tool_call("read_file", json!({ "path": "Cargo.toml" }))
            .then_text("It is a Rust workspace."),
    );
    let mut h = start(mock.clone(), Mode::Cowork, None);
    h.turn("read Cargo.toml", Some(ApprovalChoice::AllowOnce))
        .await;
    let id = h.session_id();
    let (_tmp, _work, data, seen) = h.quit().await;

    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Approval(p) if p.tool == "read_file" && p.inside))
    );
    assert!(seen.iter().any(|e| matches!(
        e,
        UiEvent::ToolFinished {
            status: ToolStatus::Succeeded,
            ..
        }
    )));
    assert!(seen.iter().any(
        |e| matches!(e, UiEvent::AssistantDelta { text, .. } if text == "It is a Rust workspace.")
    ));
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Usage(u) if u.ctx_total == 32_768 && u.session_in > 0))
    );

    assert_eq!(
        kinds(&data, &id),
        [
            "SessionStart",
            "UserMessage",
            "Usage",
            "AssistantMessage",
            "ToolCall",
            "Approval",
            "ToolResult",
            "Usage",
            "AssistantMessage",
            "SessionEnd"
        ]
    );

    let reqs = mock.requests();
    assert_eq!(reqs.len(), 3);
    assert!(
        reqs[0].response_format.is_some() && reqs[0].tools.is_empty(),
        "triage request"
    );
    assert!(reqs[1].tools.iter().any(|t| t.name == "read_file"));
    let tool_msg = reqs[2].messages.iter().find_map(|m| match m {
        Message::Tool { content, .. } => Some(content.clone()),
        _ => None,
    });
    let tool_msg = tool_msg.unwrap();
    assert!(
        tool_msg.starts_with("<untrusted source=\"tool:read_file\">")
            && tool_msg.contains("[workspace]")
    );
}

#[tokio::test]
async fn tools_are_sent_even_when_triage_says_none_are_needed() {
    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("Ownership moves values."),
    );
    let mut h = start(mock.clone(), Mode::Cowork, None);
    h.turn("what is ownership?", None).await;
    h.quit().await;
    assert!(
        mock.requests()[1]
            .tools
            .iter()
            .any(|t| t.name == "read_file")
    );
}

#[tokio::test]
async fn denied_tool_call_is_reported_to_the_model() {
    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(true))
            .then_tool_call("shell", json!({ "command": "echo hi" }))
            .then_text("Understood, I won't run it."),
    );
    let mut h = start(mock.clone(), Mode::Cowork, None);
    h.turn(
        "run echo",
        Some(ApprovalChoice::Deny {
            note: Some("not now".into()),
        }),
    )
    .await;
    let (.., seen) = h.quit().await;
    assert!(seen.iter().any(|e| matches!(
        e,
        UiEvent::ToolFinished {
            status: ToolStatus::Denied,
            ..
        }
    )));
    let last = mock.requests().pop().unwrap();
    assert!(
        last.messages
            .iter()
            .any(|m| matches!(m, Message::Tool { content, .. } if content.contains("not now")))
    );
}

#[tokio::test]
async fn auto_mode_reads_without_asking_and_mode_can_change() {
    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(true))
            .then_tool_call("read_file", json!({ "path": "Cargo.toml" }))
            .then_text("done"),
    );
    let mut h = start(mock, Mode::Cowork, None);
    h.send(CoreCommand::Slash(SlashCommand::Mode(Mode::Auto)));
    h.until(
        |e| matches!(e, UiEvent::ModeChanged { mode: Mode::Auto }),
        None,
    )
    .await;
    h.turn("read Cargo.toml", None).await;
    let id = h.session_id();
    let (_tmp, _w, data, seen) = h.quit().await;
    assert!(!seen.iter().any(|e| matches!(e, UiEvent::Approval(_))));
    assert!(
        kinds(&data, &id).contains(&"Approval"),
        "auto-allowed decisions are still logged"
    );
}

/// Answers triage, then streams one chunk and never finishes.
struct Hanging;

#[async_trait]
impl LlmProvider for Hanging {
    fn id(&self) -> &'static str {
        "hanging"
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn model_info(&self, _: &str) -> Result<ModelInfo, LlmError> {
        Ok(ModelInfo {
            context_length: Some(8192),
            ..ModelInfo::default()
        })
    }
    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let stream: ChatStream = if req.response_format.is_some() {
            Box::pin(futures::stream::iter(vec![
                Ok(ChatChunk::Text(triage(false))),
                Ok(ChatChunk::Done {
                    usage: Default::default(),
                    stop_reason: Default::default(),
                }),
            ]))
        } else {
            Box::pin(
                futures::stream::iter(vec![Ok(ChatChunk::Text("partial".into()))])
                    .chain(futures::stream::pending()),
            )
        };
        Ok(cancellable(stream, cancel))
    }
}

use futures::StreamExt as _;

#[tokio::test]
async fn cancel_stops_a_streaming_turn() {
    let mut h = start(Arc::new(Hanging), Mode::Cowork, None);
    h.send(CoreCommand::UserPrompt("hello".into()));
    h.until(|e| matches!(e, UiEvent::AssistantDelta { .. }), None)
        .await;
    h.send(CoreCommand::CancelTurn);
    let ev = h
        .until(|e| matches!(e, UiEvent::TurnFinished { .. }), None)
        .await;
    assert_eq!(ev, UiEvent::TurnFinished { cancelled: true });
    let id = h.session_id();
    let (_tmp, _w, data, _) = h.quit().await;
    let records = read_log(&data.sessions_dir().join(id.log_file_name()))
        .unwrap()
        .records;
    assert!(
        records
            .iter()
            .any(|r| matches!(&r.body, Record::AssistantMessage { text, .. } if text == "partial"))
    );
    assert_eq!(records.last().unwrap().body.kind(), "SessionEnd");
}

#[tokio::test]
async fn resume_replays_and_continues_the_conversation() {
    let first = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("Answer one.")
            .then_text(&triage(false))
            .then_text("Answer two."),
    );
    let mut h = start(first, Mode::Cowork, None);
    h.turn("question one", None).await;
    h.turn("question two", None).await;
    let id = h.session_id();
    let (tmp, work, data, _) = h.quit().await;

    let second = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("Answer three."),
    );
    let mut h = start(second.clone(), Mode::Cowork, Some((tmp, work, data)));
    // `/resume` without an id offers this directory's sessions in a picker.
    h.send(CoreCommand::Slash(SlashCommand::Resume { id: None }));
    let UiEvent::Picker(picker) = h.until(|e| matches!(e, UiEvent::Picker(_)), None).await else {
        unreachable!()
    };
    assert_eq!(picker.kind, PickerKind::Session);
    assert_eq!(picker.items.len(), 1);
    assert_eq!(picker.items[0].value, id.to_string());
    assert!(picker.items[0].detail.contains(&id.to_string()));
    h.send(CoreCommand::Picked {
        kind: PickerKind::Session,
        value: Some(picker.items[0].value.clone()),
    });
    h.until(
        |e| matches!(e, UiEvent::UserMessage { text } if text == "question two"),
        None,
    )
    .await;
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("resumed")),
        None,
    )
    .await;
    h.turn("question three", None).await;
    assert_eq!(h.session_id_or_status(), Some(id));
    let (_tmp, _w, data, _) = h.quit().await;

    let req = &second.requests()[1];
    let users: Vec<&str> = req
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(users, ["question one", "question two", "question three"]);
    let k = kinds(&data, &id);
    assert!(k.contains(&"SessionResumed"));
    assert_eq!(k.iter().filter(|x| **x == "SessionEnd").count(), 2);
}

impl Harness {
    fn session_id_or_status(&self) -> Option<SessionId> {
        self.seen.iter().rev().find_map(|e| match e {
            UiEvent::Status(s) => Some(s.session),
            _ => None,
        })?
    }
}

#[tokio::test]
async fn start_up_asks_for_a_model_and_holds_prompts_until_one_is_chosen() {
    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("Hi."),
    );
    let mut h = start_with(mock.clone(), Mode::Cowork, None, true);
    let UiEvent::Picker(picker) = h.until(|e| matches!(e, UiEvent::Picker(_)), None).await else {
        unreachable!()
    };
    assert_eq!(picker.kind, PickerKind::Model);
    assert_eq!(picker.items[0].value, "mock");
    assert!(
        h.seen
            .iter()
            .any(|e| matches!(e, UiEvent::Status(s) if s.model.is_empty())),
        "no model until one is picked"
    );

    h.send(CoreCommand::UserPrompt("hello".into()));
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("choose a model first")),
        None,
    )
    .await;
    assert!(mock.requests().is_empty(), "the prompt waits for the model");

    h.send(CoreCommand::Picked {
        kind: PickerKind::Model,
        value: Some("mock".into()),
    });
    h.until(|e| matches!(e, UiEvent::TurnFinished { .. }), None)
        .await;
    let (.., seen) = h.quit().await;
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Status(s) if s.model == "mock"))
    );
    assert_eq!(mock.requests()[1].model, "mock");
}

#[tokio::test]
async fn dismissing_the_model_picker_leaves_no_model() {
    let mock = Arc::new(MockProvider::new());
    let mut h = start_with(mock.clone(), Mode::Cowork, None, true);
    h.until(|e| matches!(e, UiEvent::Picker(_)), None).await;
    h.send(CoreCommand::Picked {
        kind: PickerKind::Model,
        value: None,
    });
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("no model selected")),
        None,
    )
    .await;
    h.send(CoreCommand::UserPrompt("hello".into()));
    h.until(
        |e| matches!(e, UiEvent::Notice { level: NoticeLevel::Error, text } if text.contains("/model")),
        None,
    )
    .await;
    h.quit().await;
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn resuming_from_another_directory_asks_first_and_prune_deletes_old_sessions() {
    let first = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("Answer one."),
    );
    let mut h = start(first, Mode::Cowork, None);
    h.turn("question one", None).await;
    let id = h.session_id();
    let (tmp, work, data, _) = h.quit().await;

    // The same data directory, but a different project folder.
    let other = work.parent().unwrap().join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    let second = Arc::new(MockProvider::new());
    let mut h = start(second, Mode::Cowork, Some((tmp, other, data)));
    h.send(CoreCommand::Slash(SlashCommand::Resume {
        id: Some(id.to_string()),
    }));
    let asked = h.until(|e| matches!(e, UiEvent::Question(_)), None).await;
    let UiEvent::Question(q) = asked else {
        unreachable!()
    };
    assert!(
        q.text.contains("was recorded in") && q.text.contains("Continue here?"),
        "{q:?}"
    );
    assert_eq!(q.options, ["continue here", "cancel"]);
    assert!(
        h.session_id_or_status().is_none(),
        "nothing is resumed before the answer"
    );
    h.send(CoreCommand::QuestionAnswer {
        id: q.id,
        answer: "continue here".into(),
    });
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.starts_with("resumed")),
        None,
    )
    .await;
    assert!(h.seen.iter().any(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.contains("was recorded in") && text.contains("continuing in"))
    ));
    assert_eq!(h.session_id_or_status(), Some(id));

    // A fresh session, then prune with a cutoff that catches the first (closed) one only.
    h.send(CoreCommand::Slash(SlashCommand::New));
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.contains("ready for a new session")),
        None,
    )
    .await;
    h.send(CoreCommand::Slash(SlashCommand::SessionsPrune { days: 0 }));
    let note = h
        .until(
            |e| matches!(e, UiEvent::Notice { text, .. } if text.contains("session(s) older than")),
            None,
        )
        .await;
    let UiEvent::Notice { text, .. } = note else {
        unreachable!()
    };
    assert!(text.starts_with("deleted 1 closed session(s)"), "{text}");
    let (_tmp, _w, data, _) = h.quit().await;
    assert!(!data.sessions_dir().join(id.log_file_name()).exists());
}

#[tokio::test]
async fn new_session_and_crash_recovery() {
    let (tmp, work, data) = dirs();
    let store = SessionStore::new(data.clone());
    let crashed = store
        .start(
            SessionStart {
                workdir: &dunce::canonicalize(&work).unwrap(),
                provider: "ollama",
                model: "m",
                mode: Mode::Cowork,
            },
            h_agent_core::AgentId::new(),
        )
        .unwrap();
    let crashed_id = crashed.id;
    drop(crashed);

    let mock = Arc::new(
        MockProvider::new()
            .then_text(&triage(false))
            .then_text("one")
            .then_text(&triage(false))
            .then_text("two"),
    );
    let mut h = start(mock, Mode::Cowork, Some((tmp, work, data)));
    h.until(
        |e| matches!(e, UiEvent::Notice { text, .. } if text.contains(&crashed_id.to_string())),
        None,
    )
    .await;
    h.turn("first", None).await;
    let a = h.session_id();
    h.send(CoreCommand::Slash(SlashCommand::New));
    h.until(|e| matches!(e, UiEvent::SessionClosed), None).await;
    h.turn("second", None).await;
    let b = h.session_id();
    assert_ne!(a, b);
    let (_tmp, _w, data, _) = h.quit().await;
    let store = SessionStore::new(data);
    let entry = store.index().get(&crashed_id).unwrap().unwrap();
    assert!(entry.recovered && entry.closed_at.is_some());
    assert!(store.index().get(&a).unwrap().unwrap().closed_at.is_some());
}
