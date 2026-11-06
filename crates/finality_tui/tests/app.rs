#![allow(clippy::unwrap_used)]

use std::time::{Duration, Instant};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use finality_core::events::{
    AgentLine, ApprovalChoice, ApprovalPrompt, CoreCommand, PickerItem, PickerKind, PickerPrompt,
    PlanView, QuestionPrompt, SlashCommand, StatusInfo, TaskLine, ToolStatus, UiEvent,
    UsageSnapshot,
};
use finality_core::{AgentId, Mode, ProviderKind, RiskClass, Role, TaskStatus, ThinkLevel};
use finality_tui::App;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::style::Color;

fn status() -> StatusInfo {
    StatusInfo {
        workdir: r"C:\src\app".into(),
        session: None,
        mode: Mode::Cowork,
        provider: ProviderKind::Ollama,
        model: "qwen3:14b".into(),
        think: ThinkLevel::Medium,
    }
}

fn draw(app: &mut App, w: u16, h: u16) -> Terminal<TestBackend> {
    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
    t.draw(|f| app.render(f)).unwrap();
    t
}

fn key(app: &mut App, code: KeyCode) -> Vec<CoreCommand> {
    key_at(app, code, KeyModifiers::NONE, Instant::now())
}

fn key_at(app: &mut App, code: KeyCode, mods: KeyModifiers, now: Instant) -> Vec<CoreCommand> {
    app.handle_terminal_event(Event::Key(KeyEvent::new(code, mods)), now)
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        key(app, KeyCode::Char(c));
    }
}

fn ctrl_c(app: &mut App, now: Instant) -> Vec<CoreCommand> {
    key_at(app, KeyCode::Char('c'), KeyModifiers::CONTROL, now)
}

fn approval(id: u64, inside: bool) -> ApprovalPrompt {
    ApprovalPrompt {
        id,
        agent_label: "worker1".into(),
        task: Some("task 2".into()),
        tool: "shell".into(),
        risk: RiskClass::Exec,
        summary: "run `git -C ../other status` in pwsh".into(),
        detail: Some("pwsh (C:\\pwsh.exe)\ngit -C ../other status".into()),
        reasons: if inside {
            vec![]
        } else {
            vec!["`../other` resolves outside the working directory".into()]
        },
        inside,
        allow_session: inside,
        default_deny: !inside,
    }
}

#[test]
fn idle_screen() {
    let mut app = App::new(status());
    insta::assert_snapshot!(draw(&mut app, 80, 12).backend());
}

#[test]
fn streaming_reply_with_tool_call() {
    let mut app = App::new(status());
    let agent = AgentId::new();
    type_text(&mut app, "read Cargo.toml");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::UserPrompt("read Cargo.toml".into())]
    );
    for ev in [
        UiEvent::SessionStarted {
            id: "000042_20261105-143012".parse().unwrap(),
        },
        UiEvent::TurnStarted,
        UiEvent::ThinkingDelta {
            agent,
            text: "Need to read the file.".into(),
        },
        UiEvent::ToolStarted {
            call_id: "c1".into(),
            tool: "read_file".into(),
            summary: "read Cargo.toml".into(),
        },
        UiEvent::ToolFinished {
            call_id: "c1".into(),
            status: ToolStatus::Succeeded,
            output: "[package]".into(),
        },
        UiEvent::AssistantDelta {
            agent,
            text: "The crate is **finality**.\n".into(),
        },
        UiEvent::AssistantDelta {
            agent,
            text: "- uses `tokio`".into(),
        },
        UiEvent::Usage(UsageSnapshot {
            last_input: 12_340,
            last_output: 2_105,
            ctx_used: 14_445,
            ctx_total: 32_000,
            session_in: 41_200,
            session_out: 7_900,
            tools_loaded: 0,
            mcp_running: 2,
        }),
    ] {
        app.handle_ui_event(ev);
    }
    insta::assert_snapshot!(draw(&mut app, 80, 20).backend());
}

#[test]
fn approval_modal_outside_defaults_to_deny() {
    let mut app = App::new(status());
    app.handle_ui_event(UiEvent::Approval(approval(7, false)));
    app.handle_ui_event(UiEvent::Approval(approval(8, true)));
    let t = draw(&mut app, 80, 20);
    insta::assert_snapshot!(t.backend());
    let buf = t.backend().buffer();
    let (x, y) = (0..20)
        .flat_map(|y| (0..80).map(move |x| (x, y)))
        .find(|&(x, y)| buf[(x, y)].symbol() == "R" && buf[(x + 1, y)].symbol() == "i")
        .unwrap();
    assert_eq!(
        buf[(x, y)].bg,
        Color::Indexed(236),
        "modals have their own background"
    );
    assert_eq!(
        buf[(0, y)].bg,
        Color::Reset,
        "the chat keeps the terminal's"
    );

    assert!(
        key(&mut app, KeyCode::Char('a')).is_empty(),
        "no session grant for outside actions"
    );
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::ApprovalResponse {
            id: 7,
            choice: ApprovalChoice::Deny { note: None }
        }]
    );
    assert_eq!(
        key(&mut app, KeyCode::Char('a')),
        vec![CoreCommand::ApprovalResponse {
            id: 8,
            choice: ApprovalChoice::AllowSession
        }]
    );
}

#[test]
fn approval_deny_with_note_and_withdrawal() {
    let mut app = App::new(status());
    app.handle_ui_event(UiEvent::Approval(approval(1, true)));
    key(&mut app, KeyCode::Char('e'));
    type_text(&mut app, "use src/");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::ApprovalResponse {
            id: 1,
            choice: ApprovalChoice::Deny {
                note: Some("use src/".into())
            }
        }]
    );
    app.handle_ui_event(UiEvent::Approval(approval(2, true)));
    app.handle_ui_event(UiEvent::ApprovalClosed { id: 2 });
    type_text(&mut app, "hi");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::UserPrompt("hi".into())]
    );
}

#[test]
fn status_bar_turns_red_above_80_percent() {
    let mut app = App::new(status());
    app.handle_ui_event(UiEvent::Usage(UsageSnapshot {
        last_input: 29_000,
        last_output: 1_000,
        ctx_used: 30_000,
        ctx_total: 32_000,
        session_in: 90_000,
        session_out: 4_000,
        tools_loaded: 0,
        mcp_running: 0,
    }));
    let t = draw(&mut app, 80, 8);
    insta::assert_snapshot!(t.backend());
    let buf = t.backend().buffer();
    let row = 7;
    let x = (0..80)
        .find(|&x| buf[(x, row)].symbol() == "c" && buf[(x + 1, row)].symbol() == "t")
        .unwrap();
    assert_eq!(buf[(x, row)].fg, Color::Red);

    app.handle_ui_event(UiEvent::Usage(UsageSnapshot {
        ctx_used: 20_000,
        ctx_total: 32_000,
        ..Default::default()
    }));
    let t = draw(&mut app, 80, 8);
    let buf = t.backend().buffer();
    let x = (0..80)
        .find(|&x| buf[(x, row)].symbol() == "c" && buf[(x + 1, row)].symbol() == "t")
        .unwrap();
    assert_eq!(buf[(x, row)].fg, Color::Yellow);
}

#[test]
fn ctrl_c_cancels_clears_and_quits() {
    let mut app = App::new(status());
    let t0 = Instant::now();

    app.handle_ui_event(UiEvent::TurnStarted);
    assert_eq!(ctrl_c(&mut app, t0), vec![CoreCommand::CancelTurn]);
    app.handle_ui_event(UiEvent::TurnFinished { cancelled: true });

    type_text(&mut app, "draft");
    assert!(ctrl_c(&mut app, t0).is_empty());
    assert_eq!(key(&mut app, KeyCode::Enter), vec![], "input was cleared");

    assert!(ctrl_c(&mut app, t0).is_empty());
    assert!(
        ctrl_c(&mut app, t0 + Duration::from_secs(3)).is_empty(),
        "second press too late"
    );
    assert!(!app.should_quit());
    assert_eq!(
        ctrl_c(&mut app, t0 + Duration::from_secs(4)),
        vec![CoreCommand::Quit]
    );
    assert!(app.should_quit());
}

#[test]
fn slash_commands_complete_and_parse() {
    let mut app = App::new(status());
    type_text(&mut app, "/mo");
    insta::assert_snapshot!(draw(&mut app, 80, 14).backend());
    key(&mut app, KeyCode::Down);
    key(&mut app, KeyCode::Tab);
    type_text(&mut app, "x");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::Slash(SlashCommand::Model("x".into()))]
    );

    type_text(&mut app, "/mode auto");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::Slash(SlashCommand::Mode(Mode::Auto))]
    );

    type_text(&mut app, "/mdoe plan");
    assert!(key(&mut app, KeyCode::Enter).is_empty());
    let screen = draw(&mut app, 80, 14).backend().to_string();
    assert!(screen.contains("did you mean /mode?"), "{screen}");

    type_text(&mut app, "/quit");
    assert_eq!(key(&mut app, KeyCode::Enter), vec![CoreCommand::Quit]);
}

#[test]
fn multiline_input_and_history() {
    let mut app = App::new(status());
    type_text(&mut app, "line one");
    assert!(key_at(&mut app, KeyCode::Enter, KeyModifiers::ALT, Instant::now()).is_empty());
    type_text(&mut app, "line two");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::UserPrompt("line one\nline two".into())]
    );

    type_text(&mut app, "second");
    key(&mut app, KeyCode::Enter);
    key(&mut app, KeyCode::Up);
    key(&mut app, KeyCode::Up);
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::UserPrompt("line one\nline two".into())]
    );
}

fn agent(label: &str, role: Role, phase: &str, task: Option<u32>) -> AgentLine {
    AgentLine {
        label: label.into(),
        role,
        phase: phase.into(),
        task,
        elapsed_secs: 12,
        ctx_used: 8_192,
        ctx_total: 32_768,
    }
}

fn task(number: u32, title: &str, status: TaskStatus, attempt: u32) -> TaskLine {
    TaskLine {
        number,
        title: title.into(),
        role: "worker".into(),
        status,
        attempt,
        max_attempts: 8,
        note: if status == TaskStatus::Failed {
            "compile error".into()
        } else {
            String::new()
        },
    }
}

fn plan_app() -> App {
    let mut app = App::new(status());
    app.handle_ui_event(UiEvent::AgentStatus(vec![
        agent("orchestrator", Role::Orchestrator, "tool:tasks_await", None),
        agent("worker1", Role::Worker, "thinking", Some(2)),
    ]));
    app.handle_ui_event(UiEvent::PlanUpdated(Some(PlanView {
        goal: "Add a config flag".into(),
        tasks: vec![
            task(1, "Find the parser", TaskStatus::Success, 1),
            task(2, "Add the flag", TaskStatus::Running, 2),
            task(3, "Update docs", TaskStatus::Pending, 0),
        ],
    })));
    app
}

#[test]
fn agents_panel_and_tasks_strip() {
    let mut app = plan_app();
    insta::assert_snapshot!(draw(&mut app, 100, 18).backend());

    // Tab hides the panel; narrow terminals never show it.
    assert!(key(&mut app, KeyCode::Tab).is_empty());
    let hidden = format!("{:?}", draw(&mut app, 100, 18).backend());
    assert!(!hidden.contains("Agents"));
    key(&mut app, KeyCode::Tab);
    let narrow = format!("{:?}", draw(&mut app, 70, 18).backend());
    assert!(!narrow.contains("Agents") && narrow.contains("Add the flag"));

    app.handle_ui_event(UiEvent::PlanUpdated(None));
    let cleared = format!("{:?}", draw(&mut app, 100, 18).backend());
    assert!(!cleared.contains("Add the flag"));
}

#[test]
fn failed_plan_leaves_the_screen_when_the_turn_ends() {
    let mut app = App::new(status());
    let plan = |second| {
        UiEvent::PlanUpdated(Some(PlanView {
            goal: "Add a config flag".into(),
            tasks: vec![
                task(1, "Find the parser", TaskStatus::Success, 1),
                task(2, "Add the flag", second, 3),
                task(3, "Update docs", TaskStatus::Pending, 0),
            ],
        }))
    };
    let shown =
        |app: &mut App| format!("{:?}", draw(app, 100, 18).backend()).contains("Add the flag");
    app.handle_ui_event(UiEvent::TurnStarted);
    app.handle_ui_event(plan(TaskStatus::Failed));
    assert!(shown(&mut app), "shown while the turn runs");
    app.handle_ui_event(UiEvent::TurnFinished { cancelled: false });
    assert!(!shown(&mut app), "gone once the turn stopped");

    // A plan still waiting to start stays (e.g. made in plan mode for review), and /tasks lists any plan.
    app.handle_ui_event(plan(TaskStatus::Pending));
    assert!(shown(&mut app));
    app.handle_ui_event(plan(TaskStatus::Failed));
    type_text(&mut app, "/tasks");
    key(&mut app, KeyCode::Enter);
    let screen = format!("{:?}", draw(&mut app, 120, 30).backend());
    assert!(screen.contains("plan: Add a config flag"));
}

fn picker() -> UiEvent {
    let item = |n: u32| PickerItem {
        value: format!("00000{n}_20261105-143012"),
        label: format!("session {n}"),
        detail: format!("00000{n}_20261105-143012 · 2026-11-05"),
    };
    UiEvent::Picker(PickerPrompt {
        kind: PickerKind::Session,
        title: "Resume a session".into(),
        items: vec![item(1), item(2), item(3)],
        selected: 0,
    })
}

#[test]
fn picker_selects_with_arrows_and_enter() {
    let mut app = App::new(status());
    app.handle_ui_event(picker());
    key(&mut app, KeyCode::Down);
    insta::assert_snapshot!(draw(&mut app, 80, 12).backend());

    type_text(&mut app, "xyz");
    key(&mut app, KeyCode::Down);
    key(&mut app, KeyCode::Down);
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::Picked {
            kind: PickerKind::Session,
            value: Some("000001_20261105-143012".into())
        }],
        "Down wraps around; typing does nothing"
    );
    assert!(
        key(&mut app, KeyCode::Enter).is_empty(),
        "the picker closed"
    );

    app.handle_ui_event(picker());
    key(&mut app, KeyCode::Up);
    assert_eq!(
        key(&mut app, KeyCode::Esc),
        vec![CoreCommand::Picked {
            kind: PickerKind::Session,
            value: None
        }]
    );
}

#[test]
fn question_modal_takes_options_text_and_skip() {
    let mut app = App::new(status());
    let question = |id| {
        UiEvent::Question(QuestionPrompt {
            id,
            agent_label: "orchestrator".into(),
            text: "Which config format should the flag use?".into(),
            options: vec!["TOML".into(), "environment variable".into()],
        })
    };
    app.handle_ui_event(question(1));
    app.handle_ui_event(question(2));
    app.handle_ui_event(question(3));
    insta::assert_snapshot!(draw(&mut app, 80, 16).backend());

    assert_eq!(
        key(&mut app, KeyCode::Char('2')),
        vec![CoreCommand::QuestionAnswer {
            id: 1,
            answer: "environment variable".into()
        }]
    );
    type_text(&mut app, "both");
    key(&mut app, KeyCode::Backspace);
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::QuestionAnswer {
            id: 2,
            answer: "bot".into()
        }]
    );
    // Approvals go first; the question waits behind them.
    app.handle_ui_event(UiEvent::Approval(approval(9, true)));
    assert!(matches!(
        key(&mut app, KeyCode::Char('y')).as_slice(),
        [CoreCommand::ApprovalResponse { id: 9, .. }]
    ));
    assert_eq!(
        key(&mut app, KeyCode::Esc),
        vec![CoreCommand::QuestionAnswer {
            id: 3,
            answer: String::new()
        }]
    );

    app.handle_ui_event(question(4));
    app.handle_ui_event(UiEvent::QuestionClosed { id: 4 });
    type_text(&mut app, "hi");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::UserPrompt("hi".into())]
    );
}

#[test]
fn phase7_slash_commands() {
    let mut app = plan_app();
    type_text(&mut app, "/subagents 2");
    assert_eq!(
        key(&mut app, KeyCode::Enter),
        vec![CoreCommand::Slash(SlashCommand::Subagents(2))]
    );
    type_text(&mut app, "/tasks");
    assert!(key(&mut app, KeyCode::Enter).is_empty());
    type_text(&mut app, "/agents");
    assert!(key(&mut app, KeyCode::Enter).is_empty());
    let screen = format!("{:?}", draw(&mut app, 120, 40).backend());
    assert!(screen.contains("plan: Add a config flag"));
    assert!(screen.contains("2. Add the flag [worker] running · attempt 2/8"));
    assert!(screen.contains("worker1 (worker) task 2 · thinking · 12s"));
}
