//! UI state, key handling and rendering. Pure with respect to I/O so it can be driven by tests.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};
use finality_core::events::{
    AgentLine, ApprovalChoice, ApprovalPrompt, CoreCommand, NoticeLevel, PickerPrompt, PlanView,
    QuestionPrompt, SkillInfo, StatusInfo, ToolStatus, UiEvent, UsageSnapshot,
};
use finality_core::{Mode, TaskStatus};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};
use tui_textarea::{CursorMove, TextArea};

use crate::format::{compact, free_pct, thousands};
use crate::markdown;
use crate::slash::{self, COMMANDS, SlashAction};

const QUIT_WINDOW: Duration = Duration::from_secs(2);
const MAX_INPUT_ROWS: u16 = 6;
const MAX_TOOL_OUTPUT_LINES: usize = 20;
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const AGENTS_PANEL_WIDTH: u16 = 32;
/// Narrower terminals hide the agents panel.
const AGENTS_PANEL_MIN_TERMINAL: u16 = 80;
/// Task rows in the plan box; finished tasks give way when there are more.
const MAX_STRIP_TASKS: usize = 5;
const MAX_PICKER_ROWS: usize = 15;
/// Modals (approvals, questions, pickers) get their own background so they stand apart from the chat.
const MODAL: Style = Style::new().fg(Color::White).bg(Color::Indexed(236));
/// Prompts and answers are brighter than thinking, tool lines and notices.
const BODY: Style = Style::new().fg(Color::White);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChatItem {
    User(String),
    Assistant {
        text: String,
        thinking: String,
        streaming: bool,
        cancelled: bool,
    },
    Tool {
        call_id: String,
        tool: String,
        summary: String,
        status: ToolStatus,
        output: String,
    },
    Notice {
        level: NoticeLevel,
        text: String,
    },
}

struct NoteEntry {
    text: String,
}

pub struct App {
    status: StatusInfo,
    usage: UsageSnapshot,
    chat: Vec<ChatItem>,
    input: TextArea<'static>,
    history: Vec<String>,
    history_pos: Option<usize>,
    approvals: VecDeque<ApprovalPrompt>,
    note: Option<NoteEntry>,
    detail_scroll: u16,
    busy: bool,
    spinner: usize,
    last_ctrl_c: Option<Instant>,
    /// Lines scrolled up from the bottom of the chat.
    scroll: u16,
    show_details: bool,
    completion: usize,
    quit: bool,
    agents: Vec<AgentLine>,
    show_agents: bool,
    plan: Option<PlanView>,
    questions: VecDeque<QuestionPrompt>,
    answer_text: String,
    picker: Option<PickerPrompt>,
    skills: Vec<SkillInfo>,
    decide: bool,
    direct: bool,
}

fn new_input(lines: Vec<String>) -> TextArea<'static> {
    let mut t = TextArea::new(lines);
    t.set_cursor_line_style(Style::default());
    t.set_placeholder_text("Ask anything, or type / for commands");
    t.set_placeholder_style(Style::default().fg(Color::DarkGray));
    t.set_block(
        Block::bordered()
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" › ")
            .title_bottom(
                Line::from(" Enter send · Shift/Alt+Enter newline · Ctrl+O details ")
                    .right_aligned(),
            ),
    );
    t.move_cursor(CursorMove::Bottom);
    t.move_cursor(CursorMove::End);
    t
}

impl App {
    pub fn new(status: StatusInfo) -> Self {
        Self {
            status,
            usage: UsageSnapshot::default(),
            chat: Vec::new(),
            input: new_input(vec![String::new()]),
            history: Vec::new(),
            history_pos: None,
            approvals: VecDeque::new(),
            note: None,
            detail_scroll: 0,
            busy: false,
            spinner: 0,
            last_ctrl_c: None,
            scroll: 0,
            show_details: false,
            completion: 0,
            quit: false,
            agents: Vec::new(),
            show_agents: true,
            plan: None,
            questions: VecDeque::new(),
            answer_text: String::new(),
            picker: None,
            skills: Vec::new(),
            decide: false,
            direct: false,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.quit
    }

    /// Whether an approval modal is open.
    pub fn approval_pending(&self) -> bool {
        !self.approvals.is_empty()
    }

    /// Shows a prompt that did not come from the input box (the CLI's initial prompt).
    pub fn prompt(&mut self, text: &str) -> CoreCommand {
        self.push(ChatItem::User(text.to_owned()));
        self.history.push(text.to_owned());
        CoreCommand::UserPrompt(text.to_owned())
    }

    pub fn tick(&mut self) {
        if self.busy {
            self.spinner = (self.spinner + 1) % SPINNER.len();
        }
    }

    fn notice(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.push(ChatItem::Notice {
            level,
            text: text.into(),
        });
    }

    fn push(&mut self, item: ChatItem) {
        self.chat.push(item);
    }

    fn input_text(&self) -> String {
        self.input.lines().join("\n")
    }

    fn input_is_empty(&self) -> bool {
        self.input.lines().iter().all(String::is_empty)
    }

    fn set_input(&mut self, text: &str) {
        self.input = new_input(text.split('\n').map(str::to_owned).collect());
    }

    /// The completion list is shown while the input is a single `/word` without spaces.
    fn completion_matches(&self) -> Vec<slash::Completion> {
        let text = self.input_text();
        match text.strip_prefix('/') {
            Some(rest) if !rest.contains(char::is_whitespace) && self.input.lines().len() == 1 => {
                slash::completions_with(rest, &self.skills)
            }
            _ => Vec::new(),
        }
    }

    // ------------------------------------------------------------------ events from the core

    pub fn handle_ui_event(&mut self, event: UiEvent) {
        match event {
            UiEvent::Notice { level, text } => self.notice(level, text),
            UiEvent::Status(s) => self.status = s,
            UiEvent::SessionStarted { id } => {
                self.status.session = Some(id);
                self.notice(NoticeLevel::Info, format!("session {id} started"));
            }
            UiEvent::SessionClosed => self.status.session = None,
            UiEvent::UserMessage { text } => self.push(ChatItem::User(text)),
            UiEvent::AssistantDelta { text, .. } => self.assistant_mut().0.push_str(&text),
            UiEvent::ThinkingDelta { text, .. } => self.assistant_mut().1.push_str(&text),
            UiEvent::ToolStarted {
                call_id,
                tool,
                summary,
            } => {
                self.end_streaming(false);
                self.push(ChatItem::Tool {
                    call_id,
                    tool,
                    summary,
                    status: ToolStatus::Running,
                    output: String::new(),
                });
            }
            UiEvent::ToolFinished {
                call_id,
                status: s,
                output: o,
            } => {
                let found = self.chat.iter_mut().rev().find_map(|item| match item {
                    ChatItem::Tool {
                        call_id: id,
                        status,
                        output,
                        ..
                    } if *id == call_id => Some((status, output)),
                    _ => None,
                });
                if let Some((status, output)) = found {
                    *status = s;
                    *output = o;
                }
            }
            UiEvent::Approval(p) => self.approvals.push_back(p),
            UiEvent::ApprovalClosed { id } => {
                let was_front = self.approvals.front().is_some_and(|p| p.id == id);
                self.approvals.retain(|p| p.id != id);
                if was_front {
                    self.note = None;
                    self.detail_scroll = 0;
                }
            }
            UiEvent::Usage(u) => self.usage = u,
            UiEvent::ModeChanged { mode } => {
                self.status.mode = mode;
                self.notice(NoticeLevel::Info, format!("mode: {mode}"));
            }
            UiEvent::TurnStarted => self.busy = true,
            UiEvent::TurnFinished { cancelled } => {
                self.busy = false;
                self.end_streaming(cancelled);
            }
            UiEvent::AgentStatus(agents) => self.agents = agents,
            UiEvent::PlanUpdated(plan) => self.plan = plan,
            UiEvent::Question(q) => self.questions.push_back(q),
            UiEvent::QuestionClosed { id } => {
                let was_front = self.questions.front().is_some_and(|q| q.id == id);
                self.questions.retain(|q| q.id != id);
                if was_front {
                    self.answer_text.clear();
                }
            }
            UiEvent::Skills(skills) => self.skills = skills,
            UiEvent::DecideChanged { on } => self.decide = on,
            UiEvent::DirectChanged { on } => self.direct = on,
            UiEvent::Picker(mut p) if !p.items.is_empty() => {
                p.selected = p.selected.min(p.items.len() - 1);
                self.picker = Some(p);
            }
            UiEvent::Picker(_) => {}
        }
    }

    /// (text, thinking) of the assistant message being streamed, creating one if needed.
    fn assistant_mut(&mut self) -> (&mut String, &mut String) {
        if !matches!(
            self.chat.last(),
            Some(ChatItem::Assistant {
                streaming: true,
                ..
            })
        ) {
            self.push(ChatItem::Assistant {
                text: String::new(),
                thinking: String::new(),
                streaming: true,
                cancelled: false,
            });
        }
        match self.chat.last_mut() {
            Some(ChatItem::Assistant { text, thinking, .. }) => (text, thinking),
            _ => unreachable!("an assistant item was just ensured"),
        }
    }

    fn end_streaming(&mut self, was_cancelled: bool) {
        if let Some(ChatItem::Assistant {
            streaming,
            cancelled,
            ..
        }) = self.chat.last_mut()
            && *streaming
        {
            *streaming = false;
            *cancelled = was_cancelled;
        }
    }

    // ------------------------------------------------------------------ terminal input

    pub fn handle_terminal_event(&mut self, event: Event, now: Instant) -> Vec<CoreCommand> {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.handle_key(key, now),
            Event::Paste(text) => {
                if let Some(note) = &mut self.note {
                    note.text.push_str(&text.replace(['\r', '\n'], " "));
                } else if !self.approvals.is_empty() || self.picker.is_some() {
                } else if !self.questions.is_empty() {
                    self.answer_text.push_str(&text.replace(['\r', '\n'], " "));
                } else {
                    self.input.insert_str(text.replace("\r\n", "\n"));
                }
                Vec::new()
            }
            Event::Mouse(m) => {
                match m.kind {
                    MouseEventKind::ScrollUp => self.scroll = self.scroll.saturating_add(3),
                    MouseEventKind::ScrollDown => self.scroll = self.scroll.saturating_sub(3),
                    _ => {}
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent, now: Instant) -> Vec<CoreCommand> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && key.code == KeyCode::Char('c') {
            return self.ctrl_c(now);
        }
        if !self.approvals.is_empty() {
            return self.approval_key(key);
        }
        if !self.questions.is_empty() {
            return self.question_key(key);
        }
        if self.picker.is_some() {
            return self.picker_key(key);
        }
        if ctrl && key.code == KeyCode::Char('o') {
            self.show_details = !self.show_details;
            return Vec::new();
        }

        let matches = self.completion_matches();
        if !matches.is_empty() {
            match key.code {
                KeyCode::Tab => {
                    let pick = &matches[self.completion.min(matches.len() - 1)];
                    let text = format!("/{} ", pick.name);
                    self.set_input(&text);
                    self.completion = 0;
                    return Vec::new();
                }
                KeyCode::Up => {
                    self.completion = self.completion.checked_sub(1).unwrap_or(matches.len() - 1);
                    return Vec::new();
                }
                KeyCode::Down => {
                    self.completion = (self.completion + 1) % matches.len();
                    return Vec::new();
                }
                KeyCode::Esc => {
                    self.set_input("");
                    return Vec::new();
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
            {
                self.input.insert_newline();
                Vec::new()
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Tab => {
                self.show_agents = !self.show_agents;
                Vec::new()
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_add(10);
                Vec::new()
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_sub(10);
                Vec::new()
            }
            KeyCode::Up if self.history_pos.is_some() || self.input_is_empty() => {
                if !self.history.is_empty() {
                    let pos = self
                        .history_pos
                        .map_or(self.history.len() - 1, |p| p.saturating_sub(1));
                    self.history_pos = Some(pos);
                    let text = self.history[pos].clone();
                    self.set_input(&text);
                }
                Vec::new()
            }
            KeyCode::Down if self.history_pos.is_some() => {
                match self.history_pos {
                    Some(p) if p + 1 < self.history.len() => {
                        self.history_pos = Some(p + 1);
                        let text = self.history[p + 1].clone();
                        self.set_input(&text);
                    }
                    _ => {
                        self.history_pos = None;
                        self.set_input("");
                    }
                }
                Vec::new()
            }
            _ => {
                self.input.input(key);
                self.history_pos = None;
                self.completion = 0;
                Vec::new()
            }
        }
    }

    fn ctrl_c(&mut self, now: Instant) -> Vec<CoreCommand> {
        if self.busy {
            self.notice(NoticeLevel::Warning, "cancelling…");
            return vec![CoreCommand::CancelTurn];
        }
        if self.note.is_some() {
            self.note = None;
            return Vec::new();
        }
        if !self.input_is_empty() {
            self.set_input("");
            return Vec::new();
        }
        if self
            .last_ctrl_c
            .is_some_and(|t| now.duration_since(t) <= QUIT_WINDOW)
        {
            self.quit = true;
            return vec![CoreCommand::Quit];
        }
        self.last_ctrl_c = Some(now);
        self.notice(NoticeLevel::Info, "press Ctrl+C again to quit");
        Vec::new()
    }

    fn submit(&mut self) -> Vec<CoreCommand> {
        let text = self.input_text();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        let text = trimmed.to_owned();
        self.set_input("");
        self.history_pos = None;
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        self.scroll = 0;
        if !text.starts_with('/') {
            self.push(ChatItem::User(text.clone()));
            return vec![CoreCommand::UserPrompt(text)];
        }
        match slash::parse_with(
            &text,
            &self
                .skills
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
        ) {
            Ok(SlashAction::Help) => {
                self.notice(NoticeLevel::Info, help_text(&self.skills));
                Vec::new()
            }
            Ok(SlashAction::Quit) => {
                self.quit = true;
                vec![CoreCommand::Quit]
            }
            Ok(SlashAction::Agents) => {
                self.notice(NoticeLevel::Info, self.agents_text());
                Vec::new()
            }
            Ok(SlashAction::Tasks) => {
                self.notice(NoticeLevel::Info, self.tasks_text());
                Vec::new()
            }
            Ok(SlashAction::Core(cmd)) => {
                if cmd == finality_core::events::SlashCommand::Mode(Mode::Auto) {
                    self.notice(
                        NoticeLevel::Warning,
                        "auto mode: tools run without asking inside the working directory",
                    );
                }
                vec![CoreCommand::Slash(cmd)]
            }
            Err(msg) => {
                self.notice(NoticeLevel::Error, msg);
                Vec::new()
            }
        }
    }

    fn approval_key(&mut self, key: KeyEvent) -> Vec<CoreCommand> {
        let Some(prompt) = self.approvals.front() else {
            return Vec::new();
        };
        let id = prompt.id;
        if let Some(note) = &mut self.note {
            match key.code {
                KeyCode::Enter => {
                    let text = note.text.trim().to_owned();
                    return self.answer(
                        id,
                        ApprovalChoice::Deny {
                            note: (!text.is_empty()).then_some(text),
                        },
                    );
                }
                KeyCode::Esc => self.note = None,
                KeyCode::Backspace => {
                    note.text.pop();
                }
                KeyCode::Char(c) => note.text.push(c),
                _ => {}
            }
            return Vec::new();
        }
        let choice = match key.code {
            KeyCode::Char('y' | 'Y') => ApprovalChoice::AllowOnce,
            KeyCode::Char('a' | 'A') if prompt.allow_session => ApprovalChoice::AllowSession,
            KeyCode::Char('n' | 'N') | KeyCode::Esc => ApprovalChoice::Deny { note: None },
            KeyCode::Char('e' | 'E') => {
                self.note = Some(NoteEntry {
                    text: String::new(),
                });
                return Vec::new();
            }
            KeyCode::Enter if prompt.default_deny => ApprovalChoice::Deny { note: None },
            KeyCode::Enter => ApprovalChoice::AllowOnce,
            KeyCode::Down => {
                self.detail_scroll = self.detail_scroll.saturating_add(1);
                return Vec::new();
            }
            KeyCode::Up => {
                self.detail_scroll = self.detail_scroll.saturating_sub(1);
                return Vec::new();
            }
            KeyCode::PageDown => {
                self.detail_scroll = self.detail_scroll.saturating_add(10);
                return Vec::new();
            }
            KeyCode::PageUp => {
                self.detail_scroll = self.detail_scroll.saturating_sub(10);
                return Vec::new();
            }
            _ => return Vec::new(),
        };
        self.answer(id, choice)
    }

    fn answer(&mut self, id: u64, choice: ApprovalChoice) -> Vec<CoreCommand> {
        self.approvals.pop_front();
        self.note = None;
        self.detail_scroll = 0;
        vec![CoreCommand::ApprovalResponse { id, choice }]
    }

    fn question_key(&mut self, key: KeyEvent) -> Vec<CoreCommand> {
        let Some(q) = self.questions.front() else {
            return Vec::new();
        };
        let answer = match key.code {
            KeyCode::Char(c @ '1'..='9') if self.answer_text.is_empty() => {
                let i = c as usize - '1' as usize;
                match q.options.get(i) {
                    Some(option) => option.clone(),
                    None => {
                        self.answer_text.push(c);
                        return Vec::new();
                    }
                }
            }
            KeyCode::Char(c) => {
                self.answer_text.push(c);
                return Vec::new();
            }
            KeyCode::Backspace => {
                self.answer_text.pop();
                return Vec::new();
            }
            KeyCode::Enter => self.answer_text.trim().to_owned(),
            // An empty answer tells the asker the user skipped the question.
            KeyCode::Esc => String::new(),
            _ => return Vec::new(),
        };
        let id = q.id;
        self.questions.pop_front();
        self.answer_text.clear();
        vec![CoreCommand::QuestionAnswer { id, answer }]
    }

    fn picker_key(&mut self, key: KeyEvent) -> Vec<CoreCommand> {
        let Some(p) = &mut self.picker else {
            return Vec::new();
        };
        let last = p.items.len().saturating_sub(1);
        match key.code {
            KeyCode::Up => p.selected = p.selected.checked_sub(1).unwrap_or(last),
            KeyCode::Down => {
                p.selected = if p.selected >= last {
                    0
                } else {
                    p.selected + 1
                }
            }
            KeyCode::PageUp => p.selected = p.selected.saturating_sub(10),
            KeyCode::PageDown => p.selected = (p.selected + 10).min(last),
            KeyCode::Home => p.selected = 0,
            KeyCode::End => p.selected = last,
            KeyCode::Enter | KeyCode::Esc => {
                let kind = p.kind;
                let value = (key.code == KeyCode::Enter)
                    .then(|| p.items.get(p.selected).map(|i| i.value.clone()))
                    .flatten();
                self.picker = None;
                return vec![CoreCommand::Picked { kind, value }];
            }
            _ => {}
        }
        Vec::new()
    }

    fn agents_text(&self) -> String {
        if self.agents.is_empty() {
            return "no agents are running".into();
        }
        let mut out = String::from("agents:");
        for a in &self.agents {
            let task = a.task.map(|t| format!(" task {t}")).unwrap_or_default();
            out.push_str(&format!(
                "\n  {} ({}){task} · {} · {}s · ctx {}/{}",
                a.label,
                a.role,
                a.phase,
                a.elapsed_secs,
                compact(a.ctx_used.into()),
                compact(a.ctx_total.into())
            ));
        }
        out
    }

    fn tasks_text(&self) -> String {
        let Some(plan) = &self.plan else {
            return "no plan".into();
        };
        let mut out = format!("plan: {}", plan.goal);
        for t in &plan.tasks {
            out.push_str(&format!(
                "\n  {} {}. {} [{}] {} · attempt {}/{}",
                task_icon(t.status),
                t.number,
                t.title,
                t.role,
                t.status,
                t.attempt,
                t.max_attempts
            ));
            if !t.note.is_empty() {
                out.push_str(&format!("\n      {}", t.note));
            }
        }
        out
    }

    // ------------------------------------------------------------------ rendering

    /// The plan box shows while work runs; once the turn is over and the plan finished or failed it
    /// goes away (`/tasks` still lists it).
    fn plan_visible(&self) -> Option<&PlanView> {
        let plan = self.plan.as_ref()?;
        let active = plan.tasks.iter().any(|t| {
            matches!(
                t.status,
                TaskStatus::Running | TaskStatus::Queued | TaskStatus::AwaitingInput
            )
        });
        let over = plan.tasks.iter().all(|t| task_finished(t.status))
            || plan.tasks.iter().any(|t| {
                matches!(
                    t.status,
                    TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::Blocked
                )
            });
        (self.busy || active || !over).then_some(plan)
    }

    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let input_rows = (self.input.lines().len() as u16).clamp(1, MAX_INPUT_ROWS) + 2;
        let strip_rows = self
            .plan_visible()
            .map_or(0, |p| p.tasks.len().min(MAX_STRIP_TASKS) as u16 + 2);
        let [header, body, strip, input, status] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(strip_rows),
            Constraint::Length(input_rows),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let panel = self.show_agents
            && !self.agents.is_empty()
            && frame.area().width >= AGENTS_PANEL_MIN_TERMINAL;
        let (chat, agents) = if panel {
            let [chat, agents] =
                Layout::horizontal([Constraint::Min(1), Constraint::Length(AGENTS_PANEL_WIDTH)])
                    .areas(body);
            (chat, Some(agents))
        } else {
            (body, None)
        };

        frame.render_widget(
            Paragraph::new(self.header_line()).style(Style::default().bg(Color::DarkGray)),
            header,
        );
        self.render_chat(frame, chat);
        if let Some(area) = agents {
            self.render_agents(frame, area);
        }
        if let Some(plan) = self.plan_visible()
            && strip_rows > 0
        {
            let done = plan
                .tasks
                .iter()
                .filter(|t| matches!(t.status, TaskStatus::Success | TaskStatus::Partial))
                .count();
            let block = Block::bordered()
                .border_style(Style::default().fg(Color::LightBlue))
                .title(Line::from(vec![
                    Span::styled(" Plan ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(
                        format!("{done}/{} · ", plan.tasks.len()),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::raw(format!("{} ", plan.goal)),
                ]));
            frame.render_widget(Paragraph::new(self.strip_lines(plan)).block(block), strip);
        }
        frame.render_widget(&self.input, input);
        frame.render_widget(Paragraph::new(self.status_line()), status);

        let matches = self.completion_matches();
        if !matches.is_empty()
            && self.approvals.is_empty()
            && self.questions.is_empty()
            && self.picker.is_none()
        {
            render_completion(
                frame,
                chat,
                &matches,
                self.completion.min(matches.len() - 1),
            );
        }
        if let Some(prompt) = self.approvals.front() {
            self.render_approval(frame, chat, prompt);
        } else if let Some(q) = self.questions.front() {
            self.render_question(frame, chat, q);
        } else if let Some(p) = &self.picker {
            render_picker(frame, chat, p);
        }
    }

    fn render_agents(&self, frame: &mut Frame<'_>, area: Rect) {
        let dim = Style::default().fg(Color::Gray);
        let mut lines = Vec::new();
        for a in &self.agents {
            let task = a.task.map(|t| format!(" #{t}")).unwrap_or_default();
            lines.push(Line::from(vec![
                Span::styled("● ", Style::default().fg(phase_color(&a.phase))),
                Span::styled(
                    format!("{}{task}", a.label),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" · {}s", a.elapsed_secs), dim),
            ]));
            let ctx = if a.ctx_total == 0 {
                String::new()
            } else {
                format!(" · {}% free", free_pct(a.ctx_used, a.ctx_total))
            };
            lines.push(Line::styled(format!("  {}{ctx}", a.phase), dim));
        }
        let block = Block::bordered()
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Agents ")
            .title_bottom(Line::from(" Tab hide ").right_aligned());
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn strip_lines(&self, plan: &PlanView) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        // When the plan is taller than the box, keep finished tasks out of the way.
        let room = MAX_STRIP_TASKS;
        let mut shown: Vec<_> = plan.tasks.iter().collect();
        if shown.len() > room {
            shown.sort_by_key(|t| (task_finished(t.status), t.number));
            shown.truncate(room);
            shown.sort_by_key(|t| t.number);
        }
        for t in shown {
            let icon = if t.status == TaskStatus::Running {
                SPINNER[self.spinner]
            } else {
                task_icon(t.status)
            };
            let attempt = if t.attempt > 1 {
                format!(" (attempt {}/{})", t.attempt, t.max_attempts)
            } else {
                String::new()
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {icon} "),
                    Style::default().fg(task_color(t.status)),
                ),
                Span::raw(format!("{}. {}", t.number, t.title)),
                Span::styled(
                    format!(" [{}]{attempt}", t.role),
                    Style::default().fg(Color::Gray),
                ),
            ]));
        }
        lines
    }

    fn render_question(&self, frame: &mut Frame<'_>, area: Rect, q: &QuestionPrompt) {
        let width = area.width.saturating_sub(4).min(100);
        let mut lines: Vec<Line<'static>> = q
            .text
            .lines()
            .map(|l| Line::styled(l.to_owned(), Style::default().add_modifier(Modifier::BOLD)))
            .collect();
        if !q.options.is_empty() {
            lines.push(Line::default());
            for (i, o) in q.options.iter().take(9).enumerate() {
                lines.push(Line::from(vec![
                    Span::styled(format!("[{}]", i + 1), Style::default().fg(Color::Cyan)),
                    Span::raw(format!(" {o}")),
                ]));
            }
        }
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::raw("Answer: "),
            Span::raw(self.answer_text.clone()),
            Span::raw("▌"),
        ]));
        let pick = if q.options.is_empty() {
            ""
        } else {
            "1-9 pick · "
        };
        lines.push(Line::styled(
            format!("{pick}type an answer · Enter send · Esc skip"),
            Style::default().fg(Color::Gray),
        ));
        let inner = usize::from(width.saturating_sub(2)).max(1);
        let rows: usize = lines.iter().map(|l| l.width().div_ceil(inner).max(1)).sum();
        let height = (u16::try_from(rows).unwrap_or(u16::MAX).saturating_add(2)).min(area.height);
        let rect = Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        };
        let count = if self.questions.len() > 1 {
            format!(" · 1 of {}", self.questions.len())
        } else {
            String::new()
        };
        let block = Block::bordered()
            .border_style(Style::default().fg(Color::Cyan))
            .title(format!(" Question ─ {}{count} ", q.agent_label));
        frame.render_widget(Clear, rect);
        frame.render_widget(
            Paragraph::new(lines)
                .style(MODAL)
                .block(block)
                .wrap(Wrap { trim: false }),
            rect,
        );
    }

    fn header_line(&self) -> Line<'static> {
        let s = &self.status;
        let sep = || Span::styled(" │ ", Style::default().fg(Color::Gray));
        let session = s
            .session
            .map_or_else(|| "no session".to_owned(), |id| id.to_string());
        Line::from(vec![
            Span::styled(
                format!(" {}", finality_core::APP_NAME),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            sep(),
            Span::raw(s.workdir.clone()),
            sep(),
            Span::raw(session),
            sep(),
            Span::styled(format!("mode: {}", s.mode), mode_style(s.mode)),
            sep(),
            Span::raw(format!(
                "{}:{}",
                s.provider,
                if s.model.is_empty() {
                    "no model"
                } else {
                    &s.model
                }
            )),
            sep(),
            Span::raw(format!("think: {}", s.think)),
        ])
    }

    fn status_line(&self) -> Line<'static> {
        let u = &self.usage;
        let dim = Style::default().fg(Color::Gray);
        let mut spans = vec![Span::raw(format!(
            " in {} · out {} · ",
            thousands(u.last_input.into()),
            thousands(u.last_output.into())
        ))];
        if u.ctx_total == 0 {
            spans.push(Span::styled("ctx –", dim));
        } else {
            let used_pct = 100 - free_pct(u.ctx_used, u.ctx_total);
            let color = match used_pct {
                0..60 => Color::Green,
                60..=80 => Color::Yellow,
                _ => Color::Red,
            };
            spans.push(Span::styled(
                format!(
                    "ctx {}/{} ({}% free)",
                    compact(u.ctx_used.into()),
                    compact(u.ctx_total.into()),
                    free_pct(u.ctx_used, u.ctx_total)
                ),
                Style::default().fg(color),
            ));
        }
        spans.push(Span::styled(
            format!(" · Σ {}/{}", compact(u.session_in), compact(u.session_out)),
            dim,
        ));
        if u.tools_loaded > 0 {
            spans.push(Span::styled(format!(" · +{} tools", u.tools_loaded), dim));
        }
        if u.mcp_running > 0 {
            spans.push(Span::styled(format!(" · MCP {}", u.mcp_running), dim));
        }
        if self.decide {
            spans.push(Span::styled(" · decide on", dim));
        }
        if self.direct {
            spans.push(Span::styled(
                " · direct",
                Style::default().fg(Color::Magenta),
            ));
        }
        if self.busy {
            spans.push(Span::styled(
                format!("  {} working", SPINNER[self.spinner]),
                Style::default().fg(Color::Cyan),
            ));
        }
        if !self.approvals.is_empty() {
            spans.push(Span::styled(
                format!("  ⚠ {} approval(s) pending", self.approvals.len()),
                Style::default().fg(Color::Yellow),
            ));
        }
        if !self.questions.is_empty() {
            spans.push(Span::styled(
                format!("  ? {} question(s)", self.questions.len()),
                Style::default().fg(Color::Cyan),
            ));
        }
        Line::from(spans)
    }

    fn chat_lines(&self) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        for item in &self.chat {
            if !lines.is_empty() {
                lines.push(Line::default());
            }
            match item {
                ChatItem::User(text) => {
                    for (i, l) in text.split('\n').enumerate() {
                        let prefix = if i == 0 { "you › " } else { "      " };
                        lines.push(Line::from(vec![
                            Span::styled(
                                prefix,
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(l.to_owned(), BODY),
                        ]));
                    }
                }
                ChatItem::Assistant {
                    text,
                    thinking,
                    streaming,
                    cancelled,
                } => {
                    let thought = Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC);
                    if !thinking.is_empty() {
                        if self.show_details {
                            lines.push(Line::styled("  ▾ thinking", thought));
                            lines.extend(
                                thinking
                                    .lines()
                                    .map(|l| Line::styled(format!("    {l}"), thought)),
                            );
                        } else {
                            let n = thinking.chars().count();
                            lines.push(Line::styled(format!("  ▸ thinking ({n} chars)"), thought));
                        }
                    }
                    // A message that so far only has thinking (e.g. before a tool call) shows no body.
                    let show_body = !text.is_empty() || (*streaming && thinking.is_empty());
                    if show_body {
                        lines.push(Line::styled(
                            format!("{} ›", finality_core::APP_NAME),
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ));
                        let mut body = markdown::render(text, BODY);
                        if *streaming && let Some(last) = body.last_mut() {
                            last.spans
                                .push(Span::styled("▌", Style::default().fg(Color::Green)));
                        }
                        lines.extend(body);
                    }
                    if *cancelled {
                        lines.push(Line::styled("[cancelled]", Style::default().fg(Color::Red)));
                    }
                }
                ChatItem::Tool {
                    tool,
                    summary,
                    status,
                    output,
                    ..
                } => {
                    let (icon, color) = match status {
                        ToolStatus::Running => (SPINNER[self.spinner], Color::Cyan),
                        ToolStatus::Succeeded => ("✓", Color::Green),
                        ToolStatus::Failed => ("✗", Color::Red),
                        ToolStatus::Denied => ("⊘", Color::Yellow),
                    };
                    let arrow = if self.show_details { "▾" } else { "▸" };
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("  {arrow} {tool} "),
                            Style::default().fg(Color::Magenta),
                        ),
                        Span::raw(summary.clone()),
                        Span::styled(format!(" {icon}"), Style::default().fg(color)),
                    ]));
                    if self.show_details && !output.is_empty() {
                        let dim = Style::default().fg(Color::Gray);
                        let all: Vec<&str> = output.lines().collect();
                        for l in all.iter().take(MAX_TOOL_OUTPUT_LINES) {
                            lines.push(Line::styled(format!("    {l}"), dim));
                        }
                        if all.len() > MAX_TOOL_OUTPUT_LINES {
                            lines.push(Line::styled(
                                format!("    … {} more lines", all.len() - MAX_TOOL_OUTPUT_LINES),
                                dim,
                            ));
                        }
                    }
                }
                ChatItem::Notice { level, text } => {
                    let (prefix, style) = match level {
                        NoticeLevel::Info => ("· ", Style::default().fg(Color::Gray)),
                        NoticeLevel::Warning => ("! ", Style::default().fg(Color::Yellow)),
                        NoticeLevel::Error => ("✗ ", Style::default().fg(Color::Red)),
                    };
                    for (i, l) in text.split('\n').enumerate() {
                        let p = if i == 0 { prefix } else { "  " };
                        lines.push(Line::styled(format!("{p}{l}"), style));
                    }
                }
            }
        }
        lines
    }

    fn render_chat(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let lines = self.chat_lines();
        if lines.is_empty() {
            let hint = Paragraph::new(vec![
                Line::default(),
                Line::styled(
                    "  Type a request to start a session.",
                    Style::default().fg(Color::Gray),
                ),
                Line::styled(
                    "  /help lists commands · Ctrl+C twice quits",
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            frame.render_widget(hint, area);
            return;
        }
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let total = u16::try_from(paragraph.line_count(area.width)).unwrap_or(u16::MAX);
        let max_scroll = total.saturating_sub(area.height);
        self.scroll = self.scroll.min(max_scroll);
        frame.render_widget(paragraph.scroll((max_scroll - self.scroll, 0)), area);
    }

    fn render_approval(&self, frame: &mut Frame<'_>, area: Rect, p: &ApprovalPrompt) {
        let width = area.width.saturating_sub(4).min(100);
        let mut lines = vec![
            Line::from(Span::styled(
                p.summary.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::raw(format!("Risk: {} · ", p.risk)),
                if p.inside {
                    Span::styled("inside workdir ✓", Style::default().fg(Color::Green))
                } else {
                    Span::styled("outside or unverifiable ✗", Style::default().fg(Color::Red))
                },
            ]),
        ];
        for r in &p.reasons {
            lines.push(Line::styled(
                format!("• {r}"),
                Style::default().fg(Color::Yellow),
            ));
        }
        let detail: Vec<Line<'static>> = p
            .detail
            .as_deref()
            .unwrap_or("")
            .lines()
            .skip(usize::from(self.detail_scroll))
            .map(detail_line)
            .collect();
        if !detail.is_empty() {
            lines.push(Line::default());
        }
        let footer = self.approval_footer(p);
        let fixed = lines.len() + footer.len() + 3;
        let room = usize::from(area.height.saturating_sub(2)).saturating_sub(fixed);
        lines.extend(detail.into_iter().take(room));
        lines.push(Line::default());
        lines.extend(footer);

        let height = (lines.len() as u16 + 2).min(area.height);
        let rect = Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        };
        let task = p
            .task
            .as_ref()
            .map(|t| format!(" ({t})"))
            .unwrap_or_default();
        let count = if self.approvals.len() > 1 {
            format!(" · 1 of {}", self.approvals.len())
        } else {
            String::new()
        };
        let border = if p.inside { Color::Yellow } else { Color::Red };
        let block = Block::bordered()
            .border_style(Style::default().fg(border))
            .title(format!(
                " Approval needed ─ {}{task}{count} ",
                p.agent_label
            ));
        frame.render_widget(Clear, rect);
        frame.render_widget(
            Paragraph::new(lines)
                .style(MODAL)
                .block(block)
                .wrap(Wrap { trim: false }),
            rect,
        );
    }

    fn approval_footer(&self, p: &ApprovalPrompt) -> Vec<Line<'static>> {
        if let Some(note) = &self.note {
            return vec![
                Line::from(vec![
                    Span::raw("Note: "),
                    Span::raw(note.text.clone()),
                    Span::raw("▌"),
                ]),
                Line::styled(
                    "Enter send denial · Esc back",
                    Style::default().fg(Color::Gray),
                ),
            ];
        }
        let key = |k: &str, label: &str, default: bool| {
            let style = if default {
                Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            } else {
                Style::default()
            };
            vec![
                Span::styled(format!("[{k}]"), Style::default().fg(Color::Cyan)),
                Span::styled(format!(" {label}  "), style),
            ]
        };
        let mut spans = key("y", "allow once", !p.default_deny);
        if p.allow_session {
            spans.extend(key("a", "allow for session", false));
        }
        spans.extend(key("n", "deny", p.default_deny));
        spans.extend(key("e", "deny + note", false));
        vec![Line::from(spans)]
    }
}

fn detail_line(l: &str) -> Line<'static> {
    let style = if l.starts_with("+++") || l.starts_with("---") {
        Style::default().fg(Color::Gray)
    } else if l.starts_with('+') {
        Style::default().fg(Color::Green)
    } else if l.starts_with('-') {
        Style::default().fg(Color::Red)
    } else if l.starts_with("@@") {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };
    Line::styled(l.to_owned(), style)
}

fn mode_style(mode: Mode) -> Style {
    match mode {
        Mode::Plan => Style::default().fg(Color::LightBlue),
        Mode::Cowork => Style::default().fg(Color::White),
        Mode::AutoEdit => Style::default().fg(Color::LightYellow),
        Mode::Auto => Style::default().fg(Color::LightRed).bold(),
    }
}

fn task_icon(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Pending => "·",
        TaskStatus::Queued => "…",
        TaskStatus::Running => "⟳",
        TaskStatus::AwaitingInput => "?",
        TaskStatus::Success => "✓",
        TaskStatus::Partial => "◐",
        TaskStatus::Failed => "✗",
        TaskStatus::Cancelled => "⊘",
        TaskStatus::Blocked => "⊗",
    }
}

fn task_color(status: TaskStatus) -> Color {
    match status {
        TaskStatus::Pending | TaskStatus::Queued => Color::Gray,
        TaskStatus::Running => Color::Cyan,
        TaskStatus::AwaitingInput | TaskStatus::Partial => Color::Yellow,
        TaskStatus::Success => Color::Green,
        TaskStatus::Failed | TaskStatus::Blocked => Color::Red,
        TaskStatus::Cancelled => Color::DarkGray,
    }
}

fn task_finished(status: TaskStatus) -> bool {
    matches!(
        status,
        TaskStatus::Success | TaskStatus::Partial | TaskStatus::Failed | TaskStatus::Cancelled
    )
}

fn phase_color(phase: &str) -> Color {
    match phase {
        "idle" => Color::DarkGray,
        "awaiting approval" | "awaiting answer" | "waiting for tasks" => Color::Yellow,
        _ => Color::Green,
    }
}

fn render_completion(
    frame: &mut Frame<'_>,
    chat: Rect,
    matches: &[slash::Completion],
    selected: usize,
) {
    let height = (matches.len() as u16 + 2).min(chat.height);
    let width = chat.width.min(70);
    let rect = Rect {
        x: chat.x,
        y: chat.y + chat.height - height,
        width,
        height,
    };
    let lines: Vec<Line<'static>> = matches
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let style = if i == selected {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(vec![
                Span::styled(format!(" /{} {}", c.name, c.args), style.fg(Color::Cyan)),
                Span::styled(format!("  {}", c.help), style.fg(Color::Gray)),
            ])
        })
        .collect();
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" Tab complete · ↑↓ select ")),
        rect,
    );
}

fn render_picker(frame: &mut Frame<'_>, area: Rect, p: &PickerPrompt) {
    let width = area.width.saturating_sub(4).min(100);
    let inner = usize::from(width.saturating_sub(2));
    let rows = p
        .items
        .len()
        .min(MAX_PICKER_ROWS)
        .min(usize::from(area.height.saturating_sub(2)).max(1));
    let start = (p.selected + 1).saturating_sub(rows);
    let lines: Vec<Line<'static>> = p
        .items
        .iter()
        .enumerate()
        .skip(start)
        .take(rows)
        .map(|(i, item)| {
            let chosen = i == p.selected;
            let marker = if chosen { "› " } else { "  " };
            let mut line = Line::from(vec![
                Span::styled(
                    format!("{marker}{}", item.label),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    if item.detail.is_empty() {
                        String::new()
                    } else {
                        format!("  {}", item.detail)
                    },
                    Style::default().fg(Color::Gray),
                ),
            ]);
            if chosen {
                let pad = inner.saturating_sub(line.width());
                line.spans.push(Span::raw(" ".repeat(pad)));
                line = line.style(Style::default().add_modifier(Modifier::REVERSED));
            }
            line
        })
        .collect();
    let height = (rows as u16 + 2).min(area.height);
    let rect = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    let block = Block::bordered()
        .border_style(Style::default().fg(Color::Cyan))
        .title(format!(
            " {} ─ {} of {} ",
            p.title,
            p.selected + 1,
            p.items.len()
        ))
        .title_bottom(Line::from(" ↑↓ select · Enter choose · Esc cancel ").right_aligned());
    frame.render_widget(Clear, rect);
    frame.render_widget(Paragraph::new(lines).style(MODAL).block(block), rect);
}

fn help_text(skills: &[SkillInfo]) -> String {
    let mut out = String::from("commands:");
    for c in COMMANDS {
        out.push_str(&format!("\n  /{} {} — {}", c.name, c.args, c.help));
    }
    if !skills.is_empty() {
        out.push_str("\nskills (run with /<name> [args]):");
        for s in skills {
            out.push_str(&format!("\n  /{} — {}", s.name, s.description));
        }
    }
    out.push_str(
        "\nkeys: Enter send · Shift/Alt+Enter newline · ↑↓ history · PgUp/PgDn scroll · Ctrl+O details · \
         Tab agents panel · Ctrl+C cancel (twice to quit)",
    );
    out
}
