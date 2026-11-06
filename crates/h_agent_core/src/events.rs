//! Channel messages between the UI and the core. Variants grow as later phases land.

use crate::{AgentId, Mode, ProviderKind, RiskClass, Role, SessionId, TaskStatus, ThinkLevel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Info,
    Warning,
    Error,
}

/// Header information shown by the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusInfo {
    pub workdir: String,
    pub session: Option<SessionId>,
    pub mode: Mode,
    pub provider: ProviderKind,
    pub model: String,
    pub think: ThinkLevel,
}

/// Token accounting for the status bar (DESIGN §11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UsageSnapshot {
    pub last_input: u32,
    pub last_output: u32,
    pub ctx_used: u32,
    pub ctx_total: u32,
    pub session_in: u64,
    pub session_out: u64,
    /// Deferred tools currently loaded for the Orchestrator.
    pub tools_loaded: u32,
    /// MCP servers with a live connection.
    pub mcp_running: u32,
}

/// An approval the user must answer (DESIGN §11.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalPrompt {
    pub id: u64,
    pub agent_label: String,
    pub task: Option<String>,
    pub tool: String,
    pub risk: RiskClass,
    pub summary: String,
    pub detail: Option<String>,
    pub reasons: Vec<String>,
    /// False for outside or unverifiable actions.
    pub inside: bool,
    pub allow_session: bool,
    pub default_deny: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalChoice {
    AllowOnce,
    AllowSession,
    Deny { note: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Succeeded,
    Failed,
    Denied,
}

/// Core → UI.
#[derive(Debug, Clone, PartialEq)]
pub enum UiEvent {
    Notice {
        level: NoticeLevel,
        text: String,
    },
    Status(StatusInfo),
    SessionStarted {
        id: SessionId,
    },
    SessionClosed,
    /// A user prompt replayed from a resumed session.
    UserMessage {
        text: String,
    },
    AssistantDelta {
        agent: AgentId,
        text: String,
    },
    ThinkingDelta {
        agent: AgentId,
        text: String,
    },
    ToolStarted {
        call_id: String,
        tool: String,
        summary: String,
    },
    ToolFinished {
        call_id: String,
        status: ToolStatus,
        output: String,
    },
    Approval(ApprovalPrompt),
    /// An approval was answered or withdrawn elsewhere (e.g. the turn was cancelled).
    ApprovalClosed {
        id: u64,
    },
    Usage(UsageSnapshot),
    ModeChanged {
        mode: Mode,
    },
    TurnStarted,
    TurnFinished {
        cancelled: bool,
    },
    /// Agents panel rows, sent at most twice a second.
    AgentStatus(Vec<AgentLine>),
    /// `None` clears the tasks strip.
    PlanUpdated(Option<PlanView>),
    Question(QuestionPrompt),
    QuestionClosed {
        id: u64,
    },
    /// Active skills, for slash completion and `/<skill-name>`.
    Skills(Vec<SkillInfo>),
    /// The decision model was switched on (after a passing probe) or off.
    DecideChanged {
        on: bool,
    },
    /// A list to choose one entry from (`/resume` without an id, the model at start-up).
    Picker(PickerPrompt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerKind {
    /// Values are session ids.
    Session,
    /// Values are model names for the current provider.
    Model,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerItem {
    pub value: String,
    pub label: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerPrompt {
    pub kind: PickerKind,
    pub title: String,
    pub items: Vec<PickerItem>,
    /// Index of the entry selected when the list opens.
    pub selected: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
}

/// One agent in the agents panel (DESIGN §6.3, §11.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentLine {
    pub label: String,
    pub role: Role,
    pub phase: String,
    pub task: Option<u32>,
    pub elapsed_secs: u64,
    pub ctx_used: u32,
    pub ctx_total: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskLine {
    pub number: u32,
    pub title: String,
    pub role: String,
    pub status: TaskStatus,
    pub attempt: u32,
    pub max_attempts: u32,
    /// Latest outcome reason or summary.
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanView {
    pub goal: String,
    pub tasks: Vec<TaskLine>,
}

/// A question for the user from `ask_user`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionPrompt {
    pub id: u64,
    pub agent_label: String,
    pub text: String,
    /// Suggested answers; the user may also type free text.
    pub options: Vec<String>,
}

/// Parsed slash command (DESIGN §11.5). `/help` is handled by the UI itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    New,
    Sessions {
        all: bool,
    },
    /// `/sessions prune <days>`: delete closed sessions older than that.
    SessionsPrune {
        days: u32,
    },
    Resume {
        id: Option<String>,
    },
    /// Internal follow-up after the user confirmed resuming a session recorded elsewhere.
    ResumeHere {
        id: String,
    },
    Mode(Mode),
    Think(ThinkLevel),
    Provider {
        provider: ProviderKind,
        model: Option<String>,
    },
    Model(String),
    /// `/model` without a name: pick one of the provider's models from a list.
    ChooseModel,
    Context,
    Subagents(u32),
    /// `/memory` shows the digest; `/memory recall <query>` searches past sessions.
    Memory {
        recall: Option<String>,
    },
    /// `/memory review`: keep or drop profile notes the Learner proposed.
    ProfileReview,
    /// `/skill <name> [args]` or `/<name> [args]`.
    Skill {
        name: String,
        args: String,
    },
    /// `/skills` lists skills; `/skills review` walks through pending ones.
    Skills {
        review: bool,
    },
    Tools,
    /// `/knowledge` lists topics; `/knowledge <topic>` shows an entry.
    Knowledge {
        topic: Option<String>,
    },
    Mcp(McpAction),
    /// `/agent list|show <name>|delete <name>` (DESIGN §6.5).
    Agent(AgentAction),
    /// `/decide on|off` (DESIGN §10).
    Decide(bool),
}

/// Custom agent management.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAction {
    List,
    Show(String),
    Delete(String),
}

/// `/mcp list|login <server>|enable|disable` (DESIGN §9.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpAction {
    List,
    Login(String),
    Enable,
    Disable,
}

/// UI → core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreCommand {
    UserPrompt(String),
    Slash(SlashCommand),
    ApprovalResponse {
        id: u64,
        choice: ApprovalChoice,
    },
    QuestionAnswer {
        id: u64,
        answer: String,
    },
    /// The entry chosen in a picker; `None` when it was dismissed.
    Picked {
        kind: PickerKind,
        value: Option<String>,
    },
    CancelTurn,
    Quit,
}
