use finality_core::{AgentId, ApprovalDecision, Mode, Role, TaskId, TaskStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordEnvelope {
    pub seq: u64,
    pub ts_unix_ms: i64,
    pub agent: AgentId,
    pub body: Record,
}

/// One session log entry (DESIGN §5.2).
/// postcard encodes the variant index and fields positionally: append new variants at the end only,
/// and bump `hlog::FORMAT_VERSION` with a migration reader for any field change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Record {
    SessionStart {
        workdir: String,
        provider: String,
        model: String,
        mode: Mode,
    },
    UserMessage {
        text: String,
    },
    AssistantMessage {
        text: String,
        thinking: Option<String>,
    },
    ToolCall {
        call_id: String,
        tool: String,
        args_json: String,
    },
    ToolResult {
        call_id: String,
        output: String,
        is_error: bool,
        truncated: bool,
    },
    Approval {
        call_id: String,
        decision: ApprovalDecision,
        reasons: Vec<String>,
        note: Option<String>,
    },
    AgentSpawn {
        child: AgentId,
        role: Role,
        task_id: Option<TaskId>,
    },
    AgentMessage {
        from: AgentId,
        to: AgentId,
        msg_json: String,
    },
    PlanUpdate {
        plan_json: String,
    },
    TaskUpdate {
        task_id: TaskId,
        status: TaskStatus,
        note: String,
    },
    Usage {
        input_tokens: u32,
        output_tokens: u32,
        ctx_used: u32,
        ctx_total: u32,
    },
    ModeChange {
        mode: Mode,
    },
    ProviderChange {
        provider: String,
        model: String,
    },
    SessionResumed {
        workdir: String,
    },
    SessionEnd {
        reason: String,
    },
    /// A user message delivered into a running turn; not an exchange boundary.
    UserInterjection {
        text: String,
    },
}

impl Record {
    pub fn kind(&self) -> &'static str {
        match self {
            Record::SessionStart { .. } => "SessionStart",
            Record::UserMessage { .. } => "UserMessage",
            Record::AssistantMessage { .. } => "AssistantMessage",
            Record::ToolCall { .. } => "ToolCall",
            Record::ToolResult { .. } => "ToolResult",
            Record::Approval { .. } => "Approval",
            Record::AgentSpawn { .. } => "AgentSpawn",
            Record::AgentMessage { .. } => "AgentMessage",
            Record::PlanUpdate { .. } => "PlanUpdate",
            Record::TaskUpdate { .. } => "TaskUpdate",
            Record::Usage { .. } => "Usage",
            Record::ModeChange { .. } => "ModeChange",
            Record::ProviderChange { .. } => "ProviderChange",
            Record::SessionResumed { .. } => "SessionResumed",
            Record::SessionEnd { .. } => "SessionEnd",
            Record::UserInterjection { .. } => "UserInterjection",
        }
    }
}
