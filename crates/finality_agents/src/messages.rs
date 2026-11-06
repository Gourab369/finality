//! Orchestrator ↔ sub-agent messages (DESIGN §6.2). Sub-agents run in-process, so these are the
//! logged form (`Record::AgentMessage`) of calls made through the supervisor.

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToSubAgent {
    Assign {
        task: u32,
        attempt: u32,
        strategy: String,
        instructions: String,
        allowed_tools: Vec<String>,
    },
    Answer {
        task: u32,
        text: String,
    },
    Cancel {
        task: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FromSubAgent {
    NeedInput {
        task: u32,
        question: String,
    },
    Result {
        task: u32,
        outcome: String,
        summary: String,
        evidence: Vec<String>,
    },
}
