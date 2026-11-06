//! Agent engine, orchestrator core, triage and (later) the sub-agent runtime.

pub mod agent_tools;
pub mod approver;
pub mod control;
pub mod conversation;
pub mod core;
pub mod decision;
pub mod engine;
pub mod gate;
pub mod heartbeat;
pub mod knowledge_tools;
pub mod learner;
pub mod lookup_tools;
pub mod memory;
pub mod memory_tools;
pub mod messages;
pub mod plan;
pub mod questions;
pub mod scorer;
pub mod skill_tools;
pub mod subagent;
pub mod supervisor;
pub mod triage;

pub use crate::core::{Core, CoreDeps};
pub use approver::UiApprover;
pub use conversation::Conversation;
pub use engine::{Engine, EngineSettings, TokenTotals, TurnEnd, TurnSink};
pub use memory::{MemoryAgent, MemoryDeps, MemoryJob, RecallHit, SessionCtx};
pub use plan::{Outcome, Plan, StepSpec, TaskRole};
pub use supervisor::Supervisor;
pub use triage::{Complexity, Triage};
