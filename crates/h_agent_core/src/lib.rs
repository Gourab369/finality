//! Shared types for all h_agent crates: ids, enums, config, UI/core messages and the workspace.

pub mod config;
pub mod events;
pub mod ids;
pub mod injection;
pub mod secrets;
pub mod text;
pub mod types;
pub mod workspace;

pub use ids::{AgentId, EventId, PlanId, SessionId, TaskId};
pub use types::{
    ApprovalDecision, Mode, ParseEnumError, ProviderKind, RiskClass, Role, TaskStatus, ThinkLevel,
};
pub use workspace::Workspace;

/// Product name shown to users and models; code, env vars and data paths still use `h_agent`.
pub const APP_NAME: &str = "Finality";
