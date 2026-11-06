//! Tool trait, registry, executor and built-in tools (DESIGN §9).

pub mod active;
pub mod browser;
pub mod edit;
pub mod executor;
pub mod html;
pub mod read;
pub mod registry;
pub mod search;
pub mod shell;
pub mod tool;
pub mod web;

pub use active::{ActiveToolSet, TOOL_SEARCH, ToolSearch, ToolSets};
pub use browser::{BrowserAction, BrowserSession, BrowserTool};
pub use executor::{ApprovalRequest, ApprovalResponse, Approver, Execution, ToolExecutor};
pub use registry::{ToolRegistry, ToolSource};
pub use tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolEnv, ToolError, ToolLimits, ToolOutput, ToolSpec,
    filter_env,
};
pub use web::{WebFetch, WebSearch};
