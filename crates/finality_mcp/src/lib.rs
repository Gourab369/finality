//! MCP client manager that adapts MCP server tools into the tool registry (DESIGN §9.4).

pub mod auth;
pub mod config;
pub mod hub;
pub mod tool;

pub use auth::{KeyringVault, MemoryVault, TokenVault, open_browser};
pub use config::{Origin, ServerDef, Transport};
pub use hub::{HubSettings, McpHub, ServerState, ServerStatus, Trust};
pub use tool::{McpTool, tool_name};
