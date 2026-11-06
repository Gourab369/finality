//! Terminal UI (DESIGN §11): chat, status bar, approvals and slash commands.
//! Talks to the core only through `UiEvent` / `CoreCommand`.

pub mod app;
pub mod format;
pub mod markdown;
pub mod slash;
pub mod terminal;

pub use app::App;
pub use terminal::{restore_terminal, run};
