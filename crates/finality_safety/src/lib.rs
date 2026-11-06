//! Safety core (DESIGN §12): permission engine, path guard, shell command analyzer and network guard.

pub mod command;
pub mod network;
pub mod path;
pub mod permission;
pub mod shell;

pub use command::{Category, CommandAnalysis, CommandAnalyzer, EnvSnapshot, Finding, Location};
pub use network::{AddrClass, NetVerdict, NetworkGuard, classify_ip};
pub use path::{PathGuard, PathVerdict};
pub use permission::{ActionRequest, AskInfo, CommandShape, Decision, PermissionEngine};
pub use shell::ShellKind;
