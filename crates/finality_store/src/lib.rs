//! Persistence primitives: data directory layout, atomic writes, cross-process locks, session logs and indexes.

pub mod atomic;
pub mod counter;
pub mod error;
pub mod hlog;
pub mod index;
pub mod paths;
pub mod record;
pub mod session;
pub mod session_log;

pub use atomic::{LockError, with_lock, write_atomic};
pub use counter::{SessionLease, allocate_session_id};
pub use error::StoreError;
pub use hlog::{LogContents, LogError, LogWriter, Tail, read_log};
pub use index::{McpTrust, ProjectEntry, ProjectIndex, SessionEntry, SessionIndex};
pub use paths::DataDir;
pub use record::{Record, RecordEnvelope};
pub use session::{ActiveSession, OrphanedSession, SessionStart, SessionStore};
pub use session_log::SessionLog;
