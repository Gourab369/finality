use std::io;

use crate::atomic::LockError;
use crate::hlog::LogError;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Log(#[from] LogError),
    #[error("invalid JSON in {file}: {source}")]
    Json {
        file: String,
        source: serde_json::Error,
    },
    #[error("corrupt {0}")]
    Corrupt(String),
    #[error("session {0} not found")]
    SessionNotFound(String),
    #[error("session {0} is in use by another Finality process")]
    SessionInUse(String),
    #[error("{0}")]
    Other(String),
}

impl From<LockError<StoreError>> for StoreError {
    fn from(e: LockError<StoreError>) -> Self {
        match e {
            LockError::InvalidName(n) => StoreError::Other(format!("invalid lock name {n}")),
            LockError::Io(e) => StoreError::Io(e),
            LockError::Inner(e) => e,
        }
    }
}
