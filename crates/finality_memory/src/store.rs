//! Reading and writing events files under the shared memory lock (DESIGN §7.1).

use std::io;
use std::path::{Path, PathBuf};

use chrono::Utc;
use finality_core::Role;
use finality_store::{DataDir, LockError, with_lock, write_atomic};

use crate::tree::{EventsFile, SCHEMA_VERSION};

const LOCK: &str = "memory";

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("cannot access {file}: {source}")]
    Io { file: String, source: io::Error },
    #[error("cannot parse {file}: {source}")]
    Json {
        file: String,
        source: serde_json::Error,
    },
    #[error("{file} has schema version {found}; this build reads up to {SCHEMA_VERSION}")]
    Schema { file: String, found: u32 },
    #[error("memory lock: {0}")]
    Lock(String),
}

#[derive(Debug, Clone)]
pub struct EventStore {
    path: PathBuf,
    locks_dir: PathBuf,
}

impl EventStore {
    pub fn main(dir: &DataDir) -> Self {
        Self {
            path: dir.main_events_file(),
            locks_dir: dir.locks_dir(),
        }
    }

    pub fn for_role(dir: &DataDir, role: Role) -> Self {
        Self {
            path: dir.memory_agents_dir().join(format!("{role}.events")),
            locks_dir: dir.locks_dir(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn io_err(&self, source: io::Error) -> MemoryError {
        MemoryError::Io {
            file: self.path.display().to_string(),
            source,
        }
    }

    /// A missing file reads as a new, seeded tree.
    pub fn load(&self) -> Result<EventsFile, MemoryError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(EventsFile::new()),
            Err(e) => return Err(self.io_err(e)),
        };
        let file: EventsFile = serde_json::from_str(&text).map_err(|source| MemoryError::Json {
            file: self.path.display().to_string(),
            source,
        })?;
        if file.schema_version > SCHEMA_VERSION {
            return Err(MemoryError::Schema {
                file: self.path.display().to_string(),
                found: file.schema_version,
            });
        }
        Ok(file)
    }

    /// Re-reads the file under the lock, applies `f` and writes it back atomically.
    pub fn update<T>(&self, f: impl FnOnce(&mut EventsFile) -> T) -> Result<T, MemoryError> {
        with_lock(&self.locks_dir, LOCK, || {
            let mut file = self.load()?;
            let out = f(&mut file);
            file.schema_version = SCHEMA_VERSION;
            file.updated_at = Utc::now();
            let bytes = serde_json::to_vec_pretty(&file).map_err(|source| MemoryError::Json {
                file: self.path.display().to_string(),
                source,
            })?;
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| self.io_err(e))?;
            }
            write_atomic(&self.path, &bytes).map_err(|e| self.io_err(e))?;
            Ok(out)
        })
        .map_err(|e| match e {
            LockError::Inner(e) => e,
            other => MemoryError::Lock(other.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::tests::info;

    #[test]
    fn missing_file_is_seeded_and_updates_persist() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let store = EventStore::main(&dir);
        assert!(store.load().unwrap().is_empty());
        store
            .update(|f| {
                f.ensure_ref("question", "physics", &info(1));
            })
            .unwrap();
        assert_eq!(store.load().unwrap().refs(&info(1).session_id).len(), 1);

        let role = EventStore::for_role(&dir, Role::Worker);
        assert!(
            role.path().ends_with("agents/worker.events")
                || role.path().ends_with("agents\\worker.events")
        );
    }

    #[test]
    fn newer_schema_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let store = EventStore::main(&dir);
        std::fs::create_dir_all(store.path().parent().unwrap()).unwrap();
        std::fs::write(
            store.path(),
            r#"{"schema_version":99,"updated_at":"2026-11-05T00:00:00Z","types":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            store.load(),
            Err(MemoryError::Schema { found: 99, .. })
        ));
    }
}
