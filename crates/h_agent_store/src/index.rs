//! `sessions/index.jsonl` and `projects.json` (DESIGN §4.5, §5.1.3).

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDate, Utc};
use h_agent_core::SessionId;
use serde::{Deserialize, Serialize};

use crate::StoreError;
use crate::atomic::{with_lock, write_atomic};

const INDEX_LOCK: &str = "session-index";
const PROJECTS_LOCK: &str = "projects";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: SessionId,
    pub project_id: String,
    pub workdir: String,
    pub log: String,
    pub created_at: DateTime<Utc>,
    /// Local dates the session was active on; grows when resumed on a new day.
    pub dates: Vec<NaiveDate>,
    pub closed_at: Option<DateTime<Utc>>,
    /// One-line summary written by the Memory agent.
    #[serde(default)]
    pub title: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub recovered: bool,
}

#[derive(Debug, Clone)]
pub struct SessionIndex {
    path: PathBuf,
    locks_dir: PathBuf,
}

impl SessionIndex {
    pub fn new(path: PathBuf, locks_dir: PathBuf) -> Self {
        Self { path, locks_dir }
    }

    pub fn append(&self, entry: &SessionEntry) -> Result<(), StoreError> {
        let line = to_json_line(entry, &self.path)?;
        with_lock(&self.locks_dir, INDEX_LOCK, || {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            f.write_all(line.as_bytes())?;
            Ok::<_, StoreError>(())
        })?;
        Ok(())
    }

    /// Unparseable lines (e.g. a torn final line) are skipped with a warning.
    pub fn list(&self) -> Result<Vec<SessionEntry>, StoreError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        Ok(text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .filter_map(|(n, l)| match serde_json::from_str(l) {
                Ok(e) => Some(e),
                Err(err) => {
                    tracing::warn!(line = n + 1, error = %err, "skipping bad session index line");
                    None
                }
            })
            .collect())
    }

    pub fn get(&self, id: &SessionId) -> Result<Option<SessionEntry>, StoreError> {
        Ok(self.list()?.into_iter().find(|e| e.id == *id))
    }

    pub fn update(
        &self,
        id: &SessionId,
        f: impl FnOnce(&mut SessionEntry),
    ) -> Result<SessionEntry, StoreError> {
        with_lock(&self.locks_dir, INDEX_LOCK, || {
            let mut entries = self.list()?;
            let entry = entries
                .iter_mut()
                .find(|e| e.id == *id)
                .ok_or_else(|| StoreError::SessionNotFound(id.to_string()))?;
            f(entry);
            let updated = entry.clone();
            let mut text = String::new();
            for e in &entries {
                text.push_str(&to_json_line(e, &self.path)?);
            }
            write_atomic(&self.path, text.as_bytes())?;
            Ok(updated)
        })
        .map_err(StoreError::from)
    }

    /// Drops the entries `keep` rejects and returns them.
    pub fn retain(
        &self,
        keep: impl Fn(&SessionEntry) -> bool,
    ) -> Result<Vec<SessionEntry>, StoreError> {
        with_lock(&self.locks_dir, INDEX_LOCK, || {
            let entries = self.list()?;
            let (kept, dropped): (Vec<_>, Vec<_>) = entries.into_iter().partition(|e| keep(e));
            if !dropped.is_empty() {
                let mut text = String::new();
                for e in &kept {
                    text.push_str(&to_json_line(e, &self.path)?);
                }
                write_atomic(&self.path, text.as_bytes())?;
            }
            Ok(dropped)
        })
        .map_err(StoreError::from)
    }
}

fn to_json_line<T: Serialize>(value: &T, file: &Path) -> Result<String, StoreError> {
    let mut line = serde_json::to_string(value).map_err(|source| StoreError::Json {
        file: file.display().to_string(),
        source,
    })?;
    line.push('\n');
    Ok(line)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub project_id: String,
    pub workdir: String,
    #[serde(default)]
    pub session_ids: Vec<SessionId>,
    /// Trust decisions for the project's `.mcp.json` servers, by server name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp_trust: BTreeMap<String, McpTrust>,
}

/// A decision holds only while the server's configuration keeps the same fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpTrust {
    pub fingerprint: String,
    pub trusted: bool,
}

/// Maps canonical working directories to stable project ids.
#[derive(Debug, Clone)]
pub struct ProjectIndex {
    path: PathBuf,
    locks_dir: PathBuf,
}

impl ProjectIndex {
    pub fn new(path: PathBuf, locks_dir: PathBuf) -> Self {
        Self { path, locks_dir }
    }

    /// Registers `workdir` (and optionally a session) and returns its project id.
    pub fn register(
        &self,
        workdir: &Path,
        session: Option<SessionId>,
    ) -> Result<String, StoreError> {
        self.update(workdir, |entry| {
            if let Some(id) = session
                && !entry.session_ids.contains(&id)
            {
                entry.session_ids.push(id);
            }
        })
    }

    pub fn set_mcp_trust(
        &self,
        workdir: &Path,
        server: &str,
        trust: McpTrust,
    ) -> Result<(), StoreError> {
        self.update(workdir, |entry| {
            entry.mcp_trust.insert(server.to_owned(), trust);
        })
        .map(drop)
    }

    fn update(
        &self,
        workdir: &Path,
        change: impl FnOnce(&mut ProjectEntry),
    ) -> Result<String, StoreError> {
        let key = workdir_key(workdir);
        with_lock(&self.locks_dir, PROJECTS_LOCK, || {
            let mut map = self.read()?;
            let entry = map.entry(key.clone()).or_insert_with(|| ProjectEntry {
                project_id: project_id_for(workdir),
                workdir: workdir.display().to_string(),
                session_ids: Vec::new(),
                mcp_trust: BTreeMap::new(),
            });
            change(entry);
            let project_id = entry.project_id.clone();
            let text = serde_json::to_vec_pretty(&map).map_err(|source| StoreError::Json {
                file: self.path.display().to_string(),
                source,
            })?;
            write_atomic(&self.path, &text)?;
            Ok(project_id)
        })
        .map_err(StoreError::from)
    }

    pub fn get(&self, workdir: &Path) -> Result<Option<ProjectEntry>, StoreError> {
        Ok(self.read()?.remove(&workdir_key(workdir)))
    }

    fn read(&self) -> Result<BTreeMap<String, ProjectEntry>, StoreError> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|source| StoreError::Json {
                file: self.path.display().to_string(),
                source,
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(e) => Err(e.into()),
        }
    }
}

/// Windows paths compare case-insensitively.
fn workdir_key(workdir: &Path) -> String {
    let s = workdir.display().to_string();
    if cfg!(windows) { s.to_lowercase() } else { s }
}

/// Stable 64-bit FNV-1a of the normalized workdir, as 16 hex chars.
pub fn project_id_for(workdir: &Path) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in workdir_key(workdir).bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn entry(serial: u64) -> SessionEntry {
        let started = NaiveDate::from_ymd_opt(2026, 11, 5)
            .unwrap()
            .and_hms_opt(9, 0, 0)
            .unwrap();
        let id = SessionId::new(serial, started);
        SessionEntry {
            id,
            project_id: "p".into(),
            workdir: "/w".into(),
            log: id.log_file_name(),
            created_at: Utc.with_ymd_and_hms(2026, 11, 5, 9, 0, 0).unwrap(),
            dates: vec![started.date()],
            closed_at: None,
            title: String::new(),
            provider: "ollama".into(),
            model: "m".into(),
            recovered: false,
        }
    }

    #[test]
    fn append_list_update() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = SessionIndex::new(tmp.path().join("index.jsonl"), tmp.path().join("locks"));
        assert!(idx.list().unwrap().is_empty());
        idx.append(&entry(1)).unwrap();
        idx.append(&entry(2)).unwrap();
        let updated = idx
            .update(&entry(2).id, |e| e.title = "Fixed CRC bug".into())
            .unwrap();
        assert_eq!(updated.title, "Fixed CRC bug");
        let all = idx.list().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].title, "Fixed CRC bug");
        assert!(matches!(
            idx.update(&entry(9).id, |_| {}),
            Err(StoreError::SessionNotFound(_))
        ));
    }

    #[test]
    fn torn_last_line_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.jsonl");
        let idx = SessionIndex::new(path.clone(), tmp.path().join("locks"));
        idx.append(&entry(1)).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"{\"id\":\"0000").unwrap();
        assert_eq!(idx.list().unwrap().len(), 1);
    }

    #[test]
    fn projects_register_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let projects =
            ProjectIndex::new(tmp.path().join("projects.json"), tmp.path().join("locks"));
        let w = tmp.path().join("work");
        let id1 = projects.register(&w, Some(entry(1).id)).unwrap();
        let id2 = projects.register(&w, Some(entry(1).id)).unwrap();
        projects.register(&w, Some(entry(2).id)).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 16);
        let p = projects.get(&w).unwrap().unwrap();
        assert_eq!(p.session_ids.len(), 2);
        assert_ne!(project_id_for(&tmp.path().join("other")), id1);

        let trust = McpTrust {
            fingerprint: "abc".into(),
            trusted: true,
        };
        projects.set_mcp_trust(&w, "github", trust.clone()).unwrap();
        let p = projects.get(&w).unwrap().unwrap();
        assert_eq!(p.mcp_trust["github"], trust);
        assert_eq!(p.session_ids.len(), 2, "other fields are kept");
    }
}
