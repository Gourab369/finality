//! Session lifecycle on top of the counter, logs, index and leases (DESIGN §5.1).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Local, Utc};
use finality_core::{AgentId, Mode, SessionId};

use crate::counter::{SessionLease, allocate_session_id};
use crate::hlog::{LogContents, LogWriter};
use crate::index::{ProjectIndex, SessionEntry, SessionIndex};
use crate::record::Record;
use crate::session_log::SessionLog;
use crate::{DataDir, StoreError};

pub struct SessionStart<'a> {
    pub workdir: &'a Path,
    pub provider: &'a str,
    pub model: &'a str,
    pub mode: Mode,
}

/// An open session owned by this process. Dropping it without [`SessionStore::close`] leaves it
/// unclosed in the index, so the next start treats it as crashed.
pub struct ActiveSession {
    pub id: SessionId,
    pub project_id: String,
    pub log: Arc<SessionLog>,
    log_path: PathBuf,
    _lease: SessionLease,
}

impl ActiveSession {
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }
}

/// A crashed session whose lease this process now holds.
pub struct OrphanedSession {
    pub entry: SessionEntry,
    lease: SessionLease,
}

pub struct SessionStore {
    dir: DataDir,
    index: SessionIndex,
    projects: ProjectIndex,
}

impl SessionStore {
    pub fn new(dir: DataDir) -> Self {
        let index = SessionIndex::new(dir.session_index_file(), dir.locks_dir());
        let projects = ProjectIndex::new(dir.projects_file(), dir.locks_dir());
        Self {
            dir,
            index,
            projects,
        }
    }

    pub fn index(&self) -> &SessionIndex {
        &self.index
    }

    pub fn projects(&self) -> &ProjectIndex {
        &self.projects
    }

    pub fn log_path(&self, id: &SessionId) -> PathBuf {
        self.dir.sessions_dir().join(id.log_file_name())
    }

    pub fn start(
        &self,
        meta: SessionStart<'_>,
        agent: AgentId,
    ) -> Result<ActiveSession, StoreError> {
        let now = Local::now();
        let id = allocate_session_id(&self.dir, now.naive_local())?;
        let lease = SessionLease::try_acquire(&self.dir.locks_dir(), &id)?
            .ok_or_else(|| StoreError::SessionInUse(id.to_string()))?;
        let project_id = self.projects.register(meta.workdir, Some(id))?;
        let log_path = self.log_path(&id);
        let log = Arc::new(SessionLog::spawn(LogWriter::create(&log_path, &id)?)?);
        let workdir = meta.workdir.display().to_string();
        log.append(
            agent,
            Record::SessionStart {
                workdir: workdir.clone(),
                provider: meta.provider.to_owned(),
                model: meta.model.to_owned(),
                mode: meta.mode,
            },
        )?;
        self.index.append(&SessionEntry {
            id,
            project_id: project_id.clone(),
            workdir,
            log: id.log_file_name(),
            created_at: now.with_timezone(&Utc),
            dates: vec![now.date_naive()],
            closed_at: None,
            title: String::new(),
            provider: meta.provider.to_owned(),
            model: meta.model.to_owned(),
            recovered: false,
        })?;
        Ok(ActiveSession {
            id,
            project_id,
            log,
            log_path,
            _lease: lease,
        })
    }

    pub async fn close(
        &self,
        session: ActiveSession,
        agent: AgentId,
        reason: &str,
    ) -> Result<(), StoreError> {
        session.log.append(
            agent,
            Record::SessionEnd {
                reason: reason.to_owned(),
            },
        )?;
        session.log.close().await?;
        self.index
            .update(&session.id, |e| e.closed_at = Some(Utc::now()))?;
        Ok(())
    }

    /// Reopens a past session for appending. Returns the existing records for context replay.
    pub fn resume(
        &self,
        id: &SessionId,
        workdir: &Path,
        agent: AgentId,
    ) -> Result<(ActiveSession, LogContents), StoreError> {
        let entry = self
            .index
            .get(id)?
            .ok_or_else(|| StoreError::SessionNotFound(id.to_string()))?;
        let lease = SessionLease::try_acquire(&self.dir.locks_dir(), id)?
            .ok_or_else(|| StoreError::SessionInUse(id.to_string()))?;
        let log_path = self.log_path(id);
        let (writer, contents) = LogWriter::open_append(&log_path)?;
        let log = Arc::new(SessionLog::spawn(writer)?);
        log.append(
            agent,
            Record::SessionResumed {
                workdir: workdir.display().to_string(),
            },
        )?;
        let today = Local::now().date_naive();
        self.index.update(id, |e| {
            e.closed_at = None;
            if !e.dates.contains(&today) {
                e.dates.push(today);
            }
        })?;
        Ok((
            ActiveSession {
                id: *id,
                project_id: entry.project_id,
                log,
                log_path,
                _lease: lease,
            },
            contents,
        ))
    }

    /// Sessions for one project (or all), newest first.
    pub fn list(&self, project_id: Option<&str>) -> Result<Vec<SessionEntry>, StoreError> {
        let mut all: Vec<_> = self
            .index
            .list()?
            .into_iter()
            .filter(|e| project_id.is_none_or(|p| e.project_id == p))
            .collect();
        all.sort_by_key(|e| std::cmp::Reverse(e.id.serial));
        Ok(all)
    }

    /// Deletes closed sessions (log, index entry, saved plans) that ended before `cutoff`, except
    /// any still held by a process. Returns the removed ids.
    pub fn prune(&self, cutoff: DateTime<Utc>) -> Result<Vec<SessionId>, StoreError> {
        let locks = self.dir.locks_dir();
        let dropped = self.index.retain(|e| {
            let old = e.closed_at.is_some_and(|c| c < cutoff);
            if !old {
                return true;
            }
            // A lease that cannot be taken means another process still owns the session.
            !matches!(SessionLease::try_acquire(&locks, &e.id), Ok(Some(_)))
        })?;
        let mut removed = Vec::new();
        for e in dropped {
            let log = self.log_path(&e.id);
            match std::fs::remove_file(&log) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => {
                    tracing::warn!(path = %log.display(), error = %err, "cannot delete session log")
                }
            }
            let plans = self.dir.plans_dir().join(e.id.to_string());
            if plans.is_dir()
                && let Err(err) = std::fs::remove_dir_all(&plans)
            {
                tracing::warn!(path = %plans.display(), error = %err, "cannot delete saved plans");
            }
            removed.push(e.id);
        }
        Ok(removed)
    }

    /// Unclosed sessions of `project_id` that no live process holds. Their leases are taken.
    pub fn find_orphans(&self, project_id: &str) -> Result<Vec<OrphanedSession>, StoreError> {
        let mut orphans = Vec::new();
        for entry in self.list(Some(project_id))? {
            if entry.closed_at.is_some() {
                continue;
            }
            if let Some(lease) = SessionLease::try_acquire(&self.dir.locks_dir(), &entry.id)? {
                orphans.push(OrphanedSession { entry, lease });
            }
        }
        Ok(orphans)
    }

    /// Repairs the log tail, appends `SessionEnd` and marks the session closed and recovered.
    /// Returns the log contents for the Memory agent's finalize job.
    pub fn recover(
        &self,
        orphan: OrphanedSession,
        agent: AgentId,
    ) -> Result<LogContents, StoreError> {
        let path = self.log_path(&orphan.entry.id);
        let (mut writer, contents) = LogWriter::open_append(&path)?;
        writer.append(
            agent,
            Record::SessionEnd {
                reason: "recovered after crash".into(),
            },
        )?;
        writer.sync()?;
        self.index.update(&orphan.entry.id, |e| {
            e.closed_at = Some(Utc::now());
            e.recovered = true;
        })?;
        drop(orphan.lease);
        Ok(contents)
    }
}
