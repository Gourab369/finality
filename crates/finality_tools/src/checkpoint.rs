//! Checkpoints: the contents of every file an edit touches, saved before a turn's first change to it, so the
//! user (`/undo`) or the agent (`checkpoint_restore`) can roll edits back. Shell commands are not tracked.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use finality_core::RiskClass;
use finality_safety::Location;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, display_path, parse_args,
};

const MANIFEST: &str = "manifest.json";
/// Larger files are not saved; their edits cannot be undone.
const MAX_SAVED_BYTES: u64 = 16 * 1024 * 1024;
const LABEL_CHARS: usize = 80;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedFile {
    pub path: PathBuf,
    /// Workdir-relative path for messages.
    pub display: String,
    /// Blob file in the checkpoint folder; `None` when the file did not exist before the edit.
    pub blob: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: u32,
    /// The request that led to the edits.
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub files: Vec<SavedFile>,
}

#[derive(Debug, Default)]
struct State {
    /// `<root>/<session id>` while a session is open.
    dir: Option<PathBuf>,
    list: Vec<Checkpoint>,
    /// Index in `list` of the checkpoint of the running turn, once it changed something.
    current: Option<usize>,
    label: String,
}

#[derive(Debug)]
pub struct Checkpoints {
    root: PathBuf,
    state: Mutex<State>,
}

fn io_err(e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

impl Checkpoints {
    pub fn new(root: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            root,
            state: Mutex::new(State::default()),
        })
    }

    fn lock(&self) -> std::io::Result<std::sync::MutexGuard<'_, State>> {
        self.state.lock().map_err(io_err)
    }

    /// Switches to `session` (`None` when no session is open), loading checkpoints saved for it earlier.
    pub fn set_session(&self, session: Option<&str>) {
        let Ok(mut st) = self.lock() else { return };
        *st = State::default();
        let Some(id) = session else { return };
        let dir = self.root.join(id);
        let mut list: Vec<Checkpoint> = std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| std::fs::read(e.path().join(MANIFEST)).ok())
                    .filter_map(|b| serde_json::from_slice::<Checkpoint>(&b).ok())
                    .collect()
            })
            .unwrap_or_default();
        list.sort_by_key(|c| c.id);
        st.list = list;
        st.dir = Some(dir);
    }

    /// Starts a turn: its first edit opens a new checkpoint labelled with `label`.
    pub fn begin(&self, label: &str) {
        if let Ok(mut st) = self.lock() {
            st.current = None;
            st.label = label.split_whitespace().collect::<Vec<_>>().join(" ");
            if st.label.chars().count() > LABEL_CHARS {
                st.label = st.label.chars().take(LABEL_CHARS).collect::<String>() + "…";
            }
        }
    }

    /// Saves `path` as it is now, unless this turn's checkpoint already has it. No-op without a session.
    pub fn record(&self, path: &Path, display: &str) -> std::io::Result<()> {
        let mut st = self.lock()?;
        let Some(dir) = st.dir.clone() else {
            return Ok(());
        };
        let index = match st.current {
            Some(i) => i,
            None => {
                let id = st.list.last().map_or(1, |c| c.id + 1);
                let label = if st.label.is_empty() {
                    "edits".to_owned()
                } else {
                    st.label.clone()
                };
                st.list.push(Checkpoint {
                    id,
                    label,
                    created_at: Utc::now(),
                    files: Vec::new(),
                });
                let i = st.list.len() - 1;
                st.current = Some(i);
                i
            }
        };
        if st.list[index].files.iter().any(|f| f.path == path) {
            return Ok(());
        }
        let cp_dir = dir.join(st.list[index].id.to_string());
        std::fs::create_dir_all(&cp_dir)?;
        let blob = match std::fs::metadata(path) {
            Ok(meta) if meta.is_file() && meta.len() > MAX_SAVED_BYTES => {
                return Err(io_err(format!(
                    "{display} is larger than {} MiB; not saved in the checkpoint",
                    MAX_SAVED_BYTES / (1024 * 1024)
                )));
            }
            Ok(meta) if meta.is_file() => {
                let name = format!("{}.bin", st.list[index].files.len());
                std::fs::copy(path, cp_dir.join(&name))?;
                Some(name)
            }
            Ok(_) => return Err(io_err(format!("{display} is not a file"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        st.list[index].files.push(SavedFile {
            path: path.to_owned(),
            display: display.to_owned(),
            blob,
        });
        let bytes = serde_json::to_vec_pretty(&st.list[index]).map_err(io_err)?;
        finality_store::write_atomic(&cp_dir.join(MANIFEST), &bytes)
    }

    /// Checkpoints of the open session, oldest first.
    pub fn list(&self) -> Vec<Checkpoint> {
        self.lock().map(|st| st.list.clone()).unwrap_or_default()
    }

    /// The files restoring to before checkpoint `id` would change, and whether that deletes created files.
    pub fn plan_restore(&self, id: u32) -> Option<(Vec<SavedFile>, bool)> {
        let st = self.lock().ok()?;
        if !st.list.iter().any(|c| c.id == id) {
            return None;
        }
        let mut files: Vec<SavedFile> = Vec::new();
        // The oldest saved state of each file is the one to restore.
        for c in st.list.iter().filter(|c| c.id >= id) {
            for f in &c.files {
                if !files.iter().any(|g| g.path == f.path) {
                    files.push(f.clone());
                }
            }
        }
        let deletes = files.iter().any(|f| f.blob.is_none() && f.path.exists());
        Some((files, deletes))
    }

    /// Puts every file back as it was before checkpoint `id` and drops that checkpoint and later ones.
    /// Returns the restored files' display paths.
    pub fn restore(&self, id: u32) -> std::io::Result<Vec<String>> {
        let mut st = self.lock()?;
        let dir = st.dir.clone().ok_or_else(|| io_err("no session is open"))?;
        if !st.list.iter().any(|c| c.id == id) {
            return Err(io_err(format!("no checkpoint {id}")));
        }
        let mut restored = Vec::new();
        let undone: Vec<Checkpoint> = st.list.iter().filter(|c| c.id >= id).cloned().collect();
        for c in undone.iter().rev() {
            let cp_dir = dir.join(c.id.to_string());
            for f in &c.files {
                match &f.blob {
                    Some(blob) => {
                        if let Some(parent) = f.path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        let bytes = std::fs::read(cp_dir.join(blob))?;
                        finality_store::write_atomic(&f.path, &bytes)?;
                    }
                    None => match std::fs::remove_file(&f.path) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e),
                    },
                }
                if !restored.contains(&f.display) {
                    restored.push(f.display.clone());
                }
            }
        }
        for c in &undone {
            let _ = std::fs::remove_dir_all(dir.join(c.id.to_string()));
        }
        st.list.retain(|c| c.id < id);
        st.current = None;
        restored.sort();
        Ok(restored)
    }

    /// One line per checkpoint, newest first.
    pub fn describe(&self) -> String {
        let list = self.list();
        if list.is_empty() {
            return "no checkpoints in this session".into();
        }
        let mut out = String::from("checkpoints (newest first):");
        for c in list.iter().rev() {
            let files: Vec<&str> = c.files.iter().map(|f| f.display.as_str()).collect();
            out.push_str(&format!(
                "\n  {} · {} · \"{}\" · {}",
                c.id,
                c.created_at
                    .with_timezone(&chrono::Local)
                    .format("%H:%M:%S"),
                c.label,
                files.join(", ")
            ));
        }
        out
    }
}

/// `checkpoint_list`: the session's checkpoints.
pub struct CheckpointList;

#[async_trait]
impl Tool for CheckpointList {
    fn name(&self) -> &str {
        "checkpoint_list"
    }

    fn description(&self) -> &str {
        "List this session's checkpoints: the files edit_file changed in each turn, saved before the change. \
         Use with checkpoint_restore to undo edits. Changes made by shell commands are not tracked."
    }

    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": {}, "additionalProperties": false })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, _args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        Ok(ActionPreview::new("list checkpoints", Location::Inside))
    }

    async fn call(
        &self,
        _args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::ok(match &ctx.env.checkpoints {
            Some(c) => c.describe(),
            None => "checkpoints are not available".to_owned(),
        }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoreArgs {
    checkpoint: u32,
}

/// `checkpoint_restore`: undo the edits of a checkpoint and every later one.
pub struct CheckpointRestore;

#[async_trait]
impl Tool for CheckpointRestore {
    fn name(&self) -> &str {
        "checkpoint_restore"
    }

    fn description(&self) -> &str {
        "Undo edits: put every file back as it was before the given checkpoint (from checkpoint_list). That \
         checkpoint and all later ones are removed. Files created since then are deleted."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "checkpoint": { "type": "integer", "minimum": 1 } },
            "required": ["checkpoint"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Write
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: RestoreArgs = parse_args(args)?;
        let cps = ctx
            .env
            .checkpoints
            .as_ref()
            .ok_or_else(|| ToolError::Failed("checkpoints are not available".into()))?;
        let (files, deletes) = cps
            .plan_restore(a.checkpoint)
            .ok_or_else(|| ToolError::Failed(format!("no checkpoint {}", a.checkpoint)))?;
        let outside = files.iter().any(|f| !ctx.env.guard.contains(&f.path));
        let mut p = ActionPreview::new(
            format!(
                "restore {} file(s) to before checkpoint {}",
                files.len(),
                a.checkpoint
            ),
            if outside {
                Location::Outside
            } else {
                Location::Inside
            },
        );
        p.paths = files.iter().map(|f| f.path.clone()).collect();
        p.deletes = deletes;
        p.detail = Some(
            files
                .iter()
                .map(|f| {
                    let what = if f.blob.is_some() {
                        "restore"
                    } else {
                        "delete"
                    };
                    format!("{what} {}", display_path(&ctx.env, &f.path))
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: RestoreArgs = parse_args(&args)?;
        let cps = ctx
            .env
            .checkpoints
            .as_ref()
            .ok_or_else(|| ToolError::Failed("checkpoints are not available".into()))?;
        let restored = cps.restore(a.checkpoint)?;
        Ok(ToolOutput::ok(format!(
            "restored to before checkpoint {}: {}",
            a.checkpoint,
            restored.join(", ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_undoes_turns_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let work = tmp.path().join("work");
        std::fs::create_dir_all(&work).unwrap();
        let a = work.join("a.txt");
        let b = work.join("b.txt");
        std::fs::write(&a, "a0").unwrap();
        let cps = Checkpoints::new(tmp.path().join("cp"));
        cps.record(&a, "a.txt").unwrap();
        assert!(cps.list().is_empty(), "nothing is saved without a session");

        cps.set_session(Some("s1"));
        cps.begin("first turn");
        cps.record(&a, "a.txt").unwrap();
        std::fs::write(&a, "a1").unwrap();
        cps.record(&a, "a.txt").unwrap();
        std::fs::write(&a, "a1b").unwrap();
        cps.begin("second turn");
        cps.record(&a, "a.txt").unwrap();
        std::fs::write(&a, "a2").unwrap();
        cps.record(&b, "b.txt").unwrap();
        std::fs::write(&b, "new").unwrap();
        let list = cps.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].label, "first turn");
        assert_eq!(list[0].files.len(), 1, "one save per file per turn");

        assert!(cps.plan_restore(2).unwrap().1, "b.txt was created");
        assert_eq!(cps.restore(2).unwrap(), ["a.txt", "b.txt"]);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "a1b");
        assert!(!b.exists());

        // Reloaded from disk, e.g. after a resume.
        cps.set_session(Some("s1"));
        assert_eq!(cps.list().len(), 1);
        cps.restore(1).unwrap();
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "a0");
        assert!(cps.list().is_empty());
        assert!(cps.restore(1).is_err());
    }
}
