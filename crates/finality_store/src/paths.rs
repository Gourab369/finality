use std::io;
use std::path::{Path, PathBuf};

pub const HOME_ENV: &str = "FINALITY_HOME";
const DEFAULT_DIR_NAME: &str = ".finality";

/// Root of all persistent finality state (DESIGN §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataDir {
    root: PathBuf,
}

impl DataDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Precedence: `FINALITY_HOME` > `general.data_dir` > `~/.finality`.
    pub fn resolve(env_home: Option<&str>, configured: Option<&Path>) -> io::Result<Self> {
        if let Some(home) = env_home.filter(|h| !h.trim().is_empty()) {
            return Ok(Self::new(expand_tilde(Path::new(home))?));
        }
        if let Some(dir) = configured {
            return Ok(Self::new(expand_tilde(dir)?));
        }
        Ok(Self::new(home_dir()?.join(DEFAULT_DIR_NAME)))
    }

    /// Creates the directory tree if missing. On Unix the tree is private to the user (0700).
    pub fn ensure(&self) -> io::Result<()> {
        for dir in [
            self.root.clone(),
            self.sessions_dir(),
            self.memory_dir(),
            self.memory_agents_dir(),
            self.memory_tmp_dir(),
            self.lookup_dir(),
            self.plans_dir(),
            self.skills_dir(),
            self.pending_skills_dir(),
            self.knowledgepedia_dir(),
            self.locks_dir(),
            self.logs_dir(),
        ] {
            std::fs::create_dir_all(&dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn config_file(&self) -> PathBuf {
        self.root.join("config.toml")
    }
    pub fn mcp_file(&self) -> PathBuf {
        self.root.join("mcp.json")
    }
    /// Settings changed from the UI that outlive a session (e.g. direct mode).
    pub fn state_file(&self) -> PathBuf {
        self.root.join("state.json")
    }
    /// Tool lists of MCP servers from earlier runs.
    pub fn mcp_cache_file(&self) -> PathBuf {
        self.root.join("mcp").join("tools-cache.json")
    }
    pub fn projects_file(&self) -> PathBuf {
        self.root.join("projects.json")
    }
    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }
    pub fn session_counter_file(&self) -> PathBuf {
        self.sessions_dir().join("counter")
    }
    pub fn session_index_file(&self) -> PathBuf {
        self.sessions_dir().join("index.jsonl")
    }
    pub fn memory_dir(&self) -> PathBuf {
        self.root.join("memory")
    }
    pub fn main_events_file(&self) -> PathBuf {
        self.memory_dir().join("main.events")
    }
    pub fn memory_agents_dir(&self) -> PathBuf {
        self.memory_dir().join("agents")
    }
    pub fn memory_tmp_dir(&self) -> PathBuf {
        self.memory_dir().join("tmp")
    }
    pub fn user_profile_file(&self) -> PathBuf {
        self.memory_dir().join("user_profile.md")
    }
    /// Learner-proposed profile facts awaiting `/memory review`.
    pub fn user_profile_pending_file(&self) -> PathBuf {
        self.memory_dir().join("user_profile.pending.md")
    }
    pub fn lookup_dir(&self) -> PathBuf {
        self.root.join("lookup")
    }
    pub fn plan_lookup_file(&self) -> PathBuf {
        self.lookup_dir().join("plan_lookup.json")
    }
    pub fn plans_dir(&self) -> PathBuf {
        self.root.join("plans")
    }
    pub fn skills_dir(&self) -> PathBuf {
        self.root.join("skills")
    }
    pub fn pending_skills_dir(&self) -> PathBuf {
        self.skills_dir().join("_pending")
    }
    /// Custom agent definitions (DESIGN §6.5); not `memory/agents`, which holds role memories.
    pub fn custom_agents_dir(&self) -> PathBuf {
        self.root.join("agents")
    }
    pub fn knowledgepedia_dir(&self) -> PathBuf {
        self.root.join("knowledgepedia")
    }
    pub fn locks_dir(&self) -> PathBuf {
        self.root.join("locks")
    }
    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }
    /// File contents saved before edits, per session, for undo.
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }
    /// Pasted images, per session.
    pub fn attachments_dir(&self) -> PathBuf {
        self.root.join("attachments")
    }
    /// Global instructions added to every session's system prompt.
    pub fn instructions_file(&self) -> PathBuf {
        self.root.join("FINALITY.md")
    }
    /// Lock files written by the Finality editor extension.
    pub fn ide_dir(&self) -> PathBuf {
        self.root.join("ide")
    }
}

fn home_dir() -> io::Result<PathBuf> {
    dirs::home_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "cannot determine home directory"))
}

/// True for a filesystem root or the home directory: everything under them would count as inside.
pub fn is_broad_workdir(dir: &Path) -> bool {
    dir.parent().is_none()
        || home_dir()
            .ok()
            .and_then(|h| dunce::canonicalize(h).ok())
            .is_some_and(|h| h == dir)
}

fn expand_tilde(path: &Path) -> io::Result<PathBuf> {
    let mut comps = path.components();
    match comps.next() {
        Some(std::path::Component::Normal(first)) if first == "~" => {
            Ok(home_dir()?.join(comps.as_path()))
        }
        _ => Ok(path.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_beats_config_beats_default() {
        let cfg = Path::new("/from/config");
        assert_eq!(
            DataDir::resolve(Some("/from/env"), Some(cfg))
                .unwrap()
                .root(),
            Path::new("/from/env")
        );
        assert_eq!(DataDir::resolve(None, Some(cfg)).unwrap().root(), cfg);
        assert_eq!(DataDir::resolve(Some("  "), Some(cfg)).unwrap().root(), cfg);
        let default = DataDir::resolve(None, None).unwrap();
        assert!(default.root().ends_with(DEFAULT_DIR_NAME));
    }

    #[test]
    fn tilde_expands_to_home() {
        let d = DataDir::resolve(None, Some(Path::new("~/custom"))).unwrap();
        assert_eq!(d.root(), dirs::home_dir().unwrap().join("custom"));
    }

    #[test]
    fn ensure_creates_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let d = DataDir::new(tmp.path().join("h"));
        d.ensure().unwrap();
        for p in [
            d.sessions_dir(),
            d.memory_tmp_dir(),
            d.pending_skills_dir(),
            d.locks_dir(),
            d.logs_dir(),
        ] {
            assert!(p.is_dir(), "{}", p.display());
        }
        d.ensure().unwrap();
    }
}
