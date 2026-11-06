//! Per-job scratch files of the Memory agent (DESIGN §7.3).

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use h_agent_core::EventId;
use h_agent_store::DataDir;

const PREFIX: &str = "memjob_";
const SUFFIX: &str = ".tmp";

/// `memory/tmp/memjob_<ulid>.tmp`, deleted when dropped (also on error or panic).
pub struct JobScratch {
    path: PathBuf,
    file: Option<File>,
}

impl JobScratch {
    pub fn create(dir: &DataDir) -> io::Result<Self> {
        let tmp = dir.memory_tmp_dir();
        std::fs::create_dir_all(&tmp)?;
        let path = tmp.join(format!("{PREFIX}{}{SUFFIX}", EventId::new()));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: Some(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, label: &str, text: &str) -> io::Result<()> {
        match &mut self.file {
            Some(f) => writeln!(f, "## {label}\n{text}\n"),
            None => Ok(()),
        }
    }
}

impl Drop for JobScratch {
    fn drop(&mut self) {
        // Windows cannot delete a file that is still open.
        self.file.take();
        if let Err(e) = std::fs::remove_file(&self.path)
            && e.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path.display(), error = %e, "cannot delete memory scratch file");
        }
    }
}

/// Deletes scratch files left by crashed processes. Only files older than `older_than` are touched,
/// so jobs of other running processes are safe.
pub fn cleanup_orphans(dir: &DataDir, older_than: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(dir.memory_tmp_dir()) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with(PREFIX) && name.ends_with(SUFFIX)
        })
        .filter(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age >= older_than)
        })
        .filter(|e| std::fs::remove_file(e.path()).is_ok())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleted_on_drop_even_after_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let mut s = JobScratch::create(&dir).unwrap();
        s.append("request", "hello").unwrap();
        let path = s.path().to_owned();
        assert!(path.exists());
        drop(s);
        assert!(!path.exists());

        let seen = std::sync::Mutex::new(PathBuf::new());
        let result = std::panic::catch_unwind(|| {
            let s = JobScratch::create(&dir).unwrap();
            *seen.lock().unwrap() = s.path().to_owned();
            panic!("job failed");
        });
        assert!(result.is_err());
        assert!(!seen.lock().unwrap().exists());
    }

    #[test]
    fn orphan_cleanup_respects_age() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        std::fs::create_dir_all(dir.memory_tmp_dir()).unwrap();
        let orphan = dir.memory_tmp_dir().join("memjob_x.tmp");
        let other = dir.memory_tmp_dir().join("notes.txt");
        std::fs::write(&orphan, "x").unwrap();
        std::fs::write(&other, "x").unwrap();
        assert_eq!(cleanup_orphans(&dir, Duration::from_secs(3600)), 0);
        assert_eq!(cleanup_orphans(&dir, Duration::ZERO), 1);
        assert!(!orphan.exists() && other.exists());
    }
}
