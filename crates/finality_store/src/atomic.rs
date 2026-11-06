use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use tempfile::NamedTempFile;

/// Writes `bytes` to a temp file in the same directory, fsyncs it, then renames it over `path`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    const MAX_ATTEMPTS: u32 = 20;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut tmp = NamedTempFile::new_in(parent)?;
    tmp.as_file_mut().write_all(bytes)?;
    tmp.as_file().sync_all()?;
    let mut attempt = 1;
    loop {
        match tmp.persist(path) {
            Ok(_) => return Ok(()),
            // Windows: scanners or a pending delete of the replaced file briefly deny the rename.
            Err(e)
                if e.error.kind() == io::ErrorKind::PermissionDenied && attempt < MAX_ATTEMPTS =>
            {
                tmp = e.file;
                std::thread::sleep(Duration::from_millis(u64::from(attempt) * 5));
                attempt += 1;
            }
            Err(e) => return Err(e.error),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LockError<E> {
    #[error("invalid lock name `{0}`")]
    InvalidName(String),
    #[error("lock io error: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Inner(E),
}

/// Runs `f` while holding an exclusive cross-process lock on `<locks_dir>/<name>.lock`. Blocks until acquired.
pub fn with_lock<T, E>(
    locks_dir: &Path,
    name: &str,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, LockError<E>> {
    if !valid_lock_name(name) {
        return Err(LockError::InvalidName(name.to_owned()));
    }
    std::fs::create_dir_all(locks_dir)?;
    let file: File = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(locks_dir.join(format!("{name}.lock")))?;
    file.lock()?;
    let result = f().map_err(LockError::Inner);
    drop(file);
    result
}

pub(crate) fn valid_lock_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_creates_and_replaces() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f.json");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two");
        let leftovers = std::fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(leftovers, 1);
    }

    #[test]
    fn lock_rejects_path_like_names() {
        let tmp = tempfile::tempdir().unwrap();
        for bad in ["", "../x", "a/b", "A", "x.lock"] {
            let r = with_lock(tmp.path(), bad, || Ok::<_, io::Error>(()));
            assert!(matches!(r, Err(LockError::InvalidName(_))), "{bad}");
        }
    }

    #[test]
    fn lock_propagates_inner_result() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            with_lock(tmp.path(), "t", || Ok::<_, io::Error>(5)).unwrap(),
            5
        );
        let err = with_lock(tmp.path(), "t", || Err::<(), _>("boom")).unwrap_err();
        assert!(matches!(err, LockError::Inner("boom")));
    }
}
