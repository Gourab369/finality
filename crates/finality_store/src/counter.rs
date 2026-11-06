use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use chrono::NaiveDateTime;
use finality_core::SessionId;

use crate::atomic::{with_lock, write_atomic};
use crate::{DataDir, StoreError};

/// Allocates the next globally unique, monotonic session serial (DESIGN §5.1.2).
pub fn allocate_session_id(dir: &DataDir, started: NaiveDateTime) -> Result<SessionId, StoreError> {
    let counter = dir.session_counter_file();
    let serial = with_lock(&dir.locks_dir(), "counter", || {
        let current = match std::fs::read_to_string(&counter) {
            Ok(s) => s.trim().parse::<u64>().map_err(|_| {
                StoreError::Corrupt(format!("session counter {}", counter.display()))
            })?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e.into()),
        };
        let next = current + 1;
        write_atomic(&counter, next.to_string().as_bytes())?;
        Ok(next)
    })?;
    Ok(SessionId::new(serial, started))
}

/// Exclusive per-session lock held by the process that has the session open.
/// A session that is unclosed in the index but whose lease can be taken was abandoned by a crash.
#[derive(Debug)]
pub struct SessionLease {
    _file: File,
}

impl SessionLease {
    /// `Ok(None)` if another handle (in any process) holds the lease.
    pub fn try_acquire(locks_dir: &Path, id: &SessionId) -> io::Result<Option<Self>> {
        std::fs::create_dir_all(locks_dir)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(locks_dir.join(format!("session-{id}.lock")))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    fn now() -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 11, 5)
            .unwrap()
            .and_hms_opt(10, 0, 0)
            .unwrap()
    }

    #[test]
    fn serials_increase() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        dir.ensure().unwrap();
        let a = allocate_session_id(&dir, now()).unwrap();
        let b = allocate_session_id(&dir, now()).unwrap();
        assert_eq!((a.serial, b.serial), (1, 2));
        assert_eq!(a.to_string(), "000001_20261105-100000");
    }

    #[test]
    fn corrupt_counter_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        dir.ensure().unwrap();
        std::fs::write(dir.session_counter_file(), "garbage").unwrap();
        assert!(matches!(
            allocate_session_id(&dir, now()),
            Err(StoreError::Corrupt(_))
        ));
    }

    #[test]
    fn lease_is_exclusive_until_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let id = SessionId::new(3, now());
        let first = SessionLease::try_acquire(tmp.path(), &id).unwrap();
        assert!(first.is_some());
        assert!(
            SessionLease::try_acquire(tmp.path(), &id)
                .unwrap()
                .is_none()
        );
        drop(first);
        assert!(
            SessionLease::try_acquire(tmp.path(), &id)
                .unwrap()
                .is_some()
        );
    }
}
