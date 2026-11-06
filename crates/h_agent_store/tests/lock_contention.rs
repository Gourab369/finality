//! Several processes increment a shared counter file under `with_lock`; no update may be lost.
#![allow(clippy::unwrap_used)]

use std::path::Path;
use std::process::Command;

use h_agent_store::{with_lock, write_atomic};

const CHILD_ENV: &str = "H_AGENT_LOCK_TEST_DIR";
const PROCESSES: usize = 4;
const INCREMENTS: usize = 50;

fn increment(dir: &Path) -> std::io::Result<()> {
    let counter = dir.join("counter");
    let current: usize = match std::fs::read_to_string(&counter) {
        Ok(s) => s.trim().parse().unwrap_or(0),
        Err(_) => 0,
    };
    write_atomic(&counter, (current + 1).to_string().as_bytes())
}

/// Acts as a worker when spawned by `parallel_processes_do_not_lose_updates`; a no-op otherwise.
#[test]
fn child_worker() {
    let Ok(dir) = std::env::var(CHILD_ENV) else {
        return;
    };
    let dir = Path::new(&dir);
    for _ in 0..INCREMENTS {
        with_lock(&dir.join("locks"), "counter", || increment(dir)).unwrap();
    }
}

#[test]
fn parallel_processes_do_not_lose_updates() {
    let tmp = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let children: Vec<_> = (0..PROCESSES)
        .map(|_| {
            Command::new(&exe)
                .args(["child_worker", "--exact", "--test-threads=1"])
                .env(CHILD_ENV, tmp.path())
                .spawn()
                .unwrap()
        })
        .collect();
    for mut child in children {
        assert!(child.wait().unwrap().success());
    }
    let total: usize = std::fs::read_to_string(tmp.path().join("counter"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(total, PROCESSES * INCREMENTS);
}
