//! Concurrent processes allocating session ids never collide.
#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use chrono::NaiveDate;
use h_agent_store::{DataDir, allocate_session_id};

const CHILD_ENV: &str = "H_AGENT_ID_TEST_DIR";
const CHILD_OUT_ENV: &str = "H_AGENT_ID_TEST_OUT";
const PROCESSES: usize = 4;
const PER_PROCESS: usize = 25;

#[test]
fn child_allocator() {
    let (Ok(dir), Ok(out)) = (std::env::var(CHILD_ENV), std::env::var(CHILD_OUT_ENV)) else {
        return;
    };
    let data = DataDir::new(dir);
    let started = NaiveDate::from_ymd_opt(2026, 11, 5)
        .unwrap()
        .and_hms_opt(12, 0, 0)
        .unwrap();
    let ids: Vec<String> = (0..PER_PROCESS)
        .map(|_| allocate_session_id(&data, started).unwrap().to_string())
        .collect();
    std::fs::write(out, ids.join("\n")).unwrap();
}

#[test]
fn parallel_processes_get_unique_serials() {
    let tmp = tempfile::tempdir().unwrap();
    let data = DataDir::new(tmp.path().join("data"));
    data.ensure().unwrap();
    let exe = std::env::current_exe().unwrap();
    let outs: Vec<_> = (0..PROCESSES)
        .map(|i| tmp.path().join(format!("out{i}.txt")))
        .collect();
    let children: Vec<_> = outs
        .iter()
        .map(|out| {
            Command::new(&exe)
                .args(["child_allocator", "--exact", "--test-threads=1"])
                .env(CHILD_ENV, data.root())
                .env(CHILD_OUT_ENV, out)
                .spawn()
                .unwrap()
        })
        .collect();
    for mut c in children {
        assert!(c.wait().unwrap().success());
    }
    let ids: Vec<String> = outs
        .iter()
        .flat_map(|p: &std::path::PathBuf| read_lines(p))
        .collect();
    let unique: HashSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), PROCESSES * PER_PROCESS);
    assert_eq!(unique.len(), ids.len(), "duplicate session ids");
    let counter = std::fs::read_to_string(data.session_counter_file()).unwrap();
    assert_eq!(counter.trim(), (PROCESSES * PER_PROCESS).to_string());
}

fn read_lines(p: &Path) -> Vec<String> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}
