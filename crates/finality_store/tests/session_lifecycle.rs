#![allow(clippy::unwrap_used)]

use std::fs::OpenOptions;

use finality_core::{AgentId, Mode, SessionId};
use finality_store::{
    DataDir, LogWriter, Record, SessionStart, SessionStore, StoreError, Tail, read_log,
};

fn store() -> (tempfile::TempDir, SessionStore, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let data = DataDir::new(tmp.path().join("data"));
    data.ensure().unwrap();
    let workdir = tmp.path().join("project");
    std::fs::create_dir(&workdir).unwrap();
    (tmp, SessionStore::new(data), workdir)
}

fn meta(workdir: &std::path::Path) -> SessionStart<'_> {
    SessionStart {
        workdir,
        provider: "ollama",
        model: "qwen3:14b",
        mode: Mode::Cowork,
    }
}

#[test]
fn ten_thousand_records_survive_a_torn_tail() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("big.hlog");
    let id: SessionId = "000001_20261105-120000".parse().unwrap();
    let agent = AgentId::new();
    let mut w = LogWriter::create(&path, &id).unwrap();
    for i in 0..10_000 {
        w.append(
            agent,
            Record::UserMessage {
                text: format!("message number {i}"),
            },
        )
        .unwrap();
    }
    drop(w);

    let len = std::fs::metadata(&path).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len - 5)
        .unwrap();

    let c = read_log(&path).unwrap();
    assert_eq!(c.records.len(), 9_999);
    assert!(matches!(c.tail, Tail::Truncated { .. }));
    assert!(c.records.iter().enumerate().all(|(i, r)| r.seq == i as u64));

    let (mut w, _) = LogWriter::open_append(&path).unwrap();
    assert_eq!(w.next_seq(), 9_999);
    w.append(
        agent,
        Record::UserMessage {
            text: "after repair".into(),
        },
    )
    .unwrap();
    drop(w);
    let c = read_log(&path).unwrap();
    assert_eq!(c.tail, Tail::Clean);
    assert_eq!(c.records.len(), 10_000);
    assert_eq!(
        c.records[9_999].body,
        Record::UserMessage {
            text: "after repair".into()
        }
    );
}

#[tokio::test]
async fn start_and_close_updates_index_and_log() {
    let (_tmp, store, workdir) = store();
    let agent = AgentId::new();
    let s = store.start(meta(&workdir), agent).unwrap();
    let id = s.id;
    s.log
        .append(agent, Record::UserMessage { text: "hi".into() })
        .unwrap();
    store.close(s, agent, "quit").await.unwrap();

    let entry = store.index().get(&id).unwrap().unwrap();
    assert!(entry.closed_at.is_some());
    assert_eq!(entry.log, id.log_file_name());
    let c = read_log(&store.log_path(&id)).unwrap();
    let kinds: Vec<_> = c.records.iter().map(|r| r.body.kind()).collect();
    assert_eq!(kinds, ["SessionStart", "UserMessage", "SessionEnd"]);
    assert_eq!(
        store.projects().get(&workdir).unwrap().unwrap().session_ids,
        vec![id]
    );
}

#[tokio::test]
async fn crashed_session_is_detected_and_recovered() {
    let (_tmp, store, workdir) = store();
    let agent = AgentId::new();
    let crashed = store.start(meta(&workdir), agent).unwrap();
    let crashed_id = crashed.id;
    crashed
        .log
        .append(
            agent,
            Record::UserMessage {
                text: "work".into(),
            },
        )
        .unwrap();
    crashed.log.flush().await.unwrap();

    let live = store.start(meta(&workdir), agent).unwrap();
    let project = live.project_id.clone();

    // While the owner is alive the session is not an orphan.
    assert!(store.find_orphans(&project).unwrap().is_empty());

    drop(crashed);
    let orphans = store.find_orphans(&project).unwrap();
    assert_eq!(
        orphans.len(),
        1,
        "only the dropped session; the live one holds its lease"
    );
    assert_eq!(orphans[0].entry.id, crashed_id);

    let contents = store
        .recover(orphans.into_iter().next().unwrap(), agent)
        .unwrap();
    assert_eq!(contents.records.len(), 2);
    let entry = store.index().get(&crashed_id).unwrap().unwrap();
    assert!(entry.recovered && entry.closed_at.is_some());
    let last = read_log(&store.log_path(&crashed_id))
        .unwrap()
        .records
        .pop()
        .unwrap();
    assert!(matches!(last.body, Record::SessionEnd { .. }));
    assert!(store.find_orphans(&project).unwrap().is_empty());

    store.close(live, agent, "quit").await.unwrap();
}

#[tokio::test]
async fn resume_continues_sequence_and_rejects_double_open() {
    let (_tmp, store, workdir) = store();
    let agent = AgentId::new();
    let s = store.start(meta(&workdir), agent).unwrap();
    let id = s.id;
    for i in 0..3 {
        s.log
            .append(
                agent,
                Record::UserMessage {
                    text: format!("q{i}"),
                },
            )
            .unwrap();
        s.log
            .append(
                agent,
                Record::AssistantMessage {
                    text: format!("a{i}"),
                    thinking: None,
                },
            )
            .unwrap();
    }
    store.close(s, agent, "new").await.unwrap();

    let (resumed, contents) = store.resume(&id, &workdir, agent).unwrap();
    assert_eq!(contents.last_exchanges(2).len(), 2);
    assert!(matches!(
        store.resume(&id, &workdir, agent),
        Err(StoreError::SessionInUse(_))
    ));
    assert!(store.index().get(&id).unwrap().unwrap().closed_at.is_none());

    let seq = resumed
        .log
        .append(
            agent,
            Record::UserMessage {
                text: "again".into(),
            },
        )
        .unwrap();
    assert_eq!(
        seq, 9,
        "SessionStart + 6 + SessionEnd + SessionResumed precede it"
    );
    store.close(resumed, agent, "quit").await.unwrap();
}

#[test]
fn listing_is_per_project_newest_first() {
    let (tmp, store, workdir) = store();
    let other = tmp.path().join("other");
    std::fs::create_dir(&other).unwrap();
    let agent = AgentId::new();
    let a = store.start(meta(&workdir), agent).unwrap();
    let b = store.start(meta(&other), agent).unwrap();
    let c = store.start(meta(&workdir), agent).unwrap();
    let mine = store.list(Some(&a.project_id)).unwrap();
    assert_eq!(
        mine.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![c.id, a.id]
    );
    assert_eq!(store.list(None).unwrap().len(), 3);
    assert_ne!(a.project_id, b.project_id);
    assert!(matches!(
        store.resume(&"999999_20261105-000000".parse().unwrap(), &workdir, agent),
        Err(StoreError::SessionNotFound(_))
    ));
}

#[tokio::test]
async fn prune_removes_only_old_closed_sessions() {
    let (_tmp, store, workdir) = store();
    let agent = AgentId::new();
    let old = store.start(meta(&workdir), agent).unwrap();
    let old_id = old.id;
    store.close(old, agent, "quit").await.unwrap();
    let fresh = store.start(meta(&workdir), agent).unwrap();
    let fresh_id = fresh.id;
    store.close(fresh, agent, "quit").await.unwrap();
    let open = store.start(meta(&workdir), agent).unwrap();
    let open_id = open.id;
    // Backdate the first session's close time; the second closed just now.
    store
        .index()
        .update(&old_id, |e| {
            e.closed_at = Some(chrono::Utc::now() - chrono::Duration::days(40));
        })
        .unwrap();
    let plans = store.index().get(&old_id).unwrap().unwrap();
    assert!(plans.closed_at.is_some());

    let removed = store
        .prune(chrono::Utc::now() - chrono::Duration::days(30))
        .unwrap();
    assert_eq!(removed, vec![old_id]);
    assert!(!store.log_path(&old_id).exists());
    assert!(store.log_path(&fresh_id).exists());
    assert!(store.log_path(&open_id).exists());
    let left: Vec<_> = store
        .list(None)
        .unwrap()
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(left, vec![open_id, fresh_id]);
    store.close(open, agent, "quit").await.unwrap();
}
