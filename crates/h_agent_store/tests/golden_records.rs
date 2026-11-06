//! Pins the postcard encoding of every `Record` variant. Regenerate with `H_AGENT_UPDATE_GOLDEN=1`
//! only for an intentional format change, and bump `hlog::FORMAT_VERSION` alongside it.
#![allow(clippy::unwrap_used)]

use std::fmt::Write;
use std::path::PathBuf;
use std::str::FromStr;

use h_agent_core::{AgentId, ApprovalDecision, Mode, Role, TaskId, TaskStatus};
use h_agent_store::hlog::encode_payload;
use h_agent_store::{Record, RecordEnvelope};

fn agent(n: u8) -> AgentId {
    AgentId::from_str(&format!("01JC{n:022}")).unwrap()
}

fn task() -> TaskId {
    TaskId::from_str("01JC0000000000000000000099").unwrap()
}

/// Exhaustive on purpose: a new variant must be added here (and to the golden file).
fn sample(kind: &Record) -> Record {
    match kind {
        Record::SessionStart { .. } => Record::SessionStart {
            workdir: "C:\\src\\app".into(),
            provider: "ollama".into(),
            model: "qwen3:14b".into(),
            mode: Mode::Cowork,
        },
        Record::UserMessage { .. } => Record::UserMessage {
            text: "add a --verbose flag".into(),
        },
        Record::AssistantMessage { .. } => Record::AssistantMessage {
            text: "Done.".into(),
            thinking: Some("plan".into()),
        },
        Record::ToolCall { .. } => Record::ToolCall {
            call_id: "c1".into(),
            tool: "read_file".into(),
            args_json: "{\"path\":\"Cargo.toml\"}".into(),
        },
        Record::ToolResult { .. } => Record::ToolResult {
            call_id: "c1".into(),
            output: "[package]".into(),
            is_error: false,
            truncated: true,
        },
        Record::Approval { .. } => Record::Approval {
            call_id: "c1".into(),
            decision: ApprovalDecision::Denied,
            reasons: vec!["outside workdir".into()],
            note: Some("use src/".into()),
        },
        Record::AgentSpawn { .. } => Record::AgentSpawn {
            child: agent(2),
            role: Role::Worker,
            task_id: Some(task()),
        },
        Record::AgentMessage { .. } => Record::AgentMessage {
            from: agent(1),
            to: agent(2),
            msg_json: "{}".into(),
        },
        Record::PlanUpdate { .. } => Record::PlanUpdate {
            plan_json: "[]".into(),
        },
        Record::TaskUpdate { .. } => Record::TaskUpdate {
            task_id: task(),
            status: TaskStatus::Partial,
            note: "tests flaky".into(),
        },
        Record::Usage { .. } => Record::Usage {
            input_tokens: 12_340,
            output_tokens: 2_105,
            ctx_used: 14_445,
            ctx_total: 32_768,
        },
        Record::ModeChange { .. } => Record::ModeChange {
            mode: Mode::AutoEdit,
        },
        Record::ProviderChange { .. } => Record::ProviderChange {
            provider: "anthropic".into(),
            model: "m".into(),
        },
        Record::SessionResumed { .. } => Record::SessionResumed {
            workdir: "/w".into(),
        },
        Record::SessionEnd { .. } => Record::SessionEnd {
            reason: "quit".into(),
        },
        Record::UserInterjection { .. } => Record::UserInterjection {
            text: "skip the docs step".into(),
        },
    }
}

fn all_samples() -> Vec<Record> {
    let placeholders = [
        Record::SessionStart {
            workdir: String::new(),
            provider: String::new(),
            model: String::new(),
            mode: Mode::Plan,
        },
        Record::UserMessage {
            text: String::new(),
        },
        Record::AssistantMessage {
            text: String::new(),
            thinking: None,
        },
        Record::ToolCall {
            call_id: String::new(),
            tool: String::new(),
            args_json: String::new(),
        },
        Record::ToolResult {
            call_id: String::new(),
            output: String::new(),
            is_error: false,
            truncated: false,
        },
        Record::Approval {
            call_id: String::new(),
            decision: ApprovalDecision::AutoAllowed,
            reasons: vec![],
            note: None,
        },
        Record::AgentSpawn {
            child: agent(0),
            role: Role::Worker,
            task_id: None,
        },
        Record::AgentMessage {
            from: agent(0),
            to: agent(0),
            msg_json: String::new(),
        },
        Record::PlanUpdate {
            plan_json: String::new(),
        },
        Record::TaskUpdate {
            task_id: task(),
            status: TaskStatus::Pending,
            note: String::new(),
        },
        Record::Usage {
            input_tokens: 0,
            output_tokens: 0,
            ctx_used: 0,
            ctx_total: 0,
        },
        Record::ModeChange { mode: Mode::Plan },
        Record::ProviderChange {
            provider: String::new(),
            model: String::new(),
        },
        Record::SessionResumed {
            workdir: String::new(),
        },
        Record::SessionEnd {
            reason: String::new(),
        },
        Record::UserInterjection {
            text: String::new(),
        },
    ];
    placeholders.iter().map(sample).collect()
}

fn render() -> String {
    let mut out = String::new();
    for (i, body) in all_samples().into_iter().enumerate() {
        let kind = body.kind();
        let env = RecordEnvelope {
            seq: i as u64,
            ts_unix_ms: 1_793_000_000_000,
            agent: agent(1),
            body,
        };
        let hex: String = encode_payload(&env)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        writeln!(out, "{kind} {hex}").unwrap();
    }
    out
}

#[test]
fn record_encoding_matches_golden_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/records.txt");
    let actual = render();
    if std::env::var_os("H_AGENT_UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| {
            panic!(
                "missing {}; run with H_AGENT_UPDATE_GOLDEN=1",
                path.display()
            )
        })
        .replace("\r\n", "\n");
    assert_eq!(
        actual, expected,
        "session log encoding changed; see file header"
    );
}

#[test]
fn every_variant_round_trips_through_a_log_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.hlog");
    let id = h_agent_core::SessionId::from_str("000001_20261105-000000").unwrap();
    let mut w = h_agent_store::LogWriter::create(&path, &id).unwrap();
    let written: Vec<RecordEnvelope> = all_samples()
        .into_iter()
        .enumerate()
        .map(|(i, body)| RecordEnvelope {
            seq: i as u64,
            ts_unix_ms: 2,
            agent: agent(1),
            body,
        })
        .collect();
    for env in &written {
        w.append_envelope(env).unwrap();
    }
    drop(w);
    assert_eq!(h_agent_store::read_log(&path).unwrap().records, written);
}
