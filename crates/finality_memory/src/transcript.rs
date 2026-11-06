//! Turning session-log records into exchanges and condensed transcripts (DESIGN §7.3, §7.6).

use std::collections::HashSet;

use finality_core::AgentId;
use finality_store::{LogContents, Record, RecordEnvelope};

use crate::decision::Span;
use crate::tree::clip;

/// Characters of transcript per exchange given to the Memory agent.
pub const SPAN_CHARS: usize = 3000;
const ENTRY_CHARS: usize = 600;
const TOOL_ARGS_CHARS: usize = 200;
const TOOL_OUTPUT_CHARS: usize = 300;

/// Sub-agents spawned in this log.
pub fn spawned_agents(records: &[RecordEnvelope]) -> HashSet<AgentId> {
    records
        .iter()
        .filter_map(|r| match r.body {
            Record::AgentSpawn { child, .. } => Some(child),
            _ => None,
        })
        .collect()
}

/// One line per meaningful record, cut to `max_chars` overall.
pub fn condense<'a>(
    records: impl IntoIterator<Item = &'a RecordEnvelope>,
    max_chars: usize,
) -> String {
    let mut out = String::new();
    for r in records {
        let line = match &r.body {
            Record::UserMessage { text } => format!("user: {}", clip(text, ENTRY_CHARS)),
            Record::UserInterjection { text } => {
                format!("user (while the agent worked): {}", clip(text, ENTRY_CHARS))
            }
            Record::AssistantMessage { text, .. } if !text.trim().is_empty() => {
                format!("assistant: {}", clip(text, ENTRY_CHARS))
            }
            Record::ToolCall {
                tool, args_json, ..
            } => format!("tool call {tool} {}", clip(args_json, TOOL_ARGS_CHARS)),
            Record::ToolResult {
                output, is_error, ..
            } => format!(
                "tool {}: {}",
                if *is_error { "error" } else { "result" },
                clip(output, TOOL_OUTPUT_CHARS)
            ),
            Record::TaskUpdate { status, note, .. } => {
                format!("task update: {status} {}", clip(note, TOOL_OUTPUT_CHARS))
            }
            _ => continue,
        };
        if out.chars().count() + line.chars().count() > max_chars {
            out.push_str("…(truncated)");
            break;
        }
        out.push_str(&line);
        out.push('\n');
    }
    out.trim_end().to_owned()
}

/// The Orchestrator's exchanges: sub-agent records are left out.
pub fn main_spans(contents: &LogContents) -> Vec<Span> {
    let children = spawned_agents(&contents.records);
    contents
        .exchanges()
        .into_iter()
        .enumerate()
        .filter_map(|(i, ex)| {
            let own: Vec<&RecordEnvelope> =
                ex.iter().filter(|r| !children.contains(&r.agent)).collect();
            let headline = own.iter().find_map(|r| match &r.body {
                Record::UserMessage { text } => Some(text.clone()),
                _ => None,
            })?;
            Some(Span {
                index: u32::try_from(i).unwrap_or(u32::MAX),
                seq: (own.first()?.seq, own.last()?.seq),
                headline,
                text: condense(own, SPAN_CHARS),
            })
        })
        .collect()
}

/// Steps of one sub-agent thread; each starts at a model reply.
pub fn thread_spans(contents: &LogContents, thread: AgentId) -> Vec<Span> {
    let assignment = contents.records.iter().find_map(|r| match &r.body {
        Record::AgentMessage { to, msg_json, .. } if *to == thread => {
            serde_json::from_str::<serde_json::Value>(msg_json)
                .ok()?
                .get("instructions")?
                .as_str()
                .map(str::to_owned)
        }
        _ => None,
    });
    let mut groups: Vec<Vec<&RecordEnvelope>> = Vec::new();
    for r in contents.records.iter().filter(|r| r.agent == thread) {
        // A step is logged as Usage, then the reply, then its tool calls and results.
        let starts_step = matches!(
            r.body,
            Record::Usage { .. } | Record::AssistantMessage { .. }
        );
        let current_has_reply = groups.last().is_some_and(|g| {
            g.iter()
                .any(|x| matches!(x.body, Record::AssistantMessage { .. }))
        });
        if groups.is_empty() || (starts_step && current_has_reply) {
            groups.push(Vec::new());
        }
        if let Some(g) = groups.last_mut() {
            g.push(r);
        }
    }
    groups
        .into_iter()
        .enumerate()
        .filter_map(|(i, g)| {
            let reply = g.iter().find_map(|r| match &r.body {
                Record::AssistantMessage { text, .. } if !text.trim().is_empty() => {
                    Some(text.clone())
                }
                _ => None,
            });
            let headline = match (&assignment, i) {
                (Some(a), 0) => a.clone(),
                _ => reply.unwrap_or_default(),
            };
            let mut text = condense(g.iter().copied(), SPAN_CHARS);
            if i == 0
                && let Some(a) = &assignment
            {
                text = format!("assignment: {}\n{text}", clip(a, ENTRY_CHARS));
            }
            Some(Span {
                index: u32::try_from(i).unwrap_or(u32::MAX),
                seq: (g.first()?.seq, g.last()?.seq),
                headline,
                text,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use finality_core::Role;
    use finality_store::Tail;

    use super::*;

    fn rec(seq: u64, agent: AgentId, body: Record) -> RecordEnvelope {
        RecordEnvelope {
            seq,
            ts_unix_ms: 0,
            agent,
            body,
        }
    }

    fn contents(records: Vec<RecordEnvelope>) -> LogContents {
        LogContents {
            session_id: "s".into(),
            records,
            valid_len: 0,
            tail: Tail::Clean,
        }
    }

    #[test]
    fn splits_main_and_thread_records() {
        let (o, w) = (AgentId::new(), AgentId::new());
        let say = |text: &str| Record::AssistantMessage {
            text: text.into(),
            thinking: None,
        };
        let c = contents(vec![
            rec(
                0,
                o,
                Record::UserMessage {
                    text: "Build it. Now.".into(),
                },
            ),
            rec(
                1,
                o,
                Record::AgentSpawn {
                    child: w,
                    role: Role::Worker,
                    task_id: None,
                },
            ),
            rec(
                2,
                o,
                Record::AgentMessage {
                    from: o,
                    to: w,
                    msg_json: r#"{"type":"assign","instructions":"Edit lib.rs"}"#.into(),
                },
            ),
            rec(3, w, say("")),
            rec(
                4,
                w,
                Record::ToolCall {
                    call_id: "c".into(),
                    tool: "read_file".into(),
                    args_json: "{}".into(),
                },
            ),
            rec(
                5,
                w,
                Record::ToolResult {
                    call_id: "c".into(),
                    output: "fn main".into(),
                    is_error: false,
                    truncated: false,
                },
            ),
            rec(
                6,
                w,
                Record::Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ctx_used: 2,
                    ctx_total: 10,
                },
            ),
            rec(7, w, say("edited")),
            rec(8, o, say("Done.")),
            rec(
                9,
                o,
                Record::UserMessage {
                    text: "Thanks".into(),
                },
            ),
        ]);
        let main = main_spans(&c);
        assert_eq!(main.len(), 2);
        assert_eq!(
            (main[0].seq, main[0].headline.as_str()),
            ((0, 8), "Build it. Now.")
        );
        assert!(main[0].text.contains("assistant: Done.") && !main[0].text.contains("edited"));
        assert_eq!(main[1].index, 1);

        let thread = thread_spans(&c, w);
        assert_eq!(thread.len(), 2);
        assert_eq!(
            thread[1].seq,
            (6, 7),
            "the usage record opens the next step"
        );
        assert_eq!(thread[0].seq, (3, 5));
        assert_eq!(thread[0].headline, "Edit lib.rs");
        assert!(thread[0].text.starts_with("assignment: Edit lib.rs"));
        assert!(thread[0].text.contains("tool call read_file"));
        assert_eq!(thread[1].headline, "edited");
    }

    #[test]
    fn condense_respects_the_limit() {
        let a = AgentId::new();
        let records: Vec<RecordEnvelope> = (0..50)
            .map(|i| {
                rec(
                    i,
                    a,
                    Record::UserMessage {
                        text: "x".repeat(100),
                    },
                )
            })
            .collect();
        let text = condense(&records, 1000);
        assert!(text.chars().count() <= 1000 + 20);
        assert!(text.ends_with("…(truncated)"));
    }
}
