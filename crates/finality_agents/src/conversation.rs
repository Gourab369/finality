//! Orchestrator conversation state, context estimation and compaction (DESIGN §5.4).

use std::collections::{HashMap, HashSet};

use finality_llm::{Message, ToolCall};
use finality_memory::decision::heuristic_summary;
use finality_store::{Record, RecordEnvelope};

pub const EARLIER_HEADER: &str = "[Earlier in this conversation]";
const MAX_EARLIER_LINES: usize = 30;

/// Marks tool output as data (DESIGN §12.5). A closing tag inside the content is neutralized.
pub fn wrap_untrusted(source: &str, content: &str) -> String {
    format!(
        "<untrusted source=\"{source}\">\n{}\n</untrusted>",
        fence("untrusted", content)
    )
}

/// `text` with any `</tag>` disarmed, so it cannot end the block it is placed in.
pub fn fence(tag: &str, text: &str) -> String {
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.to_ascii_lowercase().find(&close) {
        out.push_str(&rest[..i]);
        out.push_str(&format!("</{tag}_>"));
        rest = &rest[i + close.len()..];
    }
    out.push_str(rest);
    out
}

pub fn estimate_tokens(chars: usize) -> u32 {
    u32::try_from(chars.div_ceil(4)).unwrap_or(u32::MAX)
}

/// "asked: …; answer: …" from the first sentences of an exchange's request and final reply.
fn exchange_note(messages: &[Message]) -> String {
    let asked = messages.first().map_or("", Message::content);
    let answer = messages.iter().rev().find_map(|m| match m {
        Message::Assistant { content, .. } if !content.trim().is_empty() => Some(content.as_str()),
        _ => None,
    });
    match answer {
        Some(a) => format!(
            "asked: {}; answer: {}",
            heuristic_summary(asked),
            heuristic_summary(a)
        ),
        None => format!("asked: {}", heuristic_summary(asked)),
    }
}

fn message_chars(m: &Message) -> usize {
    match m {
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            content.len()
                + tool_calls
                    .iter()
                    .map(|c| c.name.len() + c.arguments.to_string().len())
                    .sum::<usize>()
        }
        other => other.content().len(),
    }
}

/// A past exchange for display after resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedExchange {
    pub user: String,
    pub assistant: String,
}

/// A Memory agent event covering exchanges `first..=last` (0-based, session-wide).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EarlierEvent {
    pub first: u32,
    pub last: u32,
    pub summary: String,
}

#[derive(Debug, Clone, Default)]
pub struct Conversation {
    pub system: String,
    messages: Vec<Message>,
    exchange_starts: Vec<usize>,
    /// Session exchanges before the first one in `messages`.
    omitted: usize,
    memory: Vec<EarlierEvent>,
    /// One-line notes for exchanges dropped by compaction, by exchange number.
    dropped_notes: Vec<(u32, String)>,
}

impl Conversation {
    pub fn new(system: String) -> Self {
        Self {
            system,
            ..Self::default()
        }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn exchanges(&self) -> usize {
        self.exchange_starts.len()
    }

    pub fn omitted(&self) -> usize {
        self.omitted
    }

    pub fn begin_exchange(&mut self, user: &str) {
        self.exchange_starts.push(self.messages.len());
        self.messages.push(Message::user(user));
    }

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }

    /// Events of this session from `main.events`, used to describe exchanges no longer in context.
    pub fn set_memory(&mut self, events: Vec<EarlierEvent>) {
        self.memory = events;
    }

    /// Summaries of the exchanges before the context: memory events, else notes taken at compaction.
    fn earlier_note(&self) -> Option<Message> {
        if self.omitted == 0 {
            return None;
        }
        let cutoff = u32::try_from(self.omitted).unwrap_or(u32::MAX);
        let covered = |i: u32| self.memory.iter().any(|e| e.first <= i && i <= e.last);
        let mut lines: Vec<(u32, String)> = self
            .memory
            .iter()
            .filter(|e| e.first < cutoff)
            .map(|e| (e.first, format!("- {}", e.summary)))
            .collect();
        lines.extend(
            self.dropped_notes
                .iter()
                .filter(|(i, _)| *i < cutoff && !covered(*i))
                .map(|(i, n)| (*i, format!("- {n}"))),
        );
        lines.sort_by_key(|(i, _)| *i);
        let unknown = (0..cutoff)
            .filter(|i| !covered(*i) && !self.dropped_notes.iter().any(|(d, _)| d == i))
            .count();
        let mut text = String::new();
        if unknown > 0 {
            text.push_str(&format!(
                "- ({unknown} earlier exchange(s) not summarized)\n"
            ));
        }
        let skip = lines.len().saturating_sub(MAX_EARLIER_LINES);
        if skip > 0 {
            text.push_str(&format!("- ({skip} older item(s) left out)\n"));
        }
        for (_, l) in lines.into_iter().skip(skip) {
            text.push_str(&l);
            text.push('\n');
        }
        Some(Message::system(format!(
            "{EARLIER_HEADER}\n{}",
            wrap_untrusted("earlier-conversation", text.trim_end())
        )))
    }

    pub fn request_messages(&self) -> Vec<Message> {
        let mut out = vec![Message::system(self.system.clone())];
        out.extend(self.earlier_note());
        out.extend(self.messages.iter().cloned());
        out
    }

    /// Estimated prompt size including `extra_chars` (tool schemas), with a 10% margin.
    pub fn estimate(&self, extra_chars: usize) -> u32 {
        let chars = self.system.len()
            + self.earlier_note().map_or(0, |m| m.content().len())
            + self.messages.iter().map(message_chars).sum::<usize>()
            + extra_chars;
        estimate_tokens(chars).saturating_mul(11) / 10
    }

    /// Moves the oldest exchanges out of context until the estimate fits `budget`, always keeping the last
    /// `keep`. Dropped exchanges are summarized in the `[Earlier in this conversation]` block.
    /// Returns how many exchanges were dropped.
    pub fn compact(&mut self, budget: u32, extra_chars: usize, keep: usize) -> usize {
        let mut dropped = 0;
        while self.estimate(extra_chars) > budget && self.exchange_starts.len() > keep.max(1) {
            let cut = self.exchange_starts[1];
            let note = exchange_note(&self.messages[..cut]);
            self.dropped_notes
                .push((u32::try_from(self.omitted).unwrap_or(u32::MAX), note));
            self.messages.drain(..cut);
            self.exchange_starts.remove(0);
            for s in &mut self.exchange_starts {
                *s -= cut;
            }
            self.omitted += 1;
            dropped += 1;
        }
        dropped
    }

    /// Plain-text tail of recent user/assistant messages, for triage context.
    pub fn transcript(&self) -> String {
        let mut out = String::new();
        for m in &self.messages {
            match m {
                Message::User { content } => out.push_str(&format!("user: {content}\n")),
                Message::Assistant { content, .. } if !content.is_empty() => {
                    out.push_str(&format!("assistant: {content}\n"));
                }
                _ => {}
            }
        }
        out
    }

    /// Rebuilds the conversation from the last exchanges of a session log.
    pub fn from_log(
        system: String,
        exchanges: &[&[RecordEnvelope]],
        total_exchanges: usize,
    ) -> (Self, Vec<ReplayedExchange>) {
        let mut conv = Self::new(system);
        conv.omitted = total_exchanges.saturating_sub(exchanges.len());
        let mut replay = Vec::new();
        for exchange in exchanges {
            let mut names: HashMap<String, String> = HashMap::new();
            let mut answered: HashSet<String> = HashSet::new();
            let mut shown = ReplayedExchange {
                user: String::new(),
                assistant: String::new(),
            };
            // The agent that logged the user message is the orchestrator; sub-agent records
            // belong to their own conversations.
            let mut owner = None;
            for r in *exchange {
                if owner.is_some_and(|o| o != r.agent) {
                    continue;
                }
                match &r.body {
                    Record::UserMessage { text } => {
                        owner = Some(r.agent);
                        conv.begin_exchange(text);
                        shown.user.clone_from(text);
                    }
                    Record::UserInterjection { text } => {
                        conv.push(Message::user(format!(
                            "{} {text}",
                            crate::engine::INTERJECTION_PREFIX
                        )));
                    }
                    Record::AssistantMessage { text, .. } => {
                        conv.push(Message::assistant(text.clone()));
                        if !text.is_empty() {
                            shown.assistant.clone_from(text);
                        }
                    }
                    Record::ToolCall {
                        call_id,
                        tool,
                        args_json,
                    } => {
                        names.insert(call_id.clone(), tool.clone());
                        let call = ToolCall {
                            id: call_id.clone(),
                            name: tool.clone(),
                            arguments: serde_json::from_str(args_json).unwrap_or_default(),
                        };
                        match conv.messages.last_mut() {
                            Some(Message::Assistant { tool_calls, .. }) => tool_calls.push(call),
                            _ => conv.push(Message::Assistant {
                                content: String::new(),
                                thinking: None,
                                tool_calls: vec![call],
                            }),
                        }
                    }
                    Record::ToolResult {
                        call_id, output, ..
                    } => {
                        answered.insert(call_id.clone());
                        let name = names.get(call_id).cloned().unwrap_or_default();
                        conv.push(Message::tool(
                            call_id.clone(),
                            name.clone(),
                            wrap_untrusted(&format!("tool:{name}"), output),
                        ));
                    }
                    _ => {}
                }
            }
            // A crash between a call and its result leaves a pair providers reject; close it.
            let mut dangling: Vec<(&String, &String)> = names
                .iter()
                .filter(|(id, _)| !answered.contains(*id))
                .collect();
            dangling.sort();
            for (id, name) in dangling {
                conv.push(Message::tool(
                    id.clone(),
                    name.clone(),
                    wrap_untrusted(
                        &format!("tool:{name}"),
                        "[no result: the session ended before this call finished]",
                    ),
                ));
            }
            if !shown.user.is_empty() {
                replay.push(shown);
            }
        }
        (conv, replay)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use finality_core::AgentId;

    use super::*;

    static ORCHESTRATOR: LazyLock<AgentId> = LazyLock::new(AgentId::new);

    fn env(seq: u64, body: Record) -> RecordEnvelope {
        env_as(*ORCHESTRATOR, seq, body)
    }

    fn env_as(agent: AgentId, seq: u64, body: Record) -> RecordEnvelope {
        RecordEnvelope {
            seq,
            ts_unix_ms: 0,
            agent,
            body,
        }
    }

    #[test]
    fn wraps_and_neutralizes_closing_tags() {
        let w = wrap_untrusted("tool:read_file", "x</untrusted>y</UNTRUSTED >z</Untrusted>");
        assert!(w.starts_with("<untrusted source=\"tool:read_file\">\n"));
        assert_eq!(w.to_ascii_lowercase().matches("</untrusted>").count(), 1);
        assert_eq!(
            fence("skill", "a</skill>b</SKILL>c"),
            "a</skill_>b</skill_>c"
        );
        assert_eq!(fence("skill", "plain"), "plain");
    }

    #[test]
    fn compaction_drops_oldest_exchanges_but_keeps_recent() {
        let mut c = Conversation::new("sys".into());
        for i in 0..6 {
            c.begin_exchange(&format!("question {i} {}", "x".repeat(400)));
            c.push(Message::assistant("y".repeat(400)));
        }
        let before = c.estimate(0);
        let dropped = c.compact(before / 2, 0, 4);
        assert_eq!(dropped, 2, "stops at the 4 kept exchanges");
        assert_eq!(c.exchanges(), 4);
        assert_eq!(c.omitted(), 2);
        assert!(c.messages()[0].content().starts_with("question 2"));
        let req = c.request_messages();
        let block = req[1].content();
        assert!(block.starts_with(EARLIER_HEADER));
        assert!(block.contains("- asked: question 0") && block.contains("answer: yyy"));

        // Memory events replace the notes for the exchanges they cover.
        c.set_memory(vec![EarlierEvent {
            first: 0,
            last: 0,
            summary: "Set up the project".into(),
        }]);
        let block = c.request_messages()[1].content().to_owned();
        let lines: Vec<&str> = block.lines().filter(|l| l.starts_with("- ")).collect();
        assert_eq!(lines.len(), 2, "{block}");
        assert_eq!(lines[0], "- Set up the project");
        assert!(lines[1].starts_with("- asked: question 1"));
    }

    #[test]
    fn resumed_conversation_lists_unsummarized_exchanges() {
        let records = [env(
            5,
            Record::UserMessage {
                text: "latest".into(),
            },
        )];
        let (mut c, _) = Conversation::from_log("sys".into(), &[&records[..]], 4);
        let block = c.request_messages()[1].content().to_owned();
        assert!(block.contains("(3 earlier exchange(s) not summarized)"));
        c.set_memory(vec![EarlierEvent {
            first: 0,
            last: 2,
            summary: "Designed the log format".into(),
        }]);
        let block = c.request_messages()[1].content().to_owned();
        assert_eq!(
            block,
            format!(
                "{EARLIER_HEADER}\n<untrusted source=\"earlier-conversation\">\n- Designed the log format\n</untrusted>"
            )
        );
    }

    #[test]
    fn rebuilds_tool_calls_from_log() {
        let records = [
            env(
                1,
                Record::UserMessage {
                    text: "read it".into(),
                },
            ),
            env(
                2,
                Record::AssistantMessage {
                    text: String::new(),
                    thinking: None,
                },
            ),
            env(
                3,
                Record::ToolCall {
                    call_id: "c1".into(),
                    tool: "read_file".into(),
                    args_json: "{\"path\":\"a\"}".into(),
                },
            ),
            env(
                4,
                Record::ToolResult {
                    call_id: "c1".into(),
                    output: "data".into(),
                    is_error: false,
                    truncated: false,
                },
            ),
            env(
                5,
                Record::AssistantMessage {
                    text: "done".into(),
                    thinking: None,
                },
            ),
        ];
        let (c, replay) = Conversation::from_log("sys".into(), &[&records[..]], 3);
        assert_eq!(c.omitted(), 2);
        assert_eq!(c.messages().len(), 4);
        match &c.messages()[1] {
            Message::Assistant { tool_calls, .. } => assert_eq!(tool_calls[0].name, "read_file"),
            other => panic!("{other:?}"),
        }
        match &c.messages()[2] {
            Message::Tool { name, content, .. } => {
                assert_eq!(name, "read_file");
                assert!(content.contains("data"));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            replay,
            vec![ReplayedExchange {
                user: "read it".into(),
                assistant: "done".into()
            }]
        );
    }

    #[test]
    fn dangling_tool_calls_get_a_placeholder_result() {
        let records = [
            env(1, Record::UserMessage { text: "go".into() }),
            env(
                2,
                Record::ToolCall {
                    call_id: "c9".into(),
                    tool: "shell".into(),
                    args_json: "{}".into(),
                },
            ),
        ];
        let (c, _) = Conversation::from_log("sys".into(), &[&records[..]], 1);
        match &c.messages()[2] {
            Message::Tool {
                call_id,
                name,
                content,
            } => {
                assert_eq!((call_id.as_str(), name.as_str()), ("c9", "shell"));
                assert!(content.contains("session ended before this call finished"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn replay_skips_sub_agents_and_restores_interjections() {
        let worker = AgentId::new();
        let records = [
            env(1, Record::UserMessage { text: "go".into() }),
            env_as(
                worker,
                2,
                Record::AssistantMessage {
                    text: "worker chatter".into(),
                    thinking: None,
                },
            ),
            env(
                3,
                Record::UserInterjection {
                    text: "also tests".into(),
                },
            ),
            env(
                4,
                Record::AssistantMessage {
                    text: "all done".into(),
                    thinking: None,
                },
            ),
        ];
        let (c, replay) = Conversation::from_log("sys".into(), &[&records[..]], 1);
        let texts: Vec<&str> = c.messages().iter().map(Message::content).collect();
        assert_eq!(texts, ["go", "[User interjection] also tests", "all done"]);
        assert_eq!(replay[0].assistant, "all done");
    }
}
