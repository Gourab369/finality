//! Triage (DESIGN §6.1.2): classify a prompt before acting on it.

use finality_core::{ProviderKind, ThinkLevel};
use finality_llm::{ChatRequest, LlmError, LlmProvider, Message, Usage, collect};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

pub const SEED_QUERY_TYPES: &[&str] = &[
    "command",
    "question",
    "analysis",
    "creation",
    "conversation",
];
const MAX_HISTORY_CHARS: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    Simple,
    MultiStep,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Triage {
    pub query_type: String,
    pub topic: String,
    pub complexity: Complexity,
    pub needs_tools: bool,
    #[serde(default)]
    pub tool_hints: Vec<String>,
    #[serde(default)]
    pub knowledge_topics: Vec<String>,
}

impl Triage {
    /// Used when the model's answer cannot be parsed: assume tools may be needed.
    pub fn fallback() -> Self {
        Self {
            query_type: "question".into(),
            topic: "general".into(),
            complexity: Complexity::Simple,
            needs_tools: true,
            tool_hints: Vec::new(),
            knowledge_topics: Vec::new(),
        }
    }
}

const QUICK_MAX_CHARS: usize = 160;
const QUICK_COMMAND_MAX_CHARS: usize = 80;
const SMALL_TALK: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "yes",
    "no",
    "cool",
    "great",
    "nice",
    "bye",
    "good morning",
    "good night",
];
const QUESTION_WORDS: &[&str] = &[
    "what", "why", "how", "when", "where", "who", "which", "is", "are", "can", "could", "does",
    "do", "should", "would", "will",
];
/// Words that suggest several steps; such messages always get a full triage.
const MULTI_STEP_MARKERS: &[&str] = &[
    " and ",
    " then ",
    "after that",
    "first",
    "step",
    "plan",
    "refactor",
    "implement",
    "migrate",
    "build a",
    "create a",
    "set up",
    "setup",
];

/// A triage without a model request for obviously simple messages: small talk, short questions and short
/// single-step commands. `None` means the message needs the real triage.
pub fn quick(text: &str) -> Option<Triage> {
    let t = text.trim();
    if t.is_empty() || t.len() > QUICK_MAX_CHARS || t.contains('\n') || t.contains("```") {
        return None;
    }
    let lower = t.to_lowercase();
    let padded = format!(" {lower} ");
    if MULTI_STEP_MARKERS.iter().any(|m| padded.contains(m)) {
        return None;
    }
    let bare = lower.trim_end_matches(['.', '!', '?', ' ']);
    let first = lower.split_whitespace().next().unwrap_or("");
    let query_type = if SMALL_TALK.contains(&bare) {
        "conversation"
    } else if lower.ends_with('?') || QUESTION_WORDS.contains(&first) {
        "question"
    } else if t.len() <= QUICK_COMMAND_MAX_CHARS && !t.contains([',', ';']) {
        "command"
    } else {
        return None;
    };
    Some(Triage {
        query_type: query_type.into(),
        topic: "general".into(),
        complexity: Complexity::Simple,
        needs_tools: query_type != "conversation",
        tool_hints: Vec::new(),
        knowledge_topics: Vec::new(),
    })
}

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query_type": { "type": "string" },
            "topic": { "type": "string" },
            "complexity": { "type": "string", "enum": ["simple", "multi_step"] },
            "needs_tools": { "type": "boolean" },
            "tool_hints": { "type": "array", "items": { "type": "string" } },
            "knowledge_topics": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["query_type", "topic", "complexity", "needs_tools"]
    })
}

pub fn parse(text: &str) -> Option<Triage> {
    let trimmed = text.trim();
    // Some models wrap JSON in a code fence despite the schema.
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    let mut t: Triage = serde_json::from_str(trimmed.get(start..=end)?).ok()?;
    t.query_type = t.query_type.trim().to_lowercase();
    t.topic = t.topic.trim().to_lowercase();
    if t.query_type.is_empty() || t.topic.is_empty() {
        return None;
    }
    Some(t)
}

/// Classifies `prompt`, using the tail of `recent` (prior exchanges as text) for context.
pub async fn triage(
    provider: &dyn LlmProvider,
    kind: Option<ProviderKind>,
    model: &str,
    prompt: &str,
    recent: &str,
    cancel: CancellationToken,
) -> Result<(Triage, Usage), LlmError> {
    let instructions = include_str!("../prompts/triage.md")
        .replace("{{query_types}}", &SEED_QUERY_TYPES.join(", "));
    let tail: String = {
        let chars: Vec<char> = recent.chars().collect();
        chars[chars.len().saturating_sub(MAX_HISTORY_CHARS)..]
            .iter()
            .collect()
    };
    let mut user = String::new();
    if !tail.is_empty() {
        user.push_str(&format!("Recent conversation:\n{tail}\n\n"));
    }
    user.push_str(&format!("Latest request:\n{prompt}"));
    let req = ChatRequest {
        provider: kind,
        model: model.to_owned(),
        messages: vec![Message::system(instructions), Message::user(user)],
        think: ThinkLevel::Off,
        response_format: Some(schema()),
        temperature: Some(0.0),
        ..ChatRequest::default()
    };
    let response = collect(provider.chat(req, cancel).await?).await?;
    let triage = parse(&response.text).unwrap_or_else(|| {
        tracing::warn!(reply = %response.text, "triage reply was not valid JSON; using fallback");
        Triage::fallback()
    });
    Ok((triage, response.usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_triage_takes_only_obviously_simple_messages() {
        let kind = |t: &str| quick(t).map(|q| q.query_type);
        assert_eq!(kind("thanks!").as_deref(), Some("conversation"));
        assert_eq!(kind("What does main.rs do?").as_deref(), Some("question"));
        assert_eq!(
            kind("how is the config loaded").as_deref(),
            Some("question")
        );
        assert_eq!(kind("read Cargo.toml").as_deref(), Some("command"));
        for needs_triage in [
            "add a --verbose flag and update the docs",
            "first find the parser, then add a flag",
            "refactor the store module",
            "fix the build, run the tests; commit",
            "line one\nline two",
            &"x".repeat(200),
        ] {
            assert!(quick(needs_triage).is_none(), "{needs_triage}");
        }
        assert!(!quick("hi").unwrap().needs_tools);
    }

    #[test]
    fn parses_plain_and_fenced_json() {
        let t = parse(r#"{"query_type":"Command","topic":"Coding/Rust","complexity":"simple","needs_tools":true}"#).unwrap();
        assert_eq!(t.query_type, "command");
        assert_eq!(t.topic, "coding/rust");
        assert!(t.tool_hints.is_empty());
        let fenced = "```json\n{\"query_type\":\"question\",\"topic\":\"physics\",\"complexity\":\"multi_step\",\"needs_tools\":false}\n```";
        assert_eq!(parse(fenced).unwrap().complexity, Complexity::MultiStep);
        assert!(parse("not json").is_none());
        assert!(
            parse(r#"{"query_type":"","topic":"x","complexity":"simple","needs_tools":true}"#)
                .is_none()
        );
    }
}
