//! Triage (DESIGN §6.1.2): classify a prompt before acting on it.

use h_agent_core::{ProviderKind, ThinkLevel};
use h_agent_llm::{ChatRequest, LlmError, LlmProvider, Message, Usage, collect};
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
