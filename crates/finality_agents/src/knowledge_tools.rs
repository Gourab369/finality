//! `knowledge_lookup(query)` (DESIGN §8.3): knowledgepedia entries for the Orchestrator.

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_memory::knowledge::Knowledgepedia;
use finality_safety::Location;
use finality_tools::tool::parse_args;
use finality_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

pub const KNOWLEDGE_LOOKUP: &str = "knowledge_lookup";
/// About 2k tokens.
pub const KNOWLEDGE_CHARS: usize = 8000;

#[derive(Deserialize)]
struct Args {
    query: String,
}

/// Matching entries' text, or a list of known topics when nothing matches.
pub fn lookup(kb: &Knowledgepedia, query: &str) -> String {
    if let Some(text) = kb.lookup_text(query, KNOWLEDGE_CHARS) {
        return text;
    }
    let topics: Vec<String> = kb
        .index()
        .map(|i| i.topics.into_iter().map(|t| t.topic).collect())
        .unwrap_or_default();
    format!(
        "no knowledgepedia entry for \"{}\". Known topics: {}",
        query.trim(),
        if topics.is_empty() {
            "(none)".into()
        } else {
            topics.join(", ")
        }
    )
}

pub struct KnowledgeLookup {
    kb: Knowledgepedia,
}

impl KnowledgeLookup {
    pub fn new(kb: Knowledgepedia) -> Self {
        Self { kb }
    }
}

#[async_trait]
impl Tool for KnowledgeLookup {
    fn name(&self) -> &str {
        KNOWLEDGE_LOOKUP
    }

    fn description(&self) -> &str {
        "Look up verified facts from the knowledgepedia (with sources). Prefer it over your own recall for topics it covers."
    }

    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        Ok(ActionPreview::new(
            format!("look up \"{}\"", a.query.trim()),
            Location::Inside,
        ))
    }

    async fn call(
        &self,
        args: Value,
        _ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        Ok(ToolOutput::ok(lookup(&self.kb, &a.query)))
    }
}
