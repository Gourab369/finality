//! `memory_recall` and `memory_expand` (DESIGN §7.6), answered by the Memory agent.

use std::sync::Arc;

use async_trait::async_trait;
use finality_core::{EventId, RiskClass};
use finality_safety::Location;
use finality_tools::tool::parse_args;
use finality_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::memory::{MemoryAgent, RecallHit};

pub const MEMORY_TOOLS: &[&str] = &["memory_recall", "memory_expand"];

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
}

#[derive(Deserialize)]
struct ExpandArgs {
    event_id: String,
}

pub fn format_hits(hits: &[RecallHit]) -> String {
    if hits.is_empty() {
        return "no matching memories".into();
    }
    hits.iter()
        .map(|h| {
            format!(
                "- [{}] {} · {} · {} (session {})",
                h.event_id, h.date, h.topic, h.summary, h.session_id
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

struct MemoryTool {
    expand: bool,
    memory: Arc<MemoryAgent>,
}

pub fn memory_tools(memory: Arc<MemoryAgent>) -> Vec<Arc<dyn Tool>> {
    [false, true]
        .into_iter()
        .map(|expand| {
            Arc::new(MemoryTool {
                expand,
                memory: memory.clone(),
            }) as Arc<dyn Tool>
        })
        .collect()
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &str {
        MEMORY_TOOLS[usize::from(self.expand)]
    }

    fn description(&self) -> &str {
        if self.expand {
            "Read a condensed transcript of one memory event (from memory_recall) to see the details of past work."
        } else {
            "Search memories of earlier sessions. Returns up to 10 events with ids, dates, topics and summaries."
        }
    }

    fn schema(&self) -> Value {
        if self.expand {
            json!({ "type": "object", "properties": { "event_id": { "type": "string" } }, "required": ["event_id"] })
        } else {
            json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
        }
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let summary = if self.expand {
            let a: ExpandArgs = parse_args(args)?;
            a.event_id
                .trim()
                .parse::<EventId>()
                .map_err(|e| ToolError::InvalidArgs(format!("event_id: {e}")))?;
            format!("expand memory {}", a.event_id.trim())
        } else {
            let a: RecallArgs = parse_args(args)?;
            if a.query.trim().is_empty() {
                return Err(ToolError::InvalidArgs("query must not be empty".into()));
            }
            format!("recall \"{}\"", a.query.trim())
        };
        Ok(ActionPreview::new(summary, Location::Inside))
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let work = async {
            if self.expand {
                let a: ExpandArgs = parse_args(&args)?;
                let id = a
                    .event_id
                    .trim()
                    .parse::<EventId>()
                    .map_err(|e| ToolError::InvalidArgs(format!("event_id: {e}")))?;
                Ok(match self.memory.expand(id).await {
                    Ok(text) => ToolOutput::ok(text),
                    Err(e) => ToolOutput::error(e),
                })
            } else {
                let a: RecallArgs = parse_args(&args)?;
                Ok(ToolOutput::ok(format_hits(
                    &self.memory.recall(a.query.trim()).await,
                )))
            }
        };
        tokio::select! {
            out = work => out,
            () = ctx.cancel.cancelled() => Ok(ToolOutput::error("cancelled")),
        }
    }
}
