//! `lookup_table_query(description)` (DESIGN §8.1): proven plan templates for the Orchestrator.

use async_trait::async_trait;
use h_agent_core::RiskClass;
use h_agent_core::config::ScorerConfig;
use h_agent_memory::lookup::{LookupEntry, LookupStore};
use h_agent_safety::Location;
use h_agent_tools::tool::parse_args;
use h_agent_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

pub const LOOKUP_TABLE_QUERY: &str = "lookup_table_query";
const LIMIT: usize = 5;

#[derive(Deserialize)]
struct Args {
    description: String,
}

/// One entry as the Orchestrator sees it.
pub fn describe(e: &LookupEntry, cfg: &ScorerConfig) -> String {
    let s = &e.stats;
    let reuse = if e.reusable(cfg) {
        "reusable"
    } else {
        "not proven enough to reuse"
    };
    let mut out = format!(
        "[{}] score {:.2} \u{b7} {} run(s): {} success, {} partial, {} failed \u{b7} median {}s \u{b7} {reuse}\n  {} ({}/{})",
        e.id,
        e.score,
        s.runs,
        s.success,
        s.partial,
        s.failed,
        s.median_duration_s,
        e.event_signature,
        e.query_type,
        e.topic
    );
    for (i, step) in e.steps.iter().enumerate() {
        let mut extra = Vec::new();
        if !step.tools.is_empty() {
            extra.push(format!("tools {}", step.tools.join(",")));
        }
        if !step.skills.is_empty() {
            extra.push(format!("skills {}", step.skills.join(",")));
        }
        if !step.depends_on.is_empty() {
            let deps: Vec<String> = step.depends_on.iter().map(|d| format!("#{d}")).collect();
            extra.push(format!("after {}", deps.join(",")));
        }
        out.push_str(&format!(
            "\n  {}. {} [{}]{}",
            i + 1,
            step.title,
            step.role,
            if extra.is_empty() {
                String::new()
            } else {
                format!(" ({})", extra.join("; "))
            }
        ));
    }
    out
}

pub struct LookupTableQuery {
    store: LookupStore,
    config: ScorerConfig,
}

impl LookupTableQuery {
    pub fn new(store: LookupStore, config: ScorerConfig) -> Self {
        Self { store, config }
    }
}

#[async_trait]
impl Tool for LookupTableQuery {
    fn name(&self) -> &str {
        LOOKUP_TABLE_QUERY
    }

    fn description(&self) -> &str {
        "Find proven plan templates for a kind of request, with success statistics. \
         Reusable ones can seed plan_create via template_id."
    }

    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": { "description": { "type": "string" } }, "required": ["description"] })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        Ok(ActionPreview::new(
            format!("look up plans for \"{}\"", a.description.trim()),
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
        let table = self
            .store
            .load()
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        let mut hits = table.search(&a.description, LIMIT);
        hits.sort_by(|x, y| y.score.total_cmp(&x.score));
        if hits.is_empty() {
            return Ok(ToolOutput::ok(
                "no matching plan templates; plan from scratch",
            ));
        }
        let text: Vec<String> = hits.iter().map(|e| describe(e, &self.config)).collect();
        Ok(ToolOutput::ok(text.join("\n")))
    }
}
