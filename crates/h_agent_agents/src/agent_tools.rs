//! `agent_create` (DESIGN §6.5): the Orchestrator defines a reusable custom sub-agent.

use async_trait::async_trait;
use h_agent_core::RiskClass;
use h_agent_core::events::{NoticeLevel, UiEvent};
use h_agent_memory::agents::{AgentDraft, AgentLibrary, check_name, parse_base};
use h_agent_safety::Location;
use h_agent_tools::tool::parse_args;
use h_agent_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::subagent::{delegable, role_defaults};

pub const AGENT_CREATE: &str = "agent_create";

#[derive(Deserialize)]
struct Args {
    name: String,
    description: String,
    base: String,
    instructions: String,
    #[serde(default)]
    tools: Vec<String>,
}

pub struct AgentCreate {
    library: AgentLibrary,
    ui: mpsc::UnboundedSender<UiEvent>,
}

impl AgentCreate {
    pub fn new(library: AgentLibrary, ui: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self { library, ui }
    }

    fn draft(&self, args: &Value) -> Result<AgentDraft, ToolError> {
        let a: Args = parse_args(args)?;
        let invalid = ToolError::InvalidArgs;
        check_name(&a.name).map_err(invalid)?;
        let base = parse_base(&a.base).map_err(invalid)?;
        if a.description.trim().is_empty() || a.instructions.trim().is_empty() {
            return Err(invalid(
                "description and instructions must not be empty".into(),
            ));
        }
        let bad: Vec<&str> = a
            .tools
            .iter()
            .map(String::as_str)
            .filter(|t| !delegable(t))
            .collect();
        if !bad.is_empty() {
            let mut allowed: Vec<&str> = role_defaults(h_agent_core::Role::Worker)
                .iter()
                .chain(role_defaults(h_agent_core::Role::Research))
                .copied()
                .collect();
            allowed.sort_unstable();
            allowed.dedup();
            return Err(invalid(format!(
                "custom agents cannot use {}; allowed: {} and mcp__* tools",
                bad.join(", "),
                allowed.join(", ")
            )));
        }
        if self.library.load().get(&a.name).is_some() {
            return Err(invalid(format!(
                "a custom agent named `{}` already exists; use it, or pick another name",
                a.name
            )));
        }
        Ok(AgentDraft {
            name: a.name,
            description: a.description,
            base,
            tools: a.tools,
            instructions: a.instructions,
            created_by: "orchestrator".into(),
        })
    }
}

#[async_trait]
impl Tool for AgentCreate {
    fn name(&self) -> &str {
        AGENT_CREATE
    }

    fn description(&self) -> &str {
        "Create a reusable custom sub-agent: a worker or research base role plus its own instructions and optional \
         tool list. Use it when the user asks for one, or when a recurring kind of task needs specialist instructions; \
         not for one-off tasks. Then name it in plan steps with \"agent\"."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "lowercase letters, digits and hyphens, e.g. sql-reviewer" },
                "description": { "type": "string", "description": "One line: what the agent is for" },
                "base": { "type": "string", "enum": ["worker", "research"], "description": "worker edits and runs commands, research only reads and browses" },
                "instructions": { "type": "string", "description": "Markdown instructions: focus, method, and what the report must contain" },
                "tools": { "type": "array", "items": { "type": "string" }, "description": "Optional: replaces the base role's default tools (worker/research tools and mcp__* only)" }
            },
            "required": ["name", "description", "base", "instructions"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Write
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let d = self.draft(args)?;
        let tools = if d.tools.is_empty() {
            format!("{} defaults", d.base)
        } else {
            d.tools.join(", ")
        };
        // Harness data rather than a workspace path, so it follows the inside-workdir rules.
        let mut p = ActionPreview::new(
            format!(
                "create custom agent `{}` ({}; tools: {tools})",
                d.name, d.base
            ),
            Location::Inside,
        );
        p.detail = Some(format!(
            "{}\n\n{}",
            d.description.trim(),
            d.instructions.trim()
        ));
        // The definition persists across sessions and projects, so no mode may skip the user.
        p.always_ask = Some(
            "creating a custom agent changes what future sub-agents do, so it always needs approval"
                .into(),
        );
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        _ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let draft = self.draft(&args)?;
        let agent = self.library.create(&draft).map_err(ToolError::Failed)?;
        let _ = self.ui.send(UiEvent::Notice {
            level: NoticeLevel::Info,
            text: format!(
                "created custom agent `{}` ({}) at {}; remove it with /agent delete {}",
                agent.name,
                agent.base,
                agent.file().display(),
                agent.name
            ),
        });
        Ok(ToolOutput::ok(format!(
            "created custom agent `{}` (base {}). Use it in plan steps with \"agent\": \"{}\".",
            agent.name, agent.base, agent.name
        )))
    }
}
