//! `skill_load(name)` (DESIGN §8.4): loads a skill body from the catalog into the conversation.

use async_trait::async_trait;
use h_agent_core::RiskClass;
use h_agent_memory::skills::{SKILL_FILE, Skill, SkillLibrary};
use h_agent_safety::Location;
use h_agent_tools::tool::parse_args;
use h_agent_tools::{ActionPreview, Tool, ToolCtx, ToolError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

pub const SKILL_LOAD: &str = "skill_load";

#[derive(Deserialize)]
struct Args {
    name: String,
}

/// The text an agent sees for a loaded skill.
pub fn render_skill(skill: &Skill) -> String {
    let listing = if skill.is_builtin() {
        Err(std::io::ErrorKind::NotFound.into())
    } else {
        std::fs::read_dir(&skill.dir)
    };
    let mut extra: Vec<String> = listing
        .map(|d| {
            d.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != SKILL_FILE)
                .collect()
        })
        .unwrap_or_default();
    extra.sort();
    let mut out = format!(
        "<skill name=\"{}\">\n{}\n</skill>",
        skill.name,
        crate::conversation::fence("skill", &skill.body)
    );
    if !extra.is_empty() {
        out.push_str(&format!(
            "\nOther files in the skill folder: {}",
            extra.join(", ")
        ));
    }
    out
}

pub struct SkillLoad {
    library: SkillLibrary,
}

impl SkillLoad {
    pub fn new(library: SkillLibrary) -> Self {
        Self { library }
    }
}

#[async_trait]
impl Tool for SkillLoad {
    fn name(&self) -> &str {
        SKILL_LOAD
    }

    fn description(&self) -> &str {
        "Load the full instructions of a skill from the skills catalog before doing a task it covers."
    }

    fn schema(&self) -> Value {
        json!({ "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Internal
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        Ok(ActionPreview::new(
            format!("load skill {}", a.name.trim()),
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
        let catalog = self.library.load();
        Ok(match catalog.get(a.name.trim()) {
            Some(skill) => ToolOutput::ok(render_skill(skill)),
            None => {
                let names: Vec<&str> = catalog.active.iter().map(|s| s.name.as_str()).collect();
                ToolOutput::error(format!(
                    "no skill `{}`; available: {}",
                    a.name.trim(),
                    if names.is_empty() {
                        "(none)".into()
                    } else {
                        names.join(", ")
                    }
                ))
            }
        })
    }
}
