//! MCP tools as deferred, External-risk registry tools named `mcp__<server>__<tool>`.

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_safety::Location;
use finality_tools::{ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput};
use serde_json::{Value, json};

use crate::hub::{CachedTool, Server};

const MAX_NAME: usize = 64;
const MAX_DESCRIPTION: usize = 1024;
const PREVIEW_ARGS_CHARS: usize = 2000;

/// `mcp__<server>__<tool>` with characters providers reject replaced by `_`, cut at 64. Runs of
/// underscores inside a part collapse to one, so `__` only ever separates server from tool.
pub fn tool_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            let c = if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            };
            if c == '_' && out.ends_with('_') {
                continue;
            }
            out.push(c);
        }
        out
    };
    format!("mcp__{}__{}", clean(server), clean(tool))
        .chars()
        .take(MAX_NAME)
        .collect()
}

/// Distinct exposed names for a server's tools, in order; a repeat gets a numeric suffix.
pub fn unique_tool_names(server: &str, tools: &[String]) -> Vec<String> {
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
    tools
        .iter()
        .map(|t| {
            let base = tool_name(server, t);
            let mut name = base.clone();
            let mut n = 2;
            while !taken.insert(name.clone()) {
                let suffix = format!("_{n}");
                name = base
                    .chars()
                    .take(MAX_NAME - suffix.len())
                    .collect::<String>()
                    + &suffix;
                n += 1;
            }
            name
        })
        .collect()
}

pub struct McpTool {
    server: Weak<Server>,
    server_name: String,
    remote: String,
    name: String,
    description: String,
    schema: Value,
}

impl McpTool {
    pub(crate) fn new(server: &Arc<Server>, tool: &CachedTool, name: String) -> Self {
        let mut description = format!("[MCP server {}] {}", server.name(), tool.description.trim());
        if description.chars().count() > MAX_DESCRIPTION {
            description = description
                .chars()
                .take(MAX_DESCRIPTION)
                .collect::<String>()
                + "…";
        }
        let schema = match &tool.input_schema {
            Value::Object(m) if !m.is_empty() => {
                let mut m = m.clone();
                m.entry("type").or_insert_with(|| json!("object"));
                Value::Object(m)
            }
            _ => json!({ "type": "object", "properties": {} }),
        };
        Self {
            server: Arc::downgrade(server),
            server_name: server.name().to_owned(),
            remote: tool.name.clone(),
            name,
            description,
            schema,
        }
    }
}

/// Text for the model from a `CallToolResult`; the flag is the result's `isError`.
pub fn render_result(result: &Value) -> (String, bool) {
    let is_error = result.get("isError").and_then(Value::as_bool) == Some(true);
    let mut parts: Vec<String> = Vec::new();
    for item in result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let field = |k: &str| item.get(k).and_then(Value::as_str).unwrap_or("");
        parts.push(match field("type") {
            "text" => field("text").to_owned(),
            kind @ ("image" | "audio") => format!(
                "[{kind}: {}, {} bytes base64]",
                field("mimeType"),
                field("data").len()
            ),
            "resource" => {
                let r = item.get("resource").cloned().unwrap_or(Value::Null);
                let get = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
                if r.get("text").is_some() {
                    format!("[resource {}]\n{}", get("uri"), get("text"))
                } else {
                    format!("[resource {} ({}), binary]", get("uri"), get("mimeType"))
                }
            }
            "resource_link" => format!("[resource link: {} {}]", field("uri"), field("name")),
            _ => item.to_string(),
        });
    }
    if parts.iter().all(|p| p.trim().is_empty())
        && let Some(structured) = result.get("structuredContent")
    {
        parts = vec![serde_json::to_string_pretty(structured).unwrap_or_default()];
    }
    let text = parts.join("\n");
    (
        if text.trim().is_empty() {
            "(no content)".into()
        } else {
            text
        },
        is_error,
    )
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    fn risk(&self) -> RiskClass {
        RiskClass::External
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        if !matches!(args, Value::Object(_) | Value::Null) {
            return Err(ToolError::InvalidArgs("arguments must be an object".into()));
        }
        let mut p = ActionPreview::new(
            format!(
                "call `{}` on MCP server `{}`",
                self.remote, self.server_name
            ),
            Location::Inside,
        );
        let pretty = serde_json::to_string_pretty(args).unwrap_or_default();
        let (detail, _) = finality_tools::tool::truncate_middle(&pretty, PREVIEW_ARGS_CHARS);
        p.detail = Some(detail);
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let server = self
            .server
            .upgrade()
            .ok_or_else(|| ToolError::Failed("MCP is shutting down".into()))?;
        let result = match server.call(&self.remote, args, &ctx.cancel).await {
            Ok(r) => r,
            Err(e) if ctx.cancel.is_cancelled() => {
                tracing::debug!("{e}");
                return Err(ToolError::Cancelled);
            }
            Err(e) => return Ok(ToolOutput::error(e)),
        };
        let (text, is_error) = render_result(&result);
        let (content, truncated) =
            finality_tools::tool::truncate_middle(&text, ctx.limits.max_output_chars);
        Ok(ToolOutput {
            content,
            is_error,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_sanitized_and_capped() {
        assert_eq!(
            tool_name("github", "create_issue"),
            "mcp__github__create_issue"
        );
        assert_eq!(tool_name("fs", "read.file v2"), "mcp__fs__read_file_v2");
        assert_eq!(tool_name("s", &"x".repeat(100)).len(), 64);
        assert_ne!(
            tool_name("a__b", "c"),
            tool_name("a", "b__c"),
            "the separator cannot be forged by a name"
        );
        assert_eq!(tool_name("a__b", "c"), "mcp__a_b__c");
    }

    #[test]
    fn repeated_names_get_suffixes() {
        let long = "x".repeat(100);
        let names = unique_tool_names(
            "s",
            &[
                "read_file".into(),
                "read__file".into(),
                "read.file".into(),
                long.clone(),
                long,
            ],
        );
        assert_eq!(
            &names[..3],
            [
                "mcp__s__read_file",
                "mcp__s__read_file_2",
                "mcp__s__read_file_3"
            ]
        );
        assert_eq!(names[3].len(), 64);
        assert!(names[4].ends_with("_2") && names[4].len() == 64);
        assert_eq!(
            names.iter().collect::<std::collections::HashSet<_>>().len(),
            5
        );
    }

    #[test]
    fn renders_content_blocks() {
        let r = json!({
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "image", "mimeType": "image/png", "data": "AAAA" },
                { "type": "resource", "resource": { "uri": "file:///a.txt", "text": "body" } },
                { "type": "resource_link", "uri": "file:///b", "name": "b" }
            ],
            "isError": false
        });
        let (text, err) = render_result(&r);
        assert!(!err);
        assert_eq!(
            text,
            "hello\n[image: image/png, 4 bytes base64]\n[resource file:///a.txt]\nbody\n[resource link: file:///b b]"
        );
        let (text, err) = render_result(
            &json!({ "content": [], "structuredContent": { "n": 1 }, "isError": true }),
        );
        assert!(err);
        assert_eq!(text, "{\n  \"n\": 1\n}");
        assert_eq!(render_result(&json!({})).0, "(no content)");
    }
}
