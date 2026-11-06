use async_trait::async_trait;
use h_agent_core::RiskClass;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tool::{
    ActionPreview, Tool, ToolCtx, ToolError, ToolOutput, decode_text, display_path, parse_args,
    resolve_path,
};

const MAX_FILE_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a text file with line numbers. Optionally limit to a 1-based inclusive line range."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, relative to the working directory" },
                "start_line": { "type": "integer", "minimum": 1, "description": "First line to read (1-based)" },
                "end_line": { "type": "integer", "minimum": 1, "description": "Last line to read (inclusive)" }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Read
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        if a.start_line == Some(0) || a.end_line == Some(0) {
            return Err(ToolError::InvalidArgs("line numbers start at 1".into()));
        }
        if let (Some(s), Some(e)) = (a.start_line, a.end_line)
            && e < s
        {
            return Err(ToolError::InvalidArgs(format!(
                "end_line {e} is before start_line {s}"
            )));
        }
        let (location, path, reasons) = resolve_path(ctx, &a.path);
        let mut p = ActionPreview::new(format!("read {}", a.path), location);
        p.reasons = reasons;
        p.paths = vec![path];
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = &preview.paths[0];
        let shown = display_path(&ctx.env, path);
        let meta = match tokio::fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => return Ok(ToolOutput::error(format!("cannot read {shown}: {e}"))),
        };
        if meta.is_dir() {
            return Ok(ToolOutput::error(format!(
                "{shown} is a directory; use search_files to list it"
            )));
        }
        if meta.len() > MAX_FILE_BYTES {
            return Ok(ToolOutput::error(format!(
                "{shown} is {} bytes; too large to read",
                meta.len()
            )));
        }
        let bytes = tokio::fs::read(path).await?;
        let Some(text) = decode_text(&bytes) else {
            return Ok(ToolOutput::ok(format!(
                "{shown}: binary file, {} bytes",
                bytes.len()
            )));
        };
        Ok(render(
            &text,
            a.start_line,
            a.end_line,
            ctx.limits.read_max_lines,
        ))
    }
}

fn render(text: &str, start: Option<usize>, end: Option<usize>, max_lines: usize) -> ToolOutput {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    if total == 0 {
        return ToolOutput::ok("(empty file)");
    }
    let start = start.unwrap_or(1);
    if start > total {
        return ToolOutput::error(format!(
            "start_line {start} is past the end of the file ({total} lines)"
        ));
    }
    let requested_end = end.unwrap_or(total).min(total);
    let end = requested_end.min(start + max_lines.max(1) - 1);
    let mut out = String::new();
    for (n, line) in lines[start - 1..end].iter().enumerate() {
        out.push_str(&format!("{:>6}\t{line}\n", start + n));
    }
    let truncated = end < requested_end;
    if truncated || start > 1 || end < total {
        out.push_str(&format!("[lines {start}-{end} of {total}]"));
        if truncated {
            out.push_str("; use start_line/end_line to read more");
        }
    }
    ToolOutput {
        content: out,
        is_error: false,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_caps() {
        let text = (1..=10)
            .map(|n| format!("l{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let all = render(&text, None, None, 100);
        assert!(all.content.starts_with("     1\tl1\n") && !all.content.contains("[lines"));
        let part = render(&text, Some(3), Some(4), 100);
        assert_eq!(part.content, "     3\tl3\n     4\tl4\n[lines 3-4 of 10]");
        let capped = render(&text, None, None, 4);
        assert!(capped.truncated && capped.content.contains("[lines 1-4 of 10]; use start_line"));
        assert!(render(&text, Some(11), None, 100).is_error);
        assert_eq!(render("", None, None, 10).content, "(empty file)");
        assert!(
            render("a\r\nb\r\n", None, None, 10)
                .content
                .ends_with("\tb\n")
        );
    }

    #[test]
    fn decodes_boms_and_detects_binary() {
        assert_eq!(decode_text(b"\xEF\xBB\xBFhi").unwrap(), "hi");
        assert_eq!(decode_text(&[0xFF, 0xFE, b'h', 0, b'i', 0]).unwrap(), "hi");
        assert_eq!(decode_text(&[0xFE, 0xFF, 0, b'h', 0, b'i']).unwrap(), "hi");
        assert!(decode_text(b"PNG\0\0data").is_none());
    }
}
