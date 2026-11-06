use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;

use async_trait::async_trait;
use h_agent_core::RiskClass;
use h_agent_safety::Location;
use serde::Deserialize;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};

use crate::tool::{
    ActionPreview, Tool, ToolCtx, ToolError, ToolOutput, display_path, parse_args, resolve_path,
    truncate_middle,
};

const UTF8_BOM: &str = "\u{feff}";
const MAX_DIFF_IN_OUTPUT: usize = 4000;
const MAX_EDIT_BYTES: u64 = 8 * 1024 * 1024;
const OP_PREVIEW_CHARS: usize = 300;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Occurrence {
    Nth(usize),
    Word(String),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Op {
    Insert {
        line: usize,
        content: String,
    },
    ReplaceLines {
        start_line: usize,
        end_line: usize,
        content: String,
    },
    ReplaceText {
        old: String,
        new: String,
        #[serde(default)]
        occurrence: Option<Occurrence>,
    },
    Write {
        content: String,
    },
}

impl Op {
    fn kind(&self) -> &'static str {
        match self {
            Op::Insert { .. } => "insert",
            Op::ReplaceLines { .. } => "replace_lines",
            Op::ReplaceText { .. } => "replace_text",
            Op::Write { .. } => "write",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
    operations: Vec<Op>,
}

/// Original file as seen by the edit; `None` when it does not exist yet.
struct Original {
    text: Option<String>,
    crlf: bool,
    bom: bool,
    fingerprint: Option<u64>,
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn load(path: &Path) -> Result<Original, ToolError> {
    if let Ok(meta) = std::fs::metadata(path)
        && meta.is_file()
        && meta.len() > MAX_EDIT_BYTES
    {
        return Err(ToolError::Failed(format!(
            "{} is larger than {} MiB; edit refused",
            path.display(),
            MAX_EDIT_BYTES / (1024 * 1024)
        )));
    }
    match std::fs::read(path) {
        Ok(bytes) => {
            let fp = fingerprint(&bytes);
            let text = String::from_utf8(bytes)
                .map_err(|_| ToolError::Failed("file is not UTF-8 text; edit refused".into()))?;
            let bom = text.starts_with(UTF8_BOM);
            let text = text.strip_prefix(UTF8_BOM).unwrap_or(&text).to_owned();
            Ok(Original {
                crlf: text.contains("\r\n"),
                text: Some(text.replace("\r\n", "\n")),
                bom,
                fingerprint: Some(fp),
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Original {
            text: None,
            crlf: false,
            bom: false,
            fingerprint: None,
        }),
        Err(_) if path.is_dir() => Err(ToolError::Failed(format!(
            "{} is a directory",
            path.display()
        ))),
        Err(e) => Err(e.into()),
    }
}

fn content_lines(content: &str) -> Vec<String> {
    let c = content.replace("\r\n", "\n");
    if c.is_empty() {
        return Vec::new();
    }
    let c = c.strip_suffix('\n').unwrap_or(&c);
    c.split('\n').map(str::to_owned).collect()
}

/// The operations as text, for approval prompts where no diff can be shown yet.
fn describe_ops(ops: &[Op]) -> String {
    let clip = |s: &str| truncate_middle(s, OP_PREVIEW_CHARS).0;
    ops.iter()
        .enumerate()
        .map(|(i, op)| {
            let what = match op {
                Op::Insert { line, content } => {
                    format!("insert before line {line}:\n{}", clip(content))
                }
                Op::ReplaceLines {
                    start_line,
                    end_line,
                    content,
                } => format!(
                    "replace lines {start_line}-{end_line} with:\n{}",
                    clip(content)
                ),
                Op::ReplaceText { old, new, .. } => {
                    format!("replace:\n{}\nwith:\n{}", clip(old), clip(new))
                }
                Op::Write { content } => format!(
                    "write the whole file ({} lines)",
                    content_lines(content).len()
                ),
            };
            format!("{}. {what}", i + 1)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn split_lines(text: &str) -> (Vec<String>, bool) {
    if text.is_empty() {
        return (Vec::new(), true);
    }
    let trailing = text.ends_with('\n');
    let body = text.strip_suffix('\n').unwrap_or(text);
    (body.split('\n').map(str::to_owned).collect(), trailing)
}

fn join_lines(lines: &[String], trailing: bool) -> String {
    let mut s = lines.join("\n");
    if trailing && !lines.is_empty() {
        s.push('\n');
    }
    s
}

/// Applies operations in order; line numbers refer to the content produced by earlier operations.
fn apply(original: Option<&str>, ops: &[Op]) -> Result<String, ToolError> {
    if ops.is_empty() {
        return Err(ToolError::InvalidArgs(
            "operations must not be empty".into(),
        ));
    }
    if original.is_none() && !matches!(ops[0], Op::Write { .. }) {
        return Err(ToolError::InvalidArgs(
            "file does not exist; start with a `write` operation to create it".into(),
        ));
    }
    let mut text = original.unwrap_or_default().to_owned();
    for (i, op) in ops.iter().enumerate() {
        let err =
            |m: String| ToolError::InvalidArgs(format!("operation {} ({}): {m}", i + 1, op.kind()));
        match op {
            Op::Write { content } => text = content.replace("\r\n", "\n"),
            Op::Insert { line, content } => {
                let (mut lines, trailing) = split_lines(&text);
                if *line == 0 || *line > lines.len() + 1 {
                    return Err(err(format!(
                        "line {line} is out of range 1..={}",
                        lines.len() + 1
                    )));
                }
                let at = line - 1;
                lines.splice(at..at, content_lines(content));
                text = join_lines(&lines, trailing);
            }
            Op::ReplaceLines {
                start_line,
                end_line,
                content,
            } => {
                let (mut lines, trailing) = split_lines(&text);
                if *start_line == 0 || start_line > end_line || *end_line > lines.len() {
                    return Err(err(format!(
                        "range {start_line}-{end_line} is invalid for a file with {} lines",
                        lines.len()
                    )));
                }
                lines.splice(start_line - 1..*end_line, content_lines(content));
                text = join_lines(&lines, trailing);
            }
            Op::ReplaceText {
                old,
                new,
                occurrence,
            } => {
                let old = old.replace("\r\n", "\n");
                let new = new.replace("\r\n", "\n");
                if old.is_empty() {
                    return Err(err("`old` must not be empty".into()));
                }
                let count = text.matches(old.as_str()).count();
                match occurrence {
                    None if count == 1 => text = text.replacen(&old, &new, 1),
                    None if count == 0 => return Err(err("`old` text not found".into())),
                    None => {
                        return Err(err(format!(
                            "`old` matches {count} times; pass occurrence (1-based) or \"all\""
                        )));
                    }
                    Some(Occurrence::Word(w)) if w == "all" => {
                        if count == 0 {
                            return Err(err("`old` text not found".into()));
                        }
                        text = text.replace(&old, &new);
                    }
                    Some(Occurrence::Word(w)) => {
                        return Err(err(format!("occurrence `{w}` must be a number or \"all\"")));
                    }
                    Some(Occurrence::Nth(n)) => {
                        if *n == 0 || *n > count {
                            return Err(err(format!("occurrence {n} is out of range 1..={count}")));
                        }
                        let (start, _) = text
                            .match_indices(old.as_str())
                            .nth(n - 1)
                            .unwrap_or((0, ""));
                        text.replace_range(start..start + old.len(), &new);
                    }
                }
            }
        }
    }
    Ok(text)
}

fn render(original: &Original, new_text: &str) -> Vec<u8> {
    let mut out = if original.crlf {
        new_text.replace('\n', "\r\n")
    } else {
        new_text.to_owned()
    };
    if original.bom {
        out.insert_str(0, UTF8_BOM);
    }
    out.into_bytes()
}

fn diff(old: &str, new: &str, name: &str) -> (String, usize, usize) {
    let d = TextDiff::from_lines(old, new);
    let (mut added, mut removed) = (0, 0);
    for c in d.iter_all_changes() {
        match c.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }
    let text = d
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{name}"), &format!("b/{name}"))
        .to_string();
    (text, added, removed)
}

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit or create a text file with a batch of operations applied in order (line numbers refer to the \
         result of earlier operations). The batch is all-or-nothing. Operations: insert {line, content} \
         inserts before `line` (use last+1 to append); replace_lines {start_line, end_line, content} \
         (empty content deletes); replace_text {old, new, occurrence?} where `old` must be unique unless \
         occurrence is a 1-based number or \"all\"; write {content} replaces or creates the whole file."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, relative to the working directory" },
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": { "type": "string", "enum": ["insert", "replace_lines", "replace_text", "write"] },
                            "line": { "type": "integer", "minimum": 1 },
                            "start_line": { "type": "integer", "minimum": 1 },
                            "end_line": { "type": "integer", "minimum": 1 },
                            "content": { "type": "string" },
                            "old": { "type": "string" },
                            "new": { "type": "string" },
                            "occurrence": { "oneOf": [ { "type": "integer", "minimum": 1 }, { "const": "all" } ] }
                        },
                        "required": ["op"]
                    }
                }
            },
            "required": ["path", "operations"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Write
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        let (location, path, reasons) = resolve_path(ctx, &a.path);
        let shown = display_path(&ctx.env, &path);
        if location != Location::Inside {
            // Read nothing before the user decides, or the model could probe files it may not see.
            let mut p = ActionPreview::new(
                format!("edit {shown} ({} operation(s))", a.operations.len()),
                location,
            );
            p.reasons = reasons;
            p.paths = vec![path];
            p.detail = Some(describe_ops(&a.operations));
            p.state = json!({ "deferred": true });
            return Ok(p);
        }
        let original = load(&path)?;
        let new_text = apply(original.text.as_deref(), &a.operations)?;
        let (diff_text, added, removed) =
            diff(original.text.as_deref().unwrap_or(""), &new_text, &shown);
        let summary = match original.text {
            None => format!("create {shown} ({added} lines)"),
            Some(_) => format!("edit {shown} (+{added} -{removed})"),
        };
        let mut p = ActionPreview::new(summary, location);
        p.reasons = reasons;
        p.paths = vec![path];
        p.detail = Some(diff_text);
        p.state = json!({ "fingerprint": original.fingerprint });
        Ok(p)
    }

    async fn call(
        &self,
        args: Value,
        _ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = preview.paths[0].clone();
        let original = load(&path)?;
        let deferred = preview.state["deferred"].as_bool().unwrap_or(false);
        if !deferred && json!(original.fingerprint) != preview.state["fingerprint"] {
            return Ok(ToolOutput::error(
                "the file changed after the edit was reviewed; read it again and retry",
            ));
        }
        let new_text = apply(original.text.as_deref(), &a.operations)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        h_agent_store::write_atomic(&path, &render(&original, &new_text))?;
        let (diff_text, _, _) = diff(original.text.as_deref().unwrap_or(""), &new_text, "");
        let (diff_text, _) = truncate_middle(&diff_text, MAX_DIFF_IN_OUTPUT);
        Ok(ToolOutput::ok(format!(
            "{}\n{diff_text}",
            preview
                .summary
                .replacen("edit ", "edited ", 1)
                .replacen("create ", "created ", 1)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops(v: Value) -> Vec<Op> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn line_operations() {
        let src = "a\nb\nc\n";
        let out = apply(
            Some(src),
            &ops(json!([
                { "op": "insert", "line": 1, "content": "top" },
                { "op": "insert", "line": 5, "content": "end\n" },
                { "op": "replace_lines", "start_line": 3, "end_line": 3, "content": "B1\nB2" },
                { "op": "replace_lines", "start_line": 5, "end_line": 5, "content": "" }
            ])),
        )
        .unwrap();
        assert_eq!(out, "top\na\nB1\nB2\nend\n");
    }

    #[test]
    fn text_operations() {
        let src = "x = 1\ny = 1\nz = 2\n";
        assert_eq!(
            apply(
                Some(src),
                &ops(json!([{ "op": "replace_text", "old": "z = 2", "new": "z = 3" }]))
            )
            .unwrap(),
            "x = 1\ny = 1\nz = 3\n"
        );
        let err = apply(
            Some(src),
            &ops(json!([{ "op": "replace_text", "old": "= 1", "new": "= 9" }])),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("matches 2 times"), "{err}");
        assert_eq!(
            apply(
                Some(src),
                &ops(
                    json!([{ "op": "replace_text", "old": "= 1", "new": "= 9", "occurrence": 2 }])
                )
            )
            .unwrap(),
            "x = 1\ny = 9\nz = 2\n"
        );
        assert_eq!(
            apply(Some(src), &ops(json!([{ "op": "replace_text", "old": "= 1", "new": "= 9", "occurrence": "all" }]))).unwrap(),
            "x = 9\ny = 9\nz = 2\n"
        );
        assert!(
            apply(
                Some(src),
                &ops(json!([{ "op": "replace_text", "old": "nope", "new": "" }]))
            )
            .is_err()
        );
    }

    #[test]
    fn failing_operation_is_named() {
        let err = apply(
            Some("a\n"),
            &ops(json!([
                { "op": "insert", "line": 1, "content": "x" },
                { "op": "replace_lines", "start_line": 2, "end_line": 9, "content": "" }
            ])),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("operation 2 (replace_lines)"), "{err}");
    }

    #[test]
    fn create_requires_write_first() {
        assert!(
            apply(
                None,
                &ops(json!([{ "op": "insert", "line": 1, "content": "x" }]))
            )
            .is_err()
        );
        assert_eq!(
            apply(None, &ops(json!([{ "op": "write", "content": "new\n" }]))).unwrap(),
            "new\n"
        );
        assert_eq!(
            apply(
                Some(""),
                &ops(json!([{ "op": "insert", "line": 1, "content": "x" }]))
            )
            .unwrap(),
            "x\n"
        );
    }

    #[test]
    fn preserves_crlf_and_bom() {
        let original = Original {
            text: Some("a\nb\n".into()),
            crlf: true,
            bom: true,
            fingerprint: None,
        };
        assert_eq!(render(&original, "a\nc\n"), "\u{feff}a\r\nc\r\n".as_bytes());
    }
}
