use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_safety::Location;
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
    Delete,
}

impl Op {
    fn kind(&self) -> &'static str {
        match self {
            Op::Insert { .. } => "insert",
            Op::ReplaceLines { .. } => "replace_lines",
            Op::ReplaceText { .. } => "replace_text",
            Op::Write { .. } => "write",
            Op::Delete => "delete",
        }
    }
}

fn is_delete(ops: &[Op]) -> bool {
    matches!(ops, [Op::Delete])
}

/// Size and modification time, to notice a file changing between review and deletion.
fn stamp(meta: &std::fs::Metadata) -> Value {
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    json!({ "len": meta.len(), "modified": modified })
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
                Op::Delete => "delete the file".to_owned(),
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

fn indent(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Replaces the one run of whole lines that equals `old` when each line's surrounding whitespace is
/// ignored, shifting `new`'s indentation by the difference. `Ok(None)` when there is no such run.
fn fuzzy_replace(text: &str, old: &str, new: &str) -> Result<Option<String>, String> {
    let old_lines: Vec<&str> = old.trim_matches('\n').split('\n').collect();
    if old_lines.iter().all(|l| l.trim().is_empty()) {
        return Ok(None);
    }
    let (mut lines, trailing) = split_lines(text);
    if old_lines.len() > lines.len() {
        return Ok(None);
    }
    let starts: Vec<usize> = (0..=lines.len() - old_lines.len())
        .filter(|&s| {
            old_lines
                .iter()
                .enumerate()
                .all(|(k, l)| lines[s + k].trim() == l.trim())
        })
        .collect();
    match starts.as_slice() {
        [] => Ok(None),
        &[s] => {
            let k = old_lines
                .iter()
                .position(|l| !l.trim().is_empty())
                .unwrap_or(0);
            let from = indent(old_lines[k]).to_owned();
            let to = indent(&lines[s + k]).to_owned();
            let replacement: Vec<String> = content_lines(new)
                .into_iter()
                .map(|l| {
                    if l.trim().is_empty() {
                        String::new()
                    } else if let Some(rest) = l.strip_prefix(from.as_str()) {
                        format!("{to}{rest}")
                    } else {
                        l
                    }
                })
                .collect();
            lines.splice(s..s + old_lines.len(), replacement);
            Ok(Some(join_lines(&lines, trailing)))
        }
        many => Err(format!(
            "`old` is not in the file exactly, and matches {} places when indentation is ignored (lines {}); \
             include more surrounding lines",
            many.len(),
            many.iter()
                .map(|s| (s + 1).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// "Not found" with the file lines most like `old`'s first line, so the model can correct it without rereading.
fn not_found(text: &str, old: &str) -> String {
    let mut msg =
        "`old` text not found; it must match the file exactly (indentation may differ only when \
                   `old` is whole lines)"
            .to_owned();
    let Some(probe) = old.split('\n').map(str::trim).find(|l| !l.is_empty()) else {
        return msg;
    };
    let probe: String = probe.chars().take(200).collect();
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.len() > 20_000 {
        return msg;
    }
    let mut scored: Vec<(f32, usize)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let l: String = l.trim().chars().take(200).collect();
            (TextDiff::from_chars(probe.as_str(), l.as_str()).ratio(), i)
        })
        .filter(|(r, _)| *r >= 0.6)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    if !scored.is_empty() {
        msg.push_str("; closest lines:");
        for &(_, i) in scored.iter().take(3) {
            msg.push_str(&format!(
                "\n{}: {}",
                i + 1,
                truncate_middle(lines[i], 200).0
            ));
        }
    }
    msg
}

/// Applies operations in order; line numbers refer to the content produced by earlier operations.
fn apply(original: Option<&str>, ops: &[Op]) -> Result<String, ToolError> {
    if ops.is_empty() {
        return Err(ToolError::InvalidArgs(
            "operations must not be empty".into(),
        ));
    }
    if ops.iter().any(|op| matches!(op, Op::Delete)) {
        return Err(ToolError::InvalidArgs(
            "`delete` must be the only operation".into(),
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
            Op::Delete => {}
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
                    None if count == 0 => match fuzzy_replace(&text, &old, &new) {
                        Ok(Some(t)) => text = t,
                        Ok(None) => return Err(err(not_found(&text, &old))),
                        Err(m) => return Err(err(m)),
                    },
                    None => {
                        return Err(err(format!(
                            "`old` matches {count} times; pass occurrence (1-based) or \"all\""
                        )));
                    }
                    Some(Occurrence::Word(w)) if w == "all" => {
                        if count == 0 {
                            return Err(err(not_found(&text, &old)));
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
         occurrence is a 1-based number or \"all\" (when `old` is whole lines, differing indentation is \
         tolerated and `new` is re-indented to match); write {content} replaces or creates the whole file; \
         delete {} deletes the file and must be the only operation. Changes are saved in checkpoints and \
         can be undone."
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
                            "op": { "type": "string", "enum": ["insert", "replace_lines", "replace_text", "write", "delete"] },
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
        let delete = is_delete(&a.operations);
        let (location, path, reasons) = resolve_path(ctx, &a.path);
        let shown = display_path(&ctx.env, &path);
        if location != Location::Inside {
            // Read nothing before the user decides, or the model could probe files it may not see.
            let summary = if delete {
                format!("delete {shown}")
            } else {
                format!("edit {shown} ({} operation(s))", a.operations.len())
            };
            let mut p = ActionPreview::new(summary, location);
            p.reasons = reasons;
            p.paths = vec![path];
            p.detail = Some(describe_ops(&a.operations));
            p.deletes = delete;
            p.state = json!({ "deferred": true });
            return Ok(p);
        }
        if delete {
            let meta = std::fs::metadata(&path)
                .map_err(|_| ToolError::Failed(format!("{shown} does not exist")))?;
            if !meta.is_file() {
                return Err(ToolError::Failed(format!(
                    "{shown} is not a file; only files can be deleted"
                )));
            }
            let mut p = ActionPreview::new(format!("delete {shown}"), location);
            p.reasons = reasons;
            p.paths = vec![path];
            p.detail = Some(format!("delete {shown} ({} bytes)", meta.len()));
            p.deletes = true;
            p.state = json!({ "stamp": stamp(&meta) });
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
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = preview.paths[0].clone();
        let shown = display_path(&ctx.env, &path);
        let deferred = preview.state["deferred"].as_bool().unwrap_or(false);
        let changed = "the file changed after the edit was reviewed; read it again and retry";
        if is_delete(&a.operations) {
            let meta = match std::fs::metadata(&path) {
                Ok(m) if m.is_file() => m,
                _ => return Ok(ToolOutput::error(format!("{shown} is not a file"))),
            };
            if !deferred && stamp(&meta) != preview.state["stamp"] {
                return Ok(ToolOutput::error(changed));
            }
            if let Some(cps) = &ctx.env.checkpoints {
                cps.record(&path, &shown).map_err(|e| {
                    ToolError::Failed(format!(
                        "could not save {shown} for undo, so it was not deleted: {e}"
                    ))
                })?;
            }
            std::fs::remove_file(&path)?;
            return Ok(ToolOutput::ok(format!("deleted {shown}")));
        }
        let original = load(&path)?;
        if !deferred && json!(original.fingerprint) != preview.state["fingerprint"] {
            return Ok(ToolOutput::error(changed));
        }
        let new_text = apply(original.text.as_deref(), &a.operations)?;
        let undo_note = match &ctx.env.checkpoints {
            Some(cps) => match cps.record(&path, &shown) {
                Ok(()) => String::new(),
                Err(e) => format!("\n(not saved for undo: {e})"),
            },
            None => String::new(),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        finality_store::write_atomic(&path, &render(&original, &new_text))?;
        let (diff_text, _, _) = diff(original.text.as_deref().unwrap_or(""), &new_text, "");
        let (diff_text, _) = truncate_middle(&diff_text, MAX_DIFF_IN_OUTPUT);
        Ok(ToolOutput::ok(format!(
            "{}{undo_note}\n{diff_text}",
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
    fn whitespace_tolerant_replace_reindents() {
        let src = "fn main() {\n    if x {\n        run();\n    }\n}\n";
        let out = apply(
            Some(src),
            &ops(json!([{ "op": "replace_text", "old": "if x {\n    run();\n}", "new": "if x {\n    run();\n    stop();\n}" }])),
        )
        .unwrap();
        assert_eq!(
            out,
            "fn main() {\n    if x {\n        run();\n        stop();\n    }\n}\n"
        );

        let err = apply(
            Some("a\n  b\nc\n    b\n"),
            &ops(json!([{ "op": "replace_text", "old": "\tb", "new": "x" }])),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("matches 2 places") && err.contains("lines 2, 4"),
            "{err}"
        );
    }

    #[test]
    fn not_found_names_the_closest_lines() {
        let err = apply(
            Some("let total = price * qty;\nprintln!(\"{total}\");\n"),
            &ops(json!([{ "op": "replace_text", "old": "let total = price * quantity;", "new": "" }])),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("not found") && err.contains("\n1: let total = price * qty;"),
            "{err}"
        );
    }

    #[test]
    fn delete_must_be_alone() {
        let err = apply(
            Some("a\n"),
            &ops(json!([{ "op": "delete" }, { "op": "write", "content": "x" }])),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("only operation"), "{err}");
        assert!(is_delete(&ops(json!([{ "op": "delete" }]))));
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
