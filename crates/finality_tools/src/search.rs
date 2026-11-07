use std::path::Path;

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_safety::PathGuard;
use globset::{Glob, GlobMatcher};
use regex::{Regex, RegexBuilder};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::tool::{
    ActionPreview, Tool, ToolCtx, ToolError, ToolOutput, decode_text, parse_args, resolve_path,
};

const MAX_SEARCH_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_LINE_CHARS: usize = 300;
const DEFAULT_RESULTS: usize = 100;
const MAX_RESULTS: usize = 1000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Target {
    Name,
    #[default]
    Content,
    Both,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    pattern: String,
    #[serde(default)]
    target: Target,
    path: Option<String>,
    include_glob: Option<String>,
    #[serde(default)]
    case_sensitive: bool,
    max_results: Option<usize>,
}

struct Query {
    regex: Regex,
    glob: Option<GlobMatcher>,
    target: Target,
    max: usize,
}

fn build_query(a: &Args) -> Result<Query, ToolError> {
    let regex = RegexBuilder::new(&a.pattern)
        .case_insensitive(!a.case_sensitive)
        .size_limit(1 << 20)
        .build()
        .map_err(|e| ToolError::InvalidArgs(format!("invalid regex: {e}")))?;
    let glob = a
        .include_glob
        .as_deref()
        .map(|g| Glob::new(g).map(|g| g.compile_matcher()))
        .transpose()
        .map_err(|e| ToolError::InvalidArgs(format!("invalid include_glob: {e}")))?;
    Ok(Query {
        regex,
        glob,
        target: a.target,
        max: a
            .max_results
            .unwrap_or(DEFAULT_RESULTS)
            .clamp(1, MAX_RESULTS),
    })
}

pub struct SearchFiles;

#[async_trait]
impl Tool for SearchFiles {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search the working directory by regex. target=name matches file and folder names, content matches \
         lines inside files (default), both does both. Respects .gitignore and never follows symlinks."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression" },
                "target": { "type": "string", "enum": ["name", "content", "both"], "default": "content" },
                "path": { "type": "string", "description": "Subdirectory or file to search (default: working directory)" },
                "include_glob": { "type": "string", "description": "Only files whose relative path matches, e.g. \"*.rs\"" },
                "case_sensitive": { "type": "boolean", "default": false },
                "max_results": { "type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": DEFAULT_RESULTS }
            },
            "required": ["pattern"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Read
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: Args = parse_args(args)?;
        build_query(&a)?;
        let raw = a.path.as_deref().unwrap_or(".");
        let (location, path, reasons) = resolve_path(ctx, raw);
        let mut p = ActionPreview::new(format!("search `{}` in {raw}", a.pattern), location);
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
        let query = build_query(&parse_args(&args)?)?;
        let root = preview.paths[0].clone();
        let env = ctx.env.clone();
        let cancel = ctx.cancel.clone();
        let (results, more) =
            tokio::task::spawn_blocking(move || search(&root, &env.guard, &query, &cancel))
                .await
                .map_err(|e| ToolError::Failed(format!("search task failed: {e}")))??;
        if results.is_empty() {
            return Ok(ToolOutput::ok("no matches"));
        }
        let mut out = results.join("\n");
        if more {
            out.push_str(&format!(
                "\n[results limited to {}; narrow the search]",
                results.len()
            ));
        }
        Ok(ToolOutput {
            content: out,
            is_error: false,
            truncated: more,
        })
    }
}

fn relative(path: &Path, workdir: &Path) -> String {
    path.strip_prefix(workdir)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Paths relative to `root` of files and folders (folders end in `/`) down to `depth` levels, honoring
/// `.gitignore`. At most `max` entries. Used for `@`-mentions.
pub fn list_paths(root: &Path, depth: Option<usize>, max: usize) -> Vec<String> {
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .git_global(false)
        .max_depth(depth)
        .filter_entry(|e| e.file_name() != ".git")
        .build();
    let mut out = Vec::new();
    for entry in walker.flatten() {
        let rel = relative(entry.path(), root);
        if rel.is_empty() {
            continue;
        }
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            out.push(format!("{rel}/"));
        } else {
            out.push(rel);
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

fn search(
    root: &Path,
    guard: &PathGuard,
    q: &Query,
    cancel: &CancellationToken,
) -> Result<(Vec<String>, bool), ToolError> {
    let workdir = guard.workdir();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .git_global(false)
        .filter_entry(|e| e.file_name() != ".git")
        .build();
    let mut results = Vec::new();
    for entry in walker {
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // The walker does not descend into links, but reading one would still follow it.
        if entry.path_is_symlink() && !guard.check(&path.to_string_lossy()).is_inside() {
            continue;
        }
        let rel = relative(path, workdir);
        if rel.is_empty() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
        let wants_name = matches!(q.target, Target::Name | Target::Both);
        let wants_content = matches!(q.target, Target::Content | Target::Both);

        if is_dir {
            if wants_name && q.glob.is_none() && q.regex.is_match(&name) {
                results.push(format!("{rel}/"));
            }
        } else if q.glob.as_ref().is_none_or(|g| g.is_match(&rel)) {
            if wants_name && q.regex.is_match(&name) {
                results.push(rel.clone());
            }
            if wants_content {
                search_content(path, &rel, q, &mut results);
            }
        }
        if results.len() >= q.max {
            results.truncate(q.max);
            return Ok((results, true));
        }
    }
    Ok((results, false))
}

fn search_content(path: &Path, rel: &str, q: &Query, results: &mut Vec<String>) {
    let too_big = std::fs::metadata(path).map_or(true, |m| m.len() > MAX_SEARCH_FILE_BYTES);
    if too_big {
        return;
    }
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Some(text) = decode_text(&bytes) else {
        return;
    };
    for (n, line) in text.lines().enumerate() {
        if q.regex.is_match(line) {
            let shown: String = line.trim().chars().take(MAX_LINE_CHARS).collect();
            results.push(format!("{rel}:{}: {shown}", n + 1));
            if results.len() >= q.max {
                return;
            }
        }
    }
}
