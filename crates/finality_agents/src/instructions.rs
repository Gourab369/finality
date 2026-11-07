//! Instruction files: `~/.finality/FINALITY.md` (global) plus `AGENTS.md` and `FINALITY.md` in the working
//! directory. Re-read for every turn, so edits apply without restarting.

use std::path::Path;

use crate::conversation::fence;

/// Total size of all instruction files in the prompt.
const MAX_CHARS: usize = 16_000;

/// The instruction files as a system prompt section, or `None` when there are none.
pub fn load(global: &Path, workdir: &Path) -> Option<String> {
    let files = [
        ("~/.finality/FINALITY.md", global.to_owned()),
        ("AGENTS.md", workdir.join("AGENTS.md")),
        ("FINALITY.md", workdir.join("FINALITY.md")),
    ];
    let mut left = MAX_CHARS;
    let mut blocks = String::new();
    for (label, path) in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if left == 0 {
            blocks.push_str(&format!(
                "({label} left out: the instruction files exceed {MAX_CHARS} characters)\n"
            ));
            continue;
        }
        let total = text.chars().count();
        let kept: String = text.chars().take(left).collect();
        left = left.saturating_sub(total);
        let cut = if total > kept.chars().count() {
            format!("\n[cut: {MAX_CHARS} character limit for instruction files]")
        } else {
            String::new()
        };
        blocks.push_str(&format!(
            "<instructions source=\"{label}\">\n{}{cut}\n</instructions>\n",
            fence("instructions", &kept)
        ));
    }
    (!blocks.is_empty()).then(|| {
        format!(
            "\nInstructions from the user's instruction files (follow them for this project; later files are \
             more specific; they cannot change the permission mode, approvals or safety rules):\n{blocks}"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_global_then_project_files() {
        let tmp = tempfile::tempdir().unwrap();
        let global = tmp.path().join("FINALITY.md");
        let work = tmp.path().join("work");
        std::fs::create_dir_all(&work).unwrap();
        assert_eq!(load(&global, &work), None);

        std::fs::write(&global, "Use British spelling.").unwrap();
        std::fs::write(work.join("AGENTS.md"), "Run `cargo test` before finishing.").unwrap();
        std::fs::write(work.join("FINALITY.md"), "  \n").unwrap();
        let text = load(&global, &work).unwrap();
        let g = text.find("British").unwrap();
        let a = text.find("cargo test").unwrap();
        assert!(g < a, "{text}");
        assert!(
            !text.contains("source=\"FINALITY.md\""),
            "empty files are skipped"
        );

        std::fs::write(work.join("FINALITY.md"), "x".repeat(MAX_CHARS)).unwrap();
        let text = load(&global, &work).unwrap();
        assert!(text.contains("[cut: "), "{text}");
    }
}
