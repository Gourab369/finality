//! Skills (DESIGN §8.4): `skills/<name>/SKILL.md`, Claude Code compatible frontmatter plus a markdown body.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use finality_store::DataDir;

pub const SKILL_FILE: &str = "SKILL.md";
const MAX_NAME: usize = 64;
/// Skills compiled into the binary. They have no folder (`dir` is empty).
const BUILTIN: &[&str] = &[include_str!("../skills/stubborn/SKILL.md")];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Tools the skill needs; can only narrow what an agent already has.
    pub allowed_tools: Option<Vec<String>>,
    pub created_by: Option<String>,
    pub source_events: Vec<String>,
    pub status: Option<String>,
    pub body: String,
    pub dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Value {
    Text(String),
    List(Vec<String>),
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            return s[1..s.len() - 1].to_owned();
        }
    }
    s.to_owned()
}

/// The YAML subset skills use: `key: value`, `key: [a, b]`, block lists and `|`/`>` block scalars.
fn parse_frontmatter(text: &str) -> Result<BTreeMap<String, Value>, String> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    let mut current: Option<(String, Option<char>)> = None;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let indented = line.starts_with([' ', '\t']);
        if indented {
            let Some((key, block)) = &current else {
                return Err(format!(
                    "frontmatter line {}: unexpected indentation",
                    n + 1
                ));
            };
            let item = line.trim();
            let list_item = item.strip_prefix("- ").or((item == "-").then_some(""));
            let entry = out.entry(key.clone()).or_insert(Value::Text(String::new()));
            let starts_list = block.is_none() && matches!(entry, Value::Text(t) if t.is_empty());
            match (entry, list_item) {
                (Value::List(items), Some(v)) => items.push(unquote(v)),
                (Value::List(_), None) => {
                    return Err(format!("frontmatter line {}: expected a list item", n + 1));
                }
                (e, Some(v)) if starts_list => *e = Value::List(vec![unquote(v)]),
                (Value::Text(t), _) => {
                    if !t.is_empty() {
                        t.push(if *block == Some('|') { '\n' } else { ' ' });
                    }
                    t.push_str(item);
                }
            }
            continue;
        }
        let (key, rest) = line
            .split_once(':')
            .ok_or_else(|| format!("frontmatter line {}: expected `key: value`", n + 1))?;
        let key = key.trim().to_owned();
        let rest = rest.trim();
        let block = match rest {
            "|" | "|-" | "|+" => Some('|'),
            ">" | ">-" | ">+" => Some('>'),
            _ => None,
        };
        let value = if block.is_some() || rest.is_empty() {
            Value::Text(String::new())
        } else if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            Value::List(
                inner
                    .split(',')
                    .map(unquote)
                    .filter(|s| !s.is_empty())
                    .collect(),
            )
        } else {
            Value::Text(unquote(rest))
        };
        out.insert(key.clone(), value);
        current = Some((key, block));
    }
    Ok(out)
}

pub(crate) fn as_list(v: Option<&Value>) -> Option<Vec<String>> {
    match v? {
        Value::List(items) => Some(items.clone()),
        Value::Text(t) => Some(
            t.split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect(),
        ),
    }
}

pub(crate) fn as_text(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::Text(t) if !t.trim().is_empty() => Some(t.trim().to_owned()),
        _ => None,
    }
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Splits `---` frontmatter from the markdown body.
pub(crate) fn split_frontmatter(text: &str) -> Result<(BTreeMap<String, Value>, String), String> {
    let text = text.trim_start_matches('\u{feff}');
    let mut lines = text.lines();
    if lines.next().map(str::trim_end) != Some("---") {
        return Err("missing frontmatter (the file must start with ---)".into());
    }
    let rest: Vec<&str> = lines.collect();
    let end = rest
        .iter()
        .position(|l| l.trim_end() == "---")
        .ok_or("frontmatter is not closed with ---")?;
    let fm = parse_frontmatter(&rest[..end].join("\n"))?;
    Ok((fm, rest[end + 1..].join("\n").trim().to_owned()))
}

/// Parses a SKILL.md file. `dir` is the skill's folder.
pub fn parse(text: &str, dir: &Path) -> Result<Skill, String> {
    let (fm, body) = split_frontmatter(text)?;
    let name = as_text(fm.get("name")).ok_or("frontmatter needs a `name`")?;
    if !valid_name(&name) {
        return Err(format!(
            "invalid name `{name}`: use lowercase letters, digits and hyphens (at most {MAX_NAME})"
        ));
    }
    let description = as_text(fm.get("description")).ok_or("frontmatter needs a `description`")?;
    Ok(Skill {
        name,
        description,
        allowed_tools: as_list(fm.get("allowed-tools")),
        created_by: as_text(fm.get("created_by")),
        source_events: as_list(fm.get("source_events")).unwrap_or_default(),
        status: as_text(fm.get("status")),
        body,
        dir: dir.to_owned(),
    })
}

/// Maps Claude Code tool names (`Read`, `Bash(git:*)`, …) to finality tool names.
pub fn map_tool_name(name: &str) -> String {
    let base = name.split('(').next().unwrap_or(name).trim();
    match base {
        "Read" => "read_file",
        "Grep" | "Glob" | "LS" => "search_files",
        "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => "edit_file",
        "Bash" => "shell",
        "WebFetch" => "web_fetch",
        "WebSearch" => "web_search",
        other => other,
    }
    .to_owned()
}

impl Skill {
    /// Shipped with Finality rather than read from the data directory.
    pub fn is_builtin(&self) -> bool {
        self.dir.as_os_str().is_empty()
    }

    /// `allowed-tools` as finality tool names.
    pub fn tool_names(&self) -> Option<Vec<String>> {
        self.allowed_tools
            .as_ref()
            .map(|t| t.iter().map(|n| map_tool_name(n)).collect())
    }
}

#[derive(Debug, Default, Clone)]
pub struct Catalog {
    pub active: Vec<Skill>,
    pub pending: Vec<Skill>,
    /// Skills that could not be loaded, as "path: reason".
    pub problems: Vec<String>,
}

impl Catalog {
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.active.iter().find(|s| s.name == name)
    }

    /// `- name: description` lines of active skills.
    pub fn text(&self) -> String {
        self.active
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Active skills in `skills/`, pending ones in `skills/_pending/`. Read from disk on every call.
#[derive(Debug, Clone)]
pub struct SkillLibrary {
    active_dir: PathBuf,
    pending_dir: PathBuf,
}

fn load_dir(dir: &Path, out: &mut Vec<Skill>, problems: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with(['_', '.']))
        })
        .collect();
    dirs.sort();
    for d in dirs {
        let file = d.join(SKILL_FILE);
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        match parse(&text, &d) {
            Ok(s) if out.iter().any(|o| o.name == s.name) => {
                problems.push(format!(
                    "{}: duplicate skill name `{}`",
                    file.display(),
                    s.name
                ));
            }
            Ok(s) => out.push(s),
            Err(e) => problems.push(format!("{}: {e}", file.display())),
        }
    }
}

impl SkillLibrary {
    pub fn new(dir: &DataDir) -> Self {
        Self {
            active_dir: dir.skills_dir(),
            pending_dir: dir.pending_skills_dir(),
        }
    }

    pub fn load(&self) -> Catalog {
        let mut c = Catalog::default();
        load_dir(&self.active_dir, &mut c.active, &mut c.problems);
        load_dir(&self.pending_dir, &mut c.pending, &mut c.problems);
        // A user skill with the same name replaces the built-in one.
        for text in BUILTIN {
            match parse(text, Path::new("")) {
                Ok(s) if c.get(&s.name).is_none() => c.active.push(s),
                Ok(_) => {}
                Err(e) => c.problems.push(format!("built-in skill: {e}")),
            }
        }
        c
    }

    /// Moves a pending skill into `skills/<name>/` and marks it active.
    pub fn approve(&self, name: &str) -> Result<(), String> {
        let catalog = self.load();
        let skill = catalog
            .pending
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("no pending skill `{name}`"))?;
        if catalog.get(name).is_some() {
            return Err(format!("an active skill named `{name}` already exists"));
        }
        let target = self.active_dir.join(name);
        if target.exists() {
            return Err(format!("{} already exists", target.display()));
        }
        std::fs::rename(&skill.dir, &target).map_err(|e| e.to_string())?;
        let file = target.join(SKILL_FILE);
        if let Ok(text) = std::fs::read_to_string(&file) {
            let updated: Vec<String> = text
                .lines()
                .map(|l| {
                    if l.trim_start().starts_with("status:") {
                        "status: active".to_owned()
                    } else {
                        l.to_owned()
                    }
                })
                .collect();
            std::fs::write(&file, updated.join("\n") + "\n").map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Deletes a pending skill's folder.
    pub fn reject(&self, name: &str) -> Result<(), String> {
        let skill = self
            .load()
            .pending
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("no pending skill `{name}`"))?;
        std::fs::remove_dir_all(&skill.dir).map_err(|e| e.to_string())
    }

    /// True when an active or pending skill already has this name.
    pub fn exists(&self, name: &str) -> bool {
        let c = self.load();
        c.active.iter().chain(&c.pending).any(|s| s.name == name)
            || self.active_dir.join(name).exists()
            || self.pending_dir.join(name).exists()
    }

    /// Writes a Learner-made skill, pending review unless `pending` is false.
    pub fn create(&self, draft: &SkillDraft, pending: bool) -> Result<PathBuf, String> {
        if !valid_name(&draft.name) {
            return Err(format!("invalid skill name `{}`", draft.name));
        }
        let description = draft
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if description.is_empty() || draft.body.trim().is_empty() {
            return Err("a skill needs a description and a body".into());
        }
        if self.exists(&draft.name) {
            return Err(format!("a skill named `{}` already exists", draft.name));
        }
        let dir = if pending {
            &self.pending_dir
        } else {
            &self.active_dir
        }
        .join(&draft.name);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut text = format!(
            "---\nname: {}\ndescription: \"{}\"\n",
            draft.name,
            description.replace('"', "'")
        );
        if !draft.allowed_tools.is_empty() {
            text.push_str(&format!(
                "allowed-tools: {}\n",
                draft.allowed_tools.join(", ")
            ));
        }
        text.push_str(&format!(
            "created_by: learner\nsource_events: [{}]\nstatus: {}\n---\n\n{}\n",
            draft.source_events.join(", "),
            if pending { "pending" } else { "active" },
            draft.body.trim()
        ));
        std::fs::write(dir.join(SKILL_FILE), text).map_err(|e| e.to_string())?;
        Ok(dir)
    }
}

/// A skill proposed by the Learner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDraft {
    pub name: String,
    pub description: String,
    pub allowed_tools: Vec<String>,
    pub body: String,
    pub source_events: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learner_skills_round_trip_and_never_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        let lib = SkillLibrary::new(&data);
        let draft = SkillDraft {
            name: "zig-build".into(),
            description: "Build Zig projects: \"zig build\" with steps".into(),
            allowed_tools: vec!["shell".into(), "read_file".into()],
            body: "1. Write build.zig\n2. Run zig build".into(),
            source_events: vec!["01JA".into(), "01JB".into()],
        };
        let dir = lib.create(&draft, true).unwrap();
        assert_eq!(dir, data.pending_skills_dir().join("zig-build"));
        let c = lib.load();
        let s = &c.pending[0];
        assert_eq!(s.description, "Build Zig projects: 'zig build' with steps");
        assert_eq!(
            s.allowed_tools,
            Some(vec!["shell".into(), "read_file".into()])
        );
        assert_eq!(s.source_events, ["01JA", "01JB"]);
        assert_eq!(s.created_by.as_deref(), Some("learner"));
        assert_eq!(s.status.as_deref(), Some("pending"));
        assert_eq!(s.body, "1. Write build.zig\n2. Run zig build");
        assert!(
            lib.create(&draft, false).is_err(),
            "names are unique across active and pending"
        );
        let auto = SkillDraft {
            name: "zig-test".into(),
            ..draft
        };
        lib.create(&auto, false).unwrap();
        assert_eq!(
            lib.load().get("zig-test").unwrap().status.as_deref(),
            Some("active")
        );
    }

    const CLAP: &str = "---\nname: rust-clap\ndescription: \"Add CLI flags with clap\"\nallowed-tools: Read, Bash(cargo:*)\n\
                        created_by: learner\nsource_events:\n  - 01JABC\n  - 01JABD\nstatus: pending\n---\n\n# Steps\nUse derive.\n";

    #[test]
    fn parses_frontmatter_variants() {
        let s = parse(CLAP, Path::new("x")).unwrap();
        assert_eq!(s.name, "rust-clap");
        assert_eq!(s.description, "Add CLI flags with clap");
        assert_eq!(
            s.allowed_tools,
            Some(vec!["Read".into(), "Bash(cargo:*)".into()])
        );
        assert_eq!(
            s.tool_names(),
            Some(vec!["read_file".into(), "shell".into()])
        );
        assert_eq!(s.source_events, ["01JABC", "01JABD"]);
        assert_eq!(s.status.as_deref(), Some("pending"));
        assert_eq!(s.body, "# Steps\nUse derive.");

        let folded = "---\nname: a\ndescription: >\n  first line\n  second line\nallowed-tools: [read_file, 'shell']\n---\nbody";
        let s = parse(folded, Path::new("x")).unwrap();
        assert_eq!(s.description, "first line second line");
        assert_eq!(
            s.allowed_tools,
            Some(vec!["read_file".into(), "shell".into()])
        );
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse("no frontmatter", Path::new("x")).is_err());
        assert!(parse("---\nname: a\n", Path::new("x")).is_err());
        assert!(
            parse("---\ndescription: d\n---\n", Path::new("x"))
                .unwrap_err()
                .contains("name")
        );
        assert!(
            parse("---\nname: a\n---\n", Path::new("x"))
                .unwrap_err()
                .contains("description")
        );
        assert!(parse("---\nname: Bad Name\ndescription: d\n---\n", Path::new("x")).is_err());
    }

    fn write(dir: &Path, folder: &str, text: &str) {
        std::fs::create_dir_all(dir.join(folder)).unwrap();
        std::fs::write(dir.join(folder).join(SKILL_FILE), text).unwrap();
    }

    #[test]
    fn library_loads_skips_invalid_and_reviews() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        data.ensure().unwrap();
        let lib = SkillLibrary::new(&data);
        write(
            &data.skills_dir(),
            "good",
            "---\nname: good\ndescription: works\n---\nbody",
        );
        write(&data.skills_dir(), "broken", "---\nname: broken\n---\n");
        write(&data.pending_skills_dir(), "rust-clap", CLAP);

        let c = lib.load();
        assert_eq!(
            c.active.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["good", "stubborn"]
        );
        assert_eq!(c.pending.len(), 1);
        assert_eq!(c.problems.len(), 1);
        assert!(c.problems[0].contains("broken"));
        assert!(c.text().starts_with("- good: works\n- stubborn: "));
        assert!(c.get("stubborn").unwrap().is_builtin());
        assert!(!c.get("good").unwrap().is_builtin());

        lib.approve("rust-clap").unwrap();
        let c = lib.load();
        assert!(c.pending.is_empty());
        let s = c.get("rust-clap").unwrap();
        assert_eq!(s.status.as_deref(), Some("active"));
        assert_eq!(s.dir, data.skills_dir().join("rust-clap"));

        write(
            &data.pending_skills_dir(),
            "other",
            "---\nname: other\ndescription: d\n---\n",
        );
        lib.reject("other").unwrap();
        assert!(!data.pending_skills_dir().join("other").exists());
        assert!(lib.approve("nope").is_err());
    }

    #[test]
    fn user_skill_overrides_the_builtin_one() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        data.ensure().unwrap();
        let lib = SkillLibrary::new(&data);
        let builtin = lib.load();
        let s = builtin.get("stubborn").unwrap();
        assert!(s.body.contains("Hard blocks") && builtin.problems.is_empty());
        assert!(lib.exists("stubborn"), "the Learner cannot shadow it");

        write(
            &data.skills_dir(),
            "stubborn",
            "---\nname: stubborn\ndescription: mine\n---\nbody",
        );
        let c = lib.load();
        assert_eq!(c.active.len(), 1);
        assert_eq!(c.get("stubborn").unwrap().description, "mine");
    }
}
