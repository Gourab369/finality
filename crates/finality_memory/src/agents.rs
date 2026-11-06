//! Custom agents (DESIGN §6.5): `agents/<name>/AGENT.md`, a base role plus instructions and an optional tool list.

use std::path::{Path, PathBuf};

use finality_core::Role;
use finality_store::DataDir;

use crate::skills::{as_list, as_text, split_frontmatter, valid_name};

pub const AGENT_FILE: &str = "AGENT.md";
/// Built-in role names that custom agents cannot take.
pub const RESERVED: &[&str] = &[
    "worker",
    "research",
    "self",
    "orchestrator",
    "memory",
    "scorer",
    "learner",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomAgent {
    pub name: String,
    pub description: String,
    /// `Role::Worker` or `Role::Research`.
    pub base: Role,
    /// Replaces the base role's default tools when set.
    pub tools: Option<Vec<String>>,
    pub created_by: Option<String>,
    pub created_at: Option<String>,
    pub instructions: String,
    pub dir: PathBuf,
}

impl CustomAgent {
    pub fn file(&self) -> PathBuf {
        self.dir.join(AGENT_FILE)
    }
}

pub fn parse_base(s: &str) -> Result<Role, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "worker" => Ok(Role::Worker),
        "research" => Ok(Role::Research),
        other => Err(format!(
            "invalid base `{other}`; expected worker or research"
        )),
    }
}

/// Validates an agent name; it is also what keeps names safe to join to a path.
pub fn check_name(name: &str) -> Result<(), String> {
    if !valid_name(name) {
        return Err(format!(
            "invalid agent name `{name}`: use lowercase letters, digits and hyphens (at most 64)"
        ));
    }
    if RESERVED.contains(&name) {
        return Err(format!("`{name}` is a built-in role name; choose another"));
    }
    Ok(())
}

/// Parses an AGENT.md file. `dir` is the agent's folder.
pub fn parse(text: &str, dir: &Path) -> Result<CustomAgent, String> {
    let (fm, body) = split_frontmatter(text)?;
    let name = as_text(fm.get("name")).ok_or("frontmatter needs a `name`")?;
    check_name(&name)?;
    let description = as_text(fm.get("description")).ok_or("frontmatter needs a `description`")?;
    let base = as_text(fm.get("base")).ok_or("frontmatter needs a `base` (worker or research)")?;
    if body.is_empty() {
        return Err("the agent has no instructions (the body is empty)".into());
    }
    Ok(CustomAgent {
        name,
        description,
        base: parse_base(&base)?,
        tools: as_list(fm.get("tools")),
        created_by: as_text(fm.get("created_by")),
        created_at: as_text(fm.get("created_at")),
        instructions: body,
        dir: dir.to_owned(),
    })
}

#[derive(Debug, Default, Clone)]
pub struct AgentCatalog {
    pub agents: Vec<CustomAgent>,
    /// Definitions that could not be loaded, as "path: reason".
    pub problems: Vec<String>,
}

impl AgentCatalog {
    pub fn get(&self, name: &str) -> Option<&CustomAgent> {
        self.agents.iter().find(|a| a.name == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.agents.iter().map(|a| a.name.as_str()).collect()
    }

    /// `- name (base): description` lines.
    pub fn text(&self) -> String {
        self.agents
            .iter()
            .map(|a| format!("- {} ({}): {}", a.name, a.base, a.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A new custom agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentDraft {
    pub name: String,
    pub description: String,
    pub base: Role,
    /// Empty means the base role's defaults.
    pub tools: Vec<String>,
    pub instructions: String,
    pub created_by: String,
}

/// Custom agents in `agents/`. Read from disk on every call so edits and deletes apply at once.
#[derive(Debug, Clone)]
pub struct AgentLibrary {
    dir: PathBuf,
}

impl AgentLibrary {
    pub fn new(data: &DataDir) -> Self {
        Self {
            dir: data.custom_agents_dir(),
        }
    }

    pub fn load(&self) -> AgentCatalog {
        let mut c = AgentCatalog::default();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return c;
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
            let file = d.join(AGENT_FILE);
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            match parse(&text, &d) {
                Ok(a) if c.get(&a.name).is_some() => c.problems.push(format!(
                    "{}: duplicate agent name `{}`",
                    file.display(),
                    a.name
                )),
                Ok(a) => c.agents.push(a),
                Err(e) => c.problems.push(format!("{}: {e}", file.display())),
            }
        }
        c
    }

    /// Writes `agents/<name>/AGENT.md`; never overwrites an existing agent or folder.
    pub fn create(&self, draft: &AgentDraft) -> Result<CustomAgent, String> {
        check_name(&draft.name)?;
        if !matches!(draft.base, Role::Worker | Role::Research) {
            return Err(format!(
                "invalid base `{}`; expected worker or research",
                draft.base
            ));
        }
        let description = draft
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if description.is_empty() || draft.instructions.trim().is_empty() {
            return Err("a custom agent needs a description and instructions".into());
        }
        if self.load().get(&draft.name).is_some() {
            return Err(format!(
                "a custom agent named `{}` already exists",
                draft.name
            ));
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        let dir = self.dir.join(&draft.name);
        std::fs::create_dir(&dir).map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => format!("{} already exists", dir.display()),
            _ => e.to_string(),
        })?;
        let mut text = format!(
            "---\nname: {}\ndescription: \"{}\"\nbase: {}\n",
            draft.name,
            description.replace('"', "'"),
            draft.base
        );
        if !draft.tools.is_empty() {
            text.push_str(&format!("tools: [{}]\n", draft.tools.join(", ")));
        }
        text.push_str(&format!(
            "created_by: {}\ncreated_at: {}\n---\n\n{}\n",
            draft.created_by,
            chrono::Local::now().format("%Y-%m-%d"),
            draft.instructions.trim()
        ));
        let file = dir.join(AGENT_FILE);
        std::fs::write(&file, &text).map_err(|e| e.to_string())?;
        parse(&text, &dir)
    }

    /// Removes the agent's folder.
    pub fn delete(&self, name: &str) -> Result<CustomAgent, String> {
        let agent = self
            .load()
            .agents
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| format!("no custom agent named `{name}`"))?;
        std::fs::remove_dir_all(&agent.dir).map_err(|e| e.to_string())?;
        Ok(agent)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn draft(name: &str) -> AgentDraft {
        AgentDraft {
            name: name.into(),
            description: "Reviews SQL migrations:\n \"locks\" and data loss".into(),
            base: Role::Research,
            tools: vec!["read_file".into(), "search_files".into()],
            instructions: "Check every ALTER TABLE.\n\nReport risks by severity.".into(),
            created_by: "orchestrator".into(),
        }
    }

    #[test]
    fn create_load_and_delete_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        let lib = AgentLibrary::new(&data);
        let made = lib.create(&draft("sql-reviewer")).unwrap();
        assert_eq!(made.dir, data.custom_agents_dir().join("sql-reviewer"));

        let c = lib.load();
        let a = c.get("sql-reviewer").unwrap();
        assert_eq!(a, &made);
        assert_eq!(
            a.description,
            "Reviews SQL migrations: 'locks' and data loss"
        );
        assert_eq!(a.base, Role::Research);
        assert_eq!(
            a.tools,
            Some(vec!["read_file".into(), "search_files".into()])
        );
        assert_eq!(a.created_by.as_deref(), Some("orchestrator"));
        assert!(a.created_at.is_some());
        assert_eq!(
            a.instructions,
            "Check every ALTER TABLE.\n\nReport risks by severity."
        );
        assert_eq!(
            c.text(),
            "- sql-reviewer (research): Reviews SQL migrations: 'locks' and data loss"
        );

        let again = lib.create(&draft("sql-reviewer")).unwrap_err();
        assert!(again.contains("already exists"), "{again}");

        lib.delete("sql-reviewer").unwrap();
        assert!(lib.load().agents.is_empty());
        assert!(!made.dir.exists());
        assert!(
            lib.delete("sql-reviewer")
                .unwrap_err()
                .contains("no custom agent")
        );
    }

    #[test]
    fn rejects_bad_names_and_bases() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = AgentLibrary::new(&DataDir::new(tmp.path()));
        for bad in ["worker", "Self", "../escape", "a/b", "", "has space"] {
            assert!(lib.create(&draft(bad)).is_err(), "{bad}");
        }
        let mut memory = draft("helper");
        memory.base = Role::Memory;
        assert!(lib.create(&memory).is_err());
        assert!(!tmp.path().join("agents").join("escape").exists());
        assert!(parse_base("Worker").is_ok() && parse_base("scorer").is_err());
    }

    #[test]
    fn broken_definitions_are_reported_and_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        let lib = AgentLibrary::new(&data);
        lib.create(&draft("good")).unwrap();
        let bad = data.custom_agents_dir().join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(
            bad.join(AGENT_FILE),
            "---\nname: bad\ndescription: x\nbase: scorer\n---\nbody\n",
        )
        .unwrap();
        let c = lib.load();
        assert_eq!(c.names(), ["good"]);
        assert_eq!(c.problems.len(), 1);
        assert!(c.problems[0].contains("invalid base"), "{:?}", c.problems);
    }
}
