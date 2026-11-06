//! Slash command table, parser and completion (DESIGN §11.5). Later phases add commands here.

use h_agent_core::events::{AgentAction, McpAction, SkillInfo, SlashCommand};
use h_agent_core::{Mode, ProviderKind, ThinkLevel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandInfo {
    pub name: &'static str,
    pub args: &'static str,
    pub help: &'static str,
}

pub const COMMANDS: &[CommandInfo] = &[
    CommandInfo {
        name: "help",
        args: "",
        help: "List commands and keys",
    },
    CommandInfo {
        name: "new",
        args: "",
        help: "Close this session and start fresh",
    },
    CommandInfo {
        name: "sessions",
        args: "[--all|prune <days>]",
        help: "List past sessions, or delete closed ones older than N days",
    },
    CommandInfo {
        name: "resume",
        args: "[id]",
        help: "Resume a past session (no id: pick one of this directory's)",
    },
    CommandInfo {
        name: "mode",
        args: "<plan|cowork|auto-edit|auto>",
        help: "Change the permission mode",
    },
    CommandInfo {
        name: "think",
        args: "<off|low|medium|high>",
        help: "Set the thinking level",
    },
    CommandInfo {
        name: "provider",
        args: "<ollama|openai|anthropic|llamacpp> [model]",
        help: "Switch provider",
    },
    CommandInfo {
        name: "model",
        args: "[name]",
        help: "Switch model (no name: pick from a list)",
    },
    CommandInfo {
        name: "context",
        args: "",
        help: "Show token usage",
    },
    CommandInfo {
        name: "agents",
        args: "",
        help: "Show running agents in detail",
    },
    CommandInfo {
        name: "tasks",
        args: "",
        help: "Show the current plan in detail",
    },
    CommandInfo {
        name: "subagents",
        args: "<n>",
        help: "Set how many sub-agents may run at once",
    },
    CommandInfo {
        name: "decide",
        args: "<on|off>",
        help: "Use the fast decision model where it is confident",
    },
    CommandInfo {
        name: "memory",
        args: "[recall <query>|review]",
        help: "Show the memory digest, search past sessions, or review learned notes",
    },
    CommandInfo {
        name: "skills",
        args: "[review]",
        help: "List skills, or review pending ones",
    },
    CommandInfo {
        name: "skill",
        args: "<name> [args]",
        help: "Run a skill (also /<name>)",
    },
    CommandInfo {
        name: "tools",
        args: "",
        help: "Show loaded and loadable tools",
    },
    CommandInfo {
        name: "knowledge",
        args: "[topic]",
        help: "List knowledgepedia topics or show one",
    },
    CommandInfo {
        name: "mcp",
        args: "[list|login <server>|enable|disable]",
        help: "Manage MCP servers",
    },
    CommandInfo {
        name: "agent",
        args: "[list|show <name>|delete <name>]",
        help: "List, show or delete custom agents",
    },
    CommandInfo {
        name: "quit",
        args: "",
        help: "Finalize the session and exit",
    },
    CommandInfo {
        name: "exit",
        args: "",
        help: "Same as /quit",
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashAction {
    Help,
    Quit,
    Agents,
    Tasks,
    Core(SlashCommand),
}

fn usage(name: &str) -> String {
    let args = COMMANDS
        .iter()
        .find(|c| c.name == name)
        .map_or("", |c| c.args);
    format!("usage: /{name} {args}")
}

fn parse_arg<T: std::str::FromStr>(name: &str, arg: Option<&str>) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let arg = arg.ok_or_else(|| usage(name))?;
    arg.parse().map_err(|e: T::Err| e.to_string())
}

/// Parses `/name args…` with no skills known. `Err` carries a message for the user.
pub fn parse(input: &str) -> Result<SlashAction, String> {
    parse_with(input, &[])
}

/// Parses `/name args…`; names that are not built-in commands may be `skills`.
pub fn parse_with(input: &str, skills: &[String]) -> Result<SlashAction, String> {
    let body = input
        .trim()
        .strip_prefix('/')
        .ok_or("commands start with /")?;
    let mut parts = body.split_whitespace();
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let args: Vec<&str> = parts.collect();
    let first = args.first().copied();
    // Text after the name, as typed.
    let rest = body
        .trim_start()
        .get(name.len()..)
        .unwrap_or("")
        .trim()
        .to_owned();
    let cmd = match name.as_str() {
        "help" | "?" => return Ok(SlashAction::Help),
        "quit" | "exit" => return Ok(SlashAction::Quit),
        "agents" => return Ok(SlashAction::Agents),
        "tasks" => return Ok(SlashAction::Tasks),
        "skills" => match first {
            None => SlashCommand::Skills { review: false },
            Some("review") => SlashCommand::Skills { review: true },
            _ => return Err(usage(&name)),
        },
        "skill" => {
            let skill = first.ok_or_else(|| usage(&name))?;
            SlashCommand::Skill {
                name: skill.to_owned(),
                args: rest.get(skill.len()..).unwrap_or("").trim().to_owned(),
            }
        }
        "tools" => SlashCommand::Tools,
        "knowledge" => SlashCommand::Knowledge {
            topic: (!rest.is_empty()).then(|| rest.clone()),
        },
        "mcp" => SlashCommand::Mcp(match (first, args.get(1), args.len()) {
            (None | Some("list"), _, 0 | 1) => McpAction::List,
            (Some("login"), Some(server), 2) => McpAction::Login((*server).to_owned()),
            (Some("enable"), _, 1) => McpAction::Enable,
            (Some("disable"), _, 1) => McpAction::Disable,
            _ => return Err(usage(&name)),
        }),
        "agent" => SlashCommand::Agent(match (first, args.get(1), args.len()) {
            (None | Some("list"), _, 0 | 1) => AgentAction::List,
            (Some("show"), Some(agent), 2) => AgentAction::Show((*agent).to_owned()),
            (Some("delete"), Some(agent), 2) => AgentAction::Delete((*agent).to_owned()),
            _ => return Err(usage(&name)),
        }),
        "subagents" => match parse_arg::<u32>(&name, first)? {
            0 => return Err("the sub-agent limit must be at least 1".into()),
            n => SlashCommand::Subagents(n),
        },
        "decide" => match (first, args.len()) {
            (Some("on"), 1) => SlashCommand::Decide(true),
            (Some("off"), 1) => SlashCommand::Decide(false),
            _ => return Err(usage(&name)),
        },
        "memory" => match first {
            None => SlashCommand::Memory { recall: None },
            Some("review") if args.len() == 1 => SlashCommand::ProfileReview,
            Some("recall") if args.len() > 1 => SlashCommand::Memory {
                recall: Some(args[1..].join(" ")),
            },
            _ => return Err(usage(&name)),
        },
        "new" => SlashCommand::New,
        "sessions" => match (first, args.get(1), args.len()) {
            (Some("prune"), Some(days), 2) => SlashCommand::SessionsPrune {
                days: days
                    .parse()
                    .map_err(|_| "usage: /sessions prune <days>".to_owned())?,
            },
            (Some("prune"), ..) => return Err("usage: /sessions prune <days>".into()),
            _ => SlashCommand::Sessions {
                all: args.iter().any(|a| matches!(*a, "--all" | "-a" | "all")),
            },
        },
        "resume" => SlashCommand::Resume {
            id: first.map(str::to_owned),
        },
        "mode" => SlashCommand::Mode(parse_arg::<Mode>(&name, first)?),
        "think" => SlashCommand::Think(parse_arg::<ThinkLevel>(&name, first)?),
        "provider" => SlashCommand::Provider {
            provider: parse_arg::<ProviderKind>(&name, first)?,
            model: args.get(1).map(|m| (*m).to_owned()),
        },
        "model" => first.map_or(SlashCommand::ChooseModel, |m| {
            SlashCommand::Model(m.to_owned())
        }),
        "context" => SlashCommand::Context,
        "" => return Err("type a command name after /".into()),
        other if skills.iter().any(|s| s == other) => SlashCommand::Skill {
            name: other.to_owned(),
            args: rest,
        },
        other => {
            let hint = suggest(other, skills)
                .map(|s| format!("; did you mean /{s}?"))
                .unwrap_or_default();
            return Err(format!("unknown command /{other}{hint}"));
        }
    };
    Ok(SlashAction::Core(cmd))
}

/// A completion entry: a built-in command or a skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub name: String,
    pub args: String,
    pub help: String,
}

/// Built-in commands whose name starts with `prefix` (without the leading `/`).
pub fn completions(prefix: &str) -> Vec<&'static CommandInfo> {
    let prefix = prefix.to_ascii_lowercase();
    COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(&prefix))
        .collect()
}

/// Built-in commands, then skills (not shadowed by a built-in), whose name starts with `prefix`.
pub fn completions_with(prefix: &str, skills: &[SkillInfo]) -> Vec<Completion> {
    let mut out: Vec<Completion> = completions(prefix)
        .into_iter()
        .map(|c| Completion {
            name: c.name.into(),
            args: c.args.into(),
            help: c.help.into(),
        })
        .collect();
    let prefix = prefix.to_ascii_lowercase();
    out.extend(
        skills
            .iter()
            .filter(|s| s.name.starts_with(&prefix) && !COMMANDS.iter().any(|c| c.name == s.name))
            .map(|s| Completion {
                name: s.name.clone(),
                args: "[args]".into(),
                help: format!("skill: {}", s.description),
            }),
    );
    out
}

fn suggest<'a>(name: &str, skills: &'a [String]) -> Option<&'a str> {
    COMMANDS
        .iter()
        .map(|c| c.name)
        .chain(skills.iter().map(String::as_str))
        .map(|n| (levenshtein(name, n), n))
        .filter(|(d, _)| *d <= 2)
        .min_by_key(|(d, _)| *d)
        .map(|(_, n)| n)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands() {
        assert_eq!(parse("/help"), Ok(SlashAction::Help));
        assert_eq!(parse(" /EXIT "), Ok(SlashAction::Quit));
        assert_eq!(
            parse("/mode auto-edit"),
            Ok(SlashAction::Core(SlashCommand::Mode(Mode::AutoEdit)))
        );
        assert_eq!(
            parse("/think off"),
            Ok(SlashAction::Core(SlashCommand::Think(ThinkLevel::Off)))
        );
        assert_eq!(
            parse("/provider anthropic some-model"),
            Ok(SlashAction::Core(SlashCommand::Provider {
                provider: ProviderKind::Anthropic,
                model: Some("some-model".into())
            }))
        );
        assert_eq!(
            parse("/sessions --all"),
            Ok(SlashAction::Core(SlashCommand::Sessions { all: true }))
        );
        assert_eq!(
            parse("/resume"),
            Ok(SlashAction::Core(SlashCommand::Resume { id: None }))
        );
        assert_eq!(parse("/agents"), Ok(SlashAction::Agents));
        assert_eq!(parse("/tasks"), Ok(SlashAction::Tasks));
        assert_eq!(
            parse("/subagents 3"),
            Ok(SlashAction::Core(SlashCommand::Subagents(3)))
        );
        assert!(parse("/subagents 0").is_err());
        assert!(parse("/subagents").is_err());
        assert_eq!(
            parse("/memory"),
            Ok(SlashAction::Core(SlashCommand::Memory { recall: None }))
        );
        assert_eq!(
            parse("/memory recall crc  bug"),
            Ok(SlashAction::Core(SlashCommand::Memory {
                recall: Some("crc bug".into())
            }))
        );
        assert!(parse("/memory recall").is_err());
        assert!(parse("/memory forget").is_err());
        assert_eq!(
            parse("/memory review"),
            Ok(SlashAction::Core(SlashCommand::ProfileReview))
        );
        assert!(parse("/memory review now").is_err());
        assert_eq!(
            parse("/sessions prune 30"),
            Ok(SlashAction::Core(SlashCommand::SessionsPrune { days: 30 }))
        );
        assert_eq!(
            parse("/sessions --all"),
            Ok(SlashAction::Core(SlashCommand::Sessions { all: true }))
        );
        for bad in [
            "/sessions prune",
            "/sessions prune x",
            "/sessions prune 1 2",
        ] {
            assert_eq!(
                parse(bad).unwrap_err(),
                "usage: /sessions prune <days>",
                "{bad}"
            );
        }
    }

    #[test]
    fn reports_errors_with_hints() {
        assert_eq!(
            parse("/mode").unwrap_err(),
            "usage: /mode <plan|cowork|auto-edit|auto>"
        );
        assert!(parse("/mode yolo").unwrap_err().contains("invalid mode"));
        assert_eq!(
            parse("/mdoe auto").unwrap_err(),
            "unknown command /mdoe; did you mean /mode?"
        );
        assert_eq!(parse("/zzzzzz").unwrap_err(), "unknown command /zzzzzz");
        assert_eq!(
            parse("/model"),
            Ok(SlashAction::Core(SlashCommand::ChooseModel))
        );
    }

    #[test]
    fn parses_mcp_actions() {
        let mcp = |a| Ok(SlashAction::Core(SlashCommand::Mcp(a)));
        assert_eq!(parse("/mcp"), mcp(McpAction::List));
        assert_eq!(parse("/mcp list"), mcp(McpAction::List));
        assert_eq!(
            parse("/mcp login github"),
            mcp(McpAction::Login("github".into()))
        );
        assert_eq!(parse("/mcp enable"), mcp(McpAction::Enable));
        assert_eq!(parse("/mcp disable"), mcp(McpAction::Disable));
        for bad in [
            "/mcp login",
            "/mcp login a b",
            "/mcp restart",
            "/mcp list x",
        ] {
            assert_eq!(
                parse(bad).unwrap_err(),
                "usage: /mcp [list|login <server>|enable|disable]",
                "{bad}"
            );
        }
    }

    #[test]
    fn parses_agent_actions() {
        let agent = |a| Ok(SlashAction::Core(SlashCommand::Agent(a)));
        assert_eq!(parse("/agent"), agent(AgentAction::List));
        assert_eq!(parse("/agent list"), agent(AgentAction::List));
        assert_eq!(
            parse("/agent show sql-reviewer"),
            agent(AgentAction::Show("sql-reviewer".into()))
        );
        assert_eq!(
            parse("/agent delete sql-reviewer"),
            agent(AgentAction::Delete("sql-reviewer".into()))
        );
        assert_eq!(
            parse("/agents"),
            Ok(SlashAction::Agents),
            "the live view is separate"
        );
        for bad in [
            "/agent delete",
            "/agent delete a b",
            "/agent remove a",
            "/agent list x",
        ] {
            assert_eq!(
                parse(bad).unwrap_err(),
                "usage: /agent [list|show <name>|delete <name>]",
                "{bad}"
            );
        }
    }

    #[test]
    fn parses_decide() {
        assert_eq!(
            parse("/decide on"),
            Ok(SlashAction::Core(SlashCommand::Decide(true)))
        );
        assert_eq!(
            parse("/decide OFF".to_lowercase().as_str()),
            Ok(SlashAction::Core(SlashCommand::Decide(false)))
        );
        for bad in ["/decide", "/decide maybe", "/decide on now"] {
            assert_eq!(parse(bad).unwrap_err(), "usage: /decide <on|off>", "{bad}");
        }
    }

    #[test]
    fn completes_by_prefix() {
        let names: Vec<&str> = completions("m").iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["mode", "model", "memory", "mcp"]);
        assert_eq!(completions("").len(), COMMANDS.len());
        assert!(completions("zz").is_empty());
    }

    fn skill(name: &str) -> SkillInfo {
        SkillInfo {
            name: name.into(),
            description: "d".into(),
        }
    }

    #[test]
    fn skills_parse_and_complete_after_built_ins() {
        let skills = vec!["rust-clap".to_owned(), "help".to_owned()];
        assert_eq!(
            parse_with("/rust-clap add  --verbose", &skills),
            Ok(SlashAction::Core(SlashCommand::Skill {
                name: "rust-clap".into(),
                args: "add  --verbose".into()
            }))
        );
        assert_eq!(
            parse_with("/help", &skills),
            Ok(SlashAction::Help),
            "built-ins win"
        );
        assert_eq!(
            parse_with("/skill help me", &skills),
            Ok(SlashAction::Core(SlashCommand::Skill {
                name: "help".into(),
                args: "me".into()
            }))
        );
        assert_eq!(
            parse_with("/rust-clpa", &skills).unwrap_err(),
            "unknown command /rust-clpa; did you mean /rust-clap?"
        );
        assert!(parse("/rust-clap").is_err(), "unknown without the catalog");
        assert_eq!(
            parse("/skills review"),
            Ok(SlashAction::Core(SlashCommand::Skills { review: true }))
        );
        assert_eq!(parse("/tools"), Ok(SlashAction::Core(SlashCommand::Tools)));
        assert!(parse("/skill").is_err());

        let infos = [skill("rust-clap"), skill("help"), skill("tools-guide")];
        let names: Vec<String> = completions_with("t", &infos)
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["think", "tasks", "tools", "tools-guide"]);
        assert!(
            completions_with("he", &infos)
                .iter()
                .all(|c| c.help != "skill: d")
        );
    }
}
