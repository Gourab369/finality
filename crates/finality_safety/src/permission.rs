//! Permission engine (DESIGN §12.1). The single place that turns an action into Allow / Deny / Ask.

use std::collections::{BTreeSet, HashSet};

use finality_core::{Mode, RiskClass};

pub use crate::command::Location;

/// One simple command reduced to its shape, so `always_ask_patterns` match options in any order,
/// clustered (`-rf`) or split (`-r -f`), quoted or not.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandShape {
    /// Lowercased base command (no directory, no `.exe`).
    pub name: String,
    /// Letters of single-dash clusters such as `-rf`.
    pub short: BTreeSet<char>,
    /// `--long[=value]`, PowerShell `-Parameter` and cmd `/switch`, lowercased.
    pub long: BTreeSet<String>,
    /// The remaining words, lowercased.
    pub positional: Vec<String>,
}

/// `-rf`: a dash and up to four letters (longer single-dash words are PowerShell parameters).
fn is_short_cluster(t: &str) -> bool {
    t.len() >= 2
        && t.len() <= 5
        && t.starts_with('-')
        && t[1..].chars().all(|c| c.is_ascii_alphabetic())
}

/// `/q`, `/s`, `/A:H` style switches.
fn is_slash_switch(t: &str) -> bool {
    let Some(rest) = t.strip_prefix('/') else {
        return false;
    };
    (rest.len() <= 2 || rest.get(1..2) == Some(":"))
        && rest.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && !rest.contains(['/', '\\'])
}

impl CommandShape {
    /// `tokens` are the dequoted arguments; `slash_switches` treats `/x` as a switch rather than a path.
    pub fn new(name: &str, tokens: impl IntoIterator<Item = String>, slash_switches: bool) -> Self {
        let mut shape = Self {
            name: name.to_lowercase(),
            ..Self::default()
        };
        let mut options_done = false;
        for t in tokens {
            let t = t.to_lowercase();
            if !options_done {
                if t == "--" {
                    options_done = true;
                    continue;
                }
                if is_short_cluster(&t) {
                    shape.short.extend(t[1..].chars());
                    continue;
                }
                if (t.len() > 1 && t.starts_with('-')) || (slash_switches && is_slash_switch(&t)) {
                    shape.long.insert(t);
                    continue;
                }
            }
            shape.positional.push(t);
        }
        shape
    }

    /// A pattern such as `rm -rf` or `Remove-Item -Recurse -Force`, split on whitespace.
    pub fn from_pattern(pattern: &str) -> Option<Self> {
        let mut words = pattern.split_whitespace();
        let name = words.next()?;
        let name = name.rsplit(['/', '\\']).next().unwrap_or(name);
        let name = name
            .strip_suffix(".exe")
            .or_else(|| name.strip_suffix(".EXE"))
            .unwrap_or(name);
        Some(Self::new(name, words.map(str::to_owned), true))
    }

    pub fn has_options(&self) -> bool {
        !self.short.is_empty() || !self.long.is_empty()
    }

    /// True when this command carries every option of `pattern` and its positional words in order.
    pub fn matches(&self, pattern: &Self) -> bool {
        if self.name != pattern.name
            || !pattern.short.is_subset(&self.short)
            || !pattern.long.is_subset(&self.long)
        {
            return false;
        }
        let mut rest = self.positional.iter();
        pattern
            .positional
            .iter()
            .all(|p| rest.by_ref().any(|w| w == p))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionRequest {
    pub tool: String,
    pub risk: RiskClass,
    pub location: Location,
    pub reasons: Vec<String>,
    /// Shell command text, matched against `safety.always_ask_patterns`.
    pub command: Option<String>,
    /// The command's simple commands, for option-order-independent pattern matching.
    pub shapes: Vec<CommandShape>,
    /// Set by tools whose action must be confirmed in every mode but plan; the reason shown to the user.
    pub always_ask: Option<String>,
    /// Keys for "allow for session"; empty means the tool name alone.
    pub grant_keys: Vec<String>,
}

impl ActionRequest {
    pub fn new(tool: impl Into<String>, risk: RiskClass, location: Location) -> Self {
        Self {
            tool: tool.into(),
            risk,
            location,
            reasons: Vec::new(),
            command: None,
            shapes: Vec::new(),
            always_ask: None,
            grant_keys: Vec::new(),
        }
    }

    fn keys(&self) -> Vec<String> {
        if self.grant_keys.is_empty() {
            vec![self.tool.clone()]
        } else {
            self.grant_keys
                .iter()
                .map(|k| format!("{}:{k}", self.tool))
                .collect()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskInfo {
    pub reasons: Vec<String>,
    /// Whether "allow for session" may be offered.
    pub allow_session: bool,
    /// Whether the prompt should preselect deny.
    pub default_deny: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny { reason: String },
    Ask(AskInfo),
}

pub struct PermissionEngine {
    mode: Mode,
    /// Normalized pattern text and, for patterns with options, its shape.
    always_ask: Vec<(String, Option<CommandShape>)>,
    grants: HashSet<String>,
}

fn normalize(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

impl PermissionEngine {
    pub fn new(mode: Mode, always_ask_patterns: &[String]) -> Self {
        Self {
            mode,
            always_ask: always_ask_patterns
                .iter()
                .map(|p| normalize(p))
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let shape = CommandShape::from_pattern(&p).filter(CommandShape::has_options);
                    (p, shape)
                })
                .collect(),
            grants: HashSet::new(),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    /// Patterns match the normalized text (also with quotes removed) or the shape of any simple command.
    fn always_ask_match(&self, req: &ActionRequest) -> Option<&str> {
        let cmd = req.command.as_deref()?;
        // Trailing space lets patterns such as "format " match at the end of a command.
        let text = format!("{} ", normalize(cmd));
        let dequoted = format!("{} ", normalize(&cmd.replace(['\'', '"'], "")));
        self.always_ask
            .iter()
            .find(|(p, shape)| {
                text.contains(p.as_str())
                    || dequoted.contains(p.as_str())
                    || shape
                        .as_ref()
                        .is_some_and(|s| req.shapes.iter().any(|c| c.matches(s)))
            })
            .map(|(p, _)| p.as_str())
    }

    pub fn decide(&self, req: &ActionRequest) -> Decision {
        if req.risk == RiskClass::Internal {
            return Decision::Allow;
        }
        if self.mode == Mode::Plan {
            return if req.risk == RiskClass::Read && req.location == Location::Inside {
                Decision::Allow
            } else {
                Decision::Deny {
                    reason: if req.location == Location::Inside {
                        format!("plan mode is read-only; `{}` is not allowed", req.tool)
                    } else {
                        "plan mode cannot touch paths outside the working directory".into()
                    },
                }
            };
        }
        if req.location != Location::Inside {
            let mut reasons = req.reasons.clone();
            reasons.push(match req.location {
                Location::Outside => "touches a location outside the working directory".into(),
                _ => "cannot verify that it stays inside the working directory".into(),
            });
            return Decision::Ask(AskInfo {
                reasons,
                allow_session: false,
                default_deny: true,
            });
        }
        if let Some(reason) = &req.always_ask {
            let mut reasons = req.reasons.clone();
            reasons.push(reason.clone());
            return Decision::Ask(AskInfo {
                reasons,
                allow_session: false,
                default_deny: false,
            });
        }
        if let Some(pattern) = self.always_ask_match(req) {
            let mut reasons = req.reasons.clone();
            reasons.push(format!("matches always-ask pattern `{pattern}`"));
            return Decision::Ask(AskInfo {
                reasons,
                allow_session: false,
                default_deny: true,
            });
        }
        let automatic = match self.mode {
            Mode::Auto => true,
            Mode::AutoEdit => matches!(req.risk, RiskClass::Read | RiskClass::Write),
            Mode::Cowork | Mode::Plan => false,
        };
        if automatic || req.keys().iter().all(|k| self.grants.contains(k)) {
            return Decision::Allow;
        }
        Decision::Ask(AskInfo {
            reasons: req.reasons.clone(),
            allow_session: true,
            default_deny: false,
        })
    }

    /// Records "allow for session". Refused for anything that would not have offered it.
    pub fn grant_session(&mut self, req: &ActionRequest) -> bool {
        let grantable = req.location == Location::Inside
            && req.risk != RiskClass::Internal
            && self.mode != Mode::Plan
            && req.always_ask.is_none()
            && self.always_ask_match(req).is_none();
        if grantable {
            self.grants.extend(req.keys());
        }
        grantable
    }

    pub fn clear_grants(&mut self) {
        self.grants.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    enum Expect {
        Allow,
        Deny,
        Ask,
    }

    /// DESIGN §12.1, written out independently of the engine.
    fn expected(mode: Mode, risk: RiskClass, location: Location) -> Expect {
        use Expect::*;
        if risk == RiskClass::Internal {
            return Allow;
        }
        if location != Location::Inside {
            return if mode == Mode::Plan { Deny } else { Ask };
        }
        match (mode, risk) {
            (Mode::Plan, RiskClass::Read) => Allow,
            (Mode::Plan, _) => Deny,
            (Mode::Cowork, _) => Ask,
            (Mode::AutoEdit, RiskClass::Read | RiskClass::Write) => Allow,
            (Mode::AutoEdit, _) => Ask,
            (Mode::Auto, _) => Allow,
        }
    }

    #[test]
    fn full_permission_matrix() {
        let mut checked = 0;
        for &mode in Mode::ALL {
            let engine = PermissionEngine::new(mode, &[]);
            for &risk in RiskClass::ALL {
                for location in [Location::Inside, Location::Unverifiable, Location::Outside] {
                    let req = ActionRequest::new("t", risk, location);
                    let got = match engine.decide(&req) {
                        Decision::Allow => Expect::Allow,
                        Decision::Deny { .. } => Expect::Deny,
                        Decision::Ask(info) => {
                            if location != Location::Inside {
                                assert!(
                                    info.default_deny && !info.allow_session,
                                    "{mode} {risk} {location:?}"
                                );
                            }
                            Expect::Ask
                        }
                    };
                    assert_eq!(
                        got,
                        expected(mode, risk, location),
                        "{mode} {risk} {location:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 4 * 7 * 3);
    }

    fn shell_req(cmd: &str, keys: &[&str]) -> ActionRequest {
        ActionRequest {
            command: Some(cmd.into()),
            grant_keys: keys.iter().map(|k| (*k).to_owned()).collect(),
            ..ActionRequest::new("shell", RiskClass::Exec, Location::Inside)
        }
    }

    #[test]
    fn always_ask_patterns_apply_even_in_auto() {
        let patterns = [
            "git push --force".to_owned(),
            "rm -rf".to_owned(),
            "format ".to_owned(),
        ];
        let engine = PermissionEngine::new(Mode::Auto, &patterns);
        for cmd in [
            "git  push   --force origin main",
            "RM -RF target",
            "format c:",
            "echo x; format",
        ] {
            assert!(
                matches!(engine.decide(&shell_req(cmd, &[])), Decision::Ask(_)),
                "{cmd}"
            );
        }
        assert_eq!(
            engine.decide(&shell_req("git push origin", &[])),
            Decision::Allow
        );
        assert_eq!(engine.decide(&shell_req("cargo fmt", &[])), Decision::Allow);
        let plan = PermissionEngine::new(Mode::Plan, &patterns);
        assert!(matches!(
            plan.decide(&shell_req("rm -rf x", &[])),
            Decision::Deny { .. }
        ));
    }
    fn shaped(cmd: &str, shapes: &[(&str, &[&str])]) -> ActionRequest {
        ActionRequest {
            shapes: shapes
                .iter()
                .map(|(name, args)| {
                    CommandShape::new(name, args.iter().map(|a| (*a).to_owned()), false)
                })
                .collect(),
            ..shell_req(cmd, &[])
        }
    }

    #[test]
    fn always_ask_patterns_survive_option_order_and_quoting() {
        let patterns = [
            "rm -rf".to_owned(),
            "git push --force".to_owned(),
            "Remove-Item -Recurse -Force".to_owned(),
        ];
        let engine = PermissionEngine::new(Mode::Auto, &patterns);
        let asks = |req: &ActionRequest| matches!(engine.decide(req), Decision::Ask(_));
        assert!(asks(&shell_req("rm -r'f' target", &[])), "quotes removed");
        for (cmd, shape) in [
            ("rm -fr target", ("rm", &["-fr", "target"][..])),
            ("rm -r -f target", ("rm", &["-r", "-f", "target"][..])),
            (
                "rm -f -i -r target",
                ("rm", &["-f", "-i", "-r", "target"][..]),
            ),
            ("/bin/rm -Rf target", ("rm", &["-Rf", "target"][..])),
            (
                "git -c x=y push --force origin",
                ("git", &["-c", "x=y", "push", "--force", "origin"][..]),
            ),
            (
                "rm -r -fo x",
                ("remove-item", &["-recurse", "-force", "x"][..]),
            ),
        ] {
            assert!(asks(&shaped(cmd, &[shape])), "{cmd}");
        }
        for (cmd, shape) in [
            ("rm -r target", ("rm", &["-r", "target"][..])),
            ("rm -f target", ("rm", &["-f", "target"][..])),
            ("git --force", ("git", &["--force"][..])),
            (
                "Remove-Item -Recurse x",
                ("remove-item", &["-recurse", "x"][..]),
            ),
        ] {
            assert!(!asks(&shaped(cmd, &[shape])), "{cmd}");
        }
        // A nested command's shape counts too.
        assert!(asks(&shaped(
            "sh -c 'rm -fr x'",
            &[("sh", &["-c", "rm -fr x"]), ("rm", &["-fr", "x"])]
        )));
    }

    #[test]
    fn tool_declared_always_ask_overrides_automatic_modes() {
        let engine = PermissionEngine::new(Mode::Auto, &[]);
        let req = ActionRequest {
            always_ask: Some("creates a custom agent".into()),
            ..ActionRequest::new("agent_create", RiskClass::Write, Location::Inside)
        };
        match engine.decide(&req) {
            Decision::Ask(info) => {
                assert!(!info.allow_session && !info.default_deny);
                assert_eq!(info.reasons, ["creates a custom agent"]);
            }
            other => panic!("{other:?}"),
        }
        let mut cowork = PermissionEngine::new(Mode::Cowork, &[]);
        assert!(!cowork.grant_session(&req));
        assert!(matches!(
            PermissionEngine::new(Mode::Plan, &[]).decide(&req),
            Decision::Deny { .. }
        ));
    }
    #[test]
    fn session_grants_cover_all_keys_only() {
        let mut engine = PermissionEngine::new(Mode::Cowork, &["rm -rf".to_owned()]);
        let test = shell_req("cargo test", &["cargo test"]);
        assert!(matches!(
            engine.decide(&test),
            Decision::Ask(AskInfo {
                allow_session: true,
                ..
            })
        ));
        assert!(engine.grant_session(&test));
        assert_eq!(
            engine.decide(&shell_req("cargo test --release", &["cargo test"])),
            Decision::Allow
        );
        assert!(matches!(
            engine.decide(&shell_req("cargo publish", &["cargo publish"])),
            Decision::Ask(_)
        ));
        let compound = shell_req("cargo test && curl x", &["cargo test", "curl"]);
        assert!(matches!(engine.decide(&compound), Decision::Ask(_)));

        assert!(!engine.grant_session(&shell_req("rm -rf target", &["rm"])));
        let outside = ActionRequest::new("read_file", RiskClass::Read, Location::Outside);
        assert!(!engine.grant_session(&outside));
        assert!(matches!(engine.decide(&outside), Decision::Ask(_)));

        let read = ActionRequest::new("read_file", RiskClass::Read, Location::Inside);
        assert!(engine.grant_session(&read));
        assert_eq!(engine.decide(&read), Decision::Allow);
        engine.clear_grants();
        assert!(matches!(engine.decide(&read), Decision::Ask(_)));
    }
}
