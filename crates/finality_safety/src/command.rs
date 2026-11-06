//! Shell command analyzer (DESIGN §12.3): classifies a command as Inside, Outside or Unverifiable.
//!
//! Directory changes are tracked rather than rejected outright: every directory the command may be
//! in is collected, and each path is checked against all of them. `cd src && cargo test` stays
//! Inside, while any target outside (or unknown) is reported.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::path::{PathGuard, PathVerdict};
use crate::permission::CommandShape;
use crate::shell::{Part, ShellKind, SimpleCommand, Word, parse};

const MAX_DEPTH: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Location {
    Inside,
    Unverifiable,
    Outside,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    DirectoryChange,
    DirFlag,
    OutsidePath,
    VariableExpansion,
    NestedInterpreter,
    Substitution,
    Redirection,
    LinkCreation,
    Privilege,
    Remote,
    Unparseable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub category: Category,
    pub location: Location,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandAnalysis {
    pub location: Location,
    pub findings: Vec<Finding>,
    /// `command subcommand` keys used for "allow for session" grants, one per simple command.
    pub grant_keys: Vec<String>,
    /// Every simple command (nested ones included) for always-ask pattern matching.
    pub shapes: Vec<CommandShape>,
}

impl CommandAnalysis {
    pub fn has(&self, category: Category) -> bool {
        self.findings.iter().any(|f| f.category == category)
    }

    pub fn reasons(&self) -> Vec<String> {
        self.findings.iter().map(|f| f.detail.clone()).collect()
    }
}

/// Environment used to expand variables during analysis (normally the shell tool's filtered env).
#[derive(Debug, Clone, Default)]
pub struct EnvSnapshot {
    vars: HashMap<String, String>,
}

impl EnvSnapshot {
    pub fn from_process() -> Self {
        Self::from_pairs(std::env::vars())
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            vars: pairs.into_iter().collect(),
        }
    }

    pub fn get(&self, name: &str, case_insensitive: bool) -> Option<&str> {
        if let Some(v) = self.vars.get(name) {
            return Some(v);
        }
        if case_insensitive {
            return self
                .vars
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_str());
        }
        None
    }

    pub fn home(&self) -> Option<PathBuf> {
        self.get("HOME", false)
            .or_else(|| self.get("USERPROFILE", true))
            .map(PathBuf::from)
    }
}

// ------------------------------------------------------------------ rule table

const POSIX_KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "do", "done", "while", "until", "for", "case", "esac",
    "in", "select", "function", "!", "{", "}",
];
const PS_KEYWORDS: &[&str] = &[
    "if", "elseif", "else", "foreach", "for", "while", "do", "until", "switch", "try", "catch",
    "finally", "trap", "return", "throw", "exit", "break", "continue", "function", "filter",
    "param", "begin", "process", "end", "in", "&", ".",
];
/// PowerShell automatic variables whose values never come from the environment.
const PS_AUTOMATIC: &[&str] = &[
    "_",
    "psitem",
    "true",
    "false",
    "null",
    "args",
    "input",
    "lastexitcode",
    "?",
    "matches",
    "error",
    "pwd",
    "psscriptroot",
    "pscommandpath",
    "this",
];

const WRAPPERS: &[&str] = &[
    "nohup", "time", "command", "builtin", "exec", "nice", "timeout", "stdbuf", "ionice",
];
const CD_COMMANDS: &[&str] = &[
    "cd",
    "chdir",
    "pushd",
    "set-location",
    "sl",
    "push-location",
];
const POP_COMMANDS: &[&str] = &["popd", "pop-location"];
const ALWAYS_NESTED: &[&str] = &[
    "eval",
    "invoke-expression",
    "iex",
    "invoke-command",
    "icm",
    "start-job",
    "sajb",
    "xargs",
    "parallel",
    "watch",
    "wsl",
];
const PRIVILEGE: &[&str] = &["sudo", "su", "doas", "runas", "pkexec", "gsudo"];
const REMOTE: &[&str] = &[
    "ssh",
    "scp",
    "sftp",
    "telnet",
    "ftp",
    "mstsc",
    "enter-pssession",
    "new-pssession",
    "psexec",
];
const SHELLS: &[&str] = &["bash", "sh", "zsh", "dash", "ksh", "fish"];

/// Interpreter → flags that take inline code.
const INLINE_CODE: &[(&str, &[&str])] = &[
    ("python", &["-c"]),
    ("python3", &["-c"]),
    ("python2", &["-c"]),
    ("py", &["-c"]),
    ("pypy", &["-c"]),
    ("node", &["-e", "--eval", "-p", "--print"]),
    ("nodejs", &["-e", "--eval", "-p", "--print"]),
    ("perl", &["-e", "-E"]),
    ("ruby", &["-e"]),
    ("php", &["-r"]),
    ("deno", &["eval", "-e"]),
    ("bun", &["-e", "--eval"]),
    ("osascript", &["-e"]),
];

/// Flags whose value becomes the working directory of the command.
const DIR_FLAGS: &[(&str, &[&str])] = &[
    ("git", &["-c", "--git-dir", "--work-tree"]),
    ("make", &["-c", "--directory"]),
    ("gmake", &["-c", "--directory"]),
    ("tar", &["-c", "--directory"]),
    ("npm", &["--prefix", "-c"]),
    ("pnpm", &["--dir", "-c", "--prefix"]),
    ("yarn", &["--cwd"]),
    ("cargo", &["--manifest-path", "--target-dir", "-c"]),
    ("go", &["-c"]),
    ("start", &["/d"]),
];
const GENERIC_DIR_FLAGS: &[&str] = &[
    "--cwd",
    "--directory",
    "--working-directory",
    "-workingdirectory",
    "-wd",
];
/// Native Windows programs whose `/x` arguments are switches even when run from PowerShell.
const SLASH_SWITCH_PROGRAMS: &[&str] = &[
    "cmd", "robocopy", "xcopy", "taskkill", "tasklist", "reg", "sc", "netsh", "icacls", "attrib",
    "findstr", "where", "ipconfig", "shutdown", "msiexec", "runas", "start", "mklink", "subst",
    "schtasks", "wmic", "certutil", "dism", "sfc", "net",
];

// ------------------------------------------------------------------ analyzer

pub struct CommandAnalyzer<'a> {
    guard: &'a PathGuard,
    env: &'a EnvSnapshot,
}

struct State {
    cwds: Vec<PathBuf>,
    findings: Vec<Finding>,
    grant_keys: Vec<String>,
    shapes: Vec<CommandShape>,
}

impl State {
    fn add(&mut self, category: Category, location: Location, detail: impl Into<String>) {
        self.findings.push(Finding {
            category,
            location,
            detail: detail.into(),
        });
    }
}

enum Value {
    Known(String),
    /// Only PowerShell automatic variables were unknown; `skeleton` is the literal remainder.
    Automatic {
        skeleton: String,
    },
    Unknown {
        category: Category,
        reason: String,
        skeleton: String,
    },
}

impl<'a> CommandAnalyzer<'a> {
    pub fn new(guard: &'a PathGuard, env: &'a EnvSnapshot) -> Self {
        Self { guard, env }
    }

    pub fn analyze(&self, shell: ShellKind, command: &str) -> CommandAnalysis {
        let mut st = State {
            cwds: vec![self.guard.workdir().to_path_buf()],
            findings: Vec::new(),
            grant_keys: Vec::new(),
            shapes: Vec::new(),
        };
        self.run(shell, command, &mut st, 0);
        let location = st
            .findings
            .iter()
            .map(|f| f.location)
            .max()
            .unwrap_or(Location::Inside);
        st.grant_keys.dedup();
        CommandAnalysis {
            location,
            findings: st.findings,
            grant_keys: st.grant_keys,
            shapes: st.shapes,
        }
    }

    fn run(&self, shell: ShellKind, command: &str, st: &mut State, depth: u32) {
        if depth > MAX_DEPTH {
            st.add(
                Category::Unparseable,
                Location::Unverifiable,
                "commands nested too deeply",
            );
            return;
        }
        let parsed = match parse(shell, command) {
            Ok(p) => p,
            Err(e) => {
                st.add(
                    Category::Unparseable,
                    Location::Unverifiable,
                    format!("cannot parse command: {e}"),
                );
                return;
            }
        };
        if parsed.heredoc {
            st.add(
                Category::Unparseable,
                Location::Unverifiable,
                "here-documents are not analyzed",
            );
        }
        for cmd in &parsed.commands {
            self.simple(shell, cmd, st, depth);
        }
    }

    fn simple(&self, shell: ShellKind, cmd: &SimpleCommand, st: &mut State, depth: u32) {
        let targets = cmd.redirects.iter().filter_map(|r| r.target.as_ref());
        for w in cmd.words.iter().chain(targets) {
            for sub in &w.subcommands {
                self.run(shell, sub, st, depth + 1);
            }
        }
        for r in &cmd.redirects {
            if let Some(t) = &r.target
                && r.op != "<<<"
            {
                self.redirect_target(shell, t, st);
            }
        }
        self.dispatch(shell, &cmd.words, st, depth);
    }

    fn redirect_target(&self, shell: ShellKind, target: &Word, st: &mut State) {
        match self.expand(shell, target) {
            Value::Known(t) => {
                let lower = t.to_ascii_lowercase();
                if matches!(
                    lower.as_str(),
                    "/dev/null" | "/dev/stdout" | "/dev/stderr" | "/dev/tty" | "nul" | "nul:"
                ) {
                    return;
                }
                self.check_path(shell, &t, st, Category::Redirection);
            }
            Value::Automatic { .. } => {}
            Value::Unknown { reason, .. } => {
                st.add(
                    Category::Redirection,
                    Location::Unverifiable,
                    format!("redirect target {reason}"),
                );
            }
        }
    }

    fn dispatch(&self, shell: ShellKind, words: &[Word], st: &mut State, depth: u32) {
        let mut i = 0;
        while let Some(w) = words.get(i) {
            let lit = w.literal().map(|l| l.to_ascii_lowercase());
            let keyword = match shell {
                ShellKind::Posix => POSIX_KEYWORDS,
                ShellKind::PowerShell => PS_KEYWORDS,
                ShellKind::Cmd => &[],
            };
            if lit.as_deref().is_some_and(|l| keyword.contains(&l)) {
                i += 1;
                continue;
            }
            if shell == ShellKind::Posix && self.assignment(w, st) {
                i += 1;
                continue;
            }
            let ps_assign = shell == ShellKind::PowerShell
                && matches!(w.parts.as_slice(), [Part::Var(_)])
                && words.get(i + 1).and_then(Word::literal).is_some_and(|op| {
                    matches!(op.as_str(), "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "??=")
                });
            if ps_assign {
                i += 2;
                continue;
            }
            break;
        }
        let Some(name_word) = words.get(i) else {
            return;
        };
        let args = &words[i + 1..];

        let name = match self.expand(shell, name_word) {
            Value::Known(n) => n,
            // PowerShell expression statement such as `$_.Name` inside a script block.
            Value::Automatic { .. } => {
                self.check_args(shell, "", args, st);
                return;
            }
            Value::Unknown { .. } => {
                st.add(
                    Category::NestedInterpreter,
                    Location::Unverifiable,
                    format!(
                        "command name `{}` is computed at run time",
                        name_word.display()
                    ),
                );
                self.check_args(shell, "", args, st);
                return;
            }
        };
        if path_like(&name, shell) {
            self.check_path(shell, &name, st, Category::OutsidePath);
        }
        let base = command_base(&name, shell);
        st.grant_keys.push(grant_key(&base, args, shell));
        st.shapes.push(shape(shell, &base, args));

        if self.apply_rules(shell, &base, args, st, depth) {
            return;
        }
        self.check_args(shell, &base, args, st);
    }

    /// Returns true when the rule fully handled the arguments.
    fn apply_rules(
        &self,
        shell: ShellKind,
        base: &str,
        args: &[Word],
        st: &mut State,
        depth: u32,
    ) -> bool {
        let lits: Vec<String> = args
            .iter()
            .map(|a| a.literal().unwrap_or_default().to_ascii_lowercase())
            .collect();
        let has = |flags: &[&str]| lits.iter().any(|l| flags.contains(&l.as_str()));

        if shell == ShellKind::Posix && WRAPPERS.contains(&base) {
            let start = lits
                .iter()
                .position(|l| !(l.starts_with('-') || is_duration(l)))
                .unwrap_or(args.len());
            self.dispatch(shell, &args[start..], st, depth);
            return true;
        }
        if shell == ShellKind::Posix && base == "env" {
            let mut start = 0;
            while let Some(a) = args.get(start) {
                if lits[start].starts_with('-') {
                    st.add(
                        Category::NestedInterpreter,
                        Location::Unverifiable,
                        "env with options",
                    );
                } else if !self.assignment(a, st) {
                    break;
                }
                start += 1;
            }
            self.dispatch(shell, &args[start..], st, depth);
            return true;
        }
        if PRIVILEGE.contains(&base) {
            st.add(
                Category::Privilege,
                Location::Outside,
                format!("`{base}` runs with elevated privileges"),
            );
            if shell == ShellKind::Posix && base != "su" {
                self.dispatch(shell, args, st, depth);
                return true;
            }
            return false;
        }
        if REMOTE.contains(&base) {
            st.add(
                Category::Remote,
                Location::Outside,
                format!("`{base}` operates on a remote host"),
            );
            return false;
        }
        if base == "rsync" && lits.iter().any(|l| is_remote_spec(l)) {
            st.add(
                Category::Remote,
                Location::Outside,
                "rsync to or from a remote host",
            );
            return false;
        }
        if CD_COMMANDS.contains(&base) {
            self.change_dir(shell, args, st);
            return true;
        }
        if POP_COMMANDS.contains(&base) {
            return true;
        }
        if ALWAYS_NESTED.contains(&base) {
            st.add(
                Category::NestedInterpreter,
                Location::Unverifiable,
                format!("`{base}` runs commands that cannot be analyzed"),
            );
            return false;
        }
        if base == "find" && has(&["-exec", "-execdir", "-ok", "-okdir"]) {
            st.add(
                Category::NestedInterpreter,
                Location::Unverifiable,
                "find -exec runs other commands",
            );
            return false;
        }
        if SHELLS.contains(&base) {
            let inline = lits.iter().any(|l| {
                l == "-c" || (l.starts_with('-') && !l.starts_with("--") && l.contains('c'))
            });
            if inline {
                st.add(
                    Category::NestedInterpreter,
                    Location::Unverifiable,
                    format!("`{base} -c` runs inline code"),
                );
            }
            return false;
        }
        if matches!(base, "pwsh" | "powershell") {
            let inline = lits.iter().any(|l| {
                l.len() >= 2
                    && l.starts_with('-')
                    && ("-command".starts_with(l.as_str())
                        || "-encodedcommand".starts_with(l.as_str())
                        || "-commandwithargs".starts_with(l.as_str())
                        || l == "-ec")
            });
            let file = lits
                .iter()
                .any(|l| l.len() >= 2 && l.starts_with("-f") && "-file".starts_with(l.as_str()));
            if inline || !file {
                st.add(
                    Category::NestedInterpreter,
                    Location::Unverifiable,
                    format!("`{base}` runs code that cannot be analyzed"),
                );
            }
            return false;
        }
        if base == "cmd"
            && lits
                .iter()
                .any(|l| l.starts_with("/c") || l.starts_with("/k") || l.starts_with("/r"))
        {
            st.add(
                Category::NestedInterpreter,
                Location::Unverifiable,
                "`cmd /c` runs inline code",
            );
            return false;
        }
        if let Some((_, flags)) = INLINE_CODE.iter().find(|(n, _)| *n == base)
            && has(flags)
        {
            st.add(
                Category::NestedInterpreter,
                Location::Unverifiable,
                format!("`{base}` runs inline code"),
            );
            return false;
        }
        if shell == ShellKind::Cmd && matches!(base, "if" | "for") {
            st.add(
                Category::NestedInterpreter,
                Location::Unverifiable,
                format!("cmd `{base}` constructs are not analyzed"),
            );
            return false;
        }
        match base {
            "ln" | "mklink" => {
                st.add(
                    Category::LinkCreation,
                    Location::Unverifiable,
                    format!("`{base}` creates a link"),
                );
            }
            "new-item" | "ni" => {
                let link = lits
                    .iter()
                    .any(|l| matches!(l.as_str(), "symboliclink" | "junction" | "hardlink"));
                if link {
                    st.add(
                        Category::LinkCreation,
                        Location::Unverifiable,
                        "New-Item creates a link",
                    );
                }
            }
            "subst" | "mount" => {
                st.add(
                    Category::LinkCreation,
                    Location::Outside,
                    format!("`{base}` maps another location"),
                );
            }
            "net" if lits.first().is_some_and(|l| l == "use") => {
                st.add(
                    Category::LinkCreation,
                    Location::Outside,
                    "`net use` maps a network drive",
                );
            }
            "start-process" | "saps" | "start"
                if lits
                    .windows(2)
                    .any(|w| w[0].starts_with("-verb") && w[1] == "runas") =>
            {
                st.add(
                    Category::Privilege,
                    Location::Outside,
                    "Start-Process -Verb RunAs elevates",
                );
            }
            _ => {}
        }
        false
    }

    fn change_dir(&self, shell: ShellKind, args: &[Word], st: &mut State) {
        let mut target: Option<&Word> = None;
        let mut k = 0;
        while let Some(a) = args.get(k) {
            let lit = a.literal().unwrap_or_default().to_ascii_lowercase();
            let is_path_param = shell == ShellKind::PowerShell
                && lit.len() >= 2
                && ("-path".starts_with(lit.as_str()) || "-literalpath".starts_with(lit.as_str()));
            if is_path_param {
                target = args.get(k + 1);
                break;
            }
            let skip = match shell {
                ShellKind::Posix => matches!(lit.as_str(), "-l" | "-p" | "-e" | "-@" | "--"),
                ShellKind::Cmd => lit == "/d",
                ShellKind::PowerShell => lit.starts_with('-') && lit.len() > 1,
            };
            if !skip {
                target = Some(a);
                break;
            }
            k += 1;
        }
        let text = match target {
            None if shell == ShellKind::Cmd => return,
            None => match self.env.home() {
                Some(h) => h.display().to_string(),
                None => {
                    st.add(
                        Category::DirectoryChange,
                        Location::Unverifiable,
                        "cd to an unknown home directory",
                    );
                    return;
                }
            },
            Some(w) if w.literal().as_deref() == Some("-") => {
                st.add(
                    Category::DirectoryChange,
                    Location::Unverifiable,
                    "`cd -` returns to an unknown directory",
                );
                return;
            }
            Some(w) => match self.expand(shell, w) {
                Value::Known(t) => t,
                _ => {
                    st.add(
                        Category::DirectoryChange,
                        Location::Unverifiable,
                        format!("cd target `{}` is computed at run time", w.display()),
                    );
                    return;
                }
            },
        };
        self.enter_dir(shell, &text, st, Category::DirectoryChange);
    }

    /// Adds `dir` (relative to every possible cwd) to the possible cwds, or reports it.
    fn enter_dir(&self, shell: ShellKind, dir: &str, st: &mut State, category: Category) {
        if shell == ShellKind::PowerShell && is_provider_path(dir) {
            st.add(
                category,
                Location::Outside,
                format!("`{dir}` is a PowerShell provider path"),
            );
            return;
        }
        let mut entered = Vec::new();
        for cwd in &st.cwds {
            match self.guard.check_from(cwd, dir) {
                PathVerdict::Inside { resolved } => entered.push(resolved),
                PathVerdict::Outside { reason, .. } => {
                    st.add(category, Location::Outside, format!("`{dir}` {reason}"));
                    return;
                }
            }
        }
        for e in entered {
            if !st.cwds.contains(&e) {
                st.cwds.push(e);
            }
        }
    }

    fn check_args(&self, shell: ShellKind, base: &str, args: &[Word], st: &mut State) {
        let slash = shell == ShellKind::Cmd || SLASH_SWITCH_PROGRAMS.contains(&base);
        let dir_flags: Vec<&str> = DIR_FLAGS
            .iter()
            .filter(|(n, _)| *n == base)
            .flat_map(|(_, f)| f.iter().copied())
            .chain(GENERIC_DIR_FLAGS.iter().copied())
            .collect();
        let is_dir_flag = |f: &str| dir_flags.contains(&f.to_ascii_lowercase().as_str());

        let mut k = 0;
        while let Some(w) = args.get(k) {
            k += 1;
            let text = match self.expand(shell, w) {
                Value::Known(t) => t,
                Value::Automatic { skeleton } => {
                    if has_path_syntax(&skeleton) {
                        st.add(
                            Category::VariableExpansion,
                            Location::Unverifiable,
                            format!("`{}` builds a path from a run-time value", w.display()),
                        );
                    }
                    continue;
                }
                Value::Unknown {
                    category,
                    reason,
                    skeleton,
                } => {
                    let ps_block = shell == ShellKind::PowerShell
                        && category == Category::Substitution
                        && !has_path_syntax(&skeleton);
                    if !ps_block {
                        st.add(
                            category,
                            Location::Unverifiable,
                            format!("`{}`: {reason}", w.display()),
                        );
                    }
                    continue;
                }
            };

            if let Some((flag, value)) = split_flag_value(&text, shell) {
                if is_dir_flag(flag) {
                    self.enter_dir(shell, value, st, Category::DirFlag);
                } else if path_like(value, shell) {
                    self.check_path(shell, value, st, Category::OutsidePath);
                }
                continue;
            }
            if is_flag(&text, slash) {
                if is_dir_flag(&text) {
                    match args.get(k).map(|v| self.expand(shell, v)) {
                        Some(Value::Known(v)) => self.enter_dir(shell, &v, st, Category::DirFlag),
                        Some(_) => st.add(
                            Category::DirFlag,
                            Location::Unverifiable,
                            format!("`{text}` value is computed at run time"),
                        ),
                        None => {}
                    }
                    k += 1;
                } else if !slash && !text.starts_with("--") && text.len() > 2 {
                    let attached = &text[2..];
                    if attached.contains(['/', '\\']) {
                        self.check_path(shell, attached, st, Category::OutsidePath);
                    }
                }
                continue;
            }
            if path_like(&text, shell) {
                self.check_path(shell, &text, st, Category::OutsidePath);
            }
        }
    }

    fn check_path(&self, shell: ShellKind, text: &str, st: &mut State, category: Category) {
        if shell == ShellKind::PowerShell && is_provider_path(text) {
            st.add(
                category,
                Location::Outside,
                format!("`{text}` is a PowerShell provider path"),
            );
            return;
        }
        for cwd in &st.cwds {
            if let PathVerdict::Outside { reason, .. } = self.guard.check_from(cwd, text) {
                st.add(category, Location::Outside, format!("`{text}` {reason}"));
                return;
            }
        }
    }

    /// `NAME=value`: checks a path-like value and returns true if the word was an assignment.
    fn assignment(&self, w: &Word, st: &mut State) -> bool {
        let Some(Part::Lit(first)) = w.parts.first() else {
            return false;
        };
        let Some((name, rest)) = first.split_once('=') else {
            return false;
        };
        let valid_name = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid_name {
            return false;
        }
        let mut value = Word {
            parts: vec![Part::Lit(rest.to_owned())],
            subcommands: Vec::new(),
        };
        value.parts.extend(w.parts[1..].iter().cloned());
        if let Value::Known(v) = self.expand(ShellKind::Posix, &value)
            && path_like(&v, ShellKind::Posix)
        {
            self.check_path(ShellKind::Posix, &v, st, Category::OutsidePath);
        }
        true
    }

    fn expand(&self, shell: ShellKind, w: &Word) -> Value {
        let mut out = String::new();
        let mut skeleton = String::new();
        let mut automatic = false;
        let mut unknown: Option<(Category, String)> = None;
        for p in &w.parts {
            match p {
                Part::Lit(s) => {
                    out.push_str(s);
                    skeleton.push_str(s);
                }
                Part::Tilde(user) if user.is_empty() => match self.env.home() {
                    Some(h) => out.push_str(&h.display().to_string()),
                    None => {
                        unknown.get_or_insert((
                            Category::VariableExpansion,
                            "home directory is unknown".into(),
                        ));
                    }
                },
                Part::Tilde(user) => {
                    unknown.get_or_insert((
                        Category::VariableExpansion,
                        format!("~{user} is another user's home"),
                    ));
                }
                Part::Var(name) => match self.lookup(shell, name) {
                    Some(v) => out.push_str(&v),
                    None if shell == ShellKind::PowerShell
                        && PS_AUTOMATIC.contains(&name.to_ascii_lowercase().as_str()) =>
                    {
                        automatic = true;
                    }
                    None => {
                        unknown.get_or_insert((
                            Category::VariableExpansion,
                            format!("variable `{name}` is not set"),
                        ));
                    }
                },
                Part::Dynamic(raw) => {
                    unknown.get_or_insert((
                        Category::Substitution,
                        format!("`{raw}` is computed at run time"),
                    ));
                }
            }
        }
        match unknown {
            Some((category, reason)) => Value::Unknown {
                category,
                reason,
                skeleton,
            },
            None if automatic => Value::Automatic { skeleton },
            None => Value::Known(out),
        }
    }

    fn lookup(&self, shell: ShellKind, name: &str) -> Option<String> {
        match shell {
            ShellKind::Posix => self.env.get(name, false).map(str::to_owned),
            ShellKind::Cmd => self.env.get(name, true).map(str::to_owned),
            ShellKind::PowerShell => {
                let lower = name.to_ascii_lowercase();
                if let Some(var) = lower.strip_prefix("env:") {
                    return self.env.get(var, true).map(str::to_owned);
                }
                if lower == "home" {
                    return self.env.home().map(|h| h.display().to_string());
                }
                None
            }
        }
    }
}

// ------------------------------------------------------------------ helpers

fn command_base(name: &str, shell: ShellKind) -> String {
    let name = if shell == ShellKind::Cmd {
        name.trim_start_matches('@')
    } else {
        name
    };
    let file = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase();
    for ext in [".exe", ".com", ".cmd", ".bat"] {
        if let Some(stripped) = file.strip_suffix(ext) {
            return stripped.to_owned();
        }
    }
    file
}

fn grant_key(base: &str, args: &[Word], shell: ShellKind) -> String {
    let slash = shell == ShellKind::Cmd || SLASH_SWITCH_PROGRAMS.contains(&base);
    let sub = args.first().and_then(Word::literal).filter(|s| {
        !s.is_empty()
            && s.len() <= 32
            && !is_flag(s, slash)
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    });
    match sub {
        Some(s) => format!("{base} {}", s.to_ascii_lowercase()),
        None => base.to_owned(),
    }
}

fn is_flag(text: &str, slash_switches: bool) -> bool {
    let dash = text.len() > 1 && text.starts_with('-') && text != "--";
    dash || (slash_switches && is_cmd_switch(text))
}

/// The command's shape for always-ask matching. PowerShell's `rm`/`del`/`ri`/... aliases become
/// `remove-item`, and its abbreviated `-Recurse`/`-Force` parameters are spelled out.
fn shape(shell: ShellKind, base: &str, args: &[Word]) -> CommandShape {
    let mut name = base.to_owned();
    let mut tokens: Vec<String> = args
        .iter()
        .map(|w| w.literal().unwrap_or_else(|| w.display()))
        .collect();
    if shell == ShellKind::PowerShell {
        if matches!(
            name.as_str(),
            "rm" | "ri" | "del" | "erase" | "rd" | "rmdir"
        ) {
            name = "remove-item".into();
        }
        if name == "remove-item" {
            for t in &mut tokens {
                let lower = t.to_ascii_lowercase();
                let Some(p) = lower.strip_prefix('-') else {
                    continue;
                };
                if p.is_empty() || p.starts_with('-') {
                    continue;
                }
                if "recurse".starts_with(p) {
                    *t = "-recurse".into();
                } else if "force".starts_with(p) {
                    *t = "-force".into();
                }
            }
        }
    }
    let slash = shell == ShellKind::Cmd || SLASH_SWITCH_PROGRAMS.contains(&base);
    CommandShape::new(&name, tokens, slash)
}

/// `/q`, `/s`, `/A:H` — cmd switches rather than root-relative paths.
fn is_cmd_switch(text: &str) -> bool {
    let Some(rest) = text.strip_prefix('/') else {
        return false;
    };
    let short = rest.len() <= 2 || rest.get(1..2) == Some(":");
    short
        && rest
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '?')
        && !rest.contains(['/', '\\'])
}

/// `--flag=value` or PowerShell `-Flag:value`.
fn split_flag_value(text: &str, shell: ShellKind) -> Option<(&str, &str)> {
    if text.starts_with("--") {
        return text.split_once('=');
    }
    if shell == ShellKind::PowerShell && text.starts_with('-') {
        return text.split_once(':');
    }
    None
}

fn is_url(text: &str) -> bool {
    let Some((scheme, _)) = text.split_once("://") else {
        return false;
    };
    scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
}

fn has_path_syntax(text: &str) -> bool {
    text.contains(['/', '\\']) || text.contains("..")
}

/// `HKLM:`, `Env:\x`, `Cert:` — not a filesystem path (single letters are drives).
fn is_provider_path(text: &str) -> bool {
    match text.split_once(':') {
        Some((drive, _)) => drive.len() >= 2 && drive.chars().all(|c| c.is_ascii_alphanumeric()),
        None => false,
    }
}

fn path_like(text: &str, shell: ShellKind) -> bool {
    if text.is_empty() || is_url(text) {
        return false;
    }
    let b = text.as_bytes();
    text.starts_with('~')
        || has_path_syntax(text)
        || (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
        || (shell == ShellKind::PowerShell && is_provider_path(text))
}

fn is_duration(text: &str) -> bool {
    !text.is_empty()
        && text
            .trim_end_matches(['s', 'm', 'h', 'd'])
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.')
}

/// `host:path` or `user@host:path` (but not a drive letter).
fn is_remote_spec(text: &str) -> bool {
    match text.split_once(':') {
        Some((host, _)) => host.len() > 1 && !host.contains(['/', '\\']),
        None => false,
    }
}
