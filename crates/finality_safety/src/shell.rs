//! Tokenizers for POSIX shells, PowerShell and cmd. They only need to be precise enough for the
//! command analyzer: anything they cannot model is reported as an error so the caller treats the
//! command as unverifiable.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShellKind {
    Posix,
    PowerShell,
    Cmd,
}

impl ShellKind {
    pub fn from_program(program: &str) -> Option<Self> {
        let stem = Path::new(program)
            .file_stem()?
            .to_string_lossy()
            .to_ascii_lowercase();
        match stem.as_str() {
            "bash" | "sh" | "zsh" | "dash" | "ksh" => Some(ShellKind::Posix),
            "pwsh" | "powershell" => Some(ShellKind::PowerShell),
            "cmd" => Some(ShellKind::Cmd),
            _ => None,
        }
    }

    pub fn case_insensitive(self) -> bool {
        !matches!(self, ShellKind::Posix)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Lit(String),
    /// Variable reference by name (`HOME`, `env:TEMP`, `USERPROFILE`).
    Var(String),
    /// Leading `~` or `~user` (user is empty for the current user).
    Tilde(String),
    /// Value only known at run time: command/process substitution, arithmetic, special parameters.
    Dynamic(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Word {
    pub parts: Vec<Part>,
    /// Bodies of `$(...)`, backticks, `<(...)`, PowerShell `$(...)`, `@(...)` and `{...}` blocks.
    pub subcommands: Vec<String>,
}

impl Word {
    pub fn lit(text: &str) -> Self {
        Self {
            parts: vec![Part::Lit(text.to_owned())],
            subcommands: Vec::new(),
        }
    }

    fn push_char(&mut self, c: char) {
        if let Some(Part::Lit(s)) = self.parts.last_mut() {
            s.push(c);
        } else {
            self.parts.push(Part::Lit(c.to_string()));
        }
    }

    /// The text if the word contains no expansions.
    pub fn literal(&self) -> Option<String> {
        let mut out = String::new();
        for p in &self.parts {
            match p {
                Part::Lit(s) => out.push_str(s),
                _ => return None,
            }
        }
        Some(out)
    }

    /// Source-like rendering for messages.
    pub fn display(&self) -> String {
        self.parts
            .iter()
            .map(|p| match p {
                Part::Lit(s) => s.clone(),
                Part::Var(v) => format!("${v}"),
                Part::Tilde(u) => format!("~{u}"),
                Part::Dynamic(d) => d.clone(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redirect {
    pub op: String,
    /// `None` for descriptor duplication such as `2>&1`.
    pub target: Option<Word>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SimpleCommand {
    pub words: Vec<Word>,
    pub redirects: Vec<Redirect>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Parsed {
    pub commands: Vec<SimpleCommand>,
    pub heredoc: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ParseError(pub String);

pub fn parse(shell: ShellKind, input: &str) -> Result<Parsed, ParseError> {
    let items = match shell {
        ShellKind::Posix => lex_posix(input)?,
        ShellKind::PowerShell => lex_powershell(input)?,
        ShellKind::Cmd => lex_cmd(input)?,
    };
    assemble(items)
}

enum Item {
    Word(Word),
    Op(String),
}

fn is_control(op: &str) -> bool {
    matches!(op, ";" | "&&" | "||" | "|" | "&" | "(" | ")")
}

fn is_dup(op: &str) -> bool {
    (op.contains(">&") || op.contains("<&"))
        && op
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_digit() || c == '-')
}

fn assemble(items: Vec<Item>) -> Result<Parsed, ParseError> {
    let mut parsed = Parsed::default();
    let mut cur = SimpleCommand::default();
    let mut iter = items.into_iter();
    let flush = |cur: &mut SimpleCommand, parsed: &mut Parsed| {
        if !cur.words.is_empty() || !cur.redirects.is_empty() {
            parsed.commands.push(std::mem::take(cur));
        }
    };
    while let Some(item) = iter.next() {
        match item {
            Item::Word(w) => cur.words.push(w),
            Item::Op(op) if is_control(&op) => flush(&mut cur, &mut parsed),
            Item::Op(op) if op == "<<" || op == "<<-" => {
                parsed.heredoc = true;
                let _delimiter = iter.next();
            }
            Item::Op(op) if is_dup(&op) => cur.redirects.push(Redirect { op, target: None }),
            Item::Op(op) => match iter.next() {
                Some(Item::Word(target)) => cur.redirects.push(Redirect {
                    op,
                    target: Some(target),
                }),
                _ => return Err(ParseError(format!("missing target after `{op}`"))),
            },
        }
    }
    flush(&mut cur, &mut parsed);
    Ok(parsed)
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            i: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.chars.get(self.i + n).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    fn starts_with(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(n, c)| self.peek_at(n) == Some(c))
    }

    fn skip_line(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                return;
            }
            self.i += 1;
        }
    }

    fn take_while(&mut self, f: impl Fn(char) -> bool) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek().filter(|&c| f(c)) {
            s.push(c);
            self.i += 1;
        }
        s
    }

    /// Reads up to the matching `close` (the opener is already consumed), honoring quotes and nesting.
    fn balanced(&mut self, open: char, close: char, escape: char) -> Result<String, ParseError> {
        let mut depth = 1;
        let mut out = String::new();
        let mut quote: Option<char> = None;
        while let Some(c) = self.bump() {
            if c == escape && quote != Some('\'') {
                out.push(c);
                if let Some(n) = self.bump() {
                    out.push(n);
                }
                continue;
            }
            match quote {
                Some(q) if c == q => quote = None,
                Some(_) => {}
                None if c == '\'' || c == '"' => quote = Some(c),
                None if c == open => depth += 1,
                None if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(out);
                    }
                }
                None => {}
            }
            out.push(c);
        }
        Err(ParseError(format!("unterminated `{open}`")))
    }

    /// Consumes digits (or `*`) followed by a redirection operator, if present at the cursor.
    fn fd_redirect(&mut self, allow_star: bool) -> Option<String> {
        let mut j = 0;
        while self
            .peek_at(j)
            .is_some_and(|c| c.is_ascii_digit() || (allow_star && c == '*'))
        {
            j += 1;
        }
        if j == 0 || !matches!(self.peek_at(j), Some('>' | '<')) {
            return None;
        }
        let mut op: String = (0..j).filter_map(|n| self.peek_at(n)).collect();
        self.i += j;
        op.push_str(&self.redirect_op()?);
        Some(op)
    }

    /// At `>` or `<`: reads `>`, `>>`, `>|`, `>&N`, `>&`, `<`, `<&N`.
    fn redirect_op(&mut self) -> Option<String> {
        let first = self.bump()?;
        let mut op = first.to_string();
        if first == '>' && matches!(self.peek(), Some('>' | '|')) {
            op.push(self.bump()?);
        }
        if self.peek() == Some('&') {
            op.push('&');
            self.i += 1;
            op.push_str(&self.take_while(|c| c.is_ascii_digit() || c == '-'));
        }
        Some(op)
    }
}

fn is_blank(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r')
}

// ---------------------------------------------------------------- POSIX

fn lex_posix(input: &str) -> Result<Vec<Item>, ParseError> {
    let mut lx = Lexer::new(input);
    let mut items = Vec::new();
    while let Some(c) = lx.peek() {
        match c {
            c if is_blank(c) => {
                lx.bump();
            }
            '\n' => {
                lx.bump();
                items.push(Item::Op(";".into()));
            }
            '#' => lx.skip_line(),
            ';' => {
                lx.bump();
                if lx.peek() == Some(';') {
                    lx.bump();
                }
                items.push(Item::Op(";".into()));
            }
            '&' if lx.starts_with("&&") => {
                lx.i += 2;
                items.push(Item::Op("&&".into()));
            }
            '&' if lx.starts_with("&>") => {
                lx.i += 2;
                let op = if lx.peek() == Some('>') {
                    lx.bump();
                    "&>>"
                } else {
                    "&>"
                };
                items.push(Item::Op(op.into()));
            }
            '&' => {
                lx.bump();
                items.push(Item::Op("&".into()));
            }
            '|' => {
                lx.bump();
                let op = match lx.peek() {
                    Some('|') => {
                        lx.bump();
                        "||"
                    }
                    Some('&') => {
                        lx.bump();
                        "|"
                    }
                    _ => "|",
                };
                items.push(Item::Op(op.into()));
            }
            '(' | ')' => {
                lx.bump();
                items.push(Item::Op(c.to_string()));
            }
            '<' if lx.starts_with("<<<") => {
                lx.i += 3;
                items.push(Item::Op("<<<".into()));
            }
            '<' if lx.starts_with("<<") => {
                lx.i += 2;
                let op = if lx.peek() == Some('-') {
                    lx.bump();
                    "<<-"
                } else {
                    "<<"
                };
                items.push(Item::Op(op.into()));
            }
            '<' | '>' if lx.peek_at(1) == Some('(') => items.push(Item::Word(posix_word(&mut lx)?)),
            '<' | '>' => {
                let op = lx
                    .redirect_op()
                    .ok_or_else(|| ParseError("bad redirection".into()))?;
                items.push(Item::Op(op));
            }
            d if d.is_ascii_digit() => match lx.fd_redirect(false) {
                Some(op) => items.push(Item::Op(op)),
                None => items.push(Item::Word(posix_word(&mut lx)?)),
            },
            _ => items.push(Item::Word(posix_word(&mut lx)?)),
        }
    }
    Ok(items)
}

fn posix_word(lx: &mut Lexer) -> Result<Word, ParseError> {
    let mut w = Word::default();
    if lx.peek() == Some('~') {
        lx.bump();
        let user = lx.take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
        w.parts.push(Part::Tilde(user));
    }
    while let Some(c) = lx.peek() {
        match c {
            c if is_blank(c) || matches!(c, '\n' | ';' | '&' | '|' | '(' | ')') => break,
            '<' | '>' if lx.peek_at(1) == Some('(') => {
                lx.i += 2;
                let inner = lx.balanced('(', ')', '\\')?;
                w.parts.push(Part::Dynamic(format!("{c}({inner})")));
                w.subcommands.push(inner);
            }
            '<' | '>' => break,
            '\\' => {
                lx.bump();
                match lx.bump() {
                    Some('\n') | None => {}
                    Some(n) => w.push_char(n),
                }
            }
            '\'' => {
                lx.bump();
                let mut closed = false;
                while let Some(n) = lx.bump() {
                    if n == '\'' {
                        closed = true;
                        break;
                    }
                    w.push_char(n);
                }
                if !closed {
                    return Err(ParseError("unterminated single quote".into()));
                }
                ensure_part(&mut w);
            }
            '"' => {
                lx.bump();
                let mut closed = false;
                while let Some(n) = lx.peek() {
                    match n {
                        '"' => {
                            lx.bump();
                            closed = true;
                            break;
                        }
                        '\\' => {
                            lx.bump();
                            match lx.bump() {
                                Some(e @ ('$' | '`' | '"' | '\\')) => w.push_char(e),
                                Some('\n') => {}
                                Some(e) => {
                                    w.push_char('\\');
                                    w.push_char(e);
                                }
                                None => break,
                            }
                        }
                        '$' => posix_dollar(lx, &mut w)?,
                        '`' => posix_backtick(lx, &mut w)?,
                        _ => {
                            lx.bump();
                            w.push_char(n);
                        }
                    }
                }
                if !closed {
                    return Err(ParseError("unterminated double quote".into()));
                }
                ensure_part(&mut w);
            }
            '$' => posix_dollar(lx, &mut w)?,
            '`' => posix_backtick(lx, &mut w)?,
            _ => {
                lx.bump();
                w.push_char(c);
            }
        }
    }
    Ok(w)
}

/// Keeps `''` and `""` as an explicit empty word.
fn ensure_part(w: &mut Word) {
    if w.parts.is_empty() {
        w.parts.push(Part::Lit(String::new()));
    }
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn posix_dollar(lx: &mut Lexer, w: &mut Word) -> Result<(), ParseError> {
    lx.bump();
    match lx.peek() {
        Some('(') if lx.peek_at(1) == Some('(') => {
            lx.i += 2;
            let inner = lx.balanced('(', ')', '\\')?;
            if lx.peek() == Some(')') {
                lx.bump();
            }
            w.parts.push(Part::Dynamic(format!("$(({inner}))")));
        }
        Some('(') => {
            lx.bump();
            let inner = lx.balanced('(', ')', '\\')?;
            w.parts.push(Part::Dynamic(format!("$({inner})")));
            w.subcommands.push(inner);
        }
        Some('{') => {
            lx.bump();
            let inner = lx.balanced('{', '}', '\\')?;
            let plain =
                inner.chars().next().is_some_and(is_name_start) && inner.chars().all(is_name_char);
            w.parts.push(if plain {
                Part::Var(inner)
            } else {
                Part::Dynamic(format!("${{{inner}}}"))
            });
        }
        Some(c) if is_name_start(c) => {
            let name = lx.take_while(is_name_char);
            w.parts.push(Part::Var(name));
        }
        Some(c) if c.is_ascii_digit() || matches!(c, '@' | '*' | '#' | '?' | '$' | '!' | '-') => {
            lx.bump();
            w.parts.push(Part::Dynamic(format!("${c}")));
        }
        _ => w.push_char('$'),
    }
    Ok(())
}

fn posix_backtick(lx: &mut Lexer, w: &mut Word) -> Result<(), ParseError> {
    lx.bump();
    let mut inner = String::new();
    loop {
        match lx.bump() {
            Some('`') => break,
            Some('\\') => {
                if let Some(n) = lx.bump() {
                    inner.push(n);
                }
            }
            Some(c) => inner.push(c),
            None => return Err(ParseError("unterminated backtick".into())),
        }
    }
    w.parts.push(Part::Dynamic(format!("`{inner}`")));
    w.subcommands.push(inner);
    Ok(())
}

// ---------------------------------------------------------------- PowerShell

fn lex_powershell(input: &str) -> Result<Vec<Item>, ParseError> {
    let mut lx = Lexer::new(input);
    let mut items = Vec::new();
    while let Some(c) = lx.peek() {
        match c {
            c if is_blank(c) => {
                lx.bump();
            }
            '\n' | ';' => {
                lx.bump();
                items.push(Item::Op(";".into()));
            }
            '#' => lx.skip_line(),
            '<' if lx.peek_at(1) == Some('#') => {
                lx.i += 2;
                while !lx.starts_with("#>") {
                    if lx.bump().is_none() {
                        return Err(ParseError("unterminated block comment".into()));
                    }
                }
                lx.i += 2;
            }
            '|' => {
                lx.bump();
                if lx.peek() == Some('|') {
                    lx.bump();
                    items.push(Item::Op("||".into()));
                } else {
                    items.push(Item::Op("|".into()));
                }
            }
            '&' if lx.starts_with("&&") => {
                lx.i += 2;
                items.push(Item::Op("&&".into()));
            }
            '&' => {
                lx.bump();
                items.push(Item::Word(Word::lit("&")));
            }
            '(' | ')' => {
                lx.bump();
                items.push(Item::Op(c.to_string()));
            }
            '{' => {
                lx.bump();
                let inner = lx.balanced('{', '}', '`')?;
                items.push(Item::Word(Word {
                    parts: vec![Part::Dynamic(format!("{{{inner}}}"))],
                    subcommands: vec![inner],
                }));
            }
            '}' => return Err(ParseError("unbalanced `}`".into())),
            '>' | '<' => {
                let op = lx
                    .redirect_op()
                    .ok_or_else(|| ParseError("bad redirection".into()))?;
                items.push(Item::Op(op));
            }
            d if d.is_ascii_digit() || d == '*' => match lx.fd_redirect(true) {
                Some(op) => items.push(Item::Op(op)),
                None => items.push(Item::Word(ps_word(&mut lx)?)),
            },
            _ => items.push(Item::Word(ps_word(&mut lx)?)),
        }
    }
    Ok(items)
}

fn ps_word(lx: &mut Lexer) -> Result<Word, ParseError> {
    let mut w = Word::default();
    if lx.peek() == Some('~')
        && lx
            .peek_at(1)
            .is_none_or(|c| matches!(c, '/' | '\\') || is_blank(c))
    {
        lx.bump();
        w.parts.push(Part::Tilde(String::new()));
    }
    while let Some(c) = lx.peek() {
        match c {
            c if is_blank(c)
                || matches!(c, '\n' | ';' | '|' | '(' | ')' | '{' | '}' | '>' | '<') =>
            {
                break;
            }
            '&' if lx.peek_at(1) == Some('&') => break,
            '`' => {
                lx.bump();
                if let Some(n) = lx.bump() {
                    w.push_char(n);
                }
            }
            '\'' => {
                lx.bump();
                loop {
                    match lx.bump() {
                        Some('\'') if lx.peek() == Some('\'') => {
                            lx.bump();
                            w.push_char('\'');
                        }
                        Some('\'') => break,
                        Some(n) => w.push_char(n),
                        None => return Err(ParseError("unterminated single quote".into())),
                    }
                }
                ensure_part(&mut w);
            }
            '"' => {
                lx.bump();
                loop {
                    match lx.peek() {
                        Some('"') if lx.peek_at(1) == Some('"') => {
                            lx.i += 2;
                            w.push_char('"');
                        }
                        Some('"') => {
                            lx.bump();
                            break;
                        }
                        Some('`') => {
                            lx.bump();
                            if let Some(n) = lx.bump() {
                                w.push_char(n);
                            }
                        }
                        Some('$') => ps_dollar(lx, &mut w)?,
                        Some(n) => {
                            lx.bump();
                            w.push_char(n);
                        }
                        None => return Err(ParseError("unterminated double quote".into())),
                    }
                }
                ensure_part(&mut w);
            }
            '$' => ps_dollar(lx, &mut w)?,
            '@' if matches!(lx.peek_at(1), Some('(' | '{')) => {
                let (open, close) = if lx.peek_at(1) == Some('(') {
                    ('(', ')')
                } else {
                    ('{', '}')
                };
                lx.i += 2;
                let inner = lx.balanced(open, close, '`')?;
                w.parts
                    .push(Part::Dynamic(format!("@{open}{inner}{close}")));
                w.subcommands.push(inner);
            }
            _ => {
                lx.bump();
                w.push_char(c);
            }
        }
    }
    Ok(w)
}

fn ps_dollar(lx: &mut Lexer, w: &mut Word) -> Result<(), ParseError> {
    lx.bump();
    match lx.peek() {
        Some('(') => {
            lx.bump();
            let inner = lx.balanced('(', ')', '`')?;
            w.parts.push(Part::Dynamic(format!("$({inner})")));
            w.subcommands.push(inner);
        }
        Some('{') => {
            lx.bump();
            let mut name = String::new();
            loop {
                match lx.bump() {
                    Some('}') => break,
                    Some(c) => name.push(c),
                    None => return Err(ParseError("unterminated `${`".into())),
                }
            }
            w.parts.push(Part::Var(name));
        }
        Some(c) if is_name_char(c) || c == '?' => {
            let mut name = lx.take_while(|c| is_name_char(c) || c == '?');
            // Scope or drive qualifier such as `env:`, `global:`.
            if lx.peek() == Some(':') && lx.peek_at(1).is_some_and(is_name_char) {
                lx.bump();
                name.push(':');
                name.push_str(&lx.take_while(is_name_char));
            }
            w.parts.push(Part::Var(name));
        }
        Some(c @ ('$' | '^')) => {
            lx.bump();
            w.parts.push(Part::Dynamic(format!("${c}")));
        }
        _ => w.push_char('$'),
    }
    Ok(())
}

// ---------------------------------------------------------------- cmd

fn lex_cmd(input: &str) -> Result<Vec<Item>, ParseError> {
    let mut lx = Lexer::new(input);
    let mut items = Vec::new();
    while let Some(c) = lx.peek() {
        match c {
            c if is_blank(c) => {
                lx.bump();
            }
            '\n' => {
                lx.bump();
                items.push(Item::Op(";".into()));
            }
            '&' | '|' => {
                lx.bump();
                if lx.peek() == Some(c) {
                    lx.bump();
                    items.push(Item::Op(format!("{c}{c}")));
                } else {
                    items.push(Item::Op(c.to_string()));
                }
            }
            '(' | ')' => {
                lx.bump();
                items.push(Item::Op(c.to_string()));
            }
            '>' | '<' => {
                let op = lx
                    .redirect_op()
                    .ok_or_else(|| ParseError("bad redirection".into()))?;
                items.push(Item::Op(op));
            }
            d if d.is_ascii_digit() => match lx.fd_redirect(false) {
                Some(op) => items.push(Item::Op(op)),
                None => items.push(Item::Word(cmd_word(&mut lx)?)),
            },
            _ => items.push(Item::Word(cmd_word(&mut lx)?)),
        }
    }
    Ok(items)
}

fn cmd_word(lx: &mut Lexer) -> Result<Word, ParseError> {
    let mut w = Word::default();
    let mut in_quotes = false;
    while let Some(c) = lx.peek() {
        if !in_quotes && (is_blank(c) || matches!(c, '\n' | '&' | '|' | '(' | ')' | '<' | '>')) {
            break;
        }
        match c {
            '^' if !in_quotes => {
                lx.bump();
                if let Some(n) = lx.bump() {
                    w.push_char(n);
                }
            }
            '"' => {
                lx.bump();
                in_quotes = !in_quotes;
                if !in_quotes {
                    ensure_part(&mut w);
                }
            }
            '%' => cmd_percent(lx, &mut w),
            _ => {
                lx.bump();
                w.push_char(c);
            }
        }
    }
    if in_quotes {
        return Err(ParseError("unterminated double quote".into()));
    }
    Ok(w)
}

fn cmd_percent(lx: &mut Lexer, w: &mut Word) {
    lx.bump();
    match lx.peek() {
        Some('%') => {
            lx.bump();
            w.push_char('%');
        }
        Some(c) if c == '~' || c == '*' || c.is_ascii_digit() => {
            let m = lx.take_while(|c| {
                c == '~' || c == '*' || c.is_ascii_alphanumeric() || c == '$' || c == ':'
            });
            w.parts.push(Part::Dynamic(format!("%{m}")));
        }
        _ => {
            let rest: String = lx.chars[lx.i..].iter().collect();
            match rest.find('%') {
                Some(end)
                    if end > 0
                        && !rest[..end].contains(char::is_whitespace)
                        && !rest[..end].contains('"') =>
                {
                    let name = rest[..end].to_owned();
                    lx.i += name.chars().count() + 1;
                    w.parts.push(if name.contains(':') {
                        Part::Dynamic(format!("%{name}%"))
                    } else {
                        Part::Var(name)
                    });
                }
                _ => w.push_char('%'),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(shell: ShellKind, input: &str) -> Vec<Vec<String>> {
        parse(shell, input)
            .unwrap()
            .commands
            .iter()
            .map(|c| c.words.iter().map(Word::display).collect())
            .collect()
    }

    #[test]
    fn posix_quoting_and_operators() {
        assert_eq!(
            words(
                ShellKind::Posix,
                r#"echo 'a b' "c $HOME" d\ e; ls -la && cat x || true | wc"#
            ),
            vec![
                vec!["echo", "a b", "c $HOME", "d e"],
                vec!["ls", "-la"],
                vec!["cat", "x"],
                vec!["true"],
                vec!["wc"]
            ]
        );
    }

    #[test]
    fn posix_expansions() {
        let p = parse(
            ShellKind::Posix,
            r#"cat ~/x ~bob/y $(ls /etc) `pwd` ${HOME}/z ${X:-y} $1 <(sort f)"#,
        )
        .unwrap();
        let w = &p.commands[0].words;
        assert_eq!(w[1].parts[0], Part::Tilde(String::new()));
        assert_eq!(w[2].parts[0], Part::Tilde("bob".into()));
        assert_eq!(w[3].subcommands, vec!["ls /etc"]);
        assert_eq!(w[4].subcommands, vec!["pwd"]);
        assert_eq!(w[5].parts[0], Part::Var("HOME".into()));
        assert!(matches!(w[6].parts[0], Part::Dynamic(_)));
        assert!(matches!(w[7].parts[0], Part::Dynamic(_)));
        assert_eq!(w[8].subcommands, vec!["sort f"]);
        assert_eq!(w.len(), 9);
    }

    #[test]
    fn posix_redirections() {
        let p = parse(
            ShellKind::Posix,
            "make 2>&1 >out.log 2>> err.txt < in &> all",
        )
        .unwrap();
        let r = &p.commands[0].redirects;
        assert_eq!(
            r[0],
            Redirect {
                op: "2>&1".into(),
                target: None
            }
        );
        assert_eq!(r[1].target.as_ref().unwrap().display(), "out.log");
        assert_eq!(r[2].op, "2>>");
        assert_eq!(r[3].op, "<");
        assert_eq!(r[4].op, "&>");
        assert_eq!(p.commands[0].words.len(), 1);
    }

    #[test]
    fn posix_heredoc_and_errors() {
        assert!(
            parse(ShellKind::Posix, "cat <<EOF\nhi\nEOF")
                .unwrap()
                .heredoc
        );
        for bad in ["echo 'x", "echo \"x", "echo $(ls", "echo `x", "cat >"] {
            assert!(parse(ShellKind::Posix, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn powershell_tokens() {
        let p = parse(
            ShellKind::PowerShell,
            r#"Get-ChildItem -Path "$env:TEMP\x" | ForEach-Object { Remove-Item $_ } ; & 'C:\a b\t.exe' `"q 2>&1 > $null"#,
        )
        .unwrap();
        let c = &p.commands;
        assert_eq!(c[0].words[2].parts[0], Part::Var("env:TEMP".into()));
        assert_eq!(c[1].words[1].subcommands, vec![" Remove-Item $_ "]);
        assert_eq!(c[2].words[0].display(), "&");
        assert_eq!(c[2].words[1].display(), r"C:\a b\t.exe");
        assert_eq!(c[2].words[2].display(), "\"q");
        assert_eq!(c[2].redirects[0].op, "2>&1");
        assert_eq!(
            c[2].redirects[1].target.as_ref().unwrap().parts[0],
            Part::Var("null".into())
        );
    }

    #[test]
    fn powershell_subexpressions_and_comments() {
        let p = parse(
            ShellKind::PowerShell,
            "Write-Output \"a $(Get-Location) b\" # trailing\n<# block #> Get-Item @(1,2)",
        )
        .unwrap();
        assert_eq!(p.commands[0].words[1].subcommands, vec!["Get-Location"]);
        assert_eq!(p.commands[1].words[0].display(), "Get-Item");
        assert!(parse(ShellKind::PowerShell, "Get-Item 'x").is_err());
        assert!(parse(ShellKind::PowerShell, "}").is_err());
    }

    #[test]
    fn cmd_tokens() {
        let p = parse(
            ShellKind::Cmd,
            r#"type "%USERPROFILE%\a b.txt" & del /q x^&y > NUL 2>&1 && echo %~dp0 100%%"#,
        )
        .unwrap();
        let c = &p.commands;
        assert_eq!(c[0].words[1].parts[0], Part::Var("USERPROFILE".into()));
        assert_eq!(c[0].words[1].display(), r"$USERPROFILE\a b.txt");
        assert_eq!(
            c[1].words.iter().map(Word::display).collect::<Vec<_>>(),
            vec!["del", "/q", "x&y"]
        );
        assert_eq!(c[1].redirects.len(), 2);
        assert!(matches!(c[2].words[1].parts[0], Part::Dynamic(_)));
        assert_eq!(c[2].words[2].display(), "100%");
        assert!(parse(ShellKind::Cmd, "echo \"x").is_err());
    }

    #[test]
    fn shell_detection() {
        assert_eq!(ShellKind::from_program("/bin/bash"), Some(ShellKind::Posix));
        assert_eq!(
            ShellKind::from_program(r"C:\Program Files\PowerShell\7\pwsh.exe"),
            Some(ShellKind::PowerShell)
        );
        assert_eq!(ShellKind::from_program("CMD.EXE"), Some(ShellKind::Cmd));
        assert_eq!(ShellKind::from_program("python"), None);
    }
}
