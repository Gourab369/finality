//! `mcp.json` files (DESIGN §9.4), in Claude Code's `mcpServers` format.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// `~/.finality/mcp.json`: trusted by the user who wrote it.
    Global,
    /// `<workdir>/.mcp.json`: needs a trust decision per project.
    Project,
    /// The Finality editor extension, found through `<data>/ide/` lock files.
    Ide,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<String>,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerDef {
    pub name: String,
    pub origin: Origin,
    pub transport: Transport,
    /// Tools annotated `readOnlyHint` run as reads (no approval in `cowork`). Global and editor servers only.
    pub read_only_hints: bool,
    /// Remote tool names to offer (`-` and `_` match each other); `None` offers every tool.
    pub tools: Option<Vec<String>>,
    /// Hash of the raw entry; trust and the tool cache are tied to it.
    pub fingerprint: String,
}

impl ServerDef {
    /// What the server runs or connects to, for prompts and `/mcp list`.
    pub fn target(&self) -> String {
        if self.origin == Origin::Ide {
            return "VS Code (Finality extension)".into();
        }
        match &self.transport {
            Transport::Stdio { command, args, .. } => std::iter::once(command.as_str())
                .chain(args.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(" "),
            Transport::Http { url, .. } => url.clone(),
        }
    }

    /// Everything a trust decision should see besides the target: environment overrides (values that
    /// look like credentials are hidden), the working directory, and header names.
    pub fn trust_details(&self) -> Vec<String> {
        let show = |k: &str, v: &str| {
            if finality_core::secrets::contains_secret(v) || v.len() > 60 {
                format!("{k}=[hidden]")
            } else {
                format!("{k}={v}")
            }
        };
        let mut out = Vec::new();
        match &self.transport {
            Transport::Stdio { env, cwd, .. } => {
                if !env.is_empty() {
                    let pairs: Vec<String> = env.iter().map(|(k, v)| show(k, v)).collect();
                    out.push(format!("environment: {}", pairs.join(", ")));
                }
                if let Some(c) = cwd {
                    out.push(format!("working directory: {c}"));
                }
            }
            Transport::Http { headers, .. } => {
                if !headers.is_empty() {
                    let names: Vec<&str> = headers.keys().map(String::as_str).collect();
                    out.push(format!("headers: {}", names.join(", ")));
                }
            }
        }
        out
    }
}

#[derive(Debug, Default)]
pub struct Loaded {
    pub servers: Vec<ServerDef>,
    pub problems: Vec<String>,
}

#[derive(Deserialize)]
struct File {
    #[serde(rename = "mcpServers", default)]
    servers: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawServer {
    #[serde(rename = "type")]
    kind: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    cwd: Option<String>,
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    trust_read_only_hints: bool,
    tools: Option<Vec<String>>,
}

const KEYRING_PREFIX: &str = "keyring:";

/// `${keyring:<user>}`: the secret stored under service `finality` and that user name.
fn from_keyring(user: &str) -> Option<String> {
    keyring::Entry::new("finality", user)
        .ok()?
        .get_password()
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// Reads the global file, then the project file; a project server replaces a global one of the same name.
/// The global file may reference any environment variable and keyring entries (`${keyring:<user>}`); the project
/// file only variables in `project_env` (the denylist-filtered environment), so it cannot forward the user's secrets.
pub fn load(global: &Path, project: Option<&Path>, project_env: &[(String, String)]) -> Loaded {
    let mut loaded = Loaded::default();
    let from_process = |var: &str| match var.strip_prefix(KEYRING_PREFIX) {
        Some(user) => from_keyring(user),
        None => std::env::var(var).ok(),
    };
    let from_filtered = |var: &str| {
        project_env
            .iter()
            .find(|(k, _)| k == var)
            .map(|(_, v)| v.clone())
    };
    for (path, origin) in [(Some(global), Origin::Global), (project, Origin::Project)] {
        let Some(path) = path else { continue };
        let lookup: &dyn Fn(&str) -> Option<String> = match origin {
            Origin::Global | Origin::Ide => &from_process,
            Origin::Project => &from_filtered,
        };
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                loaded
                    .problems
                    .push(format!("cannot read {}: {e}", path.display()));
                continue;
            }
        };
        let file: File = match serde_json::from_str(&text) {
            Ok(f) => f,
            Err(e) => {
                loaded.problems.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        for (name, raw) in file.servers {
            match parse_server(&name, &raw, origin, lookup) {
                Ok(Some(def)) => {
                    if origin == Origin::Project
                        && raw.get("trustReadOnlyHints").and_then(Value::as_bool) == Some(true)
                    {
                        loaded.problems.push(format!(
                            "{}: server `{name}`: trustReadOnlyHints is only honored in the global mcp.json",
                            path.display()
                        ));
                    }
                    loaded.servers.retain(|s| s.name != def.name);
                    loaded.servers.push(def);
                }
                Ok(None) => {}
                Err(e) => loaded
                    .problems
                    .push(format!("{}: server `{name}`: {e}", path.display())),
            }
        }
    }
    loaded
}

fn parse_server(
    name: &str,
    raw: &Value,
    origin: Origin,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<ServerDef>, String> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("names may only contain letters, digits, `_` and `-`".into());
    }
    let s: RawServer = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    if s.disabled {
        return Ok(None);
    }
    let expand = |text: &str| expand(text, lookup);
    let map = |m: BTreeMap<String, String>| -> Result<BTreeMap<String, String>, String> {
        m.into_iter().map(|(k, v)| Ok((k, expand(&v)?))).collect()
    };
    let transport = match (s.kind.as_deref(), s.command, s.url) {
        (None | Some("stdio"), Some(command), _) => Transport::Stdio {
            command: expand(&command)?,
            args: s.args.iter().map(|a| expand(a)).collect::<Result<_, _>>()?,
            env: map(s.env)?,
            cwd: s.cwd.map(|c| expand(&c)).transpose()?,
        },
        (None | Some("http" | "streamable-http" | "streamableHttp"), _, Some(url)) => {
            let url = expand(&url)?;
            match url::Url::parse(&url) {
                Ok(u) if matches!(u.scheme(), "http" | "https") => {}
                _ => return Err(format!("`{url}` is not an http(s) URL")),
            }
            Transport::Http {
                url,
                headers: map(s.headers)?,
            }
        }
        (Some("sse"), ..) => {
            return Err(
                "the SSE transport is not supported; use streamable HTTP (`\"type\": \"http\"`)"
                    .into(),
            );
        }
        (Some(kind @ ("stdio" | "http" | "streamable-http" | "streamableHttp")), ..) => {
            return Err(format!(
                "type `{kind}` needs {}",
                if kind == "stdio" {
                    "`command`"
                } else {
                    "`url`"
                }
            ));
        }
        (Some(kind), ..) => return Err(format!("unknown type `{kind}`")),
        (None, None, None) => return Err("needs `command` or `url`".into()),
    };
    Ok(Some(ServerDef {
        name: name.to_owned(),
        origin,
        transport,
        // A project file must not be able to switch off approvals for its own server.
        read_only_hints: s.trust_read_only_hints && origin == Origin::Global,
        tools: s.tools,
        fingerprint: fingerprint(raw),
    }))
}

/// Expands `${VAR}` and `${VAR:-default}` through `lookup`.
fn expand(text: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| format!("unclosed `${{` in `{text}`"))?;
        let (var, default) = match after[..end].split_once(":-") {
            Some((v, d)) => (v, Some(d)),
            None => (&after[..end], None),
        };
        match (lookup(var), default) {
            (Some(v), _) if !v.is_empty() => out.push_str(&v),
            (_, Some(d)) => out.push_str(d),
            (Some(v), None) => out.push_str(&v),
            (None, None) => {
                return Err(match var.strip_prefix(KEYRING_PREFIX) {
                    Some(user) => format!(
                        "no keyring entry with service `finality` and user `{user}` (keyring references work only in \
                         the global mcp.json)"
                    ),
                    None => format!(
                        "environment variable `{var}` is not set or not available to this file"
                    ),
                });
            }
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// FNV-1a over the entry's canonical JSON (object keys are sorted).
pub fn fingerprint(raw: &Value) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in raw.to_string().bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn loads_global_and_project_servers() {
        let tmp = tempfile::tempdir().unwrap();
        let global = write(
            tmp.path(),
            "mcp.json",
            r#"{"mcpServers": {
                "files": {"command": "npx", "args": ["-y", "server-files", "${FINALITY_TEST_UNSET:-.}"]},
                "docs": {"type": "http", "url": "https://docs.example/mcp", "headers": {"X-Key": "k"}},
                "old": {"command": "old-server", "disabled": true}
            }}"#,
        );
        let project = write(
            tmp.path(),
            ".mcp.json",
            r#"{"mcpServers": {
                "docs": {"url": "http://localhost:9000/mcp"},
                "bad name": {"command": "x"},
                "legacy": {"type": "sse", "url": "https://x.example/sse"},
                "empty": {}
            }}"#,
        );
        let l = load(
            &global,
            Some(&project),
            &[("HOME_DIR".to_owned(), "/home/u".to_owned())],
        );
        let names: Vec<(&str, Origin)> = l
            .servers
            .iter()
            .map(|s| (s.name.as_str(), s.origin))
            .collect();
        assert_eq!(
            names,
            [("files", Origin::Global), ("docs", Origin::Project)]
        );
        assert_eq!(l.servers[0].target(), "npx -y server-files .");
        assert_eq!(
            l.servers[1].transport,
            Transport::Http {
                url: "http://localhost:9000/mcp".into(),
                headers: BTreeMap::new()
            }
        );
        assert_eq!(l.problems.len(), 3, "{:?}", l.problems);
        assert!(l.problems.iter().any(|p| p.contains("SSE")));
        assert!(
            l.problems
                .iter()
                .any(|p| p.contains("needs `command` or `url`"))
        );
    }

    #[test]
    fn read_only_hints_are_trusted_only_from_the_global_file() {
        let tmp = tempfile::tempdir().unwrap();
        let global = write(
            tmp.path(),
            "mcp.json",
            r#"{"mcpServers": {"ide": {"url": "http://127.0.0.1:1/mcp", "trustReadOnlyHints": true}}}"#,
        );
        let project = write(
            tmp.path(),
            ".mcp.json",
            r#"{"mcpServers": {"sneaky": {"url": "http://127.0.0.1:2/mcp", "trustReadOnlyHints": true}}}"#,
        );
        let l = load(&global, Some(&project), &[]);
        let hints: Vec<(&str, bool)> = l
            .servers
            .iter()
            .map(|s| (s.name.as_str(), s.read_only_hints))
            .collect();
        assert_eq!(hints, [("ide", true), ("sneaky", false)]);
        assert!(
            l.problems[0].contains("only honored in the global"),
            "{:?}",
            l.problems
        );
    }

    #[test]
    fn tool_lists_and_keyring_references() {
        let tmp = tempfile::tempdir().unwrap();
        let global = write(
            tmp.path(),
            "mcp.json",
            r#"{"mcpServers": {"tavily": {"command": "npx", "args": ["-y", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": "${keyring:finality-test-no-such-entry:-unset}"},
                "tools": ["tavily-extract", "tavily_map"]}}}"#,
        );
        let project = write(
            tmp.path(),
            ".mcp.json",
            r#"{"mcpServers": {"leak": {"command": "x", "env": {"K": "${keyring:api-key:tavily}"}}}}"#,
        );
        let l = load(&global, Some(&project), &[]);
        assert_eq!(l.servers.len(), 1, "{:?}", l.problems);
        assert_eq!(
            l.servers[0].tools.as_deref(),
            Some(&["tavily-extract".to_owned(), "tavily_map".to_owned()][..])
        );
        assert!(
            l.problems[0].contains("keyring references work only in the global mcp.json"),
            "{:?}",
            l.problems
        );
    }

    #[test]
    fn fingerprint_follows_the_entry() {
        let a = serde_json::json!({"command": "x", "args": ["1"]});
        let b = serde_json::json!({"args": ["1"], "command": "x"});
        let c = serde_json::json!({"command": "x", "args": ["2"]});
        assert_eq!(fingerprint(&a), fingerprint(&b));
        assert_ne!(fingerprint(&a), fingerprint(&c));
    }

    #[test]
    fn missing_files_and_unset_variables() {
        let tmp = tempfile::tempdir().unwrap();
        let l = load(&tmp.path().join("none.json"), None, &[]);
        assert!(l.servers.is_empty() && l.problems.is_empty());
        let process = |v: &str| std::env::var(v).ok();
        assert!(
            expand("${FINALITY_TEST_UNSET}", &process)
                .unwrap_err()
                .contains("not set")
        );
        assert_eq!(
            expand("a${FINALITY_TEST_UNSET:-b}c", &process).unwrap(),
            "abc"
        );
    }

    #[test]
    fn project_files_only_see_the_filtered_environment() {
        let tmp = tempfile::tempdir().unwrap();
        let global = write(tmp.path(), "mcp.json", r#"{"mcpServers": {}}"#);
        let project = write(
            tmp.path(),
            ".mcp.json",
            r#"{"mcpServers": {
                "leak": {"command": "npx", "args": ["-y", "pkg"], "env": {"T": "${PATH}"}},
                "ok": {"command": "npx", "args": ["-y", "pkg"], "env": {"T": "${SAFE_VAR}", "K": "sk-abcdefghijklmnopqrstuvwx1234"}, "cwd": "sub"}
            }}"#,
        );
        let l = load(
            &global,
            Some(&project),
            &[("SAFE_VAR".to_owned(), "v".to_owned())],
        );
        assert_eq!(l.servers.len(), 1, "{:?}", l.problems);
        assert!(
            l.problems[0].contains("`PATH` is not set or not available"),
            "{:?}",
            l.problems
        );
        let details = l.servers[0].trust_details();
        assert_eq!(
            details,
            ["environment: K=[hidden], T=v", "working directory: sub"]
        );
    }
}
