//! Lazily started MCP clients and their tools (DESIGN §9.4).

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use finality_store::{McpTrust, write_atomic};
use finality_tools::{Tool, ToolSource};
use process_wrap::tokio::{CommandWrap, KillOnDrop};
use rmcp::model::{CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{Peer, RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio_util::sync::CancellationToken;

use crate::auth::{self, TokenVault};
use crate::config::{self, Origin, ServerDef, Transport};
use crate::tool::McpTool;

const STDERR_LINES: usize = 20;

pub struct HubSettings {
    pub global_file: PathBuf,
    pub project_file: Option<PathBuf>,
    /// Tool lists from earlier runs, so servers need not start to be searchable.
    pub cache_file: PathBuf,
    pub workdir: PathBuf,
    /// Environment for stdio servers (already filtered by `safety.env_denylist`).
    pub env: Vec<(String, String)>,
    pub vault: Arc<dyn TokenVault>,
    pub start_timeout: Duration,
    /// Bound on one `tools/call`; a server that stalls returns an error instead of holding the agent.
    pub call_timeout: Duration,
    /// Lock files of the Finality editor extension; `None` turns the editor server off.
    pub ide_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    Trusted,
    /// A project server the user has not decided on (or whose entry changed).
    Pending,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerState {
    Stopped,
    Running,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerStatus {
    pub name: String,
    pub origin: Origin,
    pub target: String,
    pub http: bool,
    pub trust: Trust,
    pub state: ServerState,
    /// `None` until the server has been started once (or its cached list is stale).
    pub tools: Option<usize>,
    pub logged_in: bool,
}

struct Conn {
    service: RunningService<RoleClient, ClientConfig>,
    peer: Peer<RoleClient>,
}

pub struct Server {
    pub def: ServerDef,
    settings: Arc<HubSettings>,
    cache_lock: Arc<Mutex<()>>,
    trust: Mutex<Trust>,
    state: Mutex<ServerState>,
    conn: tokio::sync::Mutex<Option<Conn>>,
    tools: RwLock<Option<Vec<Arc<dyn Tool>>>>,
    stderr: Arc<Mutex<VecDeque<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub annotations: Value,
}

#[derive(Default, Serialize, Deserialize)]
struct CacheFile {
    #[serde(default)]
    servers: BTreeMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    fingerprint: String,
    tools: Vec<CachedTool>,
}

fn read_cache(path: &Path) -> CacheFile {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn client_info() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new(finality_core::APP_NAME, env!("CARGO_PKG_VERSION")),
    )
}

/// The error and its sources on one line.
fn chain(e: &dyn std::error::Error) -> String {
    let mut text = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        let s_text = s.to_string();
        if !text.contains(&s_text) {
            text.push_str(": ");
            text.push_str(&s_text);
        }
        source = s.source();
    }
    text
}

impl Server {
    fn new(
        def: ServerDef,
        settings: Arc<HubSettings>,
        cache_lock: Arc<Mutex<()>>,
        trust: Trust,
    ) -> Arc<Self> {
        Arc::new(Self {
            def,
            settings,
            cache_lock,
            trust: Mutex::new(trust),
            state: Mutex::new(ServerState::Stopped),
            conn: tokio::sync::Mutex::new(None),
            tools: RwLock::new(None),
            stderr: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    pub fn name(&self) -> &str {
        &self.def.name
    }

    pub fn trust(&self) -> Trust {
        self.trust.lock().map_or(Trust::Denied, |t| *t)
    }

    pub fn state(&self) -> ServerState {
        self.state
            .lock()
            .map_or(ServerState::Stopped, |s| s.clone())
    }

    fn set_state(&self, state: ServerState) {
        if let Ok(mut s) = self.state.lock() {
            *s = state;
        }
    }

    pub fn tools(&self) -> Option<Vec<Arc<dyn Tool>>> {
        self.tools.read().ok().and_then(|t| t.clone())
    }

    /// Failed servers wait for `/mcp enable`; the editor may simply not have been open yet.
    fn retryable(&self) -> bool {
        self.def.origin == Origin::Ide || !matches!(self.state(), ServerState::Failed(_))
    }

    fn set_tools(self: &Arc<Self>, tools: &[CachedTool]) {
        let wanted = |name: &str| {
            let norm = |s: &str| s.to_ascii_lowercase().replace('-', "_");
            self.def
                .tools
                .as_ref()
                .is_none_or(|list| list.iter().any(|w| norm(w) == norm(name)))
        };
        let tools: Vec<&CachedTool> = tools.iter().filter(|t| wanted(&t.name)).collect();
        if let Some(list) = &self.def.tools
            && tools.len() < list.len()
        {
            tracing::warn!(server = %self.def.name, wanted = ?list, "some tools listed in mcp.json do not exist");
        }
        let remote: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        let names = crate::tool::unique_tool_names(self.name(), &remote);
        let adapted: Vec<Arc<dyn Tool>> = tools
            .into_iter()
            .zip(names)
            .map(|(t, name)| Arc::new(McpTool::new(self, t, name)) as Arc<dyn Tool>)
            .collect();
        if let Ok(mut slot) = self.tools.write() {
            *slot = Some(adapted);
        }
    }

    fn store_cache(&self, tools: Vec<CachedTool>) {
        let Ok(_guard) = self.cache_lock.lock() else {
            return;
        };
        let path = &self.settings.cache_file;
        let mut cache = read_cache(path);
        cache.servers.insert(
            self.def.name.clone(),
            CacheEntry {
                fingerprint: self.def.fingerprint.clone(),
                tools,
            },
        );
        match serde_json::to_vec_pretty(&cache) {
            Ok(bytes) => {
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                if let Err(e) = write_atomic(path, &bytes) {
                    tracing::warn!("cannot write the MCP tool cache: {e}");
                }
            }
            Err(e) => tracing::warn!("cannot encode the MCP tool cache: {e}"),
        }
    }

    pub async fn is_running(&self) -> bool {
        self.conn
            .lock()
            .await
            .as_ref()
            .is_some_and(|c| !c.peer.is_transport_closed())
    }

    /// The live connection, starting the server first if needed. Starting refreshes the tool list.
    pub async fn connect(self: &Arc<Self>) -> Result<Peer<RoleClient>, String> {
        if self.trust() != Trust::Trusted {
            return Err(format!(
                "MCP server `{}` is not trusted for this project",
                self.def.name
            ));
        }
        let mut conn = self.conn.lock().await;
        if let Some(c) = conn.as_ref() {
            if !c.peer.is_transport_closed() {
                return Ok(c.peer.clone());
            }
            tracing::warn!(server = %self.def.name, "MCP server connection closed; restarting");
        }
        *conn = None;
        let started = tokio::time::timeout(self.settings.start_timeout, self.start()).await;
        let service = match started {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(self.fail(e)),
            Err(_) => {
                return Err(self.fail(format!(
                    "did not start within {} s",
                    self.settings.start_timeout.as_secs()
                )));
            }
        };
        let peer = service.peer().clone();
        match peer.list_all_tools().await {
            Ok(list) => {
                let tools: Vec<CachedTool> = list
                    .iter()
                    .filter_map(|t| serde_json::to_value(t).ok())
                    .filter_map(|v| serde_json::from_value(v).ok())
                    .collect();
                self.set_tools(&tools);
                self.store_cache(tools);
            }
            Err(e) => {
                let _ = service.cancel().await;
                return Err(self.fail(format!("cannot list tools: {}", chain(&e))));
            }
        }
        *conn = Some(Conn {
            service,
            peer: peer.clone(),
        });
        self.set_state(ServerState::Running);
        Ok(peer)
    }

    fn fail(&self, error: String) -> String {
        let mut message = format!("MCP server `{}`: {error}", self.def.name);
        if matches!(self.def.transport, Transport::Http { .. })
            && self.def.origin != Origin::Ide
            && error.to_lowercase().contains("auth")
        {
            message.push_str(&format!("; run /mcp login {}", self.def.name));
        }
        let tail: Vec<String> = self
            .stderr
            .lock()
            .map(|s| s.iter().rev().take(5).rev().cloned().collect())
            .unwrap_or_default();
        if !tail.is_empty() {
            message.push_str(&format!("\nserver stderr:\n{}", tail.join("\n")));
        }
        self.set_state(ServerState::Failed(message.clone()));
        message
    }

    async fn start(&self) -> Result<RunningService<RoleClient, ClientConfig>, String> {
        match &self.def.transport {
            Transport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let path_var = self
                    .settings
                    .env
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
                    .map(|(_, v)| v.clone());
                let (program, prefix) = resolve_program(command, path_var.as_deref());
                let workdir = match cwd {
                    Some(c) => self.settings.workdir.join(c),
                    None => self.settings.workdir.clone(),
                };
                let base_env = self.settings.env.clone();
                let env = env.clone();
                let args = args.clone();
                let mut cmd = CommandWrap::with_new(&program, move |c| {
                    c.args(&prefix)
                        .args(&args)
                        .current_dir(&workdir)
                        .env_clear()
                        .envs(base_env)
                        .envs(env);
                });
                #[cfg(unix)]
                cmd.wrap(process_wrap::tokio::ProcessGroup::leader());
                #[cfg(windows)]
                cmd.wrap(process_wrap::tokio::JobObject);
                cmd.wrap(KillOnDrop);
                let (transport, stderr) = TokioChildProcess::builder(cmd)
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("cannot run `{command}`: {e}"))?;
                if let Some(stderr) = stderr {
                    let (tail, name) = (self.stderr.clone(), self.def.name.clone());
                    tokio::spawn(async move {
                        let mut lines = tokio::io::BufReader::new(stderr).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            tracing::debug!(server = %name, "{line}");
                            if let Ok(mut t) = tail.lock() {
                                if t.len() == STDERR_LINES {
                                    t.pop_front();
                                }
                                t.push_back(line);
                            }
                        }
                    });
                }
                client_info()
                    .serve(transport)
                    .await
                    .map_err(|e| format!("initialization failed: {}", chain(&e)))
            }
            Transport::Http { url, headers } => {
                let ide = self.def.origin == Origin::Ide;
                let (url, headers) = match &self.settings.ide_dir {
                    Some(dir) if ide => {
                        let (url, token) = crate::ide::locate(dir, &self.settings.workdir)?;
                        let mut headers = BTreeMap::from([(
                            "Authorization".to_owned(),
                            format!("Bearer {token}"),
                        )]);
                        // Relative paths and the read-only scope follow Finality's working directory.
                        if let Ok(uri) = url::Url::from_file_path(&self.settings.workdir) {
                            headers.insert(crate::ide::WORKDIR_HEADER.to_owned(), uri.to_string());
                        }
                        (url, headers)
                    }
                    _ => (url.clone(), headers.clone()),
                };
                let mut config = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                let mut custom = std::collections::HashMap::new();
                for (k, v) in &headers {
                    let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                        .map_err(|e| format!("bad header name `{k}`: {e}"))?;
                    let value = reqwest::header::HeaderValue::from_str(v)
                        .map_err(|e| format!("bad value for header `{k}`: {e}"))?;
                    custom.insert(name, value);
                }
                config = config.custom_headers(custom);
                if !ide
                    && let Some(token) =
                        auth::access_token(&url, &self.def.name, &self.settings.vault).await?
                {
                    config = config.auth_header(token);
                }
                client_info()
                    .serve(StreamableHttpClientTransport::from_config(config))
                    .await
                    .map_err(|e| format!("cannot connect to {url}: {}", chain(&e)))
            }
        }
    }

    /// Calls a tool on the server and returns the raw `CallToolResult` JSON.
    pub async fn call(
        self: &Arc<Self>,
        tool: &str,
        args: Value,
        cancel: &CancellationToken,
    ) -> Result<Value, String> {
        let arguments = match args {
            Value::Object(m) => m,
            Value::Null => serde_json::Map::new(),
            other => return Err(format!("arguments must be an object, got {other}")),
        };
        let peer = tokio::select! {
            p = self.connect() => p?,
            () = cancel.cancelled() => return Err("cancelled".into()),
        };
        let request = CallToolRequestParams::new(tool.to_owned()).with_arguments(arguments);
        let result = tokio::select! {
            r = peer.call_tool(request) => r,
            () = cancel.cancelled() => return Err("cancelled".into()),
            () = tokio::time::sleep(self.settings.call_timeout) => {
                return Err(format!(
                    "MCP server `{}` did not answer `{tool}` within {}s",
                    self.def.name,
                    self.settings.call_timeout.as_secs()
                ));
            }
        };
        match result {
            Ok(r) => serde_json::to_value(r).map_err(|e| e.to_string()),
            Err(e) => {
                if peer.is_transport_closed() {
                    self.set_state(ServerState::Failed("connection closed".into()));
                }
                Err(format!("MCP server `{}`: {}", self.def.name, chain(&e)))
            }
        }
    }

    async fn stop(&self) {
        if let Some(c) = self.conn.lock().await.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), c.service.cancel()).await;
        }
        if matches!(self.state(), ServerState::Running) {
            self.set_state(ServerState::Stopped);
        }
    }
}

/// On Windows, finds `npx` as `npx.cmd` on PATH and runs batch files through `cmd /C`.
fn resolve_program(command: &str, path_var: Option<&str>) -> (PathBuf, Vec<String>) {
    #[cfg(windows)]
    {
        let p = Path::new(command);
        let found = if p.extension().is_some() || p.components().count() > 1 {
            Some(p.to_path_buf())
        } else {
            let exts: Vec<String> = std::env::var("PATHEXT")
                .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into())
                .split(';')
                .filter(|e| !e.is_empty())
                .map(str::to_owned)
                .collect();
            path_var.and_then(|pv| {
                std::env::split_paths(pv).find_map(|dir| {
                    exts.iter()
                        .map(|e| dir.join(format!("{command}{e}")))
                        .find(|c| c.is_file())
                })
            })
        };
        if let Some(found) = found {
            let batch = found
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"));
            if batch {
                return (
                    PathBuf::from("cmd.exe"),
                    vec!["/D".into(), "/C".into(), found.display().to_string()],
                );
            }
            return (found, Vec::new());
        }
    }
    let _ = path_var;
    (PathBuf::from(command), Vec::new())
}

/// All configured MCP servers. Nothing is read until MCP is enabled, and no server starts
/// until its tools are needed.
pub struct McpHub {
    settings: Arc<HubSettings>,
    cache_lock: Arc<Mutex<()>>,
    servers: RwLock<Vec<Arc<Server>>>,
    loaded: AtomicBool,
    enabled: AtomicBool,
}

impl McpHub {
    pub fn new(settings: HubSettings) -> Arc<Self> {
        Arc::new(Self {
            settings: Arc::new(settings),
            cache_lock: Arc::new(Mutex::new(())),
            servers: RwLock::new(Vec::new()),
            loaded: AtomicBool::new(false),
            enabled: AtomicBool::new(false),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Reads the config files on first use and enables MCP. Failed servers get another try.
    /// `trust` is this project's stored decisions. Returns config problems.
    pub fn enable(&self, trust: &BTreeMap<String, McpTrust>) -> Vec<String> {
        let mut problems = Vec::new();
        if !self.loaded.swap(true, Ordering::SeqCst) {
            let loaded = config::load(
                &self.settings.global_file,
                self.settings.project_file.as_deref(),
                &self.settings.env,
            );
            problems = loaded.problems;
            let cache = read_cache(&self.settings.cache_file);
            let mut defs = loaded.servers;
            if let Some(def) = self
                .settings
                .ide_dir
                .as_deref()
                .and_then(crate::ide::server_def)
                && !defs.iter().any(|d| d.name == def.name)
            {
                defs.push(def);
            }
            let servers: Vec<Arc<Server>> = defs
                .into_iter()
                .map(|def| {
                    let decision = match def.origin {
                        Origin::Global | Origin::Ide => Trust::Trusted,
                        Origin::Project => match trust.get(&def.name) {
                            Some(t) if t.fingerprint == def.fingerprint => {
                                if t.trusted {
                                    Trust::Trusted
                                } else {
                                    Trust::Denied
                                }
                            }
                            _ => Trust::Pending,
                        },
                    };
                    let cached = cache
                        .servers
                        .get(&def.name)
                        .filter(|c| c.fingerprint == def.fingerprint)
                        .map(|c| c.tools.clone());
                    let server = Server::new(
                        def,
                        self.settings.clone(),
                        self.cache_lock.clone(),
                        decision,
                    );
                    if let Some(tools) = cached {
                        server.set_tools(&tools);
                    }
                    server
                })
                .collect();
            if let Ok(mut s) = self.servers.write() {
                *s = servers;
            }
        }
        for s in self.servers() {
            if matches!(s.state(), ServerState::Failed(_)) {
                s.set_state(ServerState::Stopped);
            }
        }
        self.enabled.store(true, Ordering::SeqCst);
        problems
    }

    /// Disables MCP and stops running servers.
    pub async fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        self.shutdown().await;
    }

    pub fn servers(&self) -> Vec<Arc<Server>> {
        self.servers.read().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn server(&self, name: &str) -> Option<Arc<Server>> {
        self.servers().into_iter().find(|s| s.def.name == name)
    }

    /// Project servers waiting for a trust decision.
    pub fn pending_trust(&self) -> Vec<ServerDef> {
        self.servers()
            .into_iter()
            .filter(|s| s.trust() == Trust::Pending)
            .map(|s| s.def.clone())
            .collect()
    }

    pub fn set_trust(&self, name: &str, trusted: bool) {
        if let Some(s) = self.server(name)
            && let Ok(mut t) = s.trust.lock()
        {
            *t = if trusted {
                Trust::Trusted
            } else {
                Trust::Denied
            };
        }
    }

    pub async fn status(&self) -> Vec<ServerStatus> {
        let mut out = Vec::new();
        for s in self.servers() {
            let running = s.is_running().await;
            let state = match s.state() {
                ServerState::Running if !running => ServerState::Stopped,
                other => other,
            };
            let http =
                matches!(s.def.transport, Transport::Http { .. }) && s.def.origin != Origin::Ide;
            out.push(ServerStatus {
                name: s.def.name.clone(),
                origin: s.def.origin,
                target: s.def.target(),
                http,
                trust: s.trust(),
                state,
                tools: s.tools().map(|t| t.len()),
                logged_in: http
                    && self
                        .settings
                        .vault
                        .load(&s.def.name)
                        .ok()
                        .flatten()
                        .is_some(),
            });
        }
        out
    }

    /// Servers whose last start succeeded and that have not stopped or failed since.
    pub fn running(&self) -> usize {
        self.servers()
            .iter()
            .filter(|s| s.state() == ServerState::Running)
            .count()
    }

    pub async fn shutdown(&self) {
        futures::future::join_all(self.servers().iter().map(|s| s.stop())).await;
    }

    /// `/mcp login <server>`: OAuth for an HTTP server, then reconnects. Returns its tool count.
    pub async fn login(
        &self,
        name: &str,
        show_url: impl FnOnce(&str),
        cancel: &CancellationToken,
    ) -> Result<usize, String> {
        let server = self
            .server(name)
            .ok_or_else(|| format!("no MCP server named `{name}`"))?;
        let Transport::Http { url, .. } = &server.def.transport else {
            return Err(format!(
                "`{name}` is a stdio server; only HTTP servers log in"
            ));
        };
        if server.def.origin == Origin::Ide {
            return Err(format!(
                "`{name}` is the editor extension; it needs no login"
            ));
        }
        auth::login(url, name, &self.settings.vault, show_url, cancel).await?;
        server.stop().await;
        server.set_state(ServerState::Stopped);
        server.connect().await?;
        Ok(server.tools().map_or(0, |t| t.len()))
    }
}

#[async_trait]
impl ToolSource for McpHub {
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        if !self.is_enabled() {
            return Vec::new();
        }
        self.servers()
            .iter()
            .filter(|s| s.trust() == Trust::Trusted)
            .filter_map(|s| s.tools())
            .flatten()
            .collect()
    }

    async fn discover(&self, cancel: &CancellationToken) -> Vec<String> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let todo: Vec<Arc<Server>> = self
            .servers()
            .into_iter()
            .filter(|s| s.trust() == Trust::Trusted && s.tools().is_none() && s.retryable())
            .collect();
        let work = futures::future::join_all(todo.iter().map(|s| s.connect()));
        tokio::select! {
            results = work => results.into_iter().filter_map(Result::err).collect(),
            () = cancel.cancelled() => Vec::new(),
        }
    }

    fn undiscovered(&self) -> Vec<String> {
        if !self.is_enabled() {
            return Vec::new();
        }
        self.servers()
            .iter()
            .filter(|s| s.trust() == Trust::Trusted && s.tools().is_none() && s.retryable())
            .map(|s| s.def.name.clone())
            .collect()
    }
}
