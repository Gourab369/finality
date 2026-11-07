#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use finality_core::config::SafetyConfig;
use finality_core::{AgentId, Mode};
use finality_mcp::{HubSettings, McpHub, MemoryVault, ServerState, TokenVault, Trust};
use finality_safety::{PathGuard, PermissionEngine};
use finality_store::McpTrust;
use finality_tools::{
    ActiveToolSet, ApprovalRequest, ApprovalResponse, Approver, Tool, ToolCtx, ToolEnv,
    ToolExecutor, ToolLimits, ToolRegistry, ToolSearch, ToolSets, ToolSource,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const ECHO: &str = env!("CARGO_BIN_EXE_finality_mcp_echo");

struct Fx {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    work: PathBuf,
    marker: PathBuf,
    vault: Arc<MemoryVault>,
}

fn fx() -> Fx {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let marker = root.join("started.txt");
    Fx {
        _tmp: tmp,
        root,
        work,
        marker,
        vault: Arc::new(MemoryVault::default()),
    }
}

impl Fx {
    fn echo_entry(&self) -> Value {
        json!({ "command": ECHO, "env": { "FINALITY_MCP_ECHO_MARKER": self.marker.display().to_string() } })
    }

    fn global(&self, servers: Value) {
        std::fs::write(
            self.root.join("mcp.json"),
            json!({ "mcpServers": servers }).to_string(),
        )
        .unwrap();
    }

    fn project(&self, servers: Value) {
        std::fs::write(
            self.work.join(".mcp.json"),
            json!({ "mcpServers": servers }).to_string(),
        )
        .unwrap();
    }

    fn hub(&self) -> Arc<McpHub> {
        McpHub::new(HubSettings {
            global_file: self.root.join("mcp.json"),
            project_file: Some(self.work.join(".mcp.json")),
            cache_file: self.root.join("mcp/tools-cache.json"),
            workdir: self.work.clone(),
            env: std::env::vars().collect(),
            vault: self.vault.clone(),
            start_timeout: Duration::from_secs(20),
            call_timeout: Duration::from_secs(20),
            ide_dir: Some(self.root.join("ide")),
        })
    }

    fn starts(&self) -> usize {
        std::fs::read_to_string(&self.marker)
            .map(|s| s.lines().count())
            .unwrap_or(0)
    }

    fn ctx(&self) -> ToolCtx {
        let safety = SafetyConfig::default();
        ToolCtx {
            env: Arc::new(ToolEnv::new(
                PathGuard::new(&self.work).unwrap(),
                [],
                &safety.env_denylist,
            )),
            limits: ToolLimits::default(),
            cancel: CancellationToken::new(),
            agent: AgentId::new(),
        }
    }
}

fn names(tools: &[Arc<dyn Tool>]) -> Vec<String> {
    let mut n: Vec<String> = tools.iter().map(|t| t.name().to_owned()).collect();
    n.sort();
    n
}

#[derive(Default)]
struct Scripted {
    responses: Mutex<VecDeque<ApprovalResponse>>,
    seen: Mutex<Vec<ApprovalRequest>>,
}

#[async_trait]
impl Approver for Scripted {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse {
        self.seen.lock().unwrap().push(req);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ApprovalResponse::Deny { note: None })
    }
}

#[tokio::test]
async fn servers_start_lazily_and_tool_lists_are_cached() {
    let f = fx();
    f.global(json!({ "echo": f.echo_entry() }));
    let hub = f.hub();
    assert!(
        hub.tools().is_empty() && f.starts() == 0,
        "nothing before enable"
    );
    assert!(hub.enable(&BTreeMap::new()).is_empty());
    assert!(hub.tools().is_empty());
    assert_eq!(hub.undiscovered(), ["echo"]);
    assert_eq!(f.starts(), 0, "enabling starts nothing");

    assert!(hub.discover(&CancellationToken::new()).await.is_empty());
    assert_eq!(f.starts(), 1);
    assert_eq!(
        names(&hub.tools()),
        ["mcp__echo__add", "mcp__echo__echo", "mcp__echo__fail"]
    );
    assert!(hub.undiscovered().is_empty());
    assert_eq!(hub.running(), 1);
    hub.shutdown().await;
    assert_eq!(hub.running(), 0);

    // A later run knows the tools from the cache and starts the server only for a call.
    let again = f.hub();
    again.enable(&BTreeMap::new());
    assert_eq!(names(&again.tools()).len(), 3);
    assert_eq!(f.starts(), 1);
    let tool = again
        .tools()
        .into_iter()
        .find(|t| t.name() == "mcp__echo__echo")
        .unwrap();
    assert_eq!(tool.risk(), finality_core::RiskClass::External);
    assert_eq!(tool.tier(), finality_tools::Tier::Deferred);
    assert!(
        tool.description()
            .starts_with("[MCP server echo] Echo the given text")
    );
    let ctx = f.ctx();
    let args = json!({ "text": "hi" });
    let preview = tool.preview(&args, &ctx).await.unwrap();
    assert_eq!(f.starts(), 1, "previews do not start servers");
    let out = tool.call(args, &ctx, &preview).await.unwrap();
    assert_eq!(out.content, "echo: hi");
    assert_eq!(f.starts(), 2);
    again.shutdown().await;
}

#[tokio::test]
async fn tool_search_discovers_and_calls_need_approval() {
    let f = fx();
    f.global(json!({ "echo": f.echo_entry() }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());

    let sets = ToolSets::new();
    let mut registry = ToolRegistry::with_builtins();
    registry.add_source(hub.clone());
    registry.register(Arc::new(
        ToolSearch::new(sets.clone()).with_sources(vec![hub.clone()]),
    ));
    let ctx = f.ctx();
    let set = Arc::new(Mutex::new(
        ActiveToolSet::new(
            registry.specs(|t| t.tier() == finality_tools::Tier::Deferred),
            3,
        )
        .open(),
    ));
    sets.register(ctx.agent, set.clone());
    let approver = Arc::new(Scripted::default());
    let exec = ToolExecutor::new(
        Arc::new(registry),
        Arc::new(Mutex::new(PermissionEngine::new(
            Mode::Cowork,
            &SafetyConfig::default().always_ask_patterns,
        ))),
        approver.clone(),
    );

    let found = exec
        .execute(
            "tool_search",
            json!({ "query": "echo text back" }),
            &ctx,
            None,
        )
        .await;
    assert!(
        found
            .output
            .content
            .contains("- mcp__echo__echo: [MCP server echo]"),
        "{}",
        found.output.content
    );
    assert!(set.lock().unwrap().is_loaded("mcp__echo__echo"));
    assert_eq!(f.starts(), 1, "discovery started the server");

    approver
        .responses
        .lock()
        .unwrap()
        .push_back(ApprovalResponse::Deny { note: None });
    let denied = exec
        .execute("mcp__echo__echo", json!({ "text": "no" }), &ctx, None)
        .await;
    assert!(denied.output.is_error);
    let req = approver.seen.lock().unwrap()[0].clone();
    assert_eq!(req.summary, "call `echo` on MCP server `echo`");
    assert!(req.detail.unwrap().contains("\"text\": \"no\""));

    approver
        .responses
        .lock()
        .unwrap()
        .extend([ApprovalResponse::AllowOnce, ApprovalResponse::AllowOnce]);
    let ok = exec
        .execute("mcp__echo__add", json!({ "a": 2, "b": 3.5 }), &ctx, None)
        .await;
    assert_eq!(ok.output.content, "5.5");
    assert!(!ok.output.is_error);
    let failed = exec.execute("mcp__echo__fail", json!({}), &ctx, None).await;
    assert!(failed.output.is_error && failed.output.content == "this tool always fails");
    hub.shutdown().await;
}

#[tokio::test]
async fn project_servers_need_trust() {
    let f = fx();
    let entry = f.echo_entry();
    f.project(json!({ "local": entry }));
    let fingerprint = finality_mcp::config::fingerprint(&entry);

    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    let pending = hub.pending_trust();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].name, "local");
    assert_eq!(pending[0].fingerprint, fingerprint);
    assert!(pending[0].target().starts_with(ECHO));
    assert!(hub.undiscovered().is_empty());
    assert!(hub.discover(&CancellationToken::new()).await.is_empty());
    assert_eq!(f.starts(), 0, "untrusted servers never start");

    hub.set_trust("local", true);
    hub.discover(&CancellationToken::new()).await;
    assert_eq!(hub.tools().len(), 3);
    hub.shutdown().await;

    let decided = |trusted, fingerprint: &str| {
        BTreeMap::from([(
            "local".to_owned(),
            McpTrust {
                fingerprint: fingerprint.to_owned(),
                trusted,
            },
        )])
    };
    let denied = f.hub();
    denied.enable(&decided(false, &fingerprint));
    assert_eq!(denied.status().await[0].trust, Trust::Denied);
    assert!(
        denied.tools().is_empty(),
        "denied servers hide their cached tools"
    );
    let changed = f.hub();
    changed.enable(&decided(true, "0000000000000000"));
    assert_eq!(
        changed.status().await[0].trust,
        Trust::Pending,
        "a changed entry asks again"
    );
}

#[tokio::test]
async fn tool_lists_limit_what_a_server_offers() {
    let f = fx();
    let mut entry = f.echo_entry();
    entry["tools"] = json!(["ECHO", "add", "missing"]);
    f.global(json!({ "echo": entry }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    assert!(hub.discover(&CancellationToken::new()).await.is_empty());
    assert_eq!(
        names(&hub.tools()),
        ["mcp__echo__add", "mcp__echo__echo"],
        "`fail` is not listed"
    );
    hub.shutdown().await;
}

#[tokio::test]
async fn broken_servers_are_reported_and_not_retried() {
    let f = fx();
    f.global(json!({
        "missing": { "command": "finality-no-such-mcp-server" },
        "echo": f.echo_entry()
    }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    let problems = hub.discover(&CancellationToken::new()).await;
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("MCP server `missing`: cannot run"),
        "{}",
        problems[0]
    );
    assert_eq!(hub.tools().len(), 3, "the working server is unaffected");
    let status = hub.status().await;
    let missing = status.iter().find(|s| s.name == "missing").unwrap();
    assert!(matches!(&missing.state, ServerState::Failed(m) if m.contains("cannot run")));
    assert!(
        hub.undiscovered().is_empty(),
        "failed servers wait for /mcp enable"
    );
    hub.enable(&BTreeMap::new());
    assert_eq!(hub.undiscovered(), ["missing"]);
    hub.disable().await;
    assert!(hub.tools().is_empty());
    assert_eq!(hub.running(), 0);
}

/// Streamable HTTP MCP endpoint answering JSON. Optionally requires a bearer token.
struct HttpMcp {
    token: Option<&'static str>,
    /// `lookup` declares `readOnlyHint`.
    annotated: bool,
}

impl Respond for HttpMcp {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        if let Some(t) = self.token {
            let auth = req
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if auth != format!("Bearer {t}") {
                return ResponseTemplate::new(401).insert_header("www-authenticate", "Bearer");
            }
        }
        if req.method.as_str() != "POST" {
            return ResponseTemplate::new(405);
        }
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let Some(id) = body.get("id").cloned() else {
            return ResponseTemplate::new(202);
        };
        let team = req
            .headers
            .get("x-team")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("none")
            .to_owned();
        let result = match body["method"].as_str().unwrap_or("") {
            "initialize" => json!({
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "docs", "version": "1" }
            }),
            "tools/list" if self.annotated => json!({ "tools": [
                { "name": "lookup", "description": "Look up docs", "inputSchema": { "type": "object" },
                  "annotations": { "readOnlyHint": true } },
                { "name": "publish", "description": "Publish docs", "inputSchema": { "type": "object" } }
            ] }),
            "tools/list" => json!({ "tools": [
                { "name": "lookup", "description": "Look up docs", "inputSchema": { "type": "object" } }
            ] }),
            "tools/call" => {
                json!({ "content": [ { "type": "text", "text": format!("docs for team {team}") } ] })
            }
            _ => json!({}),
        };
        ResponseTemplate::new(200)
            .set_body_json(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
    }
}

#[tokio::test]
async fn editor_bridge_is_found_by_its_lock_file_and_read_only_tools_are_reads() {
    let server = MockServer::start().await;
    Mock::given(path("/mcp"))
        .respond_with(HttpMcp {
            token: Some("ide-token"),
            annotated: true,
        })
        .mount(&server)
        .await;
    let f = fx();
    let ide = f.root.join("ide");
    std::fs::create_dir_all(&ide).unwrap();
    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    assert_eq!(hub.undiscovered(), ["vscode"]);
    let problems = hub.discover(&CancellationToken::new()).await;
    assert!(problems[0].contains("no VS Code window"), "{problems:?}");
    assert_eq!(
        hub.undiscovered(),
        ["vscode"],
        "retried once the editor opens"
    );

    let port = server.address().port();
    std::fs::write(
        ide.join(format!("{port}.json")),
        json!({ "port": port, "token": "ide-token", "workspaceFolders": [f.root.display().to_string()] })
            .to_string(),
    )
    .unwrap();
    assert!(hub.discover(&CancellationToken::new()).await.is_empty());
    let tools = hub.tools();
    let risk = |name: &str| tools.iter().find(|t| t.name() == name).unwrap().risk();
    assert_eq!(risk("mcp__vscode__lookup"), finality_core::RiskClass::Read);
    assert_eq!(
        risk("mcp__vscode__publish"),
        finality_core::RiskClass::External
    );
    let status = hub.status().await;
    assert_eq!(status[0].target, "VS Code (Finality extension)");
    assert!(!status[0].http, "no login for the editor");
    hub.shutdown().await;
}

#[tokio::test]
async fn http_servers_send_configured_headers() {
    let server = MockServer::start().await;
    Mock::given(path("/mcp"))
        .respond_with(HttpMcp {
            token: None,
            annotated: false,
        })
        .mount(&server)
        .await;
    let f = fx();
    f.global(json!({ "docs": { "type": "http", "url": format!("{}/mcp", server.uri()), "headers": { "X-Team": "core" } } }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    assert!(hub.discover(&CancellationToken::new()).await.is_empty());
    let tool = hub.tools().pop().unwrap();
    assert_eq!(tool.name(), "mcp__docs__lookup");
    let ctx = f.ctx();
    let preview = tool.preview(&json!({}), &ctx).await.unwrap();
    let out = tool.call(json!({}), &ctx, &preview).await.unwrap();
    assert_eq!(out.content, "docs for team core");
    hub.shutdown().await;
}

async fn mount_oauth(server: &MockServer) {
    let base = server.uri();
    Mock::given(method("GET"))
        .and(path("/.well-known/oauth-authorization-server"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "issuer": base,
            "authorization_endpoint": format!("{base}/authorize"),
            "token_endpoint": format!("{base}/token"),
            "registration_endpoint": format!("{base}/register"),
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"]
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/register"))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "client_id": "client-123",
            "redirect_uris": ["http://127.0.0.1/callback"]
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "secret-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh-1"
        })))
        .mount(server)
        .await;
}

/// Plays the browser: follows the authorization URL's redirect_uri with a code and the state.
async fn consent(auth_url: String) {
    let url = url::Url::parse(&auth_url).unwrap();
    let q: BTreeMap<String, String> = url.query_pairs().into_owned().collect();
    assert_eq!(q["code_challenge_method"], "S256");
    assert_eq!(q["client_id"], "client-123");
    let callback = format!("{}?code=abc&state={}", q["redirect_uri"], q["state"]);
    let page = reqwest::get(callback).await.unwrap().text().await.unwrap();
    assert!(page.contains("Login complete"));
}

#[tokio::test]
async fn login_stores_tokens_and_reconnects() {
    let server = MockServer::start().await;
    mount_oauth(&server).await;
    Mock::given(path("/mcp"))
        .respond_with(HttpMcp {
            token: Some("secret-token"),
            annotated: false,
        })
        .mount(&server)
        .await;
    let f = fx();
    f.global(json!({ "docs": { "type": "http", "url": format!("{}/mcp", server.uri()) } }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());

    let problems = hub.discover(&CancellationToken::new()).await;
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("/mcp login docs"), "{}", problems[0]);
    assert!(!hub.status().await[0].logged_in);

    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    let browser = tokio::spawn(async move { consent(rx.await.unwrap()).await });
    let tools = hub
        .login(
            "docs",
            move |url| {
                let _ = tx.send(url.to_owned());
            },
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    browser.await.unwrap();
    assert_eq!(tools, 1);
    let status = &hub.status().await[0];
    assert!(status.logged_in);
    assert_eq!(status.state, ServerState::Running);
    let stored = f.vault.load("docs").unwrap().unwrap();
    assert!(stored.contains("secret-token") && stored.contains("client-123"));

    // A new hub (next run) uses the stored token without logging in again.
    hub.shutdown().await;
    let again = f.hub();
    again.enable(&BTreeMap::new());
    let tool = again.tools().pop().unwrap();
    let ctx = f.ctx();
    let preview = tool.preview(&json!({}), &ctx).await.unwrap();
    let out = tool.call(json!({}), &ctx, &preview).await.unwrap();
    assert_eq!(out.content, "docs for team none");
    again.shutdown().await;
}

#[test]
fn stdio_login_is_refused() {
    let f = fx();
    f.global(json!({ "echo": f.echo_entry() }));
    let hub = f.hub();
    hub.enable(&BTreeMap::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let err = rt
        .block_on(hub.login("echo", |_| {}, &CancellationToken::new()))
        .unwrap_err();
    assert!(err.contains("only HTTP servers"), "{err}");
    let err = rt
        .block_on(hub.login("nope", |_| {}, &CancellationToken::new()))
        .unwrap_err();
    assert!(err.contains("no MCP server named `nope`"));
}
