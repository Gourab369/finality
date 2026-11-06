//! Phase 13 acceptance: MCP servers through the Core.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use h_agent_agents::{Core, CoreDeps};
use h_agent_core::config::Config;
use h_agent_core::events::{
    ApprovalChoice, CoreCommand, McpAction, NoticeLevel, SlashCommand, UiEvent,
};
use h_agent_core::{RiskClass, Workspace};
use h_agent_llm::{
    ChatChunk, ChatRequest, ChatStream, LlmError, LlmProvider, Message, MockProvider, ModelInfo,
    ToolCall, cancellable,
};
use h_agent_store::{DataDir, ProjectIndex};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::path;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const WAIT: Duration = Duration::from_secs(20);

/// Streamable HTTP MCP endpoint with one tool, `lookup`.
struct HttpMcp;

impl Respond for HttpMcp {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        if req.method.as_str() != "POST" {
            return ResponseTemplate::new(405);
        }
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let Some(id) = body.get("id").cloned() else {
            return ResponseTemplate::new(202);
        };
        let result = match body["method"].as_str().unwrap_or("") {
            "initialize" => json!({
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "docs", "version": "1" }
            }),
            "tools/list" => json!({ "tools": [{
                "name": "lookup",
                "description": "Look up internal documentation pages",
                "inputSchema": { "type": "object", "properties": { "page": { "type": "string" } } }
            }] }),
            "tools/call" => json!({ "content": [{
                "type": "text",
                "text": format!("page {}: deploy with make ship", body["params"]["arguments"]["page"].as_str().unwrap_or("?"))
            }] }),
            _ => json!({}),
        };
        ResponseTemplate::new(200)
            .set_body_json(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
    }
}

fn last_user(req: &ChatRequest) -> &str {
    req.messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .unwrap_or("")
}

fn turn_results(req: &ChatRequest) -> Vec<String> {
    let start = req
        .messages
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }))
        .unwrap_or(0);
    req.messages[start..]
        .iter()
        .filter_map(|m| match m {
            Message::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// Orchestrator: tool_search, then the MCP tool, then an answer.
#[derive(Default)]
struct Scripted {
    requests: Mutex<Vec<ChatRequest>>,
}

impl Scripted {
    fn orchestrator_requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.response_format.is_none())
            .cloned()
            .collect()
    }
}

#[async_trait]
impl LlmProvider for Scripted {
    fn id(&self) -> &'static str {
        "scripted"
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn model_info(&self, _: &str) -> Result<ModelInfo, LlmError> {
        Ok(ModelInfo {
            context_length: Some(32_768),
            ..ModelInfo::default()
        })
    }
    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        self.requests.lock().unwrap().push(req.clone());
        let n = req.messages.len();
        let call = |name: &str, arguments: Value| {
            ChatChunk::ToolCall(ToolCall {
                id: format!("call_{n}"),
                name: name.to_owned(),
                arguments,
            })
        };
        let chunk = if req.response_format.is_some() {
            ChatChunk::Text(
                json!({ "query_type": "question", "topic": "deploy", "complexity": "simple",
                        "needs_tools": true, "tool_hints": [] })
                .to_string(),
            )
        } else if !last_user(&req).contains("docs") {
            ChatChunk::Text("ok".into())
        } else {
            match turn_results(&req).len() {
                0 => call("tool_search", json!({ "query": "internal documentation" })),
                1 if turn_results(&req)[0].contains("mcp__docs__lookup") => {
                    call("mcp__docs__lookup", json!({ "page": "deploy" }))
                }
                _ => ChatChunk::Text("Done.".into()),
            }
        };
        let chunks = vec![
            Ok(chunk),
            Ok(ChatChunk::Done {
                usage: Default::default(),
                stop_reason: Default::default(),
            }),
        ];
        Ok(cancellable(Box::pin(futures::stream::iter(chunks)), cancel))
    }
}

struct Run {
    _tmp: tempfile::TempDir,
    work: PathBuf,
    data: DataDir,
    ui: mpsc::UnboundedReceiver<UiEvent>,
    cmd: mpsc::UnboundedSender<CoreCommand>,
    core: tokio::task::JoinHandle<()>,
    seen: Vec<UiEvent>,
    /// Answer for trust questions.
    trust: &'static str,
}

/// `global` goes to `<data>/mcp.json`, `project` to `<work>/.mcp.json`.
fn start(
    provider: Arc<dyn LlmProvider>,
    config: Config,
    global: Option<Value>,
    project: Option<Value>,
    trust: &'static str,
) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let data = DataDir::new(root.join("data"));
    data.ensure().unwrap();
    if let Some(servers) = global {
        std::fs::write(
            data.mcp_file(),
            json!({ "mcpServers": servers }).to_string(),
        )
        .unwrap();
    }
    if let Some(servers) = project {
        std::fs::write(
            work.join(".mcp.json"),
            json!({ "mcpServers": servers }).to_string(),
        )
        .unwrap();
    }
    let deps = CoreDeps {
        config,
        workspace: Workspace::from_dir(&work).unwrap(),
        data_dir: data.clone(),
        provider,
        memory_provider: Some(Arc::new(MockProvider::new())),
        extra_tools: vec![],
        env_vars: vec![],
        api_keys: None,
    };
    let (ui_tx, ui) = mpsc::unbounded_channel();
    let (cmd, cmd_rx) = mpsc::unbounded_channel();
    let core = tokio::spawn(Core::new(deps, ui_tx).unwrap().run(cmd_rx));
    Run {
        _tmp: tmp,
        work,
        data,
        ui,
        cmd,
        core,
        seen: Vec::new(),
        trust,
    }
}

impl Run {
    async fn until(&mut self, mut pred: impl FnMut(&UiEvent) -> bool) -> UiEvent {
        loop {
            let ev = tokio::time::timeout(WAIT, self.ui.recv())
                .await
                .expect("timed out")
                .expect("core ended");
            match &ev {
                UiEvent::Approval(p) => self
                    .cmd
                    .send(CoreCommand::ApprovalResponse {
                        id: p.id,
                        choice: ApprovalChoice::AllowOnce,
                    })
                    .unwrap(),
                UiEvent::Question(q) => self
                    .cmd
                    .send(CoreCommand::QuestionAnswer {
                        id: q.id,
                        answer: self.trust.into(),
                    })
                    .unwrap(),
                _ => {}
            }
            self.seen.push(ev.clone());
            if pred(&ev) {
                return ev;
            }
        }
    }

    async fn turn(&mut self, prompt: &str) {
        self.cmd
            .send(CoreCommand::UserPrompt(prompt.into()))
            .unwrap();
        self.until(|e| matches!(e, UiEvent::TurnFinished { .. }))
            .await;
    }

    async fn notice(&mut self, needle: &str) -> String {
        match self
            .until(|e| matches!(e, UiEvent::Notice { text, .. } if text.contains(needle)))
            .await
        {
            UiEvent::Notice { text, .. } => text,
            _ => unreachable!(),
        }
    }

    async fn slash(&mut self, action: McpAction, needle: &str) -> String {
        self.cmd
            .send(CoreCommand::Slash(SlashCommand::Mcp(action)))
            .unwrap();
        self.notice(needle).await
    }

    async fn quit(self) -> Vec<UiEvent> {
        self.cmd.send(CoreCommand::Quit).unwrap();
        tokio::time::timeout(WAIT, self.core)
            .await
            .unwrap()
            .unwrap();
        self.seen
    }
}

async fn docs_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(path("/mcp"))
        .respond_with(HttpMcp)
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn mcp_tool_is_found_by_tool_search_and_runs_after_approval() {
    let server = docs_server().await;
    let provider = Arc::new(Scripted::default());
    let mut run = start(
        provider.clone(),
        Config::default(),
        Some(json!({ "docs": { "type": "http", "url": format!("{}/mcp", server.uri()) } })),
        None,
        "",
    );
    let listed = run.slash(McpAction::List, "MCP servers").await;
    assert!(
        listed.contains("- docs (global, http://")
            && listed.contains("stopped, tools unknown, not logged in"),
        "{listed}"
    );
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "lazy start"
    );

    run.turn("check the docs for deploy").await;
    let reqs = provider.orchestrator_requests();
    assert!(
        reqs[0].messages[0]
            .content()
            .contains("MCP servers whose tools tool_search can also find: docs")
    );
    let results = turn_results(reqs.last().unwrap());
    assert!(
        results[0]
            .contains("- mcp__docs__lookup: [MCP server docs] Look up internal documentation"),
        "{results:?}"
    );
    assert!(results[1].contains(
        "<untrusted source=\"tool:mcp__docs__lookup\">\npage deploy: deploy with make ship\n"
    ));
    assert!(
        reqs.last()
            .unwrap()
            .tools
            .iter()
            .any(|t| t.name == "mcp__docs__lookup")
    );

    let listed = run.slash(McpAction::List, "MCP servers").await;
    assert!(listed.contains("running, 1 tool(s)"), "{listed}");
    let seen = run.quit().await;
    let approval = seen
        .iter()
        .find_map(|e| match e {
            UiEvent::Approval(p) => Some(p.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(approval.tool, "mcp__docs__lookup");
    assert_eq!(approval.risk, RiskClass::External);
    assert!(
        seen.iter()
            .any(|e| matches!(e, UiEvent::Usage(u) if u.mcp_running == 1))
    );
}

#[tokio::test]
async fn disabled_mcp_starts_nothing_until_enabled() {
    let server = docs_server().await;
    let provider = Arc::new(Scripted::default());
    let mut config = Config::default();
    config.mcp.enabled = false;
    let mut run = start(
        provider.clone(),
        config,
        Some(json!({ "docs": { "type": "http", "url": format!("{}/mcp", server.uri()) } })),
        None,
        "",
    );
    run.turn("check the docs for deploy").await;
    let reqs = provider.orchestrator_requests();
    assert!(!reqs[0].messages[0].content().contains("MCP servers"));
    let results = turn_results(reqs.last().unwrap());
    assert!(results[0].contains("Loaded") || results[0].contains("no matching"));
    assert!(!results[0].contains("mcp__"), "{}", results[0]);
    assert!(server.received_requests().await.unwrap().is_empty());
    let listed = run.slash(McpAction::List, "MCP").await;
    assert_eq!(listed, "MCP is disabled; /mcp enable reads the config");

    let enabled = run.slash(McpAction::Enable, "MCP enabled").await;
    assert_eq!(enabled, "MCP enabled: 1 server(s) configured");
    run.turn("check the docs again").await;
    let results = turn_results(provider.orchestrator_requests().last().unwrap());
    assert!(results[1].contains("page deploy: deploy with make ship"));

    let off = run.slash(McpAction::Disable, "MCP disabled").await;
    assert!(off.contains("for this session"));
    let listed = run.slash(McpAction::List, "MCP servers").await;
    assert!(listed.starts_with("MCP servers (disabled):"), "{listed}");
    run.quit().await;
}

#[tokio::test]
async fn project_servers_are_trusted_once_per_project() {
    let server = docs_server().await;
    let entry = json!({ "type": "http", "url": format!("{}/mcp", server.uri()) });
    let mut run = start(
        Arc::new(Scripted::default()),
        Config::default(),
        None,
        Some(json!({ "docs": entry })),
        "trust",
    );
    let question = run.until(|e| matches!(e, UiEvent::Question(_))).await;
    let UiEvent::Question(q) = question else {
        unreachable!()
    };
    assert!(
        q.text.contains("declares the MCP server `docs`"),
        "{}",
        q.text
    );
    assert!(
        q.text
            .contains(&format!("It connects to {}/mcp", server.uri()))
    );
    assert_eq!(q.options, ["trust", "don't trust"]);
    let done = run.notice("is trusted for this project").await;
    assert_eq!(done, "MCP server `docs` is trusted for this project");

    let projects = ProjectIndex::new(run.data.projects_file(), run.data.locks_dir());
    let trust = projects.get(&run.work).unwrap().unwrap().mcp_trust;
    assert!(trust["docs"].trusted);
    assert_eq!(
        trust["docs"].fingerprint,
        h_agent_mcp::config::fingerprint(&entry)
    );
    let listed = run.slash(McpAction::List, "MCP servers").await;
    assert!(listed.contains("- docs (project, "), "{listed}");
    assert!(!listed.contains("awaiting trust"), "{listed}");
    let seen = run.quit().await;
    assert!(!seen.iter().any(
        |e| matches!(e, UiEvent::Notice { level: NoticeLevel::Warning | NoticeLevel::Error, text } if text.contains("MCP"))
    ));
}
