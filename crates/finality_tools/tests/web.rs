#![allow(clippy::unwrap_used)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use finality_core::config::{BrowserConfig, SafetyConfig, SearchBackend, WebConfig};
use finality_core::{AgentId, Mode};
use finality_safety::{Location, PathGuard, PermissionEngine};
use finality_tools::{
    ActionPreview, ApprovalRequest, ApprovalResponse, Approver, BrowserAction, BrowserSession,
    BrowserTool, Execution, Tool, ToolCtx, ToolEnv, ToolExecutor, ToolLimits, ToolRegistry,
    WebFetch, WebSearch,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

struct Fx {
    _tmp: tempfile::TempDir,
    work: PathBuf,
    ctx: ToolCtx,
    approver: Arc<Scripted>,
    exec: ToolExecutor,
}

fn fx(web: WebConfig, browser: BrowserConfig) -> Fx {
    let tmp = tempfile::tempdir().unwrap();
    let work = dunce::canonicalize(tmp.path()).unwrap();
    let safety = SafetyConfig::default();
    let env = ToolEnv::new(PathGuard::new(&work).unwrap(), [], &safety.env_denylist);
    let ctx = ToolCtx {
        env: Arc::new(env),
        limits: ToolLimits::default(),
        cancel: CancellationToken::new(),
        agent: AgentId::new(),
    };
    let mut registry = ToolRegistry::with_builtins();
    registry.register_web(&web, &BrowserSession::new(browser, &web));
    let approver = Arc::new(Scripted::default());
    let exec = ToolExecutor::new(
        Arc::new(registry),
        Arc::new(Mutex::new(PermissionEngine::new(
            Mode::Auto,
            &safety.always_ask_patterns,
        ))),
        approver.clone(),
    );
    Fx {
        _tmp: tmp,
        work,
        ctx,
        approver,
        exec,
    }
}

fn private_web() -> WebConfig {
    WebConfig {
        allow_private_network: true,
        ..WebConfig::default()
    }
}

impl Fx {
    async fn run(&self, tool: &str, args: Value) -> Execution {
        self.exec.execute(tool, args, &self.ctx, None).await
    }

    fn approve(&self, n: usize) {
        let mut q = self.approver.responses.lock().unwrap();
        q.extend((0..n).map(|_| ApprovalResponse::AllowOnce));
    }

    fn asked(&self) -> usize {
        self.approver.seen.lock().unwrap().len()
    }
}

async fn direct(tool: &dyn Tool, args: Value, ctx: &ToolCtx) -> finality_tools::ToolOutput {
    let preview = tool.preview(&args, ctx).await.unwrap();
    tool.call(args, ctx, &preview).await.unwrap()
}

const DDG_PAGE: &str = r##"<html><body>
  <div class="result results_links result--ad"><a class="result__a" href="https://ads.example/">Buy now</a></div>
  <div class="result results_links web-result">
    <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fziglang.org%2Flearn%2Fbuild-system%2F&amp;rut=1">Zig <b>Build</b> System</a></h2>
    <a class="result__snippet" href="#">The Zig build system provides a cross-platform way.</a>
  </div>
  <div class="result results_links web-result">
    <h2 class="result__title"><a class="result__a" href="https://example.com/zig">Second hit</a></h2>
    <a class="result__snippet">More about zig.</a>
  </div>
</body></html>"##;

fn ddg_web() -> WebConfig {
    WebConfig {
        search_backend: SearchBackend::DuckDuckGo,
        ..WebConfig::default()
    }
}

#[tokio::test]
async fn duckduckgo_results_are_parsed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/html/"))
        .and(query_param("q", "zig build system"))
        .respond_with(ResponseTemplate::new(200).set_body_string(DDG_PAGE))
        .expect(2)
        .mount(&server)
        .await;
    let f = fx(WebConfig::default(), BrowserConfig::default());
    let tool = WebSearch::new(ddg_web()).with_duckduckgo_url(format!("{}/html/", server.uri()));

    let out = direct(&tool, json!({ "query": "zig build system" }), &f.ctx).await;
    assert!(!out.is_error, "{}", out.content);
    assert_eq!(
        out.content,
        "1. Zig Build System\n   https://ziglang.org/learn/build-system/\n   The Zig build system provides a cross-platform way.\n\
         2. Second hit\n   https://example.com/zig\n   More about zig."
    );
    assert!(!out.content.contains("Buy now"));

    let one = direct(
        &tool,
        json!({ "query": "zig build system", "max_results": 1 }),
        &f.ctx,
    )
    .await;
    assert!(!one.content.contains("Second hit"));

    let again = direct(&tool, json!({ "query": "zig build system" }), &f.ctx).await;
    assert_eq!(
        again.content, out.content,
        "served from the cache (the mock expects 2 requests)"
    );
}

const DDG_BOT_CHECK: &str = r#"<html><body><form id="challenge-form">
  <div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div></form></body></html>"#;

const DDG_LITE_PAGE: &str = r#"<table>
  <tr><td>1.</td><td><a class="result-link" href="https://ziglang.org/">Zig</a></td></tr>
  <tr><td></td><td class="result-snippet">A general-purpose language.</td></tr>
</table>"#;

#[tokio::test]
async fn duckduckgo_bot_checks_fall_back_to_lite_and_are_reported() {
    let server = MockServer::start().await;
    Mock::given(path("/html/"))
        .respond_with(ResponseTemplate::new(202).set_body_string(DDG_BOT_CHECK))
        .mount(&server)
        .await;
    Mock::given(path("/lite/"))
        .and(query_param("q", "zig"))
        .respond_with(ResponseTemplate::new(200).set_body_string(DDG_LITE_PAGE))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/lite/"))
        .and(query_param("q", "blocked"))
        .respond_with(ResponseTemplate::new(200).set_body_string(DDG_BOT_CHECK))
        .expect(2)
        .mount(&server)
        .await;
    let f = fx(WebConfig::default(), BrowserConfig::default());
    let tool = WebSearch::new(ddg_web())
        .with_duckduckgo_url(format!("{}/html/", server.uri()))
        .with_duckduckgo_lite_url(format!("{}/lite/", server.uri()))
        .with_retry_delay(std::time::Duration::ZERO);

    let out = direct(&tool, json!({ "query": "zig" }), &f.ctx).await;
    assert_eq!(
        out.content,
        "1. Zig\n   https://ziglang.org/\n   A general-purpose language."
    );
    let blocked = direct(&tool, json!({ "query": "blocked" }), &f.ctx).await;
    assert!(
        blocked.is_error && blocked.content.contains("refusing automated searches"),
        "{}",
        blocked.content
    );
}

#[tokio::test]
async fn tavily_is_used_with_a_key_and_falls_back_to_duckduckgo() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(header("authorization", "Bearer tvly-good"))
        .and(body_partial_json(json!({ "query": "rust async", "max_results": 8 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "query": "rust async",
            "results": [
                { "title": "Async Book", "url": "https://rust-lang.github.io/async-book/", "content": "Asynchronous   Programming in Rust", "score": 0.9 }
            ]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({ "detail": { "error": "Unauthorized: missing or invalid API key." } }),
        ))
        .mount(&server)
        .await;
    Mock::given(path("/html/"))
        .respond_with(ResponseTemplate::new(200).set_body_string(DDG_PAGE))
        .mount(&server)
        .await;
    let f = fx(WebConfig::default(), BrowserConfig::default());
    let tavily = |key: Option<&str>| {
        WebSearch::new(WebConfig::default())
            .with_tavily(format!("{}/search", server.uri()), key.map(str::to_owned))
            .with_duckduckgo_url(format!("{}/html/", server.uri()))
    };

    let out = direct(
        &tavily(Some("tvly-good")),
        json!({ "query": "rust async" }),
        &f.ctx,
    )
    .await;
    assert_eq!(
        out.content,
        "1. Async Book\n   https://rust-lang.github.io/async-book/\n   Asynchronous Programming in Rust"
    );
    let fallback = direct(&tavily(Some("tvly-bad")), json!({ "query": "zig" }), &f.ctx).await;
    assert!(
        fallback.content.starts_with(
            "(Tavily returned 401 Unauthorized: Unauthorized: missing or invalid API key.; these results are from DuckDuckGo)\n1. Zig Build System"
        ),
        "{}",
        fallback.content
    );
    assert!(
        !fallback.content.contains("tvly-bad"),
        "the key never shows"
    );
    let no_key = direct(&tavily(None), json!({ "query": "zig" }), &f.ctx).await;
    assert!(
        no_key.content.starts_with("1. Zig Build System"),
        "{}",
        no_key.content
    );

    let required = WebSearch::new(WebConfig {
        search_backend: SearchBackend::Tavily,
        ..WebConfig::default()
    })
    .with_tavily(format!("{}/search", server.uri()), None);
    let missing = direct(&required, json!({ "query": "zig" }), &f.ctx).await;
    assert!(
        missing.is_error && missing.content.contains("TAVILY_API_KEY"),
        "{}",
        missing.content
    );
}

#[tokio::test]
async fn searxng_backend_uses_json_api() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/search"))
        .and(query_param("format", "json"))
        .and(query_param("q", "rust async"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "title": "Async Book", "url": "https://rust-lang.github.io/async-book/", "content": "Asynchronous Programming in Rust" },
                { "title": "Tokio", "url": "https://tokio.rs/" }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let web = WebConfig {
        search_backend: SearchBackend::SearXng,
        searxng_url: format!("{}/", server.uri()),
        ..WebConfig::default()
    };
    let f = fx(web.clone(), BrowserConfig::default());
    let out = direct(
        &WebSearch::new(web),
        json!({ "query": "rust async" }),
        &f.ctx,
    )
    .await;
    assert_eq!(
        out.content,
        "1. Async Book\n   https://rust-lang.github.io/async-book/\n   Asynchronous Programming in Rust\n\
         2. Tokio\n   https://tokio.rs/"
    );
}

#[tokio::test]
async fn fetch_blocks_local_targets_by_default() {
    let server = MockServer::start().await;
    let f = fx(WebConfig::default(), BrowserConfig::default());
    for url in [
        format!("{}/", server.uri()),
        "http://127.0.0.1/".to_owned(),
        "http://169.254.169.254/latest/meta-data/".to_owned(),
        "file:///etc/passwd".to_owned(),
    ] {
        let r = f.run("web_fetch", json!({ "url": url })).await;
        assert!(r.output.is_error, "{url}: {}", r.output.content);
        assert!(r.approval.is_none(), "{url} must fail before authorization");
        assert!(
            r.output.content.contains("blocked") || r.output.content.contains("scheme"),
            "{url}: {}",
            r.output.content
        );
    }
    assert_eq!(f.asked(), 0);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn fetch_private_host_asks_then_returns_readable_text() {
    let server = MockServer::start().await;
    Mock::given(path("/start"))
        .respond_with(ResponseTemplate::new(302).insert_header("Location", "/article"))
        .mount(&server)
        .await;
    Mock::given(path("/article"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "<html><head><title>Notes</title><script>var secret = 1;</script></head>\
             <body><h1>Heading</h1><p>First   paragraph.</p></body></html>",
            "text/html; charset=utf-8",
        ))
        .mount(&server)
        .await;
    Mock::given(path("/data.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "ok": true })))
        .mount(&server)
        .await;

    let f = fx(private_web(), BrowserConfig::default());
    f.approve(1);
    let r = f
        .run(
            "web_fetch",
            json!({ "url": format!("{}/start", server.uri()) }),
        )
        .await;
    assert!(!r.output.is_error, "{}", r.output.content);
    assert_eq!(f.asked(), 1, "a private address always needs approval");
    let req = f.approver.seen.lock().unwrap()[0].clone();
    assert!(
        req.reasons.iter().any(|x| x.contains("Loopback")),
        "{:?}",
        req.reasons
    );
    let text = r.output.content;
    assert!(
        text.contains(&format!("URL: {}/article", server.uri())),
        "{text}"
    );
    assert!(text.contains("Title: Notes"));
    assert!(text.contains("Heading\n\nFirst paragraph."), "{text}");
    assert!(!text.contains("secret"));

    let json = direct(
        &WebFetch::new(&private_web()),
        json!({ "url": format!("{}/data.json", server.uri()) }),
        &f.ctx,
    )
    .await;
    assert!(json.content.ends_with(r#"{"ok":true}"#), "{}", json.content);
}

#[tokio::test]
async fn fetch_redirects_are_checked_and_limited() {
    let server = MockServer::start().await;
    Mock::given(path("/to-private"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("Location", "http://10.255.255.1/admin"),
        )
        .mount(&server)
        .await;
    Mock::given(path("/to-file"))
        .respond_with(ResponseTemplate::new(301).insert_header("Location", "file:///etc/passwd"))
        .mount(&server)
        .await;
    Mock::given(path("/loop"))
        .respond_with(ResponseTemplate::new(302).insert_header("Location", "/loop"))
        .mount(&server)
        .await;
    let f = fx(private_web(), BrowserConfig::default());
    let tool = WebFetch::new(&private_web());

    let out = direct(
        &tool,
        json!({ "url": format!("{}/to-private", server.uri()) }),
        &f.ctx,
    )
    .await;
    assert!(out.is_error);
    assert!(
        out.content.contains("http://10.255.255.1/admin") && out.content.contains("needs approval"),
        "{}",
        out.content
    );

    let out = direct(
        &tool,
        json!({ "url": format!("{}/to-file", server.uri()) }),
        &f.ctx,
    )
    .await;
    assert!(
        out.is_error && out.content.contains("scheme `file`"),
        "{}",
        out.content
    );

    let out = direct(
        &tool,
        json!({ "url": format!("{}/loop", server.uri()) }),
        &f.ctx,
    )
    .await;
    assert!(
        out.is_error && out.content.contains("more than 5 redirects"),
        "{}",
        out.content
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1 + 1 + 6);
}

#[tokio::test]
async fn fetch_output_is_truncated_to_max_chars() {
    let server = MockServer::start().await;
    Mock::given(path("/big.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("x".repeat(50_000), "text/plain"))
        .mount(&server)
        .await;
    let f = fx(private_web(), BrowserConfig::default());
    let out = direct(
        &WebFetch::new(&private_web()),
        json!({ "url": format!("{}/big.txt", server.uri()), "max_chars": 1000 }),
        &f.ctx,
    )
    .await;
    assert!(out.truncated);
    assert!(out.content.chars().count() < 1100);
}

#[tokio::test]
async fn long_pages_are_read_in_parts_from_the_cache() {
    let server = MockServer::start().await;
    let lines: Vec<String> = (1..=300)
        .map(|n| format!("line {n:03} of the manual"))
        .collect();
    Mock::given(path("/manual.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(lines.join("\n"), "text/plain"))
        .expect(2)
        .mount(&server)
        .await;
    let f = fx(private_web(), BrowserConfig::default());
    let tool = WebFetch::new(&private_web());
    let url = format!("{}/manual.txt", server.uri());

    let first = direct(&tool, json!({ "url": url, "max_chars": 1000 }), &f.ctx).await;
    assert!(first.truncated);
    let next: usize = first
        .content
        .rsplit("offset ")
        .next()
        .and_then(|s| s.split_whitespace().next())
        .unwrap()
        .parse()
        .unwrap();
    assert!(first.content.contains("line 001") && !first.content.contains("line 300"));
    let body_end = first.content.rfind("\n\n[characters").unwrap();
    assert!(
        first.content[..body_end].ends_with("of the manual"),
        "cut at a line end"
    );

    let second = direct(
        &tool,
        json!({ "url": url, "max_chars": 1000, "offset": next }),
        &f.ctx,
    )
    .await;
    assert!(
        second.content.contains("Cached: fetched"),
        "{}",
        second.content
    );
    let first_line = second
        .content
        .split("\n\n")
        .nth(1)
        .unwrap()
        .lines()
        .next()
        .unwrap();
    assert!(
        first_line.starts_with("line "),
        "continues at a line start: {first_line}"
    );
    assert!(
        !first.content.contains(first_line),
        "no overlap: {first_line}"
    );

    let end = direct(&tool, json!({ "url": url, "offset": 1_000_000 }), &f.ctx).await;
    assert!(end.content.contains("past the end"), "{}", end.content);
    let fresh = direct(&tool, json!({ "url": url, "refresh": true }), &f.ctx).await;
    assert!(!fresh.content.contains("Cached:"));
}

#[tokio::test]
async fn fetched_html_is_markdown_without_page_chrome() {
    let server = MockServer::start().await;
    let body = format!(
        "<html><head><title>Guide</title></head><body><nav><a href='/'>Home</a></nav>\
         <main><h2>Install</h2><p>{}</p><pre><code class='language-sh'>cargo install finality</code></pre>\
         <p>See <a href='/docs/setup'>setup</a>.</p></main><footer>© Example</footer></body></html>",
        "Finality installs with cargo. ".repeat(10)
    );
    Mock::given(path("/guide"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/html"))
        .mount(&server)
        .await;
    let f = fx(private_web(), BrowserConfig::default());
    let out = direct(
        &WebFetch::new(&private_web()),
        json!({ "url": format!("{}/guide", server.uri()) }),
        &f.ctx,
    )
    .await;
    let text = out.content;
    assert!(
        text.contains("Title: Guide\n\n## Install\n\nFinality installs"),
        "{text}"
    );
    assert!(
        text.contains("```sh\ncargo install finality\n```"),
        "{text}"
    );
    assert!(
        text.contains(&format!("See [setup]({}/docs/setup).", server.uri())),
        "{text}"
    );
    assert!(
        !text.contains("Home") && !text.contains("Example"),
        "{text}"
    );
}

#[tokio::test]
async fn missing_browser_gives_actionable_error_and_web_tools_still_work() {
    let server = MockServer::start().await;
    Mock::given(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
        .mount(&server)
        .await;
    let browser = BrowserConfig {
        chrome_path: r"C:\definitely\missing\chrome.exe".into(),
        ..BrowserConfig::default()
    };
    let session = BrowserSession::new(browser.clone(), &private_web());
    let err = session.executable().unwrap_err();
    assert!(err.contains("browser.chrome_path"), "{err}");

    let f = fx(private_web(), browser);
    let open = BrowserTool::new(BrowserAction::Open, session.clone());
    let out = open
        .call(
            json!({ "url": server.uri() }),
            &f.ctx,
            &ActionPreview::new("open", Location::Inside),
        )
        .await
        .unwrap();
    assert!(
        out.is_error && out.content.contains("browser.chrome_path"),
        "{}",
        out.content
    );

    let snap = f.run("browser_snapshot", json!({})).await;
    assert!(
        snap.output.content.contains("call browser_open first"),
        "{}",
        snap.output.content
    );
    let click = f.run("browser_click", json!({})).await;
    assert!(
        click.output.content.contains("exactly one of"),
        "{}",
        click.output.content
    );

    f.approve(1);
    let fetched = f.run("web_fetch", json!({ "url": server.uri() })).await;
    assert!(
        fetched.output.content.ends_with("hello"),
        "{}",
        fetched.output.content
    );
    assert!(!session.close().await);
}

#[tokio::test]
#[ignore = "needs Chrome, Chromium or Edge installed"]
async fn browser_fills_and_submits_a_local_form() {
    let server = MockServer::start().await;
    // A second private server the page tries to reach on its own; the guard must stop it.
    let other = MockServer::start().await;
    Mock::given(path("/pixel"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("x", "text/plain"))
        .mount(&other)
        .await;
    let page = format!(
        r#"<html><head><title>Signup</title></head><body>
           <img src="{}/pixel"><script>fetch("{}/pixel").catch(() => {{}})</script>
           <form action="/done" method="get">
             <label for="name">Name</label> <input id="name" name="name">
             <label><input type="checkbox" name="agree"> Agree</label>
             <button type="submit">Send</button>
           </form></body></html>"#,
        other.uri(),
        other.uri()
    );
    Mock::given(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(page, "text/html"))
        .mount(&server)
        .await;
    Mock::given(path("/done"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "<html><head><title>Done</title></head><body><p>Thanks!</p></body></html>",
            "text/html",
        ))
        .mount(&server)
        .await;
    let f = fx(private_web(), BrowserConfig::default());
    f.approve(1);

    let open = f.run("browser_open", json!({ "url": server.uri() })).await;
    let snap = open.output.content;
    assert!(!open.output.is_error, "{snap}");
    assert!(snap.contains("Title: Signup"), "{snap}");
    assert!(snap.contains(r#"[1] text "Name" value="""#), "{snap}");
    assert!(
        snap.contains(r#"[2] checkbox "Agree" value="unchecked""#),
        "{snap}"
    );
    assert!(snap.contains(r#"[3] button "Send""#), "{snap}");

    let fill = f
        .run(
            "browser_fill_form",
            json!({ "fields": [ { "element": 1, "value": "Ada Lovelace" }, { "element": 2, "value": "true" } ], "submit": true }),
        )
        .await;
    assert!(
        fill.output.content.contains("submitted the form"),
        "{}",
        fill.output.content
    );

    let after = f.run("browser_snapshot", json!({})).await.output.content;
    assert!(
        after.contains("Title: Done") && after.contains("Thanks!"),
        "{after}"
    );
    assert!(
        after.contains("/done?name=Ada+Lovelace&agree=on"),
        "{after}"
    );

    let shot = f
        .run("browser_screenshot", json!({ "path": "shots/done.png" }))
        .await;
    assert!(!shot.output.is_error, "{}", shot.output.content);
    let png = std::fs::read(f.work.join("shots/done.png")).unwrap();
    assert!(png.starts_with(b"\x89PNG"));

    let close = f.run("browser_close", json!({})).await;
    assert_eq!(close.output.content, "Browser closed");
    assert!(
        other
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "the page reached a private host the user never approved"
    );
}
