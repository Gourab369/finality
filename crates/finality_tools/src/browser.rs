//! Browser automation over the Chrome DevTools Protocol (DESIGN §9.2). One lazily started
//! Chrome/Chromium/Edge per session with a single page shared by all agents.
//!
//! Every request the page makes (navigations, redirects, scripts, frames, fetches) is paused through
//! the Fetch domain and checked by the network guard; a private address passes only when it is the
//! host and port the user approved in `browser_open`. Chrome resolves names itself, so the check
//! sees the guard's own DNS answer rather than the one Chrome connects to.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chromiumoxide::cdp::browser_protocol::fetch::{
    ContinueRequestParams, EnableParams as FetchEnableParams, EventRequestPaused,
    FailRequestParams, RequestPattern, RequestStage,
};
use chromiumoxide::cdp::browser_protocol::network::ErrorReason;
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::detection::{DetectionOptions, default_executable};
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::{Browser, Page};
use finality_core::RiskClass;
use finality_core::config::{BrowserConfig, WebConfig};
use finality_safety::{Location, NetVerdict, NetworkGuard};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, display_path, parse_args,
    resolve_path, truncate_middle,
};
use crate::web::{approved_target, preview_url};

const SNAPSHOT_TEXT_CHARS: usize = 6000;
const SETTLE_TIMEOUT: Duration = Duration::from_secs(10);
/// Bound on the guard's DNS lookup for one intercepted request; slower means blocked.
const INTERCEPT_TIMEOUT: Duration = Duration::from_secs(10);

/// The private `host:port` the user approved for the current navigation, if any.
type Approved = Arc<std::sync::Mutex<Option<(String, u16)>>>;

struct Live {
    browser: Browser,
    page: Page,
    handler: JoinHandle<()>,
    interceptor: JoinHandle<()>,
}

pub struct BrowserSession {
    cfg: BrowserConfig,
    guard: NetworkGuard,
    approved: Approved,
    live: Mutex<Option<Live>>,
}

impl BrowserSession {
    pub fn new(cfg: BrowserConfig, web: &WebConfig) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            guard: NetworkGuard::new(web.allow_private_network),
            approved: Arc::default(),
            live: Mutex::new(None),
        })
    }

    /// The browser executable: `browser.chrome_path`, else an installed Chrome, Chromium or Edge.
    pub fn executable(&self) -> Result<PathBuf, String> {
        if !self.cfg.chrome_path.trim().is_empty() {
            let path = PathBuf::from(self.cfg.chrome_path.trim());
            return if path.is_file() {
                Ok(path)
            } else {
                Err(format!(
                    "browser.chrome_path `{}` does not exist; fix it in config.toml",
                    path.display()
                ))
            };
        }
        default_executable(DetectionOptions::default()).map_err(|_| {
            "no Chrome, Chromium or Edge found; install one or set browser.chrome_path in config.toml"
                .to_owned()
        })
    }

    async fn page(&self) -> Result<Page, String> {
        let mut live = self.live.lock().await;
        if let Some(l) = live.as_ref() {
            return Ok(l.page.clone());
        }
        let exe = self.executable()?;
        static RUN: AtomicU64 = AtomicU64::new(0);
        let profile = std::env::temp_dir().join(format!(
            "finality-browser-{}-{}",
            std::process::id(),
            RUN.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = chromiumoxide::BrowserConfig::builder()
            .chrome_executable(&exe)
            .user_data_dir(&profile)
            .window_size(1280, 900);
        if !self.cfg.headless {
            builder = builder.with_head();
        }
        let config = builder.build()?;
        let (mut browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
        let handler = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if event.is_err() {
                    break;
                }
            }
        });
        let page = match browser.new_page("about:blank").await {
            Ok(p) => p,
            Err(e) => {
                handler.abort();
                let _ = browser.kill().await;
                return Err(format!("cannot open a browser tab: {e}"));
            }
        };
        let interceptor = match intercept(&page, self.guard, self.approved.clone()).await {
            Ok(task) => task,
            Err(e) => {
                handler.abort();
                let _ = browser.kill().await;
                return Err(format!("cannot guard browser requests: {e}"));
            }
        };
        *live = Some(Live {
            browser,
            page: page.clone(),
            handler,
            interceptor,
        });
        Ok(page)
    }

    /// Records the private target `browser_open` was approved for (none for public pages).
    fn set_approved(&self, target: Option<(String, u16)>) {
        if let Ok(mut a) = self.approved.lock() {
            *a = target;
        }
    }

    async fn current(&self) -> Result<Page, String> {
        self.live
            .lock()
            .await
            .as_ref()
            .map(|l| l.page.clone())
            .ok_or_else(|| "no page is open; call browser_open first".to_owned())
    }

    /// Closes the browser if running. Returns whether one was running.
    pub async fn close(&self) -> bool {
        let Some(mut live) = self.live.lock().await.take() else {
            return false;
        };
        if live.browser.close().await.is_err() {
            let _ = live.browser.kill().await;
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), live.browser.wait()).await;
        live.handler.abort();
        live.interceptor.abort();
        true
    }
}

/// Pauses every request of `page` and lets through only what the guard allows.
async fn intercept(
    page: &Page,
    guard: NetworkGuard,
    approved: Approved,
) -> Result<JoinHandle<()>, String> {
    let mut events = page
        .event_listener::<EventRequestPaused>()
        .await
        .map_err(|e| e.to_string())?;
    page.execute(
        FetchEnableParams::builder()
            .pattern(
                RequestPattern::builder()
                    .url_pattern("*")
                    .request_stage(RequestStage::Request)
                    .build(),
            )
            .build(),
    )
    .await
    .map_err(|e| e.to_string())?;
    let page = page.clone();
    Ok(tokio::spawn(async move {
        while let Some(ev) = events.next().await {
            let (page, approved) = (page.clone(), approved.clone());
            tokio::spawn(async move {
                let url = ev.request.url.clone();
                let ok = matches!(
                    tokio::time::timeout(
                        INTERCEPT_TIMEOUT,
                        request_allowed(&guard, &approved, &url)
                    )
                    .await,
                    Ok(true)
                );
                let id = ev.request_id.clone();
                let sent = if ok {
                    page.execute(ContinueRequestParams::new(id)).await.map(drop)
                } else {
                    tracing::info!(%url, "browser request blocked by the network guard");
                    page.execute(FailRequestParams::new(id, ErrorReason::BlockedByClient))
                        .await
                        .map(drop)
                };
                if let Err(e) = sent {
                    tracing::debug!(error = %e, "cannot answer a paused browser request");
                }
            });
        }
    }))
}

async fn request_allowed(guard: &NetworkGuard, approved: &Approved, url: &str) -> bool {
    // These never reach the network.
    if ["data:", "blob:", "about:"]
        .iter()
        .any(|s| url.starts_with(s))
    {
        return true;
    }
    match guard.check(url).await {
        NetVerdict::Allow { .. } => true,
        NetVerdict::Ask { .. } => {
            let target = guard.target(url).ok();
            approved
                .lock()
                .is_ok_and(|a| target.is_some() && *a == target)
        }
        NetVerdict::Block { .. } => false,
    }
}

async fn eval<T: serde::de::DeserializeOwned>(page: &Page, js: String) -> Result<T, String> {
    page.evaluate(js)
        .await
        .map_err(|e| format!("page script failed: {e}"))?
        .into_value::<T>()
        .map_err(|e| format!("unexpected page script result: {e}"))
}

/// Waits briefly for a navigation started by a click or submit to finish loading.
async fn settle(page: &Page) {
    tokio::time::sleep(Duration::from_millis(300)).await;
    let deadline = tokio::time::Instant::now() + SETTLE_TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        let state: Result<String, String> = eval(page, "document.readyState".into()).await;
        if state.as_deref() == Ok("complete") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

const JS_HELPERS: &str = r#"
function pick(t) {
  if (t.element) return document.querySelector('[data-finality="' + t.element + '"]');
  if (t.selector) return document.querySelector(t.selector);
  return null;
}
function describe(t) { return t.element ? ('element ' + t.element) : ('selector ' + t.selector); }
function setValue(e, v) {
  e.focus();
  if (e.type === 'checkbox' || e.type === 'radio') {
    const want = !(v === '' || v === 'false' || v === '0' || v === 'off');
    if (e.checked !== want) e.click();
    return;
  }
  const proto = e.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype
    : e.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  const d = Object.getOwnPropertyDescriptor(proto, 'value');
  if (e.isContentEditable) e.innerText = v; else if (d && d.set) d.set.call(e, v); else e.value = v;
  e.dispatchEvent(new Event('input', { bubbles: true }));
  e.dispatchEvent(new Event('change', { bubbles: true }));
}
function submitFrom(e) {
  const f = (e && (e.form || e.closest('form'))) || document.forms[0];
  if (!f) return false;
  if (f.requestSubmit) f.requestSubmit(); else f.submit();
  return true;
}
"#;

const JS_SNAPSHOT: &str = r#"
document.querySelectorAll('[data-finality]').forEach(e => e.removeAttribute('data-finality'));
const sel = 'a[href],button,input:not([type=hidden]),select,textarea,[role=button],[role=link],[role=checkbox],[role=tab],[contenteditable=true],[onclick]';
const visible = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
  return s.visibility !== 'hidden' && s.display !== 'none' && (r.width > 0 || r.height > 0); };
const out = [];
Array.from(document.querySelectorAll(sel)).filter(visible).slice(0, 300).forEach((e, i) => {
  const n = i + 1;
  e.setAttribute('data-finality', String(n));
  const tag = e.tagName.toLowerCase();
  const role = e.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'input' ? (e.type || 'text') : tag);
  const label = (e.labels && e.labels[0] && e.labels[0].innerText) || e.getAttribute('aria-label')
    || e.getAttribute('placeholder') || e.innerText || e.getAttribute('title') || e.getAttribute('name')
    || (tag === 'input' ? e.value : '') || '';
  const field = tag === 'input' || tag === 'textarea' || tag === 'select';
  const value = !field ? null : (e.type === 'checkbox' || e.type === 'radio') ? (e.checked ? 'checked' : 'unchecked')
    : e.type === 'password' ? (e.value ? '***' : '') : String(e.value || '').slice(0, 80);
  out.push({ n, role, name: label.trim().replace(/\s+/g, ' ').slice(0, 80), value });
});
return { title: document.title, url: location.href,
  text: (document.body ? document.body.innerText : '').slice(0, LIMIT), elements: out };
"#;

fn script(body: &str, args: &Value) -> String {
    format!("(function(){{ const ARGS = {args}; {JS_HELPERS} {body} }})()")
}

#[derive(Deserialize)]
struct Snapshot {
    title: String,
    url: String,
    text: String,
    elements: Vec<SnapElement>,
}

#[derive(Deserialize)]
struct SnapElement {
    n: u32,
    role: String,
    name: String,
    value: Option<String>,
}

async fn snapshot(page: &Page) -> Result<String, String> {
    let js = script(
        &JS_SNAPSHOT.replace("LIMIT", &SNAPSHOT_TEXT_CHARS.to_string()),
        &Value::Null,
    );
    let s: Snapshot = eval(page, js).await?;
    let mut out = format!("Title: {}\nURL: {}\n\nElements:\n", s.title, s.url);
    if s.elements.is_empty() {
        out.push_str("(none)\n");
    }
    for e in &s.elements {
        out.push_str(&format!("[{}] {} \"{}\"", e.n, e.role, e.name));
        if let Some(v) = &e.value {
            out.push_str(&format!(" value=\"{v}\""));
        }
        out.push('\n');
    }
    out.push_str("\nText:\n");
    out.push_str(s.text.trim());
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowserAction {
    Open,
    Snapshot,
    Click,
    Type,
    FillForm,
    Screenshot,
    Close,
}

impl BrowserAction {
    pub const ALL: [Self; 7] = [
        Self::Open,
        Self::Snapshot,
        Self::Click,
        Self::Type,
        Self::FillForm,
        Self::Screenshot,
        Self::Close,
    ];
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    element: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selector: Option<String>,
}

impl Target {
    fn check(&self) -> Result<(), ToolError> {
        match (&self.element, &self.selector) {
            (Some(_), None) | (None, Some(_)) => Ok(()),
            _ => Err(ToolError::InvalidArgs(
                "give exactly one of `element` (number from browser_snapshot) or `selector`".into(),
            )),
        }
    }

    fn label(&self) -> String {
        match (&self.element, &self.selector) {
            (Some(n), _) => format!("element {n}"),
            (_, Some(s)) => format!("`{s}`"),
            _ => "?".into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    url: String,
}

type ClickArgs = Target;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TypeArgs {
    element: Option<u32>,
    selector: Option<String>,
    text: String,
    #[serde(default)]
    submit: bool,
}

impl TypeArgs {
    fn target(&self) -> Target {
        Target {
            element: self.element,
            selector: self.selector.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Field {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    element: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selector: Option<String>,
    value: String,
}

impl Field {
    fn target(&self) -> Target {
        Target {
            element: self.element,
            selector: self.selector.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FillArgs {
    fields: Vec<Field>,
    #[serde(default)]
    submit: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScreenshotArgs {
    path: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[derive(Deserialize)]
struct ActResult {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    submitted: bool,
}

pub struct BrowserTool {
    action: BrowserAction,
    session: Arc<BrowserSession>,
}

impl BrowserTool {
    pub fn new(action: BrowserAction, session: Arc<BrowserSession>) -> Self {
        Self { action, session }
    }

    pub fn all(session: &Arc<BrowserSession>) -> Vec<Self> {
        BrowserAction::ALL
            .into_iter()
            .map(|a| Self::new(a, session.clone()))
            .collect()
    }

    async fn act(&self, page: &Page, body: &str, args: Value) -> Result<ActResult, String> {
        let r: ActResult = eval(page, script(body, &args)).await?;
        if !r.ok {
            return Err(r.error.unwrap_or_else(|| "action failed".into()));
        }
        Ok(r)
    }

    async fn after_action(&self, page: &Page, what: String, navigates: bool) -> String {
        if navigates {
            settle(page).await;
        }
        let title: String = eval(page, "document.title".into())
            .await
            .unwrap_or_default();
        let url = page.url().await.ok().flatten().unwrap_or_default();
        format!("{what}. Page: \"{title}\" {url}\nCall browser_snapshot to see the updated page.")
    }

    async fn run(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<String, String> {
        match self.action {
            BrowserAction::Open => {
                let a: OpenArgs = parse_args(&args).map_err(|e| e.to_string())?;
                let page = self.session.page().await?;
                self.session.set_approved(approved_target(preview));
                page.goto(a.url.as_str())
                    .await
                    .map_err(|e| format!("cannot open {}: {e}", a.url))?;
                settle(&page).await;
                snapshot(&page).await
            }
            BrowserAction::Snapshot => snapshot(&self.session.current().await?).await,
            BrowserAction::Click => {
                let a: ClickArgs = parse_args(&args).map_err(|e| e.to_string())?;
                let page = self.session.current().await?;
                let label = a.label();
                self.act(
                    &page,
                    "const e = pick(ARGS); if (!e) return { ok: false, error: describe(ARGS) + ' not found; take a new browser_snapshot' };
                     e.scrollIntoView({ block: 'center' }); e.click(); return { ok: true };",
                    json!(a),
                )
                .await?;
                Ok(self
                    .after_action(&page, format!("Clicked {label}"), true)
                    .await)
            }
            BrowserAction::Type => {
                let a: TypeArgs = parse_args(&args).map_err(|e| e.to_string())?;
                let page = self.session.current().await?;
                let label = a.target().label();
                let r = self
                    .act(
                        &page,
                        "const e = pick(ARGS.target); if (!e) return { ok: false, error: describe(ARGS.target) + ' not found; take a new browser_snapshot' };
                         setValue(e, ARGS.text); return { ok: true, submitted: ARGS.submit ? submitFrom(e) : false };",
                        json!({ "target": a.target(), "text": a.text, "submit": a.submit }),
                    )
                    .await?;
                let mut what = format!("Typed into {label}");
                if a.submit {
                    what.push_str(if r.submitted {
                        " and submitted the form"
                    } else {
                        "; no form to submit"
                    });
                }
                Ok(self.after_action(&page, what, r.submitted).await)
            }
            BrowserAction::FillForm => {
                let a: FillArgs = parse_args(&args).map_err(|e| e.to_string())?;
                let page = self.session.current().await?;
                let r = self
                    .act(
                        &page,
                        "let last = null;
                         for (const f of ARGS.fields) { const e = pick(f); if (!e) return { ok: false, error: describe(f) + ' not found; take a new browser_snapshot' }; setValue(e, f.value); last = e; }
                         return { ok: true, submitted: ARGS.submit ? submitFrom(last) : false };",
                        json!({ "fields": a.fields, "submit": a.submit }),
                    )
                    .await?;
                let mut what = format!("Filled {} field(s)", a.fields.len());
                if a.submit {
                    what.push_str(if r.submitted {
                        " and submitted the form"
                    } else {
                        "; no form to submit"
                    });
                }
                Ok(self.after_action(&page, what, r.submitted).await)
            }
            BrowserAction::Screenshot => {
                let page = self.session.current().await?;
                let png = page
                    .screenshot(
                        ScreenshotParams::builder()
                            .format(CaptureScreenshotFormat::Png)
                            .full_page(true)
                            .build(),
                    )
                    .await
                    .map_err(|e| format!("screenshot failed: {e}"))?;
                let path = &preview.paths[0];
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| e.to_string())?;
                }
                tokio::fs::write(path, &png)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(format!(
                    "Saved screenshot ({} bytes) to {}",
                    png.len(),
                    display_path(&ctx.env, path)
                ))
            }
            BrowserAction::Close => Ok(if self.session.close().await {
                "Browser closed".into()
            } else {
                "No browser was open".into()
            }),
        }
    }
}

fn target_props() -> Value {
    json!({
        "element": { "type": "integer", "minimum": 1, "description": "Element number from browser_snapshot" },
        "selector": { "type": "string", "description": "CSS selector, when no element number fits" }
    })
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        match self.action {
            BrowserAction::Open => "browser_open",
            BrowserAction::Snapshot => "browser_snapshot",
            BrowserAction::Click => "browser_click",
            BrowserAction::Type => "browser_type",
            BrowserAction::FillForm => "browser_fill_form",
            BrowserAction::Screenshot => "browser_screenshot",
            BrowserAction::Close => "browser_close",
        }
    }

    fn description(&self) -> &str {
        match self.action {
            BrowserAction::Open => {
                "Open a URL in a real browser (Chrome/Chromium/Edge) and return a snapshot: \
                 title, numbered interactive elements and visible text. Use for JavaScript-heavy \
                 pages and forms; prefer web_fetch for plain pages."
            }
            BrowserAction::Snapshot => {
                "Snapshot the open browser page: title, URL, numbered interactive elements and visible text. \
                 Element numbers change with every snapshot."
            }
            BrowserAction::Click => {
                "Click an element of the open browser page, by snapshot number or CSS selector."
            }
            BrowserAction::Type => {
                "Set the text of an input or textarea on the open browser page; optionally submit its form."
            }
            BrowserAction::FillForm => {
                "Fill several form fields on the open browser page (checkboxes take \"true\"/\"false\"); optionally submit."
            }
            BrowserAction::Screenshot => "Save a PNG screenshot of the full open browser page.",
            BrowserAction::Close => "Close the browser.",
        }
    }

    fn schema(&self) -> Value {
        let empty = json!({ "type": "object", "properties": {}, "additionalProperties": false });
        match self.action {
            BrowserAction::Open => json!({
                "type": "object",
                "properties": { "url": { "type": "string", "description": "http or https URL" } },
                "required": ["url"],
                "additionalProperties": false
            }),
            BrowserAction::Snapshot | BrowserAction::Close => empty,
            BrowserAction::Click => json!({
                "type": "object",
                "properties": target_props(),
                "additionalProperties": false
            }),
            BrowserAction::Type => {
                let mut props = target_props();
                props["text"] = json!({ "type": "string" });
                props["submit"] = json!({ "type": "boolean", "description": "Submit the element's form afterwards" });
                json!({ "type": "object", "properties": props, "required": ["text"], "additionalProperties": false })
            }
            BrowserAction::FillForm => {
                let mut field = target_props();
                field["value"] = json!({ "type": "string" });
                json!({
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "array",
                            "minItems": 1,
                            "items": { "type": "object", "properties": field, "required": ["value"], "additionalProperties": false }
                        },
                        "submit": { "type": "boolean", "description": "Submit the form afterwards" }
                    },
                    "required": ["fields"],
                    "additionalProperties": false
                })
            }
            BrowserAction::Screenshot => json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "PNG file path (default screenshot.png)" } },
                "additionalProperties": false
            }),
        }
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Browser
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let summary = match self.action {
            BrowserAction::Open => {
                let a: OpenArgs = parse_args(args)?;
                return preview_url(&self.session.guard, "open in browser", &a.url).await;
            }
            BrowserAction::Snapshot => {
                parse_args::<NoArgs>(args)?;
                "snapshot the browser page".to_owned()
            }
            BrowserAction::Close => {
                parse_args::<NoArgs>(args)?;
                "close the browser".to_owned()
            }
            BrowserAction::Click => {
                let a: ClickArgs = parse_args(args)?;
                a.check()?;
                format!("click {} in the browser", a.label())
            }
            BrowserAction::Type => {
                let a: TypeArgs = parse_args(args)?;
                a.target().check()?;
                format!("type into {} in the browser", a.target().label())
            }
            BrowserAction::FillForm => {
                let a: FillArgs = parse_args(args)?;
                if a.fields.is_empty() {
                    return Err(ToolError::InvalidArgs("fields is empty".into()));
                }
                for f in &a.fields {
                    f.target().check()?;
                }
                let verb = if a.submit { "fill and submit" } else { "fill" };
                format!("{verb} {} form field(s) in the browser", a.fields.len())
            }
            BrowserAction::Screenshot => {
                let a: ScreenshotArgs = parse_args(args)?;
                let raw = a.path.unwrap_or_else(|| "screenshot.png".into());
                let (location, path, reasons) = resolve_path(ctx, &raw);
                let mut p =
                    ActionPreview::new(format!("save a browser screenshot to {raw}"), location);
                p.reasons = reasons;
                p.paths = vec![path];
                return Ok(p);
            }
        };
        Ok(ActionPreview::new(summary, Location::Inside))
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let result = tokio::select! {
            r = self.run(args, ctx, preview) => r,
            () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        Ok(match result {
            Ok(text) => {
                let (content, truncated) = truncate_middle(&text, ctx.limits.max_output_chars);
                ToolOutput {
                    content,
                    is_error: false,
                    truncated,
                }
            }
            Err(e) => ToolOutput::error(e),
        })
    }
}
