//! `web_search` and `web_fetch` (DESIGN §9.2, §12.6).

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use finality_core::RiskClass;
use finality_core::config::{SearchBackend, WebConfig};
use finality_safety::{Location, NetVerdict, NetworkGuard};
use futures::StreamExt;
use reqwest::header::{CONTENT_TYPE, LOCATION};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use crate::html::{
    SearchHit, duckduckgo_blocked, html_to_markdown, parse_duckduckgo, parse_duckduckgo_lite,
};
use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, decode_text, parse_args,
    truncate_middle,
};

pub const DUCKDUCKGO_URL: &str = "https://html.duckduckgo.com/html/";
pub const DUCKDUCKGO_LITE_URL: &str = "https://lite.duckduckgo.com/lite/";
pub const TAVILY_URL: &str = "https://api.tavily.com/search";
/// Keyring service and user of the Tavily key; the same service as the model providers' keys.
const KEYRING_SERVICE: &str = "finality";
const TAVILY_KEYRING_USER: &str = "api-key:tavily";
const MAX_REDIRECTS: usize = 5;
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_RESULTS: usize = 8;
const MAX_RESULTS: usize = 20;
const MAX_SNIPPET_CHARS: usize = 300;
/// Pause before asking DuckDuckGo Lite again after a bot check.
const RETRY_DELAY: Duration = Duration::from_secs(2);
const CACHE_TTL: Duration = Duration::from_secs(15 * 60);
const CACHE_ENTRIES: usize = 64;
/// Characters kept for the footer that says how to read on.
const FOOTER_RESERVE: usize = 120;
const MIN_WINDOW: usize = 200;
const MAX_PDF_CHARS: usize = 2_000_000;

fn failed(e: impl std::fmt::Display) -> ToolError {
    ToolError::Failed(e.to_string())
}

/// Recent results, kept for 15 minutes, so a repeated search or reading on with `offset` costs no request.
struct Cache<T> {
    entries: Mutex<VecDeque<(String, Instant, T)>>,
}

impl<T: Clone> Cache<T> {
    fn new() -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
        }
    }

    /// The value and its age.
    fn get(&self, key: &str) -> Option<(T, Duration)> {
        let mut entries = self.entries.lock().ok()?;
        entries.retain(|(_, at, _)| at.elapsed() < CACHE_TTL);
        entries
            .iter()
            .find(|(k, _, _)| k == key)
            .map(|(_, at, v)| (v.clone(), at.elapsed()))
    }

    fn put(&self, key: String, value: T) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.retain(|(k, _, _)| *k != key);
            entries.push_back((key, Instant::now(), value));
            while entries.len() > CACHE_ENTRIES {
                entries.pop_front();
            }
        }
    }
}

/// Reads at most `MAX_BODY_BYTES`; the flag says whether the body was cut.
async fn read_capped(resp: reqwest::Response, what: &str) -> Result<(Vec<u8>, bool), String> {
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("reading {what} failed: {e}"))?;
        let room = MAX_BODY_BYTES - body.len();
        if chunk.len() > room {
            body.extend_from_slice(&chunk[..room]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body, false))
}

/// Preview for a URL the agent wants to reach: blocked targets fail, private ones (when allowed
/// by config) need approval, public ones are treated like workspace-internal actions.
pub(crate) async fn preview_url(
    guard: &NetworkGuard,
    verb: &str,
    raw: &str,
) -> Result<ActionPreview, ToolError> {
    let (host, port) = guard.target(raw).map_err(failed)?;
    let mut p = ActionPreview::new(format!("{verb} {raw}"), Location::Inside);
    match guard.check(raw).await {
        NetVerdict::Block { reason } => return Err(failed(format!("{raw} is blocked: {reason}"))),
        NetVerdict::Ask { reason, .. } => {
            p.location = Location::Outside;
            p.reasons.push(reason);
            // Only a target the user approved may be reused when execution resolves it again.
            p.state = json!({ "host": host, "port": port });
        }
        NetVerdict::Allow { .. } => {}
    }
    p.grant_keys = vec![host.clone()];
    Ok(p)
}

pub(crate) fn approved_target(preview: &ActionPreview) -> Option<(String, u16)> {
    let host = preview.state.get("host")?.as_str()?.to_owned();
    let port = u16::try_from(preview.state.get("port")?.as_u64()?).ok()?;
    Some((host, port))
}

pub struct Fetched {
    pub url: Url,
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub truncated: bool,
}

/// GET with manual redirects; every hop is checked by the guard and connects only to the
/// addresses it approved. A hop needing approval is allowed only for the approved host:port.
pub async fn guarded_get(
    guard: &NetworkGuard,
    user_agent: &str,
    raw: &str,
    approved: Option<&(String, u16)>,
) -> Result<Fetched, String> {
    let mut url = Url::parse(raw).map_err(|e| format!("invalid URL: {e}"))?;
    for _ in 0..=MAX_REDIRECTS {
        let target = guard.target(url.as_str())?;
        let addrs: Vec<SocketAddr> = match guard.check(url.as_str()).await {
            NetVerdict::Allow { addrs } => addrs,
            NetVerdict::Ask { addrs, .. } if approved == Some(&target) => addrs,
            NetVerdict::Ask { reason, .. } => {
                return Err(format!(
                    "redirect to {url} needs approval ({reason}); fetch it directly instead"
                ));
            }
            NetVerdict::Block { reason } => return Err(format!("{url} is blocked: {reason}")),
        };
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(user_agent)
            .timeout(TIMEOUT)
            .resolve_to_addrs(&target.0, &addrs)
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .get(url.clone())
            .send()
            .await
            .map_err(|e| format!("request to {url} failed: {e}"))?;
        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("{url} redirected without a Location header"))?;
            url = url
                .join(location)
                .map_err(|e| format!("bad redirect from {url}: {e}"))?;
            continue;
        }
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let final_url = resp.url().clone();
        let (body, truncated) = read_capped(resp, url.as_str()).await?;
        return Ok(Fetched {
            url: final_url,
            status: status.as_u16(),
            content_type,
            body,
            truncated,
        });
    }
    Err(format!("more than {MAX_REDIRECTS} redirects"))
}

// ---------------------------------------------------------------- web_search

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    max_results: Option<usize>,
}

type KeyLookup = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// The Tavily key: the configured environment variable, else the OS keyring.
fn tavily_key(env_var: &str) -> Option<String> {
    if !env_var.is_empty()
        && let Ok(v) = std::env::var(env_var)
        && !v.trim().is_empty()
    {
        return Some(v.trim().to_owned());
    }
    keyring::Entry::new(KEYRING_SERVICE, TAVILY_KEYRING_USER)
        .ok()?
        .get_password()
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

enum DdgPage {
    Hits(Vec<SearchHit>),
    Blocked,
}

pub struct WebSearch {
    cfg: WebConfig,
    duckduckgo_url: String,
    lite_url: String,
    tavily_url: String,
    tavily_key: KeyLookup,
    retry_delay: Duration,
    cache: Cache<String>,
}

impl WebSearch {
    pub fn new(cfg: WebConfig) -> Self {
        let env_var = cfg.tavily_api_key_env.clone();
        Self {
            cfg,
            duckduckgo_url: DUCKDUCKGO_URL.to_owned(),
            lite_url: DUCKDUCKGO_LITE_URL.to_owned(),
            tavily_url: TAVILY_URL.to_owned(),
            tavily_key: Arc::new(move || tavily_key(&env_var)),
            retry_delay: RETRY_DELAY,
            cache: Cache::new(),
        }
    }

    pub fn with_duckduckgo_url(mut self, url: impl Into<String>) -> Self {
        self.duckduckgo_url = url.into();
        self
    }

    pub fn with_duckduckgo_lite_url(mut self, url: impl Into<String>) -> Self {
        self.lite_url = url.into();
        self
    }

    /// Another Tavily endpoint and key (`None`: no key), for tests.
    pub fn with_tavily(mut self, url: impl Into<String>, key: Option<String>) -> Self {
        self.tavily_url = url.into();
        self.tavily_key = Arc::new(move || key.clone());
        self
    }

    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    fn client(&self) -> Result<reqwest::Client, String> {
        reqwest::Client::builder()
            .user_agent(&self.cfg.user_agent)
            .timeout(TIMEOUT)
            .build()
            .map_err(|e| e.to_string())
    }

    /// Hits and, when another backend had to stand in, a note saying so.
    async fn search(
        &self,
        query: &str,
        max: usize,
    ) -> Result<(Vec<SearchHit>, Option<String>), String> {
        let client = self.client()?;
        match self.cfg.search_backend {
            SearchBackend::Tavily => {
                let key = (self.tavily_key)().ok_or_else(|| {
                    format!(
                        "no Tavily API key: set {} or store it in the OS keyring (service `{KEYRING_SERVICE}`, user \
                         `{TAVILY_KEYRING_USER}`)",
                        self.cfg.tavily_api_key_env
                    )
                })?;
                Ok((self.tavily(&client, &key, query, max).await?, None))
            }
            SearchBackend::Auto => match (self.tavily_key)() {
                Some(key) => match self.tavily(&client, &key, query, max).await {
                    Ok(hits) => Ok((hits, None)),
                    Err(e) => {
                        let hits = self.duckduckgo(&client, query, max).await?;
                        Ok((
                            hits,
                            Some(format!("{e}; these results are from DuckDuckGo")),
                        ))
                    }
                },
                None => Ok((self.duckduckgo(&client, query, max).await?, None)),
            },
            SearchBackend::DuckDuckGo => Ok((self.duckduckgo(&client, query, max).await?, None)),
            SearchBackend::SearXng => Ok((self.searxng(&client, query, max).await?, None)),
        }
    }

    async fn tavily(
        &self,
        client: &reqwest::Client,
        key: &str,
        query: &str,
        max: usize,
    ) -> Result<Vec<SearchHit>, String> {
        #[derive(Deserialize)]
        struct Results {
            #[serde(default)]
            results: Vec<Item>,
        }
        #[derive(Deserialize)]
        struct Item {
            #[serde(default)]
            title: String,
            url: String,
            #[serde(default)]
            content: String,
        }
        let resp = client
            .post(&self.tavily_url)
            .bearer_auth(key)
            .json(&json!({ "query": query, "max_results": max, "search_depth": "basic" }))
            .send()
            .await
            .map_err(|e| format!("Tavily search failed: {e}"))?;
        let status = resp.status();
        let (bytes, truncated) = read_capped(resp, "Tavily").await?;
        if !status.is_success() {
            let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            let detail = body["detail"]["error"]
                .as_str()
                .or_else(|| body["detail"].as_str())
                .or_else(|| body["error"].as_str())
                .map_or_else(
                    || truncate_middle(String::from_utf8_lossy(&bytes).trim(), 200).0,
                    str::to_owned,
                );
            return Err(format!("Tavily returned {status}: {detail}"));
        }
        if truncated {
            return Err("Tavily response too large".into());
        }
        let body: Results = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Tavily returned unexpected JSON: {e}"))?;
        Ok(body
            .results
            .into_iter()
            .take(max)
            .map(|i| SearchHit {
                title: i.title,
                url: i.url,
                snippet: i.content,
            })
            .collect())
    }

    /// The HTML endpoint, then DuckDuckGo Lite (twice, with a pause) when DuckDuckGo answers with its bot check.
    async fn duckduckgo(
        &self,
        client: &reqwest::Client,
        query: &str,
        max: usize,
    ) -> Result<Vec<SearchHit>, String> {
        if let DdgPage::Hits(hits) = self
            .duckduckgo_page(client, &self.duckduckgo_url, query, max, parse_duckduckgo)
            .await?
        {
            return Ok(hits);
        }
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(self.retry_delay).await;
            }
            if let DdgPage::Hits(hits) = self
                .duckduckgo_page(client, &self.lite_url, query, max, parse_duckduckgo_lite)
                .await?
            {
                return Ok(hits);
            }
        }
        Err(format!(
            "DuckDuckGo is refusing automated searches right now (it answers with a bot check), so this is not \
             \"no results\". Try again in a few minutes, or set a Tavily API key ({}) or web.search_backend = \
             \"searxng\".",
            self.cfg.tavily_api_key_env
        ))
    }

    async fn duckduckgo_page(
        &self,
        client: &reqwest::Client,
        url: &str,
        query: &str,
        max: usize,
        parse: fn(&str, usize) -> Vec<SearchHit>,
    ) -> Result<DdgPage, String> {
        let resp = client
            .get(url)
            .query(&[("q", query)])
            .send()
            .await
            .map_err(|e| format!("DuckDuckGo search failed: {e}"))?;
        let status = resp.status().as_u16();
        if matches!(status, 403 | 429) {
            return Ok(DdgPage::Blocked);
        }
        let resp = resp
            .error_for_status()
            .map_err(|e| format!("DuckDuckGo search failed: {e}"))?;
        let (bytes, _) = read_capped(resp, "DuckDuckGo").await?;
        let html = String::from_utf8_lossy(&bytes);
        let hits = parse(&html, max);
        // A bot check comes back as 200 or 202 with a challenge page instead of results.
        if hits.is_empty() && (status == 202 || duckduckgo_blocked(&html)) {
            return Ok(DdgPage::Blocked);
        }
        Ok(DdgPage::Hits(hits))
    }

    async fn searxng(
        &self,
        client: &reqwest::Client,
        query: &str,
        max: usize,
    ) -> Result<Vec<SearchHit>, String> {
        #[derive(Deserialize)]
        struct Results {
            #[serde(default)]
            results: Vec<Item>,
        }
        #[derive(Deserialize)]
        struct Item {
            #[serde(default)]
            title: String,
            url: String,
            #[serde(default)]
            content: String,
        }
        let base = self.cfg.searxng_url.trim_end_matches('/');
        let resp = client
            .get(format!("{base}/search"))
            .query(&[("q", query), ("format", "json")])
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| format!("SearXNG search failed: {e}"))?;
        let (bytes, truncated) = read_capped(resp, "SearXNG").await?;
        if truncated {
            return Err("SearXNG response too large".into());
        }
        let body: Results = serde_json::from_slice(&bytes)
            .map_err(|e| format!("SearXNG returned unexpected JSON: {e}"))?;
        Ok(body
            .results
            .into_iter()
            .take(max)
            .map(|i| SearchHit {
                title: i.title,
                url: i.url,
                snippet: i.content,
            })
            .collect())
    }
}

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web. Returns titles, URLs and snippets; use web_fetch to read a result. Repeating a search \
         within 15 minutes returns the earlier results."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "max_results": { "type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "description": "Results to return (default 8)" }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Network
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: SearchArgs = parse_args(args)?;
        if a.query.trim().is_empty() {
            return Err(ToolError::InvalidArgs("query is empty".into()));
        }
        // The search backend is trusted configuration, not an agent-chosen target.
        Ok(ActionPreview::new(
            format!("search the web for \"{}\"", a.query),
            Location::Inside,
        ))
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        _preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: SearchArgs = parse_args(&args)?;
        let max = a
            .max_results
            .unwrap_or(DEFAULT_RESULTS)
            .clamp(1, MAX_RESULTS);
        let key = format!("{:?}\n{max}\n{}", self.cfg.search_backend, a.query.trim());
        if let Some((text, _)) = self.cache.get(&key) {
            return Ok(ToolOutput::ok(text));
        }
        let found = tokio::select! {
            r = self.search(&a.query, max) => r,
            () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        let (hits, note) = match found {
            Ok(found) => found,
            Err(e) => return Ok(ToolOutput::error(e)),
        };
        let mut out = String::new();
        if let Some(note) = note {
            out.push_str(&format!("({note})\n"));
        }
        if hits.is_empty() {
            out.push_str(&format!("no results for \"{}\"", a.query));
        }
        for (i, h) in hits.iter().enumerate() {
            out.push_str(&format!("{}. {}\n   {}\n", i + 1, h.title, h.url));
            let snippet = h.snippet.split_whitespace().collect::<Vec<_>>().join(" ");
            if !snippet.is_empty() {
                let snippet = if snippet.chars().count() > MAX_SNIPPET_CHARS {
                    snippet.chars().take(MAX_SNIPPET_CHARS).collect::<String>() + "…"
                } else {
                    snippet
                };
                out.push_str(&format!("   {snippet}\n"));
            }
        }
        let out = out.trim_end().to_owned();
        self.cache.put(key, out.clone());
        Ok(ToolOutput::ok(out))
    }
}

// ---------------------------------------------------------------- web_fetch

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    url: String,
    max_chars: Option<usize>,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    refresh: bool,
}

/// A fetched page as the model reads it.
struct Page {
    url: String,
    status: u16,
    content_type: String,
    title: Option<String>,
    /// Said before the text, e.g. why a body is not shown.
    note: Option<String>,
    text: String,
}

/// Text of a PDF, page by page. lopdf may panic on malformed files, so callers catch unwinds.
fn pdf_text(bytes: &[u8]) -> Result<String, String> {
    let doc = lopdf::Document::load_mem(bytes).map_err(|e| format!("cannot read the PDF: {e}"))?;
    if doc.is_encrypted() {
        return Err("the PDF is encrypted".into());
    }
    let mut out = String::new();
    for (i, number) in doc.get_pages().keys().enumerate() {
        if out.len() > MAX_PDF_CHARS {
            out.push_str("\n\n[later pages left out]");
            break;
        }
        if let Ok(text) = doc.extract_text(&[*number]) {
            let text = text.trim();
            if !text.is_empty() {
                out.push_str(&format!("\n\n--- page {} ---\n{text}", i + 1));
            }
        }
    }
    let out = out.trim().to_owned();
    if out.is_empty() {
        Err("the PDF has no extractable text (scanned pages?)".into())
    } else {
        Ok(out)
    }
}

async fn render_page(f: Fetched) -> Page {
    let ct = f.content_type.to_ascii_lowercase();
    let mut page = Page {
        url: f.url.to_string(),
        status: f.status,
        content_type: f.content_type.clone(),
        title: None,
        note: None,
        text: String::new(),
    };
    let looks_html =
        ct.contains("html") || (ct.is_empty() && f.body.trim_ascii_start().starts_with(b"<"));
    let pdf = ct.contains("pdf") || f.body.starts_with(b"%PDF-");
    if looks_html {
        let (title, text) = html_to_markdown(&String::from_utf8_lossy(&f.body), Some(&f.url));
        page.title = title;
        page.text = text;
    } else if pdf && f.truncated {
        page.note = Some(format!(
            "PDF larger than {} MB; not extracted",
            MAX_BODY_BYTES / 1024 / 1024
        ));
    } else if pdf {
        let body = f.body;
        let extracted = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(|| pdf_text(&body))
                .unwrap_or_else(|_| Err("the PDF could not be parsed".into()))
        })
        .await
        .unwrap_or_else(|e| Err(e.to_string()));
        match extracted {
            Ok(text) => page.text = text,
            Err(e) => page.note = Some(e),
        }
    } else if ct.starts_with("image/")
        || ct.starts_with("audio/")
        || ct.starts_with("video/")
        || ct.contains("octet-stream")
        || ct.contains("zip")
    {
        page.note = Some(format!("binary content, {} bytes; not shown", f.body.len()));
    } else {
        match decode_text(&f.body) {
            Some(t) => page.text = t,
            None => {
                page.note = Some(format!("binary content, {} bytes; not shown", f.body.len()));
            }
        }
    }
    if f.truncated && !pdf {
        let cut = format!("body cut at {} MB", MAX_BODY_BYTES / 1024 / 1024);
        page.note = Some(match page.note.take() {
            Some(n) => format!("{n}; {cut}"),
            None => cut,
        });
    }
    page
}

/// The page from character `offset`, cut to about `max_chars` at a line end, with a footer that says how to read on.
/// The flag is true when text follows the window.
fn window(page: &Page, offset: usize, max_chars: usize, age: Option<Duration>) -> (String, bool) {
    let mut head = format!("URL: {}\nStatus: {}\n", page.url, page.status);
    if !page.content_type.is_empty() {
        head.push_str(&format!("Content-Type: {}\n", page.content_type));
    }
    if let Some(t) = &page.title {
        head.push_str(&format!("Title: {t}\n"));
    }
    if let Some(age) = age {
        head.push_str(&format!(
            "Cached: fetched {} s ago (refresh: true fetches again)\n",
            age.as_secs()
        ));
    }
    if let Some(n) = &page.note {
        head.push_str(&format!("Note: {n}\n"));
    }
    let total = page.text.chars().count();
    if total == 0 {
        return (head.trim_end().to_owned(), false);
    }
    if offset >= total {
        return (
            format!("{head}\n[offset {offset} is past the end: the text has {total} characters]"),
            false,
        );
    }
    let budget = max_chars
        .saturating_sub(head.chars().count() + FOOTER_RESERVE)
        .max(MIN_WINDOW);
    let start = page
        .text
        .char_indices()
        .nth(offset)
        .map_or(page.text.len(), |(i, _)| i);
    let rest = &page.text[start..];
    let mut end = rest
        .char_indices()
        .nth(budget)
        .map_or(rest.len(), |(i, _)| i);
    if end < rest.len()
        && let Some(nl) = rest[..end].rfind('\n')
        && nl > end * 4 / 5
    {
        end = nl + 1;
    }
    let shown = &rest[..end];
    let last = offset + shown.chars().count();
    let more = last < total;
    let mut out = format!("{head}\n{}", shown.trim_end());
    if more {
        out.push_str(&format!(
            "\n\n[characters {offset}-{last} of {total}; call web_fetch with offset {last} to read on]"
        ));
    } else if offset > 0 {
        out.push_str(&format!(
            "\n\n[characters {offset}-{last} of {total}; end of the page]"
        ));
    }
    (out, more)
}

pub struct WebFetch {
    guard: NetworkGuard,
    user_agent: String,
    cache: Cache<Arc<Page>>,
}

impl WebFetch {
    pub fn new(cfg: &WebConfig) -> Self {
        Self {
            guard: NetworkGuard::new(cfg.allow_private_network),
            user_agent: cfg.user_agent.clone(),
            cache: Cache::new(),
        }
    }
}

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL over http(s) and return its main content as Markdown (menus, headers, footers and sidebars \
         left out; links and code blocks kept), or the text of a PDF. Long pages come in parts: the footer gives the \
         offset to read on. Pages are cached for 15 minutes; refresh: true fetches again. Private and local network \
         addresses are blocked."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "http or https URL" },
                "max_chars": { "type": "integer", "minimum": 500, "description": "Maximum characters to return" },
                "offset": { "type": "integer", "minimum": 0, "description": "Character offset in the page text, to read on" },
                "refresh": { "type": "boolean", "description": "Fetch again instead of using the cached copy" }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }

    fn risk(&self) -> RiskClass {
        RiskClass::Network
    }

    fn tier(&self) -> Tier {
        Tier::Deferred
    }

    async fn preview(&self, args: &Value, _ctx: &ToolCtx) -> Result<ActionPreview, ToolError> {
        let a: FetchArgs = parse_args(args)?;
        preview_url(&self.guard, "fetch", &a.url).await
    }

    async fn call(
        &self,
        args: Value,
        ctx: &ToolCtx,
        preview: &ActionPreview,
    ) -> Result<ToolOutput, ToolError> {
        let a: FetchArgs = parse_args(&args)?;
        let cached = if a.refresh {
            None
        } else {
            self.cache.get(&a.url)
        };
        let (page, age) = match cached {
            Some((page, age)) => (page, Some(age)),
            None => {
                let approved = approved_target(preview);
                let fetched = tokio::select! {
                    r = guarded_get(&self.guard, &self.user_agent, &a.url, approved.as_ref()) => r,
                    () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
                };
                let fetched = match fetched {
                    Ok(f) => f,
                    Err(e) => return Ok(ToolOutput::error(e)),
                };
                let page = Arc::new(render_page(fetched).await);
                if page.status < 400 {
                    self.cache.put(a.url.clone(), page.clone());
                }
                (page, None)
            }
        };
        let max = a.max_chars.unwrap_or(ctx.limits.max_output_chars);
        let (content, more) = window(&page, a.offset, max, age);
        Ok(ToolOutput {
            content,
            is_error: page.status >= 400,
            truncated: more,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn sample_pdf(lines: &[&str]) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{Document, Object, Stream, dictionary};
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Courier",
        });
        let resources_id =
            doc.add_object(dictionary! { "Font" => dictionary! { "F1" => font_id } });
        let mut kids: Vec<Object> = Vec::new();
        for text in lines {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![100.into(), 600.into()]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
            });
            kids.push(page_id.into());
        }
        let count = i64::try_from(kids.len()).unwrap();
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => kids, "Count" => count, "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[tokio::test]
    async fn pdfs_become_text_page_by_page() {
        let fetched = |body: Vec<u8>, truncated| Fetched {
            url: Url::parse("https://example.com/paper.pdf").unwrap(),
            status: 200,
            content_type: "application/pdf".into(),
            body,
            truncated,
        };
        let page = render_page(fetched(sample_pdf(&["Hello PDF", "Second page"]), false)).await;
        assert_eq!(page.note, None);
        assert!(
            page.text.starts_with("--- page 1 ---\nHello PDF"),
            "{}",
            page.text
        );
        assert!(
            page.text.contains("--- page 2 ---\nSecond page"),
            "{}",
            page.text
        );

        let broken = render_page(fetched(b"%PDF-1.5 not really".to_vec(), false)).await;
        assert!(broken.note.unwrap().contains("PDF"));
        let big = render_page(fetched(b"%PDF-1.5".to_vec(), true)).await;
        assert!(big.note.unwrap().contains("not extracted"));
    }

    /// A public host approved at preview time must not pre-approve a private address that the
    /// same name resolves to later (DNS rebinding).
    #[tokio::test]
    async fn only_private_targets_that_asked_are_remembered() {
        let guard = NetworkGuard::new(true);
        let public = preview_url(&guard, "fetch", "http://93.184.216.34/")
            .await
            .unwrap();
        assert_eq!(public.location, Location::Inside);
        assert_eq!(approved_target(&public), None);

        let private = preview_url(&guard, "fetch", "http://127.0.0.1:8080/x")
            .await
            .unwrap();
        assert_eq!(private.location, Location::Outside);
        assert_eq!(approved_target(&private), Some(("127.0.0.1".into(), 8080)));

        let err = match guarded_get(&guard, "ua", "http://127.0.0.1:8080/x", None).await {
            Err(e) => e,
            Ok(f) => panic!("fetched {} without approval", f.url),
        };
        assert!(err.contains("needs approval"), "{err}");
    }
}
