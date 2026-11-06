//! `web_search` and `web_fetch` (DESIGN §9.2, §12.6).

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use h_agent_core::RiskClass;
use h_agent_core::config::{SearchBackend, WebConfig};
use h_agent_safety::{Location, NetVerdict, NetworkGuard};
use reqwest::header::{CONTENT_TYPE, LOCATION};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use crate::html::{SearchHit, html_to_text, parse_duckduckgo};
use crate::tool::{
    ActionPreview, Tier, Tool, ToolCtx, ToolError, ToolOutput, decode_text, parse_args,
    truncate_middle,
};

pub const DUCKDUCKGO_URL: &str = "https://html.duckduckgo.com/html/";
const MAX_REDIRECTS: usize = 5;
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_RESULTS: usize = 8;
const MAX_RESULTS: usize = 20;

fn failed(e: impl std::fmt::Display) -> ToolError {
    ToolError::Failed(e.to_string())
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

pub struct WebSearch {
    cfg: WebConfig,
    duckduckgo_url: String,
}

impl WebSearch {
    pub fn new(cfg: WebConfig) -> Self {
        Self {
            cfg,
            duckduckgo_url: DUCKDUCKGO_URL.to_owned(),
        }
    }

    pub fn with_duckduckgo_url(mut self, url: impl Into<String>) -> Self {
        self.duckduckgo_url = url.into();
        self
    }

    async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchHit>, String> {
        let client = reqwest::Client::builder()
            .user_agent(&self.cfg.user_agent)
            .timeout(TIMEOUT)
            .build()
            .map_err(|e| e.to_string())?;
        match self.cfg.search_backend {
            SearchBackend::DuckDuckGo => {
                let resp = client
                    .get(&self.duckduckgo_url)
                    .query(&[("q", query)])
                    .send()
                    .await
                    .and_then(|r| r.error_for_status())
                    .map_err(|e| format!("DuckDuckGo search failed: {e}"))?;
                let (bytes, _) = read_capped(resp, "DuckDuckGo").await?;
                Ok(parse_duckduckgo(&String::from_utf8_lossy(&bytes), max))
            }
            SearchBackend::SearXng => {
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
    }
}

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web. Returns titles, URLs and snippets; use web_fetch to read a result."
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
        let hits = tokio::select! {
            r = self.search(&a.query, max) => r,
            () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        let hits = match hits {
            Ok(h) => h,
            Err(e) => return Ok(ToolOutput::error(e)),
        };
        if hits.is_empty() {
            return Ok(ToolOutput::ok(format!("no results for \"{}\"", a.query)));
        }
        let mut out = String::new();
        for (i, h) in hits.iter().enumerate() {
            out.push_str(&format!("{}. {}\n   {}\n", i + 1, h.title, h.url));
            if !h.snippet.is_empty() {
                out.push_str(&format!("   {}\n", h.snippet));
            }
        }
        Ok(ToolOutput::ok(out.trim_end()))
    }
}

// ---------------------------------------------------------------- web_fetch

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    url: String,
    max_chars: Option<usize>,
}

pub struct WebFetch {
    guard: NetworkGuard,
    user_agent: String,
}

impl WebFetch {
    pub fn new(cfg: &WebConfig) -> Self {
        Self {
            guard: NetworkGuard::new(cfg.allow_private_network),
            user_agent: cfg.user_agent.clone(),
        }
    }
}

fn render_fetched(f: &Fetched) -> String {
    let ct = f.content_type.to_ascii_lowercase();
    let mut head = format!("URL: {}\nStatus: {}\n", f.url, f.status);
    if !f.content_type.is_empty() {
        head.push_str(&format!("Content-Type: {}\n", f.content_type));
    }
    let looks_html =
        ct.contains("html") || (ct.is_empty() && f.body.trim_ascii_start().starts_with(b"<"));
    let body = if looks_html {
        let (title, text) = html_to_text(&String::from_utf8_lossy(&f.body));
        if let Some(t) = title {
            head.push_str(&format!("Title: {t}\n"));
        }
        text
    } else if ct.starts_with("image/")
        || ct.starts_with("audio/")
        || ct.starts_with("video/")
        || ct.contains("octet-stream")
        || ct.contains("pdf")
        || ct.contains("zip")
    {
        format!("(binary content, {} bytes; not shown)", f.body.len())
    } else {
        match decode_text(&f.body) {
            Some(t) => t,
            None => format!("(binary content, {} bytes; not shown)", f.body.len()),
        }
    };
    if f.truncated {
        head.push_str(&format!(
            "Note: body cut at {} MB\n",
            MAX_BODY_BYTES / 1024 / 1024
        ));
    }
    format!("{head}\n{body}")
}

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL over http(s) and return its readable text (HTML is converted to text). \
         Private and local network addresses are blocked."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "http or https URL" },
                "max_chars": { "type": "integer", "minimum": 200, "description": "Maximum characters of text to return" }
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
        let approved = approved_target(preview);
        let fetched = tokio::select! {
            r = guarded_get(&self.guard, &self.user_agent, &a.url, approved.as_ref()) => r,
            () = ctx.cancel.cancelled() => return Err(ToolError::Cancelled),
        };
        let fetched = match fetched {
            Ok(f) => f,
            Err(e) => return Ok(ToolOutput::error(e)),
        };
        let text = render_fetched(&fetched);
        let max = a.max_chars.unwrap_or(ctx.limits.max_output_chars);
        let (content, truncated) = truncate_middle(&text, max);
        Ok(ToolOutput {
            content,
            is_error: fetched.status >= 400,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

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
