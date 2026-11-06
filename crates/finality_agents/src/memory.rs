//! The Memory agent (DESIGN §7.3): one worker that turns session logs into events and answers recalls.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use finality_core::config::Config;
use finality_core::{AgentId, EventId, Role, SessionId, ThinkLevel};
use finality_llm::{ChatRequest, LlmProvider, Message, ModelSelector, collect};
use finality_memory::decision::{self, extract_json, heuristic_summary};
use finality_memory::transcript::{condense, main_spans, spawned_agents, thread_spans};
use finality_memory::tree::{MAX_RELATED, MAX_SUMMARY_CHARS, clip, normalize_name};
use finality_memory::{
    Decision, EventStore, EventView, EventsFile, JobScratch, Placement, RefInfo, Span, apply,
};
use finality_store::{DataDir, LogContents, SessionIndex, SessionLog, read_log};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;

use crate::conversation::wrap_untrusted;
use crate::learner::{LearnJob, Learner};

const SYSTEM: &str = include_str!("../prompts/memory.md");
const LINK_CANDIDATES: usize = 20;
const RECALL_POOL: usize = 60;
const RECALL_HITS: usize = 10;
/// About 2k tokens.
const EXPAND_CHARS: usize = 8000;

/// The session a job is about.
#[derive(Clone)]
pub struct SessionCtx {
    pub id: SessionId,
    pub project_id: String,
    /// (query type, topic) from triage of the first prompt.
    pub hint: Option<(String, String)>,
    /// Flushed before the log is read; `None` once the session is closed.
    pub log: Option<Arc<SessionLog>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallHit {
    pub event_id: EventId,
    pub summary: String,
    pub session_id: SessionId,
    pub date: String,
    pub topic: String,
}

pub enum MemoryJob {
    SessionStart {
        session: SessionCtx,
        upto: u32,
    },
    UserExchanges {
        session: SessionCtx,
        upto: u32,
    },
    SubAgentExchanges {
        session: SessionCtx,
        thread: AgentId,
        role: Role,
        upto: u32,
        ended: bool,
    },
    SessionEnd {
        session: SessionCtx,
    },
    Recall {
        query: String,
        reply: oneshot::Sender<Vec<RecallHit>>,
    },
    Expand {
        event: EventId,
        reply: oneshot::Sender<Result<String, String>>,
    },
}

impl MemoryJob {
    /// Someone waits for recall and expand; session start feeds the next prompts.
    fn priority(&self) -> u8 {
        match self {
            MemoryJob::Recall { .. } | MemoryJob::Expand { .. } => 0,
            MemoryJob::SessionStart { .. } => 1,
            _ => 2,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            MemoryJob::SessionStart { .. } => "session-start",
            MemoryJob::UserExchanges { .. } => "user-exchanges",
            MemoryJob::SubAgentExchanges { .. } => "subagent-exchanges",
            MemoryJob::SessionEnd { .. } => "session-end",
            MemoryJob::Recall { .. } => "recall",
            MemoryJob::Expand { .. } => "expand",
        }
    }
}

pub struct MemoryDeps {
    pub config: Config,
    pub data: DataDir,
    /// Already gated with `Role::Memory`.
    pub provider: Arc<dyn LlmProvider>,
    pub models: Arc<ModelSelector>,
    /// Receives every event the Memory agent closes.
    pub learner: Option<Arc<Learner>>,
}

pub struct MemoryAgent {
    deps: MemoryDeps,
    queue: Mutex<BTreeMap<(u8, u64), MemoryJob>>,
    seq: AtomicU64,
    busy: AtomicBool,
    notify: Notify,
    done: AtomicU64,
}

#[derive(Deserialize)]
struct Classification {
    query_type: String,
    topic: String,
    #[serde(default)]
    session_summary: String,
    #[serde(default)]
    event_summary: String,
}

#[derive(Deserialize)]
struct Indices {
    #[serde(default, alias = "events", alias = "topics")]
    picks: Vec<u32>,
}

#[derive(Deserialize)]
struct SessionSummary {
    summary: String,
    #[serde(default)]
    outcome: Option<String>,
}

#[derive(Deserialize)]
struct Related {
    #[serde(default)]
    related: Vec<String>,
}

fn parse_as<T: for<'de> Deserialize<'de>>(text: &str) -> Result<T, String> {
    serde_json::from_str(extract_json(text).ok_or("no JSON object in the reply")?)
        .map_err(|e| e.to_string())
}

fn numbered<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    items
        .into_iter()
        .enumerate()
        .map(|(i, t)| format!("[{}] {t}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn view_line(v: &EventView<'_>) -> String {
    let date = v.event.updated_at.date_naive();
    let session = if v.session.summary.is_empty() {
        String::new()
    } else {
        format!(" (session: {})", v.session.summary)
    };
    format!(
        "{date} · {}/{} · {}{session}",
        v.query_type, v.topic, v.event.summary
    )
}

fn hit(v: &EventView<'_>) -> RecallHit {
    RecallHit {
        event_id: v.event.id,
        summary: v.event.summary.clone(),
        session_id: v.session.session_id,
        date: v.event.updated_at.date_naive().to_string(),
        topic: format!("{}/{}", v.query_type, v.topic),
    }
}

/// Fallback ranking: query words found in the event, its topic and its session summary.
fn keyword_rank<'a>(query: &str, pool: &[EventView<'a>]) -> Vec<EventView<'a>> {
    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(str::to_lowercase)
        .collect();
    let mut scored: Vec<(usize, EventView<'a>)> = pool
        .iter()
        .map(|v| {
            let hay =
                format!("{} {} {}", v.event.summary, v.topic, v.session.summary).to_lowercase();
            (
                words.iter().filter(|w| hay.contains(w.as_str())).count(),
                *v,
            )
        })
        .filter(|(score, _)| *score > 0)
        .collect();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.1.event.updated_at.cmp(&a.1.event.updated_at))
    });
    scored.into_iter().map(|(_, v)| v).collect()
}

impl MemoryAgent {
    pub fn new(deps: MemoryDeps) -> Arc<Self> {
        Arc::new(Self {
            deps,
            queue: Mutex::new(BTreeMap::new()),
            seq: AtomicU64::new(0),
            busy: AtomicBool::new(false),
            notify: Notify::new(),
            done: AtomicU64::new(0),
        })
    }

    pub fn spawn(self: &Arc<Self>, stop: CancellationToken) {
        let me = self.clone();
        tokio::spawn(async move { me.run(stop).await });
    }

    pub fn submit(&self, job: MemoryJob) {
        let key = (job.priority(), self.seq.fetch_add(1, Ordering::Relaxed));
        if let Ok(mut q) = self.queue.lock() {
            q.insert(key, job);
        }
        self.notify.notify_one();
    }

    pub async fn recall(&self, query: &str) -> Vec<RecallHit> {
        let (reply, rx) = oneshot::channel();
        self.submit(MemoryJob::Recall {
            query: query.to_owned(),
            reply,
        });
        rx.await.unwrap_or_default()
    }

    pub async fn expand(&self, event: EventId) -> Result<String, String> {
        let (reply, rx) = oneshot::channel();
        self.submit(MemoryJob::Expand { event, reply });
        rx.await
            .unwrap_or_else(|_| Err("the memory agent stopped".into()))
    }

    /// Jobs finished so far (for tests and diagnostics).
    pub fn completed(&self) -> u64 {
        self.done.load(Ordering::SeqCst)
    }

    /// Waits until the queue is empty and no job runs. `false` on timeout.
    pub async fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let empty = self.queue.lock().map_or(true, |q| q.is_empty());
            if empty && !self.busy.load(Ordering::SeqCst) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn run(self: Arc<Self>, stop: CancellationToken) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let job = self.queue.lock().ok().and_then(|mut q| {
                let job = q.pop_first().map(|(_, j)| j);
                // Set while the queue is locked so `wait_idle` never sees an empty, idle agent mid-handoff.
                self.busy.store(job.is_some(), Ordering::SeqCst);
                job
            });
            match job {
                Some(job) => {
                    let name = job.name();
                    let result = match JobScratch::create(&self.deps.data) {
                        Ok(mut scratch) => self.run_job(job, &mut scratch).await,
                        Err(e) => Err(format!("cannot create the scratch file: {e}")),
                    };
                    if let Err(e) = result {
                        tracing::warn!(job = name, error = %e, "memory job failed");
                    }
                    self.done.fetch_add(1, Ordering::SeqCst);
                    self.busy.store(false, Ordering::SeqCst);
                }
                None => tokio::select! {
                    () = notified => {}
                    () = stop.cancelled() => return,
                },
            }
        }
    }

    async fn run_job(&self, job: MemoryJob, scratch: &mut JobScratch) -> Result<(), String> {
        match job {
            MemoryJob::SessionStart { session, upto }
            | MemoryJob::UserExchanges { session, upto } => {
                self.sync_main(&session, upto, scratch).await
            }
            MemoryJob::SubAgentExchanges {
                session,
                thread,
                role,
                upto,
                ended,
            } => {
                self.sync_thread(&session, thread, role, upto, ended, scratch)
                    .await
            }
            MemoryJob::SessionEnd { session } => self.end_session(&session, scratch).await,
            MemoryJob::Recall { query, reply } => {
                let hits = self.do_recall(&query, scratch).await;
                let _ = reply.send(hits);
                Ok(())
            }
            MemoryJob::Expand { event, reply } => {
                let _ = reply.send(self.do_expand(event).await);
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------------ LLM helpers

    async fn ask(
        &self,
        scratch: &mut JobScratch,
        user: &str,
        schema: &Value,
    ) -> Result<String, String> {
        let _ = scratch.append("request", user);
        let target = self.deps.models.resolve(Role::Memory);
        let req = ChatRequest {
            provider: Some(target.provider),
            model: target.model,
            messages: vec![Message::system(SYSTEM), Message::user(user)],
            think: ThinkLevel::Off,
            response_format: Some(schema.clone()),
            num_ctx: Some(self.deps.config.provider.ollama.num_ctx),
            temperature: Some(0.0),
            ..ChatRequest::default()
        };
        let stream = self
            .deps
            .provider
            .chat(req, CancellationToken::new())
            .await
            .map_err(|e| e.to_string())?;
        let text = collect(stream).await.map_err(|e| e.to_string())?.text;
        let _ = scratch.append("response", &text);
        Ok(text)
    }

    /// Asks, validates, and asks once more with the error when the answer is invalid.
    async fn ask_parsed<T>(
        &self,
        scratch: &mut JobScratch,
        user: String,
        schema: Value,
        parse: impl Fn(&str) -> Result<T, String>,
    ) -> Option<T> {
        let mut prompt = user;
        for _ in 0..2 {
            let text = match self.ask(scratch, &prompt, &schema).await {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "memory model request failed");
                    return None;
                }
            };
            match parse(&text) {
                Ok(v) => return Some(v),
                Err(e) => {
                    prompt = format!(
                        "{prompt}\n\nYour previous answer was invalid ({e}). Reply again with JSON that matches the schema."
                    );
                }
            }
        }
        None
    }

    // ------------------------------------------------------------------ sessions

    async fn read_session(&self, s: &SessionCtx) -> Result<LogContents, String> {
        if let Some(log) = &s.log {
            let _ = log.flush().await;
        }
        let path = self.deps.data.sessions_dir().join(s.id.log_file_name());
        tokio::task::spawn_blocking(move || read_log(&path))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }

    fn ref_info(s: &SessionCtx) -> RefInfo {
        RefInfo {
            session_id: s.id,
            log_file: s.id.log_file_name(),
            project_id: s.project_id.clone(),
            date: chrono::Local::now().date_naive(),
        }
    }

    /// Files the first `upto` Orchestrator exchanges that no event covers yet.
    async fn sync_main(
        &self,
        s: &SessionCtx,
        upto: u32,
        scratch: &mut JobScratch,
    ) -> Result<(), String> {
        let store = EventStore::main(&self.deps.data);
        let contents = self.read_session(s).await?;
        let file = store.load().map_err(|e| e.to_string())?;
        let covered = file.covered_seq(&s.id, None);
        let spans: Vec<Span> = main_spans(&contents)
            .into_iter()
            .take(upto as usize)
            .filter(|sp| covered.is_none_or(|c| sp.seq.0 > c))
            .collect();
        if spans.is_empty() {
            return Ok(());
        }
        let info = Self::ref_info(s);
        let (qt, topic, rest) = match file.first_pair(&s.id) {
            Some((qt, topic)) => (qt, topic, &spans[..]),
            None => {
                let c = self.classify(&file, s, &spans[0], scratch).await;
                let place = Placement {
                    info: &info,
                    query_type: &c.query_type,
                    topic: &c.topic,
                    thread: None,
                    now: Utc::now(),
                };
                let first = Decision::Extend {
                    summary: Some(c.event_summary.clone()),
                };
                store
                    .update(|f| {
                        let r = f.ensure_ref(&c.query_type, &c.topic, &info);
                        if r.summary.is_empty() {
                            r.summary = clip(&c.session_summary, MAX_SUMMARY_CHARS);
                        }
                        apply(f, &place, &spans[..1], &first)
                    })
                    .map_err(|e| e.to_string())?;
                (c.query_type, c.topic, &spans[1..])
            }
        };
        if !rest.is_empty() {
            self.decide(&store, &info, &qt, &topic, None, rest, scratch)
                .await?;
        }
        Ok(())
    }

    async fn classify(
        &self,
        file: &EventsFile,
        s: &SessionCtx,
        first: &Span,
        scratch: &mut JobScratch,
    ) -> Classification {
        let types: Vec<&str> = file.types.iter().map(|t| t.name.as_str()).collect();
        let topics: Vec<String> = file
            .topic_counts()
            .iter()
            .map(|(t, p, _)| format!("{t} › {p}"))
            .collect();
        let hint = s
            .hint
            .as_ref()
            .map(|(t, p)| format!("Triage suggested: query type `{t}`, topic `{p}`.\n"))
            .unwrap_or_default();
        let user = format!(
            "Task: file a new session in the memory tree.\nExisting query types: {}\nExisting topics: {}\n{hint}\
             First exchange:\n{}\n\nReply with JSON {{\"query_type\", \"topic\", \"session_summary\", \"event_summary\"}}.",
            types.join(", "),
            if topics.is_empty() {
                "(none yet)".into()
            } else {
                topics.join("; ")
            },
            wrap_untrusted("session-transcript", &first.text)
        );
        let schema = json!({
            "type": "object",
            "properties": {
                "query_type": { "type": "string" },
                "topic": { "type": "string" },
                "session_summary": { "type": "string" },
                "event_summary": { "type": "string" }
            },
            "required": ["query_type", "topic", "session_summary", "event_summary"]
        });
        let parsed = self
            .ask_parsed(scratch, user, schema, |t| {
                let mut c: Classification = parse_as(t)?;
                c.query_type = normalize_name(&c.query_type);
                c.topic = normalize_name(&c.topic);
                if c.query_type.is_empty() || c.topic.is_empty() {
                    return Err("query_type and topic must not be empty".into());
                }
                Ok(c)
            })
            .await;
        let mut c = parsed.unwrap_or_else(|| {
            let (qt, topic) = s
                .hint
                .clone()
                .unwrap_or_else(|| ("conversation".into(), "general".into()));
            Classification {
                query_type: normalize_name(&qt),
                topic: normalize_name(&topic),
                session_summary: String::new(),
                event_summary: String::new(),
            }
        });
        let fallback = heuristic_summary(&first.headline);
        if c.event_summary.trim().is_empty() {
            c.event_summary.clone_from(&fallback);
        }
        if c.session_summary.trim().is_empty() {
            c.session_summary = fallback;
        }
        c
    }

    /// Asks how `spans` relate to the open event, applies the answer (or the fallback) and links closed events.
    #[allow(clippy::too_many_arguments)]
    async fn decide(
        &self,
        store: &EventStore,
        info: &RefInfo,
        qt: &str,
        topic: &str,
        thread: Option<AgentId>,
        spans: &[Span],
        scratch: &mut JobScratch,
    ) -> Result<(), String> {
        let file = store.load().map_err(|e| e.to_string())?;
        let open = file
            .open_event(&info.session_id, thread)
            .and_then(|id| file.event(id))
            .map(|v| v.event.summary.clone());
        let user = format!(
            "Task: decide how new exchanges fit this session's events.\nOpen event: {}\nNew exchanges:\n{}\n\n\
             Reply with JSON:\n\
             - {{\"action\":\"extend\",\"summary\":\"...\"}} when they continue the open event's subject (summary optional, to improve it);\n\
             - {{\"action\":\"close_and_new\",\"summary\":\"...\",\"outcome\":\"...\"}} when the subject changed (outcome of the closed event, optional);\n\
             - {{\"action\":\"split\",\"events\":[{{\"summary\":\"...\",\"exchanges\":[1,2]}}]}} when they cover several subjects; \
             every exchange number appears once, in order.",
            open.as_deref().unwrap_or("(none)"),
            wrap_untrusted(
                "session-transcript",
                &numbered(spans.iter().map(|s| s.text.as_str()))
            )
        );
        let n = spans.len();
        let decision = self
            .ask_parsed(scratch, user, decision::schema(), |t| decision::parse(t, n))
            .await
            .unwrap_or_else(|| Decision::fallback(spans, open.is_some()));
        let place = Placement {
            info,
            query_type: qt,
            topic,
            thread,
            now: Utc::now(),
        };
        let applied = store
            .update(|f| apply(f, &place, spans, &decision))
            .map_err(|e| e.to_string())?;
        self.link(store, &applied.closed, scratch).await;
        Ok(())
    }

    /// Links each closed event to similar events the model picks from same-topic candidates, and hands it to the Learner.
    async fn link(&self, store: &EventStore, events: &[EventId], scratch: &mut JobScratch) {
        if let Some(learner) = &self.deps.learner {
            for id in events {
                learner.submit(LearnJob {
                    store: store.clone(),
                    event: *id,
                });
            }
        }
        for id in events {
            let Ok(file) = store.load() else { return };
            let Some(target) = file.event(*id) else {
                continue;
            };
            let candidates = file.candidates(*id, LINK_CANDIDATES);
            if candidates.is_empty() {
                continue;
            }
            let ids: HashSet<EventId> = candidates.iter().map(|v| v.event.id).collect();
            let list: Vec<String> = candidates
                .iter()
                .map(|v| format!("{} · {}", v.event.id, view_line(v)))
                .collect();
            let user = format!(
                "Task: find earlier events about the same issue or subject as this one.\nEvent: {}\nCandidates:\n{}\n\n\
                 Reply with JSON {{\"related\": [\"<event id>\", ...]}} with only candidates that are clearly about the same thing. \
                 An empty list is fine.",
                target.event.summary,
                list.join("\n")
            );
            let schema = json!({
                "type": "object",
                "properties": { "related": { "type": "array", "items": { "type": "string" } } },
                "required": ["related"]
            });
            let picked = self
                .ask_parsed(scratch, user, schema, |t| {
                    let r: Related = parse_as(t)?;
                    Ok(r.related
                        .iter()
                        .filter_map(|s| s.trim().parse::<EventId>().ok())
                        .filter(|e| ids.contains(e))
                        .take(MAX_RELATED)
                        .collect::<Vec<_>>())
                })
                .await
                .unwrap_or_default();
            if !picked.is_empty()
                && let Err(e) = store.update(|f| {
                    for other in &picked {
                        f.link(*id, *other);
                    }
                })
            {
                tracing::warn!(error = %e, "cannot save event links");
            }
        }
    }

    async fn sync_thread(
        &self,
        s: &SessionCtx,
        thread: AgentId,
        role: Role,
        upto: u32,
        ended: bool,
        scratch: &mut JobScratch,
    ) -> Result<(), String> {
        let store = EventStore::for_role(&self.deps.data, role);
        let contents = self.read_session(s).await?;
        let file = store.load().map_err(|e| e.to_string())?;
        let covered = file.covered_seq(&s.id, Some(thread));
        let spans: Vec<Span> = thread_spans(&contents, thread)
            .into_iter()
            .take(upto as usize)
            .filter(|sp| covered.is_none_or(|c| sp.seq.0 > c))
            .collect();
        let (qt, topic) = EventStore::main(&self.deps.data)
            .load()
            .ok()
            .and_then(|f| f.first_pair(&s.id))
            .or_else(|| s.hint.clone())
            .unwrap_or_else(|| ("command".into(), "general".into()));
        let info = Self::ref_info(s);
        if !spans.is_empty() {
            self.decide(&store, &info, &qt, &topic, Some(thread), &spans, scratch)
                .await?;
        }
        if ended {
            let open = store
                .load()
                .map_err(|e| e.to_string())?
                .open_event(&s.id, Some(thread));
            if open.is_some() {
                let closed = store
                    .update(|f| f.close_open(&s.id, Some(Some(thread)), Utc::now()))
                    .map_err(|e| e.to_string())?;
                self.link(&store, &closed, scratch).await;
            }
        }
        Ok(())
    }

    /// Files the remaining exchanges, then summarizes and closes the session.
    async fn end_session(&self, s: &SessionCtx, scratch: &mut JobScratch) -> Result<(), String> {
        self.sync_main(s, u32::MAX, scratch).await?;
        let store = EventStore::main(&self.deps.data);
        let file = store.load().map_err(|e| e.to_string())?;
        let refs = file.refs(&s.id);
        if refs.is_empty() {
            return Ok(());
        }
        let mut events: Vec<_> = refs.iter().flat_map(|(_, _, r)| r.events.iter()).collect();
        events.sort_by_key(|e| e.turn_range.0);
        let fallback = refs
            .iter()
            .map(|(_, _, r)| r.summary.as_str())
            .find(|s| !s.is_empty())
            .or_else(|| events.first().map(|e| e.summary.as_str()))
            .unwrap_or("")
            .to_owned();
        let user = format!(
            "Task: summarize a finished session.\nEvents in order:\n{}\n\n\
             Reply with JSON {{\"summary\": \"...\", \"outcome\": \"...\"}}. The summary says in one line what the session \
             was about and what it achieved; outcome is optional.",
            events
                .iter()
                .map(|e| format!("- {}", e.summary))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let schema = json!({
            "type": "object",
            "properties": { "summary": { "type": "string" }, "outcome": { "type": "string" } },
            "required": ["summary"]
        });
        let parsed = self
            .ask_parsed(scratch, user, schema, |t| {
                let s: SessionSummary = parse_as(t)?;
                if s.summary.trim().is_empty() {
                    return Err("summary must not be empty".into());
                }
                Ok(s)
            })
            .await;
        let (summary, outcome) = match parsed {
            Some(p) => (
                clip(&p.summary, MAX_SUMMARY_CHARS),
                p.outcome
                    .filter(|o| !o.trim().is_empty())
                    .map(|o| clip(&o, MAX_SUMMARY_CHARS)),
            ),
            None => (fallback, None),
        };
        let today = chrono::Local::now().date_naive();
        let now = Utc::now();
        let closed = store
            .update(|f| {
                for r in f.refs_mut(&s.id) {
                    r.summary.clone_from(&summary);
                    r.outcome.clone_from(&outcome);
                    if !r.dates.contains(&today) {
                        r.dates.push(today);
                    }
                    r.finalized = true;
                }
                f.close_open(&s.id, None, now)
            })
            .map_err(|e| e.to_string())?;
        self.link(&store, &closed, scratch).await;

        for role in [Role::Worker, Role::Research] {
            let role_store = EventStore::for_role(&self.deps.data, role);
            let has_open = role_store.load().is_ok_and(|f| {
                f.refs(&s.id).iter().any(|(_, _, r)| {
                    r.events
                        .iter()
                        .any(|e| e.status == finality_memory::EventStatus::Open)
                })
            });
            if has_open && let Err(e) = role_store.update(|f| f.close_open(&s.id, None, now)) {
                tracing::warn!(error = %e, "cannot close role events");
            }
        }

        let index = SessionIndex::new(
            self.deps.data.session_index_file(),
            self.deps.data.locks_dir(),
        );
        if !summary.is_empty()
            && let Err(e) = index.update(&s.id, |entry| entry.title.clone_from(&summary))
        {
            tracing::warn!(error = %e, "cannot set the session title");
        }
        Ok(())
    }

    // ------------------------------------------------------------------ recall

    async fn do_recall(&self, query: &str, scratch: &mut JobScratch) -> Vec<RecallHit> {
        let Ok(file) = EventStore::main(&self.deps.data).load() else {
            return Vec::new();
        };
        let all = file.events();
        if all.is_empty() || query.trim().is_empty() {
            return Vec::new();
        }
        let mut pool = all;
        if pool.len() > RECALL_POOL {
            let topics = file.topic_counts();
            let list: Vec<String> = topics
                .iter()
                .map(|(t, p, n)| format!("{t} › {p} ({n} sessions)"))
                .collect();
            let user = format!(
                "Task: pick the topics most likely to hold memories relevant to the query.\nQuery: {query}\nTopics:\n{}\n\n\
                 Reply with JSON {{\"topics\": [<topic numbers>]}}.",
                numbered(list.iter().map(String::as_str))
            );
            let schema = json!({
                "type": "object",
                "properties": { "topics": { "type": "array", "items": { "type": "integer" } } },
                "required": ["topics"]
            });
            let picked: HashSet<(String, String)> = self
                .ask_parsed(scratch, user, schema, |t| {
                    Ok(parse_as::<Indices>(t)?
                        .picks
                        .iter()
                        .filter_map(|i| topics.get((*i as usize).checked_sub(1)?))
                        .map(|(t, p, _)| ((*t).to_owned(), (*p).to_owned()))
                        .collect())
                })
                .await
                .unwrap_or_default();
            if picked.is_empty() {
                let ranked = keyword_rank(query, &pool);
                if !ranked.is_empty() {
                    pool = ranked;
                }
            } else {
                pool.retain(|v| picked.contains(&(v.query_type.to_owned(), v.topic.to_owned())));
            }
            pool.sort_by_key(|v| std::cmp::Reverse(v.event.updated_at));
            pool.truncate(RECALL_POOL);
        }
        let lines: Vec<String> = pool.iter().map(view_line).collect();
        let user = format!(
            "Task: pick the memories relevant to the query, most relevant first (at most {RECALL_HITS}).\nQuery: {query}\n\
             Memories:\n{}\n\nReply with JSON {{\"events\": [<memory numbers>]}}. An empty list is fine.",
            numbered(lines.iter().map(String::as_str))
        );
        let schema = json!({
            "type": "object",
            "properties": { "events": { "type": "array", "items": { "type": "integer" } } },
            "required": ["events"]
        });
        let n = pool.len();
        let picked = self
            .ask_parsed(scratch, user, schema, |t| {
                let mut seen = HashSet::new();
                Ok(parse_as::<Indices>(t)?
                    .picks
                    .into_iter()
                    .filter_map(|i| (i as usize).checked_sub(1).filter(|i| *i < n))
                    .filter(|i| seen.insert(*i))
                    .take(RECALL_HITS)
                    .collect::<Vec<usize>>())
            })
            .await;
        match picked {
            Some(idx) => idx.into_iter().map(|i| hit(&pool[i])).collect(),
            None => keyword_rank(query, &pool)
                .iter()
                .take(RECALL_HITS)
                .map(hit)
                .collect(),
        }
    }

    async fn do_expand(&self, id: EventId) -> Result<String, String> {
        let file = EventStore::main(&self.deps.data)
            .load()
            .map_err(|e| e.to_string())?;
        let v = file
            .event(id)
            .ok_or_else(|| format!("no memory event {id}"))?;
        let path = self.deps.data.sessions_dir().join(&v.session.log_file);
        let contents = tokio::task::spawn_blocking(move || read_log(&path))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("cannot read the session log: {e}"))?;
        let children = spawned_agents(&contents.records);
        let (start, end) = v.event.turn_range;
        let records = contents
            .range(start, end)
            .iter()
            .filter(|r| match v.event.thread {
                Some(t) => r.agent == t,
                None => !children.contains(&r.agent),
            });
        let dates: Vec<String> = v.session.dates.iter().map(ToString::to_string).collect();
        Ok(format!(
            "Event {id} ({}/{}): {}\nSession {} ({})\n\n{}",
            v.query_type,
            v.topic,
            v.event.summary,
            v.session.session_id,
            dates.join(", "),
            condense(records, EXPAND_CHARS)
        ))
    }
}
