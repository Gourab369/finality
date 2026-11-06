//! The Learner (DESIGN §8.2): mines closed events for skills, knowledgepedia facts and user preferences.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use h_agent_core::config::Config;
use h_agent_core::secrets::contains_secret;
use h_agent_core::{EventId, Role, ThinkLevel};
use h_agent_llm::{ChatRequest, LlmProvider, Message, ModelSelector, collect};
use h_agent_memory::decision::extract_json;
use h_agent_memory::knowledge::{Contribution, Knowledgepedia};
use h_agent_memory::skills::{SkillDraft, SkillLibrary};
use h_agent_memory::transcript::{condense, spawned_agents};
use h_agent_memory::{EventStatus, EventStore, profile};
use h_agent_store::{DataDir, read_log};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::conversation::wrap_untrusted;

const SYSTEM: &str = include_str!("../prompts/learner.md");
const TRANSCRIPT_CHARS: usize = 8000;

/// A closed event to learn from, in the events file that holds it.
#[derive(Debug, Clone)]
pub struct LearnJob {
    pub store: EventStore,
    pub event: EventId,
}

#[derive(Debug, Default, Deserialize)]
struct Lesson {
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    skill: Option<SkillOut>,
    #[serde(default)]
    facts: Vec<FactOut>,
    #[serde(default)]
    profile: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SkillOut {
    name: String,
    description: String,
    #[serde(default)]
    allowed_tools: Vec<String>,
    body: String,
}

#[derive(Debug, Deserialize)]
struct FactOut {
    topic: String,
    #[serde(default)]
    aliases: Vec<String>,
    fact: String,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    confidence: String,
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "outcome": { "type": "string", "enum": ["success", "partial", "failed"] },
            "skill": {
                "type": ["object", "null"],
                "properties": {
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "allowed_tools": { "type": "array", "items": { "type": "string" } },
                    "body": { "type": "string" }
                },
                "required": ["name", "description", "body"]
            },
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string" },
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "fact": { "type": "string" },
                        "sources": { "type": "array", "items": { "type": "string" } },
                        "confidence": { "type": "string", "enum": ["low", "medium", "high"] }
                    },
                    "required": ["topic", "fact"]
                }
            },
            "profile": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["outcome"]
    })
}

fn parse(text: &str) -> Result<Lesson, String> {
    let lesson: Lesson =
        serde_json::from_str(extract_json(text).ok_or("no JSON object in the reply")?)
            .map_err(|e| e.to_string())?;
    if !matches!(lesson.outcome.as_str(), "success" | "partial" | "failed") {
        return Err("outcome must be success, partial or failed".into());
    }
    Ok(lesson)
}

pub struct LearnerDeps {
    pub config: Config,
    pub data: DataDir,
    /// Already gated with `Role::Learner`, the lowest priority.
    pub provider: Arc<dyn LlmProvider>,
    pub models: Arc<ModelSelector>,
}

pub struct Learner {
    deps: LearnerDeps,
    queue: Mutex<VecDeque<LearnJob>>,
    busy: AtomicBool,
    notify: Notify,
    done: AtomicU64,
}

impl Learner {
    pub fn new(deps: LearnerDeps) -> Arc<Self> {
        Arc::new(Self {
            deps,
            queue: Mutex::new(VecDeque::new()),
            busy: AtomicBool::new(false),
            notify: Notify::new(),
            done: AtomicU64::new(0),
        })
    }

    pub fn spawn(self: &Arc<Self>, stop: CancellationToken) {
        let me = self.clone();
        tokio::spawn(async move { me.run(stop).await });
    }

    pub fn submit(&self, job: LearnJob) {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(job);
        }
        self.notify.notify_one();
    }

    pub fn completed(&self) -> u64 {
        self.done.load(Ordering::SeqCst)
    }

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
            let next = self.queue.lock().ok().and_then(|mut q| {
                let j = q.pop_front();
                self.busy.store(j.is_some(), Ordering::SeqCst);
                j
            });
            match next {
                Some(job) => {
                    if let Err(e) = self.learn(&job).await {
                        tracing::warn!(event = %job.event, error = %e, "learner job failed");
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

    async fn ask(&self, user: String) -> Option<Lesson> {
        let mut prompt = user;
        for _ in 0..2 {
            let target = self.deps.models.resolve(Role::Learner);
            let req = ChatRequest {
                provider: Some(target.provider),
                model: target.model,
                messages: vec![Message::system(SYSTEM), Message::user(prompt.clone())],
                think: ThinkLevel::Off,
                response_format: Some(schema()),
                num_ctx: Some(self.deps.config.provider.ollama.num_ctx),
                temperature: Some(0.0),
                ..ChatRequest::default()
            };
            let text = match self.deps.provider.chat(req, CancellationToken::new()).await {
                Ok(stream) => collect(stream).await.ok()?.text,
                Err(e) => {
                    tracing::warn!(error = %e, "learner model request failed");
                    return None;
                }
            };
            match parse(&text) {
                Ok(l) => return Some(l),
                Err(e) => {
                    prompt = format!(
                        "{prompt}\n\nYour previous answer was invalid ({e}). Reply again with JSON that matches the schema."
                    );
                }
            }
        }
        None
    }

    async fn learn(&self, job: &LearnJob) -> Result<(), String> {
        let file = job.store.load().map_err(|e| e.to_string())?;
        let v = file
            .event(job.event)
            .ok_or_else(|| format!("event {} not found", job.event))?;
        if v.event.status != EventStatus::Closed
            || v.event
                .outcome
                .as_deref()
                .is_some_and(|o| o.to_lowercase().contains("fail"))
        {
            return Ok(());
        }
        let session = v.session.session_id;
        let path = self.deps.data.sessions_dir().join(&v.session.log_file);
        let contents = tokio::task::spawn_blocking(move || read_log(&path))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let children = spawned_agents(&contents.records);
        let (start, end) = v.event.turn_range;
        let transcript = condense(
            contents
                .range(start, end)
                .iter()
                .filter(|r| match v.event.thread {
                    Some(t) => r.agent == t,
                    None => !children.contains(&r.agent),
                }),
            TRANSCRIPT_CHARS,
        );

        let skills = SkillLibrary::new(&self.deps.data);
        let catalog = skills.load();
        let known_skills: Vec<String> = catalog
            .active
            .iter()
            .chain(&catalog.pending)
            .map(|s| format!("{}: {}", s.name, s.description))
            .collect();
        let kb = Knowledgepedia::new(&self.deps.data);
        let topics: Vec<String> = kb
            .index()
            .map(|i| i.topics.into_iter().map(|t| t.topic).collect())
            .unwrap_or_default();
        let user = format!(
            "Task: learn from this finished piece of work.\nEvent: {} (type {}, topic {}{})\nExisting skills: {}\n\
             Known knowledge topics: {}\nTranscript:\n{}\n\n\
             Reply with JSON {{\"outcome\": \"success|partial|failed\", \"skill\": null or {{\"name\", \"description\", \
             \"allowed_tools\", \"body\"}}, \"facts\": [{{\"topic\", \"aliases\", \"fact\", \"sources\", \"confidence\"}}], \
             \"profile\": [\"...\"]}}. Use empty lists and null when there is nothing worth keeping.",
            v.event.summary,
            v.query_type,
            v.topic,
            v.event
                .outcome
                .as_deref()
                .map(|o| format!(", outcome: {o}"))
                .unwrap_or_default(),
            if known_skills.is_empty() {
                "(none)".into()
            } else {
                known_skills.join("; ")
            },
            if topics.is_empty() {
                "(none)".into()
            } else {
                topics.join("; ")
            },
            wrap_untrusted("session-log", &transcript)
        );
        let Some(lesson) = self.ask(user).await else {
            return Ok(());
        };
        if lesson.outcome == "failed" {
            return Ok(());
        }
        let event_id = job.event.to_string();

        if let Some(s) = lesson.skill {
            let draft = SkillDraft {
                name: s.name.trim().to_lowercase(),
                description: s.description,
                allowed_tools: s.allowed_tools,
                body: s.body,
                source_events: vec![event_id.clone()],
            };
            let text = format!("{} {}", draft.description, draft.body);
            if contains_secret(&text) {
                tracing::warn!(skill = %draft.name, "dropped a skill that looks like it contains a secret");
            } else {
                let pending = !self.deps.config.learner.auto_approve_skills;
                match skills.create(&draft, pending) {
                    Ok(dir) => tracing::info!(skill = %dir.display(), "learner wrote a skill"),
                    Err(e) => tracing::info!(error = %e, "learner skipped a skill"),
                }
            }
        }

        let today = chrono::Local::now().date_naive();
        for f in lesson.facts {
            let (topic, fact) = (f.topic.trim(), f.fact.trim());
            if topic.is_empty() || fact.is_empty() || contains_secret(fact) {
                continue;
            }
            let mut sources: Vec<String> = f
                .sources
                .into_iter()
                .filter(|s| !s.trim().is_empty() && !contains_secret(s))
                .collect();
            sources.push(format!("session {session}"));
            let contribution = Contribution {
                topic: topic.to_owned(),
                aliases: f.aliases,
                facts: vec![fact.to_owned()],
                sources,
                confidence: if f.confidence.is_empty() {
                    "medium".into()
                } else {
                    f.confidence
                },
            };
            if let Err(e) = kb.merge(&contribution, today) {
                tracing::warn!(error = %e, "cannot update the knowledgepedia");
            }
        }

        if !lesson.profile.is_empty() {
            let source = format!("event {event_id}");
            let written = if self.deps.config.learner.auto_approve_profile {
                profile::add_facts(&self.deps.data, &lesson.profile, &source, today)
            } else {
                profile::add_pending(&self.deps.data, &lesson.profile, &source, today)
            };
            match written {
                Ok(facts) if !facts.is_empty() => {
                    tracing::info!(count = facts.len(), "learner noted profile facts");
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "cannot update the user profile"),
            }
        }
        Ok(())
    }
}
