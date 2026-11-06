//! The Scorer (DESIGN §8.1): turns finished plans into statistics in `lookup/plan_lookup.json`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use h_agent_core::config::Config;
use h_agent_core::{Role, TaskStatus, ThinkLevel};
use h_agent_llm::{ChatRequest, LlmProvider, Message, ModelSelector, collect};
use h_agent_memory::decision::extract_json;
use h_agent_memory::lookup::{
    LookupEntry, LookupStep, LookupStore, PlanRun, RunOutcome, fingerprint, normalize_signature,
};
use h_agent_memory::tree::{MAX_SUMMARY_CHARS, clip};
use h_agent_store::{DataDir, write_atomic};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::plan::{Outcome, Plan};
use crate::subagent::SUBAGENT_TOOLS;

const SYSTEM: &str = include_str!("../prompts/scorer.md");

/// `plans/<session>/<plan>.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanFile {
    pub session_id: String,
    pub outcome: String,
    pub duration_s: u64,
    pub retries: u32,
    /// Set once the Scorer has recorded it.
    #[serde(default)]
    pub scored: bool,
    pub plan: Plan,
}

/// Success when every task succeeded; Failed when none did; Partial otherwise. Tasks removed before running are ignored.
pub fn plan_outcome(plan: &Plan) -> RunOutcome {
    let tasks: Vec<_> = plan
        .tasks
        .iter()
        .filter(|t| !(t.status == TaskStatus::Cancelled && t.attempts.is_empty()))
        .collect();
    let any_good = tasks
        .iter()
        .any(|t| matches!(t.status, TaskStatus::Success | TaskStatus::Partial));
    if !tasks.is_empty() && tasks.iter().all(|t| t.status == TaskStatus::Success) {
        RunOutcome::Success
    } else if any_good {
        RunOutcome::Partial
    } else {
        RunOutcome::Failed
    }
}

/// The plan's strategy: step titles, roles, the tools the final attempt used, skills and dependencies.
pub fn lookup_steps(plan: &Plan) -> Vec<LookupStep> {
    plan.tasks
        .iter()
        .filter(|t| !(t.status == TaskStatus::Cancelled && t.attempts.is_empty()))
        .map(|t| {
            let mut tools: Vec<String> = t
                .attempts
                .iter()
                .rev()
                .find(|a| a.outcome.is_some_and(|o| o != Outcome::Failed))
                .or(t.last_attempt())
                .map(|a| a.tools_used.clone())
                .unwrap_or_default();
            tools.retain(|n| !SUBAGENT_TOOLS.contains(&n.as_str()));
            tools.sort();
            tools.dedup();
            LookupStep {
                title: t.spec.title.clone(),
                role: t.spec.role.as_str().to_owned(),
                tools,
                skills: t.spec.skills.clone(),
                depends_on: t.spec.depends_on.clone(),
            }
        })
        .collect()
}

pub fn outcome_name(o: RunOutcome) -> &'static str {
    match o {
        RunOutcome::Success => "success",
        RunOutcome::Partial => "partial",
        RunOutcome::Failed => "failed",
    }
}

fn parse_outcome(s: &str) -> RunOutcome {
    match s {
        "success" => RunOutcome::Success,
        "partial" => RunOutcome::Partial,
        _ => RunOutcome::Failed,
    }
}

#[derive(Deserialize)]
struct SameAs {
    #[serde(default)]
    same_as: u32,
}

pub struct ScorerDeps {
    pub config: Config,
    pub data: DataDir,
    /// Already gated with `Role::Scorer`, so it never runs ahead of interactive agents.
    pub provider: Arc<dyn LlmProvider>,
    pub models: Arc<ModelSelector>,
}

pub struct Scorer {
    deps: ScorerDeps,
    queue: Mutex<VecDeque<PathBuf>>,
    busy: AtomicBool,
    notify: Notify,
    done: AtomicU64,
}

impl Scorer {
    pub fn new(deps: ScorerDeps) -> Arc<Self> {
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

    /// Queues a plan file for scoring.
    pub fn submit(&self, path: PathBuf) {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(path);
        }
        self.notify.notify_one();
    }

    /// Queues plan files that were saved but never scored (e.g. the process quit first).
    pub fn catch_up(&self) {
        let Ok(sessions) = std::fs::read_dir(self.deps.data.plans_dir()) else {
            return;
        };
        let mut pending: Vec<PathBuf> = sessions
            .flatten()
            .filter_map(|d| std::fs::read_dir(d.path()).ok())
            .flatten()
            .flatten()
            .map(|f| f.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .filter(|p| {
                std::fs::read_to_string(p)
                    .ok()
                    .and_then(|t| serde_json::from_str::<PlanFile>(&t).ok())
                    .is_some_and(|f| !f.scored)
            })
            .collect();
        pending.sort();
        for p in pending {
            self.submit(p);
        }
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
                let p = q.pop_front();
                self.busy.store(p.is_some(), Ordering::SeqCst);
                p
            });
            match next {
                Some(path) => {
                    if let Err(e) = self.score(&path).await {
                        tracing::warn!(plan = %path.display(), error = %e, "cannot score plan");
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

    async fn score(&self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut file: PlanFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if file.scored {
            return Ok(());
        }
        let plan = &file.plan;
        let steps = lookup_steps(plan);
        let fp = fingerprint(&steps);
        let qt = if plan.query_type.is_empty() {
            "conversation"
        } else {
            &plan.query_type
        };
        let topic = if plan.topic.is_empty() {
            "general"
        } else {
            &plan.topic
        };
        let signature = clip(&plan.goal, MAX_SUMMARY_CHARS);
        let run = PlanRun {
            plan_id: plan.id.to_string(),
            outcome: parse_outcome(&file.outcome),
            duration_s: file.duration_s,
            retries: file.retries,
        };

        let store = LookupStore::new(&self.deps.data);
        let table = store.load().map_err(|e| e.to_string())?;
        let credited = plan
            .template_id
            .as_deref()
            .filter(|id| table.entry(id).is_some_and(|e| fingerprint(&e.steps) == fp))
            .map(str::to_owned);
        let target = match credited {
            Some(id) => Some(id),
            None => {
                let candidates: Vec<&LookupEntry> = table
                    .entries
                    .iter()
                    .filter(|e| {
                        e.query_type == qt && e.topic == topic && fingerprint(&e.steps) == fp
                    })
                    .collect();
                let wanted = normalize_signature(&signature);
                match candidates
                    .iter()
                    .find(|e| normalize_signature(&e.event_signature) == wanted)
                {
                    Some(e) => Some(e.id.clone()),
                    None if candidates.is_empty() => None,
                    None => self.same_request(&signature, &candidates).await,
                }
            }
        };

        let cfg = &self.deps.config;
        let now = Utc::now();
        store
            .update(|t| {
                match target.as_deref().and_then(|id| t.entry_mut(id)) {
                    Some(e) => e.record(&run, now),
                    None => {
                        let mut e = LookupEntry::new(&signature, qt, topic, steps, now);
                        e.record(&run, now);
                        t.entries.push(e);
                    }
                }
                t.rescore(&cfg.scorer.weights, cfg.retry.max_retries);
            })
            .map_err(|e| e.to_string())?;

        file.scored = true;
        let bytes = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
        write_atomic(path, &bytes).map_err(|e| e.to_string())
    }

    /// Asks whether `signature` is the same kind of request as one of `candidates` (which share its strategy).
    async fn same_request(&self, signature: &str, candidates: &[&LookupEntry]) -> Option<String> {
        let list: Vec<String> = candidates
            .iter()
            .enumerate()
            .map(|(i, e)| format!("[{}] {}", i + 1, e.event_signature))
            .collect();
        let user = format!(
            "Task: is this finished plan's request the same kind of request as one of these entries?\n\
             Request: {signature}\nEntries:\n{}\n\nReply with JSON {{\"same_as\": <entry number, or 0 if none>}}.",
            list.join("\n")
        );
        let target = self.deps.models.resolve(Role::Scorer);
        let req = ChatRequest {
            provider: Some(target.provider),
            model: target.model,
            messages: vec![Message::system(SYSTEM), Message::user(user)],
            think: ThinkLevel::Off,
            response_format: Some(json!({
                "type": "object",
                "properties": { "same_as": { "type": "integer" } },
                "required": ["same_as"]
            })),
            num_ctx: Some(self.deps.config.provider.ollama.num_ctx),
            temperature: Some(0.0),
            ..ChatRequest::default()
        };
        let text = match self.deps.provider.chat(req, CancellationToken::new()).await {
            Ok(stream) => collect(stream).await.ok()?.text,
            Err(e) => {
                tracing::warn!(error = %e, "scorer model request failed");
                return None;
            }
        };
        let n = serde_json::from_str::<SameAs>(extract_json(&text)?)
            .ok()?
            .same_as as usize;
        candidates.get(n.checked_sub(1)?).map(|e| e.id.clone())
    }
}
