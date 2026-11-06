//! Supervisor: owns the plan, the sub-agent pool and retry rules (DESIGN §4.4, §6.1.3–§6.2).
//! Control tools call into it; sub-agent attempts report back to it.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use chrono::Utc;
use finality_core::config::Config;
use finality_core::events::UiEvent;
use finality_core::{AgentId, Role, TaskStatus};
use finality_llm::{LlmProvider, ModelSelector};
use finality_memory::agents::AgentLibrary;
use finality_memory::lookup::{LookupEntry, LookupStore};
use finality_memory::skills::SkillLibrary;
use finality_safety::PermissionEngine;
use finality_store::{DataDir, Record, write_atomic};
use finality_tools::{ToolEnv, ToolExecutor, ToolSets};
use serde::Serialize;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::approver::UiApprover;
use crate::decision::Decider;
use crate::engine::TokenTotals;
use crate::heartbeat::{Alert, AlertKind, HeartbeatBoard, Phase};
use crate::memory::{MemoryAgent, SessionCtx};
use crate::messages::{FromSubAgent, ToSubAgent};
use crate::plan::{Attempt, Outcome, Plan, StepSpec, TaskRole, is_active, normalize_strategy};
use crate::scorer::{PlanFile, Scorer, outcome_name, plan_outcome};
use crate::subagent::{AttemptJob, run_attempt};

pub struct SupervisorDeps {
    pub config: Config,
    /// Gated at sub-agent priority.
    pub provider: Arc<dyn LlmProvider>,
    pub executor: Arc<ToolExecutor>,
    pub tool_env: Arc<ToolEnv>,
    pub ui: mpsc::UnboundedSender<UiEvent>,
    pub board: Arc<HeartbeatBoard>,
    pub approver: Arc<UiApprover>,
    pub totals: Arc<TokenTotals>,
    pub permissions: Arc<Mutex<PermissionEngine>>,
    pub orchestrator: AgentId,
    /// Receives sub-agent threads for `<role>.events`.
    pub memory: Option<Arc<MemoryAgent>>,
    pub skills: SkillLibrary,
    /// Custom agents that plan steps may name (DESIGN §6.5).
    pub agents: AgentLibrary,
    pub tool_sets: Arc<ToolSets>,
    /// Finished plans are saved under `plans/` here.
    pub data: DataDir,
    pub scorer: Option<Arc<Scorer>>,
    /// Advises retry vs give up after failures when enabled.
    pub decider: Option<Arc<Decider>>,
    /// Each role's provider and model.
    pub models: Arc<ModelSelector>,
    /// Direct mode: sub-agent prompts get `prompts/direct.md`.
    pub direct: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub outcome: Outcome,
    pub summary: String,
    pub evidence: Vec<String>,
}

/// How an attempt ended, as seen by its runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptEnd {
    Reported(Report),
    /// Finished without calling report_result; carries the final message.
    NoReport(String),
    Cancelled,
    Error(String),
}

struct Running {
    agent: AgentId,
    label: String,
    role: Role,
    cancel: CancellationToken,
    answer: Option<oneshot::Sender<String>>,
    budget_hit: Arc<AtomicBool>,
}

#[derive(Default)]
struct State {
    plan: Option<Plan>,
    plan_started: Option<Instant>,
    plan_budget_noted: bool,
    running: HashMap<u32, Running>,
    /// Sub-agent → (task number, report).
    links: HashMap<AgentId, (u32, Option<Report>)>,
    notes: Vec<String>,
    max_subagents: u32,
    /// Attempts started per role or custom agent name, for labels.
    counters: HashMap<String, u32>,
    session: Option<SessionCtx>,
    /// (query type, topic) from the current turn's triage.
    turn_context: Option<(String, String)>,
}

pub struct Supervisor {
    pub(crate) deps: SupervisorDeps,
    state: Mutex<State>,
    notify: Notify,
}

fn role_of(r: TaskRole) -> Role {
    match r {
        TaskRole::Research => Role::Research,
        TaskRole::Worker | TaskRole::Orchestrator => Role::Worker,
    }
}

fn outcome_status(o: Outcome) -> TaskStatus {
    match o {
        Outcome::Success => TaskStatus::Success,
        Outcome::Partial => TaskStatus::Partial,
        Outcome::Failed => TaskStatus::Failed,
    }
}

impl Supervisor {
    pub fn new(deps: SupervisorDeps) -> Arc<Self> {
        let max = deps.config.agents.max_subagents;
        Arc::new(Self {
            deps,
            state: Mutex::new(State {
                max_subagents: max,
                ..State::default()
            }),
            notify: Notify::new(),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>, String> {
        self.state
            .lock()
            .map_err(|_| "supervisor state poisoned".to_owned())
    }

    pub fn max_attempts(&self) -> u32 {
        self.deps.config.retry.max_retries + 1
    }

    /// The open session; its log receives plan records and sub-agent turns.
    pub fn set_session(&self, session: Option<SessionCtx>) {
        if let Ok(mut st) = self.state.lock() {
            st.session = session;
        }
    }

    /// Triage classification of the current turn, recorded on plans it creates.
    pub fn set_turn_context(&self, query_type: &str, topic: &str) {
        if let Ok(mut st) = self.state.lock() {
            st.turn_context = Some((query_type.to_owned(), topic.to_owned()));
        }
    }

    /// Saves the plan for the Scorer once it is over: when every task is terminal, or, with `force`,
    /// when it is being replaced or dropped after something ran.
    pub fn finalize(&self, force: bool) {
        if let Ok(mut st) = self.state.lock() {
            self.finalize_locked(&mut st, force);
        }
    }

    fn finalize_locked(&self, st: &mut State, force: bool) {
        let Some(plan) = st.plan.as_mut() else { return };
        let ran = plan
            .tasks
            .iter()
            .any(|t| t.attempts.iter().any(|a| a.outcome.is_some()));
        if plan.finished_at.is_some() || plan.has_active() || !ran || !(force || plan.is_finished())
        {
            return;
        }
        let finished = Utc::now();
        plan.finished_at = Some(finished);
        let Some(session) = &st.session else { return };
        let file = PlanFile {
            session_id: session.id.to_string(),
            outcome: outcome_name(plan_outcome(plan)).to_owned(),
            duration_s: u64::try_from((finished - plan.created_at).num_seconds()).unwrap_or(0),
            retries: plan.tasks.iter().map(|t| t.retries_used()).sum(),
            scored: false,
            plan: plan.clone(),
        };
        let dir = self.deps.data.plans_dir().join(session.id.to_string());
        let path = dir.join(format!("{}.json", plan.id));
        let saved = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|()| serde_json::to_vec_pretty(&file).map_err(|e| e.to_string()))
            .and_then(|bytes| write_atomic(&path, &bytes).map_err(|e| e.to_string()));
        match saved {
            Ok(()) => {
                if let Some(scorer) = &self.deps.scorer {
                    scorer.submit(path);
                }
            }
            Err(e) => tracing::warn!(error = %e, "cannot save the finished plan"),
        }
    }

    pub fn has_active(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|st| st.plan.as_ref().is_some_and(Plan::has_active))
    }

    pub fn digest(&self) -> Option<String> {
        let st = self.state.lock().ok()?;
        st.plan.as_ref().map(|p| p.digest(self.max_attempts()))
    }

    pub fn set_max_subagents(self: &Arc<Self>, n: u32) {
        if let Ok(mut st) = self.state.lock() {
            st.max_subagents = n.max(1);
            self.start_queued(&mut st);
            self.publish(&st);
        }
    }

    // ------------------------------------------------------------------ helpers

    fn record(&self, st: &State, agent: AgentId, body: Record) {
        if let Some(log) = st.session.as_ref().and_then(|s| s.log.as_ref())
            && let Err(e) = log.append(agent, body)
        {
            tracing::warn!(error = %e, "cannot write supervisor record");
        }
    }

    fn record_message<T: Serialize>(&self, st: &State, from: AgentId, to: AgentId, msg: &T) {
        let msg_json = serde_json::to_string(msg).unwrap_or_default();
        self.record(st, from, Record::AgentMessage { from, to, msg_json });
    }

    fn task_update(&self, st: &State, n: u32, note: &str) {
        if let Some(t) = st.plan.as_ref().and_then(|p| p.task(n)) {
            self.record(
                st,
                self.deps.orchestrator,
                Record::TaskUpdate {
                    task_id: t.id,
                    status: t.status,
                    note: note.to_owned(),
                },
            );
        }
    }

    fn publish(&self, st: &State) {
        let view = st.plan.as_ref().map(|p| p.view(self.max_attempts()));
        let _ = self.deps.ui.send(UiEvent::PlanUpdated(view));
        if let Some(plan) = &st.plan {
            let plan_json = serde_json::to_string(plan).unwrap_or_default();
            self.record(st, self.deps.orchestrator, Record::PlanUpdate { plan_json });
        }
    }

    fn plan_mut(st: &mut State) -> Result<&mut Plan, String> {
        st.plan
            .as_mut()
            .ok_or_else(|| "there is no plan; create one with plan_create".to_owned())
    }

    // ------------------------------------------------------------------ orchestrator operations

    /// A lookup entry that passes the reuse rule.
    fn template(&self, id: &str) -> Result<LookupEntry, String> {
        let table = LookupStore::new(&self.deps.data)
            .load()
            .map_err(|e| e.to_string())?;
        let entry = table
            .entry(id.trim())
            .ok_or_else(|| format!("no lookup entry `{id}`; plan from scratch"))?;
        let cfg = &self.deps.config.scorer;
        if !entry.reusable(cfg) {
            return Err(format!(
                "template `{id}` is not proven enough to reuse (score {:.2} over {} run(s); needs ≥ {:.2} and ≥ {} runs); plan from scratch",
                entry.score, entry.stats.runs, cfg.reuse_min_score, cfg.reuse_min_runs
            ));
        }
        Ok(entry.clone())
    }

    /// Steps built from a template; skills that no longer exist are dropped.
    fn instantiate(&self, template: &LookupEntry, goal: &str) -> Vec<StepSpec> {
        let catalog = self.deps.skills.load();
        template
            .steps
            .iter()
            .map(|s| StepSpec {
                title: s.title.clone(),
                instructions: format!("{}. Overall goal: {goal}", s.title),
                role: match s.role.as_str() {
                    "research" => TaskRole::Research,
                    "self" => TaskRole::Orchestrator,
                    _ => TaskRole::Worker,
                },
                allowed_tools: None,
                depends_on: s.depends_on.clone(),
                success_criteria: String::new(),
                skills: s
                    .skills
                    .iter()
                    .filter(|n| catalog.get(n).is_some())
                    .cloned()
                    .collect(),
                agent: None,
            })
            .collect()
    }

    fn check_steps(&self, steps: &mut [StepSpec]) -> Result<(), String> {
        let catalog = self.deps.skills.load();
        let agents = self.deps.agents.load();
        for s in steps {
            if let Some(missing) = s.skills.iter().find(|n| catalog.get(n).is_none()) {
                let known: Vec<&str> = catalog.active.iter().map(|k| k.name.as_str()).collect();
                return Err(format!(
                    "step \"{}\": unknown skill `{missing}`; available: {}",
                    s.title,
                    if known.is_empty() {
                        "(none)".into()
                    } else {
                        known.join(", ")
                    }
                ));
            }
            let Some(name) = s.agent.as_deref() else {
                continue;
            };
            if s.role == TaskRole::Orchestrator {
                return Err(format!(
                    "step \"{}\": role self cannot use custom agent `{name}`",
                    s.title
                ));
            }
            let agent = agents.get(name).ok_or_else(|| {
                let known = agents.names();
                format!(
                    "step \"{}\": unknown custom agent `{name}`; available: {}",
                    s.title,
                    if known.is_empty() {
                        "(none; create one with agent_create)".into()
                    } else {
                        known.join(", ")
                    }
                )
            })?;
            s.role = if agent.base == Role::Research {
                TaskRole::Research
            } else {
                TaskRole::Worker
            };
        }
        Ok(())
    }

    pub fn create_plan(
        &self,
        goal: &str,
        mut steps: Vec<StepSpec>,
        template_id: Option<&str>,
    ) -> Result<String, String> {
        self.check_steps(&mut steps)?;
        let template = template_id.map(|id| self.template(id)).transpose()?;
        let steps = match &template {
            Some(t) if steps.is_empty() => self.instantiate(t, goal),
            _ => steps,
        };
        let mut st = self.lock()?;
        if st.plan.as_ref().is_some_and(Plan::has_active) {
            return Err(
                "a plan is still running; use plan_revise, or cancel its tasks first".into(),
            );
        }
        self.finalize_locked(&mut st, true);
        let mut plan = Plan::new(goal, steps)?;
        if let Some((qt, topic)) = &st.turn_context {
            plan.query_type.clone_from(qt);
            plan.topic.clone_from(topic);
        }
        plan.template_id = template.map(|t| t.id);
        let digest = plan.digest(self.max_attempts());
        st.plan = Some(plan);
        st.plan_started = Some(Instant::now());
        st.plan_budget_noted = false;
        st.notes.clear();
        self.publish(&st);
        Ok(format!(
            "{digest}Dispatch ready tasks with tasks_dispatch, then collect results with tasks_await."
        ))
    }

    pub fn revise(&self, mut add: Vec<StepSpec>, remove: &[u32]) -> Result<String, String> {
        self.check_steps(&mut add)?;
        let mut st = self.lock()?;
        let max = self.max_attempts();
        let plan = Self::plan_mut(&mut st)?;
        if !remove.is_empty() {
            plan.remove(remove)?;
        }
        let added = if add.is_empty() {
            Vec::new()
        } else {
            plan.add(add)?
        };
        let digest = plan.digest(max);
        self.publish(&st);
        Ok(format!("added {added:?}, removed {remove:?}\n{digest}"))
    }

    pub fn dispatch(
        self: &Arc<Self>,
        numbers: &[u32],
        strategy: Option<&str>,
    ) -> Result<String, String> {
        let mut st = self.lock()?;
        let strategy = strategy.map_or("initial approach", str::trim).to_owned();
        let mut lines = Vec::new();
        for &n in numbers {
            let plan = Self::plan_mut(&mut st)?;
            let check = (|| {
                let t = plan
                    .task(n)
                    .ok_or_else(|| format!("task {n}: no such task"))?;
                if t.spec.role == TaskRole::Orchestrator {
                    return Err(format!(
                        "task {n} is yours (role self); do it and record it with task_set_outcome"
                    ));
                }
                match t.status {
                    TaskStatus::Pending => {}
                    TaskStatus::Failed | TaskStatus::Partial => {
                        return Err(format!(
                            "task {n} already ran; use task_retry with a new strategy"
                        ));
                    }
                    other => return Err(format!("task {n} is {other}")),
                }
                plan.deps_met(n)
            })();
            match check {
                Err(e) => lines.push(e),
                Ok(()) => {
                    if let Some(t) = plan.task_mut(n) {
                        t.attempts.push(new_attempt(t.attempts.len(), &strategy));
                        t.status = TaskStatus::Queued;
                    }
                    lines.push(format!("task {n}: queued"));
                }
            }
        }
        self.start_queued(&mut st);
        let lines = self.describe_started(&st, numbers, lines);
        self.publish(&st);
        Ok(lines.join("\n"))
    }

    /// Replaces "queued" with "running as <label>" for tasks that got a slot.
    fn describe_started(&self, st: &State, numbers: &[u32], lines: Vec<String>) -> Vec<String> {
        lines
            .into_iter()
            .zip(numbers)
            .map(|(line, n)| match st.running.get(n) {
                Some(r) if line.ends_with("queued") => {
                    format!("task {n}: running as {}", r.label)
                }
                _ if line.ends_with("queued") => {
                    format!(
                        "task {n}: queued (all {} sub-agent slots are busy)",
                        st.max_subagents
                    )
                }
                _ => line,
            })
            .collect()
    }

    pub fn retry(self: &Arc<Self>, n: u32, strategy: &str) -> Result<String, String> {
        let mut st = self.lock()?;
        let max_retries = self.deps.config.retry.max_retries;
        let plan = Self::plan_mut(&mut st)?;
        let t = plan.task(n).ok_or_else(|| format!("no task {n}"))?;
        if !matches!(t.status, TaskStatus::Failed | TaskStatus::Partial) {
            return Err(format!(
                "task {n} is {}; only failed or partial tasks can be retried",
                t.status
            ));
        }
        if t.retries_used() >= max_retries {
            return Err(format!("task {n} has used all {max_retries} retries"));
        }
        let wanted = normalize_strategy(strategy);
        if wanted.is_empty() {
            return Err("describe the new strategy".into());
        }
        if let Some(prev) = t
            .attempts
            .iter()
            .find(|a| normalize_strategy(&a.strategy) == wanted)
        {
            return Err(format!(
                "strategy already tried in attempt {} (\"{}\"); describe a different approach",
                prev.n, prev.strategy
            ));
        }
        plan.deps_met(n)?;
        if let Some(t) = plan.task_mut(n) {
            t.attempts
                .push(new_attempt(t.attempts.len(), strategy.trim()));
            t.status = TaskStatus::Queued;
        }
        self.start_queued(&mut st);
        let line = self.describe_started(&st, &[n], vec![format!("task {n}: queued")]);
        self.publish(&st);
        Ok(format!("retrying: {}", line.join("")))
    }

    pub fn set_outcome(&self, n: u32, outcome: Outcome, reason: &str) -> Result<String, String> {
        let mut st = self.lock()?;
        let max = self.max_attempts();
        let plan = Self::plan_mut(&mut st)?;
        let t = plan.task_mut(n).ok_or_else(|| format!("no task {n}"))?;
        if is_active(t.status) {
            return Err(format!(
                "task {n} is still {}; wait for it or cancel it",
                t.status
            ));
        }
        if t.attempts.is_empty() {
            let mut a = new_attempt(0, "done by the orchestrator");
            a.started_at = Some(Utc::now());
            t.attempts.push(a);
        }
        if let Some(a) = t.last_attempt_mut() {
            a.outcome = Some(outcome);
            a.reason = reason.to_owned();
            a.ended_at.get_or_insert_with(Utc::now);
        }
        t.status = outcome_status(outcome);
        let exhausted = t.attempts.len() >= max as usize;
        match outcome {
            Outcome::Success | Outcome::Partial => plan.unblock_dependents(n),
            Outcome::Failed if exhausted => {
                plan.block_dependents(n);
            }
            Outcome::Failed => {}
        }
        self.task_update(&st, n, reason);
        self.publish(&st);
        Ok(format!("task {n} marked {outcome:?}"))
    }

    pub fn cancel(&self, n: u32) -> Result<String, String> {
        let mut st = self.lock()?;
        if let Some(r) = st.running.get(&n) {
            r.cancel.cancel();
            self.record_message(
                &st,
                self.deps.orchestrator,
                r.agent,
                &ToSubAgent::Cancel { task: n },
            );
            return Ok(format!("cancelling task {n}"));
        }
        let plan = Self::plan_mut(&mut st)?;
        let t = plan.task_mut(n).ok_or_else(|| format!("no task {n}"))?;
        if !matches!(t.status, TaskStatus::Pending | TaskStatus::Queued) {
            return Err(format!("task {n} is {}", t.status));
        }
        if t.last_attempt().is_some_and(|a| a.started_at.is_none()) {
            t.attempts.pop();
        }
        t.status = TaskStatus::Cancelled;
        let blocked = plan.block_dependents(n);
        self.task_update(&st, n, "cancelled");
        self.publish(&st);
        Ok(format!("task {n} cancelled; blocked: {blocked:?}"))
    }

    /// Cancels every running and waiting task (used when the user cancels the turn).
    pub fn cancel_all(&self) {
        let numbers: Vec<u32> = match self.state.lock() {
            Ok(st) => st
                .plan
                .as_ref()
                .map(|p| {
                    p.tasks
                        .iter()
                        .filter(|t| {
                            matches!(
                                t.status,
                                TaskStatus::Queued
                                    | TaskStatus::Running
                                    | TaskStatus::AwaitingInput
                            )
                        })
                        .map(|t| t.number)
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => return,
        };
        for n in numbers {
            let _ = self.cancel(n);
        }
    }

    /// Drops the plan entirely (new session), saving it for the Scorer if it is over.
    pub fn reset(&self) {
        if let Ok(mut st) = self.state.lock() {
            self.finalize_locked(&mut st, true);
            for r in st.running.values() {
                r.cancel.cancel();
            }
            st.plan = None;
            st.notes.clear();
            let _ = self.deps.ui.send(UiEvent::PlanUpdated(None));
        }
    }

    pub fn answer(&self, n: u32, text: &str) -> Result<String, String> {
        let mut st = self.lock()?;
        let r = st
            .running
            .get_mut(&n)
            .ok_or_else(|| format!("task {n} is not running"))?;
        let tx = r
            .answer
            .take()
            .ok_or_else(|| format!("task {n} has not asked anything"))?;
        let agent = r.agent;
        tx.send(text.to_owned())
            .map_err(|_| format!("task {n} stopped waiting"))?;
        if let Some(t) = st.plan.as_mut().and_then(|p| p.task_mut(n)) {
            t.status = TaskStatus::Running;
            t.question = None;
        }
        self.record_message(
            &st,
            self.deps.orchestrator,
            agent,
            &ToSubAgent::Answer {
                task: n,
                text: text.to_owned(),
            },
        );
        self.publish(&st);
        Ok(format!("answered task {n}"))
    }

    /// Returns as soon as there is news (results, questions, alerts) or `timeout` passes.
    pub async fn await_events(&self, timeout: Duration, cancel: &CancellationToken) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let Ok(mut st) = self.state.lock() else {
                    return "supervisor unavailable".into();
                };
                self.check_plan_budget(&mut st);
                let digest = st
                    .plan
                    .as_ref()
                    .map(|p| p.digest(self.max_attempts()))
                    .unwrap_or_default();
                if !st.notes.is_empty() {
                    let notes = std::mem::take(&mut st.notes);
                    let mut out = String::from("events:\n");
                    for n in notes {
                        let _ = writeln!(out, "- {n}");
                    }
                    return out + &digest;
                }
                if !st.plan.as_ref().is_some_and(Plan::has_active) {
                    return format!("no tasks are running or queued.\n{digest}");
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let digest = self.digest().unwrap_or_default();
                return format!(
                    "no new events in {}s; tasks are still running.\n{digest}",
                    timeout.as_secs()
                );
            }
            tokio::select! {
                () = &mut notified => {}
                () = tokio::time::sleep(remaining) => {}
                () = cancel.cancelled() => return "cancelled".into(),
            }
        }
    }

    fn check_plan_budget(&self, st: &mut State) {
        let budget = Duration::from_secs(self.deps.config.retry.plan_time_budget_secs);
        if !st.plan_budget_noted && st.plan_started.is_some_and(|s| s.elapsed() > budget) {
            st.plan_budget_noted = true;
            st.notes.push(format!(
                "the plan has run longer than its {}s budget; consider wrapping up",
                budget.as_secs()
            ));
        }
    }

    pub fn on_alert(&self, alert: Alert) {
        let Ok(mut st) = self.state.lock() else {
            return;
        };
        let task = alert
            .task
            .map(|t| format!(" on task {t}"))
            .unwrap_or_default();
        st.notes.push(format!(
            "alert: {}{task} {:?}: {}",
            alert.label, alert.kind, alert.detail
        ));
        if alert.kind == AlertKind::BudgetExceeded
            && let Some(r) = alert.task.and_then(|t| st.running.get(&t))
        {
            r.budget_hit.store(true, Ordering::SeqCst);
            r.cancel.cancel();
        }
        drop(st);
        self.notify.notify_waiters();
    }

    // ------------------------------------------------------------------ sub-agent side

    pub fn report(&self, agent: AgentId, report: Report) -> Result<(), String> {
        let mut st = self.lock()?;
        let link = st.links.get_mut(&agent).ok_or("this agent has no task")?;
        link.1 = Some(report);
        Ok(())
    }

    pub async fn ask(
        &self,
        agent: AgentId,
        question: &str,
        cancel: &CancellationToken,
    ) -> Result<String, String> {
        let rx = {
            let mut st = self.lock()?;
            let n = st
                .links
                .get(&agent)
                .map(|l| l.0)
                .ok_or("this agent has no task")?;
            let (tx, rx) = oneshot::channel();
            if let Some(r) = st.running.get_mut(&n) {
                r.answer = Some(tx);
            }
            if let Some(t) = st.plan.as_mut().and_then(|p| p.task_mut(n)) {
                t.status = TaskStatus::AwaitingInput;
                t.question = Some(question.to_owned());
            }
            st.notes.push(format!(
                "task {n} asks: {question} (answer with task_answer)"
            ));
            self.record_message(
                &st,
                agent,
                self.deps.orchestrator,
                &FromSubAgent::NeedInput {
                    task: n,
                    question: question.to_owned(),
                },
            );
            self.publish(&st);
            rx
        };
        self.notify.notify_waiters();
        self.deps.board.set_phase(agent, Phase::WaitingInput);
        let answer = tokio::select! {
            r = rx => r.map_err(|_| "the orchestrator did not answer".to_owned()),
            () = cancel.cancelled() => Err("cancelled".into()),
        };
        self.deps.board.set_phase(agent, Phase::Streaming);
        answer
    }

    pub(crate) fn take_report(&self, agent: AgentId) -> Option<Report> {
        self.state.lock().ok()?.links.get_mut(&agent)?.1.take()
    }

    /// `retry_advice`: the decision model's retry-vs-give-up answer for a failed attempt.
    pub(crate) fn finish_attempt(
        self: &Arc<Self>,
        n: u32,
        end: AttemptEnd,
        tokens: u64,
        tools_used: Vec<String>,
        retry_advice: Option<bool>,
    ) {
        let Ok(mut st) = self.state.lock() else {
            return;
        };
        let running = st.running.remove(&n);
        let budget_hit = running
            .as_ref()
            .is_some_and(|r| r.budget_hit.load(Ordering::SeqCst));
        let agent = running.as_ref().map(|r| r.agent);
        if let Some(a) = agent {
            st.links.remove(&a);
        }
        let max = self.max_attempts();
        let (outcome, summary, reason, evidence) = match end {
            AttemptEnd::Reported(r) => {
                let reason = if r.outcome == Outcome::Success {
                    String::new()
                } else {
                    r.summary.clone()
                };
                (Some(r.outcome), r.summary, reason, r.evidence)
            }
            AttemptEnd::NoReport(text) => (
                Some(Outcome::Partial),
                text,
                "did not call report_result".into(),
                vec![],
            ),
            AttemptEnd::Cancelled if budget_hit => (
                Some(Outcome::Failed),
                String::new(),
                "budget exceeded".into(),
                vec![],
            ),
            AttemptEnd::Cancelled => (None, String::new(), "cancelled".into(), vec![]),
            AttemptEnd::Error(e) => (Some(Outcome::Failed), String::new(), e, vec![]),
        };
        let Some(plan) = st.plan.as_mut() else { return };
        let Some(t) = plan.task_mut(n) else { return };
        if let Some(a) = t.last_attempt_mut() {
            a.ended_at = Some(Utc::now());
            a.outcome = outcome;
            a.summary.clone_from(&summary);
            a.reason.clone_from(&reason);
            a.evidence.clone_from(&evidence);
            a.tokens = tokens;
            a.tools_used = tools_used;
        }
        t.status = outcome.map_or(TaskStatus::Cancelled, outcome_status);
        t.question = None;
        let attempts = t.attempts.len();
        let title = t.spec.title.clone();
        let status = t.status;
        let mut note = format!(
            "task {n} ({title}) {status} on attempt {attempts}/{max}: {}",
            if reason.is_empty() { &summary } else { &reason }
        );
        match status {
            TaskStatus::Success | TaskStatus::Partial => plan.unblock_dependents(n),
            TaskStatus::Failed if attempts >= max as usize => {
                let blocked = plan.block_dependents(n);
                let _ = write!(note, "; no retries left, blocked {blocked:?}");
            }
            TaskStatus::Failed => {
                note.push_str(match retry_advice {
                    Some(true) => {
                        "; retry with task_retry and a different strategy (the decision model expects one to succeed)"
                    }
                    Some(false) => {
                        "; the decision model judged another strategy unlikely to succeed: revise the plan \
                         (plan_revise) or report the failure, unless you see a clear fix"
                    }
                    None => "; retry with task_retry and a different strategy",
                });
            }
            TaskStatus::Cancelled => {
                plan.block_dependents(n);
            }
            _ => {}
        }
        st.notes.push(note);
        self.task_update(&st, n, &reason);
        if let Some(a) = agent {
            self.record_message(
                &st,
                a,
                self.deps.orchestrator,
                &FromSubAgent::Result {
                    task: n,
                    outcome: outcome
                        .map_or_else(|| "cancelled".into(), |o| format!("{o:?}").to_lowercase()),
                    summary,
                    evidence,
                },
            );
        }
        self.start_queued(&mut st);
        self.publish(&st);
        drop(st);
        self.notify.notify_waiters();
    }

    /// Starts queued tasks while sub-agent slots are free.
    fn start_queued(self: &Arc<Self>, st: &mut State) {
        let caps = &self.deps.config.agents;
        let Some(plan) = &st.plan else { return };
        let queued: Vec<u32> = plan
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Queued)
            .map(|t| t.number)
            .collect();
        for n in queued {
            let Some(plan) = &st.plan else { return };
            let Some(t) = plan.task(n) else { continue };
            let role = role_of(t.spec.role);
            let role_cap = if role == Role::Research {
                caps.max_research
            } else {
                caps.max_workers
            };
            let role_running = st.running.values().filter(|r| r.role == role).count() as u32;
            if st.running.len() as u32 >= st.max_subagents || role_running >= role_cap {
                continue;
            }
            self.start_attempt(st, n, role);
        }
    }

    fn start_attempt(self: &Arc<Self>, st: &mut State, n: u32, role: Role) {
        let custom_name = st
            .plan
            .as_ref()
            .and_then(|p| p.task(n))
            .and_then(|t| t.spec.agent.clone());
        // Read now so edits apply to the next attempt; `None` with a name means it was deleted.
        let custom = custom_name
            .as_deref()
            .and_then(|name| self.deps.agents.load().get(name).cloned());
        let base = custom_name.unwrap_or_else(|| role.to_string());
        let counter = st.counters.entry(base.clone()).or_insert(0);
        *counter += 1;
        let label = format!("{base}{counter}");
        let agent = AgentId::new();
        let cancel = CancellationToken::new();
        let goal = st.plan.as_ref().map(|p| p.goal.clone()).unwrap_or_default();
        let Some(t) = st.plan.as_mut().and_then(|p| p.task_mut(n)) else {
            return;
        };
        t.status = TaskStatus::Running;
        let prior: Vec<(u32, String, String)> = t
            .attempts
            .iter()
            .filter(|a| a.outcome.is_some())
            .map(|a| {
                let why = if a.reason.is_empty() {
                    a.summary.clone()
                } else {
                    a.reason.clone()
                };
                (
                    a.n,
                    a.strategy.clone(),
                    format!("{:?}: {why}", a.outcome.unwrap_or(Outcome::Failed)),
                )
            })
            .collect();
        let spec = t.spec.clone();
        let task_id = t.id;
        let Some(attempt) = t.last_attempt_mut() else {
            return;
        };
        attempt.agent = Some(agent);
        attempt.started_at = Some(Utc::now());
        let job = AttemptJob {
            number: n,
            attempt: attempt.n,
            agent,
            label: label.clone(),
            role,
            custom,
            goal,
            spec,
            task_id,
            strategy: attempt.strategy.clone(),
            prior,
            cancel: cancel.clone(),
            session: st.session.clone(),
        };
        st.running.insert(
            n,
            Running {
                agent,
                label,
                role,
                cancel,
                answer: None,
                budget_hit: Arc::new(AtomicBool::new(false)),
            },
        );
        st.links.insert(agent, (n, None));
        self.task_update(st, n, &format!("attempt {} started", job.attempt));
        tokio::spawn(run_attempt(self.clone(), job));
    }
}

fn new_attempt(existing: usize, strategy: &str) -> Attempt {
    Attempt {
        n: u32::try_from(existing).unwrap_or(u32::MAX) + 1,
        strategy: strategy.to_owned(),
        agent: None,
        started_at: None,
        ended_at: None,
        outcome: None,
        summary: String::new(),
        reason: String::new(),
        evidence: Vec::new(),
        tokens: 0,
        tools_used: Vec::new(),
    }
}
