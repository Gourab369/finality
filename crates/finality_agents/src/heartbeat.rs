//! Heartbeats (DESIGN §6.3): agents publish their phase and usage; a monitor turns them into
//! agents-panel rows and Stalled / NoProgress / BudgetNear / BudgetExceeded alerts.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use finality_core::events::{AgentLine, UiEvent};
use finality_core::{AgentId, Role};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const MAX_PANEL_REFRESH: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    Idle,
    WaitingLlm,
    Streaming,
    ToolCall(String),
    WaitingApproval,
    WaitingInput,
    /// Blocked in `tasks_await`.
    WaitingTasks,
}

impl Phase {
    /// The phase while `tool` runs; tools that block on other parties count as waiting.
    pub fn for_tool(tool: &str) -> Self {
        match tool {
            "tasks_await" => Phase::WaitingTasks,
            "ask_user" | "ask_orchestrator" => Phase::WaitingInput,
            other => Phase::ToolCall(other.to_owned()),
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Idle => f.write_str("idle"),
            Phase::WaitingLlm => f.write_str("waiting for model"),
            Phase::Streaming => f.write_str("thinking"),
            Phase::ToolCall(t) => write!(f, "tool:{t}"),
            Phase::WaitingApproval => f.write_str("awaiting approval"),
            Phase::WaitingInput => f.write_str("awaiting answer"),
            Phase::WaitingTasks => f.write_str("waiting for tasks"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub tokens: u64,
    pub time: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertKind {
    Stalled,
    NoProgress,
    BudgetNear,
    BudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alert {
    pub agent: AgentId,
    pub label: String,
    pub task: Option<u32>,
    pub kind: AlertKind,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Heartbeat {
    pub role: Role,
    pub label: String,
    pub task: Option<u32>,
    pub phase: Phase,
    pub started_at: Instant,
    pub last_beat: Instant,
    pub last_progress: Instant,
    pub tokens: u64,
    pub ctx_used: u32,
    pub ctx_total: u32,
    pub budget: Option<Budget>,
    alerted: HashSet<AlertKind>,
}

#[derive(Debug, Clone, Copy)]
pub struct MonitorSettings {
    pub interval: Duration,
    pub stall_after: u32,
    pub no_progress: Duration,
}

#[derive(Default)]
pub struct HeartbeatBoard {
    beats: Mutex<HashMap<AgentId, Heartbeat>>,
    order: Mutex<Vec<AgentId>>,
}

impl HeartbeatBoard {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn register(
        &self,
        agent: AgentId,
        role: Role,
        label: &str,
        task: Option<u32>,
        budget: Option<Budget>,
    ) {
        let now = Instant::now();
        if let Ok(mut b) = self.beats.lock() {
            b.insert(
                agent,
                Heartbeat {
                    role,
                    label: label.to_owned(),
                    task,
                    phase: Phase::Idle,
                    started_at: now,
                    last_beat: now,
                    last_progress: now,
                    tokens: 0,
                    ctx_used: 0,
                    ctx_total: 0,
                    budget,
                    alerted: HashSet::new(),
                },
            );
        }
        if let Ok(mut o) = self.order.lock()
            && !o.contains(&agent)
        {
            o.push(agent);
        }
    }

    pub fn remove(&self, agent: AgentId) {
        if let Ok(mut b) = self.beats.lock() {
            b.remove(&agent);
        }
        if let Ok(mut o) = self.order.lock() {
            o.retain(|a| *a != agent);
        }
    }

    pub fn get(&self, agent: AgentId) -> Option<Heartbeat> {
        self.beats.lock().ok()?.get(&agent).cloned()
    }

    fn with(&self, agent: AgentId, f: impl FnOnce(&mut Heartbeat)) {
        if let Ok(mut b) = self.beats.lock()
            && let Some(h) = b.get_mut(&agent)
        {
            f(h);
        }
    }

    pub fn beat(&self, agent: AgentId) {
        self.with(agent, |h| h.last_beat = Instant::now());
    }

    pub fn set_phase(&self, agent: AgentId, phase: Phase) {
        self.with(agent, |h| {
            let now = Instant::now();
            h.phase = phase;
            h.last_beat = now;
            h.last_progress = now;
        });
    }

    pub fn progress(&self, agent: AgentId) {
        self.with(agent, |h| {
            let now = Instant::now();
            h.last_beat = now;
            h.last_progress = now;
        });
    }

    pub fn add_usage(&self, agent: AgentId, tokens: u64, ctx_used: u32, ctx_total: u32) {
        self.with(agent, |h| {
            h.tokens += tokens;
            h.ctx_used = ctx_used;
            h.ctx_total = ctx_total;
        });
    }

    pub fn lines(&self, now: Instant) -> Vec<AgentLine> {
        let (Ok(b), Ok(o)) = (self.beats.lock(), self.order.lock()) else {
            return Vec::new();
        };
        o.iter()
            .filter_map(|a| b.get(a))
            .map(|h| AgentLine {
                label: h.label.clone(),
                role: h.role,
                phase: h.phase.to_string(),
                task: h.task,
                elapsed_secs: now.duration_since(h.started_at).as_secs(),
                ctx_used: h.ctx_used,
                ctx_total: h.ctx_total,
            })
            .collect()
    }

    /// New alerts since the last check; each kind is raised once per agent.
    pub fn check(&self, now: Instant, s: &MonitorSettings) -> Vec<Alert> {
        let Ok(mut b) = self.beats.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (agent, h) in b.iter_mut() {
            let waiting = matches!(
                h.phase,
                Phase::Idle | Phase::WaitingApproval | Phase::WaitingInput | Phase::WaitingTasks
            );
            let silent = now.duration_since(h.last_beat);
            let mut raise = |kind: AlertKind, detail: String, h: &mut Heartbeat| {
                if h.alerted.insert(kind) {
                    out.push(Alert {
                        agent: *agent,
                        label: h.label.clone(),
                        task: h.task,
                        kind,
                        detail,
                    });
                }
            };
            if !waiting && silent > s.interval * s.stall_after {
                raise(
                    AlertKind::Stalled,
                    format!("no heartbeat for {}s", silent.as_secs()),
                    h,
                );
            }
            let quiet = now.duration_since(h.last_progress);
            if matches!(h.phase, Phase::Streaming | Phase::ToolCall(_)) && quiet > s.no_progress {
                raise(
                    AlertKind::NoProgress,
                    format!("no progress for {}s while {}", quiet.as_secs(), h.phase),
                    h,
                );
            }
            if let Some(budget) = h.budget {
                let elapsed = now.duration_since(h.started_at);
                let token_frac = h.tokens as f64 / budget.tokens.max(1) as f64;
                let time_frac = elapsed.as_secs_f64() / budget.time.as_secs_f64().max(0.001);
                let frac = token_frac.max(time_frac);
                let detail = format!(
                    "{} tokens of {}, {}s of {}s",
                    h.tokens,
                    budget.tokens,
                    elapsed.as_secs(),
                    budget.time.as_secs()
                );
                if frac >= 1.0 {
                    raise(AlertKind::BudgetExceeded, detail, h);
                } else if frac >= 0.8 {
                    raise(AlertKind::BudgetNear, detail, h);
                }
            }
        }
        out
    }
}

/// Pushes panel rows to the UI (when they change) and hands alerts to `on_alert`.
pub fn spawn_monitor(
    board: Arc<HeartbeatBoard>,
    settings: MonitorSettings,
    ui: mpsc::UnboundedSender<UiEvent>,
    on_alert: impl Fn(Alert) + Send + Sync + 'static,
    stop: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(settings.interval.min(MAX_PANEL_REFRESH));
        let mut last: Vec<AgentLine> = Vec::new();
        loop {
            tokio::select! {
                () = stop.cancelled() => return,
                _ = tick.tick() => {}
            }
            let now = Instant::now();
            let lines = board.lines(now);
            if lines != last {
                let _ = ui.send(UiEvent::AgentStatus(lines.clone()));
                last = lines;
            }
            for alert in board.check(now, &settings) {
                on_alert(alert);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> MonitorSettings {
        MonitorSettings {
            interval: Duration::from_secs(1),
            stall_after: 3,
            no_progress: Duration::from_secs(120),
        }
    }

    #[test]
    fn stall_is_raised_once_and_not_while_waiting_for_approval() {
        let board = HeartbeatBoard::new();
        let (a, b) = (AgentId::new(), AgentId::new());
        board.register(a, Role::Worker, "worker1", Some(1), None);
        board.register(b, Role::Worker, "worker2", Some(2), None);
        board.set_phase(a, Phase::Streaming);
        board.set_phase(b, Phase::WaitingApproval);
        let later = Instant::now() + Duration::from_secs(5);
        let alerts = board.check(later, &settings());
        assert_eq!(alerts.len(), 1);
        assert_eq!((alerts[0].agent, alerts[0].kind), (a, AlertKind::Stalled));
        assert!(board.check(later, &settings()).is_empty(), "raised once");
    }

    #[test]
    fn no_progress_and_budget_alerts() {
        let board = HeartbeatBoard::new();
        let a = AgentId::new();
        let budget = Budget {
            tokens: 1000,
            time: Duration::from_secs(900),
        };
        board.register(a, Role::Research, "research1", Some(1), Some(budget));
        board.set_phase(a, Phase::ToolCall("shell".into()));
        board.add_usage(a, 850, 850, 8192);
        let now = Instant::now();
        let kinds: Vec<AlertKind> = board
            .check(now, &settings())
            .into_iter()
            .map(|a| a.kind)
            .collect();
        assert_eq!(kinds, vec![AlertKind::BudgetNear]);

        board.add_usage(a, 200, 1050, 8192);
        let s = MonitorSettings {
            no_progress: Duration::from_secs(2),
            stall_after: 100,
            ..settings()
        };
        let mut kinds: Vec<AlertKind> = board
            .check(now + Duration::from_secs(3), &s)
            .into_iter()
            .map(|a| a.kind)
            .collect();
        kinds.sort_by_key(|k| format!("{k:?}"));
        assert_eq!(
            kinds,
            vec![AlertKind::BudgetExceeded, AlertKind::NoProgress]
        );
    }

    #[test]
    fn lines_keep_registration_order() {
        let board = HeartbeatBoard::new();
        let (a, b) = (AgentId::new(), AgentId::new());
        board.register(a, Role::Orchestrator, "orchestrator", None, None);
        board.register(b, Role::Worker, "worker1", Some(3), None);
        board.set_phase(b, Phase::ToolCall("shell".into()));
        let lines = board.lines(Instant::now());
        assert_eq!(
            lines.iter().map(|l| l.label.as_str()).collect::<Vec<_>>(),
            ["orchestrator", "worker1"]
        );
        assert_eq!(lines[1].phase, "tool:shell");
        board.remove(a);
        assert_eq!(board.lines(Instant::now()).len(), 1);
    }
}
