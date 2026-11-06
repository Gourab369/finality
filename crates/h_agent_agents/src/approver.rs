//! Bridges tool approvals to the UI: each request becomes an `ApprovalPrompt` answered by `CoreCommand::ApprovalResponse`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use h_agent_core::AgentId;
use h_agent_core::events::{ApprovalChoice, ApprovalPrompt, UiEvent};
use h_agent_safety::Location;
use h_agent_tools::{ApprovalRequest, ApprovalResponse, Approver};
use tokio::sync::{mpsc, oneshot};

use crate::heartbeat::{HeartbeatBoard, Phase};

pub struct UiApprover {
    ui: mpsc::UnboundedSender<UiEvent>,
    pending: Mutex<HashMap<u64, oneshot::Sender<ApprovalChoice>>>,
    next: AtomicU64,
    labels: Mutex<HashMap<AgentId, (String, Option<String>)>>,
    board: Option<Arc<HeartbeatBoard>>,
}

impl UiApprover {
    pub fn new(ui: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self {
            ui,
            pending: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            labels: Mutex::new(HashMap::new()),
            board: None,
        }
    }

    /// Marks agents as awaiting approval on `board` while a prompt is open.
    pub fn with_board(mut self, board: Arc<HeartbeatBoard>) -> Self {
        self.board = Some(board);
        self
    }

    /// Name shown in approval prompts for `agent` (unregistered agents show as the orchestrator).
    pub fn label(&self, agent: AgentId, label: &str, task: Option<String>) {
        if let Ok(mut l) = self.labels.lock() {
            l.insert(agent, (label.to_owned(), task));
        }
    }

    pub fn forget(&self, agent: AgentId) {
        if let Ok(mut l) = self.labels.lock() {
            l.remove(&agent);
        }
    }

    /// Delivers the user's answer. Returns false if the request is no longer pending.
    pub fn resolve(&self, id: u64, choice: ApprovalChoice) -> bool {
        let sender = self.pending.lock().ok().and_then(|mut p| p.remove(&id));
        sender.is_some_and(|tx| tx.send(choice).is_ok())
    }

    /// Denies every outstanding request and removes it from the UI (used when a turn is cancelled).
    pub fn withdraw_all(&self) {
        let drained: Vec<(u64, oneshot::Sender<ApprovalChoice>)> = match self.pending.lock() {
            Ok(mut p) => p.drain().collect(),
            Err(_) => return,
        };
        for (id, tx) in drained {
            let _ = tx.send(ApprovalChoice::Deny {
                note: Some("cancelled".into()),
            });
            let _ = self.ui.send(UiEvent::ApprovalClosed { id });
        }
    }
}

#[async_trait]
impl Approver for UiApprover {
    async fn request(&self, req: ApprovalRequest) -> ApprovalResponse {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        if let Ok(mut p) = self.pending.lock() {
            p.insert(id, tx);
        }
        let (agent_label, task) = self
            .labels
            .lock()
            .ok()
            .and_then(|l| l.get(&req.agent).cloned())
            .unwrap_or_else(|| ("orchestrator".into(), None));
        let prompt = ApprovalPrompt {
            id,
            agent_label,
            task,
            tool: req.tool,
            risk: req.risk,
            summary: req.summary,
            detail: req.detail,
            reasons: req.reasons,
            inside: req.location == Location::Inside,
            allow_session: req.allow_session,
            default_deny: req.default_deny,
        };
        if self.ui.send(UiEvent::Approval(prompt)).is_err() {
            return ApprovalResponse::Deny {
                note: Some("no user interface to ask".into()),
            };
        }
        let agent = req.agent;
        let previous = self
            .board
            .as_ref()
            .and_then(|b| b.get(agent))
            .map(|h| h.phase);
        if let Some(b) = &self.board {
            b.set_phase(agent, Phase::WaitingApproval);
        }
        let answer = rx.await;
        if let (Some(b), Some(p)) = (&self.board, previous) {
            b.set_phase(agent, p);
        }
        match answer {
            Ok(ApprovalChoice::AllowOnce) => ApprovalResponse::AllowOnce,
            Ok(ApprovalChoice::AllowSession) => ApprovalResponse::AllowSession,
            Ok(ApprovalChoice::Deny { note }) => ApprovalResponse::Deny { note },
            Err(_) => ApprovalResponse::Deny { note: None },
        }
    }
}
