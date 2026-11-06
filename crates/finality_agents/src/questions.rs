//! `ask_user` questions shown as a UI prompt and answered by `CoreCommand::QuestionAnswer`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use finality_core::events::{QuestionPrompt, UiEvent};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub struct UiQuestions {
    ui: mpsc::UnboundedSender<UiEvent>,
    pending: Mutex<HashMap<u64, oneshot::Sender<String>>>,
    next: AtomicU64,
}

impl UiQuestions {
    pub fn new(ui: mpsc::UnboundedSender<UiEvent>) -> Self {
        Self {
            ui,
            pending: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
        }
    }

    /// `None` if the question was withdrawn or there is no UI.
    pub async fn ask(
        &self,
        agent_label: &str,
        text: &str,
        options: Vec<String>,
        cancel: &CancellationToken,
    ) -> Option<String> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().ok()?.insert(id, tx);
        let prompt = QuestionPrompt {
            id,
            agent_label: agent_label.to_owned(),
            text: text.to_owned(),
            options,
        };
        self.ui.send(UiEvent::Question(prompt)).ok()?;
        tokio::select! {
            r = rx => r.ok(),
            () = cancel.cancelled() => {
                if let Ok(mut p) = self.pending.lock() {
                    p.remove(&id);
                }
                let _ = self.ui.send(UiEvent::QuestionClosed { id });
                None
            }
        }
    }

    pub fn answer(&self, id: u64, text: String) -> bool {
        let tx = self.pending.lock().ok().and_then(|mut p| p.remove(&id));
        tx.is_some_and(|tx| tx.send(text).is_ok())
    }

    pub fn withdraw_all(&self) {
        let ids: Vec<u64> = match self.pending.lock() {
            Ok(mut p) => p.drain().map(|(id, _)| id).collect(),
            Err(_) => return,
        };
        for id in ids {
            let _ = self.ui.send(UiEvent::QuestionClosed { id });
        }
    }
}
