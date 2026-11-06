//! Priority-ordered cap on concurrent LLM requests (DESIGN §4.4).
//! Orchestrator > Worker/Research > Memory > Scorer/Learner; a permit is held for the whole stream.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use finality_core::{ProviderKind, Role};
use finality_llm::{ChatRequest, ChatStream, LlmError, LlmProvider, ModelInfo};
use futures::StreamExt;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

pub fn priority(role: Role) -> u8 {
    match role {
        Role::Orchestrator => 0,
        Role::Worker | Role::Research => 1,
        Role::Memory => 2,
        Role::Scorer | Role::Learner => 3,
    }
}

struct Waiter {
    key: (Reverse<u8>, Reverse<u64>),
    tx: oneshot::Sender<()>,
}

impl PartialEq for Waiter {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Waiter {}
impl PartialOrd for Waiter {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Waiter {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

struct State {
    free: usize,
    seq: u64,
    waiters: BinaryHeap<Waiter>,
}

pub struct LlmGate {
    state: Mutex<State>,
}

/// Returns its slot to the gate when dropped.
pub struct Permit {
    gate: Arc<LlmGate>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

impl LlmGate {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                free: capacity.max(1),
                seq: 0,
                waiters: BinaryHeap::new(),
            }),
        })
    }

    /// Waits for a slot; lower `priority` values are served first, FIFO within a priority.
    pub async fn acquire(
        self: &Arc<Self>,
        priority: u8,
        cancel: &CancellationToken,
    ) -> Result<Permit, LlmError> {
        let rx = {
            let mut s = self
                .state
                .lock()
                .map_err(|_| LlmError::Other("LLM gate poisoned".into()))?;
            if s.free > 0 && s.waiters.is_empty() {
                s.free -= 1;
                return Ok(Permit { gate: self.clone() });
            }
            let (tx, rx) = oneshot::channel();
            s.seq += 1;
            let key = (Reverse(priority), Reverse(s.seq));
            s.waiters.push(Waiter { key, tx });
            rx
        };
        let mut rx = rx;
        tokio::select! {
            biased;
            r = &mut rx => match r {
                Ok(()) => Ok(Permit { gate: self.clone() }),
                Err(_) => Err(LlmError::Other("LLM gate closed".into())),
            },
            () = cancel.cancelled() => {
                rx.close();
                // A slot handed over just before cancellation must go back to the gate.
                if rx.try_recv().is_ok() {
                    drop(Permit { gate: self.clone() });
                }
                Err(LlmError::Cancelled)
            }
        }
    }

    fn release(&self) {
        let Ok(mut s) = self.state.lock() else { return };
        // A waiter whose receiver is gone (cancelled) is skipped.
        while let Some(w) = s.waiters.pop() {
            if w.tx.send(()).is_ok() {
                return;
            }
        }
        s.free += 1;
    }
}

/// An `LlmProvider` whose chat requests pass through the gate at a fixed priority.
pub struct GatedProvider {
    inner: Arc<dyn LlmProvider>,
    gate: Arc<LlmGate>,
    priority: u8,
}

impl GatedProvider {
    pub fn new(inner: Arc<dyn LlmProvider>, gate: Arc<LlmGate>, role: Role) -> Self {
        Self {
            inner,
            gate,
            priority: priority(role),
        }
    }
}

#[async_trait]
impl LlmProvider for GatedProvider {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.inner.list_models().await
    }

    async fn model_info(&self, model: &str) -> Result<ModelInfo, LlmError> {
        self.inner.model_info(model).await
    }

    async fn model_info_for(
        &self,
        kind: Option<ProviderKind>,
        model: &str,
    ) -> Result<ModelInfo, LlmError> {
        self.inner.model_info_for(kind, model).await
    }

    async fn chat(
        &self,
        req: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let permit = self.gate.acquire(self.priority, &cancel).await?;
        let stream = self.inner.chat(req, cancel).await?;
        // The permit lives inside the stream so the slot is held until the response is fully consumed.
        Ok(Box::pin(stream.map(move |item| {
            let _held = &permit;
            item
        })))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn higher_priority_waiters_go_first() {
        let gate = LlmGate::new(1);
        let none = CancellationToken::new();
        let held = gate.acquire(0, &none).await.unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (name, prio) in [
            ("learner", 3u8),
            ("worker", 1),
            ("memory", 2),
            ("worker2", 1),
        ] {
            let (gate, order, none) = (gate.clone(), order.clone(), none.clone());
            handles.push(tokio::spawn(async move {
                let _p = gate.acquire(prio, &none).await.unwrap();
                order.lock().unwrap().push(name);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(held);
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            *order.lock().unwrap(),
            vec!["worker", "worker2", "memory", "learner"]
        );
    }

    #[tokio::test]
    async fn cancelled_waiters_do_not_leak_slots() {
        let gate = LlmGate::new(1);
        let held = gate.acquire(0, &CancellationToken::new()).await.unwrap();
        let cancel = CancellationToken::new();
        let g = gate.clone();
        let c = cancel.clone();
        let waiting = tokio::spawn(async move { g.acquire(1, &c).await.is_err() });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();
        assert!(waiting.await.unwrap());
        drop(held);
        let again = tokio::time::timeout(
            Duration::from_secs(1),
            gate.acquire(0, &CancellationToken::new()),
        )
        .await;
        assert!(
            again.is_ok(),
            "slot was returned after the cancelled waiter"
        );
    }
}
