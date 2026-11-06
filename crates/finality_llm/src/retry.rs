use std::future::Future;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::LlmError;

/// Exponential backoff for transient failures before a response starts streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
        }
    }
}

/// Longest wait a `Retry-After` header may impose.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

impl RetryPolicy {
    pub async fn run<T, F, Fut>(&self, cancel: &CancellationToken, mut op: F) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        let mut attempt = 1;
        loop {
            let result = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(LlmError::Cancelled),
                r = op() => r,
            };
            match result {
                Err(e) if e.is_transient() && attempt < self.max_attempts => {
                    let backoff = self.base_delay * 2u32.saturating_pow(attempt - 1);
                    let delay = e
                        .retry_after()
                        .map_or(backoff, |ra| ra.min(MAX_RETRY_AFTER).max(backoff));
                    tracing::warn!(attempt, ?delay, error = %e, "transient LLM error; retrying");
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => return Err(LlmError::Cancelled),
                        () = tokio::time::sleep(delay) => {}
                    }
                    attempt += 1;
                }
                other => return other,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    fn fast() -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
        }
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let calls = AtomicU32::new(0);
        let r = fast()
            .run(&CancellationToken::new(), || async {
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(LlmError::Status {
                        status: 503,
                        body: String::new(),
                        retry_after: None,
                    })
                } else {
                    Ok(7)
                }
            })
            .await;
        assert_eq!(r.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_client_errors() {
        let calls = AtomicU32::new(0);
        let r: Result<(), _> = fast()
            .run(&CancellationToken::new(), || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(LlmError::Status {
                    status: 400,
                    body: String::new(),
                    retry_after: None,
                })
            })
            .await;
        assert!(matches!(r, Err(LlmError::Status { status: 400, .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn waits_for_retry_after_within_bounds() {
        let calls = AtomicU32::new(0);
        let started = std::time::Instant::now();
        let r = fast()
            .run(&CancellationToken::new(), || async {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(LlmError::Status {
                        status: 429,
                        body: String::new(),
                        retry_after: Some(Duration::from_millis(120)),
                    })
                } else {
                    Ok(())
                }
            })
            .await;
        assert!(r.is_ok());
        assert!(started.elapsed() >= Duration::from_millis(100));
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let calls = AtomicU32::new(0);
        let r: Result<(), _> = fast()
            .run(&CancellationToken::new(), || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(LlmError::Timeout)
            })
            .await;
        assert!(matches!(r, Err(LlmError::Timeout)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn cancel_interrupts_backoff() {
        let cancel = CancellationToken::new();
        let policy = RetryPolicy {
            max_attempts: 5,
            base_delay: Duration::from_secs(60),
        };
        let c = cancel.clone();
        let r: Result<(), _> = policy
            .run(&cancel, || {
                c.cancel();
                async { Err(LlmError::Timeout) }
            })
            .await;
        assert!(matches!(r, Err(LlmError::Cancelled)));
    }
}
