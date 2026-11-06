//! Session log writer running on a dedicated thread so agents never block on disk I/O.

use std::sync::Mutex;

use h_agent_core::AgentId;
use tokio::sync::{mpsc, oneshot};

use crate::hlog::{LogError, LogWriter};
use crate::record::{Record, RecordEnvelope};

enum Cmd {
    Append(RecordEnvelope),
    Flush(oneshot::Sender<Result<(), String>>),
    Close(oneshot::Sender<Result<(), String>>),
}

/// Cloneable-by-reference handle; records are written in the order `append` is called.
pub struct SessionLog {
    tx: mpsc::UnboundedSender<Cmd>,
    // Held while assigning a seq and sending, so seq order matches write order across threads.
    next_seq: Mutex<u64>,
}

impl SessionLog {
    pub fn spawn(writer: LogWriter) -> Result<Self, LogError> {
        let next_seq = writer.next_seq();
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("h_agent-session-log".into())
            .spawn(move || run(writer, rx))?;
        Ok(Self {
            tx,
            next_seq: Mutex::new(next_seq),
        })
    }

    /// Queues a record and returns its sequence number.
    pub fn append(&self, agent: AgentId, body: Record) -> Result<u64, LogError> {
        let mut next = self
            .next_seq
            .lock()
            .map_err(|_| LogError::Writer("sequence lock poisoned".into()))?;
        let env = RecordEnvelope {
            seq: *next,
            ts_unix_ms: chrono::Utc::now().timestamp_millis(),
            agent,
            body,
        };
        let seq = env.seq;
        self.tx
            .send(Cmd::Append(env))
            .map_err(|_| LogError::Writer("writer thread has exited".into()))?;
        *next += 1;
        Ok(seq)
    }

    /// Resolves once every earlier record has been written; reports the first write error, if any.
    pub async fn flush(&self) -> Result<(), LogError> {
        self.request(Cmd::Flush).await
    }

    /// Writes pending records, syncs to disk and stops the writer.
    pub async fn close(&self) -> Result<(), LogError> {
        self.request(Cmd::Close).await
    }

    async fn request(
        &self,
        make: fn(oneshot::Sender<Result<(), String>>) -> Cmd,
    ) -> Result<(), LogError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make(tx))
            .map_err(|_| LogError::Writer("writer thread has exited".into()))?;
        rx.await
            .map_err(|_| LogError::Writer("writer thread has exited".into()))?
            .map_err(LogError::Writer)
    }
}

fn run(mut writer: LogWriter, mut rx: mpsc::UnboundedReceiver<Cmd>) {
    let mut first_error: Option<String> = None;
    let status = |e: &Option<String>| e.clone().map_or(Ok(()), Err);
    while let Some(cmd) = rx.blocking_recv() {
        match cmd {
            Cmd::Append(env) => {
                if let Err(e) = writer.append_envelope(&env) {
                    tracing::error!(path = %writer.path().display(), seq = env.seq, error = %e, "session log write failed");
                    first_error.get_or_insert_with(|| e.to_string());
                }
            }
            Cmd::Flush(reply) => {
                let _ = reply.send(status(&first_error));
            }
            Cmd::Close(reply) => {
                rx.close();
                if let Err(e) = writer.sync() {
                    first_error.get_or_insert_with(|| e.to_string());
                }
                let _ = reply.send(status(&first_error));
                return;
            }
        }
    }
    let _ = writer.sync();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::NaiveDate;
    use h_agent_core::SessionId;

    use super::*;
    use crate::hlog::read_log;

    #[tokio::test]
    async fn concurrent_appends_keep_seq_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        let id = SessionId::new(
            1,
            NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
        );
        let log = Arc::new(SessionLog::spawn(LogWriter::create(&path, &id).unwrap()).unwrap());
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let log = log.clone();
                std::thread::spawn(move || {
                    for i in 0..50 {
                        log.append(
                            AgentId::new(),
                            Record::UserMessage {
                                text: format!("{t}-{i}"),
                            },
                        )
                        .unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        log.close().await.unwrap();
        let c = read_log(&path).unwrap();
        assert_eq!(c.records.len(), 200);
        assert!(c.records.iter().enumerate().all(|(i, r)| r.seq == i as u64));
        assert!(
            log.append(
                AgentId::new(),
                Record::SessionEnd {
                    reason: String::new()
                }
            )
            .is_err()
        );
    }
}
