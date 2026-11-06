//! A tracing writer that removes credentials before log lines reach disk (DESIGN §12.6).

use std::io::{self, Write};

use h_agent_core::secrets::redact;
use tracing_subscriber::fmt::MakeWriter;

/// Wraps a [`MakeWriter`] so every write is passed through [`redact`].
#[derive(Clone)]
pub struct Redacting<M>(pub M);

impl<'a, M: MakeWriter<'a>> MakeWriter<'a> for Redacting<M> {
    type Writer = RedactingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter(self.0.make_writer())
    }
}

pub struct RedactingWriter<W>(W);

impl<W: Write> Write for RedactingWriter<W> {
    /// The fmt layer writes each formatted event in one call, so secrets are never split.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        self.0.write_all(redact(&text).as_bytes())?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone, Default)]
    struct Buf(Arc<Mutex<Vec<u8>>>);

    impl Write for Buf {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn log_lines_lose_their_secrets() {
        let buf = Buf::default();
        let sink = buf.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(Redacting(move || sink.clone()))
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(key = "sk-abcdefghijklmnopqrstuvwx1234", "calling provider");
            tracing::warn!("auth failed: Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345");
        });
        let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert!(
            out.contains("calling provider") && out.contains("auth failed"),
            "{out}"
        );
        assert!(
            !out.contains("sk-abcdef") && !out.contains("abcdefghijklmnop"),
            "{out}"
        );
        assert_eq!(out.matches("[REDACTED]").count(), 2, "{out}");
    }
}
