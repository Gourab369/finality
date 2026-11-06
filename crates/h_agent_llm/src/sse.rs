//! Server-sent events for the streaming providers: a line decoder and a stream driver.

use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;

use futures::{Stream, StreamExt};

use crate::{ChatChunk, ChatStream, LlmError};

pub(crate) type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, LlmError>> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Splits bytes into events: `field: value` lines, `:` comments, a blank line ends an event.
#[derive(Default)]
pub(crate) struct SseDecoder {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

/// Largest event (pending line included) a provider may send before the stream is abandoned.
pub(crate) const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

impl SseDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, LlmError> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n' || b == b'\r') {
            // A trailing "\r" may be the first half of "\r\n"; wait for the next read.
            if self.buf[pos] == b'\r' && pos + 1 == self.buf.len() {
                break;
            }
            let crlf = self.buf[pos] == b'\r' && self.buf[pos + 1] == b'\n';
            let line: Vec<u8> = self.buf.drain(..pos).collect();
            self.buf.drain(..if crlf { 2 } else { 1 });
            self.line(&String::from_utf8_lossy(&line), &mut out);
        }
        let pending = self.buf.len() + self.data.iter().map(String::len).sum::<usize>();
        if pending > MAX_EVENT_BYTES {
            return Err(LlmError::Decode(format!(
                "stream event larger than {} MiB",
                MAX_EVENT_BYTES / (1024 * 1024)
            )));
        }
        Ok(out)
    }

    /// Dispatches an event left without a trailing blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            let rest = String::from_utf8_lossy(&rest);
            self.line(rest.trim_end_matches('\r'), &mut out);
        }
        self.dispatch(&mut out);
        out
    }

    fn line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = self.event.take();
        if self.data.is_empty() {
            return;
        }
        out.push(SseEvent {
            event,
            data: std::mem::take(&mut self.data).join("\n"),
        });
    }
}

/// Turns provider events into chat chunks.
pub(crate) trait SseParser: Send + 'static {
    /// Handles one event; returns true once the response is complete.
    fn event(&mut self, ev: SseEvent, out: &mut VecDeque<Result<ChatChunk, LlmError>>) -> bool;
    /// The connection ended before `event` reported completion.
    fn end(&mut self, out: &mut VecDeque<Result<ChatChunk, LlmError>>);
}

struct Driver<P> {
    bytes: ByteStream,
    idle: Duration,
    decoder: SseDecoder,
    parser: P,
    pending: VecDeque<Result<ChatChunk, LlmError>>,
    finished: bool,
}

impl<P: SseParser> Driver<P> {
    fn feed(&mut self, events: Vec<SseEvent>) {
        for ev in events {
            if self.finished {
                return;
            }
            if self.parser.event(ev, &mut self.pending) {
                self.finished = true;
            }
            if matches!(self.pending.back(), Some(Err(_))) {
                self.finished = true;
            }
        }
    }
}

/// Streams chunks from an SSE body; `idle` bounds the wait for each read.
pub(crate) fn sse_chunks<P: SseParser>(bytes: ByteStream, idle: Duration, parser: P) -> ChatStream {
    let driver = Driver {
        bytes,
        idle,
        decoder: SseDecoder::default(),
        parser,
        pending: VecDeque::new(),
        finished: false,
    };
    Box::pin(futures::stream::unfold(driver, |mut d| async move {
        loop {
            if let Some(item) = d.pending.pop_front() {
                return Some((item, d));
            }
            if d.finished {
                return None;
            }
            match tokio::time::timeout(d.idle, d.bytes.next()).await {
                Err(_) => {
                    d.pending.push_back(Err(LlmError::Timeout));
                    d.finished = true;
                }
                Ok(Some(Ok(b))) => match d.decoder.push(&b) {
                    Ok(events) => d.feed(events),
                    Err(e) => {
                        d.pending.push_back(Err(e));
                        d.finished = true;
                    }
                },
                Ok(Some(Err(e))) => {
                    d.pending.push_back(Err(e));
                    d.finished = true;
                }
                Ok(None) => {
                    let events = d.decoder.finish();
                    d.feed(events);
                    if !d.finished {
                        d.parser.end(&mut d.pending);
                        d.finished = true;
                    }
                }
            }
        }
    }))
}

/// A response body as a byte stream.
pub(crate) fn body_stream(resp: reqwest::Response) -> ByteStream {
    Box::pin(resp.bytes_stream().map(|r| {
        r.map(|b| b.to_vec())
            .map_err(|e| LlmError::Decode(e.to_string()))
    }))
}

#[cfg(test)]
pub(crate) fn byte_stream(parts: Vec<String>) -> ByteStream {
    Box::pin(futures::stream::iter(
        parts.into_iter().map(|p| Ok(p.into_bytes())),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn decodes_events_split_anywhere() {
        let text = "event: message_start\ndata: {\"a\":1}\n\n: keep-alive\n\ndata: line1\ndata: line2\r\n\r\ndata: [DONE]\n\n";
        for split in 0..text.len() {
            let (a, b) = text.split_at(split);
            let mut d = SseDecoder::default();
            let mut events = d.push(a.as_bytes()).unwrap();
            events.extend(d.push(b.as_bytes()).unwrap());
            events.extend(d.finish());
            assert_eq!(
                events,
                [
                    SseEvent {
                        event: Some("message_start".into()),
                        data: "{\"a\":1}".into()
                    },
                    SseEvent {
                        event: None,
                        data: "line1\nline2".into()
                    },
                    SseEvent {
                        event: None,
                        data: "[DONE]".into()
                    },
                ],
                "split at {split}"
            );
        }
    }

    #[test]
    fn trailing_event_without_blank_line() {
        let mut d = SseDecoder::default();
        assert!(d.push(b"data: x").unwrap().is_empty());
        assert_eq!(d.finish()[0].data, "x");
    }

    #[test]
    fn oversized_events_are_refused() {
        let mut d = SseDecoder::default();
        let chunk = vec![b'a'; 1024 * 1024];
        for _ in 0..8 {
            d.push(&chunk).unwrap();
        }
        assert!(matches!(d.push(b"a"), Err(LlmError::Decode(_))));
    }
}
