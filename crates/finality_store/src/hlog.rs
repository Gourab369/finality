//! `.hlog` session log: header + length-prefixed, CRC-checked, zstd-compressed postcard records (DESIGN §5.2).

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use finality_core::{AgentId, SessionId};

use crate::record::{Record, RecordEnvelope};

pub const MAGIC: &[u8; 8] = b"FINLOG01";
pub const FORMAT_VERSION: u16 = 1;
const FRAME_HEADER_LEN: usize = 8;
const MAX_RECORD_LEN: usize = 64 << 20;
/// Decompressed size limit per record; keeps a damaged or hostile file from exhausting memory.
const MAX_DECODED_LEN: usize = 256 << 20;
const ZSTD_LEVEL: i32 = 3;

#[derive(Debug, thiserror::Error)]
pub enum LogError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("not a session log: {0}")]
    BadHeader(String),
    #[error("unsupported session log format version {0}")]
    UnsupportedVersion(u16),
    #[error("session log corrupt at byte {offset}: {reason}")]
    Corrupt { offset: u64, reason: String },
    #[error("cannot encode record: {0}")]
    Encode(String),
    #[error("session log writer stopped: {0}")]
    Writer(String),
}

pub fn encode_payload(env: &RecordEnvelope) -> Result<Vec<u8>, LogError> {
    postcard::to_allocvec(env).map_err(|e| LogError::Encode(e.to_string()))
}

pub fn encode_frame(env: &RecordEnvelope) -> Result<Vec<u8>, LogError> {
    let compressed = zstd::bulk::compress(&encode_payload(env)?, ZSTD_LEVEL)?;
    let len = u32::try_from(compressed.len())
        .ok()
        .filter(|&l| l as usize <= MAX_RECORD_LEN)
        .ok_or_else(|| {
            LogError::Encode(format!("record too large ({} bytes)", compressed.len()))
        })?;
    let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + compressed.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&crc32fast::hash(&compressed).to_le_bytes());
    frame.extend_from_slice(&compressed);
    Ok(frame)
}

fn decode_payload(compressed: &[u8]) -> Result<RecordEnvelope, String> {
    use std::io::Read as _;
    let mut raw = Vec::new();
    zstd::stream::read::Decoder::new(compressed)
        .map_err(|e| e.to_string())?
        .take(MAX_DECODED_LEN as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() > MAX_DECODED_LEN {
        return Err(format!("record larger than {} MiB", MAX_DECODED_LEN >> 20));
    }
    postcard::from_bytes(&raw).map_err(|e| e.to_string())
}

fn header_bytes(session_id: &str) -> Result<Vec<u8>, LogError> {
    let id_len = u16::try_from(session_id.len())
        .map_err(|_| LogError::Encode("session id too long".into()))?;
    let mut h = Vec::with_capacity(12 + session_id.len());
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    h.extend_from_slice(&id_len.to_le_bytes());
    h.extend_from_slice(session_id.as_bytes());
    Ok(h)
}

/// Returns `(session_id, header_len)`.
fn parse_header(bytes: &[u8]) -> Result<(String, usize), LogError> {
    if bytes.len() < 12 || &bytes[..8] != MAGIC {
        return Err(LogError::BadHeader("missing magic".into()));
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != FORMAT_VERSION {
        return Err(LogError::UnsupportedVersion(version));
    }
    let id_len = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
    let id = bytes
        .get(12..12 + id_len)
        .ok_or_else(|| LogError::BadHeader("truncated header".into()))?;
    let id = String::from_utf8(id.to_vec())
        .map_err(|_| LogError::BadHeader("session id not UTF-8".into()))?;
    Ok((id, 12 + id_len))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tail {
    Clean,
    /// Bytes from `offset` on are an incomplete or damaged final record.
    Truncated {
        offset: u64,
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct LogContents {
    pub session_id: String,
    pub records: Vec<RecordEnvelope>,
    /// Length of the file prefix holding the header and all valid records.
    pub valid_len: u64,
    pub tail: Tail,
}

impl LogContents {
    pub fn last_seq(&self) -> Option<u64> {
        self.records.last().map(|r| r.seq)
    }

    /// Records with `start <= seq <= end`.
    pub fn range(&self, start: u64, end: u64) -> &[RecordEnvelope] {
        let lo = self.records.partition_point(|r| r.seq < start);
        let hi = self.records.partition_point(|r| r.seq <= end);
        &self.records[lo..hi.max(lo)]
    }

    /// Each exchange starts at a `UserMessage` and runs until the next one.
    pub fn exchanges(&self) -> Vec<&[RecordEnvelope]> {
        let starts: Vec<usize> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.body, Record::UserMessage { .. }))
            .map(|(i, _)| i)
            .collect();
        starts
            .iter()
            .enumerate()
            .map(|(n, &s)| {
                let e = starts.get(n + 1).copied().unwrap_or(self.records.len());
                &self.records[s..e]
            })
            .collect()
    }

    pub fn last_exchanges(&self, k: usize) -> Vec<&[RecordEnvelope]> {
        let all = self.exchanges();
        let skip = all.len().saturating_sub(k);
        all.into_iter().skip(skip).collect()
    }
}

pub fn read_log(path: &Path) -> Result<LogContents, LogError> {
    let bytes = std::fs::read(path)?;
    let (session_id, header_len) = parse_header(&bytes)?;
    let total = bytes.len();
    let mut pos = header_len;
    let mut records = Vec::new();
    let mut tail = Tail::Clean;

    while pos < total {
        let truncated = |reason: &str| Tail::Truncated {
            offset: pos as u64,
            reason: reason.to_owned(),
        };
        if total - pos < FRAME_HEADER_LEN {
            tail = truncated("partial frame header");
            break;
        }
        let len = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let crc = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]);
        let end = pos + FRAME_HEADER_LEN + len;
        // A writer never produces such a length, so this is corruption, not a torn write.
        if len > MAX_RECORD_LEN {
            return Err(LogError::Corrupt {
                offset: pos as u64,
                reason: format!("impossible frame length {len}"),
            });
        }
        if end > total {
            tail = truncated("frame extends past end of file");
            break;
        }
        let payload = &bytes[pos + FRAME_HEADER_LEN..end];
        let problem = if crc32fast::hash(payload) != crc {
            Some("checksum mismatch".to_owned())
        } else {
            match decode_payload(payload) {
                Ok(env) => {
                    records.push(env);
                    None
                }
                Err(e) => Some(format!("undecodable record: {e}")),
            }
        };
        if let Some(reason) = problem {
            if end == total {
                tail = truncated(&reason);
                break;
            }
            return Err(LogError::Corrupt {
                offset: pos as u64,
                reason,
            });
        }
        pos = end;
    }

    let valid_len = match &tail {
        Tail::Clean => total as u64,
        Tail::Truncated { offset, .. } => *offset,
    };
    Ok(LogContents {
        session_id,
        records,
        valid_len,
        tail,
    })
}

/// Synchronous appender. Every record is written with a single `write_all`, so it reaches the OS immediately.
pub struct LogWriter {
    file: File,
    path: PathBuf,
    next_seq: u64,
}

impl LogWriter {
    /// Fails if the file already exists.
    pub fn create(path: &Path, session_id: &SessionId) -> Result<Self, LogError> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&header_bytes(&session_id.to_string())?)?;
        file.sync_data()?;
        Ok(Self {
            file,
            path: path.to_owned(),
            next_seq: 0,
        })
    }

    /// Opens an existing log for appending, dropping a damaged tail first.
    pub fn open_append(path: &Path) -> Result<(Self, LogContents), LogError> {
        let contents = read_log(path)?;
        if let Tail::Truncated { offset, reason } = &contents.tail {
            tracing::warn!(path = %path.display(), offset, reason, "truncating damaged session log tail");
            // Append-only handles lack the write access set_len needs on Windows.
            OpenOptions::new()
                .write(true)
                .open(path)?
                .set_len(contents.valid_len)?;
        }
        let file = OpenOptions::new().append(true).open(path)?;
        let next_seq = contents.last_seq().map_or(0, |s| s + 1);
        Ok((
            Self {
                file,
                path: path.to_owned(),
                next_seq,
            },
            contents,
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn append(&mut self, agent: AgentId, body: Record) -> Result<u64, LogError> {
        let env = RecordEnvelope {
            seq: self.next_seq,
            ts_unix_ms: chrono::Utc::now().timestamp_millis(),
            agent,
            body,
        };
        self.append_envelope(&env)?;
        Ok(env.seq)
    }

    pub fn append_envelope(&mut self, env: &RecordEnvelope) -> Result<(), LogError> {
        self.file.write_all(&encode_frame(env)?)?;
        self.next_seq = env.seq + 1;
        Ok(())
    }

    pub fn sync(&self) -> Result<(), LogError> {
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    fn sid() -> SessionId {
        SessionId::new(
            7,
            NaiveDate::from_ymd_opt(2026, 11, 5)
                .unwrap()
                .and_hms_opt(9, 0, 0)
                .unwrap(),
        )
    }

    fn user(text: &str) -> Record {
        Record::UserMessage { text: text.into() }
    }

    #[test]
    fn round_trip_and_replay_helpers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        let agent = AgentId::new();
        let mut w = LogWriter::create(&path, &sid()).unwrap();
        w.append(
            agent,
            Record::SessionEnd {
                reason: "preamble".into(),
            },
        )
        .unwrap();
        for i in 0..3 {
            w.append(agent, user(&format!("q{i}"))).unwrap();
            w.append(
                agent,
                Record::AssistantMessage {
                    text: format!("a{i}"),
                    thinking: None,
                },
            )
            .unwrap();
        }
        let c = read_log(&path).unwrap();
        assert_eq!(c.session_id, "000007_20261105-090000");
        assert_eq!(c.tail, Tail::Clean);
        assert_eq!(c.records.len(), 7);
        assert_eq!(c.last_seq(), Some(6));
        assert_eq!(
            c.range(2, 4).iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert!(c.range(9, 12).is_empty());
        assert_eq!(c.exchanges().len(), 3);
        let last = c.last_exchanges(2);
        assert_eq!(last.len(), 2);
        assert_eq!(last[0][0].body, user("q1"));
        assert_eq!(last[1].len(), 2);
    }

    #[test]
    fn create_refuses_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        LogWriter::create(&path, &sid()).unwrap();
        assert!(LogWriter::create(&path, &sid()).is_err());
    }

    #[test]
    fn rejects_foreign_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.hlog");
        std::fs::write(&path, b"not a log at all").unwrap();
        assert!(matches!(read_log(&path), Err(LogError::BadHeader(_))));
    }

    #[test]
    fn mid_file_corruption_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        let mut w = LogWriter::create(&path, &sid()).unwrap();
        for i in 0..3 {
            w.append(AgentId::new(), user(&format!("m{i}"))).unwrap();
        }
        drop(w);
        let mut bytes = std::fs::read(&path).unwrap();
        let header_len = 12 + sid().to_string().len();
        bytes[header_len + FRAME_HEADER_LEN] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(read_log(&path), Err(LogError::Corrupt { .. })));
    }

    #[test]
    fn damaged_last_record_is_a_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        let mut w = LogWriter::create(&path, &sid()).unwrap();
        w.append(AgentId::new(), user("keep")).unwrap();
        w.append(AgentId::new(), user("damage")).unwrap();
        drop(w);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let c = read_log(&path).unwrap();
        assert_eq!(c.records.len(), 1);
        assert!(matches!(c.tail, Tail::Truncated { .. }));
    }

    #[test]
    fn impossible_length_is_corruption_not_a_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.hlog");
        let mut w = LogWriter::create(&path, &sid()).unwrap();
        w.append(AgentId::new(), user("a")).unwrap();
        w.append(AgentId::new(), user("b")).unwrap();
        drop(w);
        let mut bytes = std::fs::read(&path).unwrap();
        let header_len = 12 + sid().to_string().len();
        bytes[header_len..header_len + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(read_log(&path), Err(LogError::Corrupt { .. })));
        assert!(
            LogWriter::open_append(&path).is_err(),
            "nothing is truncated"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
}
