//! Events tree (DESIGN §7.1): root → query type → topic → session ref → event.

use std::cmp::Reverse;

use chrono::{DateTime, NaiveDate, Utc};
use h_agent_core::{AgentId, EventId, SessionId};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const SEED_QUERY_TYPES: &[&str] = &[
    "command",
    "question",
    "analysis",
    "creation",
    "conversation",
];
pub const MAX_RELATED: usize = 8;
pub const MAX_SUMMARY_CHARS: usize = 160;
const DEFAULT_QUERY_TYPE: &str = "conversation";
const DEFAULT_TOPIC: &str = "general";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventStatus {
    Open,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub summary: String,
    /// Inclusive range of session-log `seq` numbers.
    pub turn_range: (u64, u64),
    /// Inclusive 0-based Orchestrator exchange numbers (main file only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchanges: Option<(u32, u32)>,
    /// Sub-agent thread the event belongs to (role files only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<AgentId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub related: Vec<EventId>,
    pub status: EventStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRef {
    pub session_id: SessionId,
    pub summary: String,
    pub dates: Vec<NaiveDate>,
    pub log_file: String,
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Set once the session-end summary has been written.
    #[serde(default)]
    pub finalized: bool,
    #[serde(default)]
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicNode {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub sessions: Vec<SessionRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryTypeNode {
    pub name: String,
    #[serde(default)]
    pub topics: Vec<TopicNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventsFile {
    pub schema_version: u32,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub types: Vec<QueryTypeNode>,
}

/// What a new `SessionRef` is created with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefInfo {
    pub session_id: SessionId,
    pub log_file: String,
    pub project_id: String,
    pub date: NaiveDate,
}

/// An event together with its place in the tree.
#[derive(Debug, Clone, Copy)]
pub struct EventView<'a> {
    pub query_type: &'a str,
    pub topic: &'a str,
    pub session: &'a SessionRef,
    pub event: &'a Event,
}

/// Lowercase, single spaces, and `a / b` → `a/b`.
pub fn normalize_name(s: &str) -> String {
    s.split('/')
        .map(|seg| {
            seg.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        })
        .filter(|seg| !seg.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

/// One line of at most `max` characters.
pub fn clip(text: &str, max: usize) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= max {
        return line;
    }
    let mut out: String = line.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

impl Default for EventsFile {
    fn default() -> Self {
        Self::new()
    }
}

impl EventsFile {
    pub fn new() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            updated_at: Utc::now(),
            types: SEED_QUERY_TYPES
                .iter()
                .map(|n| QueryTypeNode {
                    name: (*n).to_owned(),
                    topics: Vec::new(),
                })
                .collect(),
        }
    }

    fn all_refs(&self) -> impl Iterator<Item = (&str, &str, &SessionRef)> {
        self.types.iter().flat_map(|t| {
            t.topics.iter().flat_map(move |p| {
                p.sessions
                    .iter()
                    .map(move |s| (t.name.as_str(), p.name.as_str(), s))
            })
        })
    }

    fn all_refs_mut(&mut self) -> impl Iterator<Item = &mut SessionRef> {
        self.types
            .iter_mut()
            .flat_map(|t| t.topics.iter_mut())
            .flat_map(|p| p.sessions.iter_mut())
    }

    pub fn is_empty(&self) -> bool {
        self.all_refs().next().is_none()
    }

    /// Finds or creates the type, topic (matching aliases too) and session ref.
    pub fn ensure_ref(&mut self, query_type: &str, topic: &str, info: &RefInfo) -> &mut SessionRef {
        let qt = Some(normalize_name(query_type))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_QUERY_TYPE.into());
        let tp = Some(normalize_name(topic))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_TOPIC.into());
        let ti = match self.types.iter().position(|t| t.name == qt) {
            Some(i) => i,
            None => {
                self.types.push(QueryTypeNode {
                    name: qt,
                    topics: Vec::new(),
                });
                self.types.len() - 1
            }
        };
        let topics = &mut self.types[ti].topics;
        let pi = match topics
            .iter()
            .position(|t| t.name == tp || t.aliases.iter().any(|a| normalize_name(a) == tp))
        {
            Some(i) => i,
            None => {
                topics.push(TopicNode {
                    name: tp,
                    aliases: Vec::new(),
                    sessions: Vec::new(),
                });
                topics.len() - 1
            }
        };
        let sessions = &mut topics[pi].sessions;
        let si = match sessions
            .iter()
            .position(|s| s.session_id == info.session_id)
        {
            Some(i) => i,
            None => {
                sessions.push(SessionRef {
                    session_id: info.session_id,
                    summary: String::new(),
                    dates: vec![info.date],
                    log_file: info.log_file.clone(),
                    project_id: info.project_id.clone(),
                    outcome: None,
                    finalized: false,
                    events: Vec::new(),
                });
                sessions.len() - 1
            }
        };
        &mut sessions[si]
    }

    /// Every (query type, topic, ref) of `session`.
    pub fn refs(&self, session: &SessionId) -> Vec<(&str, &str, &SessionRef)> {
        self.all_refs()
            .filter(|(_, _, s)| s.session_id == *session)
            .collect()
    }

    pub fn refs_mut(&mut self, session: &SessionId) -> Vec<&mut SessionRef> {
        self.all_refs_mut()
            .filter(|s| s.session_id == *session)
            .collect()
    }

    /// The (query type, topic) the session was first filed under.
    pub fn first_pair(&self, session: &SessionId) -> Option<(String, String)> {
        self.refs(session)
            .into_iter()
            .min_by_key(|(_, _, s)| s.events.first().map_or(u64::MAX, |e| e.turn_range.0))
            .map(|(t, p, _)| (t.to_owned(), p.to_owned()))
    }

    pub fn events(&self) -> Vec<EventView<'_>> {
        self.all_refs()
            .flat_map(|(t, p, s)| {
                s.events.iter().map(move |e| EventView {
                    query_type: t,
                    topic: p,
                    session: s,
                    event: e,
                })
            })
            .collect()
    }

    pub fn event(&self, id: EventId) -> Option<EventView<'_>> {
        self.events().into_iter().find(|v| v.event.id == id)
    }

    pub fn event_mut(&mut self, id: EventId) -> Option<&mut Event> {
        self.all_refs_mut()
            .flat_map(|s| s.events.iter_mut())
            .find(|e| e.id == id)
    }

    /// Latest open event of `session` in `thread` (`None` = the Orchestrator).
    pub fn open_event(&self, session: &SessionId, thread: Option<AgentId>) -> Option<EventId> {
        self.refs(session)
            .into_iter()
            .flat_map(|(_, _, s)| s.events.iter())
            .filter(|e| e.thread == thread && e.status == EventStatus::Open)
            .max_by_key(|e| e.turn_range.1)
            .map(|e| e.id)
    }

    /// Highest log `seq` already summarized for `session` in `thread`.
    pub fn covered_seq(&self, session: &SessionId, thread: Option<AgentId>) -> Option<u64> {
        self.refs(session)
            .into_iter()
            .flat_map(|(_, _, s)| s.events.iter())
            .filter(|e| e.thread == thread)
            .map(|e| e.turn_range.1)
            .max()
    }

    /// Closes the open events of `session`; only those of `thread` when given.
    pub fn close_open(
        &mut self,
        session: &SessionId,
        thread: Option<Option<AgentId>>,
        now: DateTime<Utc>,
    ) -> Vec<EventId> {
        let mut closed = Vec::new();
        for s in self.refs_mut(session) {
            for e in &mut s.events {
                if e.status == EventStatus::Open && thread.is_none_or(|t| e.thread == t) {
                    e.status = EventStatus::Closed;
                    e.updated_at = now;
                    closed.push(e.id);
                }
            }
        }
        closed
    }

    /// Links two events both ways. A full list drops its oldest link (on both of its ends).
    pub fn link(&mut self, a: EventId, b: EventId) -> bool {
        if a == b || self.event(a).is_none() || self.event(b).is_none() {
            return false;
        }
        if self.event(a).is_some_and(|v| v.event.related.contains(&b)) {
            return true;
        }
        for (x, y) in [(a, b), (b, a)] {
            let evicted = self.event_mut(x).and_then(|e| {
                let gone = (e.related.len() >= MAX_RELATED).then(|| e.related.remove(0));
                e.related.push(y);
                gone
            });
            if let Some(g) = evicted.and_then(|g| self.event_mut(g)) {
                g.related.retain(|r| *r != x);
            }
        }
        true
    }

    /// Events that might be similar to `id`: same topic first, then the same topic family, newest first.
    pub fn candidates(&self, id: EventId, limit: usize) -> Vec<EventView<'_>> {
        let Some(target) = self.event(id) else {
            return Vec::new();
        };
        let family = |t: &str| t.split('/').next().unwrap_or("").to_owned();
        let mut out: Vec<(u8, EventView<'_>)> = self
            .events()
            .into_iter()
            .filter(|v| v.event.id != id && !target.event.related.contains(&v.event.id))
            .filter_map(|v| {
                let rank = if v.topic == target.topic && v.query_type == target.query_type {
                    0
                } else if v.topic == target.topic {
                    1
                } else if family(v.topic) == family(target.topic) {
                    2
                } else {
                    return None;
                };
                Some((rank, v))
            })
            .collect();
        out.sort_by_key(|(rank, v)| (*rank, Reverse(v.event.updated_at)));
        out.into_iter().take(limit).map(|(_, v)| v).collect()
    }

    /// (query type, topic, number of sessions) for topics with at least one session.
    pub fn topic_counts(&self) -> Vec<(&str, &str, usize)> {
        self.types
            .iter()
            .flat_map(|t| {
                t.topics
                    .iter()
                    .filter(|p| !p.sessions.is_empty())
                    .map(move |p| (t.name.as_str(), p.name.as_str(), p.sessions.len()))
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use chrono::NaiveDateTime;

    use super::*;

    pub(crate) fn info(serial: u64) -> RefInfo {
        let started =
            NaiveDateTime::parse_from_str("2026-11-05 14:30:12", "%Y-%m-%d %H:%M:%S").unwrap();
        let id = SessionId::new(serial, started);
        RefInfo {
            session_id: id,
            log_file: id.log_file_name(),
            project_id: "p1".into(),
            date: started.date(),
        }
    }

    pub(crate) fn event(seq: u64) -> Event {
        Event {
            id: EventId::new(),
            summary: format!("event at {seq}"),
            turn_range: (seq, seq),
            exchanges: None,
            thread: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            related: Vec::new(),
            status: EventStatus::Open,
            outcome: None,
        }
    }

    #[test]
    fn seeds_types_and_reuses_topics_and_aliases() {
        let mut f = EventsFile::new();
        assert_eq!(f.types.len(), 5);
        assert!(f.is_empty());
        f.ensure_ref("Question", "Coding / Rust", &info(1));
        f.types[1].topics[0].aliases.push("rust-lang".into());
        f.ensure_ref("question", "rust-lang", &info(2));
        f.ensure_ref("question", "coding/rust", &info(1));
        let topics = &f.types[1].topics;
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].name, "coding/rust");
        assert_eq!(topics[0].sessions.len(), 2);
        f.ensure_ref("brainstorm", "", &info(3));
        assert_eq!(f.types.last().unwrap().name, "brainstorm");
        assert_eq!(f.types.last().unwrap().topics[0].name, "general");
    }

    #[test]
    fn links_are_bidirectional_and_capped() {
        let mut f = EventsFile::new();
        let r = f.ensure_ref("question", "physics", &info(1));
        for seq in 0..10 {
            r.events.push(event(seq));
        }
        let ids: Vec<EventId> = f.events().iter().map(|v| v.event.id).collect();
        for other in &ids[1..] {
            assert!(f.link(ids[0], *other));
        }
        let first = f.event(ids[0]).unwrap().event.related.clone();
        assert_eq!(first.len(), MAX_RELATED);
        assert!(!first.contains(&ids[1]), "oldest link evicted");
        assert!(
            f.event(ids[1]).unwrap().event.related.is_empty(),
            "and removed on the other end"
        );
        for id in &first {
            assert_eq!(f.event(*id).unwrap().event.related, vec![ids[0]]);
        }
        assert!(!f.link(ids[0], ids[0]));
    }

    #[test]
    fn open_covered_and_close() {
        let mut f = EventsFile::new();
        let s = info(1);
        let r = f.ensure_ref("command", "git", &s);
        r.events.push(event(3));
        let mut threaded = event(9);
        threaded.thread = Some(AgentId::new());
        r.events.push(threaded);
        assert_eq!(f.covered_seq(&s.session_id, None), Some(3));
        let open = f.open_event(&s.session_id, None).unwrap();
        assert_eq!(
            f.close_open(&s.session_id, Some(None), Utc::now()),
            vec![open]
        );
        assert!(f.open_event(&s.session_id, None).is_none());
        assert_eq!(f.close_open(&s.session_id, None, Utc::now()).len(), 1);
    }

    #[test]
    fn candidates_prefer_same_topic() {
        let mut f = EventsFile::new();
        f.ensure_ref("question", "coding/rust", &info(1))
            .events
            .push(event(1));
        f.ensure_ref("question", "coding/go", &info(2))
            .events
            .push(event(1));
        f.ensure_ref("question", "coding/rust", &info(3))
            .events
            .push(event(1));
        f.ensure_ref("question", "cooking", &info(4))
            .events
            .push(event(1));
        let target = f.refs(&info(1).session_id)[0].2.events[0].id;
        let c: Vec<&str> = f.candidates(target, 10).iter().map(|v| v.topic).collect();
        assert_eq!(c, ["coding/rust", "coding/go"]);
    }

    #[test]
    fn round_trips_through_json() {
        let mut f = EventsFile::new();
        f.ensure_ref("question", "physics", &info(1))
            .events
            .push(event(4));
        let text = serde_json::to_string(&f).unwrap();
        assert!(text.contains("\"schema_version\":1"));
        let back: EventsFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back, f);
    }
}
