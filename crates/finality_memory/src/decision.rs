//! The Memory agent's event decision (DESIGN §7.2): schema, validation, fallback and applying it.

use chrono::{DateTime, Utc};
use finality_core::{AgentId, EventId};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tree::{Event, EventStatus, EventsFile, MAX_SUMMARY_CHARS, RefInfo, clip};

/// One exchange as the Memory agent sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// 0-based exchange number in the session (or step number in a sub-agent thread).
    pub index: u32,
    /// Inclusive log `seq` range.
    pub seq: (u64, u64),
    /// The request that opened the exchange; used for fallback summaries.
    pub headline: String,
    /// Condensed transcript.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Decision {
    /// Same subject: widen the open event, optionally with a better summary.
    Extend {
        #[serde(default)]
        summary: Option<String>,
    },
    /// The subject changed: close the open event and start one for the new exchanges.
    CloseAndNew {
        summary: String,
        #[serde(default)]
        outcome: Option<String>,
    },
    /// The new exchanges cover several subjects: one event per group of exchanges.
    Split { events: Vec<SplitPart> },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SplitPart {
    pub summary: String,
    /// 1-based positions in the list of new exchanges.
    pub exchanges: Vec<u32>,
}

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "enum": ["extend", "close_and_new", "split"] },
            "summary": { "type": "string" },
            "outcome": { "type": "string" },
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" },
                        "exchanges": { "type": "array", "items": { "type": "integer" } }
                    },
                    "required": ["summary", "exchanges"]
                }
            }
        },
        "required": ["action"]
    })
}

/// The JSON object in a model reply, tolerating code fences and surrounding prose.
pub fn extract_json(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    text.get(start..=end)
}

/// Parses and validates a decision about `spans` new exchanges.
pub fn parse(text: &str, spans: usize) -> Result<Decision, String> {
    let json = extract_json(text).ok_or("no JSON object in the reply")?;
    let d: Decision = serde_json::from_str(json).map_err(|e| e.to_string())?;
    d.validate(spans)?;
    Ok(d)
}

/// First sentence of `text`, as a one-line summary.
pub fn heuristic_summary(text: &str) -> String {
    let trimmed = text.trim();
    let end = trimmed
        .char_indices()
        .find(|(i, c)| {
            *c == '\n'
                || (matches!(c, '.' | '?' | '!')
                    && trimmed[i + c.len_utf8()..]
                        .chars()
                        .next()
                        .is_none_or(char::is_whitespace))
        })
        .map_or(
            trimmed.len(),
            |(i, c)| if c == '\n' { i } else { i + c.len_utf8() },
        );
    let s = clip(&trimmed[..end], MAX_SUMMARY_CHARS);
    if s.is_empty() { "(no text)".into() } else { s }
}

impl Decision {
    pub fn validate(&self, spans: usize) -> Result<(), String> {
        let blank = |s: &str| s.trim().is_empty();
        match self {
            Decision::Extend { .. } => Ok(()),
            Decision::CloseAndNew { summary, .. } if blank(summary) => {
                Err("close_and_new needs a summary".into())
            }
            Decision::CloseAndNew { .. } => Ok(()),
            Decision::Split { events } => {
                if events.is_empty() {
                    return Err("split needs at least one event".into());
                }
                if events
                    .iter()
                    .any(|p| blank(&p.summary) || p.exchanges.is_empty())
                {
                    return Err("every split event needs a summary and exchanges".into());
                }
                let mut parts: Vec<Vec<u32>> = events
                    .iter()
                    .map(|p| {
                        let mut x = p.exchanges.clone();
                        x.sort_unstable();
                        x
                    })
                    .collect();
                parts.sort_by_key(|p| p[0]);
                let flat: Vec<u32> = parts.concat();
                let expected: Vec<u32> = (1..=u32::try_from(spans).unwrap_or(u32::MAX)).collect();
                if flat != expected {
                    return Err(format!(
                        "split groups must cover exchanges 1..={spans} once each, in contiguous runs"
                    ));
                }
                Ok(())
            }
        }
    }

    /// Used when the model's answer is invalid twice: keep extending the open event.
    pub fn fallback(spans: &[Span], has_open: bool) -> Self {
        Decision::Extend {
            summary: (!has_open)
                .then(|| heuristic_summary(spans.first().map_or("", |s| s.headline.as_str()))),
        }
    }
}

/// Where new events go.
#[derive(Debug, Clone)]
pub struct Placement<'a> {
    pub info: &'a RefInfo,
    pub query_type: &'a str,
    pub topic: &'a str,
    /// `None` for the Orchestrator's own exchanges.
    pub thread: Option<AgentId>,
    pub now: DateTime<Utc>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Applied {
    pub created: Vec<EventId>,
    pub closed: Vec<EventId>,
    pub extended: Option<EventId>,
}

fn new_event(spans: &[Span], summary: &str, status: EventStatus, place: &Placement<'_>) -> Event {
    let (first, last) = (&spans[0], &spans[spans.len() - 1]);
    Event {
        id: EventId::new(),
        summary: clip(summary, MAX_SUMMARY_CHARS),
        turn_range: (first.seq.0, last.seq.1),
        exchanges: place.thread.is_none().then_some((first.index, last.index)),
        thread: place.thread,
        created_at: place.now,
        updated_at: place.now,
        related: Vec::new(),
        status,
        outcome: None,
    }
}

/// Applies a validated decision for `spans` (oldest first) of one session.
pub fn apply(
    file: &mut EventsFile,
    place: &Placement<'_>,
    spans: &[Span],
    decision: &Decision,
) -> Applied {
    let mut applied = Applied::default();
    if spans.is_empty() {
        return applied;
    }
    let session = place.info.session_id;
    let open = file.open_event(&session, place.thread);
    let mut close_open = |file: &mut EventsFile, outcome: Option<&String>| {
        if let Some(e) = open.and_then(|id| file.event_mut(id)) {
            e.status = EventStatus::Closed;
            e.updated_at = place.now;
            if let Some(o) = outcome.filter(|o| !o.trim().is_empty()) {
                e.outcome = Some(clip(o, MAX_SUMMARY_CHARS));
            }
            applied.closed.push(e.id);
        }
    };
    let add = |file: &mut EventsFile, event: Event| {
        let id = event.id;
        file.ensure_ref(place.query_type, place.topic, place.info)
            .events
            .push(event);
        id
    };
    match decision {
        Decision::Extend { summary } => {
            let summary = summary.as_deref().filter(|s| !s.trim().is_empty());
            match open.and_then(|id| file.event_mut(id)) {
                Some(e) => {
                    let last = &spans[spans.len() - 1];
                    e.turn_range = (
                        e.turn_range.0.min(spans[0].seq.0),
                        e.turn_range.1.max(last.seq.1),
                    );
                    if let Some((a, b)) = e.exchanges {
                        e.exchanges = Some((a.min(spans[0].index), b.max(last.index)));
                    }
                    if let Some(s) = summary {
                        e.summary = clip(s, MAX_SUMMARY_CHARS);
                    }
                    e.updated_at = place.now;
                    applied.extended = Some(e.id);
                }
                None => {
                    let s = summary
                        .map_or_else(|| heuristic_summary(&spans[0].headline), str::to_owned);
                    let id = add(file, new_event(spans, &s, EventStatus::Open, place));
                    applied.created.push(id);
                }
            }
        }
        Decision::CloseAndNew { summary, outcome } => {
            close_open(file, outcome.as_ref());
            let id = add(file, new_event(spans, summary, EventStatus::Open, place));
            applied.created.push(id);
        }
        Decision::Split { events } => {
            close_open(file, None);
            let mut parts: Vec<&SplitPart> = events.iter().collect();
            parts.sort_by_key(|p| p.exchanges.iter().min().copied().unwrap_or(0));
            let count = parts.len();
            for (i, part) in parts.into_iter().enumerate() {
                let lo = part.exchanges.iter().min().copied().unwrap_or(1) as usize;
                let hi = part.exchanges.iter().max().copied().unwrap_or(1) as usize;
                let slice = &spans[(lo - 1).min(spans.len() - 1)..hi.min(spans.len())];
                let status = if i + 1 == count {
                    EventStatus::Open
                } else {
                    EventStatus::Closed
                };
                let id = add(file, new_event(slice, &part.summary, status, place));
                applied.created.push(id);
                if status == EventStatus::Closed {
                    applied.closed.push(id);
                }
            }
        }
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::tests::info;

    fn spans(n: u32) -> Vec<Span> {
        (0..n)
            .map(|i| Span {
                index: i,
                seq: (u64::from(i) * 10 + 1, u64::from(i) * 10 + 9),
                headline: format!("Question {i}? More text."),
                text: String::new(),
            })
            .collect()
    }

    fn place(info: &RefInfo) -> Placement<'_> {
        Placement {
            info,
            query_type: "question",
            topic: "coding/rust",
            thread: None,
            now: Utc::now(),
        }
    }

    #[test]
    fn parses_and_validates() {
        assert_eq!(
            parse("```json\n{\"action\":\"extend\"}\n```", 2),
            Ok(Decision::Extend { summary: None })
        );
        assert!(parse("{\"action\":\"close_and_new\",\"summary\":\" \"}", 2).is_err());
        assert!(parse("{\"action\":\"rename\"}", 2).is_err());
        let ok = r#"{"action":"split","events":[{"summary":"b","exchanges":[3]},{"summary":"a","exchanges":[1,2]}]}"#;
        assert!(parse(ok, 3).is_ok());
        let gap = r#"{"action":"split","events":[{"summary":"a","exchanges":[1]},{"summary":"b","exchanges":[3]}]}"#;
        assert!(parse(gap, 3).is_err());
        let interleaved = r#"{"action":"split","events":[{"summary":"a","exchanges":[1,3]},{"summary":"b","exchanges":[2]}]}"#;
        assert!(parse(interleaved, 3).is_err());
    }

    #[test]
    fn heuristic_takes_the_first_sentence() {
        assert_eq!(
            heuristic_summary("Fix the CRC bug. Then test."),
            "Fix the CRC bug."
        );
        assert_eq!(
            heuristic_summary("version 1.2 is out\nmore"),
            "version 1.2 is out"
        );
        assert_eq!(
            heuristic_summary(&"x".repeat(500)).chars().count(),
            MAX_SUMMARY_CHARS
        );
        assert_eq!(heuristic_summary(""), "(no text)");
    }

    #[test]
    fn extend_creates_then_widens() {
        let info = info(1);
        let mut f = EventsFile::new();
        let s = spans(3);
        let a = apply(
            &mut f,
            &place(&info),
            &s[..1],
            &Decision::fallback(&s[..1], false),
        );
        assert_eq!(a.created.len(), 1);
        let b = apply(
            &mut f,
            &place(&info),
            &s[1..],
            &Decision::Extend { summary: None },
        );
        assert_eq!(b.extended, Some(a.created[0]));
        let e = f.event(a.created[0]).unwrap().event;
        assert_eq!(e.turn_range, (1, 29));
        assert_eq!(e.exchanges, Some((0, 2)));
        assert_eq!(e.summary, "Question 0?");
    }

    #[test]
    fn close_and_new_and_split() {
        let info = info(1);
        let mut f = EventsFile::new();
        let s = spans(4);
        let first = apply(
            &mut f,
            &place(&info),
            &s[..1],
            &Decision::Extend {
                summary: Some("bug".into()),
            },
        );
        let c = apply(
            &mut f,
            &place(&info),
            &s[1..2],
            &Decision::CloseAndNew {
                summary: "deploy".into(),
                outcome: Some("fixed".into()),
            },
        );
        assert_eq!(c.closed, first.created);
        assert_eq!(
            f.event(first.created[0]).unwrap().event.outcome.as_deref(),
            Some("fixed")
        );
        let split = Decision::Split {
            events: vec![
                SplitPart {
                    summary: "docs".into(),
                    exchanges: vec![1],
                },
                SplitPart {
                    summary: "tests".into(),
                    exchanges: vec![2],
                },
            ],
        };
        let d = apply(&mut f, &place(&info), &s[2..], &split);
        assert_eq!(d.created.len(), 2);
        assert_eq!(d.closed, vec![c.created[0], d.created[0]]);
        let open = f.open_event(&info.session_id, None).unwrap();
        assert_eq!(open, d.created[1]);
        assert_eq!(f.event(open).unwrap().event.turn_range, (31, 39));
    }
}
