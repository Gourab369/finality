//! The memory digest given to the Orchestrator at session start (DESIGN §7.5).

use std::collections::BTreeMap;

use finality_core::SessionId;

use crate::tree::{EventsFile, SessionRef};

const RECENT_SESSIONS: usize = 5;

/// Topics with counts, recent sessions of this project and the user profile, within `budget_tokens`.
/// Empty when there is nothing to tell.
pub fn digest(
    file: &EventsFile,
    project_id: &str,
    profile: Option<&str>,
    budget_tokens: u32,
) -> String {
    let mut out = String::new();
    let mut by_type: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (t, p, n) in file.topic_counts() {
        by_type.entry(t).or_default().push(format!("{p} ({n})"));
    }
    if !by_type.is_empty() {
        out.push_str("Known topics (sessions):\n");
        for (t, topics) in by_type {
            out.push_str(&format!("- {t}: {}\n", topics.join(", ")));
        }
    }

    let mut sessions: BTreeMap<SessionId, &SessionRef> = BTreeMap::new();
    for v in file.events() {
        if v.session.project_id == project_id {
            sessions.entry(v.session.session_id).or_insert(v.session);
        }
    }
    if !sessions.is_empty() {
        out.push_str("Recent sessions in this project:\n");
        for (id, s) in sessions.iter().rev().take(RECENT_SESSIONS) {
            let date = s.dates.last().map(ToString::to_string).unwrap_or_default();
            let summary = if s.summary.is_empty() {
                s.events.first().map_or("", |e| e.summary.as_str())
            } else {
                &s.summary
            };
            out.push_str(&format!("- {id} ({date}): {summary}\n"));
        }
    }

    if let Some(p) = profile.map(str::trim).filter(|p| !p.is_empty()) {
        out.push_str("User profile:\n");
        out.push_str(p);
        out.push('\n');
    }

    let max_chars = budget_tokens as usize * 4;
    if out.chars().count() > max_chars {
        out = out.chars().take(max_chars.saturating_sub(1)).collect();
        out.push('…');
    }
    out.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::tests::{event, info};

    #[test]
    fn lists_topics_sessions_and_profile_within_budget() {
        let mut f = EventsFile::new();
        assert_eq!(digest(&f, "p1", None, 100), "");
        for serial in 1..=7 {
            let r = f.ensure_ref("question", "coding/rust", &info(serial));
            r.summary = format!("session {serial}");
            r.events.push(event(1));
        }
        let mut other = info(8);
        other.project_id = "p2".into();
        f.ensure_ref("command", "git", &other).events.push(event(1));

        let d = digest(&f, "p1", Some("- prefers short answers"), 1000);
        assert!(d.contains("- command: git (1)"));
        assert!(d.contains("- question: coding/rust (7)"));
        assert!(d.contains("session 7") && d.contains("session 3") && !d.contains("session 2"));
        assert!(!d.contains("000008"), "other projects are left out");
        assert!(d.ends_with("- prefers short answers"));

        assert!(digest(&f, "p1", None, 10).chars().count() <= 40);
    }
}
