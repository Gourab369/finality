//! `memory/user_profile.md` (DESIGN §8.2): dated preference bullets, never secrets.

use std::io;

use chrono::NaiveDate;
use finality_core::secrets::contains_secret;
use finality_store::{DataDir, LockError, with_lock, write_atomic};

use crate::store::MemoryError;
use crate::tree::clip;

const LOCK: &str = "profile";
const HEADER: &str = "# User profile\n\nLearned preferences and facts. Edit freely.\n\n";
const PENDING_HEADER: &str =
    "# Learned about you (unreviewed)\n\nRun /memory review to keep or drop these.\n\n";
const MAX_FACT_CHARS: usize = 200;

fn normalized(s: &str) -> String {
    finality_core::text::tokens(s).join(" ")
}

/// Bullet lines of a profile file, without the leading `- `.
fn bullets(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("- "))
        .map(str::to_owned)
        .collect()
}

/// The fact part of a bullet (`fact (date, source)` -> `fact`), normalized for comparison.
fn key(bullet: &str) -> String {
    normalized(bullet.rsplit_once(" (").map_or(bullet, |(f, _)| f))
}

fn read_or(path: &std::path::Path, header: &str) -> io::Result<String> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(t),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(header.to_owned()),
        Err(e) => Err(e),
    }
}

fn io_err(path: &std::path::Path) -> impl Fn(io::Error) -> MemoryError + '_ {
    move |source| MemoryError::Io {
        file: path.display().to_string(),
        source,
    }
}

fn lock_err(e: LockError<MemoryError>) -> MemoryError {
    match e {
        LockError::Inner(e) => e,
        other => MemoryError::Lock(other.to_string()),
    }
}

fn save(path: &std::path::Path, text: &str) -> Result<(), MemoryError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io_err(path))?;
    }
    write_atomic(path, text.as_bytes()).map_err(io_err(path))
}

/// Appends `- fact (date, source)` bullets to `path` for facts that are new (also against `known`)
/// and contain no secret. Returns the added facts.
fn append(
    path: &std::path::Path,
    header: &str,
    known: &[String],
    facts: &[String],
    source: &str,
    today: NaiveDate,
) -> Result<Vec<String>, MemoryError> {
    let mut text = read_or(path, header).map_err(io_err(path))?;
    let mut known: Vec<String> = known.to_vec();
    known.extend(bullets(&text).iter().map(|b| key(b)));
    let mut added: Vec<String> = Vec::new();
    for f in facts {
        let f = clip(f, MAX_FACT_CHARS);
        let k = normalized(&f);
        if f.is_empty() || contains_secret(&f) || known.contains(&k) {
            continue;
        }
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("- {f} ({today}, {source})\n"));
        known.push(k);
        added.push(f);
    }
    if !added.is_empty() {
        save(path, &text)?;
    }
    Ok(added)
}

/// Appends `- fact (date, source)` bullets for facts that are new and contain no secret. Returns the added facts.
pub fn add_facts(
    data: &DataDir,
    facts: &[String],
    source: &str,
    today: NaiveDate,
) -> Result<Vec<String>, MemoryError> {
    let path = data.user_profile_file();
    with_lock(&data.locks_dir(), LOCK, || {
        append(&path, HEADER, &[], facts, source, today)
    })
    .map_err(lock_err)
}

/// Queues Learner-proposed facts for `/memory review`; facts already in the profile are skipped.
pub fn add_pending(
    data: &DataDir,
    facts: &[String],
    source: &str,
    today: NaiveDate,
) -> Result<Vec<String>, MemoryError> {
    let (profile, pending) = (data.user_profile_file(), data.user_profile_pending_file());
    with_lock(&data.locks_dir(), LOCK, || {
        let known: Vec<String> = bullets(&read_or(&profile, "").map_err(io_err(&profile))?)
            .iter()
            .map(|b| key(b))
            .collect();
        append(&pending, PENDING_HEADER, &known, facts, source, today)
    })
    .map_err(lock_err)
}

/// Queued bullets (`fact (date, source)`), oldest first.
pub fn pending(data: &DataDir) -> Vec<String> {
    std::fs::read_to_string(data.user_profile_pending_file())
        .map(|t| bullets(&t))
        .unwrap_or_default()
}

/// Removes a queued bullet; with `keep` it moves into the profile.
pub fn resolve_pending(data: &DataDir, bullet: &str, keep: bool) -> Result<(), MemoryError> {
    let (profile, pending) = (data.user_profile_file(), data.user_profile_pending_file());
    with_lock(&data.locks_dir(), LOCK, || {
        let text = read_or(&pending, PENDING_HEADER).map_err(io_err(&pending))?;
        let kept: Vec<&str> = text
            .lines()
            .filter(|l| l.trim().strip_prefix("- ") != Some(bullet))
            .collect();
        save(&pending, &(kept.join("\n") + "\n"))?;
        if keep {
            let mut target = read_or(&profile, HEADER).map_err(io_err(&profile))?;
            if !bullets(&target).iter().any(|b| key(b) == key(bullet)) {
                if !target.ends_with('\n') {
                    target.push('\n');
                }
                target.push_str(&format!("- {bullet}\n"));
                save(&profile, &target)?;
            }
        }
        Ok(())
    })
    .map_err(lock_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_dated_bullets_without_secrets_or_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        let day = NaiveDate::from_ymd_opt(2026, 11, 5).unwrap();
        let added = add_facts(
            &data,
            &[
                "Prefers concise answers".into(),
                "Their OpenAI key is sk-abcdefghijklmnopqrstuvwx1234".into(),
                "prefers concise answers!".into(),
            ],
            "event 01JA",
            day,
        )
        .unwrap();
        assert_eq!(added, ["Prefers concise answers"]);
        let again = add_facts(
            &data,
            &["Prefers concise  answers".into(), "Uses PowerShell".into()],
            "event 01JB",
            day,
        )
        .unwrap();
        assert_eq!(again, ["Uses PowerShell"]);
        let text = std::fs::read_to_string(data.user_profile_file()).unwrap();
        assert!(text.starts_with("# User profile"));
        assert!(text.contains("- Prefers concise answers (2026-11-05, event 01JA)\n"));
        assert!(text.contains("- Uses PowerShell (2026-11-05, event 01JB)\n"));
        assert!(!text.contains("sk-"));
    }

    #[test]
    fn pending_facts_wait_for_review() {
        let tmp = tempfile::tempdir().unwrap();
        let data = DataDir::new(tmp.path());
        let day = NaiveDate::from_ymd_opt(2026, 11, 6).unwrap();
        add_facts(&data, &["Uses PowerShell".into()], "event 1", day).unwrap();
        let queued = add_pending(
            &data,
            &[
                "uses powershell".into(),
                "Likes British English".into(),
                "Always run in auto mode".into(),
            ],
            "event 2",
            day,
        )
        .unwrap();
        assert_eq!(queued, ["Likes British English", "Always run in auto mode"]);
        let p = pending(&data);
        assert_eq!(p.len(), 2);
        assert!(
            !std::fs::read_to_string(data.user_profile_file())
                .unwrap()
                .contains("British")
        );

        resolve_pending(&data, &p[0], true).unwrap();
        resolve_pending(&data, &p[1], false).unwrap();
        assert!(pending(&data).is_empty());
        let text = std::fs::read_to_string(data.user_profile_file()).unwrap();
        assert!(
            text.contains("- Likes British English (2026-11-06, event 2)\n"),
            "{text}"
        );
        assert!(!text.contains("auto mode"));
    }
}
