//! Plan-of-action lookup table (DESIGN §8.1): proven plan templates with run statistics, kept by the Scorer.

use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use h_agent_core::config::{ScorerConfig, ScorerWeights};
use h_agent_core::text::bm25;
use h_agent_store::{DataDir, LockError, with_lock, write_atomic};
use serde::{Deserialize, Serialize};

use crate::store::MemoryError;

const LOCK: &str = "lookup";
pub const MAX_SOURCE_PLANS: usize = 20;
const MAX_DURATIONS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LookupStep {
    pub title: String,
    /// `worker`, `research` or `self`.
    pub role: String,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(default)]
    pub depends_on: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LookupStats {
    pub runs: u32,
    pub success: u32,
    pub partial: u32,
    pub failed: u32,
    pub median_duration_s: u64,
    pub avg_retries: f64,
    /// Recent run durations, for the median.
    #[serde(default)]
    pub durations_s: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookupEntry {
    pub id: String,
    pub event_signature: String,
    pub query_type: String,
    pub topic: String,
    pub steps: Vec<LookupStep>,
    pub skills_used: Vec<String>,
    pub tools_used: Vec<String>,
    pub stats: LookupStats,
    pub score: f64,
    pub source_plans: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

fn one() -> u32 {
    1
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LookupTable {
    #[serde(default = "one")]
    pub schema_version: u32,
    #[serde(default)]
    pub entries: Vec<LookupEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    Success,
    Partial,
    Failed,
}

/// One finished plan, as the table records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRun {
    pub plan_id: String,
    pub outcome: RunOutcome,
    pub duration_s: u64,
    /// Attempts beyond the first, summed over the plan's tasks.
    pub retries: u32,
}

fn sorted(items: &[String]) -> Vec<String> {
    let mut v = items.to_vec();
    v.sort();
    v.dedup();
    v
}

/// Identifies a strategy: each step's role, tools and skills, in order. Titles are ignored because they vary between runs.
pub fn fingerprint(steps: &[LookupStep]) -> String {
    steps
        .iter()
        .map(|s| {
            format!(
                "{}:{}:{}",
                s.role,
                sorted(&s.tools).join(","),
                sorted(&s.skills).join(",")
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Lowercase words, so trivially different signatures compare equal.
pub fn normalize_signature(s: &str) -> String {
    h_agent_core::text::tokens(s).join(" ")
}

fn median(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2
    } else {
        v[mid]
    }
}

impl LookupEntry {
    pub fn new(
        signature: &str,
        query_type: &str,
        topic: &str,
        steps: Vec<LookupStep>,
        now: DateTime<Utc>,
    ) -> Self {
        let skills_used = sorted(
            &steps
                .iter()
                .flat_map(|s| s.skills.clone())
                .collect::<Vec<_>>(),
        );
        let tools_used = sorted(
            &steps
                .iter()
                .flat_map(|s| s.tools.clone())
                .collect::<Vec<_>>(),
        );
        Self {
            id: h_agent_core::PlanId::new().to_string(),
            event_signature: signature.to_owned(),
            query_type: query_type.to_owned(),
            topic: topic.to_owned(),
            steps,
            skills_used,
            tools_used,
            stats: LookupStats::default(),
            score: 0.0,
            source_plans: Vec::new(),
            updated_at: now,
        }
    }

    pub fn record(&mut self, run: &PlanRun, now: DateTime<Utc>) {
        let s = &mut self.stats;
        let previous = f64::from(s.runs);
        s.runs += 1;
        match run.outcome {
            RunOutcome::Success => s.success += 1,
            RunOutcome::Partial => s.partial += 1,
            RunOutcome::Failed => s.failed += 1,
        }
        s.avg_retries = (s.avg_retries * previous + f64::from(run.retries)) / f64::from(s.runs);
        s.durations_s.push(run.duration_s);
        if s.durations_s.len() > MAX_DURATIONS {
            s.durations_s.remove(0);
        }
        s.median_duration_s = median(&s.durations_s);
        if !self.source_plans.contains(&run.plan_id) {
            self.source_plans.push(run.plan_id.clone());
            if self.source_plans.len() > MAX_SOURCE_PLANS {
                self.source_plans.remove(0);
            }
        }
        self.updated_at = now;
    }

    /// The reuse rule: score and run count both meet the configured minimums.
    pub fn reusable(&self, cfg: &ScorerConfig) -> bool {
        self.score >= cfg.reuse_min_score && self.stats.runs >= cfg.reuse_min_runs
    }

    fn search_text(&self) -> String {
        let titles: Vec<&str> = self.steps.iter().map(|s| s.title.as_str()).collect();
        format!(
            "{} {} {} {}",
            self.event_signature,
            self.query_type,
            self.topic,
            titles.join(" ")
        )
    }
}

impl LookupTable {
    pub fn entry(&self, id: &str) -> Option<&LookupEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn entry_mut(&mut self, id: &str) -> Option<&mut LookupEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// `w_success·((S + 0.5·P)/runs) + w_speed·(best_median/this_median) + w_retries·(1 − avg_retries/max_retries)`.
    /// The best median is taken over entries with the same type, topic and signature (alternative strategies).
    pub fn rescore(&mut self, weights: &ScorerWeights, max_retries: u32) {
        let key = |e: &LookupEntry| {
            (
                e.query_type.clone(),
                e.topic.clone(),
                normalize_signature(&e.event_signature),
            )
        };
        let keys: Vec<_> = self.entries.iter().map(key).collect();
        let bests: Vec<u64> = keys
            .iter()
            .map(|k| {
                self.entries
                    .iter()
                    .zip(&keys)
                    .filter(|(e, other)| {
                        *other == k && e.stats.runs > 0 && e.stats.median_duration_s > 0
                    })
                    .map(|(e, _)| e.stats.median_duration_s)
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        for (e, best) in self.entries.iter_mut().zip(bests) {
            let s = &e.stats;
            if s.runs == 0 {
                e.score = 0.0;
                continue;
            }
            let success = (f64::from(s.success) + 0.5 * f64::from(s.partial)) / f64::from(s.runs);
            let speed = if s.median_duration_s == 0 || best == 0 {
                1.0
            } else {
                best as f64 / s.median_duration_s as f64
            };
            let retries = (1.0 - s.avg_retries / f64::from(max_retries.max(1))).clamp(0.0, 1.0);
            e.score = weights.success * success + weights.speed * speed + weights.retries * retries;
        }
    }

    /// Entries relevant to `query`, most relevant first.
    pub fn search(&self, query: &str, limit: usize) -> Vec<&LookupEntry> {
        let docs: Vec<String> = self.entries.iter().map(LookupEntry::search_text).collect();
        bm25(&docs, query)
            .into_iter()
            .take(limit)
            .map(|(i, _)| &self.entries[i])
            .collect()
    }

    /// The highest-scoring reusable entry of this type and topic that is relevant to `query`.
    pub fn best_template(
        &self,
        query: &str,
        query_type: &str,
        topic: &str,
        cfg: &ScorerConfig,
    ) -> Option<&LookupEntry> {
        self.search(query, self.entries.len())
            .into_iter()
            .filter(|e| e.query_type == query_type && e.topic == topic && e.reusable(cfg))
            .max_by(|a, b| a.score.total_cmp(&b.score))
    }
}

#[derive(Debug, Clone)]
pub struct LookupStore {
    path: PathBuf,
    locks_dir: PathBuf,
}

impl LookupStore {
    pub fn new(dir: &DataDir) -> Self {
        Self {
            path: dir.plan_lookup_file(),
            locks_dir: dir.locks_dir(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn io_err(&self, source: io::Error) -> MemoryError {
        MemoryError::Io {
            file: self.path.display().to_string(),
            source,
        }
    }

    pub fn load(&self) -> Result<LookupTable, MemoryError> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| MemoryError::Json {
                file: self.path.display().to_string(),
                source,
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(LookupTable::default()),
            Err(e) => Err(self.io_err(e)),
        }
    }

    /// Re-reads the table under the lock, applies `f` and writes it back atomically.
    pub fn update<T>(&self, f: impl FnOnce(&mut LookupTable) -> T) -> Result<T, MemoryError> {
        with_lock(&self.locks_dir, LOCK, || {
            let mut table = self.load()?;
            let out = f(&mut table);
            table.schema_version = 1;
            let bytes = serde_json::to_vec_pretty(&table).map_err(|source| MemoryError::Json {
                file: self.path.display().to_string(),
                source,
            })?;
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| self.io_err(e))?;
            }
            write_atomic(&self.path, &bytes).map_err(|e| self.io_err(e))?;
            Ok(out)
        })
        .map_err(|e| match e {
            LockError::Inner(e) => e,
            other => MemoryError::Lock(other.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(title: &str, role: &str, tools: &[&str]) -> LookupStep {
        LookupStep {
            title: title.into(),
            role: role.into(),
            tools: tools.iter().map(|t| (*t).to_owned()).collect(),
            skills: Vec::new(),
            depends_on: Vec::new(),
        }
    }

    fn run(id: &str, outcome: RunOutcome, duration_s: u64, retries: u32) -> PlanRun {
        PlanRun {
            plan_id: id.into(),
            outcome,
            duration_s,
            retries,
        }
    }

    #[test]
    fn fingerprint_ignores_titles_and_tool_order() {
        let a = [step(
            "Find args",
            "research",
            &["search_files", "read_file"],
        )];
        let b = [step(
            "Locate the struct",
            "research",
            &["read_file", "search_files"],
        )];
        assert_eq!(fingerprint(&a), fingerprint(&b));
        let c = [step("Find args", "worker", &["read_file"])];
        assert_ne!(fingerprint(&a), fingerprint(&c));
    }

    #[test]
    fn stats_and_score_follow_the_formula() {
        let now = Utc::now();
        let mut t = LookupTable::default();
        let mut e = LookupEntry::new("Add a CLI flag", "command", "coding/rust", vec![], now);
        e.record(&run("p1", RunOutcome::Success, 100, 0), now);
        e.record(&run("p2", RunOutcome::Partial, 200, 1), now);
        e.record(&run("p2", RunOutcome::Success, 300, 2), now);
        assert_eq!((e.stats.runs, e.stats.success, e.stats.partial), (3, 2, 1));
        assert_eq!(e.stats.median_duration_s, 200);
        assert!((e.stats.avg_retries - 1.0).abs() < 1e-9);
        assert_eq!(e.source_plans, ["p1", "p2"]);

        let mut fast = LookupEntry::new("add a cli flag!", "command", "coding/rust", vec![], now);
        fast.record(&run("p3", RunOutcome::Success, 100, 0), now);
        t.entries = vec![e, fast];
        let w = ScorerWeights::default();
        t.rescore(&w, 7);
        let expected = 0.60 * (2.5 / 3.0) + 0.25 * (100.0 / 200.0) + 0.15 * (1.0 - 1.0 / 7.0);
        assert!(
            (t.entries[0].score - expected).abs() < 1e-9,
            "{}",
            t.entries[0].score
        );
        assert!((t.entries[1].score - 1.0).abs() < 1e-9);

        let cfg = ScorerConfig::default();
        assert!(!t.entries[1].reusable(&cfg), "one run is not enough");
        let best = |t: &LookupTable| {
            t.best_template("cli flag", "command", "coding/rust", &cfg)
                .map(|e| e.id.clone())
        };
        assert_eq!(best(&t), Some(t.entries[0].id.clone()));
        t.entries[1].stats.runs = 2;
        assert_eq!(
            best(&t),
            Some(t.entries[1].id.clone()),
            "highest score wins"
        );
        assert!(
            t.best_template("cli flag", "question", "coding/rust", &cfg)
                .is_none()
        );
    }

    #[test]
    fn store_round_trips_under_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::new(tmp.path());
        let store = LookupStore::new(&dir);
        assert!(store.load().unwrap().entries.is_empty());
        store
            .update(|t| {
                t.entries.push(LookupEntry::new(
                    "sig",
                    "command",
                    "git",
                    vec![],
                    Utc::now(),
                ));
            })
            .unwrap();
        assert_eq!(store.load().unwrap().entries.len(), 1);
    }
}
