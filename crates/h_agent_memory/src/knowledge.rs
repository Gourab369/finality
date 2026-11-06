//! Knowledgepedia (DESIGN §8.3): `knowledgepedia/<slug>.txt` topic files plus `index.json`.

use std::io;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use h_agent_core::text::{bm25, tokens};
use h_agent_store::{DataDir, LockError, with_lock, write_atomic};
use serde::{Deserialize, Serialize};

use crate::store::MemoryError;
use crate::tree::clip;

const LOCK: &str = "knowledge";
const INDEX: &str = "index.json";
const SUMMARY_CHARS: usize = 120;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    pub topic: String,
    pub file: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub summary: String,
    pub updated: NaiveDate,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeIndex {
    #[serde(default)]
    pub topics: Vec<IndexEntry>,
}

/// A topic file: header fields and facts, newest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Article {
    pub topic: String,
    pub aliases: Vec<String>,
    pub updated: NaiveDate,
    pub sources: Vec<String>,
    pub confidence: String,
    pub facts: Vec<String>,
}

/// Facts to merge into a topic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contribution {
    pub topic: String,
    pub aliases: Vec<String>,
    pub facts: Vec<String>,
    pub sources: Vec<String>,
    pub confidence: String,
}

/// Lowercase topic restricted to `[a-z0-9-]`.
pub fn slug(topic: &str) -> String {
    let raw: String = topic
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    raw.split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn same(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

fn union(into: &mut Vec<String>, more: &[String]) {
    for m in more {
        let m = m.trim();
        if !m.is_empty() && !into.iter().any(|x| same(x, m)) {
            into.push(m.to_owned());
        }
    }
}

impl Article {
    pub fn render(&self) -> String {
        let mut out = format!(
            "Topic: {}\nAliases: {}\nUpdated: {}\nSources: {}\nConfidence: {}\n---\n",
            self.topic,
            self.aliases.join(", "),
            self.updated,
            self.sources.join(" ; "),
            self.confidence
        );
        for f in &self.facts {
            out.push_str(&format!("- {f}\n"));
        }
        out
    }

    pub fn parse(text: &str) -> Option<Self> {
        let (head, body) = text.split_once("\n---\n").unwrap_or((text, ""));
        let field = |name: &str| {
            head.lines()
                .find_map(|l| l.strip_prefix(&format!("{name}:")))
                .map(|v| v.trim().to_owned())
                .unwrap_or_default()
        };
        let list = |v: String, sep: char| -> Vec<String> {
            v.split(sep)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        };
        let topic = field("Topic");
        if topic.is_empty() {
            return None;
        }
        Some(Self {
            topic,
            aliases: list(field("Aliases"), ','),
            updated: field("Updated").parse().unwrap_or_default(),
            sources: list(field("Sources"), ';'),
            confidence: field("Confidence"),
            facts: body
                .lines()
                .map(|l| l.trim().trim_start_matches("- ").trim().to_owned())
                .filter(|l| !l.is_empty())
                .collect(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Knowledgepedia {
    dir: PathBuf,
    locks_dir: PathBuf,
}

impl Knowledgepedia {
    pub fn new(data: &DataDir) -> Self {
        Self {
            dir: data.knowledgepedia_dir(),
            locks_dir: data.locks_dir(),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn io_err(&self, path: &Path, source: io::Error) -> MemoryError {
        MemoryError::Io {
            file: path.display().to_string(),
            source,
        }
    }

    pub fn index(&self) -> Result<KnowledgeIndex, MemoryError> {
        let path = self.dir.join(INDEX);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| MemoryError::Json {
                file: path.display().to_string(),
                source,
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(KnowledgeIndex::default()),
            Err(e) => Err(self.io_err(&path, e)),
        }
    }

    pub fn read(&self, entry: &IndexEntry) -> Option<String> {
        std::fs::read_to_string(self.dir.join(&entry.file)).ok()
    }

    /// Index entries for `query`: exact topic, then alias, then substring matches, then BM25 over topics, aliases and summaries.
    pub fn lookup(&self, query: &str) -> Vec<IndexEntry> {
        let Ok(index) = self.index() else {
            return Vec::new();
        };
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        let tiers: [&dyn Fn(&IndexEntry) -> bool; 3] = [
            &|e| same(&e.topic, &q),
            &|e| e.aliases.iter().any(|a| same(a, &q)),
            &|e| {
                let t = e.topic.to_lowercase();
                t.contains(&q)
                    || q.contains(&t)
                    || e.aliases.iter().any(|a| {
                        let a = a.to_lowercase();
                        a.contains(&q) || q.contains(&a)
                    })
            },
        ];
        for tier in tiers {
            let hits: Vec<IndexEntry> = index.topics.iter().filter(|e| tier(e)).cloned().collect();
            if !hits.is_empty() {
                return hits;
            }
        }
        let docs: Vec<String> = index
            .topics
            .iter()
            .map(|e| format!("{} {} {}", e.topic, e.aliases.join(" "), e.summary))
            .collect();
        bm25(&docs, &q)
            .into_iter()
            .take(3)
            .map(|(i, _)| index.topics[i].clone())
            .collect()
    }

    /// Topics whose name or an alias appears in `text` as whole words, in index order.
    pub fn mentioned_in(&self, text: &str) -> Vec<String> {
        let Ok(index) = self.index() else {
            return Vec::new();
        };
        let words = tokens(text);
        let appears = |phrase: &str| {
            let p = tokens(phrase);
            !p.is_empty() && words.windows(p.len()).any(|w| w == p.as_slice())
        };
        index
            .topics
            .iter()
            .filter(|e| appears(&e.topic) || e.aliases.iter().any(|a| appears(a)))
            .map(|e| e.topic.clone())
            .collect()
    }

    /// Text of the entries matching `query`, at most `max_chars`.
    pub fn lookup_text(&self, query: &str, max_chars: usize) -> Option<String> {
        let mut out = String::new();
        for e in self.lookup(query) {
            let Some(text) = self.read(&e) else { continue };
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&text);
        }
        if out.is_empty() {
            return None;
        }
        if out.chars().count() > max_chars {
            out = out.chars().take(max_chars).collect();
            out.push_str("\n…(truncated)");
        }
        Some(out)
    }

    /// Merges facts into the topic's file (creating it) and updates the index. Returns the file name.
    pub fn merge(&self, c: &Contribution, today: NaiveDate) -> Result<String, MemoryError> {
        with_lock(&self.locks_dir, LOCK, || {
            std::fs::create_dir_all(&self.dir).map_err(|e| self.io_err(&self.dir, e))?;
            let mut index = self.index()?;
            let existing = index.topics.iter().position(|e| {
                same(&e.topic, &c.topic)
                    || e.aliases.iter().any(|a| same(a, &c.topic))
                    || c.aliases.iter().any(|a| same(a, &e.topic))
            });
            let file = existing.map_or_else(
                || format!("{}.txt", slug(&c.topic)),
                |i| index.topics[i].file.clone(),
            );
            let path = self.dir.join(&file);
            let mut article = std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| Article::parse(&t))
                .unwrap_or_else(|| Article {
                    topic: c.topic.trim().to_owned(),
                    aliases: Vec::new(),
                    updated: today,
                    sources: Vec::new(),
                    confidence: c.confidence.clone(),
                    facts: Vec::new(),
                });
            union(&mut article.aliases, &c.aliases);
            union(&mut article.sources, &c.sources);
            article.updated = today;
            if !c.confidence.is_empty() {
                article.confidence.clone_from(&c.confidence);
            }
            let new: Vec<String> = c
                .facts
                .iter()
                .map(|f| f.trim().to_owned())
                .filter(|f| !f.is_empty() && !article.facts.iter().any(|x| same(x, f)))
                .collect();
            article.facts.splice(0..0, new);
            write_atomic(&path, article.render().as_bytes()).map_err(|e| self.io_err(&path, e))?;

            let entry = IndexEntry {
                topic: article.topic.clone(),
                file: file.clone(),
                aliases: article.aliases.clone(),
                summary: clip(
                    article.facts.first().map_or("", String::as_str),
                    SUMMARY_CHARS,
                ),
                updated: today,
            };
            match existing {
                Some(i) => index.topics[i] = entry,
                None => index.topics.push(entry),
            }
            let index_path = self.dir.join(INDEX);
            let bytes = serde_json::to_vec_pretty(&index).map_err(|source| MemoryError::Json {
                file: index_path.display().to_string(),
                source,
            })?;
            write_atomic(&index_path, &bytes).map_err(|e| self.io_err(&index_path, e))?;
            Ok(file)
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

    fn day(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, d).unwrap()
    }

    fn contribution(topic: &str, fact: &str, source: &str) -> Contribution {
        Contribution {
            topic: topic.into(),
            aliases: vec!["build.zig".into()],
            facts: vec![fact.into()],
            sources: vec![source.into()],
            confidence: "medium".into(),
        }
    }

    #[test]
    fn slugs_are_safe_file_names() {
        assert_eq!(slug("Zig build system"), "zig-build-system");
        assert_eq!(slug("  C++ / CMake!! "), "c-cmake");
    }

    #[test]
    fn merge_creates_then_extends_with_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let k = Knowledgepedia::new(&DataDir::new(tmp.path()));
        let f = k
            .merge(
                &contribution(
                    "Zig build system",
                    "zig build runs build.zig",
                    "https://ziglang.org/learn/build-system/",
                ),
                day(4),
            )
            .unwrap();
        assert_eq!(f, "zig-build-system.txt");
        k.merge(
            &contribution(
                "build.zig",
                "Steps are declared with b.step",
                "session 000042",
            ),
            day(5),
        )
        .unwrap();
        let index = k.index().unwrap();
        assert_eq!(
            index.topics.len(),
            1,
            "an alias merges into the existing topic"
        );
        let text = k.read(&index.topics[0]).unwrap();
        let a = Article::parse(&text).unwrap();
        assert_eq!(a.topic, "Zig build system");
        assert_eq!(a.updated, day(5));
        assert_eq!(
            a.sources,
            ["https://ziglang.org/learn/build-system/", "session 000042"]
        );
        assert_eq!(
            a.facts,
            ["Steps are declared with b.step", "zig build runs build.zig"]
        );
        assert!(
            text.starts_with("Topic: Zig build system\nAliases: build.zig\nUpdated: 2026-11-05\n")
        );
        assert_eq!(index.topics[0].summary, "Steps are declared with b.step");
    }

    #[test]
    fn lookup_prefers_exact_then_alias_then_substring() {
        let tmp = tempfile::tempdir().unwrap();
        let k = Knowledgepedia::new(&DataDir::new(tmp.path()));
        k.merge(&contribution("Zig build system", "a", "s"), day(5))
            .unwrap();
        let mut other = contribution("Zig", "b", "s");
        other.aliases = vec!["ziglang".into()];
        k.merge(&other, day(5)).unwrap();
        let topics = |q: &str| k.lookup(q).into_iter().map(|e| e.topic).collect::<Vec<_>>();
        assert_eq!(topics("zig"), ["Zig"]);
        assert_eq!(topics("BUILD.ZIG"), ["Zig build system"]);
        assert_eq!(
            topics("zig build system for beginners"),
            ["Zig build system", "Zig"]
        );
        assert!(topics("python").is_empty());
        assert!(
            k.lookup_text("ziglang", 1000)
                .unwrap()
                .contains("Topic: Zig\n")
        );
        assert_eq!(
            k.mentioned_in("How do I add a step in build.zig?"),
            ["Zig build system", "Zig"]
        );
        assert!(k.mentioned_in("zigzag in python").is_empty());
    }
}
