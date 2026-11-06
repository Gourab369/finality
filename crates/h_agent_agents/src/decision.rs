//! The optional decision model at the Orchestrator's decision points (DESIGN §10). Every answer
//! passes a confidence gate; below it, or on any error, callers fall back to their LLM.
//! Never used for permission, approval or path-safety decisions.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use h_agent_core::config::DecisionConfig;
use h_agent_llm::{Answer, DecisionClient, LlmError, Question};
use tokio_util::sync::CancellationToken;

use crate::engine::TokenTotals;
use crate::triage::{Complexity, SEED_QUERY_TYPES, Triage};

const PROBE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_STATE_CHARS: usize = 2000;
const MAX_LEARNED_TYPES: usize = 15;
pub const MAX_TOPICS: usize = 20;

pub const COMPLEXITY_QUESTION: &str = "Does this request need multiple steps or sub-tasks?";
pub const NEEDS_TOOLS_QUESTION: &str = "Does answering this need tools: reading or searching files, editing files, running commands, or using the web?";
pub const RETRY_QUESTION: &str = "Given these failures, is another strategy likely to succeed?";
pub const TEMPLATE_QUESTION: &str = "Does this template fit the request?";

fn seed_description(t: &str) -> &'static str {
    match t {
        "command" => "Do something: run, change, fix or build",
        "question" => "Asks for information or an explanation",
        "analysis" => "Examine, compare or review something",
        "creation" => "Write or generate something new",
        "conversation" => "Chat, greetings, thanks or feedback",
        _ => "",
    }
}

fn clip(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        text.to_owned()
    } else {
        chars[chars.len() - max..].iter().collect()
    }
}

/// What the decision model settled for triage; `None` fields go to the LLM.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FastTriage {
    pub complexity: Option<Complexity>,
    pub query_type: Option<String>,
    pub topic: Option<String>,
    pub needs_tools: Option<bool>,
}

impl FastTriage {
    /// A full triage when every field was decided. Hints and knowledge topics stay empty.
    pub fn complete(&self) -> Option<Triage> {
        Some(Triage {
            query_type: self.query_type.clone()?,
            topic: self.topic.clone()?,
            complexity: self.complexity?,
            needs_tools: self.needs_tools?,
            tool_hints: Vec::new(),
            knowledge_topics: Vec::new(),
        })
    }

    /// Confident decisions override the LLM's.
    pub fn apply(&self, t: &mut Triage) {
        if let Some(c) = self.complexity {
            t.complexity = c;
        }
        if let Some(q) = &self.query_type {
            t.query_type.clone_from(q);
        }
        if let Some(p) = &self.topic {
            t.topic.clone_from(p);
        }
        if let Some(n) = self.needs_tools {
            t.needs_tools = n;
        }
    }

    pub fn undecided(&self) -> Vec<&'static str> {
        [
            ("complexity", self.complexity.is_none()),
            ("query_type", self.query_type.is_none()),
            ("topic", self.topic.is_none()),
            ("needs_tools", self.needs_tools.is_none()),
        ]
        .into_iter()
        .filter_map(|(k, missing)| missing.then_some(k))
        .collect()
    }
}

pub struct Decider {
    client: DecisionClient,
    min_confidence: f64,
    enabled: AtomicBool,
    totals: Arc<TokenTotals>,
}

impl Decider {
    pub fn new(
        cfg: &DecisionConfig,
        base_url: &str,
        totals: Arc<TokenTotals>,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            client: DecisionClient::new(
                base_url,
                &cfg.path,
                &cfg.model,
                Duration::from_secs(cfg.timeout_secs.max(1)),
            )?,
            min_confidence: cfg.min_confidence,
            enabled: AtomicBool::new(false),
            totals,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    pub fn model(&self) -> &str {
        self.client.model()
    }

    /// Probes the endpoint; a failed probe leaves the decision model off.
    pub async fn enable(&self) -> Result<(), String> {
        let cancel = CancellationToken::new();
        let result = match tokio::time::timeout(PROBE_TIMEOUT, self.client.probe(&cancel)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(format!(
                "decision model `{}` at {} is not usable: {e}",
                self.client.model(),
                self.client.url()
            )),
            Err(_) => Err(format!(
                "decision model `{}` at {} did not answer within {} s",
                self.client.model(),
                self.client.url(),
                PROBE_TIMEOUT.as_secs()
            )),
        };
        self.enabled.store(result.is_ok(), Ordering::SeqCst);
        result
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// `None` when disabled or the request failed (the caller falls back to its LLM).
    async fn ask(
        &self,
        point: &str,
        state: &str,
        questions: BTreeMap<String, Question>,
        cancel: &CancellationToken,
    ) -> Option<BTreeMap<String, Answer>> {
        if !self.is_enabled() {
            return None;
        }
        match self
            .client
            .ask(&clip(state, MAX_STATE_CHARS), &questions, cancel)
            .await
        {
            Ok(d) => {
                self.totals.add(d.usage.input_tokens, d.usage.output_tokens);
                Some(d.answers)
            }
            Err(LlmError::Cancelled) => None,
            Err(e) => {
                tracing::warn!(point, error = %e, "decision model failed; the LLM decides");
                None
            }
        }
    }

    /// The answer if it passes the confidence gate; logs the fallback otherwise.
    fn gate<'a>(&self, answers: &'a BTreeMap<String, Answer>, key: &str) -> Option<&'a Answer> {
        match answers.get(key) {
            Some(a) if a.confidence() >= self.min_confidence => Some(a),
            Some(a) => {
                tracing::info!(
                    point = key,
                    confidence = a.confidence(),
                    min = self.min_confidence,
                    "decision model unsure; the LLM decides"
                );
                None
            }
            None => {
                tracing::info!(
                    point = key,
                    "decision model gave no answer; the LLM decides"
                );
                None
            }
        }
    }

    fn yes(&self, answers: &BTreeMap<String, Answer>, key: &str) -> Option<bool> {
        match self.gate(answers, key)? {
            Answer::Noul { p } => Some(*p >= 0.5),
            _ => None,
        }
    }

    fn pick(&self, answers: &BTreeMap<String, Answer>, key: &str) -> Option<String> {
        match self.gate(answers, key)? {
            Answer::Choice { value, .. } if value != "other" => Some(value.clone()),
            _ => {
                tracing::info!(point = key, "decision model chose `other`; the LLM decides");
                None
            }
        }
    }

    /// Triage complexity, query type, topic and needs-tools in one request.
    /// `types` are learned query types; `topics` the most used topics (most used first).
    pub async fn triage(
        &self,
        prompt: &str,
        recent: &str,
        types: &[String],
        topics: &[String],
        cancel: &CancellationToken,
    ) -> Option<FastTriage> {
        let mut criteria: BTreeMap<String, Option<String>> = SEED_QUERY_TYPES
            .iter()
            .map(|t| ((*t).to_owned(), Some(seed_description(t).to_owned())))
            .collect();
        for t in types.iter().take(MAX_LEARNED_TYPES) {
            criteria.entry(t.clone()).or_insert_with(|| Some(t.clone()));
        }
        criteria.insert("other".into(), None);
        let mut questions = BTreeMap::from([
            ("complexity".to_owned(), Question::noul(COMPLEXITY_QUESTION)),
            (
                "needs_tools".to_owned(),
                Question::noul(NEEDS_TOOLS_QUESTION),
            ),
            (
                "query_type".to_owned(),
                Question::Choice {
                    instructions: "What kind of request is this?".into(),
                    criteria,
                },
            ),
        ]);
        if !topics.is_empty() {
            let mut criteria: BTreeMap<String, Option<String>> = topics
                .iter()
                .take(MAX_TOPICS)
                .map(|t| (t.clone(), Some(t.clone())))
                .collect();
            criteria.insert("other".into(), None);
            questions.insert(
                "topic".into(),
                Question::Choice {
                    instructions: "Which subject is this request about?".into(),
                    criteria,
                },
            );
        }
        let mut state = String::new();
        if !recent.is_empty() {
            state.push_str(&format!("Recent conversation:\n{}\n\n", clip(recent, 800)));
        }
        state.push_str(&format!("Latest request:\n{prompt}"));
        let answers = self.ask("triage", &state, questions, cancel).await?;
        Some(FastTriage {
            complexity: self.yes(&answers, "complexity").map(|multi| {
                if multi {
                    Complexity::MultiStep
                } else {
                    Complexity::Simple
                }
            }),
            query_type: self.pick(&answers, "query_type"),
            topic: self.pick(&answers, "topic"),
            needs_tools: self.yes(&answers, "needs_tools"),
        })
    }

    /// Retry vs give up after a failed attempt. `None` leaves it to the Orchestrator's LLM.
    pub async fn retry_worthwhile(&self, task: &str, attempts: &str) -> Option<bool> {
        let state = format!("Task:\n{task}\n\nAttempts so far:\n{attempts}");
        let questions = BTreeMap::from([("retry".to_owned(), Question::noul(RETRY_QUESTION))]);
        let answers = self
            .ask("retry", &state, questions, &CancellationToken::new())
            .await?;
        self.yes(&answers, "retry")
    }

    /// Whether a lookup-table template fits the request. `None` leaves it to the LLM.
    pub async fn template_fits(
        &self,
        request: &str,
        template: &str,
        cancel: &CancellationToken,
    ) -> Option<bool> {
        let state = format!("Request:\n{request}\n\nTemplate:\n{template}");
        let questions =
            BTreeMap::from([("template".to_owned(), Question::noul(TEMPLATE_QUESTION))]);
        let answers = self.ask("template", &state, questions, cancel).await?;
        self.yes(&answers, "template")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_triage_completes_or_overrides() {
        let partial = FastTriage {
            complexity: Some(Complexity::MultiStep),
            query_type: None,
            topic: Some("coding/rust".into()),
            needs_tools: Some(true),
        };
        assert!(partial.complete().is_none());
        assert_eq!(partial.undecided(), ["query_type"]);
        let mut llm = Triage::fallback();
        partial.apply(&mut llm);
        assert_eq!(llm.complexity, Complexity::MultiStep);
        assert_eq!(
            llm.query_type, "question",
            "undecided fields keep the LLM's value"
        );
        assert_eq!(llm.topic, "coding/rust");

        let full = FastTriage {
            query_type: Some("command".into()),
            ..partial
        };
        let t = full.complete().unwrap();
        assert_eq!((t.query_type.as_str(), t.needs_tools), ("command", true));
        assert!(t.tool_hints.is_empty() && t.knowledge_topics.is_empty());
    }

    #[test]
    fn clip_keeps_the_tail() {
        assert_eq!(clip("abcdef", 3), "def");
        assert_eq!(clip("ab", 3), "ab");
    }
}
