//! Client for the optional decision model (DESIGN §10): `POST {base_url}{path}` with
//! `{model, state, questions}`; answers are decoded by `type`.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{LlmError, Usage};

const MAX_ERROR_BODY: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no; answered with a probability.
    Noul { instructions: String },
    /// One of `criteria` (name → description). A `None` description marks the catch-all ("other").
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Option<String>>,
    },
    /// A level of `legend` ("0".."n" → meaning).
    Score {
        instructions: String,
        legend: BTreeMap<String, String>,
    },
}

impl Question {
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Noul {
        p: f64,
    },
    Choice {
        value: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    /// In [0, 1]. For `noul` it is |p − 0.5| × 2.
    pub fn confidence(&self) -> f64 {
        match self {
            Answer::Noul { p } => ((p - 0.5).abs() * 2.0).min(1.0),
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => *confidence,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Decision {
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

fn number(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
    .filter(|n: &f64| n.is_finite())
}

fn probabilities(v: Option<&Value>) -> BTreeMap<String, f64> {
    v.and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, p)| Some((k.clone(), number(Some(p))?.clamp(0.0, 1.0))))
                .collect()
        })
        .unwrap_or_default()
}

fn decode_answer(v: &Value) -> Option<Answer> {
    let probs = || probabilities(v.get("probabilities"));
    let unit = |x: f64| x.clamp(0.0, 1.0);
    match v.get("type")?.as_str()? {
        "noul" => Some(Answer::Noul {
            p: unit(number(v.get("noul"))?),
        }),
        "choice" => {
            let value = match v.get("choice")? {
                Value::String(s) => s.clone(),
                Value::Null => "other".to_owned(),
                _ => return None,
            };
            let probabilities = probs();
            let confidence = number(v.get("confidence"))
                .or_else(|| probabilities.get(&value).copied())
                .map(unit)?;
            Some(Answer::Choice {
                value,
                probabilities,
                confidence,
            })
        }
        "score" => {
            let probabilities = probs();
            let confidence = number(v.get("confidence"))
                .or_else(|| probabilities.values().copied().reduce(f64::max))
                .map(unit)?;
            Some(Answer::Score {
                score: number(v.get("score"))?,
                probabilities,
                confidence,
            })
        }
        _ => None,
    }
}

/// Lenient: unknown types and malformed answers are skipped; numbers may be strings.
pub fn decode(body: &Value) -> Decision {
    let answers = body
        .get("answers")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), decode_answer(v)?)))
                .collect()
        })
        .unwrap_or_default();
    let usage = body.get("usage");
    let tokens = |k: &str| {
        number(usage.and_then(|u| u.get(k)))
            .map_or(0, |n| n.max(0.0).min(f64::from(u32::MAX)) as u32)
    };
    Decision {
        answers,
        usage: Usage {
            input_tokens: tokens("input_tokens"),
            output_tokens: tokens("output_tokens"),
        },
    }
}

pub struct DecisionClient {
    client: reqwest::Client,
    url: String,
    model: String,
    timeout: Duration,
}

impl DecisionClient {
    pub fn new(
        base_url: &str,
        path: &str,
        model: &str,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| LlmError::Other(format!("http client: {e}")))?;
        Ok(Self {
            client,
            url: format!(
                "{}/{}",
                base_url.trim_end_matches('/'),
                path.trim_start_matches('/')
            ),
            model: model.to_owned(),
            timeout,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn ask(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        cancel: &CancellationToken,
    ) -> Result<Decision, LlmError> {
        let body = json!({ "model": self.model, "state": state, "questions": questions });
        let send = async {
            let resp =
                tokio::time::timeout(self.timeout, self.client.post(&self.url).json(&body).send())
                    .await
                    .map_err(|_| LlmError::Timeout)?
                    .map_err(|e| LlmError::from_reqwest(&self.url, e))?;
            let status = resp.status();
            let text = tokio::time::timeout(self.timeout, resp.text())
                .await
                .map_err(|_| LlmError::Timeout)?
                .map_err(|e| LlmError::from_reqwest(&self.url, e))?;
            if !status.is_success() {
                return Err(LlmError::Status {
                    status: status.as_u16(),
                    body: text.chars().take(MAX_ERROR_BODY).collect(),
                    retry_after: None,
                });
            }
            let value: Value =
                serde_json::from_str(&text).map_err(|e| LlmError::Decode(e.to_string()))?;
            Ok(decode(&value))
        };
        tokio::select! {
            r = send => r,
            () = cancel.cancelled() => Err(LlmError::Cancelled),
        }
    }

    /// Checks that the endpoint exists and answers a `noul` question.
    pub async fn probe(&self, cancel: &CancellationToken) -> Result<(), LlmError> {
        let questions = BTreeMap::from([(
            "probe".to_owned(),
            Question::noul("Is this a connectivity check?"),
        )]);
        let decision = self
            .ask(
                &format!("Connectivity check from {}.", finality_core::APP_NAME),
                &questions,
                cancel,
            )
            .await?;
        match decision.answers.get("probe") {
            Some(Answer::Noul { .. }) => Ok(()),
            _ => Err(LlmError::Decode(format!(
                "{} did not answer the probe question with a `noul` answer; is `{}` a decision model?",
                self.url, self.model
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_source_example() {
        let body = json!({
            "model": "d1-omni",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.981},
                "team": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {"billing": 0.012, "technical": 0.973, "sales": 0.015},
                    "confidence": 0.96
                },
                "severity": {
                    "type": "score",
                    "score": 2.582,
                    "legend": {"0": "No impact", "1": "Minor", "2": "Major", "3": "Critical"},
                    "probabilities": {"0": 0.015, "1": 0.012, "2": 0.35, "3": 0.623},
                    "confidence": 0.582
                }
            },
            "usage": {"input_tokens": 175, "output_tokens": 0}
        });
        let d = decode(&body);
        assert_eq!(d.answers["urgent"], Answer::Noul { p: 0.981 });
        assert!((d.answers["urgent"].confidence() - 0.962).abs() < 1e-9);
        let Answer::Choice {
            value, confidence, ..
        } = &d.answers["team"]
        else {
            panic!("not a choice");
        };
        assert_eq!((value.as_str(), *confidence), ("technical", 0.96));
        let Answer::Score {
            score, confidence, ..
        } = d.answers["severity"]
        else {
            panic!("not a score");
        };
        assert_eq!((score, confidence), (2.582, 0.582));
        assert_eq!(
            d.usage,
            Usage {
                input_tokens: 175,
                output_tokens: 0
            }
        );
    }

    #[test]
    fn decoding_is_lenient() {
        let d = decode(&json!({
            "answers": {
                "a": {"type": "noul", "noul": "0.2"},
                "b": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}},
                "c": {"type": "choice", "choice": null, "confidence": 0.9},
                "d": {"type": "vector", "vector": [1, 2]},
                "e": {"type": "noul"},
                "f": {"noul": 0.4},
                "g": {"type": "noul", "noul": 1.7}
            }
        }));
        assert_eq!(d.answers["a"], Answer::Noul { p: 0.2 });
        assert!(matches!(&d.answers["b"], Answer::Choice { confidence, .. } if *confidence == 0.7));
        assert!(matches!(&d.answers["c"], Answer::Choice { value, .. } if value == "other"));
        assert_eq!(d.answers["g"], Answer::Noul { p: 1.0 });
        assert_eq!(
            d.answers.keys().map(String::as_str).collect::<Vec<_>>(),
            ["a", "b", "c", "g"]
        );
        assert_eq!(d.usage, Usage::default());
        assert!(decode(&json!({"error": "x"})).answers.is_empty());
    }

    #[test]
    fn questions_serialize_like_the_source() {
        let q = BTreeMap::from([
            ("urgent".to_owned(), Question::noul("Is this urgent?")),
            (
                "team".to_owned(),
                Question::Choice {
                    instructions: "Which team?".into(),
                    criteria: BTreeMap::from([
                        ("billing".to_owned(), Some("Payments".to_owned())),
                        ("other".to_owned(), None),
                    ]),
                },
            ),
        ]);
        assert_eq!(
            serde_json::to_value(&q).unwrap(),
            json!({
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "Payments", "other": null}}
            })
        );
        assert!((Answer::Noul { p: 0.55 }.confidence() - 0.1).abs() < 1e-9);
    }
}
