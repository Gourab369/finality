#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::time::Duration;

use h_agent_llm::{Answer, DecisionClient, LlmError, Question};
use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn client(server: &MockServer, timeout: Duration) -> DecisionClient {
    DecisionClient::new(
        &format!("{}/", server.uri()),
        "/v1/systemone",
        "d1-omni",
        timeout,
    )
    .unwrap()
}

#[tokio::test]
async fn sends_typed_questions_and_decodes_answers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_json(json!({
            "model": "d1-omni",
            "state": "Our app crashes on launch for every user.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this request?",
                    "criteria": {"billing": "Payments", "technical": "Outages", "other": null}
                },
                "severity": {"type": "score", "instructions": "How severe?", "legend": {"0": "None", "1": "Major"}}
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "d1-omni",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.998},
                "team": {"type": "choice", "choice": "technical",
                         "probabilities": {"billing": 0.005, "technical": 0.995}, "confidence": 0.99},
                "severity": {"type": "score", "score": 0.9, "confidence": 0.8}
            },
            "usage": {"input_tokens": 175, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let questions = BTreeMap::from([
        (
            "urgent".to_owned(),
            Question::noul("Is this support request urgent?"),
        ),
        (
            "team".to_owned(),
            Question::Choice {
                instructions: "Which team should handle this request?".into(),
                criteria: BTreeMap::from([
                    ("billing".to_owned(), Some("Payments".to_owned())),
                    ("technical".to_owned(), Some("Outages".to_owned())),
                    ("other".to_owned(), None),
                ]),
            },
        ),
        (
            "severity".to_owned(),
            Question::Score {
                instructions: "How severe?".into(),
                legend: BTreeMap::from([
                    ("0".to_owned(), "None".to_owned()),
                    ("1".to_owned(), "Major".to_owned()),
                ]),
            },
        ),
    ]);
    let d = client(&server, Duration::from_secs(5))
        .ask(
            "Our app crashes on launch for every user.",
            &questions,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(d.answers["urgent"], Answer::Noul { p: 0.998 });
    assert!(
        matches!(&d.answers["team"], Answer::Choice { value, confidence, .. } if value == "technical" && *confidence == 0.99)
    );
    assert!(matches!(d.answers["severity"], Answer::Score { score, .. } if score == 0.9));
    assert_eq!(d.usage.input_tokens, 175);
}

#[tokio::test]
async fn probe_reports_missing_endpoints_and_wrong_models() {
    let server = MockServer::start().await;
    let c = client(&server, Duration::from_secs(5));
    let cancel = CancellationToken::new();
    match c.probe(&cancel).await.unwrap_err() {
        LlmError::Status { status: 404, .. } => {}
        other => panic!("unexpected {other:?}"),
    }

    Mock::given(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "answers": {} })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let err = c.probe(&cancel).await.unwrap_err().to_string();
    assert!(err.contains("did not answer the probe question"), "{err}");

    Mock::given(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": { "probe": { "type": "noul", "noul": 0.99 } }
        })))
        .mount(&server)
        .await;
    c.probe(&cancel).await.unwrap();
}

#[tokio::test]
async fn slow_answers_time_out_and_cancel() {
    let server = MockServer::start().await;
    Mock::given(path("/v1/systemone"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "answers": {} }))
                .set_delay(Duration::from_secs(5)),
        )
        .mount(&server)
        .await;
    let q = BTreeMap::from([("x".to_owned(), Question::noul("?"))]);
    let err = client(&server, Duration::from_millis(200))
        .ask("s", &q, &CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::Timeout), "{err:?}");

    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = client(&server, Duration::from_secs(10))
        .ask("s", &q, &cancel)
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::Cancelled), "{err:?}");
}
