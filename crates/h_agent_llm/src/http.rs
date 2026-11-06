//! HTTP helpers shared by the providers.

use std::time::Duration;

use futures::StreamExt;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::LlmError;

pub(crate) const MAX_ERROR_BODY: usize = 2000;
/// Non-streaming responses (model lists, model info) larger than this are refused.
const MAX_JSON_BODY: usize = 32 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 64 * 1024;
const BODY_TIMEOUT: Duration = Duration::from_secs(30);

/// Reads at most `max` bytes of the body within `timeout`; anything beyond `max` is dropped.
pub(crate) async fn read_body(
    resp: reqwest::Response,
    max: usize,
    timeout: Duration,
) -> Result<Vec<u8>, LlmError> {
    let read = async {
        let mut out = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::Decode(e.to_string()))?;
            let room = max.saturating_sub(out.len());
            out.extend_from_slice(&chunk[..chunk.len().min(room)]);
            if chunk.len() > room {
                break;
            }
        }
        Ok::<_, LlmError>(out)
    };
    tokio::time::timeout(timeout, read)
        .await
        .map_err(|_| LlmError::Timeout)?
}

/// Decodes a bounded JSON body.
pub(crate) async fn read_json<T: DeserializeOwned>(
    resp: reqwest::Response,
    timeout: Duration,
) -> Result<T, LlmError> {
    let bytes = read_body(resp, MAX_JSON_BODY + 1, timeout).await?;
    if bytes.len() > MAX_JSON_BODY {
        return Err(LlmError::Decode("response body too large".into()));
    }
    serde_json::from_slice(&bytes).map_err(|e| LlmError::Decode(e.to_string()))
}

/// `0.2f32` as `0.2`, not `0.20000000298023224`.
pub(crate) fn f32_json(v: f32) -> Value {
    v.to_string()
        .parse::<f64>()
        .map_or(Value::Null, Value::from)
}

/// The provider's error text: `error` as a string, `error.message`, or `message`; else the raw body.
pub(crate) fn error_message(text: &str) -> String {
    let parsed = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| match v.get("error") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(e) => e.get("message").and_then(Value::as_str).map(str::to_owned),
            None => v.get("message").and_then(Value::as_str).map(str::to_owned),
        });
    parsed
        .unwrap_or_else(|| text.trim().to_owned())
        .chars()
        .take(MAX_ERROR_BODY)
        .collect()
}

pub(crate) async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response, LlmError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after);
    let text = read_body(resp, MAX_ERROR_BYTES, BODY_TIMEOUT)
        .await
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    Err(LlmError::Status {
        status: status.as_u16(),
        body: error_message(&text),
        retry_after,
    })
}

/// `Retry-After` as seconds or an HTTP date.
fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let at = httpdate::parse_http_date(value).ok()?;
    at.duration_since(std::time::SystemTime::now()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_accepts_seconds_and_dates() {
        assert_eq!(parse_retry_after("7"), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after("soon"), None);
        let future =
            httpdate::fmt_http_date(std::time::SystemTime::now() + Duration::from_secs(90));
        let d = parse_retry_after(&future).unwrap_or_default();
        assert!(
            d > Duration::from_secs(80) && d <= Duration::from_secs(90),
            "{d:?}"
        );
        assert_eq!(
            parse_retry_after("Thu, 01 Jan 1970 00:00:00 GMT"),
            None,
            "dates in the past mean no wait"
        );
    }

    #[test]
    fn extracts_provider_error_messages() {
        assert_eq!(
            error_message(r#"{"error":"model not found"}"#),
            "model not found"
        );
        assert_eq!(
            error_message(
                r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#
            ),
            "Incorrect API key"
        );
        assert_eq!(
            error_message(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
            ),
            "Overloaded"
        );
        assert_eq!(error_message(" bad gateway \n"), "bad gateway");
    }
}
