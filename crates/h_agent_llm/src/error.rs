#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("cannot reach {url}: {message}")]
    Unreachable { url: String, message: String },
    #[error("request timed out")]
    Timeout,
    #[error("HTTP {status}: {body}")]
    Status {
        status: u16,
        body: String,
        /// The server's `Retry-After`, when it sent one.
        retry_after: Option<std::time::Duration>,
    },
    #[error("invalid response: {0}")]
    Decode(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

impl LlmError {
    /// Worth retrying before any output was streamed.
    pub fn is_transient(&self) -> bool {
        match self {
            LlmError::Unreachable { .. } | LlmError::Timeout => true,
            LlmError::Status { status, .. } => *status == 429 || *status >= 500,
            _ => false,
        }
    }

    /// The wait the server asked for before the next attempt.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            LlmError::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    pub(crate) fn from_reqwest(url: &str, e: reqwest::Error) -> Self {
        if e.is_timeout() {
            LlmError::Timeout
        } else if e.is_connect() || e.is_request() {
            LlmError::Unreachable {
                url: url.to_owned(),
                message: e.to_string(),
            }
        } else if e.is_decode() || e.is_body() {
            LlmError::Decode(e.to_string())
        } else {
            LlmError::Other(e.to_string())
        }
    }
}
