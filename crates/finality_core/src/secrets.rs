//! Detection and redaction of credentials in text that is about to be stored or logged
//! (DESIGN §8.2, §12.6).

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// Replacement for redacted secrets.
pub const REDACTED: &str = "[REDACTED]";

static WORDS: LazyLock<Option<Regex>> = LazyLock::new(|| Regex::new(r"[A-Za-z0-9_+/-]{32,}").ok());

static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"tvly-[A-Za-z0-9_-]{16,}",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        r"xox[abprs]-[A-Za-z0-9-]{10,}",
        r"AIza[0-9A-Za-z_-]{35}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}",
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|token)\b\s*(?:is|[:=])\s*\S{6,}",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

/// A run of 32+ token characters mixing upper case, lower case and digits looks like a generated secret.
fn high_entropy(word: &str) -> bool {
    word.len() >= 32
        && word.chars().any(|c| c.is_ascii_uppercase())
        && word.chars().any(|c| c.is_ascii_lowercase())
        && word.chars().any(|c| c.is_ascii_digit())
}

/// True when `text` contains something that looks like a key, token, password or private key.
pub fn contains_secret(text: &str) -> bool {
    PATTERNS.iter().any(|p| p.is_match(text))
        || text
            .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+' | '/')))
            .any(high_entropy)
}

/// `text` with anything [`contains_secret`] would flag replaced by [`REDACTED`].
pub fn redact(text: &str) -> Cow<'_, str> {
    if !contains_secret(text) {
        return Cow::Borrowed(text);
    }
    let mut out = text.to_owned();
    for p in PATTERNS.iter() {
        if let Cow::Owned(s) = p.replace_all(&out, REDACTED) {
            out = s;
        }
    }
    if let Some(words) = WORDS.as_ref()
        && let Cow::Owned(s) = words.replace_all(&out, |c: &regex::Captures<'_>| {
            if high_entropy(&c[0]) {
                REDACTED.to_owned()
            } else {
                c[0].to_owned()
            }
        })
    {
        out = s;
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_common_credentials() {
        for s in [
            "my key is sk-abcdefghijklmnopqrstuvwx1234",
            "token ghp_0123456789abcdefghijABCDEFGHIJ0123",
            "AKIAIOSFODNN7EXAMPLE in the env",
            "-----BEGIN RSA PRIVATE KEY-----",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
            "password: hunter22",
            "the api_key=0123456789",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36P",
            "value Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0LmNb",
        ] {
            assert!(contains_secret(s), "{s}");
        }
    }

    #[test]
    fn leaves_ordinary_text_alone() {
        for s in [
            "Prefers concise answers in British English",
            "Uses PowerShell 5.1 on Windows",
            "Works on the finality project in C:\\source_code",
            "commit 3f2a9c1d4e5b6a7c8d9e0f1a2b3c4d5e6f7a8b9c",
            "the token budget is 200000 tokens",
        ] {
            assert!(!contains_secret(s), "{s}");
        }
    }

    #[test]
    fn redacts_secrets_and_keeps_the_rest() {
        let r = redact("calling api with key sk-abcdefghijklmnopqrstuvwx1234 for user 42");
        assert_eq!(r, "calling api with key [REDACTED] for user 42");
        let r = redact("header Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 sent");
        assert!(!r.contains("abcdefghij") && r.ends_with("sent"), "{r}");
        let r = redact("blob Zx9Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk0LmNb end");
        assert_eq!(r, "blob [REDACTED] end");
        assert!(matches!(redact("plain log line"), Cow::Borrowed(_)));
    }
}
