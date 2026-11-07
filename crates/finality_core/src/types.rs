use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
#[error("invalid {kind} `{value}`; expected one of: {expected}")]
pub struct ParseEnumError {
    kind: &'static str,
    value: String,
    expected: &'static str,
}

/// Defines a fieldless enum whose serde, `Display` and `FromStr` forms share the same names.
macro_rules! string_enum {
    ($(#[$meta:meta])* $name:ident, $kind:literal { $($variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(#[serde(rename = $text)] $variant),+
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = ParseEnumError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s.trim().to_ascii_lowercase().as_str() {
                    $($text => Ok($name::$variant),)+
                    _ => Err(ParseEnumError {
                        kind: $kind,
                        value: s.to_owned(),
                        expected: concat!($($text, " "),+),
                    }),
                }
            }
        }
    };
}

string_enum!(
    /// Permission policy; see DESIGN §12.1.
    Mode, "mode" {
        Plan => "plan",
        Cowork => "cowork",
        AutoEdit => "auto-edit",
        Auto => "auto",
    }
);

string_enum!(ThinkLevel, "thinking level" {
    Off => "off",
    Low => "low",
    Medium => "medium",
    High => "high",
});

string_enum!(ProviderKind, "provider" {
    Ollama => "ollama",
    OpenAi => "openai",
    Anthropic => "anthropic",
    LlamaCpp => "llamacpp",
});

string_enum!(Role, "role" {
    Orchestrator => "orchestrator",
    Worker => "worker",
    Research => "research",
    Memory => "memory",
    Scorer => "scorer",
    Learner => "learner",
});

string_enum!(RiskClass, "risk class" {
    Read => "read",
    Write => "write",
    Exec => "exec",
    Network => "network",
    Browser => "browser",
    External => "external",
    Internal => "internal",
});

// Variant order is part of the session log format (postcard encodes the index); append only.
string_enum!(TaskStatus, "task status" {
    Pending => "pending",
    Queued => "queued",
    Running => "running",
    AwaitingInput => "awaiting-input",
    Success => "success",
    Partial => "partial",
    Failed => "failed",
    Cancelled => "cancelled",
    Blocked => "blocked",
});

string_enum!(ApprovalDecision, "approval decision" {
    AutoAllowed => "auto-allowed",
    AllowedOnce => "allowed-once",
    AllowedSession => "allowed-session",
    Denied => "denied",
    PolicyDenied => "policy-denied",
});

#[allow(clippy::derivable_impls)]
impl Default for Mode {
    fn default() -> Self {
        Mode::Cowork
    }
}

impl Mode {
    /// The modes users choose from; `auto-edit` was merged into `cowork` in v0.3.
    pub const CHOICES: &'static [Mode] = &[Mode::Plan, Mode::Cowork, Mode::Auto];

    /// `auto-edit` behaves as `cowork`. The variant stays so session logs written before v0.3 still decode.
    pub fn canonical(self) -> Self {
        if self == Mode::AutoEdit {
            Mode::Cowork
        } else {
            self
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ThinkLevel {
    fn default() -> Self {
        ThinkLevel::Medium
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ProviderKind {
    fn default() -> Self {
        ProviderKind::Ollama
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_round_trips_through_str_and_serde() {
        for &m in Mode::ALL {
            assert_eq!(m.to_string().parse::<Mode>().unwrap(), m);
            let json = serde_json::to_string(&m).unwrap();
            assert_eq!(json, format!("\"{m}\""));
        }
        assert_eq!("AUTO-EDIT".parse::<Mode>().unwrap(), Mode::AutoEdit);
    }

    #[test]
    fn parse_error_lists_choices() {
        let err = "fast".parse::<ThinkLevel>().unwrap_err().to_string();
        assert!(err.contains("off low medium high"), "{err}");
    }
}
