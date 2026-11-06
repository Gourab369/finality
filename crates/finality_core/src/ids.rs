use std::fmt;
use std::str::FromStr;

use chrono::{NaiveDateTime, Timelike};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ulid::Ulid;

macro_rules! ulid_id {
    ($($name:ident),* $(,)?) => {$(
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Ulid);

        impl $name {
            pub fn new() -> Self {
                Self(Ulid::new())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromStr for $name {
            type Err = ulid::DecodeError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ulid::from_string(s).map(Self)
            }
        }
    )*};
}

ulid_id!(AgentId, TaskId, PlanId, EventId);

const SESSION_TIME_FORMAT: &str = "%Y%m%d-%H%M%S";

/// Session identifier rendered as `{serial:06}_{YYYYMMDD-HHMMSS}` (local time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId {
    pub serial: u64,
    pub started: NaiveDateTime,
}

impl SessionId {
    pub fn new(serial: u64, started: NaiveDateTime) -> Self {
        // Second precision keeps Display/FromStr lossless.
        let started = started.with_nanosecond(0).unwrap_or(started);
        Self { serial, started }
    }

    pub fn log_file_name(&self) -> String {
        format!("{self}.hlog")
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:06}_{}",
            self.serial,
            self.started.format(SESSION_TIME_FORMAT)
        )
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid session id `{0}`; expected `NNNNNN_YYYYMMDD-HHMMSS`")]
pub struct ParseSessionIdError(String);

impl FromStr for SessionId {
    type Err = ParseSessionIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || ParseSessionIdError(s.to_owned());
        let (serial, time) = s.split_once('_').ok_or_else(err)?;
        if serial.len() < 6 || !serial.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        let serial = serial.parse().map_err(|_| err())?;
        let started =
            NaiveDateTime::parse_from_str(time, SESSION_TIME_FORMAT).map_err(|_| err())?;
        Ok(Self { serial, started })
    }
}

impl Serialize for SessionId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn sample() -> SessionId {
        let t = NaiveDate::from_ymd_opt(2026, 11, 5)
            .unwrap()
            .and_hms_milli_opt(14, 30, 12, 345)
            .unwrap();
        SessionId::new(42, t)
    }

    #[test]
    fn session_id_format() {
        assert_eq!(sample().to_string(), "000042_20261105-143012");
        assert_eq!(sample().log_file_name(), "000042_20261105-143012.hlog");
    }

    #[test]
    fn session_id_round_trip() {
        let id = sample();
        assert_eq!(id.to_string().parse::<SessionId>().unwrap(), id);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"000042_20261105-143012\"");
        assert_eq!(serde_json::from_str::<SessionId>(&json).unwrap(), id);
    }

    #[test]
    fn session_id_large_serial() {
        let id: SessionId = "1234567_20261105-143012".parse().unwrap();
        assert_eq!(id.serial, 1_234_567);
    }

    #[test]
    fn session_id_rejects_garbage() {
        for bad in [
            "",
            "42_20261105-143012",
            "000042-20261105",
            "000042_2026-11-05",
            "abcdef_20261105-143012",
        ] {
            assert!(bad.parse::<SessionId>().is_err(), "{bad}");
        }
    }

    #[test]
    fn ulid_ids_are_unique_and_parse() {
        let a = TaskId::new();
        let b = TaskId::new();
        assert_ne!(a, b);
        assert_eq!(a.to_string().parse::<TaskId>().unwrap(), a);
    }
}
