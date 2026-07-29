// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # WorkerType
//!
//! `WorkerType` is the processing stage a worker handles in a (potentially
//! disaggregated) serving topology. Four canonical values:
//!
//! - `WorkerType::Prefill`
//! - `WorkerType::Decode`
//! - `WorkerType::Encode`
//! - `WorkerType::Aggregated` — handles Prefill+Decode in a single process
//!
//! Each worker has exactly one role; values are not combinable. To express
//! "an encode worker needs Prefill+Decode OR a single Aggregated peer," the
//! `needs` field on `ModelDeploymentCard` is in DNF form
//! (`Vec<Vec<WorkerType>>`): the outer Vec is OR, each inner Vec is an
//! AND-set of required peer worker types. See
//! `docs/proposals/health-disagg-readiness.md`.
//!
//! `WorkerType` is **orthogonal** to [`crate::model_type::ModelType`]:
//! `ModelType` answers "what OpenAI-style endpoints does this model expose"
//! (Chat, Completions, Embedding, …), while `WorkerType` answers "what
//! processing stage does this worker run." A prefill worker and a decode
//! worker serving the same Chat model both advertise `ModelType::Chat`; they
//! differ only in `WorkerType`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Processing stage a single worker handles. See module docs.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    Prefill,
    Decode,
    Encode,
    Aggregated,
}

impl WorkerType {
    /// Canonical lowercase string form. Used in error messages, logs, and
    /// (via the derived serde rename) on the wire.
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkerType::Prefill => "prefill",
            WorkerType::Decode => "decode",
            WorkerType::Encode => "encode",
            WorkerType::Aggregated => "aggregated",
        }
    }
}

impl fmt::Display for WorkerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error from parsing a [`WorkerType`] string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseWorkerTypeError {
    pub token: String,
}

impl fmt::Display for ParseWorkerTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unrecognized worker_type: {:?}", self.token)
    }
}

impl std::error::Error for ParseWorkerTypeError {}

impl FromStr for WorkerType {
    type Err = ParseWorkerTypeError;

    /// Parse a worker type. Accepts the four canonical names
    /// (`"prefill"`, `"decode"`, `"encode"`, `"aggregated"`),
    /// case-insensitive and whitespace-tolerant. Anything else errors.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefill" => Ok(WorkerType::Prefill),
            "decode" => Ok(WorkerType::Decode),
            "encode" => Ok(WorkerType::Encode),
            "aggregated" => Ok(WorkerType::Aggregated),
            _ => Err(ParseWorkerTypeError {
                token: s.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_canonical_lowercase() {
        assert_eq!(WorkerType::Prefill.to_string(), "prefill");
        assert_eq!(WorkerType::Decode.to_string(), "decode");
        assert_eq!(WorkerType::Encode.to_string(), "encode");
        assert_eq!(WorkerType::Aggregated.to_string(), "aggregated");
    }

    #[test]
    fn from_str_accepts_canonical_names() {
        assert_eq!(
            "prefill".parse::<WorkerType>().unwrap(),
            WorkerType::Prefill
        );
        assert_eq!("decode".parse::<WorkerType>().unwrap(), WorkerType::Decode);
        assert_eq!("encode".parse::<WorkerType>().unwrap(), WorkerType::Encode);
        assert_eq!(
            "aggregated".parse::<WorkerType>().unwrap(),
            WorkerType::Aggregated
        );
    }

    #[test]
    fn from_str_case_insensitive_and_whitespace_tolerant() {
        assert_eq!(
            "PREFILL".parse::<WorkerType>().unwrap(),
            WorkerType::Prefill
        );
        assert_eq!(
            "  Decode  ".parse::<WorkerType>().unwrap(),
            WorkerType::Decode
        );
    }

    #[test]
    fn from_str_rejects_unknown_and_empty() {
        assert!("wibble".parse::<WorkerType>().is_err());
        assert!("".parse::<WorkerType>().is_err());
        assert!("prefill|decode".parse::<WorkerType>().is_err());
    }

    #[test]
    fn display_from_str_round_trip() {
        for wt in [
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
        ] {
            assert_eq!(wt.to_string().parse::<WorkerType>().unwrap(), wt);
        }
    }

    #[test]
    fn serde_json_wire_format_is_canonical_lowercase() {
        assert_eq!(
            serde_json::to_string(&WorkerType::Prefill).unwrap(),
            "\"prefill\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerType::Decode).unwrap(),
            "\"decode\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerType::Encode).unwrap(),
            "\"encode\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerType::Aggregated).unwrap(),
            "\"aggregated\""
        );
    }

    #[test]
    fn serde_json_round_trip() {
        for wt in [
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
        ] {
            let j = serde_json::to_string(&wt).unwrap();
            let back: WorkerType = serde_json::from_str(&j).unwrap();
            assert_eq!(back, wt);
        }
    }

    #[test]
    fn serde_json_rejects_unknown_value() {
        assert!(serde_json::from_str::<WorkerType>("\"wibble\"").is_err());
    }
}
