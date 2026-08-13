// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical worker roles used by routing hosts and worker-selection policies.
//!
//! Each worker has exactly one role. `Aggregated` handles prefill and decode in one process;
//! `Prefill`, `Decode`, and `Encode` identify one stage in a disaggregated topology. Roles are
//! orthogonal to the model's public API surface.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Processing stage a single worker handles.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    Prefill,
    Decode,
    Encode,
    Aggregated,
}

impl WorkerType {
    /// Canonical lowercase string form used for wire values, logs, and metric labels.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::Aggregated => "aggregated",
        }
    }
}

impl fmt::Display for WorkerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "encode" => Ok(Self::Encode),
            "aggregated" => Ok(Self::Aggregated),
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
    fn parses_canonical_names_case_insensitively() {
        assert_eq!("prefill".parse(), Ok(WorkerType::Prefill));
        assert_eq!("Decode".parse(), Ok(WorkerType::Decode));
        assert_eq!(" encode ".parse(), Ok(WorkerType::Encode));
        assert_eq!("aggregated".parse(), Ok(WorkerType::Aggregated));
    }

    #[test]
    fn rejects_unknown_and_empty_names() {
        assert!("wibble".parse::<WorkerType>().is_err());
        assert!("".parse::<WorkerType>().is_err());
        assert!("prefill|decode".parse::<WorkerType>().is_err());
    }

    #[test]
    fn display_and_parse_round_trip() {
        for worker_type in [
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
        ] {
            assert_eq!(worker_type.to_string().parse(), Ok(worker_type));
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
    fn serde_json_round_trip_and_unknown_rejection() {
        for worker_type in [
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
        ] {
            let encoded = serde_json::to_string(&worker_type).unwrap();
            assert_eq!(
                serde_json::from_str::<WorkerType>(&encoded).unwrap(),
                worker_type
            );
        }
        assert!(serde_json::from_str::<WorkerType>("\"wibble\"").is_err());
    }
}
