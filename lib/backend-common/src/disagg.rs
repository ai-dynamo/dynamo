// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disaggregated-serving mode for unified backends.
//!
//! [`DisaggregationMode`] is metadata carried on [`crate::WorkerConfig`]. The
//! [`crate::Worker`] consumes it for two registration-time decisions
//! (`ModelType::Prefill` vs the parsed `endpoint_types`, and whether the model
//! advertises a local KV indexer). Engines consult it to switch their per-mode
//! protocol divergence (KV-transfer config, bootstrap handshake,
//! `disaggregated_params` codec) inside `generate` and `drain`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{BackendError, DynamoError, ErrorType};

/// Disaggregation role this worker is playing.
///
/// `Aggregated` is the default: the worker handles prefill and decode in the
/// same engine. `Prefill` and `Decode` workers split the two phases across
/// processes / GPUs and exchange KV cache via the engine-specific transport.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DisaggregationMode {
    /// Single worker handles prefill and decode in one engine. Default.
    #[default]
    #[serde(alias = "agg", alias = "aggregated")]
    #[value(name = "agg", alias = "aggregated")]
    Aggregated,
    /// Worker only runs the prefill phase and hands off KV cache to a decode
    /// peer. Registered as `ModelType::Prefill` so the frontend's prefill
    /// router targets it.
    Prefill,
    /// Worker only runs the decode phase, consuming KV cache produced by a
    /// prefill peer. Does not advertise a local KV indexer.
    Decode,
}

impl DisaggregationMode {
    /// CLI-friendly slug. Matches the value `clap` parses from the
    /// `--disaggregation-mode` flag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Aggregated => "agg",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }

    /// `true` when this worker only runs the prefill phase.
    pub fn is_prefill(&self) -> bool {
        matches!(self, Self::Prefill)
    }

    /// `true` when this worker only runs the decode phase.
    pub fn is_decode(&self) -> bool {
        matches!(self, Self::Decode)
    }
}

impl fmt::Display for DisaggregationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DisaggregationMode {
    type Err = DynamoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            // Accept both the canonical CLI slug and the long form so callers
            // that round-trip through serde JSON ("aggregated") and CLI ("agg")
            // both work.
            "agg" | "aggregated" => Ok(Self::Aggregated),
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            other => Err(DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                .message(format!(
                    "unknown disaggregation mode '{other}' (expected one of: agg, prefill, decode)"
                ))
                .build()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_slugs() {
        assert_eq!(
            "agg".parse::<DisaggregationMode>().unwrap(),
            DisaggregationMode::Aggregated
        );
        assert_eq!(
            "prefill".parse::<DisaggregationMode>().unwrap(),
            DisaggregationMode::Prefill
        );
        assert_eq!(
            "decode".parse::<DisaggregationMode>().unwrap(),
            DisaggregationMode::Decode
        );
    }

    #[test]
    fn parses_aggregated_alias_and_is_case_insensitive() {
        assert_eq!(
            "AGGREGATED".parse::<DisaggregationMode>().unwrap(),
            DisaggregationMode::Aggregated
        );
        assert_eq!(
            "  Prefill  ".parse::<DisaggregationMode>().unwrap(),
            DisaggregationMode::Prefill
        );
    }

    #[test]
    fn rejects_unknown() {
        let e = "encode".parse::<DisaggregationMode>().unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn display_round_trips_through_from_str() {
        for mode in [
            DisaggregationMode::Aggregated,
            DisaggregationMode::Prefill,
            DisaggregationMode::Decode,
        ] {
            let printed = mode.to_string();
            assert_eq!(printed.parse::<DisaggregationMode>().unwrap(), mode);
        }
    }

    #[test]
    fn predicates_match_variants() {
        assert!(DisaggregationMode::Prefill.is_prefill());
        assert!(!DisaggregationMode::Prefill.is_decode());
        assert!(DisaggregationMode::Decode.is_decode());
        assert!(!DisaggregationMode::Aggregated.is_prefill());
        assert!(!DisaggregationMode::Aggregated.is_decode());
    }

    #[test]
    fn default_is_aggregated() {
        assert_eq!(
            DisaggregationMode::default(),
            DisaggregationMode::Aggregated
        );
    }

    #[test]
    fn serde_round_trip_uses_lowercase() {
        let json = serde_json::to_string(&DisaggregationMode::Prefill).unwrap();
        assert_eq!(json, "\"prefill\"");
        let back: DisaggregationMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, DisaggregationMode::Prefill);
    }
}
