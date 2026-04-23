// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # WorkerType
//!
//! `WorkerType` is a bitflag describing which processing stage a worker
//! handles in a (potentially disaggregated) serving topology:
//!
//! - `WorkerType::Prefill`
//! - `WorkerType::Decode`
//! - `WorkerType::Encode`
//! - `WorkerType::Aggregated` — defined as `Prefill | Decode` (bitflag alias)
//!
//! The `Aggregated` alias is load-bearing: it means an encode worker declaring
//! `needs = Prefill | Decode` is satisfied equally by a P+D pair or by a single
//! Aggregated worker, with plain bitwise AND semantics. See
//! `docs/proposals/health-disagg-readiness.md`.
//!
//! `WorkerType` is **orthogonal** to [`crate::model_type::ModelType`]:
//! `ModelType` answers "what OpenAI-style endpoints does this model expose"
//! (Chat, Completions, Embedding, …), while `WorkerType` answers "what
//! processing stage does this worker run." A prefill worker and a decode
//! worker serving the same Chat model both advertise `ModelType::Chat`; they
//! differ only in `WorkerType`.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

bitflags! {
    /// Processing stage(s) a worker handles. See module docs for the full design.
    ///
    /// Canonical values (and the only ones accepted at registration):
    /// `Prefill`, `Decode`, `Encode`, `Aggregated` (= `Prefill | Decode`).
    /// Other combinations (e.g. `Prefill | Encode`) are not valid worker types.
    #[derive(Copy, Debug, Default, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
    pub struct WorkerType: u8 {
        const Prefill    = 1 << 0;
        const Decode     = 1 << 1;
        const Encode     = 1 << 2;

        /// Aggregated is an alias for `Prefill | Decode`, not a separate bit.
        /// This lets `Encode::needs = Prefill | Decode` be satisfied equally by
        /// a P+D pair or by a single Aggregated worker.
        const Aggregated = Self::Prefill.bits() | Self::Decode.bits();
    }
}

impl WorkerType {
    /// The canonical string form. Used in the WorkerSet key and on the wire.
    ///
    /// Returns the canonical lowercase name for the four valid worker types,
    /// or a `|`-joined decomposition for non-canonical combinations.
    pub fn as_str(&self) -> String {
        // Canonical single-name forms first.
        if *self == WorkerType::Aggregated {
            return "aggregated".to_string();
        }
        if *self == WorkerType::Prefill {
            return "prefill".to_string();
        }
        if *self == WorkerType::Decode {
            return "decode".to_string();
        }
        if *self == WorkerType::Encode {
            return "encode".to_string();
        }
        if self.is_empty() {
            return String::new();
        }
        // Fallback: decompose into single bits joined by '|'.
        self.units()
            .iter()
            .map(|u| u.as_str())
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn contains_prefill(&self) -> bool {
        self.contains(WorkerType::Prefill)
    }
    pub fn contains_decode(&self) -> bool {
        self.contains(WorkerType::Decode)
    }
    pub fn contains_encode(&self) -> bool {
        self.contains(WorkerType::Encode)
    }

    /// True iff this is the `Aggregated` value (i.e. `Prefill | Decode`).
    pub fn is_aggregated(&self) -> bool {
        *self == WorkerType::Aggregated
    }

    /// True iff this is one of the four canonical worker-type values.
    /// Registration should reject any card whose `worker_type` fails this check.
    pub fn is_canonical(&self) -> bool {
        matches!(
            *self,
            WorkerType::Prefill | WorkerType::Decode | WorkerType::Encode | WorkerType::Aggregated
        )
    }

    /// Decompose into single-bit components. `Aggregated` decomposes to
    /// `[Prefill, Decode]` since it is the alias `Prefill | Decode`.
    pub fn units(&self) -> Vec<WorkerType> {
        let mut result = Vec::new();
        if self.contains_prefill() {
            result.push(WorkerType::Prefill);
        }
        if self.contains_decode() {
            result.push(WorkerType::Decode);
        }
        if self.contains_encode() {
            result.push(WorkerType::Encode);
        }
        result
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
        write!(f, "unrecognized worker_type token: {:?}", self.token)
    }
}

impl std::error::Error for ParseWorkerTypeError {}

impl FromStr for WorkerType {
    type Err = ParseWorkerTypeError;

    /// Parse a worker type from its string form. Accepts:
    /// - canonical names: `"prefill"`, `"decode"`, `"encode"`, `"aggregated"`
    /// - `|`-joined decompositions: `"prefill|decode"` (== `"aggregated"`)
    /// - empty string: `WorkerType::empty()`
    ///
    /// Case-insensitive; whitespace around tokens and separators is ignored.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(WorkerType::empty());
        }
        let mut result = WorkerType::empty();
        for raw_token in trimmed.split('|') {
            let token = raw_token.trim().to_ascii_lowercase();
            let bit = match token.as_str() {
                "prefill" => WorkerType::Prefill,
                "decode" => WorkerType::Decode,
                "encode" => WorkerType::Encode,
                "aggregated" => WorkerType::Aggregated,
                _ => return Err(ParseWorkerTypeError { token }),
            };
            result |= bit;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Bitflag semantics --

    #[test]
    fn aggregated_is_prefill_or_decode_alias() {
        // The load-bearing invariant of this whole design.
        assert_eq!(
            WorkerType::Aggregated,
            WorkerType::Prefill | WorkerType::Decode
        );
        assert!(WorkerType::Aggregated.contains_prefill());
        assert!(WorkerType::Aggregated.contains_decode());
        assert!(!WorkerType::Aggregated.contains_encode());
        assert!(WorkerType::Aggregated.is_aggregated());
    }

    #[test]
    fn bitwise_operations() {
        let p = WorkerType::Prefill;
        let d = WorkerType::Decode;
        let e = WorkerType::Encode;

        assert_eq!(p | d, WorkerType::Aggregated);
        assert_eq!((p | d) & p, p);
        assert_eq!((p | d | e) & WorkerType::Aggregated, WorkerType::Aggregated);

        // Encode-needs-Prefill|Decode satisfied by Aggregated alone.
        let required = WorkerType::Prefill | WorkerType::Decode;
        let present = WorkerType::Aggregated;
        assert_eq!(required & present, required);

        // Encode-needs-P|D satisfied by P+D pair.
        let present_pd = WorkerType::Prefill | WorkerType::Decode;
        assert_eq!(required & present_pd, required);

        // Encode-needs-P|D NOT satisfied by encode-alone.
        let present_encode_only = WorkerType::Encode;
        assert_ne!(required & present_encode_only, required);
    }

    // -- Canonical value validation --

    #[test]
    fn is_canonical_matches_only_the_four() {
        assert!(WorkerType::Prefill.is_canonical());
        assert!(WorkerType::Decode.is_canonical());
        assert!(WorkerType::Encode.is_canonical());
        assert!(WorkerType::Aggregated.is_canonical());

        assert!(!WorkerType::empty().is_canonical());
        assert!(!(WorkerType::Prefill | WorkerType::Encode).is_canonical());
        assert!(!WorkerType::all().is_canonical());
    }

    // -- Display / as_str --

    #[test]
    fn display_canonicalizes_aggregated() {
        assert_eq!(WorkerType::Aggregated.to_string(), "aggregated");
        // Even though Aggregated == Prefill | Decode at the bit level, Display
        // prefers the canonical shorthand.
        assert_eq!(
            (WorkerType::Prefill | WorkerType::Decode).to_string(),
            "aggregated"
        );

        assert_eq!(WorkerType::Prefill.to_string(), "prefill");
        assert_eq!(WorkerType::Decode.to_string(), "decode");
        assert_eq!(WorkerType::Encode.to_string(), "encode");
        assert_eq!(WorkerType::empty().to_string(), "");
    }

    #[test]
    fn display_non_canonical_combinations() {
        // Sanity: non-canonical combos still render something sensible.
        assert_eq!(
            (WorkerType::Prefill | WorkerType::Encode).to_string(),
            "prefill|encode"
        );
    }

    // -- FromStr --

    #[test]
    fn from_str_canonical_names() {
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
    fn from_str_decomposed_equals_aggregated() {
        // "prefill|decode" parses to the same value as "aggregated".
        assert_eq!(
            "prefill|decode".parse::<WorkerType>().unwrap(),
            WorkerType::Aggregated
        );
    }

    #[test]
    fn from_str_empty_is_empty() {
        assert_eq!("".parse::<WorkerType>().unwrap(), WorkerType::empty());
        assert_eq!("   ".parse::<WorkerType>().unwrap(), WorkerType::empty());
    }

    #[test]
    fn from_str_case_insensitive_and_whitespace_tolerant() {
        assert_eq!(
            "Prefill".parse::<WorkerType>().unwrap(),
            WorkerType::Prefill
        );
        assert_eq!(
            "  PREFILL | Decode ".parse::<WorkerType>().unwrap(),
            WorkerType::Aggregated
        );
    }

    #[test]
    fn from_str_rejects_unknown_token() {
        let err = "wibble".parse::<WorkerType>().unwrap_err();
        assert_eq!(err.token, "wibble");
    }

    #[test]
    fn display_from_str_round_trip_canonical() {
        for wt in [
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
        ] {
            let s = wt.to_string();
            let parsed: WorkerType = s.parse().unwrap();
            assert_eq!(parsed, wt, "round-trip failed for {s:?}");
        }
    }

    // -- serde round-trip --

    #[test]
    fn serde_json_round_trip() {
        for wt in [
            WorkerType::empty(),
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
            WorkerType::Aggregated,
            WorkerType::Prefill | WorkerType::Encode, // non-canonical combo
        ] {
            let j = serde_json::to_string(&wt).unwrap();
            let back: WorkerType = serde_json::from_str(&j).unwrap();
            assert_eq!(back, wt, "serde round-trip failed for {wt:?} (json={j})");
        }
    }

    // -- units() --

    #[test]
    fn units_decomposes_to_single_bits() {
        assert_eq!(WorkerType::empty().units(), Vec::<WorkerType>::new());
        assert_eq!(WorkerType::Prefill.units(), vec![WorkerType::Prefill]);
        assert_eq!(
            WorkerType::Aggregated.units(),
            vec![WorkerType::Prefill, WorkerType::Decode],
        );
        assert_eq!(
            (WorkerType::Prefill | WorkerType::Encode).units(),
            vec![WorkerType::Prefill, WorkerType::Encode],
        );
    }
}
