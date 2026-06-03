// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disaggregation configuration for conditional prefill/decode coordination.
//!
//! When present, signals that this leader participates in the conditional
//! disaggregation topology coordinated by a `kvbm-hub`. The leader registers
//! with the hub under the configured role so that decode instances can locate
//! prefill workers (and vice versa).

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Disaggregation role.
///
/// Identifies whether the leader participates as a prefill producer or a
/// decode consumer in the disagg topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisaggregationRole {
    /// Prefill role — produces KV for decode instances.
    Prefill,
    /// Decode role — consumes prefilled KV from prefill instances.
    Decode,
}

/// Disaggregation configuration.
///
/// The hub URL is **not** here — it comes from
/// [`KvbmConfig::hub`](crate::KvbmConfig::hub). This block only carries the
/// per-instance role (and admission budget); the `disagg` feature
/// is enabled via `leader.hub.features`.
///
/// # JSON example
/// ```json
/// {
///   "role": "prefill",
///   "max_inflight_remote_prefill_tokens": 1048576
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DisaggConfig {
    /// Role this instance plays in the disagg topology.
    pub role: DisaggregationRole,

    /// Maximum number of decode-side remote-prefill tokens accepted but not
    /// yet materialized. Defaults to unlimited to preserve existing behavior
    /// unless operators opt in to admission throttling.
    #[serde(default = "default_max_inflight_remote_prefill_tokens")]
    pub max_inflight_remote_prefill_tokens: usize,

    /// Decode-side conditional-disagg threshold: the minimum number of
    /// *uncached* prefill tokens (`total − num_computed − local connector
    /// match`) at or above which a decode leader disaggregates prefill to a
    /// remote prefill worker. Requests below this prefill locally on the
    /// decode instance. `0` (default) ⇒ AlwaysRemote — every CD-eligible
    /// request disaggregates (subject to the downstream 1-full-block floor),
    /// preserving prior behavior. Only consulted for the decode role.
    #[serde(default)]
    pub min_remote_prefill_tokens: usize,
}

fn default_max_inflight_remote_prefill_tokens() -> usize {
    usize::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_prefill() {
        let json = r#"{"role": "prefill"}"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.role, DisaggregationRole::Prefill);
        assert_eq!(cfg.max_inflight_remote_prefill_tokens, usize::MAX);
    }

    #[test]
    fn test_deserialize_decode() {
        let json = r#"{"role": "decode"}"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.role, DisaggregationRole::Decode);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = DisaggConfig {
            role: DisaggregationRole::Decode,
            max_inflight_remote_prefill_tokens: 4096,
            min_remote_prefill_tokens: 0,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""role":"decode""#));
        assert!(json.contains(r#""max_inflight_remote_prefill_tokens":4096"#));

        let roundtrip: DisaggConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.role, cfg.role);
        assert_eq!(
            roundtrip.max_inflight_remote_prefill_tokens,
            cfg.max_inflight_remote_prefill_tokens
        );
    }

    #[test]
    fn test_validate_ok() {
        let cfg = DisaggConfig {
            role: DisaggregationRole::Prefill,
            max_inflight_remote_prefill_tokens: default_max_inflight_remote_prefill_tokens(),
            min_remote_prefill_tokens: 0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_min_remote_prefill_tokens_defaults_and_parses() {
        // Absent ⇒ defaults to 0 (AlwaysRemote-equivalent).
        let cfg: DisaggConfig = serde_json::from_str(r#"{"role": "decode"}"#).unwrap();
        assert_eq!(cfg.min_remote_prefill_tokens, 0);

        // Present ⇒ parsed (conditional-disagg threshold).
        let cfg: DisaggConfig =
            serde_json::from_str(r#"{"role": "decode", "min_remote_prefill_tokens": 256}"#).unwrap();
        assert_eq!(cfg.min_remote_prefill_tokens, 256);
    }

    #[test]
    fn test_deserialize_inflight_budget() {
        let json = r#"{
            "role": "decode",
            "max_inflight_remote_prefill_tokens": 64
        }"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.max_inflight_remote_prefill_tokens, 64);
    }

    #[test]
    fn test_missing_role_fails() {
        let json = r#"{"max_inflight_remote_prefill_tokens": 64}"#;
        let result: Result<DisaggConfig, _> = serde_json::from_str(json);
        assert!(result.is_err(), "role is required");
    }
}
