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
/// decode consumer in the conditional-disagg topology.
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
/// # JSON example
/// ```json
/// {
///   "hub_url": "http://127.0.0.1:1337",
///   "role": "prefill",
///   "max_inflight_remote_prefill_tokens": 1048576
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DisaggConfig {
    /// Hub control-plane URL (default: `http://127.0.0.1:1337`).
    ///
    /// This is the private discovery port; the hub also exposes a public
    /// port (`8337` by default) that serves the same registry.
    #[serde(default = "default_hub_url")]
    pub hub_url: String,

    /// Role this instance plays in the disagg topology.
    pub role: DisaggregationRole,

    /// Maximum number of decode-side remote-prefill tokens accepted but not
    /// yet materialized. Defaults to unlimited to preserve existing behavior
    /// unless operators opt in to admission throttling.
    #[serde(default = "default_max_inflight_remote_prefill_tokens")]
    pub max_inflight_remote_prefill_tokens: usize,
}

fn default_hub_url() -> String {
    "http://127.0.0.1:1337".to_string()
}

fn default_max_inflight_remote_prefill_tokens() -> usize {
    usize::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_prefill() {
        let json = r#"{"hub_url": "http://127.0.0.1:1337", "role": "prefill"}"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hub_url, "http://127.0.0.1:1337");
        assert_eq!(cfg.role, DisaggregationRole::Prefill);
        assert_eq!(cfg.max_inflight_remote_prefill_tokens, usize::MAX);
    }

    #[test]
    fn test_deserialize_decode() {
        let json = r#"{"hub_url": "http://hub.local:8337", "role": "decode"}"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hub_url, "http://hub.local:8337");
        assert_eq!(cfg.role, DisaggregationRole::Decode);
    }

    #[test]
    fn test_default_hub_url() {
        let json = r#"{"role": "prefill"}"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hub_url, "http://127.0.0.1:1337");
        assert_eq!(cfg.role, DisaggregationRole::Prefill);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = DisaggConfig {
            hub_url: "http://example.com:1337".to_string(),
            role: DisaggregationRole::Decode,
            max_inflight_remote_prefill_tokens: 4096,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""role":"decode""#));
        assert!(json.contains(r#""hub_url":"http://example.com:1337""#));
        assert!(json.contains(r#""max_inflight_remote_prefill_tokens":4096"#));

        let roundtrip: DisaggConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.hub_url, cfg.hub_url);
        assert_eq!(roundtrip.role, cfg.role);
        assert_eq!(
            roundtrip.max_inflight_remote_prefill_tokens,
            cfg.max_inflight_remote_prefill_tokens
        );
    }

    #[test]
    fn test_validate_ok() {
        let cfg = DisaggConfig {
            hub_url: default_hub_url(),
            role: DisaggregationRole::Prefill,
            max_inflight_remote_prefill_tokens: default_max_inflight_remote_prefill_tokens(),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_deserialize_inflight_budget() {
        let json = r#"{
            "hub_url": "http://127.0.0.1:1337",
            "role": "decode",
            "max_inflight_remote_prefill_tokens": 64
        }"#;
        let cfg: DisaggConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.max_inflight_remote_prefill_tokens, 64);
    }

    #[test]
    fn test_missing_role_fails() {
        let json = r#"{"hub_url": "http://127.0.0.1:1337"}"#;
        let result: Result<DisaggConfig, _> = serde_json::from_str(json);
        assert!(result.is_err(), "role is required");
    }
}
