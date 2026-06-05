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

    /// Decode-side prefill-overload local fallback (Approach B-GNMT). When a
    /// Remote (disaggregate) decision cannot reserve from the inflight budget
    /// (`max_inflight_remote_prefill_tokens` exhausted — the decode-side proxy
    /// for hub prefill-router pressure), `true` (default) DOWNGRADES the
    /// request to a local prefill on the decode worker (a no-CD-state
    /// passthrough, behaviorally identical to a policy-`Local` decision)
    /// instead of returning `(None, false)` to vLLM (which DEFERS/spins,
    /// re-queuing the request until budget frees — making the saturated
    /// prefill the TTFT-tail bottleneck). `false` preserves the prior
    /// defer-on-exhaustion behavior. Only consulted for the decode role, and
    /// only reachable when `max_inflight_remote_prefill_tokens` is FINITE (the
    /// default `usize::MAX` short-circuits the reservation, so this is inert
    /// until a budget is set). Narrows disaggregation only — never produces
    /// more Remote.
    #[serde(default = "default_cd_local_fallback_on_overload")]
    pub cd_local_fallback_on_overload: bool,
    // NOTE: the CD prefill-overload circuit breaker is configured ENTIRELY on
    // the hub (the `kvbm_hub --cd-breaker` CLI flags), NOT on the connector's
    // DisaggConfig. The breaker lives in the hub's prefill-router (it senses the
    // router's free-capacity fraction and PUSHES the resulting tier to decodes
    // over velo); the connector only consumes the pushed tier at runtime, never
    // the breaker's configuration. Earlier drafts carried `cd_breaker_*` fields
    // here, but nothing in kvbm-connector ever read them — they have been
    // removed to keep a single source of truth (the hub CLI). See
    // `kvbm_hub::BreakerConfig` and the `--cd-breaker*` flags in the hub binary.
}

fn default_max_inflight_remote_prefill_tokens() -> usize {
    usize::MAX
}

fn default_cd_local_fallback_on_overload() -> bool {
    true
}

impl Default for DisaggConfig {
    /// Every field at its `serde` default. `role` defaults to
    /// [`DisaggregationRole::Decode`] (the threshold / overflow knobs are
    /// decode-only). Lets construction sites use
    /// `DisaggConfig { role: …, ..Default::default() }` so adding a new field
    /// does not churn every literal — it stays inert unless explicitly set.
    fn default() -> Self {
        Self {
            role: DisaggregationRole::Decode,
            max_inflight_remote_prefill_tokens: default_max_inflight_remote_prefill_tokens(),
            min_remote_prefill_tokens: 0,
            cd_local_fallback_on_overload: default_cd_local_fallback_on_overload(),
        }
    }
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

    /// Build a `DisaggConfig` with every non-`role` field at its serde default,
    /// so existing tests need not spell out each field.
    fn disagg_with(role: DisaggregationRole) -> DisaggConfig {
        DisaggConfig {
            role,
            ..Default::default()
        }
    }

    #[test]
    fn test_default_role_is_decode_and_inert() {
        // The Default impl is the inert base used by `..Default::default()`
        // construction sites: decode role, threshold 0, unlimited budget.
        let cfg = DisaggConfig::default();
        assert_eq!(cfg.role, DisaggregationRole::Decode);
        assert_eq!(cfg.min_remote_prefill_tokens, 0);
        assert_eq!(cfg.max_inflight_remote_prefill_tokens, usize::MAX);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = DisaggConfig {
            max_inflight_remote_prefill_tokens: 4096,
            ..disagg_with(DisaggregationRole::Decode)
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
        let cfg = disagg_with(DisaggregationRole::Prefill);
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
    fn test_cd_local_fallback_on_overload_defaults_true_and_parses() {
        // Absent ⇒ defaults to true (downgrade Remote→Local on budget
        // exhaustion rather than defer to vLLM).
        let cfg: DisaggConfig = serde_json::from_str(r#"{"role": "decode"}"#).unwrap();
        assert!(cfg.cd_local_fallback_on_overload);

        // Present ⇒ parsed; `false` restores the prior defer-on-exhaustion
        // behavior.
        let cfg: DisaggConfig = serde_json::from_str(
            r#"{"role": "decode", "cd_local_fallback_on_overload": false}"#,
        )
        .unwrap();
        assert!(!cfg.cd_local_fallback_on_overload);

        // Roundtrips through serialize without dropping the field.
        let cfg = DisaggConfig {
            max_inflight_remote_prefill_tokens: 4096,
            cd_local_fallback_on_overload: false,
            ..disagg_with(DisaggregationRole::Decode)
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""cd_local_fallback_on_overload":false"#));
        let roundtrip: DisaggConfig = serde_json::from_str(&json).unwrap();
        assert!(!roundtrip.cd_local_fallback_on_overload);
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
