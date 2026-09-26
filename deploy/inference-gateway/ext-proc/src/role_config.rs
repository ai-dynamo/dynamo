// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A decode worker never computed the prompt, so it is never credited for prefix
//! overlap or charged prefill work; it also has no KV-event source, so its
//! selector is built with events off.

use anyhow::Result;
use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};

use crate::epp::decode_router_config_override;
use crate::epp_standalone_config::{DISAGGREGATED_TOPOLOGY, EppTopologyMode};
use crate::worker_role::WorkerRole;

const DYN_ROUTER_CONDITIONAL_DISAGG: &str = "DYN_ROUTER_CONDITIONAL_DISAGG";
const DYN_ROUTER_PREDICTED_TTL_SECS: &str = "DYN_ROUTER_PREDICTED_TTL_SECS";

/// Reject router settings the standalone EPP cannot honor under a role split, so
/// they fail at startup instead of silently meaning something else.
pub(crate) fn reject_unsupported_router_config(
    topology: EppTopologyMode,
    base: &KvRouterConfig,
) -> Result<()> {
    if topology.is_disaggregated() && base.conditional_disagg_enabled {
        // Needs decode-side KV events and a bypass the selection service does
        // not orchestrate.
        anyhow::bail!(
            "{DYN_ROUTER_CONDITIONAL_DISAGG} is not supported with \
             DYN_EPP_TOPOLOGY_MODE={DISAGGREGATED_TOPOLOGY}: conditional disaggregation requires \
             decode-side KV events and bypass orchestration the standalone selector does not \
             provide"
        );
    }
    Ok(())
}

/// Static `SelectionService` configuration for one role. Only what must be fixed
/// at construction lives here; the decode leg's selection semantics ride on
/// [`router_config_override_for_role`] instead.
pub(crate) fn kv_router_config_for_role(base: &KvRouterConfig, role: WorkerRole) -> KvRouterConfig {
    let mut cfg = base.clone();
    if role == WorkerRole::Decode {
        cfg.use_kv_events = false;
        // Predicted-TTL pruning needs the event stream this role does not have,
        // and the builder rejects the pair.
        if cfg.router_predicted_ttl_secs.take().is_some() {
            tracing::warn!(
                role = %role,
                "Clearing {DYN_ROUTER_PREDICTED_TTL_SECS} for the decode selector: it prunes a \
                 KV-event-fed index, and the decode role consumes no KV events"
            );
        }
    }
    cfg
}

/// Per-request selection semantics for one role's selector, supplied on every
/// selection and booking it performs. Decode carries the same override the
/// Dynamo-runtime EPP applies to its decode leg; prefill and aggregated carry none.
pub(crate) fn router_config_override_for_role(role: WorkerRole) -> Option<RouterConfigOverride> {
    decode_router_config_override(role == WorkerRole::Decode)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::config::RouterPrefillLoadModel;

    use super::*;

    const ROLES: [WorkerRole; 3] = [
        WorkerRole::Aggregated,
        WorkerRole::Prefill,
        WorkerRole::Decode,
    ];

    #[test]
    fn only_decode_changes_the_static_config_and_only_for_kv_events() {
        let base = KvRouterConfig::default();
        for role in ROLES {
            let cfg = kv_router_config_for_role(&base, role);
            assert_eq!(cfg.use_kv_events, role != WorkerRole::Decode, "{role}");
            // Selection semantics are per-request, never static.
            assert_eq!(
                cfg.overlap_score_credit, base.overlap_score_credit,
                "{role}"
            );
            assert_eq!(
                cfg.router_track_prefill_tokens, base.router_track_prefill_tokens,
                "{role}"
            );
            assert_eq!(
                cfg.router_assume_kv_reuse, base.router_assume_kv_reuse,
                "{role}"
            );
            assert!(cfg.validate_config().is_ok(), "{role}");
        }
    }

    #[test]
    fn decode_carries_the_runtime_decode_override_and_the_other_roles_none() {
        let decode = router_config_override_for_role(WorkerRole::Decode).expect("decode override");
        assert_eq!(decode.overlap_score_credit, Some(0.0));
        assert_eq!(decode.assume_kv_reuse, Some(false));
        assert_eq!(decode.track_prefill_tokens, Some(false));
        assert!(decode.prefill_load_scale.is_none());
        assert!(decode.router_temperature.is_none());
        assert!(decode.shared_cache_multiplier.is_none());

        assert!(router_config_override_for_role(WorkerRole::Prefill).is_none());
        assert!(router_config_override_for_role(WorkerRole::Aggregated).is_none());
    }

    #[test]
    fn decode_clears_predicted_ttl_so_the_service_can_build() {
        let base = KvRouterConfig {
            router_predicted_ttl_secs: Some(120.0),
            ..Default::default()
        };

        let decode = kv_router_config_for_role(&base, WorkerRole::Decode);
        assert!(decode.router_predicted_ttl_secs.is_none());
        assert!(decode.validate_config().is_ok(), "decode config must build");
        assert_eq!(
            kv_router_config_for_role(&base, WorkerRole::Prefill).router_predicted_ttl_secs,
            Some(120.0)
        );
    }

    #[test]
    fn decode_inherits_prefill_load_model_and_serve_indexer_and_still_validates() {
        // Both were once cleared to satisfy cross-field rules against static
        // knobs the decode arm no longer touches.
        let base = KvRouterConfig {
            router_prefill_load_model: RouterPrefillLoadModel::Aic,
            serve_indexer: true,
            ..Default::default()
        };
        assert!(
            base.validate_config().is_ok(),
            "the base itself must be valid"
        );

        let decode = kv_router_config_for_role(&base, WorkerRole::Decode);
        assert!(decode.router_prefill_load_model.is_enabled());
        assert!(decode.serve_indexer);
        assert!(decode.validate_config().is_ok(), "decode config must build");
    }

    #[test]
    fn conditional_disagg_is_rejected_only_under_disaggregated() {
        let default = KvRouterConfig::default();
        let enabled = KvRouterConfig {
            conditional_disagg_enabled: true,
            ..Default::default()
        };

        let cases = [
            (EppTopologyMode::Disaggregated, &enabled, false),
            (EppTopologyMode::Aggregated, &enabled, true),
            (EppTopologyMode::Disaggregated, &default, true),
            (EppTopologyMode::Aggregated, &default, true),
        ];
        for (topology, base, ok) in cases {
            let result = reject_unsupported_router_config(topology, base);
            assert_eq!(
                result.is_ok(),
                ok,
                "{topology:?} conditional={}",
                base.conditional_disagg_enabled
            );
            if let Err(error) = result {
                let error = error.to_string();
                assert!(error.contains(DYN_ROUTER_CONDITIONAL_DISAGG), "{error}");
                assert!(error.contains(DISAGGREGATED_TOPOLOGY), "{error}");
            }
        }
    }
}
