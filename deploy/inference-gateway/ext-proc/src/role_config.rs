// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-role router configuration for the disaggregated standalone EPP.
//!
//! The two roles need genuinely different selection behavior, and every knob
//! that expresses the difference lives on [`KvRouterConfig`], which a
//! `SelectionService` takes once at construction. That is why disaggregated mode
//! builds two services rather than one service with two partitions: a single
//! config cannot describe both roles.
//!
//! The split mirrors what the full Dynamo runtime does for the same two legs:
//!
//! | knob | prefill | decode |
//! |---|---|---|
//! | `use_kv_events` | on | off |
//! | `overlap_score_credit` | base | `0.0` |
//! | `router_track_prefill_tokens` | base | off |
//! | `router_assume_kv_reuse` | base | off |
//!
//! In words: prefill scores on prefix overlap and does not model long-lived
//! block occupancy; decode ignores overlap and models occupancy. A decode worker
//! did not compute the prompt, so charging it prefill work would mis-rank it,
//! and crediting it for a prefix it never cached would be a fiction.

use anyhow::Result;
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};

use crate::epp_standalone_config::{DISAGGREGATED_TOPOLOGY, EppTopologyMode};
use crate::worker_role::WorkerRole;

/// Env var that enables conditional disaggregation, rejected below.
const DYN_ROUTER_CONDITIONAL_DISAGG: &str = "DYN_ROUTER_CONDITIONAL_DISAGG";
/// Env var whose value the decode role must clear; named in the warning.
const DYN_ROUTER_PREDICTED_TTL_SECS: &str = "DYN_ROUTER_PREDICTED_TTL_SECS";
/// Env var the standalone selector cannot honor; warned about, not rejected.
const DYN_ROUTER_TRACK_ACTIVE_BLOCKS: &str = "DYN_ROUTER_TRACK_ACTIVE_BLOCKS";

/// Reject router settings the standalone EPP cannot honor under a role split.
///
/// Separate from [`kv_router_config_for_role`] so it can fail loudly at startup
/// instead of silently producing a config that means something else.
pub(crate) fn reject_unsupported_router_config(
    topology: EppTopologyMode,
    base: &KvRouterConfig,
) -> Result<()> {
    if !topology.is_disaggregated() {
        return Ok(());
    }

    if !base.router_track_active_blocks {
        // Not an error, because the flag is currently a no-op on this path: the
        // selection service computes tracking hashes directly rather than
        // through the `KvRouterConfig` wrapper that honors it. Worth saying out
        // loud, though — an operator who set this expects load accounting to be
        // off, and under a role split active-block occupancy is the decode
        // selector's only remaining load signal.
        tracing::warn!(
            "{DYN_ROUTER_TRACK_ACTIVE_BLOCKS}=false has no effect on the standalone selector; \
             active-block accounting stays on, and it is the decode role's only load signal \
             once overlap scoring and prefill-token tracking are disabled for it"
        );
    }

    if base.conditional_disagg_enabled {
        // Conditional disaggregation inverts the decode role's whole profile: it
        // needs decode-side KV events to decide whether to bypass prefill at
        // all, and the bypass itself is orchestration `SelectionService` does not
        // perform. Honoring the flag would be a lie; ignoring it silently would
        // be worse.
        anyhow::bail!(
            "{DYN_ROUTER_CONDITIONAL_DISAGG} is not supported with \
             DYN_EPP_TOPOLOGY_MODE={DISAGGREGATED_TOPOLOGY}: conditional disaggregation requires \
             decode-side KV events and bypass orchestration the standalone selector does not \
             provide"
        );
    }

    Ok(())
}

/// Derive the config for one role from the process-wide base.
///
/// Prefill *inherits* rather than overrides when the base already disables
/// KV-aware selection: an operator who turned it off meant it, and
/// `should_subscribe_to_kv_events()` is `use_kv_events && overlap_score_credit > 0`,
/// so either switch alone is a complete opt-out. Re-enabling one of them here
/// would resurrect a subscription the operator deliberately removed.
pub(crate) fn kv_router_config_for_role(base: &KvRouterConfig, role: WorkerRole) -> KvRouterConfig {
    let mut cfg = base.clone();

    match role {
        // Aggregated is the untouched base: this is what keeps aggregated mode
        // byte-identical to its previous behavior.
        WorkerRole::Aggregated => {}

        WorkerRole::Prefill => {
            if !kv_aware_selection_enabled(base) {
                tracing::warn!(
                    role = %WorkerRole::Prefill,
                    use_kv_events = base.use_kv_events,
                    overlap_score_credit = base.overlap_score_credit,
                    "KV-aware prefill selection is disabled by the base router config; \
                     the prefill catalog will select on load alone"
                );
            }
        }

        WorkerRole::Decode => {
            cfg.use_kv_events = false;
            cfg.overlap_score_credit = 0.0;
            cfg.router_track_prefill_tokens = false;
            cfg.router_assume_kv_reuse = false;

            // `SelectionServiceBuilder::build` validates the config before it
            // constructs anything, and predicted-TTL pruning requires the event
            // stream this role just turned off. Clearing it is the only reading
            // that keeps both settings meaningful.
            if cfg.router_predicted_ttl_secs.take().is_some() {
                tracing::warn!(
                    role = %WorkerRole::Decode,
                    "Clearing {DYN_ROUTER_PREDICTED_TTL_SECS} for the decode selector: it prunes \
                     a KV-event-fed index, and the decode role consumes no KV events"
                );
            }

            // Two more settings are cross-field validated against knobs this arm
            // just turned off: `router_prefill_load_model` requires
            // `router_track_prefill_tokens`, and `serve_indexer` requires
            // `overlap_score_credit > 0`. Inheriting either would fail the same
            // validation and take the EPP down at startup.
            //
            // Neither is settable from the EPP's only config source today:
            // `kv_router_config_from_lookup` starts from `KvRouterConfig::default()`
            // and applies a fixed list of `DYN_ROUTER_*` overrides that does not
            // include them, so both hold their disabled defaults. Clear them
            // anyway — the day one of those env knobs is added, the decode
            // selector should keep building rather than fail on a setting that
            // only ever described the prefill leg.
            if cfg.router_prefill_load_model.is_enabled() {
                cfg.router_prefill_load_model = RouterPrefillLoadModel::None;
                tracing::warn!(
                    role = %WorkerRole::Decode,
                    "Clearing router_prefill_load_model for the decode selector: it estimates \
                     prefill load from tracked prefill tokens, which this role does not track"
                );
            }
            if cfg.serve_indexer {
                cfg.serve_indexer = false;
                tracing::warn!(
                    role = %WorkerRole::Decode,
                    "Clearing serve_indexer for the decode selector: it would publish an index \
                     that no KV event ever reaches"
                );
            }
        }
    }

    cfg
}

/// Whether the base config leaves prefix-overlap selection switched on at all.
fn kv_aware_selection_enabled(base: &KvRouterConfig) -> bool {
    base.use_kv_events && base.overlap_score_credit > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> KvRouterConfig {
        KvRouterConfig::default()
    }

    #[test]
    fn aggregated_is_the_untouched_base() {
        let base = base();
        let cfg = kv_router_config_for_role(&base, WorkerRole::Aggregated);
        assert_eq!(cfg.use_kv_events, base.use_kv_events);
        assert_eq!(cfg.overlap_score_credit, base.overlap_score_credit);
        assert_eq!(
            cfg.router_track_prefill_tokens,
            base.router_track_prefill_tokens
        );
        assert_eq!(cfg.router_assume_kv_reuse, base.router_assume_kv_reuse);
    }

    #[test]
    fn prefill_keeps_the_base_selection_profile() {
        // Prefill is the role that still scores on prefix overlap, so none of
        // the four knobs may be forced here.
        let base = base();
        let cfg = kv_router_config_for_role(&base, WorkerRole::Prefill);
        assert!(cfg.use_kv_events);
        assert_eq!(cfg.overlap_score_credit, base.overlap_score_credit);
        assert!(cfg.router_track_prefill_tokens);
        assert!(cfg.router_assume_kv_reuse);
    }

    #[test]
    fn decode_drops_overlap_and_prefill_accounting() {
        let cfg = kv_router_config_for_role(&base(), WorkerRole::Decode);
        assert!(!cfg.use_kv_events);
        assert_eq!(cfg.overlap_score_credit, 0.0);
        assert!(!cfg.router_track_prefill_tokens);
        assert!(!cfg.router_assume_kv_reuse);
    }

    #[test]
    fn decode_clears_predicted_ttl_so_the_service_can_build() {
        // Left set, `validate_kv_router_config` rejects the pair and the decode
        // instance never constructs — taking the whole EPP down at startup.
        let mut base = base();
        base.router_predicted_ttl_secs = Some(120.0);

        let cfg = kv_router_config_for_role(&base, WorkerRole::Decode);
        assert!(cfg.router_predicted_ttl_secs.is_none());
        assert!(cfg.validate_config().is_ok(), "decode config must build");

        // Prefill still consumes events, so its TTL is left alone.
        let prefill = kv_router_config_for_role(&base, WorkerRole::Prefill);
        assert_eq!(prefill.router_predicted_ttl_secs, Some(120.0));
    }

    #[test]
    fn decode_clears_the_other_settings_its_own_overrides_would_invalidate() {
        // `router_prefill_load_model` requires `router_track_prefill_tokens` and
        // `serve_indexer` requires `overlap_score_credit > 0` -- both of which
        // the decode arm turns off. A base that sets them is valid on its own,
        // so the failure would land on the decode selector at startup.
        let mut base = base();
        base.router_prefill_load_model = RouterPrefillLoadModel::Aic;
        base.serve_indexer = true;
        assert!(
            base.validate_config().is_ok(),
            "the base itself must be valid"
        );

        let cfg = kv_router_config_for_role(&base, WorkerRole::Decode);
        assert!(!cfg.router_prefill_load_model.is_enabled());
        assert!(!cfg.serve_indexer);
        assert!(cfg.validate_config().is_ok(), "decode config must build");

        // Prefill keeps both: it tracks prefill tokens and scores overlap.
        let prefill = kv_router_config_for_role(&base, WorkerRole::Prefill);
        assert_eq!(
            prefill.router_prefill_load_model,
            RouterPrefillLoadModel::Aic
        );
        assert!(prefill.serve_indexer);
        assert!(prefill.validate_config().is_ok());
    }

    #[test]
    fn every_role_config_passes_the_builders_validation() {
        for role in [
            WorkerRole::Aggregated,
            WorkerRole::Prefill,
            WorkerRole::Decode,
        ] {
            let cfg = kv_router_config_for_role(&base(), role);
            assert!(
                cfg.validate_config().is_ok(),
                "{role} config must satisfy the same validation SelectionServiceBuilder runs"
            );
        }
    }

    #[test]
    fn prefill_inherits_an_explicit_kv_events_opt_out() {
        // Not an override: `should_subscribe_to_kv_events` is
        // `use_kv_events && overlap_score_credit > 0`, so forcing either back on
        // would resurrect a subscription the operator removed.
        let mut base = base();
        base.use_kv_events = false;
        assert!(!kv_router_config_for_role(&base, WorkerRole::Prefill).use_kv_events);

        let mut base = self::base();
        base.overlap_score_credit = 0.0;
        assert_eq!(
            kv_router_config_for_role(&base, WorkerRole::Prefill).overlap_score_credit,
            0.0
        );
    }

    #[test]
    fn conditional_disagg_is_rejected_only_under_disaggregated() {
        let mut base = base();
        base.conditional_disagg_enabled = true;

        let error = reject_unsupported_router_config(EppTopologyMode::Disaggregated, &base)
            .expect_err("conditional disagg is not supported by the standalone selector")
            .to_string();
        assert!(error.contains(DYN_ROUTER_CONDITIONAL_DISAGG), "{error}");
        assert!(error.contains(DISAGGREGATED_TOPOLOGY), "{error}");

        // Aggregated never builds a decode profile, so the flag is not our
        // business there.
        assert!(reject_unsupported_router_config(EppTopologyMode::Aggregated, &base).is_ok());
    }

    #[test]
    fn a_default_router_config_is_accepted() {
        assert!(reject_unsupported_router_config(EppTopologyMode::Disaggregated, &base()).is_ok());
    }

    #[test]
    fn a_disabled_track_active_blocks_warns_but_does_not_reject() {
        // The flag is a no-op on the selection path, so rejecting would block a
        // deployment over a setting that changes nothing; the warning is the
        // whole remedy until the lib/kv-router gap is closed.
        let mut base = base();
        base.router_track_active_blocks = false;
        assert!(reject_unsupported_router_config(EppTopologyMode::Disaggregated, &base).is_ok());
    }
}
