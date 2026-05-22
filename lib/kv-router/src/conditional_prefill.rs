// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-prefill bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait is pure decision logic over
//! a struct of summary signals (`ConditionalPrefillDecisionInput`) — no async,
//! no runtime, no operator dependencies — which lets both the live
//! `PrefillRouter` operator (`lib/llm/src/kv_router/prefill_router/`) and
//! offline replay (`lib/mocker/src/replay/offline/disagg.rs`) call into the
//! same code.
//!
//! One policy ships today:
//! - `TokenCapConditionalPrefillPolicy`: bypass when `net_new_tokens <= cap`.

use crate::config::{
    ConditionalPrefillPolicyKind, DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS, KvRouterConfig,
};

/// Inputs passed to a `ConditionalPrefillPolicy` when deciding whether to
/// bypass remote prefill.
///
/// Field naming convention:
/// - `decode_chosen_*` — value for the cache-hot decode worker (the AGG target,
///   picked by the agg-equation peek on the decode pool).
/// - `prefill_chosen_*` — value for the cost-equation-chosen prefill worker
///   (the DISAGG prefill target).
/// - `decode_min_*` / `decode_pool_min_*` — value for the load-min decode
///   worker picked by the post-handoff load-only peek (the DISAGG decode
///   target after prefill completes).
///
/// `Option<...>` fields are populated only when the relevant peek runs; the
/// probe gates the extra peeks behind `ConditionalPrefillPolicy::needs_cost_terms`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalPrefillDecisionInput {
    /// Total prompt token count. User-facing — TokenCap policy reads this
    /// directly without conversion.
    pub prompt_tokens: usize,

    /// KV cache block size, in tokens.
    pub block_size: usize,

    // === Decode side — cache-hot worker (AGG target) ===
    /// Device-prefix overlap on the cache-hot decode worker, in blocks.
    /// Used by TokenCap and JointSigmoid via `net_new_tokens()`.
    pub decode_chosen_overlap_blocks: u32,

    /// Tier-weighted overlap credit on the cache-hot decode worker, in
    /// block-equivalent units. Pre-computed by the probe as
    /// `overlap_score_credit*device + host_weight*host + disk_weight*disk +
    /// shared_multiplier*shared`. Used by CostEquation in the AGG cost's
    /// prefill compute term. `None` if the probe didn't compute it (e.g.
    /// test selectors).
    pub decode_chosen_tier_overlap_credit_blocks: Option<f64>,

    /// Projected active blocks on the cache-hot decode worker (i.e.
    /// `active_blocks(d_hot) + new_blocks(d_hot)`). The `decode_block` term
    /// for that worker. `None` if the selector did not surface it.
    pub decode_chosen_load_blocks: Option<usize>,

    // === Prefill side — cost-equation-chosen prefill worker (DISAGG prefill) ===
    /// Tier-weighted overlap credit on the chosen prefill worker, in
    /// block-equivalent units. Same formula as `decode_chosen_tier_overlap_credit_blocks`.
    /// Used by CostEquation in the DISAGG cost's prefill compute term.
    /// `None` if the prefill peek did not run.
    pub prefill_chosen_tier_overlap_credit_blocks: Option<f64>,

    /// Projected active blocks on the cost-equation-chosen prefill worker.
    /// `None` if the prefill peek did not run.
    pub prefill_chosen_load_blocks: Option<usize>,

    // === Decode min-load worker (DISAGG decode post-prefill re-pick) ===
    /// `min_d decode_block(d)` — projected active blocks on the load-min
    /// decode worker, from the post-handoff load-only peek
    /// (`overlap_score_credit=0`). Models the standard-disagg decode re-pick.
    /// Used only by CostEquation.
    pub decode_pool_min_load_blocks: Option<usize>,

    /// Device-prefix overlap (blocks) on the same load-min decode worker.
    /// Used by CostEquation's delta-aware transfer term as
    /// `transfer_cost_scale * (prompt_blocks − decode_min_overlap_blocks)`.
    /// `None` if the load-only peek did not run.
    pub decode_min_overlap_blocks: Option<u32>,
}

impl ConditionalPrefillDecisionInput {
    /// Effective net-new prefill in tokens after the decode-side device
    /// cache hit is subtracted. Used by TokenCap.
    pub fn net_new_tokens(self) -> usize {
        let overlap_tokens =
            (self.decode_chosen_overlap_blocks as usize).saturating_mul(self.block_size);
        self.prompt_tokens.saturating_sub(overlap_tokens)
    }
}

pub trait ConditionalPrefillPolicy: Send + Sync {
    fn is_enabled(&self) -> bool;

    /// Does this policy need the cost-equation RHS terms
    /// (`prefill_chosen_tier_overlap_credit_blocks`, `prefill_chosen_load_blocks`,
    /// `decode_pool_min_load_blocks`, `decode_min_overlap_blocks`)? The probe
    /// only does the extra prefill / load-only decode lookups when this returns
    /// true.
    fn needs_cost_terms(&self) -> bool {
        false
    }

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool;

    /// Return the computed `(agg_cost, disagg_cost)` if this policy compares
    /// mode-level costs (CostEquation does; TokenCap and JointSigmoid don't).
    /// Returns `None` if the policy doesn't model costs, or if required input
    /// fields are missing. Used by callers that want to record the cost values
    /// for offline analysis (separate from making the bypass decision).
    fn evaluate_costs(&self, _input: ConditionalPrefillDecisionInput) -> Option<(f64, f64)> {
        None
    }
}

/// Build the policy implementation from router config. Returns a boxed trait
/// object so the policy can be swapped without changing `PrefillRouter` or
/// other consumers.
pub fn make_conditional_prefill_policy(
    config: Option<&KvRouterConfig>,
) -> Box<dyn ConditionalPrefillPolicy> {
    let Some(config) = config else {
        return Box::new(TokenCapConditionalPrefillPolicy::default());
    };
    match config.conditional_prefill_policy {
        ConditionalPrefillPolicyKind::TokenCap => {
            Box::new(TokenCapConditionalPrefillPolicy::from_config(Some(config)))
        }
    }
}

// -- TokenCap policy --

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCapConditionalPrefillPolicy {
    enabled: bool,
    max_new_tokens: usize,
}

impl TokenCapConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        let Some(config) = config else {
            return Self::default();
        };

        Self {
            enabled: config.conditional_prefill_enabled,
            max_new_tokens: config.conditional_prefill_max_new_tokens,
        }
    }
}

impl Default for TokenCapConditionalPrefillPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_new_tokens: DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS,
        }
    }
}

impl ConditionalPrefillPolicy for TokenCapConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        self.enabled && input.net_new_tokens() <= self.max_new_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `overlap_blocks` is in blocks (not tokens). Caller is responsible for
    /// picking block-aligned values; the helper does no rounding.
    fn token_cap_input(
        prompt_tokens: usize,
        overlap_blocks: u32,
    ) -> ConditionalPrefillDecisionInput {
        ConditionalPrefillDecisionInput {
            prompt_tokens,
            block_size: 16,
            decode_chosen_overlap_blocks: overlap_blocks,
            decode_chosen_tier_overlap_credit_blocks: None,
            decode_chosen_load_blocks: None,
            prefill_chosen_tier_overlap_credit_blocks: None,
            prefill_chosen_load_blocks: None,
            decode_pool_min_load_blocks: None,
            decode_min_overlap_blocks: None,
        }
    }

    #[test]
    fn token_cap_policy_is_disabled_by_default() {
        let policy = TokenCapConditionalPrefillPolicy::default();

        assert!(!policy.is_enabled());
        assert_eq!(
            policy.max_new_tokens,
            DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS
        );
        assert!(!policy.should_bypass_remote_prefill(token_cap_input(1, 0)));
    }

    #[test]
    fn token_cap_policy_bypasses_at_or_below_cap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 160,
        };

        // block_size=16, so overlap_blocks=5 → 80 overlap tokens.
        // prompt=240, overlap=80 → net_new=160 ≤ 160 → bypass.
        assert!(policy.should_bypass_remote_prefill(token_cap_input(240, 5)));
        // prompt=256, overlap=80 → net_new=176 > 160 → no bypass.
        assert!(!policy.should_bypass_remote_prefill(token_cap_input(256, 5)));
    }

    #[test]
    fn token_cap_policy_allows_no_overlap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 160,
        };

        assert!(policy.should_bypass_remote_prefill(token_cap_input(160, 0)));
    }

    #[test]
    fn policy_kind_round_trips() {
        let kind = ConditionalPrefillPolicyKind::TokenCap;
        assert_eq!(
            ConditionalPrefillPolicyKind::from_str(kind.as_str()),
            Some(kind)
        );
        assert_eq!(ConditionalPrefillPolicyKind::from_str("nonsense"), None);
    }
}
