// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::config::{ConditionalPrefillPolicyKind, KvRouterConfig, RouterConfigOverride};

use crate::protocols::common::preprocessor::{BootstrapInfo, PrefillResult};

/// Inputs passed to a `ConditionalPrefillPolicy` when deciding whether to
/// bypass remote prefill. Fields beyond `prompt_tokens` / `overlap_tokens` are
/// optional because not every policy needs them; the probe populates the ones
/// it has cheap access to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ConditionalPrefillDecisionInput {
    /// Total prompt token count for this request.
    pub prompt_tokens: usize,

    /// Decode-side cache overlap (tokens) on the chosen decode worker — output
    /// of the decode router's `find_best_match_details` at probe time.
    pub overlap_tokens: usize,

    /// KV cache block size, in tokens. Used to convert between token-units and
    /// block-units when evaluating the cost equation.
    pub block_size: usize,

    /// Projected active decode blocks on the LHS-chosen decode worker (the
    /// `decode_block` term in the cost equation). `None` if the selector did
    /// not surface it (e.g. test selectors).
    pub chosen_decode_blocks: Option<usize>,

    /// `min_p logit_full(p) = min_p [ potential_prefill_block(p) + decode_block(p) ]`
    /// — the projected cost of routing through the best prefill worker.
    /// In block units. `None` if the policy didn't request this term.
    pub prefill_min_logit_full: Option<f64>,

    /// `min_d decode_block(d)` — the load on the least-loaded decode worker,
    /// the cost of the post-prefill decode hop in standard disagg (where the
    /// re-pick uses `overlap_weight = 0`). In block units. `None` if the
    /// policy didn't request this term.
    pub decode_pool_min_load_blocks: Option<usize>,
}

impl ConditionalPrefillDecisionInput {
    pub fn net_new_tokens(self) -> usize {
        self.prompt_tokens.saturating_sub(self.overlap_tokens)
    }

    /// `logit_full(d_chosen) = potential_prefill_block(d) + decode_block(d)`
    /// — the bypass cost: local prefill work on the chosen decode worker plus
    /// its existing load. Returns `None` if `chosen_decode_blocks` is missing.
    pub fn bypass_logit_full(self) -> Option<f64> {
        let chosen = self.chosen_decode_blocks? as f64;
        let block_size = self.block_size.max(1) as f64;
        let potential_prefill_block = self.net_new_tokens() as f64 / block_size;
        Some(potential_prefill_block + chosen)
    }
}

pub(super) trait ConditionalPrefillPolicy: Send + Sync {
    fn is_enabled(&self) -> bool;

    /// Does this policy need the cost-equation RHS terms
    /// (`prefill_min_logit_full`, `decode_pool_min_load_blocks`)? The probe
    /// only does the extra prefill / load-only decode lookups when this
    /// returns true.
    fn needs_cost_terms(&self) -> bool {
        false
    }

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool;
}

/// Build the policy implementation from router config. Returns a boxed trait
/// object so the policy can be swapped without changing `PrefillRouter`.
pub(super) fn make_conditional_prefill_policy(
    config: Option<&KvRouterConfig>,
) -> Box<dyn ConditionalPrefillPolicy> {
    let Some(config) = config else {
        return Box::new(TokenCapConditionalPrefillPolicy::default());
    };
    match config.conditional_prefill_policy {
        ConditionalPrefillPolicyKind::TokenCap => {
            Box::new(TokenCapConditionalPrefillPolicy::from_config(Some(config)))
        }
        ConditionalPrefillPolicyKind::CostEquation => Box::new(
            CostEquationConditionalPrefillPolicy::from_config(Some(config)),
        ),
    }
}

// -- TokenCap policy --

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenCapConditionalPrefillPolicy {
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
            max_new_tokens: dynamo_kv_router::config::DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS,
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

// -- CostEquation policy --

/// Bypass when the selector cost equation says local prefill on the chosen
/// decode worker is cheaper than remote prefill + standard decode re-pick.
///
/// LHS = `logit_full(d_chosen) = potential_prefill_block(d) + decode_block(d)`
/// RHS = `min_p logit_full(p) + min_d decode_block(d) + transfer_cost`
///
/// Falls back to a no-bypass decision if the required cost terms are missing
/// from the input (e.g. the probe wasn't able to query both routers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CostEquationConditionalPrefillPolicy {
    enabled: bool,
    /// Static config: KV transfer cost in block units, added to the RHS.
    /// Stubbed at 0 in v1; TODO model as `transfer_constant * num_transferred_blocks`
    /// once per-backend transfer-bandwidth calibration lands.
    transfer_cost_blocks: usize,
}

impl CostEquationConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        let Some(config) = config else {
            return Self::default();
        };
        Self {
            enabled: config.conditional_prefill_enabled,
            transfer_cost_blocks: config.conditional_prefill_transfer_cost_blocks,
        }
    }
}

impl Default for CostEquationConditionalPrefillPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            transfer_cost_blocks: 0,
        }
    }
}

impl ConditionalPrefillPolicy for CostEquationConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn needs_cost_terms(&self) -> bool {
        true
    }

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }
        let Some(lhs) = input.bypass_logit_full() else {
            return false;
        };
        let Some(prefill_min) = input.prefill_min_logit_full else {
            return false;
        };
        let Some(decode_pool_min) = input.decode_pool_min_load_blocks else {
            return false;
        };
        let rhs = prefill_min + decode_pool_min as f64 + self.transfer_cost_blocks as f64;
        lhs < rhs
    }
}

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
pub(super) enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result
    Completed(PrefillResult),
}

pub(super) enum PrefillResolveDecision {
    Resolved {
        worker_id: u64,
        dp_rank: Option<u32>,
        bootstrap_info: BootstrapInfo,
    },
    Unavailable,
    NotActivated,
    NoBootstrapEndpoint,
}

pub(super) fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_weight: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS;

    fn token_cap_input(prompt: usize, overlap: usize) -> ConditionalPrefillDecisionInput {
        ConditionalPrefillDecisionInput {
            prompt_tokens: prompt,
            overlap_tokens: overlap,
            block_size: 16,
            chosen_decode_blocks: None,
            prefill_min_logit_full: None,
            decode_pool_min_load_blocks: None,
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
            max_new_tokens: 10,
        };

        assert!(policy.should_bypass_remote_prefill(token_cap_input(15, 5)));
        assert!(!policy.should_bypass_remote_prefill(token_cap_input(16, 5)));
    }

    #[test]
    fn token_cap_policy_allows_no_overlap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 10,
        };

        assert!(policy.should_bypass_remote_prefill(token_cap_input(10, 0)));
    }

    // -- CostEquation policy tests --

    fn cost_input(
        prompt: usize,
        overlap: usize,
        chosen_decode: usize,
        prefill_min: f64,
        decode_pool_min: usize,
    ) -> ConditionalPrefillDecisionInput {
        ConditionalPrefillDecisionInput {
            prompt_tokens: prompt,
            overlap_tokens: overlap,
            block_size: 16,
            chosen_decode_blocks: Some(chosen_decode),
            prefill_min_logit_full: Some(prefill_min),
            decode_pool_min_load_blocks: Some(decode_pool_min),
        }
    }

    fn enabled_cost_policy(transfer: usize) -> CostEquationConditionalPrefillPolicy {
        CostEquationConditionalPrefillPolicy {
            enabled: true,
            transfer_cost_blocks: transfer,
        }
    }

    #[test]
    fn cost_policy_disabled_never_bypasses() {
        let policy = CostEquationConditionalPrefillPolicy::default();
        assert!(!policy.is_enabled());
        // even with strongly bypass-favorable inputs
        assert!(!policy.should_bypass_remote_prefill(cost_input(160, 160, 0, 100.0, 100)));
    }

    #[test]
    fn cost_policy_bypasses_when_lhs_below_rhs() {
        let policy = enabled_cost_policy(0);
        // prompt=160 tokens / 16 = 10 blocks net-new; overlap=0 → potential_prefill_block=10
        // chosen_decode_blocks=2 → LHS = 12
        // prefill_min=20, decode_pool_min=5, transfer=0 → RHS = 25
        // 12 < 25 → bypass
        assert!(policy.should_bypass_remote_prefill(cost_input(160, 0, 2, 20.0, 5)));
    }

    #[test]
    fn cost_policy_no_bypass_when_lhs_above_rhs() {
        let policy = enabled_cost_policy(0);
        // LHS=12 (same as above), but RHS=10 (prefill_min=5, decode_pool_min=5, transfer=0)
        // 12 > 10 → no bypass
        assert!(!policy.should_bypass_remote_prefill(cost_input(160, 0, 2, 5.0, 5)));
    }

    #[test]
    fn cost_policy_bypasses_more_when_overlap_high() {
        let policy = enabled_cost_policy(0);
        // overlap covers 9 of 10 blocks → potential_prefill_block=1, LHS=3
        // RHS stays modest → bypass even with light prefill load
        assert!(policy.should_bypass_remote_prefill(cost_input(160, 144, 2, 5.0, 5)));
    }

    #[test]
    fn cost_policy_transfer_cost_raises_rhs() {
        // LHS=12, base RHS=10 → no bypass. transfer_cost=5 raises RHS to 15 → bypass.
        let policy = enabled_cost_policy(5);
        assert!(policy.should_bypass_remote_prefill(cost_input(160, 0, 2, 5.0, 5)));
    }

    #[test]
    fn cost_policy_no_bypass_when_signals_missing() {
        let policy = enabled_cost_policy(0);
        let mut input = cost_input(160, 0, 2, 20.0, 5);
        input.prefill_min_logit_full = None;
        assert!(!policy.should_bypass_remote_prefill(input));

        let mut input = cost_input(160, 0, 2, 20.0, 5);
        input.decode_pool_min_load_blocks = None;
        assert!(!policy.should_bypass_remote_prefill(input));

        let mut input = cost_input(160, 0, 2, 20.0, 5);
        input.chosen_decode_blocks = None;
        assert!(!policy.should_bypass_remote_prefill(input));
    }

    #[test]
    fn policy_kind_round_trips() {
        for kind in [
            ConditionalPrefillPolicyKind::TokenCap,
            ConditionalPrefillPolicyKind::CostEquation,
        ] {
            assert_eq!(
                ConditionalPrefillPolicyKind::from_str(kind.as_str()),
                Some(kind)
            );
        }
        assert_eq!(ConditionalPrefillPolicyKind::from_str("nonsense"), None);
    }
}
