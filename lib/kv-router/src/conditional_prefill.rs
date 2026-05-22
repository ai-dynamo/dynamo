// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-prefill bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait operates over a struct of
//! summary signals (`ConditionalPrefillDecisionInput`); both the live
//! `PrefillRouter` operator (`lib/llm/src/kv_router/prefill_router/`) and
//! offline replay (`lib/mocker/src/replay/offline/disagg.rs`) call into the
//! same code.
//!
//! Trait methods are `async` so a policy's slow path can consult an external
//! service (e.g. the Regression policy's NATS round-trip to the Python
//! decision sidecar). Policies with no I/O return ready values — the await
//! is a no-op.
//!
//! Two policies ship today:
//! - `TokenCapConditionalPrefillPolicy`: bypass when `net_new_tokens <= cap`.
//! - `RegressionConditionalPrefillPolicy`: fast-path router-state checks +
//!   slow-path regression-backed cost compare (Cost 4A; see
//!   `docs/model_costs_design.md`).

use std::sync::Arc;

use async_trait::async_trait;

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

    // === Capacity / queue signals for Regression policy fast path ===
    /// Total KV block capacity (`num_gpu_blocks`) on the cache-hot decode
    /// worker. Used alongside `decode_chosen_load_blocks` to compute
    /// `available_ratio = (max − active) / max` for the `roomy(d)` predicate.
    /// `None` if the probe didn't surface it.
    pub decode_chosen_max_blocks: Option<usize>,

    /// Total KV block capacity on the cost-equation-chosen prefill worker.
    /// Used for the `roomy(p)` predicate. `None` if the prefill peek didn't
    /// run or didn't surface it.
    pub prefill_chosen_max_blocks: Option<usize>,

    /// Queued (waiting) block count on the cache-hot decode worker — requests
    /// admitted but not yet scheduled into a forward pass. Distinct from
    /// `decode_chosen_load_blocks` (which counts *active* blocks). Feeds
    /// `roomy(d)`. `None` if the probe didn't surface it.
    pub decode_chosen_queued_blocks: Option<u32>,

    /// Queued block count on the cost-equation-chosen prefill worker. Feeds
    /// `roomy(p)`. `None` if the prefill peek didn't run or didn't surface
    /// the value.
    pub prefill_chosen_queued_blocks: Option<u32>,
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

#[async_trait]
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

    /// Decide whether the request should skip remote prefill. Async so a
    /// policy's slow path may consult an external service (e.g. the
    /// Regression policy's NATS round-trip to the decision sidecar). Policies
    /// with no I/O return a ready value — `.await` is a no-op in that case.
    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool;

    /// Return the computed `(agg_cost, disagg_cost)` if this policy compares
    /// mode-level costs. Returns `None` if the policy doesn't model costs, or
    /// if required input fields are missing. Used by callers that want to
    /// record the cost values for offline analysis (separate from making the
    /// bypass decision).
    async fn evaluate_costs(&self, _input: ConditionalPrefillDecisionInput) -> Option<(f64, f64)> {
        None
    }
}

/// Slow-path cost evaluator used by the `RegressionConditionalPrefillPolicy`.
///
/// Decouples the policy from the transport that fetches `(agg_ttft, disagg_ttft)`
/// from the Planner regression models. Production wires up an RPC client
/// (NATS request/reply to the Python decision service); the mocker (Phase 5)
/// wires up an in-process PyO3 callback. Same policy code, different impl.
#[async_trait]
pub trait CostEvaluator: Send + Sync {
    async fn evaluate(&self, request: CostEvalRequest) -> CostEvalResponse;
}

/// Request features sent to a `CostEvaluator`. Wire-stable: msgpack-serialized
/// in the prod RPC path. Field names + types are the single source of truth
/// for the corresponding Python `wire.py` dataclass.
///
/// The regression's only per-request input is `kv_hit_rate` (see
/// `PrefillRegressionModel.estimate_next_ttft` / `AggRegressionModel.estimate_next_ttft`);
/// queue depth, `max_num_batched_tokens`, decode KV, and avg ISL are all
/// tracked sidecar-side from FPM + engine config. So the request carries one
/// hit rate per mode plus the prompt token count for sidecar observability.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEvalRequest {
    /// Caller-attached identifier for tracing; not consumed by the regression.
    pub request_id: String,
    /// Full prompt tokens for the request. Carried for sidecar observability
    /// (logs, tracing) and as a reference for future per-request-ISL
    /// overrides; not currently passed to the regression.
    pub prompt_tokens: usize,
    /// AGG-side KV-hit rate ∈ [0, 1]. Derived from device-prefix overlap on
    /// the cache-hot decode worker (the AGG bypass target). Passed straight
    /// into `AggRegressionModel.estimate_next_ttft(..., kv_hit_rate=...)`.
    pub agg_kv_hit_rate: f64,
    /// DISAGG-side KV-hit rate ∈ [0, 1]. Derived from the tier-weighted
    /// overlap credit on the cost-equation-chosen prefill worker. Passed
    /// straight into `PrefillRegressionModel.estimate_next_ttft(..., kv_hit_rate=...)`.
    pub disagg_kv_hit_rate: f64,
}

/// Response from a `CostEvaluator`. `agg_ttft_ms` and `disagg_ttft_ms` are
/// predicted TTFT in milliseconds for each mode; either may be `None` when
/// the corresponding regression isn't warm. The `*_warm` flags expose warmth
/// explicitly so the policy can apply conservative-DISAGG fallback semantics
/// without having to inspect the ms values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEvalResponse {
    pub agg_ttft_ms: Option<f64>,
    pub disagg_ttft_ms: Option<f64>,
    pub agg_warm: bool,
    pub disagg_warm: bool,
}

impl CostEvalResponse {
    /// Sentinel response used when the evaluator is unreachable (RPC failed,
    /// timed out, etc.). All fields set so the policy's conservative-DISAGG
    /// fallback fires.
    pub fn unavailable() -> Self {
        Self {
            agg_ttft_ms: None,
            disagg_ttft_ms: None,
            agg_warm: false,
            disagg_warm: false,
        }
    }
}

/// Shared `Arc<dyn CostEvaluator>` handed to the policy factory. Held by
/// `RegressionConditionalPrefillPolicy`; ignored by other policies.
pub type SharedCostEvaluator = Arc<dyn CostEvaluator>;

/// Build the policy implementation from router config. Returns a boxed trait
/// object so the policy can be swapped without changing `PrefillRouter` or
/// other consumers.
///
/// `evaluator` is consumed only by `ConditionalPrefillPolicyKind::Regression`;
/// the other policy kinds ignore it. Callers that never expect to route through
/// the Regression policy can pass `None` and we fall back to an evaluator that
/// always reports "unavailable" — the policy then takes its conservative
/// DISAGG fallback on every slow-path decision.
pub fn make_conditional_prefill_policy(
    config: Option<&KvRouterConfig>,
    evaluator: Option<SharedCostEvaluator>,
) -> Box<dyn ConditionalPrefillPolicy> {
    let Some(config) = config else {
        return Box::new(TokenCapConditionalPrefillPolicy::default());
    };
    match config.conditional_prefill_policy {
        ConditionalPrefillPolicyKind::TokenCap => {
            Box::new(TokenCapConditionalPrefillPolicy::from_config(Some(config)))
        }
        ConditionalPrefillPolicyKind::Regression => {
            let evaluator = evaluator.unwrap_or_else(|| Arc::new(UnavailableCostEvaluator) as _);
            Box::new(RegressionConditionalPrefillPolicy::from_config(
                Some(config),
                evaluator,
            ))
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

#[async_trait]
impl ConditionalPrefillPolicy for TokenCapConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        self.enabled && input.net_new_tokens() <= self.max_new_tokens
    }
}

// -- Regression policy --

/// Two-phase router cost policy (the "Cost 4A" design in
/// `docs/model_costs_design.md`). The fast path filters obvious cases on
/// router-side state; only contested decisions consult the regression-backed
/// slow path through an injected `CostEvaluator`.
///
/// ```text
/// fast path:
///   eff_isl = prompt_tokens - decode_chosen_overlap_blocks * block_size
///   if roomy(p) or eff_isl > large_prompt_threshold_tokens → DISAGG
///   else if roomy(d) → AGG
/// slow path:
///   resp = evaluator.evaluate(features).await
///   if !resp.agg_warm or !resp.disagg_warm → DISAGG  (conservative fallback)
///   else → AGG iff resp.agg_ttft_ms < resp.disagg_ttft_ms
/// ```
///
/// `roomy(w)` is true when both `(max-active)/max ≥ roomy_available_ratio_threshold`
/// AND `queued_blocks ≤ roomy_queued_blocks_threshold`. Missing capacity/queue
/// signals → conservatively treat as not roomy.
pub struct RegressionConditionalPrefillPolicy {
    enabled: bool,
    /// Effective ISL (post device-cache) above this triggers DISAGG on the
    /// fast path — too much prefill to commit to the decode worker.
    large_prompt_threshold_tokens: usize,
    /// `(max - active) / max` must be at least this for the worker to count
    /// as roomy. Range `[0, 1]`.
    roomy_available_ratio_threshold: f64,
    /// `queued_blocks` must be at or below this for the worker to count as
    /// roomy. `0` = no queued work allowed.
    roomy_queued_blocks_threshold: u32,
    evaluator: SharedCostEvaluator,
}

impl RegressionConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>, evaluator: SharedCostEvaluator) -> Self {
        let Some(config) = config else {
            return Self::with_defaults(evaluator);
        };
        Self {
            enabled: config.conditional_prefill_enabled,
            large_prompt_threshold_tokens: config
                .conditional_prefill_regression_large_prompt_threshold_tokens,
            roomy_available_ratio_threshold: config
                .conditional_prefill_regression_roomy_available_ratio,
            roomy_queued_blocks_threshold: config
                .conditional_prefill_regression_roomy_queued_blocks,
            evaluator,
        }
    }

    pub fn with_defaults(evaluator: SharedCostEvaluator) -> Self {
        Self {
            enabled: false,
            large_prompt_threshold_tokens: DEFAULT_REGRESSION_LARGE_PROMPT_THRESHOLD_TOKENS,
            roomy_available_ratio_threshold: DEFAULT_REGRESSION_ROOMY_AVAILABLE_RATIO,
            roomy_queued_blocks_threshold: DEFAULT_REGRESSION_ROOMY_QUEUED_BLOCKS,
            evaluator,
        }
    }

    /// Build the wire-stable features for the slow-path RPC. Returns `None`
    /// when the DISAGG-side credit isn't available (the prefill peek didn't
    /// run or didn't surface a tier credit) — caller short-circuits to
    /// conservative DISAGG without paying the RPC round-trip.
    fn cost_eval_request(&self, input: ConditionalPrefillDecisionInput) -> Option<CostEvalRequest> {
        let prefill_tier_credit = input.prefill_chosen_tier_overlap_credit_blocks?;
        let denom = input.prompt_tokens.max(1) as f64;
        let block_size = input.block_size as f64;
        let agg_overlap_tokens = (input.decode_chosen_overlap_blocks as f64) * block_size;
        let disagg_overlap_tokens = prefill_tier_credit * block_size;
        Some(CostEvalRequest {
            // The policy itself doesn't carry a request ID; callers that want
            // request-level tracing can attach one to the evaluator wrapper.
            request_id: String::new(),
            prompt_tokens: input.prompt_tokens,
            agg_kv_hit_rate: (agg_overlap_tokens / denom).clamp(0.0, 1.0),
            disagg_kv_hit_rate: (disagg_overlap_tokens / denom).clamp(0.0, 1.0),
        })
    }
}

/// Default for the `roomy(w)` fraction-available threshold. Hand-picked
/// starting point; tune from mocker data.
pub const DEFAULT_REGRESSION_ROOMY_AVAILABLE_RATIO: f64 = 0.5;
/// Default for the `roomy(w)` queued-blocks threshold. `0` = no queued work
/// allowed.
pub const DEFAULT_REGRESSION_ROOMY_QUEUED_BLOCKS: u32 = 0;
/// Default effective-ISL ceiling above which fast path forces DISAGG.
/// Roughly 16k tokens — keeps very long prompts on the disagg path even when
/// decode has headroom.
pub const DEFAULT_REGRESSION_LARGE_PROMPT_THRESHOLD_TOKENS: usize = 16_384;

/// True when worker `w` has both capacity headroom and an empty/short queue,
/// per the configured thresholds. Missing capacity/queue signals → not roomy
/// (conservative). All four fields must be present to count.
fn worker_is_roomy(
    load_blocks: Option<usize>,
    max_blocks: Option<usize>,
    queued_blocks: Option<u32>,
    available_ratio_threshold: f64,
    queued_threshold: u32,
) -> bool {
    let (Some(load), Some(max), Some(queued)) = (load_blocks, max_blocks, queued_blocks) else {
        return false;
    };
    if max == 0 {
        return false;
    }
    let available = max.saturating_sub(load) as f64;
    let ratio = available / max as f64;
    ratio >= available_ratio_threshold && queued <= queued_threshold
}

#[async_trait]
impl ConditionalPrefillPolicy for RegressionConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn needs_cost_terms(&self) -> bool {
        true
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }

        // Fast path — evaluate roomy() and large-prompt cutoff on router state.
        let p_roomy = worker_is_roomy(
            input.prefill_chosen_load_blocks,
            input.prefill_chosen_max_blocks,
            input.prefill_chosen_queued_blocks,
            self.roomy_available_ratio_threshold,
            self.roomy_queued_blocks_threshold,
        );
        let overlap_tokens =
            (input.decode_chosen_overlap_blocks as usize).saturating_mul(input.block_size);
        let eff_isl_tokens = input.prompt_tokens.saturating_sub(overlap_tokens);
        if p_roomy || eff_isl_tokens > self.large_prompt_threshold_tokens {
            return false;
        }

        let d_roomy = worker_is_roomy(
            input.decode_chosen_load_blocks,
            input.decode_chosen_max_blocks,
            input.decode_chosen_queued_blocks,
            self.roomy_available_ratio_threshold,
            self.roomy_queued_blocks_threshold,
        );
        if d_roomy {
            return true;
        }

        // Slow path — consult the regression-backed evaluator. Conservative
        // DISAGG fallback when the DISAGG-side credit is missing (skip the
        // RPC entirely) or when either regression is cold.
        let Some(request) = self.cost_eval_request(input) else {
            return false;
        };
        let resp = self.evaluator.evaluate(request).await;
        match (resp.agg_ttft_ms, resp.disagg_ttft_ms) {
            (Some(agg_ms), Some(disagg_ms)) if resp.agg_warm && resp.disagg_warm => {
                agg_ms < disagg_ms
            }
            _ => false,
        }
    }

    async fn evaluate_costs(&self, input: ConditionalPrefillDecisionInput) -> Option<(f64, f64)> {
        let request = self.cost_eval_request(input)?;
        let resp = self.evaluator.evaluate(request).await;
        match (resp.agg_ttft_ms, resp.disagg_ttft_ms) {
            (Some(agg_ms), Some(disagg_ms)) if resp.agg_warm && resp.disagg_warm => {
                Some((agg_ms, disagg_ms))
            }
            _ => None,
        }
    }
}

/// `CostEvaluator` that always reports unavailable. Used as the default when
/// the Regression policy is selected without a real evaluator wired up — the
/// policy then takes its conservative DISAGG fallback on every slow-path
/// decision. Cheap to construct; useful in tests and during bootstrap.
pub struct UnavailableCostEvaluator;

#[async_trait]
impl CostEvaluator for UnavailableCostEvaluator {
    async fn evaluate(&self, _request: CostEvalRequest) -> CostEvalResponse {
        CostEvalResponse::unavailable()
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
            decode_chosen_max_blocks: None,
            prefill_chosen_max_blocks: None,
            decode_chosen_queued_blocks: None,
            prefill_chosen_queued_blocks: None,
        }
    }

    #[tokio::test]
    async fn token_cap_policy_is_disabled_by_default() {
        let policy = TokenCapConditionalPrefillPolicy::default();

        assert!(!policy.is_enabled());
        assert_eq!(
            policy.max_new_tokens,
            DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS
        );
        assert!(
            !policy
                .should_bypass_remote_prefill(token_cap_input(1, 0))
                .await
        );
    }

    #[tokio::test]
    async fn token_cap_policy_bypasses_at_or_below_cap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 160,
        };

        // block_size=16, so overlap_blocks=5 → 80 overlap tokens.
        // prompt=240, overlap=80 → net_new=160 ≤ 160 → bypass.
        assert!(
            policy
                .should_bypass_remote_prefill(token_cap_input(240, 5))
                .await
        );
        // prompt=256, overlap=80 → net_new=176 > 160 → no bypass.
        assert!(
            !policy
                .should_bypass_remote_prefill(token_cap_input(256, 5))
                .await
        );
    }

    #[tokio::test]
    async fn token_cap_policy_allows_no_overlap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 160,
        };

        assert!(
            policy
                .should_bypass_remote_prefill(token_cap_input(160, 0))
                .await
        );
    }

    #[test]
    fn policy_kind_round_trips() {
        for kind in [
            ConditionalPrefillPolicyKind::TokenCap,
            ConditionalPrefillPolicyKind::Regression,
        ] {
            assert_eq!(
                ConditionalPrefillPolicyKind::from_str(kind.as_str()),
                Some(kind)
            );
        }
        assert_eq!(ConditionalPrefillPolicyKind::from_str("nonsense"), None);
    }

    // -- Regression policy tests --

    /// Stub `CostEvaluator` whose response is set at construction. Lets each
    /// test drive the slow-path branch deterministically without needing a
    /// real Planner regression behind it.
    struct StubCostEvaluator {
        response: CostEvalResponse,
    }

    #[async_trait]
    impl CostEvaluator for StubCostEvaluator {
        async fn evaluate(&self, _request: CostEvalRequest) -> CostEvalResponse {
            self.response
        }
    }

    /// Build an input that lands the policy in the slow path: neither side
    /// roomy (90% load), short prompt (below the 16k large-prompt cutoff),
    /// and a populated prefill-side tier credit so the slow-path RPC fires
    /// rather than short-circuiting to conservative DISAGG.
    fn regression_input_slow_path() -> ConditionalPrefillDecisionInput {
        ConditionalPrefillDecisionInput {
            prompt_tokens: 1024,
            block_size: 64,
            decode_chosen_overlap_blocks: 0,
            decode_chosen_tier_overlap_credit_blocks: None,
            decode_chosen_load_blocks: Some(900),
            prefill_chosen_tier_overlap_credit_blocks: Some(0.0),
            prefill_chosen_load_blocks: Some(900),
            decode_pool_min_load_blocks: None,
            decode_min_overlap_blocks: None,
            decode_chosen_max_blocks: Some(1000),
            prefill_chosen_max_blocks: Some(1000),
            decode_chosen_queued_blocks: Some(10),
            prefill_chosen_queued_blocks: Some(10),
        }
    }

    fn make_regression_policy(response: CostEvalResponse) -> RegressionConditionalPrefillPolicy {
        RegressionConditionalPrefillPolicy {
            enabled: true,
            large_prompt_threshold_tokens: 16_384,
            roomy_available_ratio_threshold: 0.5,
            roomy_queued_blocks_threshold: 0,
            evaluator: Arc::new(StubCostEvaluator { response }),
        }
    }

    #[tokio::test]
    async fn regression_disabled_never_bypasses() {
        let mut policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(1.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        policy.enabled = false;
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_fast_path_p_roomy_forces_disagg() {
        // prefill_chosen load=10/1000 → 99% available, well above 0.5 threshold.
        // Slow-path verdict would say bypass, but fast path overrides.
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(1.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        let mut input = regression_input_slow_path();
        input.prefill_chosen_load_blocks = Some(10);
        input.prefill_chosen_queued_blocks = Some(0);
        assert!(!policy.should_bypass_remote_prefill(input).await);
    }

    #[tokio::test]
    async fn regression_fast_path_large_prompt_forces_disagg() {
        // eff_isl = 20000 > 16384 → fast path returns DISAGG regardless.
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(1.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        let mut input = regression_input_slow_path();
        input.prompt_tokens = 20_000;
        assert!(!policy.should_bypass_remote_prefill(input).await);
    }

    #[tokio::test]
    async fn regression_fast_path_d_roomy_forces_agg() {
        // decode_chosen empty → not p-roomy (prefill_chosen still loaded),
        // prompt small → fast path proceeds to d-roomy check → AGG. Slow path
        // would say DISAGG but is not consulted.
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(999.0),
            disagg_ttft_ms: Some(1.0),
            agg_warm: true,
            disagg_warm: true,
        });
        let mut input = regression_input_slow_path();
        input.decode_chosen_load_blocks = Some(10);
        input.decode_chosen_queued_blocks = Some(0);
        assert!(policy.should_bypass_remote_prefill(input).await);
    }

    #[tokio::test]
    async fn regression_slow_path_agg_wins() {
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(10.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        assert!(
            policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_slow_path_disagg_wins() {
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(100.0),
            disagg_ttft_ms: Some(10.0),
            agg_warm: true,
            disagg_warm: true,
        });
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_slow_path_cold_side_falls_back_to_disagg() {
        // AGG side cold → conservative DISAGG even though warm side looks great.
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: None,
            disagg_ttft_ms: Some(100.0),
            agg_warm: false,
            disagg_warm: true,
        });
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_slow_path_unavailable_evaluator_falls_back_to_disagg() {
        let policy = make_regression_policy(CostEvalResponse::unavailable());
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_slow_path_missing_prefill_credit_short_circuits_to_disagg() {
        // When the prefill peek didn't surface a tier credit, the policy can't
        // form a faithful DISAGG-side hit rate. Skip the RPC entirely and
        // return conservative DISAGG. Verified by setting the evaluator
        // response to one that *would* favor AGG if consulted.
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(1.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        let mut input = regression_input_slow_path();
        input.prefill_chosen_tier_overlap_credit_blocks = None;
        assert!(!policy.should_bypass_remote_prefill(input).await);
    }

    #[tokio::test]
    async fn regression_evaluate_costs_returns_pair_when_warm() {
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(10.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: true,
            disagg_warm: true,
        });
        assert_eq!(
            policy.evaluate_costs(regression_input_slow_path()).await,
            Some((10.0, 100.0))
        );
    }

    #[tokio::test]
    async fn regression_evaluate_costs_returns_none_when_cold() {
        let policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(10.0),
            disagg_ttft_ms: Some(100.0),
            agg_warm: false,
            disagg_warm: true,
        });
        assert!(
            policy
                .evaluate_costs(regression_input_slow_path())
                .await
                .is_none()
        );
    }

    #[test]
    fn worker_is_roomy_requires_all_signals_present() {
        assert!(!worker_is_roomy(None, Some(1000), Some(0), 0.5, 0));
        assert!(!worker_is_roomy(Some(10), None, Some(0), 0.5, 0));
        assert!(!worker_is_roomy(Some(10), Some(1000), None, 0.5, 0));
        assert!(worker_is_roomy(Some(10), Some(1000), Some(0), 0.5, 0));
    }

    #[test]
    fn worker_is_roomy_checks_ratio_and_queue() {
        // 50% available, exactly at threshold → roomy.
        assert!(worker_is_roomy(Some(500), Some(1000), Some(0), 0.5, 0));
        // Just below threshold → not roomy.
        assert!(!worker_is_roomy(Some(501), Some(1000), Some(0), 0.5, 0));
        // Queue blocks above threshold → not roomy even if capacity ok.
        assert!(!worker_is_roomy(Some(10), Some(1000), Some(1), 0.5, 0));
    }

    #[test]
    fn worker_is_roomy_handles_zero_max() {
        assert!(!worker_is_roomy(Some(0), Some(0), Some(0), 0.5, 0));
    }
}
