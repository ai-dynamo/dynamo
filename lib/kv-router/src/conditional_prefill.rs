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

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::{ConditionalPrefillPolicyKind, KvRouterConfig};

/// Inputs passed to a `ConditionalPrefillPolicy` when deciding whether to
/// bypass remote prefill.
///
/// Field naming convention:
/// - `decode_chosen_*` — value for the cache-hot decode worker (the AGG target,
///   picked by the agg-equation peek on the decode pool).
/// - `prefill_chosen_*` — value for the cost-equation-chosen prefill worker
///   (the DISAGG prefill target).
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

    // === Candidate worker identities (for the slow-path RPC) ===
    /// Cache-hot decode worker's identity. Same worker handles AGG-mode if
    /// bypass is chosen. The slow-path RPC carries these so the cost_eval
    /// sidecar can look up the candidate's live FPM (queue depth, in-flight
    /// decode KV) instead of falling back to pool aggregates.
    pub decode_chosen_worker_id: u64,
    pub decode_chosen_dp_rank: u32,

    /// Cost-equation-chosen prefill worker's identity. `None` if the prefill
    /// peek didn't run (`needs_cost_terms() == false`) or returned no result.
    pub prefill_chosen_worker_id: Option<u64>,
    pub prefill_chosen_dp_rank: Option<u32>,

    // === Capacity / pending-prefill signals for Regression policy fast path ===
    //
    // Sourced router-side from KvRouter accessors (live router) or equivalent
    // mocker-side state (Phase 5). See `worker_is_roomy` for the load/queued
    // semantic gap warning — these aren't literal "free blocks" + "queue depth."
    /// Static KV block capacity advertised by the cache-hot decode worker at
    /// registration. Sourced via `KvRouter::total_kv_blocks_for`. Used with
    /// `decode_chosen_load_blocks` to compute available-ratio headroom for
    /// the `roomy(d)` check. `None` if the worker config didn't surface it.
    pub decode_chosen_max_blocks: Option<usize>,

    /// Same as `decode_chosen_max_blocks` but for the cost-equation-chosen
    /// prefill worker. Used for `roomy(p)`. `None` if the prefill peek didn't
    /// run or the worker config didn't surface it.
    pub prefill_chosen_max_blocks: Option<usize>,

    /// Decay-adjusted pending-prefill load on the cache-hot decode worker,
    /// in block-equivalent units (tokens / block_size). Sourced via
    /// `KvRouter::pending_prefill_tokens_for` and rounded up. Includes the
    /// currently-running prefill (decay-credited) plus queued-at-worker work
    /// (full weight) — NOT literal queue depth. Used by `roomy(d)` as the
    /// "pending work ahead" threshold check.
    pub decode_chosen_queued_blocks: Option<u32>,

    /// Same as `decode_chosen_queued_blocks` but for the cost-equation-chosen
    /// prefill worker. Used for `roomy(p)`. `None` if the prefill peek didn't
    /// run.
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

    /// Does this policy need the extra cost-equation RHS terms
    /// (`prefill_chosen_tier_overlap_credit_blocks`, `prefill_chosen_load_blocks`,
    /// `prefill_chosen_max_blocks`, `prefill_chosen_queued_blocks`)? The
    /// probe only does the extra prefill-router peek when this returns true.
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

    /// Late-bind a `CostEvaluator` after construction. Used by the live
    /// router's `PrefillRouter::activate()`: the policy is constructed at
    /// `PrefillRouter::new()` (sync, before the runtime is wired), and the
    /// request-plane evaluator becomes constructable only later, at activation.
    ///
    /// Default impl is a no-op — policies without a slow path (TokenCap)
    /// don't need to override. The Regression policy overrides to store the
    /// evaluator in its internal `OnceLock`; subsequent calls silently
    /// no-op (the lock is set-once).
    fn try_set_cost_evaluator(&self, _evaluator: Arc<dyn CostEvaluator>) {}
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// Cache-hot decode worker's identity (the AGG candidate). The sidecar
    /// uses this to look up the candidate's live FPM (`queued_requests
    /// .sum_prefill_tokens`, `scheduled_requests.sum_decode_kv_tokens`) and
    /// feed those into `AggRegressionModel.estimate_next_ttft` — see
    /// planner's per-worker pattern at `load_scaling.py:440-444`.
    pub decode_chosen_worker_id: u64,
    pub decode_chosen_dp_rank: u32,

    /// Cost-equation-chosen prefill worker's identity (the DISAGG candidate).
    /// Sidecar uses this for the prefill regression's `queued_prefill_tokens`
    /// lookup. `None` if the prefill peek did not run — caller should
    /// short-circuit to conservative DISAGG before issuing the RPC.
    pub prefill_chosen_worker_id: Option<u64>,
    pub prefill_chosen_dp_rank: Option<u32>,
}

/// Response from a `CostEvaluator`. `agg_ttft_ms` and `disagg_ttft_ms` are
/// predicted TTFT in milliseconds for each mode; either may be `None` when
/// the corresponding regression isn't warm. The `*_warm` flags expose warmth
/// explicitly so the policy can apply conservative-DISAGG fallback semantics
/// without having to inspect the ms values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostEvalResponse {
    pub agg_ttft_ms: Option<f64>,
    pub disagg_ttft_ms: Option<f64>,
    /// Total predicted cost per side, in ms: `ttft + itl * avg_decode_length`.
    /// `avg_decode_length` is the sidecar's rolling average decode KV per
    /// request (from `AggRegressionModel._avg_decode_len`), used as the
    /// OSL multiplier. AGG-side ITL is the 2D agg regression queried at the
    /// chunked point `(prefill_tokens>0, decode_kv)`; DISAGG-side ITL is the
    /// *same* 2D regression queried at the pure-decode slice `(0, decode_kv)`.
    pub agg_total_cost_ms: Option<f64>,
    pub disagg_total_cost_ms: Option<f64>,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
/// `evaluator` is consumed only by the Regression-flavored policy kinds
/// (`ThresholdRegression`, `ThresholdOnly`, `RegressionOnly`); other kinds
/// ignore it. When `None`, the policy is constructed with no evaluator and
/// its slow path falls back to conservative DISAGG until something (typically
/// `PrefillRouter::activate()`) calls `try_set_cost_evaluator()` on the policy.
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
        ConditionalPrefillPolicyKind::ThresholdRegression => {
            Box::new(RegressionConditionalPrefillPolicy::from_config_with_mode(
                Some(config),
                evaluator,
                true,
                true,
            ))
        }
        ConditionalPrefillPolicyKind::ThresholdOnly => {
            Box::new(RegressionConditionalPrefillPolicy::from_config_with_mode(
                Some(config),
                evaluator,
                true,
                false,
            ))
        }
        ConditionalPrefillPolicyKind::RegressionOnly => {
            Box::new(RegressionConditionalPrefillPolicy::from_config_with_mode(
                Some(config),
                evaluator,
                false,
                true,
            ))
        }
        ConditionalPrefillPolicyKind::AlwaysBypass => Box::new(
            AlwaysBypassConditionalPrefillPolicy::from_config(Some(config)),
        ),
        ConditionalPrefillPolicyKind::RandomBypass => Box::new(
            RandomBypassConditionalPrefillPolicy::from_config(Some(config)),
        ),
    }
}

// -- TokenCap policy --

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCapConditionalPrefillPolicy {
    enabled: bool,
    bypass_below_tokens: usize,
}

impl TokenCapConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        let Some(config) = config else {
            return Self::default();
        };

        Self {
            enabled: config.conditional_prefill_enabled,
            bypass_below_tokens: config.conditional_prefill_bypass_below_tokens,
        }
    }
}

impl Default for TokenCapConditionalPrefillPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            bypass_below_tokens: DEFAULT_CONDITIONAL_PREFILL_BYPASS_BELOW_TOKENS,
        }
    }
}

#[async_trait]
impl ConditionalPrefillPolicy for TokenCapConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        self.enabled && input.net_new_tokens() <= self.bypass_below_tokens
    }
}

// -- AlwaysBypass policy (test-only) --

/// Always bypasses remote prefill when enabled. Test scaffolding for
/// verifying the end-to-end bypass path against real engine workers
/// (TRT-LLM, vLLM, SGLang) — strips all decision logic so the bypass path
/// is exercised on every request. Not intended for production use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AlwaysBypassConditionalPrefillPolicy {
    enabled: bool,
}

impl AlwaysBypassConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        Self {
            enabled: config.is_some_and(|c| c.conditional_prefill_enabled),
        }
    }
}

#[async_trait]
impl ConditionalPrefillPolicy for AlwaysBypassConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, _input: ConditionalPrefillDecisionInput) -> bool {
        self.enabled
    }
}

// -- RandomBypass policy (test-only) --

/// Per-request bypass probability for `RandomBypassConditionalPrefillPolicy`.
/// Hardcoded at 50% — gives the most discriminating mix of bypass /
/// non-bypass traffic for an E2E smoke test. If you need a different value,
/// either edit this constant or add a knob.
const RANDOM_BYPASS_PROBABILITY: f64 = 0.5;

/// Bypasses remote prefill with a hardcoded 50% probability. Test
/// scaffolding for stressing the routing layer's handling of mixed
/// bypass / non-bypass traffic — exercises both code paths on every run
/// without needing carefully-shaped workloads. Not intended for
/// production use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RandomBypassConditionalPrefillPolicy {
    enabled: bool,
}

impl RandomBypassConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        Self {
            enabled: config.is_some_and(|c| c.conditional_prefill_enabled),
        }
    }
}

#[async_trait]
impl ConditionalPrefillPolicy for RandomBypassConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, _input: ConditionalPrefillDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }
        use rand::Rng;
        rand::rng().random::<f64>() < RANDOM_BYPASS_PROBABILITY
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
///   if roomy(p) or eff_isl > disagg_above_tokens → DISAGG
///   else if roomy(d) or eff_isl ≤ bypass_below_tokens → AGG
/// slow path:
///   resp = evaluator.evaluate(features).await
///   if !resp.agg_warm or !resp.disagg_warm → DISAGG  (conservative fallback)
///   else if both total_cost_ms surfaced → AGG iff agg_total_cost_ms < disagg_total_cost_ms
///                                         (total_cost = ttft + itl * avg_decode_length)
///   else → AGG iff resp.agg_ttft_ms < resp.disagg_ttft_ms  (TTFT-only fallback)
/// ```
///
/// `roomy(w)` is true when both `(max-active)/max ≥ roomy_available_ratio_threshold`
/// AND `queued_blocks ≤ roomy_queued_blocks_threshold`. Missing capacity/queue
/// signals → conservatively treat as not roomy.
pub struct RegressionConditionalPrefillPolicy {
    enabled: bool,
    /// When true, evaluate the roomy()/large-prompt checks. Disabled in the
    /// `RegressionOnly` diagnostic variant so every request hits the slow path.
    enable_fast_path: bool,
    /// When true, the slow-path `CostEvaluator` RPC may run. Disabled in the
    /// `ThresholdOnly` diagnostic variant so requests that the fast path can't
    /// decide fall back to conservative DISAGG without consulting cost_eval.
    enable_slow_path: bool,
    /// Effective ISL (post device-cache) above this triggers DISAGG on the
    /// fast path — too much prefill to commit to the decode worker.
    disagg_above_tokens: usize,
    /// Effective ISL (post device-cache) at or below this triggers AGG on the
    /// fast path — small prompts are cheap to bypass and AGG saves the
    /// prefill→decode KV transfer cost.
    bypass_below_tokens: usize,
    /// `(max - active) / max` must be at least this for the worker to count
    /// as roomy. Range `[0, 1]`.
    roomy_available_ratio_threshold: f64,
    /// `queued_blocks` must be at or below this for the worker to count as
    /// roomy. `0` = no queued work allowed.
    roomy_queued_blocks_threshold: u32,
    /// Slow-path evaluator. Late-bound via `try_set_cost_evaluator()` so the
    /// router can construct the policy synchronously at startup and fill in
    /// the request-plane evaluator only after `activate()` has resolved the
    /// cost-eval sidecar's endpoint. When unset, slow-path decisions fall
    /// back to conservative DISAGG (same outcome as the evaluator returning
    /// `CostEvalResponse::unavailable()`).
    evaluator: OnceLock<SharedCostEvaluator>,
}

impl RegressionConditionalPrefillPolicy {
    /// Build with the production `ThresholdRegression` mode (fast + slow).
    pub fn from_config(
        config: Option<&KvRouterConfig>,
        evaluator: Option<SharedCostEvaluator>,
    ) -> Self {
        Self::from_config_with_mode(config, evaluator, true, true)
    }

    /// Variant-aware constructor. `enable_fast_path` / `enable_slow_path` are
    /// set by `make_conditional_prefill_policy` based on which
    /// `ConditionalPrefillPolicyKind` was requested.
    pub fn from_config_with_mode(
        config: Option<&KvRouterConfig>,
        evaluator: Option<SharedCostEvaluator>,
        enable_fast_path: bool,
        enable_slow_path: bool,
    ) -> Self {
        let policy = match config {
            Some(cfg) => Self {
                enabled: cfg.conditional_prefill_enabled,
                enable_fast_path,
                enable_slow_path,
                disagg_above_tokens: cfg.conditional_prefill_disagg_above_tokens,
                bypass_below_tokens: cfg.conditional_prefill_bypass_below_tokens,
                roomy_available_ratio_threshold: cfg
                    .conditional_prefill_regression_roomy_available_ratio,
                roomy_queued_blocks_threshold: cfg
                    .conditional_prefill_regression_roomy_queued_blocks,
                evaluator: OnceLock::new(),
            },
            None => {
                let mut p = Self::with_defaults();
                p.enable_fast_path = enable_fast_path;
                p.enable_slow_path = enable_slow_path;
                p
            }
        };
        if let Some(evaluator) = evaluator {
            // Best-effort: only `Err`s if already set, which can't happen on
            // a freshly-constructed policy.
            let _ = policy.evaluator.set(evaluator);
        }
        policy
    }

    pub fn with_defaults() -> Self {
        Self {
            enabled: false,
            enable_fast_path: true,
            enable_slow_path: true,
            disagg_above_tokens: DEFAULT_CONDITIONAL_PREFILL_DISAGG_ABOVE_TOKENS,
            bypass_below_tokens: DEFAULT_CONDITIONAL_PREFILL_BYPASS_BELOW_TOKENS,
            roomy_available_ratio_threshold: DEFAULT_REGRESSION_ROOMY_AVAILABLE_RATIO,
            roomy_queued_blocks_threshold: DEFAULT_REGRESSION_ROOMY_QUEUED_BLOCKS,
            evaluator: OnceLock::new(),
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
            decode_chosen_worker_id: input.decode_chosen_worker_id,
            decode_chosen_dp_rank: input.decode_chosen_dp_rank,
            prefill_chosen_worker_id: input.prefill_chosen_worker_id,
            prefill_chosen_dp_rank: input.prefill_chosen_dp_rank,
        })
    }
}

/// Default for the `roomy(w)` fraction-available threshold. Hand-picked
/// starting point; tune from mocker data.
pub const DEFAULT_REGRESSION_ROOMY_AVAILABLE_RATIO: f64 = 0.8;
/// Default for the `roomy(w)` queued-blocks threshold. `0` = no queued work
/// allowed.
pub const DEFAULT_REGRESSION_ROOMY_QUEUED_BLOCKS: u32 = 0;
/// Default effective-ISL ceiling above which fast path forces DISAGG.
/// Roughly 16k tokens — keeps very long prompts on the disagg path even when
/// decode has headroom.
pub const DEFAULT_CONDITIONAL_PREFILL_DISAGG_ABOVE_TOKENS: usize = 16_384;

/// Effective ISL (net new tokens) at or below which the conditional-prefill
/// policy fast-paths to AGG. Shared by TokenCap (sole bypass criterion) and
/// Regression (small-prompt fast path before falling through to slow path).
pub const DEFAULT_CONDITIONAL_PREFILL_BYPASS_BELOW_TOKENS: usize = 1024;

/// True when worker `w` has both KV-capacity headroom and a small pending-
/// prefill backlog, per the configured thresholds. Missing capacity/queue
/// signals → not roomy (conservative). All four arguments must be present.
///
/// **Semantic warning — the inputs are not literal "queue depth".** Both
/// `load_blocks` and `queued_blocks` arrive from router-state snapshots,
/// sourced by `prefill_router/execution.rs` via:
///   - `load_blocks` ← `KvRouter::active_blocks_for(worker)` (delegates to
///     `ActiveSequencesMultiWorker::active_blocks`) — total projected KV
///     block load across all router-dispatched requests on the worker
///     (admitted + queued-at-worker), not just "admitted-and-scheduled."
///   - `queued_blocks` ← `KvRouter::pending_prefill_tokens_for(worker) /
///     block_size` (delegates to `ActiveSequencesMultiWorker::active_tokens`)
///     — decay-adjusted pending-prefill token load: includes the currently-
///     running prefill (linearly decaying to 0 over its expected duration)
///     plus all queued-at-worker prefill work (full weight). Divided by
///     block_size to get block-equivalent units.
///
/// Both signals are snapshot values (pre-request) — they don't include the
/// current request's projected contribution. For the `roomy()` decision this
/// is what we want ("is the worker busy *now*?"), and the difference vs the
/// projected view is small (one request's contribution is ~1-20 blocks
/// against typical thresholds of 500+).
///
/// In practice the two signals together capture "worker is busy" reasonably:
///   - `(max - load) / max ≥ ratio` → "enough free KV memory"
///   - `queued ≤ threshold`         → "small pending prefill compute backlog"
///
/// TODO:
/// before pushing on real-engine E2E, revisit with mocker data already in hand:
///   - Whether the snapshot semantic (this function's current shape) gives
///     materially different decisions vs a request-projected view at the
///     workloads of interest.
///   - Per-side asymmetry: AGG cares more about decode-side memory pressure;
///     DISAGG cares more about prefill-side queue depth. Today both sides
///     use the same symmetric `roomy()` form.
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
        // Skipped entirely in `RegressionOnly` mode so every request that
        // reaches the slow path RPCs cost_eval.
        if self.enable_fast_path {
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
            if p_roomy || eff_isl_tokens > self.disagg_above_tokens {
                return false;
            }

            // AGG when either the decode worker has headroom or the prompt
            // is small enough that bypass is cheap.
            let d_roomy = worker_is_roomy(
                input.decode_chosen_load_blocks,
                input.decode_chosen_max_blocks,
                input.decode_chosen_queued_blocks,
                self.roomy_available_ratio_threshold,
                self.roomy_queued_blocks_threshold,
            );
            if d_roomy || eff_isl_tokens <= self.bypass_below_tokens {
                return true;
            }
        }

        // Slow path — consult the regression-backed evaluator. Conservative
        // DISAGG fallback when:
        //  - the slow path is disabled (`ThresholdOnly` diagnostic mode):
        //    fast path didn't conclude, so commit to DISAGG.
        //  - the DISAGG-side credit is missing (skip the RPC entirely),
        //  - the evaluator hasn't been bound yet (pre-activate, or
        //    Regression selected without a sidecar), or
        //  - either regression is cold.
        if !self.enable_slow_path {
            return false;
        }
        let Some(request) = self.cost_eval_request(input) else {
            return false;
        };
        let Some(evaluator) = self.evaluator.get() else {
            return false;
        };
        let resp = evaluator.evaluate(request).await;
        if !(resp.agg_warm && resp.disagg_warm) {
            return false;
        }
        // Prefer the total-cost comparison (TTFT + ITL * avg_decode_length)
        // when the sidecar surfaced it on both sides; fall back to TTFT-only
        // when total_cost is missing (cold ITL estimate or no avg_decode_length
        // yet). Either way, both sides must be Some — otherwise conservative
        // DISAGG.
        if let (Some(agg), Some(disagg)) = (resp.agg_total_cost_ms, resp.disagg_total_cost_ms) {
            return agg < disagg;
        }
        match (resp.agg_ttft_ms, resp.disagg_ttft_ms) {
            (Some(agg_ms), Some(disagg_ms)) => agg_ms < disagg_ms,
            _ => false,
        }
    }

    async fn evaluate_costs(&self, input: ConditionalPrefillDecisionInput) -> Option<(f64, f64)> {
        let request = self.cost_eval_request(input)?;
        let evaluator = self.evaluator.get()?;
        let resp = evaluator.evaluate(request).await;
        if !(resp.agg_warm && resp.disagg_warm) {
            return None;
        }
        // Same priority order as `should_bypass_remote_prefill`: total_cost
        // when available, fall back to TTFT-only.
        if let (Some(agg), Some(disagg)) = (resp.agg_total_cost_ms, resp.disagg_total_cost_ms) {
            return Some((agg, disagg));
        }
        match (resp.agg_ttft_ms, resp.disagg_ttft_ms) {
            (Some(agg_ms), Some(disagg_ms)) => Some((agg_ms, disagg_ms)),
            _ => None,
        }
    }

    fn try_set_cost_evaluator(&self, evaluator: Arc<dyn CostEvaluator>) {
        let _ = self.evaluator.set(evaluator);
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
            decode_chosen_max_blocks: None,
            prefill_chosen_max_blocks: None,
            decode_chosen_queued_blocks: None,
            prefill_chosen_queued_blocks: None,
            decode_chosen_worker_id: 0,
            decode_chosen_dp_rank: 0,
            prefill_chosen_worker_id: None,
            prefill_chosen_dp_rank: None,
        }
    }

    #[tokio::test]
    async fn token_cap_policy_is_disabled_by_default() {
        let policy = TokenCapConditionalPrefillPolicy::default();

        assert!(!policy.is_enabled());
        assert_eq!(
            policy.bypass_below_tokens,
            DEFAULT_CONDITIONAL_PREFILL_BYPASS_BELOW_TOKENS
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
            bypass_below_tokens: 160,
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
            bypass_below_tokens: 160,
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
            ConditionalPrefillPolicyKind::ThresholdRegression,
            ConditionalPrefillPolicyKind::ThresholdOnly,
            ConditionalPrefillPolicyKind::RegressionOnly,
            ConditionalPrefillPolicyKind::AlwaysBypass,
            ConditionalPrefillPolicyKind::RandomBypass,
        ] {
            assert_eq!(
                ConditionalPrefillPolicyKind::from_str(kind.as_str()),
                Some(kind)
            );
        }
        assert_eq!(ConditionalPrefillPolicyKind::from_str("nonsense"), None);
    }

    #[test]
    fn uses_slow_path_matches_variant_intent() {
        assert!(!ConditionalPrefillPolicyKind::TokenCap.uses_slow_path());
        assert!(!ConditionalPrefillPolicyKind::AlwaysBypass.uses_slow_path());
        assert!(!ConditionalPrefillPolicyKind::RandomBypass.uses_slow_path());
        assert!(!ConditionalPrefillPolicyKind::ThresholdOnly.uses_slow_path());
        assert!(ConditionalPrefillPolicyKind::ThresholdRegression.uses_slow_path());
        assert!(ConditionalPrefillPolicyKind::RegressionOnly.uses_slow_path());
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
            decode_chosen_max_blocks: Some(1000),
            prefill_chosen_max_blocks: Some(1000),
            decode_chosen_queued_blocks: Some(10),
            prefill_chosen_queued_blocks: Some(10),
            decode_chosen_worker_id: 42,
            decode_chosen_dp_rank: 0,
            prefill_chosen_worker_id: Some(7),
            prefill_chosen_dp_rank: Some(0),
        }
    }

    fn make_regression_policy(response: CostEvalResponse) -> RegressionConditionalPrefillPolicy {
        let policy = RegressionConditionalPrefillPolicy {
            enabled: true,
            enable_fast_path: true,
            enable_slow_path: true,
            disagg_above_tokens: 16_384,
            bypass_below_tokens: 0,
            roomy_available_ratio_threshold: 0.5,
            roomy_queued_blocks_threshold: 0,
            evaluator: OnceLock::new(),
        };
        policy.try_set_cost_evaluator(Arc::new(StubCostEvaluator { response }));
        policy
    }

    /// Same as `make_regression_policy` but leaves the evaluator unbound,
    /// to verify slow-path conservative-DISAGG fallback.
    fn make_regression_policy_without_evaluator() -> RegressionConditionalPrefillPolicy {
        RegressionConditionalPrefillPolicy {
            enabled: true,
            enable_fast_path: true,
            enable_slow_path: true,
            disagg_above_tokens: 16_384,
            bypass_below_tokens: 0,
            roomy_available_ratio_threshold: 0.5,
            roomy_queued_blocks_threshold: 0,
            evaluator: OnceLock::new(),
        }
    }

    #[tokio::test]
    async fn regression_disabled_never_bypasses() {
        let mut policy = make_regression_policy(CostEvalResponse {
            agg_ttft_ms: Some(1.0),
            disagg_ttft_ms: Some(100.0),
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
    async fn regression_slow_path_unbound_evaluator_falls_back_to_disagg() {
        // Policy constructed without an evaluator (the pre-activate state for
        // the live router). Slow path must take conservative DISAGG without
        // panicking on the missing evaluator.
        let policy = make_regression_policy_without_evaluator();
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
    }

    #[tokio::test]
    async fn regression_try_set_cost_evaluator_late_binds() {
        // Construct with no evaluator → slow-path returns DISAGG. After
        // binding an AGG-favorable evaluator, slow path now picks AGG.
        let policy = make_regression_policy_without_evaluator();
        assert!(
            !policy
                .should_bypass_remote_prefill(regression_input_slow_path())
                .await
        );
        policy.try_set_cost_evaluator(Arc::new(StubCostEvaluator {
            response: CostEvalResponse {
                agg_ttft_ms: Some(10.0),
                disagg_ttft_ms: Some(100.0),
                agg_total_cost_ms: None,
                disagg_total_cost_ms: None,
                agg_warm: true,
                disagg_warm: true,
            },
        }));
        assert!(
            policy
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
            agg_total_cost_ms: None,
            disagg_total_cost_ms: None,
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
