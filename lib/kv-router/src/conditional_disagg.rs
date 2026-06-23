// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disagg bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait operates over a struct of
//! summary signals (`ConditionalDisaggDecisionInput`). v1 ships a single
//! production policy (`IslBoundingPolicy`); the `ConditionalDisaggPolicyKind`
//! enum and `ConditionalDisaggDecisionInput` struct are both designed to
//! grow — future policies (queue-aware, regression-backed) plug in here
//! without breaking the trait or the router's call site.

use async_trait::async_trait;

use crate::config::{ConditionalDisaggPolicyKind, KvRouterConfig};

/// Default effective-ISL absolute threshold (tokens). A request bypasses to
/// AGG only if its net-new prefill stays under this cap.
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD: usize = 2048;

/// Default effective-ISL ratio threshold. A request bypasses to AGG only if
/// `eff_isl / prompt_tokens` stays under this fraction (i.e. the device
/// prefix cache covers enough of the prompt).
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD: f64 = 0.7;

/// Inputs passed to a `ConditionalDisaggPolicy` when deciding whether to
/// bypass remote prefill.
///
/// **Extensibility:** `#[non_exhaustive]` so future fields (load signals,
/// pending-prefill counts, candidate worker ids, cost-eval inputs) can be
/// added without breaking external pattern matches. Callers should construct
/// via the inherent `new` constructor or struct-update syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConditionalDisaggDecisionInput {
    /// Total prompt token count.
    pub prompt_tokens: usize,

    /// KV cache block size, in tokens.
    pub block_size: usize,

    /// Device-prefix overlap on the cache-hot decode worker, in blocks.
    pub decode_chosen_overlap_blocks: u32,

    /// Whether the prefill worker the router would pick for this request is
    /// over the existing prefill-busy line (i.e.
    /// `active_tokens > router_queue_threshold * max_num_batched_tokens`).
    /// `None` when the signal isn't available (queueing disabled, no prefill
    /// workers, or the peek-selection failed); in that case the load-aware
    /// policies treat the worker as calm.
    pub prefill_chosen_worker_busy: Option<bool>,

    /// Whether the decode worker the router would pick for this request is over
    /// the decode-busy line (i.e.
    /// `active_decode_blocks > conditional_disagg_decode_busy_threshold * total_kv_blocks`).
    /// `None` when the decode gate is disabled, or the signal isn't available
    /// (unknown worker / no `total_kv_blocks` reported). The v2 decode-side
    /// circuit breaker is composed at the `PrefillRouter` call site (an AND
    /// veto over the policy's verdict), not inside a policy — this field
    /// carries the signal for logging and future use.
    pub decode_chosen_worker_busy: Option<bool>,
}

impl ConditionalDisaggDecisionInput {
    pub fn new(prompt_tokens: usize, block_size: usize, decode_chosen_overlap_blocks: u32) -> Self {
        Self {
            prompt_tokens,
            block_size,
            decode_chosen_overlap_blocks,
            prefill_chosen_worker_busy: None,
            decode_chosen_worker_busy: None,
        }
    }

    /// Set the chosen prefill-worker busy signal. Chained from `new` at call
    /// sites that have the signal (e.g. `PrefillRouter` after a peek).
    pub fn with_prefill_chosen_worker_busy(mut self, busy: Option<bool>) -> Self {
        self.prefill_chosen_worker_busy = busy;
        self
    }

    /// Set the chosen decode-worker busy signal. Chained from `new` at the
    /// `PrefillRouter` call site after the decode-busy peek.
    pub fn with_decode_chosen_worker_busy(mut self, busy: Option<bool>) -> Self {
        self.decode_chosen_worker_busy = busy;
        self
    }

    /// Effective net-new prefill in tokens after the decode-side device
    /// cache hit is subtracted.
    pub fn net_new_tokens(self) -> usize {
        let overlap_tokens =
            (self.decode_chosen_overlap_blocks as usize).saturating_mul(self.block_size);
        self.prompt_tokens.saturating_sub(overlap_tokens)
    }
}

/// Decision policy invoked by `PrefillRouter` before routing a request.
///
/// **Extensibility:** new policies that need additional inputs should extend
/// `ConditionalDisaggDecisionInput` (it is `#[non_exhaustive]`). New trait
/// methods that future policies need should have default impls so older
/// policies don't have to opt in.
#[async_trait]
pub trait ConditionalDisaggPolicy: Send + Sync {
    fn is_enabled(&self) -> bool;

    /// Decide whether the request should skip remote prefill. Async so a
    /// future policy's slow path can consult an external service. v1's
    /// `IslBoundingPolicy` is fully synchronous — the `.await` is a no-op.
    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool;

    /// True iff this policy consumes
    /// [`ConditionalDisaggDecisionInput::prefill_chosen_worker_busy`]. The
    /// `PrefillRouter` uses this to skip the extra prefill-worker peek for
    /// policies that don't need the signal (back-compat fast path for
    /// `IslBounding`).
    fn needs_prefill_worker_busy(&self) -> bool {
        false
    }
}

/// Build the configured conditional-disagg policy. Returns a disabled
/// `IslBoundingPolicy` (always returns `false`) when `config` is `None` or
/// `conditional_disagg_enabled` is false.
pub fn make_conditional_disagg_policy(
    config: Option<&KvRouterConfig>,
) -> Box<dyn ConditionalDisaggPolicy> {
    let Some(config) = config else {
        return Box::new(IslBoundingPolicy::disabled());
    };
    match config.conditional_disagg_policy {
        ConditionalDisaggPolicyKind::IslBounding => {
            Box::new(IslBoundingPolicy::from_config(config))
        }
        ConditionalDisaggPolicyKind::PrefillLoad => {
            Box::new(PrefillLoadPolicy::from_config(config))
        }
        ConditionalDisaggPolicyKind::IslOrLoad => Box::new(IslOrLoadPolicy::from_config(config)),
    }
}

/// True iff the policy needs the `prefill_chosen_worker_busy` signal
/// populated on `ConditionalDisaggDecisionInput`. Used by `PrefillRouter`
/// to skip the prefill-worker peek for policies that don't consume the
/// signal (back-compat fast path for `IslBounding` / `Disabled`).
pub fn policy_needs_prefill_worker_busy(config: Option<&KvRouterConfig>) -> bool {
    let Some(config) = config else { return false };
    if !config.conditional_disagg_enabled {
        return false;
    }
    matches!(
        config.conditional_disagg_policy,
        ConditionalDisaggPolicyKind::PrefillLoad | ConditionalDisaggPolicyKind::IslOrLoad,
    )
}

/// v1 conditional-disagg policy. Bypasses to AGG when the request is both
/// small in absolute net-new prefill AND mostly cached on the decode worker.
///
/// Predicate:
/// ```text
/// eff_isl = prompt_tokens - decode_chosen_overlap_blocks * block_size
/// ratio   = eff_isl / max(prompt_tokens, 1)
///
/// bypass = enabled
///       AND eff_isl < eff_isl_threshold
///       AND ratio   < eff_isl_ratio_threshold
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IslBoundingPolicy {
    enabled: bool,
    eff_isl_threshold: usize,
    eff_isl_ratio_threshold: f64,
}

impl IslBoundingPolicy {
    pub fn new(enabled: bool, eff_isl_threshold: usize, eff_isl_ratio_threshold: f64) -> Self {
        Self {
            enabled,
            eff_isl_threshold,
            eff_isl_ratio_threshold,
        }
    }

    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            enabled: config.conditional_disagg_enabled,
            eff_isl_threshold: config.conditional_disagg_eff_isl_threshold,
            eff_isl_ratio_threshold: config.conditional_disagg_eff_isl_ratio_threshold,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            eff_isl_threshold: DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD,
            eff_isl_ratio_threshold: DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD,
        }
    }
}

impl Default for IslBoundingPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[async_trait]
impl ConditionalDisaggPolicy for IslBoundingPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }
        let eff_isl = input.net_new_tokens();
        if eff_isl >= self.eff_isl_threshold {
            return false;
        }
        let denom = input.prompt_tokens.max(1) as f64;
        let ratio = eff_isl as f64 / denom;
        ratio < self.eff_isl_ratio_threshold
    }
}

/// v1.5 conditional-disagg policy. Bypasses to AGG when the prefill worker
/// the router would pick for this request is already over the existing
/// prefill-busy line — same predicate the scheduler uses to decide whether
/// to park a new request in the pending queue:
///
/// ```text
/// busy(worker) = active_tokens > router_queue_threshold * max_num_batched_tokens
/// ```
///
/// Semantic interpretation: if the chosen prefill worker would queue this
/// request anyway, just bypass to decode instead — avoiding queue overhead
/// and exploiting decode-side cache locality. No new operator knob; the
/// trigger is the existing `router_queue_threshold`.
///
/// The signal is `input.prefill_chosen_worker_busy`, populated by
/// `PrefillRouter` after peek-selecting the best prefill worker for this
/// request. When the signal is `None` (queueing disabled, no prefill
/// workers, or peek failure), the policy treats the worker as calm and does
/// not bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillLoadPolicy {
    enabled: bool,
}

impl PrefillLoadPolicy {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            enabled: config.conditional_disagg_enabled,
        }
    }

    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for PrefillLoadPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[async_trait]
impl ConditionalDisaggPolicy for PrefillLoadPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }
        // None ⇒ signal unavailable ⇒ treat as calm (don't bypass).
        input.prefill_chosen_worker_busy.unwrap_or(false)
    }

    fn needs_prefill_worker_busy(&self) -> bool {
        self.enabled
    }
}

/// v1.5 composition policy: bypass when EITHER `IslBoundingPolicy` says so
/// OR `PrefillLoadPolicy` says so. Pure glue — the two inner policies are
/// independently testable, and the OR captures the "each gate has an
/// independent rationale for keeping local" framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IslOrLoadPolicy {
    isl: IslBoundingPolicy,
    load: PrefillLoadPolicy,
}

impl IslOrLoadPolicy {
    pub fn new(isl: IslBoundingPolicy, load: PrefillLoadPolicy) -> Self {
        Self { isl, load }
    }

    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            isl: IslBoundingPolicy::from_config(config),
            load: PrefillLoadPolicy::from_config(config),
        }
    }

    pub fn disabled() -> Self {
        Self {
            isl: IslBoundingPolicy::disabled(),
            load: PrefillLoadPolicy::disabled(),
        }
    }
}

impl Default for IslOrLoadPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[async_trait]
impl ConditionalDisaggPolicy for IslOrLoadPolicy {
    fn is_enabled(&self) -> bool {
        self.isl.is_enabled() || self.load.is_enabled()
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool {
        self.isl.should_bypass_remote_prefill(input).await
            || self.load.should_bypass_remote_prefill(input).await
    }

    fn needs_prefill_worker_busy(&self) -> bool {
        self.isl.needs_prefill_worker_busy() || self.load.needs_prefill_worker_busy()
    }
}

// ===== Test-only policies ==================================================

/// Test-only: bypass with a configurable probability. Not exposed via the
/// `ConditionalDisaggPolicyKind` enum and not selectable from the CLI —
/// constructed directly in unit tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct RandomBypassConditionalDisaggPolicy {
    enabled: bool,
    bypass_probability: f64,
}

#[cfg(test)]
impl RandomBypassConditionalDisaggPolicy {
    pub fn new(enabled: bool, bypass_probability: f64) -> Self {
        Self {
            enabled,
            bypass_probability: bypass_probability.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
#[async_trait]
impl ConditionalDisaggPolicy for RandomBypassConditionalDisaggPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, _input: ConditionalDisaggDecisionInput) -> bool {
        if !self.enabled {
            return false;
        }
        rand::random::<f64>() < self.bypass_probability
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(
        prompt_tokens: usize,
        overlap_blocks: u32,
        block_size: usize,
    ) -> ConditionalDisaggDecisionInput {
        ConditionalDisaggDecisionInput::new(prompt_tokens, block_size, overlap_blocks)
    }

    #[tokio::test]
    async fn disabled_never_bypasses() {
        let policy = IslBoundingPolicy::new(false, 2048, 0.7);
        assert!(!policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
    }

    #[tokio::test]
    async fn small_and_mostly_cached_bypasses() {
        // 1000 prompt, 14 blocks * 64 = 896 cached → eff_isl = 104, ratio = 0.104 < 0.7
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(
            policy
                .should_bypass_remote_prefill(input(1000, 14, 64))
                .await
        );
    }

    #[tokio::test]
    async fn large_eff_isl_does_not_bypass_even_if_ratio_low() {
        // 100k prompt, 1000 blocks * 64 = 64k cached → eff_isl = 36k > 2048
        let policy = IslBoundingPolicy::new(true, 2048, 0.99);
        assert!(
            !policy
                .should_bypass_remote_prefill(input(100_000, 1000, 64))
                .await
        );
    }

    #[tokio::test]
    async fn small_eff_isl_but_ratio_at_or_above_threshold_does_not_bypass() {
        // 200 prompt, 0 overlap → eff_isl = 200, ratio = 1.0 ≥ 0.7
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(!policy.should_bypass_remote_prefill(input(200, 0, 64)).await);
    }

    #[tokio::test]
    async fn boundary_eff_isl_equals_threshold_does_not_bypass() {
        // eff_isl = 2048 = threshold → strict < fails
        let policy = IslBoundingPolicy::new(true, 2048, 0.99);
        assert!(
            !policy
                .should_bypass_remote_prefill(input(2048, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn zero_prompt_tokens_does_not_panic() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        // eff_isl = 0, denom = max(0,1) = 1, ratio = 0 < 0.7 → bypass
        assert!(policy.should_bypass_remote_prefill(input(0, 0, 64)).await);
    }

    #[tokio::test]
    async fn overlap_exceeding_prompt_clamps_to_zero_eff_isl() {
        // overlap_tokens = 10 * 64 = 640 > 500 prompt
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        // eff_isl saturates to 0, ratio = 0 → bypass
        assert!(
            policy
                .should_bypass_remote_prefill(input(500, 10, 64))
                .await
        );
    }

    #[tokio::test]
    async fn random_bypass_when_disabled_never_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(false, 1.0);
        assert!(!policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
    }

    #[tokio::test]
    async fn random_bypass_zero_probability_never_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(true, 0.0);
        for _ in 0..50 {
            assert!(!policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
        }
    }

    #[tokio::test]
    async fn random_bypass_one_probability_always_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(true, 1.0);
        for _ in 0..50 {
            assert!(policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
        }
    }

    // ===== PrefillLoadPolicy tests ==========================================

    fn input_with_busy(
        prompt_tokens: usize,
        overlap_blocks: u32,
        block_size: usize,
        busy: Option<bool>,
    ) -> ConditionalDisaggDecisionInput {
        ConditionalDisaggDecisionInput::new(prompt_tokens, block_size, overlap_blocks)
            .with_prefill_chosen_worker_busy(busy)
    }

    #[tokio::test]
    async fn prefill_load_disabled_never_bypasses() {
        let policy = PrefillLoadPolicy::new(false);
        // Even when worker is reported busy, disabled policy never bypasses.
        assert!(
            !policy
                .should_bypass_remote_prefill(input_with_busy(1000, 0, 64, Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn prefill_load_signal_none_does_not_bypass() {
        let policy = PrefillLoadPolicy::new(true);
        // No signal ⇒ treat as calm.
        assert!(
            !policy
                .should_bypass_remote_prefill(input_with_busy(1000, 0, 64, None))
                .await
        );
    }

    #[tokio::test]
    async fn prefill_load_worker_calm_does_not_bypass() {
        let policy = PrefillLoadPolicy::new(true);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_with_busy(1000, 0, 64, Some(false)))
                .await
        );
    }

    #[tokio::test]
    async fn prefill_load_worker_busy_bypasses() {
        let policy = PrefillLoadPolicy::new(true);
        assert!(
            policy
                .should_bypass_remote_prefill(input_with_busy(1000, 0, 64, Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn prefill_load_independent_of_isl_fields() {
        // PrefillLoadPolicy ignores prompt_tokens / overlap / block_size — it
        // only consults `prefill_chosen_worker_busy`. Verify a "large" request
        // bypasses purely on the busy signal.
        let policy = PrefillLoadPolicy::new(true);
        assert!(
            policy
                .should_bypass_remote_prefill(input_with_busy(100_000, 0, 64, Some(true)))
                .await
        );
        assert!(
            !policy
                .should_bypass_remote_prefill(input_with_busy(100_000, 0, 64, Some(false)))
                .await
        );
    }

    // ===== IslOrLoadPolicy tests ============================================
    //
    // Four-quadrant truth table: (isl_says_bypass, worker_busy).
    // Helper: "small + cached" trips IslBounding; "large" doesn't.

    fn small_input(busy: Option<bool>) -> ConditionalDisaggDecisionInput {
        // 1000 prompt, 14 blocks * 64 = 896 cached → eff_isl = 104 < 2048, ratio = 0.104 < 0.7
        input_with_busy(1000, 14, 64, busy)
    }

    fn large_input(busy: Option<bool>) -> ConditionalDisaggDecisionInput {
        // 100k prompt, 0 overlap → eff_isl = 100k > 2048
        input_with_busy(100_000, 0, 64, busy)
    }

    fn enabled_or_policy() -> IslOrLoadPolicy {
        IslOrLoadPolicy::new(
            IslBoundingPolicy::new(true, 2048, 0.7),
            PrefillLoadPolicy::new(true),
        )
    }

    #[tokio::test]
    async fn isl_or_load_small_and_calm_bypasses_via_isl() {
        assert!(
            enabled_or_policy()
                .should_bypass_remote_prefill(small_input(Some(false)))
                .await
        );
    }

    #[tokio::test]
    async fn isl_or_load_small_and_busy_bypasses() {
        // Both gates fire — still one bypass.
        assert!(
            enabled_or_policy()
                .should_bypass_remote_prefill(small_input(Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn isl_or_load_large_and_calm_does_not_bypass() {
        assert!(
            !enabled_or_policy()
                .should_bypass_remote_prefill(large_input(Some(false)))
                .await
        );
    }

    #[tokio::test]
    async fn isl_or_load_large_and_busy_bypasses_via_load() {
        assert!(
            enabled_or_policy()
                .should_bypass_remote_prefill(large_input(Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn isl_or_load_disabled_never_bypasses() {
        // Both inner disabled — even small+busy must not bypass.
        let policy = IslOrLoadPolicy::disabled();
        assert!(
            !policy
                .should_bypass_remote_prefill(small_input(Some(true)))
                .await
        );
        assert!(
            !policy
                .should_bypass_remote_prefill(large_input(Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn isl_or_load_signal_none_falls_back_to_isl_only() {
        // No busy signal ⇒ Load gate is a no-op ⇒ behavior matches IslBounding alone.
        let policy = enabled_or_policy();
        assert!(policy.should_bypass_remote_prefill(small_input(None)).await);
        assert!(!policy.should_bypass_remote_prefill(large_input(None)).await);
    }

    #[tokio::test]
    async fn decision_input_new_defaults_busy_to_none() {
        // Back-compat: existing call sites that build via `new` should default
        // the new fields to `None`.
        let input = ConditionalDisaggDecisionInput::new(1000, 64, 0);
        assert_eq!(input.prefill_chosen_worker_busy, None);
        assert_eq!(input.decode_chosen_worker_busy, None);
    }

    #[tokio::test]
    async fn decision_input_with_decode_chosen_worker_busy_round_trips() {
        let input = ConditionalDisaggDecisionInput::new(1000, 64, 0)
            .with_decode_chosen_worker_busy(Some(true));
        assert_eq!(input.decode_chosen_worker_busy, Some(true));
        // Independent of the prefill-side field.
        assert_eq!(input.prefill_chosen_worker_busy, None);
    }
}
