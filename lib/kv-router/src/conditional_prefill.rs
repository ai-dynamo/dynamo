// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-prefill bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait operates over a struct of
//! summary signals (`ConditionalPrefillDecisionInput`). v1 ships a single
//! production policy (`IslBoundingPolicy`); the `ConditionalPrefillPolicyKind`
//! enum and `ConditionalPrefillDecisionInput` struct are both designed to
//! grow — future policies (queue-aware, regression-backed) plug in here
//! without breaking the trait or the router's call site.

use async_trait::async_trait;

use crate::config::{ConditionalPrefillPolicyKind, KvRouterConfig};

/// Default effective-ISL absolute threshold (tokens). A request bypasses to
/// AGG only if its net-new prefill stays under this cap.
pub const DEFAULT_CONDITIONAL_PREFILL_EFF_ISL_THRESHOLD: usize = 2048;

/// Default effective-ISL ratio threshold. A request bypasses to AGG only if
/// `eff_isl / prompt_tokens` stays under this fraction (i.e. the device
/// prefix cache covers enough of the prompt).
pub const DEFAULT_CONDITIONAL_PREFILL_EFF_ISL_RATIO_THRESHOLD: f64 = 0.7;

/// Inputs passed to a `ConditionalPrefillPolicy` when deciding whether to
/// bypass remote prefill.
///
/// **Extensibility:** `#[non_exhaustive]` so future fields (load signals,
/// pending-prefill counts, candidate worker ids, cost-eval inputs) can be
/// added without breaking external pattern matches. Callers should construct
/// via the inherent `new` constructor or struct-update syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConditionalPrefillDecisionInput {
    /// Total prompt token count.
    pub prompt_tokens: usize,

    /// KV cache block size, in tokens.
    pub block_size: usize,

    /// Device-prefix overlap on the cache-hot decode worker, in blocks.
    pub decode_chosen_overlap_blocks: u32,
}

impl ConditionalPrefillDecisionInput {
    pub fn new(prompt_tokens: usize, block_size: usize, decode_chosen_overlap_blocks: u32) -> Self {
        Self {
            prompt_tokens,
            block_size,
            decode_chosen_overlap_blocks,
        }
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
/// `ConditionalPrefillDecisionInput` (it is `#[non_exhaustive]`). New trait
/// methods that future policies need should have default impls so older
/// policies don't have to opt in.
#[async_trait]
pub trait ConditionalPrefillPolicy: Send + Sync {
    fn is_enabled(&self) -> bool;

    /// Decide whether the request should skip remote prefill. Async so a
    /// future policy's slow path can consult an external service. v1's
    /// `IslBoundingPolicy` is fully synchronous — the `.await` is a no-op.
    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool;
}

/// Build the configured conditional-prefill policy. Returns a disabled
/// `IslBoundingPolicy` (always returns `false`) when `config` is `None` or
/// `conditional_prefill_enabled` is false.
pub fn make_conditional_prefill_policy(
    config: Option<&KvRouterConfig>,
) -> Box<dyn ConditionalPrefillPolicy> {
    let Some(config) = config else {
        return Box::new(IslBoundingPolicy::disabled());
    };
    match config.conditional_prefill_policy {
        ConditionalPrefillPolicyKind::IslBounding => {
            Box::new(IslBoundingPolicy::from_config(config))
        }
    }
}

/// v1 conditional-prefill policy. Bypasses to AGG when the request is both
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
            enabled: config.conditional_prefill_enabled,
            eff_isl_threshold: config.conditional_prefill_eff_isl_threshold,
            eff_isl_ratio_threshold: config.conditional_prefill_eff_isl_ratio_threshold,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            eff_isl_threshold: DEFAULT_CONDITIONAL_PREFILL_EFF_ISL_THRESHOLD,
            eff_isl_ratio_threshold: DEFAULT_CONDITIONAL_PREFILL_EFF_ISL_RATIO_THRESHOLD,
        }
    }
}

impl Default for IslBoundingPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[async_trait]
impl ConditionalPrefillPolicy for IslBoundingPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
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

// ===== Test-only policies ==================================================

/// Test-only: bypass with a configurable probability. Not exposed via the
/// `ConditionalPrefillPolicyKind` enum and not selectable from the CLI —
/// constructed directly in unit tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct RandomBypassConditionalPrefillPolicy {
    enabled: bool,
    bypass_probability: f64,
}

#[cfg(test)]
impl RandomBypassConditionalPrefillPolicy {
    pub fn new(enabled: bool, bypass_probability: f64) -> Self {
        Self {
            enabled,
            bypass_probability: bypass_probability.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
#[async_trait]
impl ConditionalPrefillPolicy for RandomBypassConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn should_bypass_remote_prefill(&self, _input: ConditionalPrefillDecisionInput) -> bool {
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
    ) -> ConditionalPrefillDecisionInput {
        ConditionalPrefillDecisionInput::new(prompt_tokens, block_size, overlap_blocks)
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
        let policy = RandomBypassConditionalPrefillPolicy::new(false, 1.0);
        assert!(!policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
    }

    #[tokio::test]
    async fn random_bypass_zero_probability_never_bypasses() {
        let policy = RandomBypassConditionalPrefillPolicy::new(true, 0.0);
        for _ in 0..50 {
            assert!(!policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
        }
    }

    #[tokio::test]
    async fn random_bypass_one_probability_always_bypasses() {
        let policy = RandomBypassConditionalPrefillPolicy::new(true, 1.0);
        for _ in 0..50 {
            assert!(policy.should_bypass_remote_prefill(input(100, 0, 64)).await);
        }
    }
}
