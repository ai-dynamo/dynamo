// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disagg bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait operates over summary
//! signals so future policies can plug in without changing the router call site.

use async_trait::async_trait;

use crate::config::{ConditionalDisaggPolicyKind, KvRouterConfig};

/// Default effective-ISL absolute threshold (tokens). A request bypasses to
/// AGG only if its net-new prefill stays under this cap.
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD: usize = 2048;

/// Default effective-ISL ratio threshold. A request bypasses to AGG only if
/// `eff_isl / prompt_tokens` stays under this fraction.
pub const DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD: f64 = 0.7;

/// Inputs passed to a `ConditionalDisaggPolicy` when deciding whether to
/// bypass remote prefill.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConditionalDisaggDecisionInput {
    /// Total prompt token count.
    pub prompt_tokens: usize,

    /// Effective cache credit on the chosen decode worker, in weighted tokens.
    pub decode_chosen_cached_tokens: usize,

    /// Whether the prefill worker the router would pick for this request is
    /// over the prefill-busy line. `None` means the signal is unavailable.
    pub prefill_chosen_worker_busy: Option<bool>,

    /// Whether the decode worker the router would pick for this request is
    /// over the decode-busy line. `None` means the gate is disabled or the
    /// signal is unavailable.
    pub decode_chosen_worker_busy: Option<bool>,
}

impl ConditionalDisaggDecisionInput {
    pub fn new(prompt_tokens: usize, decode_chosen_cached_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            decode_chosen_cached_tokens,
            prefill_chosen_worker_busy: None,
            decode_chosen_worker_busy: None,
        }
    }

    pub fn with_prefill_chosen_worker_busy(mut self, busy: Option<bool>) -> Self {
        self.prefill_chosen_worker_busy = busy;
        self
    }

    pub fn with_decode_chosen_worker_busy(mut self, busy: Option<bool>) -> Self {
        self.decode_chosen_worker_busy = busy;
        self
    }

    /// Effective net-new prefill in tokens after the decode-side weighted
    /// cache hit is subtracted.
    pub fn net_new_tokens(self) -> usize {
        self.prompt_tokens
            .saturating_sub(self.decode_chosen_cached_tokens.min(self.prompt_tokens))
    }
}

#[async_trait]
pub trait ConditionalDisaggPolicy: Send + Sync {
    fn is_enabled(&self) -> bool;

    /// Decide whether the request should skip remote prefill.
    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool;

    /// True iff this policy consumes
    /// [`ConditionalDisaggDecisionInput::prefill_chosen_worker_busy`].
    fn needs_prefill_worker_busy(&self) -> bool {
        false
    }
}

/// Build the configured conditional-disagg policy. Returns a disabled policy
/// when conditional disagg is not enabled.
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

pub fn policy_needs_prefill_worker_busy(config: Option<&KvRouterConfig>) -> bool {
    make_conditional_disagg_policy(config).needs_prefill_worker_busy()
}

/// v1 conditional-disagg policy. Bypasses to AGG when the request is both
/// small in absolute net-new prefill and mostly cached on the decode worker.
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
/// the router would pick for this request is already over the prefill-busy line.
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
        input.prefill_chosen_worker_busy.unwrap_or(false)
    }

    fn needs_prefill_worker_busy(&self) -> bool {
        self.enabled
    }
}

/// v1.5 composition policy: bypass when either the ISL policy or the load
/// policy wants bypass.
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

    fn input(prompt_tokens: usize, cached_tokens: usize) -> ConditionalDisaggDecisionInput {
        ConditionalDisaggDecisionInput::new(prompt_tokens, cached_tokens)
    }

    fn input_from_blocks(
        prompt_tokens: usize,
        overlap_blocks: u32,
        block_size: usize,
    ) -> ConditionalDisaggDecisionInput {
        input(
            prompt_tokens,
            (overlap_blocks as usize).saturating_mul(block_size),
        )
    }

    #[tokio::test]
    async fn disabled_never_bypasses() {
        let policy = IslBoundingPolicy::new(false, 2048, 0.7);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_from_blocks(100, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn small_and_mostly_cached_bypasses() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(
            policy
                .should_bypass_remote_prefill(input_from_blocks(1000, 14, 64))
                .await
        );
    }

    #[tokio::test]
    async fn large_eff_isl_does_not_bypass_even_if_ratio_low() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.99);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_from_blocks(100_000, 1000, 64))
                .await
        );
    }

    #[tokio::test]
    async fn small_eff_isl_but_ratio_at_or_above_threshold_does_not_bypass() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_from_blocks(200, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn boundary_eff_isl_equals_threshold_does_not_bypass() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.99);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_from_blocks(2048, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn zero_prompt_tokens_does_not_panic() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(
            policy
                .should_bypass_remote_prefill(input_from_blocks(0, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn overlap_exceeding_prompt_clamps_to_zero_eff_isl() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.7);
        assert!(
            policy
                .should_bypass_remote_prefill(input_from_blocks(500, 10, 64))
                .await
        );
    }

    #[tokio::test]
    async fn weighted_cached_tokens_do_not_round_up_to_full_block() {
        let policy = IslBoundingPolicy::new(true, 2048, 0.99);
        assert!(!policy.should_bypass_remote_prefill(input(2100, 48)).await);
    }

    #[tokio::test]
    async fn random_bypass_when_disabled_never_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(false, 1.0);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_from_blocks(100, 0, 64))
                .await
        );
    }

    #[tokio::test]
    async fn random_bypass_zero_probability_never_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(true, 0.0);
        for _ in 0..50 {
            assert!(
                !policy
                    .should_bypass_remote_prefill(input_from_blocks(100, 0, 64))
                    .await
            );
        }
    }

    #[tokio::test]
    async fn random_bypass_one_probability_always_bypasses() {
        let policy = RandomBypassConditionalDisaggPolicy::new(true, 1.0);
        for _ in 0..50 {
            assert!(
                policy
                    .should_bypass_remote_prefill(input_from_blocks(100, 0, 64))
                    .await
            );
        }
    }

    fn input_with_busy(
        prompt_tokens: usize,
        overlap_blocks: u32,
        block_size: usize,
        busy: Option<bool>,
    ) -> ConditionalDisaggDecisionInput {
        input_from_blocks(prompt_tokens, overlap_blocks, block_size)
            .with_prefill_chosen_worker_busy(busy)
    }

    #[tokio::test]
    async fn prefill_load_disabled_never_bypasses() {
        let policy = PrefillLoadPolicy::new(false);
        assert!(
            !policy
                .should_bypass_remote_prefill(input_with_busy(1000, 0, 64, Some(true)))
                .await
        );
    }

    #[tokio::test]
    async fn prefill_load_signal_none_does_not_bypass() {
        let policy = PrefillLoadPolicy::new(true);
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

    fn small_input(busy: Option<bool>) -> ConditionalDisaggDecisionInput {
        input_with_busy(1000, 14, 64, busy)
    }

    fn large_input(busy: Option<bool>) -> ConditionalDisaggDecisionInput {
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
        let policy = enabled_or_policy();
        assert!(policy.should_bypass_remote_prefill(small_input(None)).await);
        assert!(!policy.should_bypass_remote_prefill(large_input(None)).await);
    }

    #[tokio::test]
    async fn decision_input_new_defaults_busy_to_none() {
        let input = ConditionalDisaggDecisionInput::new(1000, 0);
        assert_eq!(input.prefill_chosen_worker_busy, None);
        assert_eq!(input.decode_chosen_worker_busy, None);
    }

    #[tokio::test]
    async fn decision_input_with_decode_chosen_worker_busy_round_trips() {
        let input =
            ConditionalDisaggDecisionInput::new(1000, 0).with_decode_chosen_worker_busy(Some(true));
        assert_eq!(input.decode_chosen_worker_busy, Some(true));
        assert_eq!(input.prefill_chosen_worker_busy, None);
    }
}
