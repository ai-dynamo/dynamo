// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decide whether a request keeps generating on its prefill worker.
//!
//! In disaggregated serving a prefill worker stops after one token and hands the
//! request to a decode worker. When the decode pool has no room, that handoff has
//! nowhere to go. This policy decides when the prefill worker keeps generating
//! instead, which converts a request that would have thrashed the decode pool into
//! one that is served on capacity that is already idle.
//!
//! The policy is a pure function of measured load and the request's own budget, so
//! it is testable without a runtime and carries no model or engine knowledge.

use crate::scheduling::config::KvRouterConfig;

/// Why a request was not allowed to keep generating on its prefill worker.
///
/// Carried instead of a bare `false` so the caller can report it: an operator
/// asking "why did the feature never fire?" needs the reason, not the outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrefillContinueSkip {
    /// The feature is off.
    Disabled,
    /// The feature is on but no decode trigger is configured, so it can never
    /// fire. Distinct from `Disabled` so the reason an operator reads is true.
    NoTrigger,
    /// The decode pool can take the request, so the normal handoff is correct.
    DecodeHasRoom,
    /// Decode load could not be read. Fail closed rather than guess.
    DecodeLoadUnknown,
    /// The prefill worker is over its own busy line, so it has nothing to donate.
    PrefillBusy,
    /// Prefill load could not be read. An unchecked safety check is not a pass.
    PrefillLoadUnknown,
    /// The request may generate more than the continuation cap allows.
    BudgetAboveCap,
    /// The request has no bounded budget, so the commitment cannot be bounded.
    BudgetUnbounded,
    /// The prefill worker already holds its maximum concurrent continuations.
    ConcurrencyCapReached,
    /// A cap is configured but the running count could not be read. Refuse
    /// rather than assume zero, or a broken counter silently lifts the cap.
    ConcurrencyUnknown,
}

/// The decision itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillContinueDecision {
    /// Keep generating on the prefill worker.
    Continue,
    /// Hand off to a decode worker, as today.
    Skip(PrefillContinueSkip),
}

impl PrefillContinueDecision {
    pub fn should_continue(self) -> bool {
        matches!(self, Self::Continue)
    }

    pub fn skip_reason(self) -> Option<PrefillContinueSkip> {
        match self {
            Self::Continue => None,
            Self::Skip(reason) => Some(reason),
        }
    }
}

/// What the router measured for one request, at the moment it must decide.
///
/// Every field is optional where the signal can genuinely be missing, so the
/// policy can distinguish "measured, and fine" from "could not measure".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct PrefillContinueDecisionInput {
    /// Decode blocks the chosen decode worker would hold after admitting this
    /// request. `None` means the signal was unavailable.
    pub potential_decode_blocks: Option<usize>,

    /// Total decode KV blocks on that worker. `None` means unavailable.
    pub total_kv_blocks: Option<usize>,

    /// Whether the prefill worker holding this request is over its busy line.
    /// `None` means the signal was unavailable.
    pub prefill_worker_busy: Option<bool>,

    /// The request's remaining token budget. `None` means unbounded.
    pub remaining_budget_tokens: Option<u32>,

    /// Continuations already running on that prefill worker. `None` means the
    /// count was unavailable, which is refused when a cap is configured.
    pub active_continuations: Option<usize>,

    /// KV block size in tokens, used to convert the output reserve into blocks.
    pub block_size: usize,
}

/// Decides whether a request keeps generating on its prefill worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillContinuePolicy {
    enabled: bool,
    force: bool,
    decode_busy_threshold: Option<f64>,
    prefill_busy_threshold_configured: bool,
    output_reserve_tokens: usize,
    max_budget_tokens: Option<u32>,
    max_concurrent: Option<usize>,
}

impl PrefillContinueDecisionInput {
    /// The decode-side measurement, which every decision needs.
    ///
    /// `#[non_exhaustive]` forbids struct-expression syntax outside this crate,
    /// so the caller in `dynamo-llm` builds through these rather than a literal.
    pub fn new(
        potential_decode_blocks: Option<usize>,
        total_kv_blocks: Option<usize>,
        block_size: usize,
    ) -> Self {
        Self {
            potential_decode_blocks,
            total_kv_blocks,
            prefill_worker_busy: None,
            remaining_budget_tokens: None,
            active_continuations: None,
            block_size,
        }
    }

    pub fn with_prefill_worker_busy(mut self, busy: Option<bool>) -> Self {
        self.prefill_worker_busy = busy;
        self
    }

    pub fn with_remaining_budget_tokens(mut self, budget: Option<u32>) -> Self {
        self.remaining_budget_tokens = budget;
        self
    }

    pub fn with_active_continuations(mut self, active: Option<usize>) -> Self {
        self.active_continuations = active;
        self
    }
}

impl PrefillContinuePolicy {
    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            enabled: config.prefill_continue_enabled,
            force: config.prefill_continue_force,
            decode_busy_threshold: config.prefill_continue_decode_busy_threshold,
            prefill_busy_threshold_configured: config
                .prefill_continue_prefill_busy_threshold
                .or(config.router_queue_threshold)
                .is_some(),
            output_reserve_tokens: config.prefill_continue_output_reserve_tokens,
            max_budget_tokens: config.prefill_continue_max_budget_tokens,
            max_concurrent: config.prefill_continue_max_concurrent,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            force: false,
            decode_busy_threshold: None,
            prefill_busy_threshold_configured: false,
            output_reserve_tokens: 0,
            max_budget_tokens: None,
            max_concurrent: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether the caller needs to probe prefill load for this policy.
    ///
    /// False when no busy threshold is configured, because there is then nothing
    /// to evaluate the signal against. Probing costs a routing preview, so the
    /// caller skips it in that case.
    pub fn needs_prefill_worker_busy(&self) -> bool {
        self.enabled && self.prefill_busy_threshold_configured
    }

    /// The decision.
    ///
    /// Order matters: every safety gate runs before `force`, so the bring-up
    /// switch relaxes the decode-load trigger and nothing else.
    pub fn decide(&self, input: PrefillContinueDecisionInput) -> PrefillContinueDecision {
        use PrefillContinueSkip as Skip;

        if !self.enabled {
            return PrefillContinueDecision::Skip(Skip::Disabled);
        }

        // The commitment cannot be undone once made, so it is bounded here, at
        // admission, against a budget the request already carries.
        if let Some(cap) = self.max_budget_tokens {
            match input.remaining_budget_tokens {
                Some(budget) if budget > cap => {
                    return PrefillContinueDecision::Skip(Skip::BudgetAboveCap);
                }
                None => return PrefillContinueDecision::Skip(Skip::BudgetUnbounded),
                Some(_) => {}
            }
        }

        if let Some(max) = self.max_concurrent {
            match input.active_continuations {
                Some(active) if active >= max => {
                    return PrefillContinueDecision::Skip(Skip::ConcurrencyCapReached);
                }
                None => return PrefillContinueDecision::Skip(Skip::ConcurrencyUnknown),
                Some(_) => {}
            }
        }

        // The interlock: the feature spends prefill capacity to relieve decode,
        // so a loaded prefill worker has nothing to give.
        if self.needs_prefill_worker_busy() {
            match input.prefill_worker_busy {
                Some(true) => return PrefillContinueDecision::Skip(Skip::PrefillBusy),
                None => return PrefillContinueDecision::Skip(Skip::PrefillLoadUnknown),
                Some(false) => {}
            }
        }

        // Force is the bring-up path: it skips only the decode-load test, so a
        // deployment whose decode pool never fills can still exercise the feature.
        if self.force {
            return PrefillContinueDecision::Continue;
        }

        let Some(threshold) = self.decode_busy_threshold else {
            return PrefillContinueDecision::Skip(Skip::NoTrigger);
        };

        match (input.potential_decode_blocks, input.total_kv_blocks) {
            (Some(potential), Some(total)) if total > 0 => {
                // Admission reserves the prompt only and nothing reserves what
                // the request will generate, so a request that exactly fits is
                // admitted and then runs out of room while decoding. Round up:
                // a partial block still occupies a block.
                let reserve = match input.block_size {
                    0 => 0,
                    block_size => self.output_reserve_tokens.div_ceil(block_size),
                };
                let projected = potential.saturating_add(reserve) as f64;
                if projected > threshold * total as f64 {
                    PrefillContinueDecision::Continue
                } else {
                    PrefillContinueDecision::Skip(Skip::DecodeHasRoom)
                }
            }
            _ => PrefillContinueDecision::Skip(Skip::DecodeLoadUnknown),
        }
    }
}

impl Default for PrefillContinuePolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A policy that is on, has a trigger, and no other gates configured.
    fn policy(threshold: f64) -> PrefillContinuePolicy {
        PrefillContinuePolicy {
            enabled: true,
            force: false,
            decode_busy_threshold: Some(threshold),
            prefill_busy_threshold_configured: false,
            output_reserve_tokens: 0,
            max_budget_tokens: None,
            max_concurrent: None,
        }
    }

    /// Decode load as blocks-out-of-total, with everything else unset.
    fn decode_load(potential: usize, total: usize) -> PrefillContinueDecisionInput {
        PrefillContinueDecisionInput {
            potential_decode_blocks: Some(potential),
            total_kv_blocks: Some(total),
            ..Default::default()
        }
    }

    fn skip(decision: PrefillContinueDecision) -> PrefillContinueSkip {
        decision.skip_reason().expect("expected a skip")
    }

    // --- the off switches ---------------------------------------------------

    #[test]
    fn disabled_policy_never_continues() {
        let decision = PrefillContinuePolicy::disabled().decide(decode_load(100, 100));
        assert_eq!(skip(decision), PrefillContinueSkip::Disabled);
    }

    #[test]
    fn enabled_without_a_threshold_never_continues() {
        let mut policy = policy(0.9);
        policy.decode_busy_threshold = None;
        // Even against a completely full decode pool.
        let decision = policy.decide(decode_load(100, 100));
        assert_eq!(skip(decision), PrefillContinueSkip::NoTrigger);
    }

    // --- the decode trigger -------------------------------------------------

    #[test]
    fn continues_only_once_decode_is_over_the_threshold() {
        let policy = policy(0.9);

        assert_eq!(
            skip(policy.decide(decode_load(89, 100))),
            PrefillContinueSkip::DecodeHasRoom
        );
        // Exactly at the threshold is still room: the test is strictly greater.
        assert_eq!(
            skip(policy.decide(decode_load(90, 100))),
            PrefillContinueSkip::DecodeHasRoom
        );
        assert!(policy.decide(decode_load(91, 100)).should_continue());
    }

    #[test]
    fn unknown_decode_load_fails_closed() {
        let policy = policy(0.9);

        for input in [
            PrefillContinueDecisionInput {
                potential_decode_blocks: None,
                total_kv_blocks: Some(100),
                ..Default::default()
            },
            PrefillContinueDecisionInput {
                potential_decode_blocks: Some(99),
                total_kv_blocks: None,
                ..Default::default()
            },
            // A zero-capacity worker is not a full worker; it is an unreadable one.
            decode_load(99, 0),
        ] {
            assert_eq!(
                skip(policy.decide(input)),
                PrefillContinueSkip::DecodeLoadUnknown
            );
        }
    }

    #[test]
    fn output_reserve_pushes_a_request_over_the_line() {
        let mut policy = policy(0.9);
        policy.output_reserve_tokens = 2048;

        // 89 of 100 blocks is under the line on the prompt alone.
        let bare = decode_load(89, 100);
        assert_eq!(
            skip(policy.decide(bare)),
            PrefillContinueSkip::DecodeHasRoom
        );

        // The same request, once its own output is reserved, is over it.
        // 2048 tokens at 512 per block is 4 blocks: 89 + 4 = 93 > 90.
        let with_block_size = PrefillContinueDecisionInput {
            block_size: 512,
            ..bare
        };
        assert!(policy.decide(with_block_size).should_continue());
    }

    #[test]
    fn output_reserve_rounds_a_partial_block_up() {
        let mut policy = policy(0.9);
        policy.output_reserve_tokens = 1;
        // One token still occupies a whole block.
        let input = PrefillContinueDecisionInput {
            block_size: 512,
            ..decode_load(90, 100)
        };
        assert!(policy.decide(input).should_continue());
    }

    #[test]
    fn output_reserve_is_ignored_without_a_block_size() {
        let mut policy = policy(0.9);
        policy.output_reserve_tokens = 4096;
        // A zero block size cannot convert the reserve. It must not be guessed
        // at, so the decision falls back to the prompt projection.
        assert_eq!(
            skip(policy.decide(decode_load(89, 100))),
            PrefillContinueSkip::DecodeHasRoom
        );
    }

    // --- the prefill interlock ----------------------------------------------

    #[test]
    fn a_busy_prefill_worker_stops_the_continuation() {
        let mut policy = policy(0.9);
        policy.prefill_busy_threshold_configured = true;

        let input = PrefillContinueDecisionInput {
            prefill_worker_busy: Some(true),
            ..decode_load(99, 100)
        };
        assert_eq!(skip(policy.decide(input)), PrefillContinueSkip::PrefillBusy);
    }

    #[test]
    fn unknown_prefill_load_fails_closed_when_the_interlock_is_configured() {
        let mut policy = policy(0.9);
        policy.prefill_busy_threshold_configured = true;

        let input = decode_load(99, 100); // prefill_worker_busy is None
        assert_eq!(
            skip(policy.decide(input)),
            PrefillContinueSkip::PrefillLoadUnknown
        );
    }

    #[test]
    fn prefill_load_is_ignored_when_the_interlock_is_not_configured() {
        let policy = policy(0.9); // prefill_busy_threshold_configured = false
        let input = PrefillContinueDecisionInput {
            prefill_worker_busy: None,
            ..decode_load(99, 100)
        };
        assert!(policy.decide(input).should_continue());
    }

    // --- the commitment bound -----------------------------------------------

    #[test]
    fn a_request_over_the_budget_cap_is_not_continued() {
        let mut policy = policy(0.9);
        policy.max_budget_tokens = Some(2048);

        let over = PrefillContinueDecisionInput {
            remaining_budget_tokens: Some(2049),
            ..decode_load(99, 100)
        };
        assert_eq!(
            skip(policy.decide(over)),
            PrefillContinueSkip::BudgetAboveCap
        );

        let at_the_cap = PrefillContinueDecisionInput {
            remaining_budget_tokens: Some(2048),
            ..decode_load(99, 100)
        };
        assert!(policy.decide(at_the_cap).should_continue());
    }

    #[test]
    fn an_unbounded_budget_is_not_continued_when_a_cap_is_set() {
        let mut policy = policy(0.9);
        policy.max_budget_tokens = Some(2048);

        // Without a budget the commitment cannot be bounded, and Mode A has no
        // way to undo it once started.
        let input = PrefillContinueDecisionInput {
            remaining_budget_tokens: None,
            ..decode_load(99, 100)
        };
        assert_eq!(
            skip(policy.decide(input)),
            PrefillContinueSkip::BudgetUnbounded
        );
    }

    #[test]
    fn an_unbounded_budget_is_fine_when_no_cap_is_set() {
        let policy = policy(0.9);
        let input = PrefillContinueDecisionInput {
            remaining_budget_tokens: None,
            ..decode_load(99, 100)
        };
        assert!(policy.decide(input).should_continue());
    }

    // --- the concurrency cap ------------------------------------------------

    #[test]
    fn the_concurrency_cap_stops_further_continuations() {
        let mut policy = policy(0.9);
        policy.max_concurrent = Some(4);

        let at_the_cap = PrefillContinueDecisionInput {
            active_continuations: Some(4),
            ..decode_load(99, 100)
        };
        assert_eq!(
            skip(policy.decide(at_the_cap)),
            PrefillContinueSkip::ConcurrencyCapReached
        );

        let below = PrefillContinueDecisionInput {
            active_continuations: Some(3),
            ..decode_load(99, 100)
        };
        assert!(policy.decide(below).should_continue());
    }

    #[test]
    fn an_unreadable_continuation_count_refuses_when_a_cap_is_set() {
        // A broken counter must not silently lift the cap. This is the one gate
        // where assuming zero would be fail-open.
        let mut policy = policy(0.9);
        policy.max_concurrent = Some(4);

        let unknown = PrefillContinueDecisionInput {
            active_continuations: None,
            ..decode_load(99, 100)
        };
        assert_eq!(
            skip(policy.decide(unknown)),
            PrefillContinueSkip::ConcurrencyUnknown
        );

        // force must not bypass it either.
        policy.force = true;
        assert_eq!(
            skip(policy.decide(unknown)),
            PrefillContinueSkip::ConcurrencyUnknown
        );
    }

    #[test]
    fn an_unreadable_continuation_count_is_fine_without_a_cap() {
        let policy = policy(0.9);
        let unknown = PrefillContinueDecisionInput {
            active_continuations: None,
            ..decode_load(99, 100)
        };
        assert!(policy.decide(unknown).should_continue());
    }

    // --- force ---------------------------------------------------------------

    #[test]
    fn force_skips_the_decode_test_but_not_the_other_gates() {
        let mut policy = policy(0.9);
        policy.force = true;

        // An idle decode pool would normally mean "hand off"; force continues.
        assert!(policy.decide(decode_load(1, 100)).should_continue());
        // It also does not need decode load to be readable at all.
        assert!(
            policy
                .decide(PrefillContinueDecisionInput::default())
                .should_continue()
        );

        // But the safety gates still apply.
        policy.max_budget_tokens = Some(16);
        let over = PrefillContinueDecisionInput {
            remaining_budget_tokens: Some(17),
            ..Default::default()
        };
        assert_eq!(
            skip(policy.decide(over)),
            PrefillContinueSkip::BudgetAboveCap
        );

        policy.max_budget_tokens = None;
        policy.max_concurrent = Some(2);
        let at_cap = PrefillContinueDecisionInput {
            active_continuations: Some(2),
            ..Default::default()
        };
        assert_eq!(
            skip(policy.decide(at_cap)),
            PrefillContinueSkip::ConcurrencyCapReached
        );

        policy.max_concurrent = None;
        policy.prefill_busy_threshold_configured = true;
        let busy = PrefillContinueDecisionInput {
            prefill_worker_busy: Some(true),
            ..Default::default()
        };
        assert_eq!(skip(policy.decide(busy)), PrefillContinueSkip::PrefillBusy);
    }

    #[test]
    fn force_does_not_override_the_off_switch() {
        let mut policy = PrefillContinuePolicy::disabled();
        policy.force = true;
        assert_eq!(
            skip(policy.decide(decode_load(1, 100))),
            PrefillContinueSkip::Disabled
        );
    }

    // --- construction --------------------------------------------------------

    #[test]
    fn from_config_reads_every_knob() {
        let config = KvRouterConfig {
            prefill_continue_enabled: true,
            prefill_continue_force: true,
            prefill_continue_decode_busy_threshold: Some(0.85),
            prefill_continue_prefill_busy_threshold: Some(0.4),
            prefill_continue_output_reserve_tokens: 4096,
            prefill_continue_max_budget_tokens: Some(2048),
            prefill_continue_max_concurrent: Some(8),
            ..Default::default()
        };
        let policy = PrefillContinuePolicy::from_config(&config);

        assert!(policy.is_enabled());
        assert!(policy.needs_prefill_worker_busy());
        assert_eq!(policy.decode_busy_threshold, Some(0.85));
        assert_eq!(policy.output_reserve_tokens, 4096);
        assert_eq!(policy.max_budget_tokens, Some(2048));
        assert_eq!(policy.max_concurrent, Some(8));
        assert!(policy.force);
    }

    #[test]
    fn the_interlock_is_probed_only_when_a_busy_line_exists() {
        let probes = |enabled, own: Option<f64>, router: Option<f64>| {
            PrefillContinuePolicy::from_config(&KvRouterConfig {
                prefill_continue_enabled: enabled,
                prefill_continue_decode_busy_threshold: Some(0.9),
                prefill_continue_prefill_busy_threshold: own,
                router_queue_threshold: router,
                ..Default::default()
            })
            .needs_prefill_worker_busy()
        };

        assert!(probes(true, Some(0.4), None));
        // Its own threshold unset, but the router-wide one is configured.
        assert!(probes(true, None, Some(4.0)));
        // Neither configured: the interlock cannot run, so it is not required.
        assert!(!probes(true, None, None));
        // Off: there is no decision to interlock.
        assert!(!probes(false, Some(0.4), None));
    }
}
