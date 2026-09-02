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
    /// The request asks for several sequences.
    ///
    /// Two reasons, and the second is why this refusal is permanent rather
    /// than a placeholder. A continuation's stream carries no sequence index,
    /// so the sequences would merge and the first to finish would end the
    /// response for all of them — that part is a few lines of Python away from
    /// being fixed. But the token budget is *per sequence*: `n` sequences
    /// commit the worker to `n` times the request's `max_tokens`, while the
    /// budget this policy reads reports one times. Emitting the index without
    /// also fixing that would give a correct-looking answer backed by a
    /// commitment nothing bounds, which is worse than refusing.
    MultipleSequences,
    /// The prefill worker already holds its maximum concurrent continuations.
    ConcurrencyCapReached,
    /// A cap is configured but the running count could not be read. Refuse
    /// rather than assume zero, or a broken counter silently lifts the cap.
    ConcurrencyUnknown,
}

impl PrefillContinueSkip {
    /// Every reason, so a caller can create the metric series up front.
    ///
    /// A counter with no observations exposes no sample at all, so an operator
    /// asking "why did it never fire?" would get an empty query rather than a
    /// zero. The enum is `#[non_exhaustive]`, which stops a caller building
    /// this list itself, so it lives here where a new variant is added.
    pub const ALL: &'static [Self] = &[
        Self::Disabled,
        Self::NoTrigger,
        Self::DecodeHasRoom,
        Self::DecodeLoadUnknown,
        Self::PrefillBusy,
        Self::PrefillLoadUnknown,
        Self::BudgetAboveCap,
        Self::BudgetUnbounded,
        Self::MultipleSequences,
        Self::ConcurrencyCapReached,
        Self::ConcurrencyUnknown,
    ];

    /// A stable, low-cardinality label for metrics.
    ///
    /// Deliberately not `Debug`: these become a Prometheus label value, so
    /// renaming a variant must not silently rename a series an operator has a
    /// dashboard on.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoTrigger => "no_trigger",
            Self::DecodeHasRoom => "decode_has_room",
            Self::DecodeLoadUnknown => "decode_load_unknown",
            Self::PrefillBusy => "prefill_busy",
            Self::PrefillLoadUnknown => "prefill_load_unknown",
            Self::BudgetAboveCap => "budget_above_cap",
            Self::BudgetUnbounded => "budget_unbounded",
            Self::MultipleSequences => "multiple_sequences",
            Self::ConcurrencyCapReached => "concurrency_cap_reached",
            Self::ConcurrencyUnknown => "concurrency_unknown",
        }
    }
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

    /// Sequences the engine will run for this request: the larger of `n` and
    /// `best_of`, because `best_of` generates sequences that `n` does not
    /// return. `None` means one.
    pub sequences: Option<u8>,

    /// KV block size in tokens, used to convert the output reserve into blocks.
    pub block_size: usize,
}

/// Decides whether a request keeps generating on its prefill worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillContinuePolicy {
    enabled: bool,
    force: bool,
    decode_busy_threshold: Option<f64>,
    prefill_busy_threshold: Option<f64>,
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
            sequences: None,
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

    pub fn with_sequences(mut self, sequences: Option<u8>) -> Self {
        self.sequences = sequences;
        self
    }
}

impl PrefillContinuePolicy {
    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            enabled: config.prefill_continue_enabled,
            force: config.prefill_continue_force,
            decode_busy_threshold: config.prefill_continue_decode_busy_threshold,
            // Resolved once, here, so a caller cannot disagree with the policy
            // about which threshold is in force.
            prefill_busy_threshold: config
                .prefill_continue_prefill_busy_threshold
                .or(config.router_queue_threshold),
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
            prefill_busy_threshold: None,
            output_reserve_tokens: 0,
            max_budget_tokens: None,
            max_concurrent: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether this policy evaluates the prefill-load interlock at all.
    ///
    /// False when no busy threshold is configured, because there is then
    /// nothing to evaluate the signal against.
    pub fn needs_prefill_worker_busy(&self) -> bool {
        self.interlock_threshold().is_some()
    }

    /// The interlock threshold in force, inheriting the router-wide queue
    /// threshold when the feature does not set its own.
    ///
    /// One source of truth: the router probes with exactly the value the policy
    /// will judge the answer against. Named apart from the field so that
    /// dropping the parens inside this file cannot silently bypass `enabled`.
    pub fn interlock_threshold(&self) -> Option<f64> {
        if self.enabled {
            self.prefill_busy_threshold
        } else {
            None
        }
    }

    /// The per-worker ceiling on concurrent continuations, if one is set.
    ///
    /// Enforced where the worker is known, which is at dispatch. [`Self::decide`]
    /// runs before a worker is chosen, so its own cap check can only ever be a
    /// pool-level filter on a count the caller measured; the authoritative
    /// bound is the router's per-worker census.
    pub fn max_concurrent(&self) -> Option<usize> {
        self.max_concurrent
    }

    /// The gates that cost nothing to evaluate.
    ///
    /// Measuring load costs a scheduler selection apiece, so the caller runs
    /// this first and only measures when it returns `None`. These are the same
    /// gates, in the same order, that [`Self::decide`] applies — it is written
    /// in terms of this so the two cannot drift apart.
    pub fn preflight(
        &self,
        remaining_budget_tokens: Option<u32>,
        active_continuations: Option<usize>,
        sequences: Option<u8>,
    ) -> Option<PrefillContinueSkip> {
        use PrefillContinueSkip as Skip;

        if !self.enabled {
            return Some(Skip::Disabled);
        }

        // The commitment cannot be undone once made, so it is bounded here, at
        // admission, against a budget the request already carries.
        //
        // A request with no budget of its own is refused whether or not a cap
        // is configured. Nesting this inside the cap left the default
        // configuration admitting an unbounded continuation, which then held
        // its worker until the model chose to stop — and clients that omit
        // `max_tokens` are the common case, not the exception.
        let Some(budget) = remaining_budget_tokens else {
            return Some(Skip::BudgetUnbounded);
        };
        if self.max_budget_tokens.is_some_and(|cap| budget > cap) {
            return Some(Skip::BudgetAboveCap);
        }

        // A handoff response carries a sequence index on every chunk; a
        // forwarded continuation carries none.
        if sequences.is_some_and(|count| count > 1) {
            return Some(Skip::MultipleSequences);
        }

        if let Some(max) = self.max_concurrent {
            match active_continuations {
                Some(active) if active >= max => return Some(Skip::ConcurrencyCapReached),
                None => return Some(Skip::ConcurrencyUnknown),
                Some(_) => {}
            }
        }

        None
    }

    /// Whether the caller needs to measure decode load for this policy.
    ///
    /// False under the bring-up override, which continues without consulting
    /// decode load at all, so measuring it would be a scheduler selection spent
    /// on an answer nothing reads.
    pub fn needs_decode_load(&self) -> bool {
        self.enabled && !self.force
    }

    /// The decision.
    ///
    /// Order matters: every safety gate runs before `force`, so the bring-up
    /// switch relaxes the decode-load trigger and nothing else.
    pub fn decide(&self, input: PrefillContinueDecisionInput) -> PrefillContinueDecision {
        use PrefillContinueSkip as Skip;

        if let Some(skip) = self.preflight(
            input.remaining_budget_tokens,
            input.active_continuations,
            input.sequences,
        ) {
            return PrefillContinueDecision::Skip(skip);
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
            prefill_busy_threshold: None,
            output_reserve_tokens: 0,
            max_budget_tokens: None,
            max_concurrent: None,
        }
    }

    /// Decode load as blocks-out-of-total, with everything else unset.
    /// A request that is admissible on every axis except the one under test.
    ///
    /// Carries a budget, because an absent one is now a refusal in its own
    /// right and would mask whichever gate the test is actually about.
    fn decode_load(potential: usize, total: usize) -> PrefillContinueDecisionInput {
        PrefillContinueDecisionInput {
            potential_decode_blocks: Some(potential),
            total_kv_blocks: Some(total),
            remaining_budget_tokens: Some(256),
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
                ..decode_load(99, 100)
            },
            PrefillContinueDecisionInput {
                total_kv_blocks: None,
                ..decode_load(99, 100)
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
        policy.prefill_busy_threshold = Some(0.8);

        let input = PrefillContinueDecisionInput {
            prefill_worker_busy: Some(true),
            ..decode_load(99, 100)
        };
        assert_eq!(skip(policy.decide(input)), PrefillContinueSkip::PrefillBusy);
    }

    #[test]
    fn unknown_prefill_load_fails_closed_when_the_interlock_is_configured() {
        let mut policy = policy(0.9);
        policy.prefill_busy_threshold = Some(0.8);

        let input = decode_load(99, 100); // prefill_worker_busy is None
        assert_eq!(
            skip(policy.decide(input)),
            PrefillContinueSkip::PrefillLoadUnknown
        );
    }

    #[test]
    fn prefill_load_is_ignored_when_the_interlock_is_not_configured() {
        let policy = policy(0.9); // no interlock threshold configured
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
    fn an_unbounded_budget_is_refused_even_with_no_cap_set() {
        // A continuation occupies its worker until the model stops, so a
        // request that names no ceiling cannot be admitted, cap or no cap.
        // Clients that omit `max_tokens` are the common case.
        let policy = policy(0.9);
        let input = PrefillContinueDecisionInput {
            remaining_budget_tokens: None,
            ..decode_load(99, 100)
        };
        assert_eq!(
            skip(policy.decide(input)),
            PrefillContinueSkip::BudgetUnbounded
        );
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
        let unreadable = PrefillContinueDecisionInput {
            potential_decode_blocks: None,
            total_kv_blocks: None,
            ..decode_load(0, 0)
        };
        assert!(policy.decide(unreadable).should_continue());

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
            ..decode_load(1, 100)
        };
        assert_eq!(
            skip(policy.decide(at_cap)),
            PrefillContinueSkip::ConcurrencyCapReached
        );

        policy.max_concurrent = None;
        policy.prefill_busy_threshold = Some(0.8);
        let busy = PrefillContinueDecisionInput {
            prefill_worker_busy: Some(true),
            ..decode_load(1, 100)
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

    #[test]
    fn several_sequences_cannot_continue() {
        // A forwarded stream carries no sequence index, so `n` sequences would
        // merge and the first to finish would end the response for all.
        let policy = policy(0.9);
        for sequences in [Some(2), Some(8)] {
            let several = PrefillContinueDecisionInput {
                sequences,
                ..decode_load(99, 100)
            };

            assert_eq!(
                skip(policy.decide(several)),
                PrefillContinueSkip::MultipleSequences,
                "{sequences:?}"
            );
        }

        // One, and unset meaning one, both continue.
        for sequences in [None, Some(1)] {
            let input = PrefillContinueDecisionInput {
                sequences,
                ..decode_load(99, 100)
            };
            assert!(policy.decide(input).should_continue(), "{sequences:?}");
        }
    }

    #[test]
    fn skip_labels_are_stable_and_distinct() {
        // These are Prometheus label values. A rename breaks an operator's
        // dashboard silently, and a duplicate merges two series just as
        // silently, so pin the exact strings and not only their shape.
        let expected = [
            (PrefillContinueSkip::Disabled, "disabled"),
            (PrefillContinueSkip::NoTrigger, "no_trigger"),
            (PrefillContinueSkip::DecodeHasRoom, "decode_has_room"),
            (
                PrefillContinueSkip::DecodeLoadUnknown,
                "decode_load_unknown",
            ),
            (PrefillContinueSkip::PrefillBusy, "prefill_busy"),
            (
                PrefillContinueSkip::PrefillLoadUnknown,
                "prefill_load_unknown",
            ),
            (PrefillContinueSkip::BudgetAboveCap, "budget_above_cap"),
            (PrefillContinueSkip::BudgetUnbounded, "budget_unbounded"),
            (PrefillContinueSkip::MultipleSequences, "multiple_sequences"),
            (
                PrefillContinueSkip::ConcurrencyCapReached,
                "concurrency_cap_reached",
            ),
            (
                PrefillContinueSkip::ConcurrencyUnknown,
                "concurrency_unknown",
            ),
        ];
        for (reason, label) in expected {
            assert_eq!(reason.as_str(), label);
        }

        let distinct: std::collections::HashSet<_> =
            expected.iter().map(|(_, label)| *label).collect();
        assert_eq!(distinct.len(), expected.len(), "labels must be distinct");
        assert_eq!(
            expected.len(),
            PrefillContinueSkip::ALL.len(),
            "every reason must be pinned here"
        );
    }

    #[test]
    fn all_lists_every_skip_reason() {
        // `ALL` drives which metric series exist, so a variant missing from it
        // is a series that never appears — the exact failure it exists to stop.
        // Adding a variant makes this match non-exhaustive and fails the build.
        for reason in PrefillContinueSkip::ALL {
            match reason {
                PrefillContinueSkip::Disabled
                | PrefillContinueSkip::NoTrigger
                | PrefillContinueSkip::DecodeHasRoom
                | PrefillContinueSkip::DecodeLoadUnknown
                | PrefillContinueSkip::PrefillBusy
                | PrefillContinueSkip::PrefillLoadUnknown
                | PrefillContinueSkip::BudgetAboveCap
                | PrefillContinueSkip::BudgetUnbounded
                | PrefillContinueSkip::MultipleSequences
                | PrefillContinueSkip::ConcurrencyCapReached
                | PrefillContinueSkip::ConcurrencyUnknown => {}
            }
        }
        assert_eq!(PrefillContinueSkip::ALL.len(), 11);
    }

    #[test]
    fn preflight_and_decide_agree_on_the_cheap_gates() {
        // The router runs `preflight` first to avoid paying for a load probe a
        // cheap gate would refuse anyway. If the two ever disagreed, the router
        // would skip for one reason and the policy report another.
        let mut policy = policy(0.9);
        policy.max_budget_tokens = Some(128);
        policy.max_concurrent = Some(2);

        for budget in [None, Some(64), Some(4096)] {
            for active in [None, Some(0), Some(2), Some(9)] {
                for sequences in [None, Some(1), Some(2)] {
                    let input = PrefillContinueDecisionInput::new(Some(0), Some(100), 16)
                        .with_remaining_budget_tokens(budget)
                        .with_active_continuations(active)
                        .with_sequences(sequences);

                    let preflight = policy.preflight(budget, active, sequences);
                    let decided = policy.decide(input).skip_reason();
                    match preflight {
                        Some(reason) => assert_eq!(
                            decided,
                            Some(reason),
                            "preflight refused {budget:?}/{active:?} but decide did not agree"
                        ),
                        // decide may still refuse later, on a gate preflight does
                        // not cover; it must not refuse for a cheap-gate reason.
                        None => assert!(
                            !matches!(
                                decided,
                                Some(PrefillContinueSkip::BudgetAboveCap)
                                    | Some(PrefillContinueSkip::BudgetUnbounded)
                                    | Some(PrefillContinueSkip::MultipleSequences)
                                    | Some(PrefillContinueSkip::ConcurrencyCapReached)
                                    | Some(PrefillContinueSkip::ConcurrencyUnknown)
                            ),
                            "preflight passed {budget:?}/{active:?} but decide refused on a cheap gate"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn the_override_does_not_need_decode_load() {
        let mut policy = policy(0.9);
        assert!(
            policy.needs_decode_load(),
            "without force, decode load decides"
        );

        policy.force = true;
        assert!(
            !policy.needs_decode_load(),
            "force continues without consulting decode load, so measuring it buys nothing"
        );

        policy.enabled = false;
        assert!(!policy.needs_decode_load());
    }
}
