// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decide whether a request keeps generating on its prefill worker.

use crate::scheduling::config::KvRouterConfig;

/// The skip reasons, written once.
///
/// The enum, `ALL` and `as_str` are generated from this one list, so a new
/// reason cannot reach the enum but miss `ALL` and lose its metric series.
macro_rules! skip_reasons {
    ($($(#[$meta:meta])* $variant:ident => $label:literal,)+) => {
        /// Why a request was not allowed to keep generating on its prefill worker.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum PrefillContinueSkip {
            $($(#[$meta])* $variant,)+
        }

        impl PrefillContinueSkip {
            /// Every reason, so a caller can create the metric series up front.
            pub const ALL: &'static [Self] = &[$(Self::$variant,)+];

            /// A stable, low-cardinality label for metrics.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $label,)+
                }
            }
        }
    };
}

skip_reasons! {
    /// The feature is off.
    Disabled => "disabled",
    /// The feature is on but no decode trigger is configured, so it can never
    /// fire. Distinct from `Disabled` so the reason an operator reads is true.
    NoTrigger => "no_trigger",
    /// The decode pool can take the request, so the normal handoff is correct.
    DecodeHasRoom => "decode_has_room",
    /// Decode load could not be read. Fail closed rather than guess.
    DecodeLoadUnknown => "decode_load_unknown",
    /// The prefill worker is over its own busy line, so it has nothing to donate.
    PrefillBusy => "prefill_busy",
    /// Prefill load could not be read. An unchecked safety check is not a pass.
    PrefillLoadUnknown => "prefill_load_unknown",
    /// The request may generate more than the continuation cap allows.
    BudgetAboveCap => "budget_above_cap",
    /// The request has no bounded budget, so the commitment cannot be bounded.
    BudgetUnbounded => "budget_unbounded",
    /// The request asks for several sequences.
    MultipleSequences => "multiple_sequences",
    /// The prefill worker already holds its maximum concurrent continuations.
    ConcurrencyCapReached => "concurrency_cap_reached",
    /// A cap is configured but the running count could not be read. Refuse
    /// rather than assume zero, or a broken counter silently lifts the cap.
    ConcurrencyUnknown => "concurrency_unknown",
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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PrefillContinueDecisionInput {
    /// What the chosen decode worker reports it is holding, as a fraction of
    /// what it can hold. A snapshot taken before this request is admitted, so
    pub decode_occupancy: Option<f64>,

    /// Whether the prefill worker holding this request is over its busy line.
    /// `None` means the signal was unavailable. It measures ordinary prefill
    /// work only, so a running continuation does not appear in it.
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
}

/// Decides whether a request keeps generating on its prefill worker.
/// The default is the off switch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PrefillContinuePolicy {
    enabled: bool,
    force: bool,
    decode_busy_threshold: Option<f64>,
    prefill_busy_threshold: Option<f64>,
    max_budget_tokens: Option<u32>,
    max_concurrent: Option<usize>,
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
            max_budget_tokens: config.prefill_continue_max_budget_tokens,
            max_concurrent: config.prefill_continue_max_concurrent,
        }
    }

    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether this policy evaluates the prefill-load interlock at all.
    pub fn needs_prefill_worker_busy(&self) -> bool {
        self.interlock_threshold().is_some()
    }

    /// The interlock threshold in force, inheriting the router-wide queue
    /// threshold when the feature does not set its own.
    pub fn interlock_threshold(&self) -> Option<f64> {
        if self.enabled {
            self.prefill_busy_threshold
        } else {
            None
        }
    }

    /// The per-worker ceiling on concurrent continuations, if one is set.
    pub fn max_concurrent(&self) -> Option<usize> {
        self.max_concurrent
    }

    /// The gates that cost nothing to evaluate.
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
    pub fn needs_decode_load(&self) -> bool {
        self.enabled && !self.force
    }

    /// The decision.
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
        // so a loaded prefill worker has nothing to give. Its threshold is a
        if self.needs_prefill_worker_busy() {
            match input.prefill_worker_busy {
                Some(true) => return PrefillContinueDecision::Skip(Skip::PrefillBusy),
                None => return PrefillContinueDecision::Skip(Skip::PrefillLoadUnknown),
                Some(false) => {}
            }
        }

        // Force is the bring-up path: it skips the decode-load threshold below,
        // which never fires on an idle deployment.
        if self.force {
            return PrefillContinueDecision::Continue;
        }

        let Some(threshold) = self.decode_busy_threshold else {
            return PrefillContinueDecision::Skip(Skip::NoTrigger);
        };

        let Some(decode_occupancy) = input.decode_occupancy else {
            return PrefillContinueDecision::Skip(Skip::DecodeLoadUnknown);
        };
        // Nothing is projected onto this reading: no worker reports the
        // physical cost of this prompt, so a router estimate would mix units.
        if decode_occupancy <= threshold {
            return PrefillContinueDecision::Skip(Skip::DecodeHasRoom);
        }

        PrefillContinueDecision::Continue
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
            max_budget_tokens: None,
            max_concurrent: None,
        }
    }

    /// Decode occupancy as blocks-out-of-total, with everything else unset.
    fn decode_load(used: usize, total: usize) -> PrefillContinueDecisionInput {
        PrefillContinueDecisionInput {
            decode_occupancy: (total > 0).then(|| used as f64 / total as f64),
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
                decode_occupancy: None,
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

    /// The 2P1D against 3P2D regression. One worker at 40 % and each of two
    /// workers at 40 % must decide alike: the gate reads a fraction of the
    /// selected rank, so pool size cannot move it.
    #[test]
    fn the_decision_follows_the_fraction_not_the_block_count() {
        let policy = policy(0.2);

        assert!(policy.decide(decode_load(1_659, 4_168)).should_continue());
        assert!(policy.decide(decode_load(40, 100)).should_continue());
        assert_eq!(
            skip(policy.decide(decode_load(400, 4_168))),
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
    fn a_light_prefill_worker_relieves_a_full_decode_pool() {
        let mut policy = policy(0.9);
        policy.prefill_busy_threshold = Some(0.8);

        // The regime the feature exists for: decode nearly full, prefill
        // carrying almost no continuation load.
        let input = PrefillContinueDecisionInput {
            prefill_worker_busy: Some(false),
            ..decode_load(99, 100)
        };
        assert_eq!(policy.decide(input), PrefillContinueDecision::Continue);
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
    fn an_unbounded_budget_is_refused_with_or_without_a_cap() {
        // A continuation occupies its worker until the model stops, so a
        // request that names no ceiling cannot be admitted, cap or no cap.
        // Clients that omit `max_tokens` are the common case.
        for cap in [None, Some(2048)] {
            let mut policy = policy(0.9);
            policy.max_budget_tokens = cap;

            let input = PrefillContinueDecisionInput {
                remaining_budget_tokens: None,
                ..decode_load(99, 100)
            };
            assert_eq!(
                skip(policy.decide(input)),
                PrefillContinueSkip::BudgetUnbounded,
                "{cap:?}"
            );
        }
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
            decode_occupancy: None,
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
            prefill_continue_max_budget_tokens: Some(2048),
            prefill_continue_max_concurrent: Some(8),
            ..Default::default()
        };
        let policy = PrefillContinuePolicy::from_config(&config);

        assert!(policy.is_enabled());
        assert!(policy.needs_prefill_worker_busy());
        assert_eq!(policy.decode_busy_threshold, Some(0.85));
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
        // silently, so pin the exact strings and not only their shape. The
        // match is exhaustive, so a new variant fails the build right here.
        let mut labels = std::collections::HashSet::new();
        for reason in PrefillContinueSkip::ALL {
            let pinned = match reason {
                PrefillContinueSkip::Disabled => "disabled",
                PrefillContinueSkip::NoTrigger => "no_trigger",
                PrefillContinueSkip::DecodeHasRoom => "decode_has_room",
                PrefillContinueSkip::DecodeLoadUnknown => "decode_load_unknown",
                PrefillContinueSkip::PrefillBusy => "prefill_busy",
                PrefillContinueSkip::PrefillLoadUnknown => "prefill_load_unknown",
                PrefillContinueSkip::BudgetAboveCap => "budget_above_cap",
                PrefillContinueSkip::BudgetUnbounded => "budget_unbounded",
                PrefillContinueSkip::MultipleSequences => "multiple_sequences",
                PrefillContinueSkip::ConcurrencyCapReached => "concurrency_cap_reached",
                PrefillContinueSkip::ConcurrencyUnknown => "concurrency_unknown",
            };
            assert_eq!(reason.as_str(), pinned);
            assert!(labels.insert(pinned), "labels must be distinct: {pinned}");
        }
        // `ALL` is generated from the same list as the enum, so it cannot be
        // short. This pins the labels themselves, which a rename would break.
        assert_eq!(labels.len(), PrefillContinueSkip::ALL.len());
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
