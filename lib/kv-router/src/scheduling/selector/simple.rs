// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_router_policy::{
    PolicyDecision, RouteCandidates, RouteContext, RoutePicker, RoutingPolicy,
};

use super::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext,
};
use crate::scheduling::types::WorkerSelectionPolicyError;

pub use dynamo_router_policy::RoutingPolicy as SimpleRoutingPolicy;

/// Native scorer for cache-unaware routing presets.
///
/// Static policies contribute a constant cost and do not request load inputs. Load-aware
/// policies project the host's active-request count into the picker's lower-is-better cost.
#[derive(Clone, Copy, Debug)]
pub struct SimpleWorkerScorer {
    policy: SimpleRoutingPolicy,
}

impl SimpleWorkerScorer {
    pub const fn new(policy: SimpleRoutingPolicy) -> Self {
        Self { policy }
    }

    pub const fn policy(&self) -> SimpleRoutingPolicy {
        self.policy
    }

    #[inline(always)]
    fn score_load(&self, load: u64) -> u64 {
        match self.policy {
            RoutingPolicy::RoundRobin | RoutingPolicy::Random => 0,
            RoutingPolicy::PowerOfTwoChoices
            | RoutingPolicy::LeastLoaded
            | RoutingPolicy::DeviceAwareWeighted => load,
        }
    }
}

impl WorkerScorer for SimpleWorkerScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        match self.policy {
            RoutingPolicy::RoundRobin | RoutingPolicy::Random => WorkerInputs::NONE,
            RoutingPolicy::PowerOfTwoChoices
            | RoutingPolicy::LeastLoaded
            | RoutingPolicy::DeviceAwareWeighted => WorkerInputs::LOAD,
        }
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let active_requests = candidate
            .load()
            .map(|load| load.active_requests() as u64)
            .unwrap_or(0);
        Ok(self.score_load(active_requests) as f64)
    }
}

/// Native picker for cache-unaware routing presets.
///
/// The same dependency-neutral kernel backs runtime's compatibility APIs and this native
/// `WorkerPicker`. LLM orchestration calls the inherent borrowed-candidate methods so static
/// round-robin and random selection remain allocation-free and candidate-count independent.
#[derive(Debug)]
pub struct SimpleWorkerPicker {
    inner: RoutePicker,
}

impl SimpleWorkerPicker {
    pub const fn new(policy: SimpleRoutingPolicy) -> Self {
        Self {
            inner: RoutePicker::new(policy),
        }
    }

    pub const fn policy(&self) -> SimpleRoutingPolicy {
        self.inner.policy()
    }

    #[inline(always)]
    pub fn peek<C: RouteCandidates + ?Sized>(
        &self,
        scorer: &SimpleWorkerScorer,
        candidates: &C,
        context: RouteContext,
    ) -> Option<PolicyDecision> {
        debug_assert_eq!(self.policy(), scorer.policy());
        self.inner.peek(&ScoredRows { candidates, scorer }, context)
    }

    #[inline(always)]
    pub fn select<C: RouteCandidates + ?Sized>(
        &self,
        scorer: &SimpleWorkerScorer,
        candidates: &C,
        context: RouteContext,
    ) -> Option<PolicyDecision> {
        debug_assert_eq!(self.policy(), scorer.policy());
        self.inner
            .select(&ScoredRows { candidates, scorer }, context)
    }
}

struct ScoredRows<'a, C: ?Sized> {
    candidates: &'a C,
    scorer: &'a SimpleWorkerScorer,
}

impl<C: RouteCandidates + ?Sized> RouteCandidates for ScoredRows<'_, C> {
    fn len(&self) -> usize {
        self.candidates.len()
    }

    fn load(&self, index: usize) -> u64 {
        self.scorer.score_load(self.candidates.load(index))
    }

    fn device(&self, index: usize) -> dynamo_router_policy::RouteDevice {
        self.candidates.device(index)
    }

    fn cache_hits(&self, index: usize) -> usize {
        self.candidates.cache_hits(index)
    }
}

struct NativeRows<'a> {
    input: WorkerInputView<'a>,
}

impl RouteCandidates for NativeRows<'_> {
    fn len(&self) -> usize {
        self.input.candidates.len()
    }

    fn load(&self, index: usize) -> u64 {
        ordered_cost(self.input.candidates()[index].cost())
    }
}

/// Map a finite `f64` onto an integer key with the same total ordering.
#[inline(always)]
fn ordered_cost(cost: f64) -> u64 {
    debug_assert!(cost.is_finite());
    let bits = cost.to_bits();
    if bits & (1 << 63) == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}

impl WorkerPicker for SimpleWorkerPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if self.policy() == RoutingPolicy::DeviceAwareWeighted {
            return Err(WorkerSelectionPolicyError::failed(
                "device-aware routing requires runtime-owned device and embedding-cache inputs",
            ));
        }
        self.inner
            .select(&NativeRows { input }, RouteContext::default())
            .map(|decision| decision.index)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible workers"))
    }

    fn peek(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if self.policy() == RoutingPolicy::DeviceAwareWeighted {
            return Err(WorkerSelectionPolicyError::failed(
                "device-aware routing requires runtime-owned device and embedding-cache inputs",
            ));
        }
        self.inner
            .peek(&NativeRows { input }, RouteContext::default())
            .map(|decision| decision.index)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible workers"))
    }
}

#[cfg(test)]
mod tests {
    use super::super::ScoredWorkerCandidate;
    use super::*;

    fn scored_candidates(costs: &[f64]) -> Vec<ScoredWorkerCandidate> {
        costs
            .iter()
            .enumerate()
            .map(|(worker_id, &cost)| ScoredWorkerCandidate {
                worker: crate::protocols::WorkerWithDpRank::from_worker_id(worker_id as u64),
                cost,
            })
            .collect()
    }

    fn selection_context() -> WorkerSelectionContext<'static> {
        WorkerSelectionContext {
            request_id: "test",
            request_blocks: 1,
            block_size: 1,
            track_prefill_tokens: false,
            weights: super::super::LogitWeights {
                overlap_score_credit: 0.0,
                overlap_score_credit_decay: 0.0,
                prefill_load_scale: 0.0,
                shared_cache_multiplier: 0.0,
            },
            min_active_prefill_tokens: 0,
            router_temperature_override: None,
            session_context: None,
            expected_output_tokens: None,
            priority_jump: 0.0,
            strict_priority: 0,
            advisory: false,
        }
    }

    struct Loads<'a>(&'a [u64]);

    impl RouteCandidates for Loads<'_> {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn load(&self, index: usize) -> u64 {
            self.0[index]
        }
    }

    #[test]
    fn native_round_robin_advisory_does_not_advance() {
        let scorer = SimpleWorkerScorer::new(SimpleRoutingPolicy::RoundRobin);
        let picker = SimpleWorkerPicker::new(SimpleRoutingPolicy::RoundRobin);
        let rows = Loads(&[0, 0]);
        assert_eq!(
            picker
                .peek(&scorer, &rows, RouteContext::default())
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            picker
                .peek(&scorer, &rows, RouteContext::default())
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            picker
                .select(&scorer, &rows, RouteContext::default())
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            picker
                .select(&scorer, &rows, RouteContext::default())
                .unwrap()
                .index,
            1
        );
    }

    #[test]
    fn native_least_loaded_reads_host_load_column() {
        let scorer = SimpleWorkerScorer::new(SimpleRoutingPolicy::LeastLoaded);
        let picker = SimpleWorkerPicker::new(SimpleRoutingPolicy::LeastLoaded);
        let rows = Loads(&[8, 1, 6]);
        assert_eq!(
            picker
                .select(&scorer, &rows, RouteContext::default())
                .unwrap()
                .index,
            1
        );
    }

    #[test]
    fn static_scorer_does_not_request_load() {
        let scorer = SimpleWorkerScorer::new(SimpleRoutingPolicy::RoundRobin);
        assert_eq!(scorer.required_worker_inputs(), WorkerInputs::NONE);
        assert_eq!(scorer.score_load(19), 0);
    }

    #[test]
    fn load_scorer_projects_active_requests() {
        let scorer = SimpleWorkerScorer::new(SimpleRoutingPolicy::PowerOfTwoChoices);
        assert_eq!(scorer.required_worker_inputs(), WorkerInputs::LOAD);
        assert_eq!(scorer.score_load(19), 19);
    }

    #[test]
    fn worker_picker_preserves_fractional_cost_ordering() {
        let mut picker = SimpleWorkerPicker::new(SimpleRoutingPolicy::LeastLoaded);
        let candidates = scored_candidates(&[0.9, 0.4]);
        let input = WorkerInputView {
            candidates: &candidates,
            cache: None,
            load: None,
            routing: None,
        };

        assert_eq!(picker.pick(&selection_context(), input).unwrap(), 1);
    }

    #[test]
    fn worker_picker_preserves_negative_cost_ordering_for_p2c() {
        let mut picker = SimpleWorkerPicker::new(SimpleRoutingPolicy::PowerOfTwoChoices);
        let candidates = scored_candidates(&[-1.0, -5.0]);
        let input = WorkerInputView {
            candidates: &candidates,
            cache: None,
            load: None,
            routing: None,
        };

        assert_eq!(picker.pick(&selection_context(), input).unwrap(), 1);
    }
}
