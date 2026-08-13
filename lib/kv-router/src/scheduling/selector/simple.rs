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
        self.input.candidates()[index].cost() as u64
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
    use super::*;

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
}
