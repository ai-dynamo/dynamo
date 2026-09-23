// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use dynamo_kv_router::{
    KvRouterConfig, SharedCacheHits, WorkerCacheInput, WorkerCandidate, WorkerCandidates,
    WorkerFilter, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct InputProbe {
    inputs: WorkerInputs,
    tier_matches: bool,
    shared_hits: Option<u32>,
    shared_identity: Arc<AtomicUsize>,
    declarations: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}

impl InputProbe {
    fn declare(&self) -> WorkerInputs {
        self.declarations.fetch_add(1, Ordering::Relaxed);
        self.inputs
    }

    fn check_candidate(&self, candidate: WorkerCandidate<'_>) {
        assert_eq!(
            candidate.cache().is_some(),
            self.inputs.contains(WorkerInputs::CACHE)
        );
        assert_eq!(
            candidate.load().is_some(),
            self.inputs.contains(WorkerInputs::LOAD)
        );
        assert_eq!(
            candidate.preferred_taint_multiplier().is_some(),
            self.inputs.contains(WorkerInputs::PREFERRED_TAINT)
        );
        if let Some(cache) = candidate.cache() {
            self.check_cache(cache);
            assert_eq!(
                cache.host_overlap_blocks(),
                if self.tier_matches { 2.0 } else { 0.0 }
            );
        }
        if let Some(load) = candidate.load() {
            assert!(load.is_available());
        }
    }

    fn check(&self, context: &WorkerSelectionContext<'_>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // Ordinary request context stays available without CACHE.
        assert_eq!(context.prompt_tokens(), 17);
    }

    fn check_cache(&self, cache: WorkerCacheInput<'_>) {
        assert_eq!(cache.has_tier_matches(), self.tier_matches);
        assert_eq!(cache.shared_hits().is_some(), self.shared_hits.is_some());
        if let Some(hits) = cache.shared_hits() {
            assert_eq!(hits.hits_beyond(0), self.shared_hits.unwrap());
            let address = std::ptr::from_ref(hits) as usize;
            if let Err(previous) = self.shared_identity.compare_exchange(
                0,
                address,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                assert_eq!(address, previous);
            }
        }
    }
}

impl WorkerFilter for InputProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        self.check(context);
        self.check_candidate(candidate);
        Ok(true)
    }
}

impl WorkerScorer for InputProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.check(context);
        assert_eq!(candidates.len(), costs.len());
        assert!(!candidates.is_empty());
        assert!(candidates.get(candidates.len()).is_none());
        for (row, candidate) in candidates.iter().enumerate() {
            self.check_candidate(candidate);
            let indexed = candidates.get(row).unwrap();
            self.check_candidate(indexed);
            assert_eq!(indexed.worker(), candidate.worker());
        }
        costs.fill(0.0);
        Ok(())
    }
}

impl WorkerPicker for InputProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        self.check(context);
        assert_eq!(
            input.cache().is_some(),
            self.inputs.contains(WorkerInputs::CACHE)
        );
        assert_eq!(
            input.load().is_some(),
            self.inputs.contains(WorkerInputs::LOAD)
        );
        if let Some(cache) = input.cache() {
            assert_eq!(cache.len(), input.candidates().len());
            assert!(!cache.is_empty());
            assert!(cache.get(cache.len()).is_none());
            for (row, entry) in cache.iter().enumerate() {
                self.check_cache(entry);
                self.check_cache(cache.get(row).unwrap());
            }
        }
        for candidate in input.candidates() {
            assert_eq!(
                candidate.preferred_taint_multiplier().is_some(),
                self.inputs.contains(WorkerInputs::PREFERRED_TAINT)
            );
        }
        Ok(0)
    }
}

fn inputs(mask: usize) -> WorkerInputs {
    [
        WorkerInputs::CACHE,
        WorkerInputs::LOAD,
        WorkerInputs::PREFERRED_TAINT,
    ]
    .into_iter()
    .enumerate()
    .fold(WorkerInputs::NONE, |inputs, (bit, group)| {
        if mask & (1 << bit) == 0 {
            inputs
        } else {
            inputs | group
        }
    })
}

#[test]
fn each_component_can_only_read_its_own_inputs() {
    for filter_mask in 0..8 {
        for scorer_mask in 0..8 {
            for picker_mask in 0..8 {
                for tier_matches in [false, true] {
                    for shared_hits in [None, Some(0), Some(4)] {
                        let (workers, mut request) = support::fixture(2, 17);
                        if !tier_matches {
                            request.overlap.tier_overlap_blocks = Default::default();
                        }
                        request.shared_cache_hits = shared_hits.map(|count| {
                            SharedCacheHits::from_ranges(if count == 0 {
                                vec![]
                            } else {
                                vec![1..3, 5..7]
                            })
                        });
                        // Even unmatched preferences produce Some(1.0), so an undeclared
                        // component leaking another component's taint value fails this test.
                        request
                            .routing_constraints
                            .preferred_taints
                            .insert("preferred".into(), 1.0);
                        let declarations = Arc::new(AtomicUsize::new(0));
                        let calls = Arc::new(AtomicUsize::new(0));
                        let shared_identity = Arc::new(AtomicUsize::new(0));
                        let probe = |mask| InputProbe {
                            inputs: inputs(mask),
                            tier_matches,
                            shared_hits,
                            shared_identity: shared_identity.clone(),
                            declarations: declarations.clone(),
                            calls: calls.clone(),
                        };
                        let policy = WorkerSelectionPolicy::new_with_filters(
                            KvRouterConfig::default(),
                            "test",
                            vec![
                                Box::new(probe(filter_mask)),
                                Box::new(probe(7 ^ filter_mask)),
                            ],
                            vec![
                                Box::new(probe(scorer_mask)),
                                Box::new(probe(7 ^ scorer_mask)),
                            ],
                            Box::new(probe(picker_mask)),
                        );
                        assert_eq!(declarations.load(Ordering::Relaxed), 5);
                        for _ in 0..2 {
                            policy
                                .select_worker(support::selection_input(&workers, &request, 16))
                                .unwrap();
                        }
                        if let Some(hits) = request.shared_cache_hits.as_ref() {
                            assert_eq!(
                                shared_identity.load(Ordering::Relaxed),
                                std::ptr::from_ref(hits) as usize
                            );
                        }
                        assert_eq!(calls.load(Ordering::Relaxed), 22);
                        assert_eq!(declarations.load(Ordering::Relaxed), 5);
                    }
                }
            }
        }
    }
}
