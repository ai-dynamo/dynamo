// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

use dynamo_kv_router::{
    KvRouterConfig, SharedCacheHits, WorkerCandidate, WorkerFilter, WorkerInputView, WorkerInputs,
    WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionInput,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CacheProbe {
    inputs: WorkerInputs,
    tier_matches: bool,
    shared_hits: bool,
    declarations: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}

impl CacheProbe {
    fn declare(&self) -> WorkerInputs {
        self.declarations.fetch_add(1, Ordering::Relaxed);
        self.inputs
    }

    fn check(&self, context: &WorkerSelectionContext<'_>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // Ordinary request context stays available without CACHE.
        assert_eq!(context.prompt_tokens(), 17);
        if self.inputs.contains(WorkerInputs::CACHE) {
            assert_eq!(context.has_tier_matches(), Some(self.tier_matches));
            assert_eq!(context.shared_cache_hits().is_some(), self.shared_hits);
            if let Some(hits) = context.shared_cache_hits() {
                assert_eq!(hits.hits_beyond(0), 4);
            }
        } else {
            assert_eq!(context.has_tier_matches(), None);
            assert!(context.shared_cache_hits().is_none());
        }
    }
}

impl WorkerFilter for CacheProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        _candidate: &WorkerCandidate,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        self.check(context);
        Ok(true)
    }
}

impl WorkerScorer for CacheProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        _candidates: &[WorkerCandidate],
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.check(context);
        costs.fill(0.0);
        Ok(())
    }
}

impl WorkerPicker for CacheProbe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.declare()
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        self.check(context);
        Ok(0)
    }
}

#[test]
fn cache_context_requires_each_components_own_declaration() {
    for mask in 0..8 {
        for unrequested in [WorkerInputs::NONE, WorkerInputs::LOAD] {
            for tier_matches in [false, true] {
                for shared_hits in [false, true] {
                    let (workers, mut request) = support::fixture(2, 17);
                    if !tier_matches {
                        request.overlap.tier_overlap_blocks = Default::default();
                    }
                    if shared_hits {
                        request.shared_cache_hits =
                            Some(SharedCacheHits::from_ranges(vec![1..3, 5..7]));
                    }
                    let declarations = Arc::new(AtomicUsize::new(0));
                    let calls = Arc::new(AtomicUsize::new(0));
                    let probe = |inputs| CacheProbe {
                        inputs,
                        tier_matches,
                        shared_hits,
                        declarations: declarations.clone(),
                        calls: calls.clone(),
                    };
                    let inputs = |bit| {
                        if mask & bit == 0 {
                            unrequested
                        } else {
                            WorkerInputs::CACHE
                        }
                    };
                    let policy = WorkerSelectionPolicy::new_with_filters(
                        KvRouterConfig::default(),
                        "test",
                        vec![Box::new(probe(inputs(1))), Box::new(probe(unrequested))],
                        vec![Box::new(probe(inputs(2))), Box::new(probe(unrequested))],
                        Box::new(probe(inputs(4))),
                    );
                    assert_eq!(declarations.load(Ordering::Relaxed), 5);
                    for _ in 0..2 {
                        policy
                            .select_worker(WorkerSelectionInput::configured(
                                &workers,
                                &request,
                                request.eligibility(),
                                16,
                            ))
                            .unwrap();
                    }
                    // Four worker/ranks pass two filters, then two scorers and one picker run.
                    assert_eq!(calls.load(Ordering::Relaxed), 22);
                    assert_eq!(declarations.load(Ordering::Relaxed), 5);
                }
            }
        }
    }
}
