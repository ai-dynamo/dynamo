// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the batch scoring contract using only public plugin inputs.
mod support;

use std::sync::{Arc, Mutex};

use dynamo_kv_router::{
    KvRouterConfig, WorkerCandidate, WorkerCandidates, WorkerFilter, WorkerInputView, WorkerInputs,
    WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelector,
};
use support::fixture;

#[derive(Debug, PartialEq)]
enum Call {
    Score(usize, Vec<u64>),
    Pick,
}

struct RelativeLoadScorer {
    index: usize,
    calls: Arc<Mutex<Vec<Call>>>,
}
impl WorkerScorer for RelativeLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        assert_eq!(context.prompt_tokens(), 17);
        assert!(candidates.iter().all(|c| c.cache().is_none()));
        let mut ids: Vec<_> = candidates.iter().map(|c| c.worker().worker_id).collect();
        ids.sort_unstable();
        self.calls
            .lock()
            .unwrap()
            .push(Call::Score(self.index, ids));
        let minimum = candidates
            .iter()
            .map(|c| c.load().unwrap().active_requests())
            .min()
            .unwrap();
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = (candidate.load().unwrap().active_requests() - minimum) as f64;
        }
        Ok(())
    }
}

struct ExcludeWorkerOne;
impl WorkerFilter for ExcludeWorkerOne {
    fn keep(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        Ok(candidate.worker().worker_id != 1)
    }
}

struct InspectCosts(Arc<Mutex<Vec<Call>>>);
impl WorkerPicker for InspectCosts {
    fn pick(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        assert!(input.load().is_none());
        assert!(input.cache().is_none());
        self.0.lock().unwrap().push(Call::Pick);
        for candidate in input.candidates() {
            // Each scorer contributes the same relative count; the host adds both.
            let expected = if candidate.worker().worker_id == 3 {
                0.0
            } else {
                6.0
            };
            assert_eq!(candidate.cost(), expected);
        }
        Ok(input
            .candidates()
            .iter()
            .position(|c| c.cost() == 0.0)
            .unwrap())
    }
}

#[test]
fn scores_batches_from_surviving_workers_and_resets_each_selection() {
    let (workers, mut request) = fixture(4, 17);
    request.allowed_worker_ids = Some([1, 2, 3].into_iter().collect());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let policy = WorkerSelectionPolicy::new_with_filters(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(ExcludeWorkerOne)],
        (0..2)
            .map(|index| {
                Box::new(RelativeLoadScorer {
                    index,
                    calls: calls.clone(),
                }) as Box<dyn WorkerScorer>
            })
            .collect(),
        Box::new(InspectCosts(calls.clone())),
    );
    // Excluded workers have zero load. Neither may lower the batch minimum.
    for round in 0..3 {
        for (worker, load) in &mut request.worker_loads {
            load.active_requests = match worker.worker_id {
                2 => 10 + round * 10 + 3,
                3 => 10 + round * 10,
                _ => 0,
            };
        }
        if round == 2 {
            request.allowed_worker_ids = Some([3].into_iter().collect());
        }
        let result = policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap();
        assert_eq!(result.worker.worker_id, 3);
        let mut calls = calls.lock().unwrap();
        let expected = if round == 2 {
            vec![3, 3]
        } else {
            vec![2, 2, 3, 3]
        };
        assert_eq!(
            *calls,
            vec![
                Call::Score(0, expected.clone()),
                Call::Score(1, expected),
                Call::Pick
            ]
        );
        calls.clear();
    }
}

struct FailScoring(Arc<Mutex<usize>>);
impl WorkerScorer for FailScoring {
    fn score(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        _: WorkerCandidates<'_>,
        _: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        *self.0.lock().unwrap() += 1;
        Err(WorkerSelectionPolicyError::failed("score failed"))
    }
}
struct NeverPick;
impl WorkerPicker for NeverPick {
    fn pick(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        _: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        panic!("failed or empty selections must stop before picking")
    }
}

#[test]
fn skips_scoring_for_empty_sets_and_aborts_on_error() {
    let (workers, mut request) = fixture(2, 17);
    let calls = Arc::new(Mutex::new(0));
    let policy = WorkerSelectionPolicy::new_with_filters(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(ExcludeWorkerOne)],
        vec![Box::new(FailScoring(calls.clone()))],
        Box::new(NeverPick),
    );
    for allowed in [vec![], vec![1]] {
        request.allowed_worker_ids = Some(allowed.into_iter().collect());
        assert!(
            policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .is_err()
        );
        assert_eq!(*calls.lock().unwrap(), 0);
    }
    request.allowed_worker_ids = None;
    let error = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap_err();
    assert!(error.to_string().contains("score failed"));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[test]
fn rejects_nonfinite_contributions_and_overflow_before_picking() {
    struct Constant(f64);
    impl WorkerScorer for Constant {
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            costs.fill(self.0);
            Ok(())
        }
    }
    let (workers, request) = fixture(1, 17);
    for costs in [
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::MAX, f64::MAX],
    ] {
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            costs
                .into_iter()
                .map(|cost| Box::new(Constant(cost)) as Box<dyn WorkerScorer>)
                .collect(),
            Box::new(NeverPick),
        );
        let error = policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
}

#[test]
fn picker_columns_and_costs_stay_aligned_after_a_scoring_error() {
    struct FailSecondScoreOnce(usize);
    impl WorkerScorer for FailSecondScoreOnce {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                self.0 += 1;
                if self.0 == 2 {
                    return Err(WorkerSelectionPolicyError::failed("score failed"));
                }
                *cost = candidate.load().unwrap().active_requests() as f64;
            }
            Ok(())
        }
    }
    struct CheckColumns;
    impl WorkerPicker for CheckColumns {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }
        fn pick(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let cache = input.cache().unwrap();
            let load = input.load().unwrap();
            assert_eq!(cache.len(), input.candidates().len());
            assert_eq!(load.len(), input.candidates().len());
            for (row, candidate) in input.candidates().iter().enumerate() {
                let worker = candidate.worker();
                let expected_overlap = (worker.worker_id * 7 + u64::from(worker.dp_rank)) % 9;
                assert_eq!(
                    cache.get(row).unwrap().device_overlap_blocks(),
                    expected_overlap as f64
                );
                assert_eq!(load[row].active_requests(), worker.worker_id as usize % 5);
                assert_eq!(candidate.cost(), load[row].active_requests() as f64);
            }
            Ok(0)
        }
    }
    let (workers, mut request) = fixture(8, 17);
    let policy = WorkerSelectionPolicy::new_with_filters(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(ExcludeWorkerOne)],
        vec![Box::new(FailSecondScoreOnce(0))],
        Box::new(CheckColumns),
    );
    let select = |request: &dynamo_kv_router::scheduling::SchedulingRequest| {
        policy.select_worker(support::selection_input(&workers, request, 16))
    };
    assert!(
        select(&request)
            .unwrap_err()
            .to_string()
            .contains("score failed")
    );
    request.allowed_worker_ids = Some([1, 3, 4].into_iter().collect());
    assert!([3, 4].contains(&select(&request).unwrap().worker.worker_id));
    request.allowed_worker_ids = None;
    assert_ne!(select(&request).unwrap().worker.worker_id, 1);
}

#[test]
fn unwritten_costs_cannot_reuse_a_previous_selection_or_scorer() {
    struct Fill;
    impl WorkerScorer for Fill {
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            costs.fill(7.0);
            Ok(())
        }
    }
    struct OmitOnSecondCall(bool);
    impl WorkerScorer for OmitOnSecondCall {
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            if std::mem::replace(&mut self.0, false) {
                costs.fill(1.0);
            }
            Ok(())
        }
    }
    struct First;
    impl WorkerPicker for First {
        fn pick(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert!(input.candidates().iter().all(|c| c.cost() == 8.0));
            Ok(0)
        }
    }
    let (workers, request) = fixture(2, 17);
    let policy = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(Fill), Box::new(OmitOnSecondCall(true))],
        Box::new(First),
    );
    let select = || policy.select_worker(support::selection_input(&workers, &request, 16));
    select().unwrap();
    let error = select().unwrap_err();
    assert!(matches!(
        error,
        dynamo_kv_router::KvSchedulerError::WorkerSelectionPolicy(
            WorkerSelectionPolicyError::NonFiniteCost {
                scorer_index: 1,
                row: 0
            }
        )
    ));
}
