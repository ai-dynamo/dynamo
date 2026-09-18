// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise scorer preparation from an external crate using only public plugin inputs.
mod support;

use std::sync::{Arc, Mutex};

use dynamo_kv_router::{
    KvRouterConfig, WorkerCandidate, WorkerFilter, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerScorer, WorkerSelectionContext, WorkerSelectionInput, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelector,
};
use support::fixture;

#[derive(Debug, PartialEq)]
enum Call {
    Prepare(usize, Vec<u64>),
    Score(usize),
    Pick,
}

struct RelativeLoadScorer {
    index: usize,
    minimum: usize,
    calls: Arc<Mutex<Vec<Call>>>,
}

impl WorkerScorer for RelativeLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn prepare(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: &[WorkerCandidate],
    ) -> Result<(), WorkerSelectionPolicyError> {
        assert_eq!(context.prompt_tokens(), 17);
        assert!(candidates.iter().all(|c| c.cache().is_none()));
        let mut ids: Vec<_> = candidates.iter().map(|c| c.worker().worker_id).collect();
        ids.sort_unstable();
        self.calls
            .lock()
            .unwrap()
            .push(Call::Prepare(self.index, ids));
        self.minimum = candidates
            .iter()
            .map(|c| c.load().unwrap().active_requests())
            .min()
            .unwrap();
        Ok(())
    }

    fn score(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        self.calls.lock().unwrap().push(Call::Score(self.index));
        Ok((candidate.load().unwrap().active_requests() - self.minimum) as f64)
    }
}

struct ExcludeWorkerOne;
impl WorkerFilter for ExcludeWorkerOne {
    fn keep(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
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
fn prepares_all_scorers_from_surviving_workers_and_resets_each_selection() {
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
                    minimum: usize::MAX,
                    calls: calls.clone(),
                }) as Box<dyn WorkerScorer>
            })
            .collect(),
        Box::new(InspectCosts(calls.clone())),
    );
    // Excluded workers have zero load. Neither may lower the preparation minimum.
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
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 3);
        let mut calls = calls.lock().unwrap();
        let expected = if round == 2 {
            vec![3, 3]
        } else {
            vec![2, 2, 3, 3]
        };
        assert_eq!(calls[0], Call::Prepare(0, expected.clone()));
        assert_eq!(calls[1], Call::Prepare(1, expected.clone()));
        assert_eq!(calls.len(), 2 + expected.len() * 2 + 1);
        for pair in calls[2..calls.len() - 1].chunks_exact(2) {
            assert_eq!(pair, [Call::Score(0), Call::Score(1)]);
        }
        assert_eq!(calls.last(), Some(&Call::Pick));
        calls.clear();
    }
}

struct FailPreparation(Arc<Mutex<usize>>);
impl WorkerScorer for FailPreparation {
    fn prepare(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        _: &[WorkerCandidate],
    ) -> Result<(), WorkerSelectionPolicyError> {
        *self.0.lock().unwrap() += 1;
        Err(WorkerSelectionPolicyError::failed("prepare failed"))
    }
    fn score(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        _: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        panic!("preparation failure must stop scoring")
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
fn skips_preparation_for_empty_sets_and_aborts_on_preparation_error() {
    let (workers, mut request) = fixture(2, 17);
    let calls = Arc::new(Mutex::new(0));
    let policy = WorkerSelectionPolicy::new_with_filters(
        KvRouterConfig::default(),
        "test",
        vec![Box::new(ExcludeWorkerOne)],
        vec![Box::new(FailPreparation(calls.clone()))],
        Box::new(NeverPick),
    );
    for allowed in [vec![], vec![1]] {
        request.allowed_worker_ids = Some(allowed.into_iter().collect());
        assert!(
            policy
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .is_err()
        );
        assert_eq!(*calls.lock().unwrap(), 0);
    }
    request.allowed_worker_ids = None;
    let error = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap_err();
    assert!(error.to_string().contains("prepare failed"));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[test]
fn rejects_nonfinite_contributions_and_overflow_before_picking() {
    struct Constant(f64);
    impl WorkerScorer for Constant {
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: &WorkerCandidate,
        ) -> Result<f64, WorkerSelectionPolicyError> {
            Ok(self.0)
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
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
}
