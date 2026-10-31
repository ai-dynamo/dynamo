// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive `WorkerSelectionPolicy` through the kernel's `WorkerSelector` with a scheduling
//! request, worker configs, and eligibility: the stage and its adapter together.

use crate::SessionContext;
use crate::protocols::WorkerWithDpRank;
use dynamo_worker_selection::WorkerInputView;
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
};

use rustc_hash::FxHashMap;

use super::DefaultWorkerSelector;
use super::test_support::*;
use super::{CandidatePreparerExt, WorkerSelectionInput, WorkerSelector};
use crate::WorkerSelectionInputTrigger;
use crate::protocols::WorkerConfigLike;
use crate::types::{KvSchedulerError, SchedulingRequest};
use dynamo_worker_selection::*;

fn uses_exclusive_affinity(selector: &impl WorkerSelector<TaintedWorkerConfig>) -> bool {
    selector.uses_exclusive_affinity_target()
}

struct FirstPicker;

impl WorkerPicker for FirstPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        Ok(0)
    }
}

#[test]
fn default_policy_matches_default_selector() {
    let worker0 = WorkerWithDpRank::from_worker_id(0);
    let worker1 = WorkerWithDpRank::from_worker_id(1);
    let workers = HashMap::from([
        (0, TaintedWorkerConfig::default()),
        (1, TaintedWorkerConfig::default()),
    ]);
    let mut request = base_request(16);
    request.worker_loads =
        worker_loads_with_active_decode(FxHashMap::from_iter([(worker0, 8), (worker1, 1)]));
    let config = SelectionConfig {
        router_temperature: 0.0,
        ..Default::default()
    };

    let expected = DefaultWorkerSelector::new(Some(config.clone()), "test")
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    let policy = WorkerSelectionPolicy::reference(config, "test");
    assert!(uses_exclusive_affinity(&policy));
    let actual = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();

    assert_eq!(actual.worker, expected.worker);
    assert_eq!(actual.required_blocks, expected.required_blocks);
    assert_eq!(
        actual.effective_overlap_blocks,
        expected.effective_overlap_blocks
    );
    assert_eq!(actual.cached_tokens, expected.cached_tokens);
    assert_eq!(
        actual.potential_decode_blocks,
        expected.potential_decode_blocks
    );
}

#[test]
fn custom_picker_counts_tracked_blocks_without_worker_projection() {
    struct LoadPicker;

    impl WorkerPicker for LoadPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let load = &input.load().expect("requested load inputs")[0];
            assert!(!load.is_available());
            assert_eq!(load.decode_cost_blocks(), 2.0);
            assert_eq!(load.active_prefill_tokens(), 0);
            assert_eq!(load.active_requests(), 0);
            Ok(0)
        }
    }

    let workers = HashMap::from([(0, TaintedWorkerConfig::default())]);
    let mut request = base_request(33);
    request.token_seq = Some(vec![0, 1]);
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(LoadPicker),
    );

    policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
}

#[test]
fn selection_trace_is_captured_only_when_requested() {
    struct DecodeLoadScorer;

    impl WorkerScorer for DecodeLoadScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                *cost = candidate.load().unwrap().decode_cost_blocks();
            }
            Ok(())
        }
    }

    /// Picks the costliest candidate so the selected row falls outside the top-k.
    struct HighestCostPicker;

    impl WorkerPicker for HighestCostPicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(input
                .candidates()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.cost().total_cmp(&b.1.cost()))
                .map(|(row, _)| row)
                .unwrap())
        }
    }

    let workers: HashMap<_, _> = (0..8u64)
        .map(|id| (id, TaintedWorkerConfig::default()))
        .collect();
    let mut request = base_request(16);
    request.worker_loads = worker_loads_with_active_decode(FxHashMap::from_iter(
        (0..8u64).map(|id| (WorkerWithDpRank::from_worker_id(id), id as usize)),
    ));
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        vec![Box::new(DecodeLoadScorer)],
        Box::new(HighestCostPicker),
    );

    let input = || WorkerSelectionInput::configured(&workers, &request, request.eligibility(), 16);
    let (untraced, no_trace) = policy.select_worker_traced(input(), None).unwrap();
    assert_eq!(untraced.worker.worker_id, 7);
    assert!(no_trace.is_none());

    let (traced, trace) = policy
        .select_worker_traced(
            input(),
            Some(crate::protocols::SelectionTraceSpec { top_k: 3 }),
        )
        .unwrap();
    assert_eq!(traced.worker.worker_id, 7);
    let trace = trace.expect("trace requested");
    assert_eq!(trace.candidate_count, 8);
    assert!(!trace.pinned);
    let mut rows = trace.rows.clone();
    rows.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    let ids: Vec<u64> = rows.iter().map(|row| row.worker.worker_id).collect();
    assert_eq!(
        ids,
        vec![0, 1, 2, 7],
        "three cheapest plus the selected worker"
    );
    assert_eq!(rows[3].cost, 7.0);
}

#[test]
fn custom_picker_receives_requested_cache_inputs() {
    struct HighestOverlapPicker;

    impl WorkerPicker for HighestOverlapPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(input
                .cache()
                .expect("requested cache inputs")
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.device_overlap_blocks()
                        .total_cmp(&right.device_overlap_blocks())
                })
                .map(|(row, _)| row)
                .expect("eligible candidate"))
        }
    }

    let worker0 = WorkerWithDpRank::from_worker_id(0);
    let worker1 = WorkerWithDpRank::from_worker_id(1);
    let workers = HashMap::from([
        (0, TaintedWorkerConfig::default()),
        (1, TaintedWorkerConfig::default()),
    ]);
    let mut request = base_request(16);
    request.overlap.tier_overlap_blocks.device = FxHashMap::from_iter([(worker0, 1), (worker1, 3)]);
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(HighestOverlapPicker),
    );

    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker, worker1);
}

#[test]
fn components_read_fetchable_prefixes_only_when_declared() {
    use crate::kv_hints::KvTransferCandidates;
    use crate::protocols::{ExternalSequenceBlockHash, KvHintTransferWorkerMetadata};
    use std::sync::{Arc, Mutex};

    struct HintWorker;

    impl WorkerConfigLike for HintWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn kv_hint_transfer_metadata_for_dp_rank(
            &self,
            _dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "prefill",
                source_control_endpoint: Some("tcp://worker:23280"),
            })
        }
    }

    type Seen = Arc<Mutex<Vec<(u64, Option<Option<u32>>)>>>;

    struct TransferScorer {
        inputs: WorkerInputs,
        seen: Seen,
    }

    impl WorkerScorer for TransferScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.inputs
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                let transfer = candidate
                    .kv_transfer()
                    .map(|transfer| transfer.source_prefix_blocks());
                self.seen
                    .lock()
                    .unwrap()
                    .push((candidate.worker().worker_id, transfer));
                *cost = 0.0;
            }
            Ok(())
        }
    }

    struct TransferFilter;

    impl WorkerFilter for TransferFilter {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::KV_TRANSFER
        }

        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            // Only scorers receive fetch inputs.
            assert_eq!(candidate.kv_transfer(), None);
            Ok(true)
        }
    }

    let warm = WorkerWithDpRank::from_worker_id(0);
    let workers = HashMap::from([(0, HintWorker), (1, HintWorker)]);
    let select = |candidates: Option<KvTransferCandidates>| {
        let (declared, undeclared) = (Seen::default(), Seen::default());
        let mut request = base_request(64);
        request.overlap.tier_overlap_blocks.device = FxHashMap::from_iter([(warm, 3)]);
        request.kv_transfer_candidates = candidates;
        let policy = WorkerSelectionPolicy::new_with_filters(
            SelectionConfig::default(),
            "test",
            vec![Box::new(TransferFilter)],
            vec![
                Box::new(TransferScorer {
                    inputs: WorkerInputs::KV_TRANSFER,
                    seen: declared.clone(),
                }),
                Box::new(TransferScorer {
                    inputs: WorkerInputs::CACHE,
                    seen: undeclared.clone(),
                }),
            ],
            Box::new(FirstPicker),
        );
        policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        let sorted = |seen: Seen| {
            let mut seen = seen.lock().unwrap().clone();
            seen.sort_unstable();
            seen
        };
        (sorted(declared), sorted(undeclared))
    };

    let (declared, undeclared) = select(Some(KvTransferCandidates {
        block_hashes: (101..=104).map(ExternalSequenceBlockHash).collect(),
        owner_prefix_blocks: vec![(warm.into(), 3)],
        routing_snapshot: None,
    }));
    // The warm worker holds the whole fetchable prefix; the cold one can fetch it.
    assert_eq!(declared, vec![(0, Some(None)), (1, Some(Some(3)))]);
    assert_eq!(undeclared, vec![(0, None), (1, None)]);
    // Without transfer candidates the host has nothing to report.
    let (declared, _) = select(None);
    assert_eq!(declared, vec![(0, None), (1, None)]);
}

#[test]
fn worker_info_reaches_every_component_without_declaration() {
    struct ReportedKvFilter;
    impl WorkerFilter for ReportedKvFilter {
        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            let info = candidate.info();
            assert_eq!(info.worker(), candidate.worker());
            Ok(info.total_kv_blocks().is_some())
        }
    }

    struct BatchBudgetScorer;
    impl WorkerScorer for BatchBudgetScorer {
        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                let info = candidate.info();
                assert_eq!(info.worker(), candidate.worker());
                *cost = info.max_num_batched_tokens().unwrap_or(0) as f64;
            }
            Ok(())
        }
    }

    struct LargestKvPicker;
    impl WorkerPicker for LargestKvPicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let rows = input.candidates().len();
            assert!(input.info(rows).is_none());
            for (row, candidate) in input.candidates().iter().enumerate() {
                let info = input.info(row).expect("in-range row");
                assert_eq!(info.worker(), candidate.worker());
                assert_eq!(
                    info.max_num_batched_tokens().unwrap_or(0) as f64,
                    candidate.cost()
                );
            }
            Ok((0..rows)
                .max_by_key(|&row| input.info(row).and_then(|info| info.total_kv_blocks()))
                .expect("eligible candidate"))
        }
    }

    let config = |max_num_batched_tokens, total_kv_blocks| crate::test_utils::SimpleWorkerConfig {
        max_num_batched_tokens,
        total_kv_blocks,
        ..Default::default()
    };
    // Worker 2 reports no KV capacity, so the filter drops it.
    let workers = HashMap::from([
        (0, config(Some(64), Some(10))),
        (1, config(None, Some(20))),
        (2, config(Some(64), None)),
    ]);
    let request = base_request(16);
    let policy = WorkerSelectionPolicy::new_with_filters(
        SelectionConfig::default(),
        "test",
        vec![Box::new(ReportedKvFilter)],
        vec![Box::new(BatchBudgetScorer)],
        Box::new(LargestKvPicker),
    );

    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker, WorkerWithDpRank::from_worker_id(1));
}

#[test]
fn prepared_candidates_select_like_rows_built_at_selection() {
    use crate::ScheduleMode;
    use crate::kv_hints::KvTransferCandidates;
    use crate::protocols::{ExternalSequenceBlockHash, KvHintTransferWorkerMetadata};

    struct Worker {
        dp_size: u32,
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for Worker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            self.dp_size
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }

        fn kv_hint_transfer_metadata_for_dp_rank(
            &self,
            _dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "prefill",
                source_control_endpoint: Some("tcp://worker:23280"),
            })
        }
    }

    struct EveryInputScorer;

    impl WorkerScorer for EveryInputScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
                | WorkerInputs::LOAD
                | WorkerInputs::PREFERRED_TAINT
                | WorkerInputs::KV_TRANSFER
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                let load = candidate.load().unwrap();
                let cache = candidate.cache().unwrap();
                let fetchable = candidate
                    .kv_transfer()
                    .and_then(|transfer| transfer.source_prefix_blocks())
                    .unwrap_or(0);
                *cost = (load.decode_cost_blocks() + load.active_prefill_tokens() as f64
                    - cache.device_overlap_blocks()
                    - 0.5 * f64::from(fetchable))
                    * candidate.preferred_taint_multiplier().unwrap_or(1.0);
            }
            Ok(())
        }
    }

    let workers: HashMap<_, _> = (0..6)
        .map(|worker_id| {
            let taints = if worker_id % 2 == 0 {
                HashSet::from(["fast".to_string()])
            } else {
                HashSet::new()
            };
            let dp_size = if worker_id == 5 { 2 } else { 1 };
            (worker_id, Worker { dp_size, taints })
        })
        .collect();
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        vec![Box::new(EveryInputScorer)],
        Box::new(FirstPicker),
    );
    let preparer = WorkerSelector::<Worker>::candidate_preparer(&policy).unwrap();
    // The trace keeps every candidate's cost, so it compares every row.
    let trace = Some(SelectionTraceSpec { top_k: 0 });
    let loads = |shift: u64| {
        (0..6)
            .flat_map(|worker_id| {
                let ranks = if worker_id == 5 { 0..2 } else { 0..1 };
                ranks.map(move |rank| WorkerWithDpRank::new(worker_id, rank))
            })
            .map(|worker| {
                let blocks = ((worker.worker_id + shift) * 7 % 11) as usize;
                (worker, blocks)
            })
            .collect()
    };

    let mut base = base_request(256);
    for (worker, blocks) in [(0, 9), (1, 4), (3, 12), (5, 6)] {
        let worker = WorkerWithDpRank::from_worker_id(worker);
        base.overlap
            .effective_overlap_blocks
            .insert(worker, blocks as f64);
        base.overlap
            .tier_overlap_blocks
            .device
            .insert(worker, blocks);
    }
    for case in ["tracked", "preferred_taint", "fetch"] {
        let mut request = base_request(256);
        request.overlap = base.overlap.clone();
        request.worker_loads = worker_loads_with_active_decode(loads(0));
        match case {
            "tracked" => {
                request.mode = ScheduleMode::Tracked {
                    request_id: "tracked".into(),
                }
            }
            "preferred_taint" => {
                request.routing_constraints.preferred_taints =
                    HashMap::from([("fast".to_string(), 1.0)])
            }
            "fetch" => {
                request.kv_transfer_candidates = Some(KvTransferCandidates {
                    block_hashes: (101..=112).map(ExternalSequenceBlockHash).collect(),
                    owner_prefix_blocks: vec![(WorkerWithDpRank::from_worker_id(3).into(), 12)],
                    routing_snapshot: None,
                })
            }
            _ => {}
        }
        fn input<'a>(
            workers: &'a HashMap<u64, Worker>,
            request: &'a SchedulingRequest,
        ) -> WorkerSelectionInput<'a, Worker> {
            WorkerSelectionInput::configured(workers, request, request.eligibility(), 16)
        }
        let prepared = preparer.prepare(input(&workers, &request)).unwrap();
        // Loads are read at selection, after any booking since preparation.
        request.worker_loads = worker_loads_with_active_decode(loads(3));

        let expected = policy
            .select_worker_traced(input(&workers, &request), trace)
            .unwrap();
        let actual = policy
            .select_worker_prepared(input(&workers, &request), Some(prepared), trace)
            .unwrap();
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{case}");
    }
}

#[test]
fn preferred_taints_are_materialized_only_when_requested() {
    struct PreferenceScorer;

    impl WorkerScorer for PreferenceScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::PREFERRED_TAINT
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                assert!(candidate.preferred_taint_multiplier().is_some());
                *cost = 0.0;
            }
            Ok(())
        }
    }

    struct PreferencePicker;

    impl WorkerPicker for PreferencePicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::PREFERRED_TAINT
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert!(input.candidates()[0].preferred_taint_multiplier().is_some());
            Ok(0)
        }
    }

    let workers = HashMap::from([(
        0,
        TaintedWorkerConfig {
            taints: HashSet::from(["preferred".to_string()]),
        },
    )]);
    let mut request = base_request(16);
    request.routing_constraints.preferred_taints = HashMap::from([("preferred".to_string(), 0.5)]);
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        vec![Box::new(PreferenceScorer)],
        Box::new(PreferencePicker),
    );

    assert_eq!(
        <WorkerSelectionPolicy as WorkerSelector<TaintedWorkerConfig>>::required_worker_inputs(
            &policy,
        ),
        WorkerInputs::PREFERRED_TAINT
    );
    policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
}

#[test]
fn custom_policy_skips_undeclared_preferred_taints() {
    struct CountingTaintConfig {
        taints: HashSet<String>,
        taint_reads: Cell<usize>,
    }

    impl WorkerConfigLike for CountingTaintConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            self.taint_reads.set(self.taint_reads.get() + 1);
            &self.taints
        }
    }

    struct NoPreferencePicker;

    impl WorkerPicker for NoPreferencePicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert!(input.candidates()[0].preferred_taint_multiplier().is_none());
            Ok(0)
        }
    }

    let workers = HashMap::from([(
        0,
        CountingTaintConfig {
            taints: HashSet::from(["preferred".to_string()]),
            taint_reads: Cell::new(0),
        },
    )]);
    let mut request = base_request(16);
    request.routing_constraints.preferred_taints = HashMap::from([("preferred".to_string(), 0.5)]);
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(NoPreferencePicker),
    );

    policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    // Eligibility checks required taints once. The preference multiplier must not perform a
    // second lookup when no policy component declares it.
    assert_eq!(workers[&0].taint_reads.get(), 1);
}

#[test]
fn custom_policy_does_not_receive_effective_overlap_as_device_overlap() {
    struct RawDeviceOverlapPicker;

    impl WorkerPicker for RawDeviceOverlapPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert_eq!(
                input
                    .cache()
                    .expect("cache input")
                    .get(0)
                    .unwrap()
                    .device_overlap_blocks(),
                0.0
            );
            Ok(0)
        }
    }

    let worker = WorkerWithDpRank::from_worker_id(0);
    let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
    let mut request = base_request(16);
    request.overlap.effective_overlap_blocks.insert(worker, 3.5);
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(RawDeviceOverlapPicker),
    );

    policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
}

#[test]
fn custom_picker_receives_session_metadata() {
    struct ContextPicker;

    impl WorkerPicker for ContextPicker {
        fn pick(
            &mut self,
            context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let session = context.session_context().expect("session context");
            assert_eq!(session.session_id(), "session-1");
            assert_eq!(session.parent_session_id(), Some("root"));
            assert_eq!(session.session_final(), Some(false));
            assert_eq!(
                session.agent_headers()["x-codex-turn-metadata"],
                ["{future schema}"]
            );
            assert_eq!(
                session.agent_headers()["x-claude-code-future"],
                ["first", "second"]
            );
            assert_eq!(
                session.input_trigger(),
                Some(WorkerSelectionInputTrigger::ToolResult)
            );
            assert_eq!(context.expected_output_tokens(), Some(128));
            assert_eq!(context.priority_jump(), 3.0);
            assert_eq!(context.strict_priority(), 2);
            Ok(0)
        }
    }

    let workers = HashMap::from([(0, TaintedWorkerConfig::default())]);
    let mut request = base_request(16);
    assert!(
        SessionContext::new("root".into(), None, None, None)
            .agent_headers()
            .is_empty()
    );
    let headers = std::sync::Arc::new(std::collections::BTreeMap::from([
        (
            "x-codex-turn-metadata".into(),
            vec!["{future schema}".into()],
        ),
        (
            "x-claude-code-future".into(),
            vec!["first".into(), "second".into()],
        ),
    ]));
    request.session_context = Some(
        SessionContext::new(
            "session-1".into(),
            Some("root".into()),
            Some(false),
            Some(WorkerSelectionInputTrigger::ToolResult),
        )
        .with_agent_headers(headers.clone()),
    );
    assert!(std::ptr::eq(
        request.session_context.as_ref().unwrap().agent_headers(),
        headers.as_ref()
    ));
    let cloned = request.session_context.clone().unwrap();
    assert!(std::ptr::eq(cloned.agent_headers(), headers.as_ref()));
    request.expected_output_tokens = Some(128);
    request.priority_jump = 3.0;
    request.strict_priority = 2;
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(ContextPicker),
    );

    policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
}

#[test]
fn custom_picker_receives_affinity_target_without_narrowing_candidates() {
    struct AffinityPicker;

    impl WorkerPicker for AffinityPicker {
        fn pick(
            &mut self,
            context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let target = context.affinity_target().expect("affinity target");
            assert_eq!(input.candidates().len(), 2);
            input
                .candidates()
                .iter()
                .position(|candidate| {
                    candidate.worker().worker_id == target.worker_id
                        && target
                            .dp_rank
                            .is_none_or(|rank| candidate.worker().dp_rank == rank)
                })
                .ok_or_else(|| WorkerSelectionPolicyError::failed("affinity target unavailable"))
        }
    }

    let worker1 = WorkerWithDpRank::from_worker_id(1);
    let workers = HashMap::from([
        (0, TaintedWorkerConfig::default()),
        (1, TaintedWorkerConfig::default()),
    ]);
    let mut request = base_request(16);
    request.affinity_target = Some(worker1.into());
    let policy = WorkerSelectionPolicy::new(
        SelectionConfig::default(),
        "test",
        Vec::new(),
        Box::new(AffinityPicker),
    );
    assert!(!uses_exclusive_affinity(&policy));

    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();
    assert_eq!(selected.worker, worker1);
}

#[test]
fn rejected_filters_do_not_materialize_scorer_inputs() {
    struct RejectWithoutSignals;

    impl WorkerFilter for RejectWithoutSignals {
        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            assert!(candidate.cache().is_none());
            assert!(candidate.load().is_none());
            assert!(candidate.preferred_taint_multiplier().is_none());
            Ok(false)
        }
    }

    struct CacheScorer;

    impl WorkerScorer for CacheScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _candidates: WorkerCandidates<'_>,
            _costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            unreachable!("rejected worker must not reach scorers")
        }
    }

    let workers = HashMap::from([(0, TaintedWorkerConfig::default())]);
    let request = base_request(16);
    let policy = WorkerSelectionPolicy::new_with_filters(
        SelectionConfig::default(),
        "test",
        vec![Box::new(RejectWithoutSignals)],
        vec![Box::new(CacheScorer)],
        Box::new(FirstPicker),
    );

    assert!(matches!(
        policy.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16
        )),
        Err(KvSchedulerError::AllEligibleWorkersFiltered)
    ));
}

#[test]
fn best_eligible_cache_reuse_skips_filtered_workers() {
    struct RejectWorker(WorkerWithDpRank);
    impl WorkerFilter for RejectWorker {
        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            Ok(candidate.worker() != self.0)
        }
    }

    struct ZeroScorer;
    impl WorkerScorer for ZeroScorer {
        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            costs.fill(0.0);
            Ok(())
        }
    }

    let rejected = WorkerWithDpRank::from_worker_id(0);
    let kept = WorkerWithDpRank::from_worker_id(1);
    let workers = HashMap::from([
        (0, TaintedWorkerConfig::default()),
        (1, TaintedWorkerConfig::default()),
    ]);
    let mut request = base_request(128);
    request.mode = crate::ScheduleMode::Tracked {
        request_id: "test".into(),
    };
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(rejected, 6);
    request.overlap.tier_overlap_blocks.device.insert(kept, 1);
    let policy = WorkerSelectionPolicy::new_with_filters(
        SelectionConfig::default(),
        "test",
        vec![Box::new(RejectWorker(rejected))],
        vec![Box::new(ZeroScorer)],
        Box::new(FirstPicker),
    );

    let selected = policy
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap();

    assert_eq!(selected.worker, kept);
    // The rejected worker's 96 cached tokens must not count as available reuse.
    assert_eq!(selected.max_raw_cached_tokens, Some(16));
    assert_eq!(selected.selected_raw_cached_tokens, Some(16));
}

#[test]
fn policy_requirements_union_all_components() {
    struct CacheFilter;
    impl WorkerFilter for CacheFilter {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            Ok(true)
        }
    }

    struct LoadScorer;
    impl WorkerScorer for LoadScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (_candidate, cost) in candidates.iter().zip(costs) {
                *cost = 0.0;
            }
            Ok(())
        }
    }

    let policy = WorkerSelectionPolicy::new_with_filters(
        SelectionConfig::default(),
        "test",
        vec![Box::new(CacheFilter)],
        vec![Box::new(LoadScorer)],
        Box::new(FirstPicker),
    );
    let inputs =
        <WorkerSelectionPolicy as WorkerSelector<TaintedWorkerConfig>>::required_worker_inputs(
            &policy,
        );

    assert!(inputs.contains(WorkerInputs::CACHE));
    assert!(inputs.contains(WorkerInputs::LOAD));
}
