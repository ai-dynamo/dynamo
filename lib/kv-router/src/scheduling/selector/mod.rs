// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    cell::{Cell, OnceCell},
    collections::HashMap,
};

mod policy;
#[cfg(any(test, feature = "bench"))]
mod reference;

#[cfg(any(test, feature = "bench"))]
pub use reference::DefaultWorkerSelector;

// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerFilter, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer,
    WorkerSelectionContext,
};
#[cfg(any(test, feature = "bench"))]
use reference::{DefaultWorkerPicker, DefaultWorkerScorer};

use crate::kv_hints::{
    KvTransferScope, KvTransferSourceChoice, KvTransferSourceSummary, KvTransferTarget,
};
use crate::plugins::worker_selection::{
    CacheSnapshot, CandidateData, WorkerCacheData, WorkerTransferInput,
};
use crate::plugins::worker_selection::{
    RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};
pub use policy::WorkerSelectionPolicy;
use policy::{ComposedPolicyState, WorkerSelectionPolicyStateRef, collect_policy_candidates};
#[cfg(any(test, feature = "bench"))]
use reference::pick_default_worker;

use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};
use crate::protocols::{
    SelectionTrace, SelectionTraceRow, SelectionTraceSpec, WorkerConfigLike, WorkerId,
    WorkerSelectionResult, WorkerWithDpRank,
};

/// Low-level selector used by routing hosts.
///
/// External policies should use [`WorkerSelectionPolicy`].
pub trait WorkerSelector<C: WorkerConfigLike> {
    /// Snapshot runtime parameters, when this selector explicitly supports updates.
    fn runtime_config(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        Err(RuntimePolicyError::Unsupported)
    }

    /// Apply a validated replacement between selections. The scheduler actor serializes calls.
    fn update_runtime_config(
        &mut self,
        _update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        Err(RuntimePolicyError::Unsupported)
    }

    /// Optional worker data required by this selector.
    fn required_worker_inputs(&self) -> WorkerInputs;

    /// Whether an eligible affinity target exclusively constrains worker selection.
    ///
    /// The default selector uses exclusive affinity. Custom policies receive affinity as
    /// advisory context and may choose another eligible worker.
    fn uses_exclusive_affinity_target(&self) -> bool {
        false
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;

    /// Select a worker and, when `trace` is set, also return the per-candidate costs
    /// the policy scored. The default ignores `trace`; policies that can report their
    /// candidates override it. Hosts call this instead of [`Self::select_worker`]
    /// when a selection trace is installed.
    fn select_worker_traced(
        &self,
        input: WorkerSelectionInput<'_, C>,
        _trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        self.select_worker(input).map(|result| (result, None))
    }
}

/// Inputs supplied by the selector's host.
#[derive(Clone, Copy)]
pub enum WorkerSelectionInput<'a, C: WorkerConfigLike> {
    Configured {
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    },
    Hosted {
        worker_ids: &'a [WorkerId],
        occupancy: Option<&'a dyn Fn(WorkerId) -> u64>,
    },
}

pub type ConfiguredSelectionInputs<'a, C> = (
    &'a HashMap<WorkerId, C>,
    &'a SchedulingRequest,
    RoutingEligibility<'a>,
    u32,
);

pub type HostedSelectionInputs<'a> = (&'a [WorkerId], Option<&'a dyn Fn(WorkerId) -> u64>);

impl<'a, C: WorkerConfigLike> WorkerSelectionInput<'a, C> {
    pub fn configured(
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    ) -> Self {
        Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
        }
    }

    pub fn hosted(
        worker_ids: &'a [WorkerId],
        occupancy: Option<&'a dyn Fn(WorkerId) -> u64>,
    ) -> Self {
        Self::Hosted {
            worker_ids,
            occupancy,
        }
    }

    pub fn into_configured(self) -> Result<ConfiguredSelectionInputs<'a, C>, KvSchedulerError> {
        match self {
            Self::Configured {
                workers,
                request,
                eligibility,
                block_size,
            } => Ok((workers, request, eligibility, block_size)),
            Self::Hosted { .. } => Err(WorkerSelectionPolicyError::failed(
                "selector requires configured worker inputs",
            )
            .into()),
        }
    }

    pub fn into_hosted(self) -> Result<HostedSelectionInputs<'a>, KvSchedulerError> {
        match self {
            Self::Hosted {
                worker_ids,
                occupancy,
            } => Ok((worker_ids, occupancy)),
            Self::Configured { .. } => Err(WorkerSelectionPolicyError::failed(
                "selector requires hosted worker inputs",
            )
            .into()),
        }
    }
}

struct MaterializedSelectionInput<'a> {
    request: &'a SchedulingRequest,
    context: WorkerSelectionContext<'a>,
    // Track the maximum over the eligible ranks row() already visits; no second scan.
    max_raw_cached_tokens: Cell<Option<usize>>,
    cache_snapshot: CacheSnapshot<'a>,
    // Built at most once and shared by every row's fetch credit and the
    // selected worker's fetch source, so the two cannot disagree.
    kv_transfer_sources: OnceCell<KvTransferSourceSummary>,
}

impl<'a> MaterializedSelectionInput<'a> {
    fn new(request: &'a SchedulingRequest, block_size: u32) -> Self {
        Self {
            request,
            cache_snapshot: CacheSnapshot {
                shared_hits: request.shared_cache_hits.as_ref(),
                has_tier_matches: !request.overlap.tier_overlap_blocks.device.is_empty()
                    || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
                    || !request.overlap.tier_overlap_blocks.disk.is_empty(),
            },
            context: WorkerSelectionContext {
                request,
                request_blocks: request.request_blocks(block_size),
                block_size,
                track_prefill_tokens: request.track_prefill_tokens,
                pinned_worker: request.pinned_worker,
                router_temperature_override: request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.router_temperature),
            },
            max_raw_cached_tokens: Cell::new(request.mode.is_tracked().then_some(0)),
            kv_transfer_sources: OnceCell::new(),
        }
    }

    /// The request's fetch sources for the targets in `scope`, built on the
    /// first call.
    fn kv_transfer_sources<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        scope: impl FnOnce() -> KvTransferScope,
    ) -> Option<&KvTransferSourceSummary> {
        let candidates = self.request.kv_transfer_candidates.as_ref()?;
        Some(
            self.kv_transfer_sources
                .get_or_init(|| KvTransferSourceSummary::new(candidates, workers, scope())),
        )
    }

    /// Fill every row's fetch input once all rows are built, so the sources
    /// skip anything shorter than the shortest local prefix. `targets` is
    /// index-aligned with `rows`: each hint-capable row's target and local prefix.
    fn fill_kv_transfer_sources<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        rows: &mut [CandidateData],
        targets: &[Option<(KvTransferTarget, usize)>],
    ) {
        debug_assert_eq!(rows.len(), targets.len());
        let scope = || KvTransferScope::of(targets.iter().flatten().copied());
        let Some(sources) = self.kv_transfer_sources(workers, scope) else {
            return;
        };
        for (row, &target) in rows.iter_mut().zip(targets) {
            row.transfer = Some(WorkerTransferInput {
                source_prefix_blocks: target
                    .and_then(|(target, local)| sources.source_for(row.worker, target, local))
                    .map(source_prefix_blocks),
            });
        }
    }

    fn selected_kv_transfer_source<C: WorkerConfigLike>(
        &self,
        workers: &HashMap<WorkerId, C>,
        worker: WorkerWithDpRank,
    ) -> Option<KvTransferSourceChoice> {
        self.request.kv_transfer_candidates.as_ref()?;
        let target = workers
            .get(&worker.worker_id)?
            .kv_transfer_keys()
            .target()?;
        let local_prefix_blocks = self.request.transfer_local_prefix_blocks_for(worker);
        let scope = || KvTransferScope::of([(target, local_prefix_blocks)]);
        self.kv_transfer_sources(workers, scope)?
            .source_for(worker, target, local_prefix_blocks)
    }

    fn max_raw_cached_tokens(&self) -> Option<usize> {
        self.max_raw_cached_tokens.get()
    }

    /// Count a worker toward the best eligible cached prefix once every policy
    /// filter has kept it.
    fn track_kept_candidate(&self, worker: WorkerWithDpRank) {
        if let Some(current_max) = self.max_raw_cached_tokens.get() {
            let raw_cached_tokens = self
                .request
                .raw_cached_tokens_for(worker, self.context.block_size);
            self.max_raw_cached_tokens
                .set(Some(current_max.max(raw_cached_tokens)));
        }
    }

    #[inline(always)]
    fn row(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
    ) -> CandidateData {
        self.row_with_device_overlap(
            worker,
            preferred_taint_multiplier,
            inputs,
            |_, device_overlap_blocks| device_overlap_blocks,
        )
    }

    #[inline(always)]
    fn row_with_device_overlap(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
        select_device_overlap: impl FnOnce(f64, f64) -> f64,
    ) -> CandidateData {
        let cached_tokens = if inputs.contains(WorkerInputs::CACHE) {
            self.request.effective_cached_tokens_for(worker)
        } else {
            0
        };
        let worker_load = if inputs.contains(WorkerInputs::LOAD) {
            self.request.worker_loads.get(&worker).copied()
        } else {
            None
        };
        let cache = if inputs.contains(WorkerInputs::CACHE) {
            let effective_overlap_blocks = self.request.effective_overlap_blocks_for(worker);
            let reported_device_overlap_blocks = self
                .request
                .overlap
                .tier_overlap_blocks
                .device
                .get(&worker)
                .copied()
                .map(|blocks| blocks as f64)
                .unwrap_or(0.0);
            let device_overlap_blocks =
                select_device_overlap(effective_overlap_blocks, reported_device_overlap_blocks);
            WorkerCacheData {
                effective_overlap_blocks,
                estimated_cached_tokens: cached_tokens,
                device_overlap_blocks,
                host_overlap_blocks: self
                    .request
                    .overlap
                    .tier_overlap_blocks
                    .host_pinned
                    .get(&worker)
                    .copied()
                    .unwrap_or(0) as f64,
                disk_overlap_blocks: self
                    .request
                    .overlap
                    .tier_overlap_blocks
                    .disk
                    .get(&worker)
                    .copied()
                    .unwrap_or(0) as f64,
            }
        } else {
            WorkerCacheData::default()
        };
        let load = if inputs.contains(WorkerInputs::LOAD) {
            let available = worker_load.is_some();
            let worker_load = worker_load.unwrap_or_else(|| self.request.worker_load_for(worker));
            WorkerLoadInput {
                available,
                active_prefill_tokens: worker_load.active_prefill_tokens,
                decode_cost_blocks: worker_load.potential_decode_blocks() as f64,
                active_requests: worker_load.active_requests,
            }
        } else {
            WorkerLoadInput::default()
        };

        CandidateData {
            worker,
            inputs,
            cache,
            load,
            transfer: None,
            preferred_taint_multiplier,
        }
    }
}

fn source_prefix_blocks(source: KvTransferSourceChoice) -> u32 {
    u32::try_from(source.prefix_blocks).unwrap_or(u32::MAX)
}

fn selection_result(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
    max_raw_cached_tokens: Option<usize>,
    kv_transfer_source: Option<KvTransferSourceChoice>,
) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: request.request_blocks(block_size),
        effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
        cached_tokens: request.effective_cached_tokens_for(worker),
        max_raw_cached_tokens,
        selected_raw_cached_tokens: max_raw_cached_tokens
            .map(|_| request.raw_cached_tokens_for(worker, block_size)),
        potential_decode_blocks: request
            .potential_decode_blocks_after_admission(worker, block_size),
        kv_transfer_source,
    }
}

/// Copy the retained candidate costs out of the policy's scratch rows in one pass.
///
/// Keeps every candidate when `spec.top_k` is zero or not below the candidate count;
/// otherwise the `top_k` lowest-cost rows plus the selected row when it is not among
/// them. Rows are unordered: this runs on the scheduler actor, so sorting is left to
/// the trace emitter on the routing task.
fn build_selection_trace(
    candidates: &[ScoredWorkerCandidate],
    selected_row: usize,
    pinned: bool,
    spec: SelectionTraceSpec,
) -> SelectionTrace {
    let candidate_count = candidates.len();
    let row_of = |candidate: &ScoredWorkerCandidate| SelectionTraceRow {
        worker: candidate.worker,
        cost: candidate.cost,
    };
    let rows = if spec.top_k == 0 || spec.top_k >= candidate_count {
        candidates.iter().map(row_of).collect()
    } else {
        let mut rows: Vec<SelectionTraceRow> = Vec::with_capacity(spec.top_k + 1);
        // Index into `rows` of the costliest retained row.
        let mut costliest = 0;
        for candidate in candidates {
            if rows.len() < spec.top_k {
                if rows.is_empty() || candidate.cost > rows[costliest].cost {
                    costliest = rows.len();
                }
                rows.push(row_of(candidate));
            } else if candidate.cost < rows[costliest].cost {
                rows[costliest] = row_of(candidate);
                for (index, row) in rows.iter().enumerate() {
                    if row.cost > rows[costliest].cost {
                        costliest = index;
                    }
                }
            }
        }
        let selected = &candidates[selected_row];
        if !rows.iter().any(|row| row.worker == selected.worker) {
            rows.push(row_of(selected));
        }
        rows
    };
    SelectionTrace {
        candidate_count,
        pinned,
        rows,
    }
}

/// Log every scored candidate at TRACE, one row per worker and rank, stamped with
/// `request_id` and `worker_type` because the scheduler actor task has no request span.
fn log_scored_candidates(
    request: &SchedulingRequest,
    worker_type: &'static str,
    candidates: &[ScoredWorkerCandidate],
    selected_row: usize,
) {
    if !tracing::enabled!(tracing::Level::TRACE) {
        return;
    }
    let request_id = request.mode.request_id().unwrap_or("-");
    for (row, candidate) in candidates.iter().enumerate() {
        tracing::trace!(
            request_id,
            worker_type,
            worker_id = candidate.worker.worker_id,
            dp_rank = candidate.worker.dp_rank,
            cost = candidate.cost,
            selected = row == selected_row,
            "Scored worker candidate"
        );
    }
}

fn log_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    worker_type: &'static str,
    cost: f64,
    effective_overlap_blocks: f64,
) {
    let request_id = request.mode.request_id().unwrap_or("-");
    let host_pinned_blocks = request
        .overlap
        .tier_overlap_blocks
        .host_pinned
        .get(&worker)
        .copied()
        .unwrap_or(0);
    let disk_blocks = request
        .overlap
        .tier_overlap_blocks
        .disk
        .get(&worker)
        .copied()
        .unwrap_or(0);

    if request.pinned_worker == Some(worker) {
        tracing::info!(
            request_id,
            "Selected pinned worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
            worker_type,
            worker.worker_id,
            worker.dp_rank,
            cost,
            effective_overlap_blocks,
        );
    } else if worker_type == "decode" {
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit = cost,
            host_pinned_blocks,
            disk_blocks,
            "Selected worker"
        );
    } else {
        let total_kv_blocks = workers
            .get(&worker.worker_id)
            .and_then(WorkerConfigLike::total_kv_blocks);
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit = cost,
            effective_cached_blocks = effective_overlap_blocks,
            host_pinned_blocks,
            disk_blocks,
            total_kv_blocks = ?total_kv_blocks,
            "Selected worker"
        );
    }
}

#[inline(always)]
// DefaultWorkerSelector and SelectionService both converge here. Only the scorer/picker stage is
// dispatched; eligibility outcomes and result construction stay host-owned and shared.
fn select_worker_with_policy<C: WorkerConfigLike>(
    worker_type: &'static str,
    state: WorkerSelectionPolicyStateRef<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    block_size: u32,
    trace: Option<SelectionTraceSpec>,
) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
    eligibility.validate_pinned_worker_allowed()?;

    if let Some(worker) = eligibility.pinned_worker() {
        match eligibility.validate_worker_rank(workers, worker) {
            Ok(_) => {}
            Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                return Err(KvSchedulerError::PinnedWorkerOverloaded {
                    worker_id: worker.worker_id,
                });
            }
            Err(_) => return Err(KvSchedulerError::NoEndpoints),
        }
    }

    let mut input = MaterializedSelectionInput::new(request, block_size);
    input.context.pinned_worker = eligibility.pinned_worker();
    let selected = match state {
        #[cfg(any(test, feature = "bench"))]
        WorkerSelectionPolicyStateRef::Reference(kv_router_config, picker) => {
            let scorer = DefaultWorkerScorer {
                kv_router_config,
                worker_type,
            };
            pick_default_worker(&scorer, picker, &input, workers, request, eligibility)
                .map(|(worker, cost)| (worker, cost, None))
        }
        WorkerSelectionPolicyStateRef::Composed(state) => {
            let mut state = state.borrow_mut();
            let has_eligible_worker =
                collect_policy_candidates(&mut state, &input, workers, request, eligibility)?;
            let ComposedPolicyState {
                picker,
                picker_inputs,
                candidates,
                cache_inputs,
                load_inputs,
                ..
            } = &mut *state;
            if candidates.is_empty() {
                if has_eligible_worker {
                    return Err(KvSchedulerError::AllEligibleWorkersFiltered);
                }
                None
            } else {
                debug_assert!(
                    !picker_inputs.contains(WorkerInputs::CACHE)
                        || cache_inputs.len() == candidates.len()
                );
                debug_assert!(
                    !picker_inputs.contains(WorkerInputs::LOAD)
                        || load_inputs.len() == candidates.len()
                );
                let picker_input = WorkerInputView {
                    candidates,
                    cache: picker_inputs.contains(WorkerInputs::CACHE).then_some(
                        WorkerCacheInputs {
                            rows: cache_inputs,
                            snapshot: &input.cache_snapshot,
                        },
                    ),
                    load: picker_inputs
                        .contains(WorkerInputs::LOAD)
                        .then_some(load_inputs.as_slice()),
                };
                let row = picker.pick(&input.context, picker_input)?;
                let Some(candidate) = candidates.get(row) else {
                    return Err(WorkerSelectionPolicyError::InvalidPickerRow {
                        row,
                        candidate_count: candidates.len(),
                    }
                    .into());
                };
                log_scored_candidates(request, worker_type, candidates, row);
                let trace = trace.map(|spec| {
                    build_selection_trace(
                        candidates,
                        row,
                        input.context.pinned_worker.is_some(),
                        spec,
                    )
                });
                Some((candidate.worker, candidate.cost, trace))
            }
        }
    };
    let Some((worker, cost, selection_trace)) = selected else {
        if eligibility.has_eligible_worker_ignoring_overload(
            workers
                .iter()
                .map(|(&worker_id, config)| (worker_id, config)),
        ) {
            return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
        }
        return Err(KvSchedulerError::NoEndpoints);
    };
    let kv_transfer_source = input.selected_kv_transfer_source(workers, worker);
    let result = selection_result(
        request,
        worker,
        block_size,
        input.max_raw_cached_tokens(),
        kv_transfer_source,
    );
    log_selection(
        workers,
        request,
        worker,
        worker_type,
        cost,
        result.effective_overlap_blocks,
    );
    Ok((result, selection_trace))
}

#[cfg(test)]
mod selection_trace_tests {
    use super::*;

    fn candidates(costs: &[f64]) -> Vec<ScoredWorkerCandidate> {
        costs
            .iter()
            .enumerate()
            .map(|(id, cost)| ScoredWorkerCandidate {
                worker: WorkerWithDpRank::from_worker_id(id as u64),
                cost: *cost,
                preferred_taint_multiplier: None,
            })
            .collect()
    }

    fn worker_ids_by_cost(trace: &SelectionTrace) -> Vec<u64> {
        let mut rows = trace.rows.clone();
        rows.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        rows.iter().map(|row| row.worker.worker_id).collect()
    }

    #[test]
    fn zero_top_k_keeps_every_candidate() {
        let rows = candidates(&[3.0, 1.0, 2.0]);
        let trace = build_selection_trace(&rows, 2, false, SelectionTraceSpec { top_k: 0 });
        assert_eq!(trace.candidate_count, 3);
        assert_eq!(worker_ids_by_cost(&trace), vec![1, 2, 0]);
        assert!(!trace.pinned);
    }

    #[test]
    fn top_k_keeps_lowest_costs_and_appends_selected_row() {
        let rows = candidates(&[5.0, 1.0, 4.0, 2.0, 3.0, 0.5]);
        // Selected row 0 costs 5.0, the worst candidate: it is appended after the three cheapest.
        let trace = build_selection_trace(&rows, 0, true, SelectionTraceSpec { top_k: 3 });
        assert_eq!(trace.candidate_count, 6);
        assert_eq!(worker_ids_by_cost(&trace), vec![5, 1, 3, 0]);
        assert!(trace.pinned);

        // Selected row already among the cheapest: nothing is appended.
        let trace = build_selection_trace(&rows, 1, false, SelectionTraceSpec { top_k: 3 });
        assert_eq!(worker_ids_by_cost(&trace), vec![5, 1, 3]);
    }

    #[test]
    fn top_k_at_or_above_candidate_count_keeps_everything() {
        let rows = candidates(&[2.0, 1.0]);
        let trace = build_selection_trace(&rows, 0, false, SelectionTraceSpec { top_k: 8 });
        assert_eq!(worker_ids_by_cost(&trace), vec![1, 0]);
    }

    #[test]
    fn top_k_pass_tracks_the_costliest_retained_row() {
        // Descending costs force a replacement on every candidate after the first k.
        let rows = candidates(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]);
        let trace = build_selection_trace(&rows, 6, false, SelectionTraceSpec { top_k: 2 });
        assert_eq!(worker_ids_by_cost(&trace), vec![6, 5]);
    }
}

#[cfg(test)]
mod test_support {
    use std::collections::HashSet;

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    #[derive(Clone, Default)]
    pub(super) struct TaintedWorkerConfig {
        pub(super) taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorkerConfig {
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
            &self.taints
        }
    }

    pub(super) fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: Default::default(),
                effective_cached_tokens: Default::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    pub(super) fn worker_loads_with_active_decode(
        decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    ) -> FxHashMap<WorkerWithDpRank, crate::sequences::WorkerLoadProjection> {
        decode_blocks
            .into_iter()
            .map(|(worker, active_decode_blocks)| {
                (
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod cache_reuse_tests {
    use super::test_support::base_request;
    use super::*;
    use crate::test_utils::SimpleWorkerConfig;

    #[test]
    fn tracked_requests_report_raw_cache_reuse() {
        let mut request = base_request(128);
        request.mode = crate::scheduling::ScheduleMode::Tracked {
            request_id: "test".into(),
        };
        let first = WorkerWithDpRank::from_worker_id(1);
        let second = WorkerWithDpRank::from_worker_id(2);
        request.overlap.effective_cached_tokens.insert(first, 120);
        request.overlap.effective_cached_tokens.insert(second, 16);
        request.overlap.tier_overlap_blocks.device.insert(first, 2);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(first, 1);
        request.overlap.tier_overlap_blocks.device.insert(second, 4);
        request.overlap.tier_overlap_blocks.disk.insert(second, 2);
        let input = MaterializedSelectionInput::new(&request, 16);

        input.track_kept_candidate(first);
        input.track_kept_candidate(second);

        assert_eq!(input.max_raw_cached_tokens(), Some(96));
        let result = selection_result(&request, first, 16, input.max_raw_cached_tokens(), None);
        assert_eq!(result.max_raw_cached_tokens, Some(96));
        assert_eq!(result.selected_raw_cached_tokens, Some(48));
        assert_eq!(result.cached_tokens, 120);
    }

    #[test]
    fn query_only_requests_skip_cache_reuse_tracking() {
        let mut request = base_request(128);
        request.mode = crate::scheduling::ScheduleMode::QueryOnly { request_id: None };
        let worker = WorkerWithDpRank::from_worker_id(1);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        let input = MaterializedSelectionInput::new(&request, 16);

        input.track_kept_candidate(worker);

        assert_eq!(input.max_raw_cached_tokens(), None);
        let result = selection_result(&request, worker, 16, input.max_raw_cached_tokens(), None);
        assert_eq!(result.max_raw_cached_tokens, None);
        assert_eq!(result.selected_raw_cached_tokens, None);
    }

    #[test]
    fn raw_overlap_is_specific_to_the_selected_dp_rank() {
        let mut request = base_request(128);
        let selected = WorkerWithDpRank::new(1, 0);
        let other_rank = WorkerWithDpRank::new(1, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(selected, 2);
        request.overlap.tier_overlap_blocks.disk.insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(other_rank, 7);
        request.overlap.effective_cached_tokens.insert(selected, 3);
        assert_eq!(request.raw_cached_tokens_for(selected, 16), 64);
        assert_eq!(request.raw_cached_tokens_for(other_rank, 16), 112);
        let result = selection_result(&request, selected, 16, Some(112), None);
        assert_eq!(result.selected_raw_cached_tokens, Some(64));
    }

    #[test]
    fn cache_reuse_tracking_excludes_out_of_range_dp_rank() {
        let workers = HashMap::from([(1, SimpleWorkerConfig::default())]);
        let mut request = base_request(128);
        request.mode = crate::scheduling::ScheduleMode::Tracked {
            request_id: "test".into(),
        };
        let invalid_rank = WorkerWithDpRank::new(1, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(invalid_rank, 6);
        let input = MaterializedSelectionInput::new(&request, 16);
        let eligibility = request.eligibility();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| {
            input.track_kept_candidate(worker);
        });

        assert!(
            eligibility
                .validate_worker_rank(&workers, invalid_rank)
                .is_err()
        );
        assert_eq!(input.max_raw_cached_tokens(), Some(0));
    }
}
