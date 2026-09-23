// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

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

use crate::plugins::worker_selection::{CacheSnapshot, CandidateData, WorkerCacheData};
pub use policy::WorkerSelectionPolicy;
use policy::{ComposedPolicyState, WorkerSelectionPolicyStateRef, collect_policy_candidates};
#[cfg(any(test, feature = "bench"))]
use reference::pick_default_worker;

use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// Low-level selector used by routing hosts.
///
/// External policies should use [`WorkerSelectionPolicy`].
pub trait WorkerSelector<C: WorkerConfigLike> {
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

/// Host-owned accounting for the DP-rank candidate universe. These fields are
/// policy-neutral and stay meaningful when a custom policy is installed.
#[derive(Debug, Clone, Copy, Default)]
struct CandidateFilterSummary {
    eligible: usize,
    not_allowed: usize,
    constraints: usize,
    overloaded: usize,
    unavailable: usize,
}

impl CandidateFilterSummary {
    fn count_worker<C: WorkerConfigLike>(
        &mut self,
        worker_id: WorkerId,
        config: &C,
        eligibility: RoutingEligibility<'_>,
    ) {
        let replicas = config.data_parallel_size() as usize;
        if !eligibility.caller_allows_worker_id(worker_id) {
            self.not_allowed += replicas;
        } else if !eligibility.is_worker_available(worker_id) {
            self.unavailable += replicas;
        } else if !eligibility.allows_worker_ignoring_overload(worker_id, config) {
            self.constraints += replicas;
        } else if eligibility.is_worker_overloaded(worker_id) {
            self.overloaded += replicas;
        } else {
            self.eligible += replicas;
        }
    }

    fn from_eligibility<C: WorkerConfigLike>(
        workers: &HashMap<WorkerId, C>,
        eligibility: RoutingEligibility<'_>,
    ) -> Self {
        let mut summary = Self::default();
        if let Some(worker) = eligibility.pinned_worker() {
            match workers.get(&worker.worker_id) {
                Some(config) => {
                    summary.count_worker(worker.worker_id, config, eligibility);
                    summary.eligible = usize::from(summary.eligible > 0);
                    summary.not_allowed = usize::from(summary.not_allowed > 0);
                    summary.constraints = usize::from(summary.constraints > 0);
                    summary.overloaded = usize::from(summary.overloaded > 0);
                    summary.unavailable = usize::from(summary.unavailable > 0);
                }
                None => summary.unavailable = 1,
            }
            return summary;
        }
        for (&worker_id, config) in workers {
            summary.count_worker(worker_id, config, eligibility);
        }
        summary
    }

    fn filtered_count(self) -> usize {
        self.not_allowed + self.constraints + self.overloaded + self.unavailable
    }
}

/// Bounded decision evidence written to a lifecycle `router.selection` span.
/// Core mode records a stable summary. Investigation mode also records at most
/// four scored candidates and their available raw cache/load inputs.
struct RouterSelectionTelemetry {
    span: tracing::Span,
    include_candidate_details: bool,
}

impl RouterSelectionTelemetry {
    fn record_candidate_envelope(&self, filters: CandidateFilterSummary, candidate_count: usize) {
        // Composed policies can filter host-eligible workers before scoring.
        let policy_filtered = filters.eligible.saturating_sub(candidate_count);
        self.span.record(
            "dynamo.router.candidate.eligible.count",
            candidate_count as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.count",
            (filters.filtered_count() + policy_filtered) as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.not_allowed",
            filters.not_allowed as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.constraints",
            filters.constraints as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.overloaded",
            filters.overloaded as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.unavailable",
            filters.unavailable as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.policy",
            policy_filtered as u64,
        );
    }

    fn record_custom(
        self,
        candidates: &[ScoredWorkerCandidate],
        inputs: &[CandidateData],
        filters: CandidateFilterSummary,
        selected: ScoredWorkerCandidate,
        pool_role: &'static str,
    ) {
        self.span
            .record("dynamo.router.candidate.count", candidates.len() as u64);
        self.record_candidate_envelope(filters, candidates.len());
        self.span.record("dynamo.router.algorithm.id", "composed");
        self.span.record("dynamo.router.algorithm.version", "v1");
        self.span
            .record("dynamo.router.decision.schema", "selection.v1");
        self.span
            .record("dynamo.router.selection.policy", "composed");
        self.span.record("dynamo.router.pool.role", pool_role);
        self.span.record(
            "dynamo.router.selected.worker.id",
            selected.worker.worker_id,
        );
        self.span.record(
            "dynamo.router.selected.dp.rank",
            selected.worker.dp_rank as u64,
        );
        self.span
            .record("dynamo.router.selected.score", selected.cost);
        if let Some(best) = candidates.iter().min_by(|left, right| {
            left.cost
                .total_cmp(&right.cost)
                .then_with(|| left.worker.cmp(&right.worker))
        }) {
            self.span
                .record("dynamo.router.best.worker.id", best.worker.worker_id);
            self.span
                .record("dynamo.router.best.dp.rank", best.worker.dp_rank as u64);
            self.span.record("dynamo.router.best.score", best.cost);
            let runner_up = candidates
                .iter()
                .filter(|candidate| candidate.worker != best.worker)
                .min_by(|left, right| left.cost.total_cmp(&right.cost));
            self.span.record(
                "dynamo.router.best.margin",
                runner_up.map_or(0.0, |candidate| candidate.cost - best.cost),
            );
        }
        if self.include_candidate_details {
            const TOP_K: usize = 4;
            let by_score = |&left: &usize, &right: &usize| {
                candidates[left]
                    .cost
                    .total_cmp(&candidates[right].cost)
                    .then_with(|| candidates[left].worker.cmp(&candidates[right].worker))
            };
            let mut detailed = (0..candidates.len()).collect::<Vec<_>>();
            detailed.sort_unstable_by(by_score);
            detailed.truncate(TOP_K);
            if let Some(selected_row) = candidates
                .iter()
                .position(|candidate| candidate.worker == selected.worker)
                && !detailed.contains(&selected_row)
            {
                detailed.pop();
                detailed.push(selected_row);
                detailed.sort_unstable_by(by_score);
            }
            let details = detailed
                .into_iter()
                .map(|row| {
                    let candidate = candidates[row];
                    let mut detail = serde_json::json!({
                        "worker_id": candidate.worker.worker_id,
                        "dp_rank": candidate.worker.dp_rank,
                        "selected": candidate.worker == selected.worker,
                        "score": candidate.cost,
                    });
                    if let Some(multiplier) = candidate.preferred_taint_multiplier {
                        detail["preferred_taint_multiplier"] = serde_json::json!(multiplier);
                    }
                    if let Some(input) = inputs.get(row) {
                        if input.inputs.contains(WorkerInputs::CACHE) {
                            detail["cached_tokens"] =
                                serde_json::json!(input.cache.estimated_cached_tokens);
                            detail["device_overlap_blocks"] =
                                serde_json::json!(input.cache.device_overlap_blocks);
                            detail["host_overlap_blocks"] =
                                serde_json::json!(input.cache.host_overlap_blocks);
                            detail["disk_overlap_blocks"] =
                                serde_json::json!(input.cache.disk_overlap_blocks);
                        }
                        if input.inputs.contains(WorkerInputs::LOAD) {
                            detail["active_prefill_tokens"] =
                                serde_json::json!(input.load.active_prefill_tokens);
                            detail["decode_cost_blocks"] =
                                serde_json::json!(input.load.decode_cost_blocks);
                            detail["active_requests"] =
                                serde_json::json!(input.load.active_requests);
                        }
                    }
                    detail
                })
                .collect::<Vec<_>>();
            self.span
                .record("dynamo.router.candidates.detail_schema", "selection.v1");
            self.span.record(
                "dynamo.router.candidates.top_k",
                serde_json::to_string(&details).expect("candidate details serialize"),
            );
        }
    }
}

fn current_router_selection_telemetry() -> Option<RouterSelectionTelemetry> {
    #[cfg(feature = "runtime-protocols")]
    {
        use dynamo_runtime::config::{
            env_is_truthy,
            environment_names::lifecycle_tracing::{
                DYN_LIFECYCLE_TRACE_ENABLED, DYN_LIFECYCLE_TRACE_MODE,
            },
        };

        let span = tracing::Span::current();
        let is_router_selection = span.metadata().is_some_and(|metadata| {
            metadata.target() == dynamo_runtime::telemetry::LIFECYCLE_TARGET
                && metadata.name() == "router.selection"
        });
        (env_is_truthy(DYN_LIFECYCLE_TRACE_ENABLED) && is_router_selection).then(|| {
            RouterSelectionTelemetry {
                span,
                include_candidate_details: std::env::var(DYN_LIFECYCLE_TRACE_MODE)
                    .is_ok_and(|mode| mode.trim().eq_ignore_ascii_case("investigation")),
            }
        })
    }
    #[cfg(not(feature = "runtime-protocols"))]
    {
        None
    }
}

struct MaterializedSelectionInput<'a> {
    request: &'a SchedulingRequest,
    context: WorkerSelectionContext<'a>,
    cache_snapshot: CacheSnapshot<'a>,
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
            let worker_load = worker_load.unwrap_or_default();
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
            preferred_taint_multiplier,
        }
    }
}

fn selection_result(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: request.request_blocks(block_size),
        effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
        cached_tokens: request.effective_cached_tokens_for(worker),
        potential_decode_blocks: request
            .potential_decode_blocks_after_admission(worker, block_size),
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
) -> Result<WorkerSelectionResult, KvSchedulerError> {
    assert!(request.isl_tokens > 0);
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
        }
        WorkerSelectionPolicyStateRef::Composed(state) => {
            let mut state = state.borrow_mut();
            let has_eligible_worker =
                collect_policy_candidates(&mut state, &input, workers, request, eligibility)?;
            let ComposedPolicyState {
                picker,
                picker_inputs,
                candidates,
                unscored_candidates,
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
                if let Some(telemetry) = current_router_selection_telemetry() {
                    telemetry.record_custom(
                        candidates,
                        unscored_candidates,
                        CandidateFilterSummary::from_eligibility(workers, eligibility),
                        *candidate,
                        worker_type,
                    );
                }
                Some((candidate.worker, candidate.cost))
            }
        }
    };
    let Some((worker, cost)) = selected else {
        if eligibility.has_eligible_worker_ignoring_overload(
            workers
                .iter()
                .map(|(&worker_id, config)| (worker_id, config)),
        ) {
            return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
        }
        return Err(KvSchedulerError::NoEndpoints);
    };
    let result = selection_result(request, worker, block_size);
    log_selection(
        workers,
        request,
        worker,
        worker_type,
        cost,
        result.effective_overlap_blocks,
    );
    Ok(result)
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
