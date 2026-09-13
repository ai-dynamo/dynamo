// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

mod default;
mod policy;

pub use default::DefaultWorkerSelector;

use default::{DefaultWorkerPicker, DefaultWorkerScorer};
pub use policy::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCandidate, WorkerFilter, WorkerInputView,
    WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy,
};

use default::{pick_default_worker, selection_weights};
use policy::{
    CustomWorkerSelectionState, WorkerSelectionPolicyStateRef, collect_custom_candidates,
};

use super::config::KvRouterConfig;
use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};
use crate::protocols::{
    BestOverlapCandidate, WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};

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

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

struct MaterializedSelectionInput<'a> {
    request: &'a SchedulingRequest,
    context: WorkerSelectionContext<'a>,
}

impl<'a> MaterializedSelectionInput<'a> {
    fn new(request: &'a SchedulingRequest, block_size: u32, weights: LogitWeights) -> Self {
        Self {
            request,
            context: WorkerSelectionContext {
                request,
                request_id: request.mode.request_id().unwrap_or("-"),
                request_blocks: request.request_blocks(block_size),
                block_size,
                track_prefill_tokens: request.track_prefill_tokens,
                weights,
                router_temperature_override: request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.router_temperature),
            },
        }
    }

    fn row(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
    ) -> WorkerCandidate {
        self.row_with_device_overlap(
            worker,
            preferred_taint_multiplier,
            inputs,
            |_, device_overlap_blocks| device_overlap_blocks,
        )
    }

    fn row_with_device_overlap(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
        select_device_overlap: impl FnOnce(f64, f64) -> f64,
    ) -> WorkerCandidate {
        let cached_tokens = if inputs.contains(WorkerInputs::CACHE)
            || (inputs.contains(WorkerInputs::LOAD) && self.request.track_prefill_tokens)
        {
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
            let shared_beyond = |device_blocks: f64| {
                self.request.shared_cache_hits.as_ref().map_or(0, |hits| {
                    // `hits_beyond` expects the unweighted device prefix depth.
                    hits.hits_beyond(device_blocks.round().max(0.0) as u32)
                })
            };
            WorkerCacheInput {
                effective_overlap_blocks,
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
                shared_beyond_device_blocks: shared_beyond(device_overlap_blocks),
            }
        } else {
            WorkerCacheInput::default()
        };
        let load = if inputs.contains(WorkerInputs::LOAD) {
            let raw_prefill_tokens = if self.request.track_prefill_tokens {
                match worker_load {
                    Some(load) => {
                        // Preserve the legacy operation order when overlap exceeds the prompt.
                        let uncached_tokens = super::prefill_load::effective_prefill_tokens(
                            self.request.isl_tokens,
                            cached_tokens,
                        );
                        let projected_tokens = load.active_prefill_tokens + uncached_tokens;
                        projected_tokens.saturating_add(cached_tokens)
                    }
                    None => self.request.isl_tokens,
                }
            } else {
                0
            } as f64;
            let worker_load = worker_load.unwrap_or_default();
            WorkerLoadInput {
                raw_prefill_blocks: raw_prefill_tokens / self.context.block_size as f64,
                active_prefill_tokens: worker_load.active_prefill_tokens,
                decode_cost_blocks: worker_load.potential_decode_blocks() as f64,
                active_requests: worker_load.active_requests,
            }
        } else {
            WorkerLoadInput::default()
        };

        WorkerCandidate {
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
    best_overlap: Option<BestOverlapCandidate>,
) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: request.request_blocks(block_size),
        effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
        cached_tokens: request.effective_cached_tokens_for(worker),
        potential_decode_blocks: request
            .potential_decode_blocks_after_admission(worker, block_size),
        best_overlap,
    }
}

/// Best effective KV overlap reachable for `request` across the eligible workers.
///
/// Applies the same eligibility filter and lower-worker-id tie-break as
/// `non_max_overlap_selection`, but reports the best candidate absolutely rather than relative
/// to the selection, so an optimal decision is still scoreable. Workers missing from the
/// overlap map hold no matching blocks, so an empty map yields a zero candidate that the
/// selection trivially matches.
fn best_overlap_candidate<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selected: WorkerWithDpRank,
) -> BestOverlapCandidate {
    let mut best = None;
    for (&worker, &overlap_blocks) in &request.overlap.effective_overlap_blocks {
        if eligibility.validate_worker_rank(workers, worker).is_err() {
            continue;
        }
        let is_better = best.is_none_or(
            |(current_worker, current_overlap): (WorkerWithDpRank, f64)| {
                overlap_blocks > current_overlap
                    || (overlap_blocks == current_overlap && worker < current_worker)
            },
        );
        if is_better {
            best = Some((worker, overlap_blocks));
        }
    }

    let Some((best_worker, effective_overlap_blocks)) = best else {
        return BestOverlapCandidate {
            effective_overlap_blocks: 0.0,
            effective_cached_tokens: 0,
            selected_has_max_overlap: true,
        };
    };
    BestOverlapCandidate {
        effective_overlap_blocks,
        effective_cached_tokens: request.effective_cached_tokens_for(best_worker),
        selected_has_max_overlap: request.effective_overlap_blocks_for(selected)
            >= effective_overlap_blocks,
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
    kv_router_config: &KvRouterConfig,
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

    let weights = selection_weights(kv_router_config, request);
    let input = MaterializedSelectionInput::new(request, block_size, weights);
    let selected = match state {
        WorkerSelectionPolicyStateRef::Default(picker) => {
            let scorer = DefaultWorkerScorer {
                kv_router_config,
                worker_type,
            };
            pick_default_worker(&scorer, picker, &input, workers, request, eligibility)
        }
        WorkerSelectionPolicyStateRef::Custom(state) => {
            let mut state = state.borrow_mut();
            let has_eligible_worker =
                collect_custom_candidates(&mut state, &input, workers, request, eligibility)?;
            let CustomWorkerSelectionState {
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
                    cache: picker_inputs
                        .contains(WorkerInputs::CACHE)
                        .then_some(cache_inputs.as_slice()),
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
    // Pinned selections are excluded: the caller named the worker, so there was no decision to
    // score, and counting one as optimal would inflate the routing-quality ratio.
    let best_overlap = eligibility
        .pinned_worker()
        .is_none()
        .then(|| best_overlap_candidate(workers, request, eligibility, worker));
    let result = selection_result(request, worker, block_size, best_overlap);
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
    use std::collections::{HashMap, HashSet};

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
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
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
mod tests {
    use std::collections::{HashMap, HashSet};

    use rustc_hash::FxHashMap;

    use super::test_support::*;
    use super::*;
    use crate::config::KvRouterConfig;
    use crate::test_utils::SimpleWorkerConfig;

    const BLOCK_SIZE: u32 = 16;

    fn selector() -> DefaultWorkerSelector {
        DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        )
    }

    fn two_workers() -> HashMap<WorkerId, SimpleWorkerConfig> {
        HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ])
    }

    fn with_overlap(
        request: &mut SchedulingRequest,
        worker_id: WorkerId,
        blocks: f64,
        tokens: usize,
    ) {
        let worker = WorkerWithDpRank::from_worker_id(worker_id);
        request
            .overlap
            .effective_overlap_blocks
            .insert(worker, blocks);
        request
            .overlap
            .effective_cached_tokens
            .insert(worker, tokens);
    }

    fn select(
        workers: &HashMap<WorkerId, SimpleWorkerConfig>,
        request: &SchedulingRequest,
        overloaded: Option<&HashSet<WorkerId>>,
    ) -> WorkerSelectionResult {
        selector()
            .select_worker(WorkerSelectionInput::configured(
                workers,
                request,
                request.eligibility_with_overloaded(overloaded),
                BLOCK_SIZE,
            ))
            .expect("selection succeeds")
    }

    /// The discriminating case: the best-overlap worker is eligible and simply loses on cost.
    /// A candidate computed from the selection alone would report this decision as optimal.
    #[test]
    fn best_overlap_reports_the_unselected_maximum() {
        let workers = two_workers();
        let mut request = base_request(64);
        with_overlap(&mut request, 0, 4.0, 64);
        request.worker_loads = worker_loads_with_active_decode(FxHashMap::from_iter([(
            WorkerWithDpRank::from_worker_id(0),
            1_000,
        )]));

        let result = select(&workers, &request, None);

        assert_eq!(
            result.worker.worker_id, 1,
            "decode load must outweigh overlap"
        );
        let best = result
            .best_overlap
            .expect("unpinned selection reports a candidate");
        assert_eq!(best.effective_overlap_blocks, 4.0);
        assert_eq!(best.effective_cached_tokens, 64);
        assert!(!best.selected_has_max_overlap);
    }

    #[test]
    fn best_overlap_marks_a_max_overlap_selection_optimal() {
        let workers = two_workers();
        let mut request = base_request(64);
        with_overlap(&mut request, 0, 4.0, 64);

        let result = select(&workers, &request, None);

        assert_eq!(result.worker.worker_id, 0);
        let best = result
            .best_overlap
            .expect("unpinned selection reports a candidate");
        assert_eq!(best.effective_overlap_blocks, 4.0);
        assert_eq!(best.effective_cached_tokens, 64);
        assert!(best.selected_has_max_overlap);
    }

    /// A tie means no strictly better worker exists, so whichever side wins is optimal.
    #[test]
    fn best_overlap_counts_a_tie_as_optimal() {
        let workers = two_workers();
        let mut request = base_request(64);
        with_overlap(&mut request, 0, 2.0, 32);
        with_overlap(&mut request, 1, 2.0, 32);

        let result = select(&workers, &request, None);

        let best = result
            .best_overlap
            .expect("unpinned selection reports a candidate");
        assert_eq!(best.effective_overlap_blocks, 2.0);
        assert!(best.selected_has_max_overlap);
    }

    /// Overlap the router could not have taken is not overlap it missed.
    #[test]
    fn best_overlap_ignores_ineligible_workers() {
        let workers = two_workers();
        let mut request = base_request(64);
        with_overlap(&mut request, 0, 4.0, 64);
        let overloaded = HashSet::from([0]);

        let result = select(&workers, &request, Some(&overloaded));

        assert_eq!(result.worker.worker_id, 1);
        let best = result
            .best_overlap
            .expect("unpinned selection reports a candidate");
        assert_eq!(best.effective_overlap_blocks, 0.0);
        assert_eq!(best.effective_cached_tokens, 0);
        assert!(best.selected_has_max_overlap);
    }

    #[test]
    fn best_overlap_is_absent_for_pinned_selections() {
        let workers = two_workers();
        let mut request = base_request(64);
        with_overlap(&mut request, 1, 4.0, 64);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));

        let result = select(&workers, &request, None);

        assert_eq!(result.worker.worker_id, 0);
        assert!(
            result.best_overlap.is_none(),
            "the caller chose, not the router"
        );
    }

    #[test]
    fn best_overlap_is_zero_when_no_worker_holds_overlap() {
        let workers = two_workers();
        let request = base_request(64);

        let result = select(&workers, &request, None);

        let best = result
            .best_overlap
            .expect("unpinned selection reports a candidate");
        assert_eq!(best.effective_overlap_blocks, 0.0);
        assert_eq!(best.effective_cached_tokens, 0);
        assert!(best.selected_has_max_overlap);
    }
}
