// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::HashMap;

mod default;
mod policy;

pub use default::DefaultWorkerSelector;

use default::{DefaultWorkerPicker, DefaultWorkerScorer};
pub use policy::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCandidate, WorkerFilter, WorkerInputView,
    WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerRoutingInput, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy,
};

use default::{pick_default_worker, selection_weights};
use policy::{
    CustomWorkerSelectionState, WorkerSelectionPolicyStateRef, collect_custom_candidates,
};

use super::config::KvRouterConfig;
use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// Low-level worker-selection contract used by the scheduler.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
/// External policies should compose [`WorkerScorer`] and [`WorkerPicker`] implementations with
/// [`WorkerSelectionPolicy`] instead of implementing this trait directly.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

struct WorkerSelectionInput<'a> {
    request: &'a SchedulingRequest,
    has_tier_overlap_blocks: bool,
    use_default_cache_fallbacks: bool,
    context: WorkerSelectionContext<'a>,
}

impl<'a> WorkerSelectionInput<'a> {
    fn new<C: WorkerConfigLike>(
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
        weights: LogitWeights,
        inputs: WorkerInputs,
    ) -> Self {
        let min_active_prefill_tokens = if inputs.contains(WorkerInputs::MIN_ACTIVE_PREFILL_TOKENS)
            && request.track_prefill_tokens
            && weights.overlap_score_credit_decay > 0.0
        {
            let mut minimum = usize::MAX;
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                minimum = minimum.min(request.worker_load_for(worker).active_prefill_tokens);
            });
            if minimum == usize::MAX { 0 } else { minimum }
        } else {
            0
        };
        let has_tier_overlap_blocks = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        Self {
            request,
            has_tier_overlap_blocks,
            use_default_cache_fallbacks: inputs.contains(WorkerInputs::DEFAULT_POLICY_CACHE),
            context: WorkerSelectionContext {
                request,
                request_id: request.mode.request_id().unwrap_or("-"),
                request_blocks: request.request_blocks(block_size),
                block_size,
                track_prefill_tokens: request.track_prefill_tokens,
                weights,
                min_active_prefill_tokens,
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
            let device_overlap_blocks = self
                .request
                .overlap
                .tier_overlap_blocks
                .device
                .get(&worker)
                .copied()
                .map(|blocks| blocks as f64)
                .unwrap_or(0.0);
            let shared_beyond = |device_blocks: f64| {
                self.request.shared_cache_hits.as_ref().map_or(0, |hits| {
                    // `hits_beyond` expects the unweighted device prefix depth.
                    hits.hits_beyond(device_blocks.round().max(0.0) as u32)
                })
            };
            let default_device_overlap_blocks = if self.has_tier_overlap_blocks {
                device_overlap_blocks
            } else {
                effective_overlap_blocks
            };
            let (default_shared_beyond_device_blocks, shared_beyond_device_blocks) =
                if self.use_default_cache_fallbacks {
                    (shared_beyond(default_device_overlap_blocks), 0)
                } else {
                    (0, shared_beyond(device_overlap_blocks))
                };
            WorkerCacheInput {
                effective_overlap_blocks,
                default_device_overlap_blocks,
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
                default_shared_beyond_device_blocks,
                shared_beyond_device_blocks,
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
            routing: if inputs.contains(WorkerInputs::ROUTING) {
                WorkerRoutingInput {
                    preferred_taint_multiplier,
                }
            } else {
                WorkerRoutingInput::default()
            },
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
    let (inputs, needs_filtered_baseline) = match &state {
        WorkerSelectionPolicyStateRef::Default(_) => (
            WorkerInputs::ALL
                | WorkerInputs::MIN_ACTIVE_PREFILL_TOKENS
                | WorkerInputs::DEFAULT_POLICY_CACHE,
            false,
        ),
        WorkerSelectionPolicyStateRef::Custom(state) => {
            let state = RefCell::borrow(state);
            let needs_filtered_baseline = !state.filters.is_empty()
                && state
                    .scorer_picker_inputs
                    .contains(WorkerInputs::MIN_ACTIVE_PREFILL_TOKENS)
                && request.track_prefill_tokens
                && weights.overlap_score_credit_decay > 0.0;
            let inputs = state.filter_inputs | state.scorer_picker_inputs;
            let inputs = if needs_filtered_baseline {
                inputs.without(WorkerInputs::MIN_ACTIVE_PREFILL_TOKENS)
            } else {
                inputs
            };
            (inputs, needs_filtered_baseline)
        }
    };
    let mut input =
        WorkerSelectionInput::new(workers, request, eligibility, block_size, weights, inputs);
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
            let has_eligible_worker = collect_custom_candidates(
                &mut state,
                &mut input,
                workers,
                request,
                eligibility,
                needs_filtered_baseline,
            )?;
            let CustomWorkerSelectionState {
                picker,
                picker_inputs,
                candidates,
                cache_inputs,
                load_inputs,
                routing_inputs,
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
                debug_assert!(
                    !picker_inputs.contains(WorkerInputs::ROUTING)
                        || routing_inputs.len() == candidates.len()
                );
                let picker_input = WorkerInputView {
                    candidates,
                    cache: picker_inputs
                        .contains(WorkerInputs::CACHE)
                        .then_some(cache_inputs.as_slice()),
                    load: picker_inputs
                        .contains(WorkerInputs::LOAD)
                        .then_some(load_inputs.as_slice()),
                    routing: picker_inputs
                        .contains(WorkerInputs::ROUTING)
                        .then_some(routing_inputs.as_slice()),
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
