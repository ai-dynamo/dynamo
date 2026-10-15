// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang EP-group worker selection.
//!
//! SGLang registers every data-parallel rank in one engine under the same worker ID. This policy
//! treats that worker ID as one expert-parallel group and prefers the admission that leaves the
//! group's DP ranks with the smallest request-count and active-token spreads. A smaller secondary
//! term keeps whole EP groups from diverging when the within-group costs are similar.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

pub const POLICY_TYPE: &str = "dynamo-sglang-ep-balance";

const DEFAULT_REQUEST_IMBALANCE_WEIGHT: f64 = 0.0;
const DEFAULT_TOKEN_IMBALANCE_WEIGHT: f64 = 1.0;
const DEFAULT_GROUP_REQUEST_WEIGHT: f64 = 1.0;
const DEFAULT_GROUP_TOKEN_WEIGHT: f64 = 0.0;
const DEFAULT_GROUP_LOAD_TOLERANCE: f64 = 0.1;
const DEFAULT_INCOMPLETE_GROUP_WEIGHT: f64 = 1.0;
const DEFAULT_CACHE_AFFINITY_MIN_FRACTION: f64 = 0.5;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Within-group terms: spread of requests / KV blocks across the chosen group's DP ranks.
    /// Requests default to 0: MoE time is per group, attention time per rank follows context.
    request_imbalance_weight: f64,
    token_imbalance_weight: f64,
    /// Group terms: mean requests / KV blocks per rank of an EP group (SGLang worker).
    group_request_weight: f64,
    group_token_weight: f64,
    /// Groups within this relative distance of the least-loaded group are all candidates;
    /// the within-group terms decide among them.
    group_load_tolerance: f64,
    incomplete_group_weight: f64,
    /// Ranks holding at least this fraction of the prompt in device cache keep the request
    /// (sticky sessions); balancing only decides among them. Values above 1.0 disable the gate.
    cache_affinity_min_fraction: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            request_imbalance_weight: DEFAULT_REQUEST_IMBALANCE_WEIGHT,
            token_imbalance_weight: DEFAULT_TOKEN_IMBALANCE_WEIGHT,
            group_request_weight: DEFAULT_GROUP_REQUEST_WEIGHT,
            group_token_weight: DEFAULT_GROUP_TOKEN_WEIGHT,
            group_load_tolerance: DEFAULT_GROUP_LOAD_TOLERANCE,
            incomplete_group_weight: DEFAULT_INCOMPLETE_GROUP_WEIGHT,
            cache_affinity_min_fraction: DEFAULT_CACHE_AFFINITY_MIN_FRACTION,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [
            ("request_imbalance_weight", self.request_imbalance_weight),
            ("token_imbalance_weight", self.token_imbalance_weight),
            ("group_request_weight", self.group_request_weight),
            ("group_token_weight", self.group_token_weight),
            ("group_load_tolerance", self.group_load_tolerance),
            ("incomplete_group_weight", self.incomplete_group_weight),
            (
                "cache_affinity_min_fraction",
                self.cache_affinity_min_fraction,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be a finite non-negative number"
                )));
            }
        }
        if self.request_imbalance_weight == 0.0 && self.token_imbalance_weight == 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "at least one within-group imbalance weight must be positive",
            ));
        }
        if self.group_request_weight == 0.0 && self.group_token_weight == 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "at least one group load weight must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct GroupStats {
    worker_id: u64,
    size: usize,
    sum_requests: usize,
    sum_blocks: usize,
    has_allowed_row: bool,
}

/// Projected per-rank (requests, context blocks) after admitting the request on
/// `selected_row`. Context blocks are the per-request sequence blocks the attention kernels
/// read, so the incoming request adds its whole prompt regardless of cache hits. A rank the
/// host reports as unavailable has no tracked sequences and counts as idle.
fn projected_load(
    load: &WorkerLoadInput,
    row: usize,
    selected_row: usize,
    request_blocks: usize,
) -> (usize, usize) {
    let selected = row == selected_row;
    if !load.is_available() {
        return (
            usize::from(selected),
            if selected { request_blocks } else { 0 },
        );
    }
    (
        load.active_requests() + usize::from(selected),
        load.active_sequence_blocks() + if selected { request_blocks } else { 0 },
    )
}

/// The selected rank's own load after admitting the request, scaled by the global means so
/// request and block terms are unitless. Picking the minimum is "least-loaded rank in the
/// group"; unlike a max−min spread it is never flat, so 16 ranks do not tie onto one of them.
fn within_group_cost(
    parameters: &Parameters,
    load: &[WorkerLoadInput],
    selected_row: usize,
    request_blocks: usize,
    global_mean_requests: f64,
    global_mean_blocks: f64,
) -> f64 {
    let (requests, blocks) = projected_load(
        &load[selected_row],
        selected_row,
        selected_row,
        request_blocks,
    );
    parameters.request_imbalance_weight * requests as f64 / (global_mean_requests + 1.0)
        + parameters.token_imbalance_weight * blocks as f64 / (global_mean_blocks + 1.0)
}

fn group_load(
    parameters: &Parameters,
    group: &GroupStats,
    max_group_size: usize,
    global_mean_requests: f64,
    global_mean_blocks: f64,
) -> f64 {
    let size = group.size as f64;
    parameters.group_request_weight * (group.sum_requests as f64 / size)
        / (global_mean_requests + 1.0)
        + parameters.group_token_weight * (group.sum_blocks as f64 / size)
            / (global_mean_blocks + 1.0)
        + parameters.incomplete_group_weight * max_group_size.saturating_sub(group.size) as f64
}

/// Sticky cached prefixes first, then the least-loaded EP group (within tolerance), then the
/// rank that leaves that group's DP ranks most even.
fn pick_row(
    parameters: &Parameters,
    candidates: &[dynamo_kv_router::ScoredWorkerCandidate],
    load: &[WorkerLoadInput],
    overlap_blocks: &[f64],
    request_blocks: u64,
) -> Option<usize> {
    if candidates.is_empty()
        || candidates.len() != load.len()
        || candidates.len() != overlap_blocks.len()
    {
        return None;
    }

    let affinity_threshold = parameters.cache_affinity_min_fraction * request_blocks as f64;
    let is_affine = |row: usize| {
        request_blocks > 0 && overlap_blocks[row] > 0.0 && overlap_blocks[row] >= affinity_threshold
    };
    let has_affine_row = (0..candidates.len()).any(is_affine);
    let allowed = |row: usize| !has_affine_row || is_affine(row);

    let mut groups: Vec<GroupStats> = Vec::new();
    let (mut sum_requests, mut sum_blocks) = (0usize, 0usize);
    for (row, (candidate, load)) in candidates.iter().zip(load).enumerate() {
        let worker_id = candidate.worker().worker_id;
        let (requests, blocks) = projected_load(load, row, usize::MAX, 0);
        sum_requests += requests;
        sum_blocks += blocks;
        let group = match groups.iter_mut().find(|g| g.worker_id == worker_id) {
            Some(group) => group,
            None => {
                groups.push(GroupStats {
                    worker_id,
                    size: 0,
                    sum_requests: 0,
                    sum_blocks: 0,
                    has_allowed_row: false,
                });
                groups.last_mut().expect("just pushed")
            }
        };
        group.size += 1;
        group.sum_requests += requests;
        group.sum_blocks += blocks;
        group.has_allowed_row |= allowed(row);
    }
    let global_mean_requests = sum_requests as f64 / candidates.len() as f64;
    let global_mean_blocks = sum_blocks as f64 / candidates.len() as f64;
    let max_group_size = groups.iter().map(|g| g.size).max()?;
    let loads: Vec<(u64, f64)> = groups
        .iter()
        .filter(|g| g.has_allowed_row)
        .map(|g| {
            (
                g.worker_id,
                group_load(
                    parameters,
                    g,
                    max_group_size,
                    global_mean_requests,
                    global_mean_blocks,
                ),
            )
        })
        .collect();
    let min_load = loads.iter().map(|(_, l)| *l).fold(f64::INFINITY, f64::min);
    let load_cutoff = min_load * (1.0 + parameters.group_load_tolerance);
    let load_of = |worker_id: u64| {
        loads
            .iter()
            .find(|(id, _)| *id == worker_id)
            .map(|(_, l)| *l)
    };

    candidates
        .iter()
        .enumerate()
        .filter(|(row, _)| allowed(*row))
        .filter_map(|(row, candidate)| {
            let group_load = load_of(candidate.worker().worker_id)?;
            (group_load <= load_cutoff).then_some((row, group_load, candidate.worker()))
        })
        .map(|(row, group_load, worker)| {
            let cost = within_group_cost(
                parameters,
                load,
                row,
                request_blocks as usize,
                global_mean_requests,
                global_mean_blocks,
            );
            (row, cost, group_load, overlap_blocks[row], worker)
        })
        .min_by(|(_, lc, ll, lo, lw), (_, rc, rl, ro, rw)| {
            lc.total_cmp(rc)
                .then_with(|| ll.total_cmp(rl))
                .then_with(|| ro.total_cmp(lo))
                .then_with(|| lw.cmp(rw))
        })
        .map(|(row, _, _, _, _)| row)
}

struct SglangEpBalancePicker {
    parameters: Parameters,
}

impl WorkerPicker for SglangEpBalancePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD | WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let overlap_blocks: Vec<f64> = cache.iter().map(|c| c.device_overlap_blocks()).collect();
        pick_row(
            &self.parameters,
            input.candidates(),
            load,
            &overlap_blocks,
            context.request_blocks(),
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker with load data"))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SglangEpBalancePicker { parameters }),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
    use dynamo_kv_router::{WorkerLoadProjection, WorkerSelectionInput, WorkerSelector};

    use super::*;

    const BLOCK_SIZE: u32 = 16;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            2
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(4096)
        }
    }

    fn select(groups: &[[(usize, usize, usize); 2]], parameters: Parameters) -> WorkerWithDpRank {
        select_with_overlap(groups, &[], parameters)
    }

    fn select_with_overlap(
        groups: &[[(usize, usize, usize); 2]],
        device_overlap: &[(u64, u32, usize)],
        parameters: Parameters,
    ) -> WorkerWithDpRank {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: BLOCK_SIZE as usize,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        let workers = groups
            .iter()
            .enumerate()
            .map(|(worker_id, ranks)| {
                for (dp_rank, (active_requests, active_blocks, additional_blocks)) in
                    ranks.iter().copied().enumerate()
                {
                    request.worker_loads.insert(
                        WorkerWithDpRank::new(worker_id as u64, dp_rank as u32),
                        WorkerLoadProjection {
                            active_requests,
                            active_decode_blocks: active_blocks,
                            active_sequence_blocks: active_blocks,
                            additional_active_blocks: additional_blocks,
                            ..Default::default()
                        },
                    );
                }
                (worker_id as u64, TestWorker)
            })
            .collect::<HashMap<_, _>>();
        for (worker_id, dp_rank, blocks) in device_overlap.iter().copied() {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(WorkerWithDpRank::new(worker_id, dp_rank), blocks);
        }

        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(SglangEpBalancePicker { parameters }),
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
    }

    #[test]
    fn least_loaded_group_wins_then_its_least_loaded_rank() {
        let selected = select(
            &[[(5, 50, 1), (4, 40, 1)], [(1, 10, 1), (0, 5, 1)]],
            Parameters::default(),
        );
        assert_eq!(selected, WorkerWithDpRank::new(1, 1));
    }

    #[test]
    fn within_group_balance_decides_among_groups_inside_tolerance() {
        let selected = select(
            &[[(9, 90, 1), (8, 80, 1)], [(9, 90, 1), (9, 90, 1)]],
            Parameters::default(),
        );
        assert_eq!(selected, WorkerWithDpRank::new(0, 1));
    }

    #[test]
    fn a_balanced_but_saturated_group_does_not_attract_new_requests() {
        let selected = select(
            &[[(50, 500, 1), (50, 500, 1)], [(10, 100, 1), (5, 50, 1)]],
            Parameters::default(),
        );
        assert_eq!(selected, WorkerWithDpRank::new(1, 1));
    }

    #[test]
    fn active_tokens_break_an_equal_request_balance() {
        let selected = select(&[[(2, 20, 4), (2, 4, 4)]], Parameters::default());
        assert_eq!(selected, WorkerWithDpRank::new(0, 1));
    }

    #[test]
    fn cached_prefix_keeps_a_session_on_its_rank() {
        let selected = select_with_overlap(
            &[[(9, 90, 1), (0, 0, 1)], [(0, 0, 1), (0, 0, 1)]],
            &[(0, 0, 1)],
            Parameters::default(),
        );
        assert_eq!(selected, WorkerWithDpRank::new(0, 0));

        let below_threshold = select_with_overlap(
            &[[(9, 90, 1), (0, 0, 1)], [(0, 0, 1), (0, 0, 1)]],
            &[(0, 0, 1)],
            Parameters {
                cache_affinity_min_fraction: 2.0,
                ..Default::default()
            },
        );
        assert_ne!(below_threshold, WorkerWithDpRank::new(0, 0));
    }

    #[test]
    fn whole_group_load_breaks_an_equal_within_group_cost() {
        let selected = select(
            &[[(4, 20, 1), (4, 20, 1)], [(1, 5, 1), (1, 5, 1)]],
            Parameters::default(),
        );
        assert_eq!(selected.worker_id, 1);
    }

    #[test]
    fn rejects_invalid_parameters() {
        let invalid = Parameters {
            request_imbalance_weight: 0.0,
            token_imbalance_weight: 0.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
        assert!(
            Parameters {
                group_request_weight: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                group_request_weight: 0.0,
                group_token_weight: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
