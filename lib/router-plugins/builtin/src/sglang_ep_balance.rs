// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang EP-group worker selection.
//!
//! SGLang registers every data-parallel rank in one engine under the same worker ID. This policy
//! treats that worker ID as one expert-parallel group: MoE time follows the group's total batch,
//! and attention time follows each rank's own context, so the group waits for its rank with the
//! most context. The policy keeps request counts even across groups and context even across the
//! ranks of a group.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use dynamo_kv_router::{KvRouterConfig, ScoredWorkerCandidate};

pub const POLICY_TYPE: &str = "dynamo-sglang-ep-balance";

/// Group-load penalty per DP rank of a group that is missing from the candidates (filtered out),
/// in units of the global mean requests per rank.
const MISSING_RANK_PENALTY: f64 = 1.0;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Groups whose mean requests per rank is within this relative distance of the least-loaded
    /// group are all candidates; per-rank context decides among them.
    group_load_tolerance: f64,
    /// Ranks holding at least this fraction of the prompt in device cache keep the request;
    /// balancing only decides among them. Values above 1.0 disable the gate.
    cache_affinity_min_fraction: f64,
    /// A session bound to a rank (the host's advisory affinity target) moves only if that lowers
    /// the group's slowest-rank context by more than this many times the request's own context
    /// times the move cost (see [`affinity_row`]). 0 moves whenever another rank is better; a
    /// large value (e.g. 1e9) gives hard stickiness.
    affinity_min_imbalance_reduction: f64,
    /// DP ranks per physical host, numbered contiguously within a worker (ranks
    /// `[k * n, (k + 1) * n)` share host `k`). 0 means unknown: every move costs a full transfer.
    dp_ranks_per_host: u32,
    /// Cost of a move to a rank on the bound rank's host, relative to a cross-host move. Such a
    /// rank reloads the session from the host's shared KV tier (e.g. Mooncake) over the chip-host
    /// link instead of the network (GB200: 50 / 225 GB/s).
    same_host_move_cost: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            group_load_tolerance: 0.1,
            cache_affinity_min_fraction: 0.5,
            affinity_min_imbalance_reduction: 0.5,
            dp_ranks_per_host: 0,
            same_host_move_cost: 50.0 / 225.0,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [
            ("group_load_tolerance", self.group_load_tolerance),
            (
                "cache_affinity_min_fraction",
                self.cache_affinity_min_fraction,
            ),
            (
                "affinity_min_imbalance_reduction",
                self.affinity_min_imbalance_reduction,
            ),
            ("same_host_move_cost", self.same_host_move_cost),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be a finite non-negative number"
                )));
            }
        }
        Ok(())
    }
}

/// One EP group (SGLang worker) among the candidates.
#[derive(Clone, Copy)]
struct Group {
    worker_id: u64,
    ranks: usize,
    requests: usize,
    max_blocks: usize,
    has_allowed_row: bool,
}

/// A rank's active (requests, context blocks). Context blocks are the per-request sequence
/// blocks the attention kernels read. A rank the host reports as unavailable counts as idle.
fn rank_load(load: &WorkerLoadInput) -> (usize, usize) {
    if load.is_available() {
        (load.active_requests(), load.active_sequence_blocks())
    } else {
        (0, 0)
    }
}

/// Selection state shared by the bound-session and the free-placement paths.
struct Candidates<'a> {
    workers: Vec<WorkerWithDpRank>,
    blocks: Vec<usize>,
    /// Index into `groups` for each candidate row.
    group_of: Vec<usize>,
    groups: Vec<Group>,
    total_requests: usize,
    parameters: &'a Parameters,
}

impl<'a> Candidates<'a> {
    fn new(
        parameters: &'a Parameters,
        candidates: &[ScoredWorkerCandidate],
        load: &[WorkerLoadInput],
        allowed: impl Fn(usize) -> bool,
    ) -> Self {
        let mut this = Self {
            workers: Vec::with_capacity(candidates.len()),
            blocks: Vec::with_capacity(candidates.len()),
            group_of: Vec::with_capacity(candidates.len()),
            groups: Vec::new(),
            total_requests: 0,
            parameters,
        };
        for (row, (candidate, load)) in candidates.iter().zip(load).enumerate() {
            let worker = candidate.worker();
            let (requests, blocks) = rank_load(load);
            let index = match this
                .groups
                .iter()
                .position(|g| g.worker_id == worker.worker_id)
            {
                Some(index) => index,
                None => {
                    this.groups.push(Group {
                        worker_id: worker.worker_id,
                        ranks: 0,
                        requests: 0,
                        max_blocks: 0,
                        has_allowed_row: false,
                    });
                    this.groups.len() - 1
                }
            };
            let group = &mut this.groups[index];
            group.ranks += 1;
            group.requests += requests;
            group.max_blocks = group.max_blocks.max(blocks);
            group.has_allowed_row |= allowed(row);
            this.total_requests += requests;
            this.workers.push(worker);
            this.blocks.push(blocks);
            this.group_of.push(index);
        }
        this
    }

    /// Rise in the group's ranks × slowest-rank context when the request lands on `row`. With
    /// attention time linear in context, a group's barrier wait Σ_r(max − att_r) is
    /// ranks·max − Σ_r att_r, and the sum does not depend on placement, so this is the change in
    /// total wait.
    fn barrier_increase(&self, row: usize, request_blocks: usize) -> f64 {
        let group = &self.groups[self.group_of[row]];
        group.ranks as f64
            * (self.blocks[row] + request_blocks).saturating_sub(group.max_blocks) as f64
    }

    /// The bound row, or the row whose saving (slowest-rank context saved, as a fraction of the
    /// request) beats the threshold times its move cost by the most. A huge request makes any
    /// destination the new straggler, so it saves little and stays.
    fn affinity_row(&self, bound_row: usize, request_blocks: usize) -> usize {
        let parameters = self.parameters;
        let bound = self.workers[bound_row];
        let host = |worker: WorkerWithDpRank| {
            (
                worker.worker_id,
                worker.dp_rank / parameters.dp_ranks_per_host,
            )
        };
        let move_cost = |row: usize| {
            if parameters.dp_ranks_per_host > 0 && host(self.workers[row]) == host(bound) {
                parameters.same_host_move_cost
            } else {
                1.0
            }
        };
        let stay = self.barrier_increase(bound_row, request_blocks);
        let scale =
            self.groups[self.group_of[bound_row]].ranks as f64 * request_blocks.max(1) as f64;
        (0..self.workers.len())
            .filter(|row| *row != bound_row)
            .map(|row| {
                let saved = (stay - self.barrier_increase(row, request_blocks)) / scale;
                let cost = move_cost(row);
                // The threshold term is non-negative, so a positive net implies a real saving.
                (
                    row,
                    saved - parameters.affinity_min_imbalance_reduction * cost,
                    cost,
                )
            })
            .filter(|(_, net, _)| *net > 0.0)
            .max_by(|(lr, ln, lc), (rr, rn, rc)| {
                ln.total_cmp(rn)
                    .then_with(|| rc.total_cmp(lc))
                    .then_with(|| self.blocks[*rr].cmp(&self.blocks[*lr]))
            })
            .map_or(bound_row, |(row, ..)| row)
    }

    /// Mean requests per rank, scaled by the global mean, plus a penalty per missing rank.
    fn group_load(&self, group: &Group, max_ranks: usize) -> f64 {
        let global_mean = self.total_requests as f64 / self.workers.len() as f64;
        (group.requests as f64 / group.ranks as f64) / (global_mean + 1.0)
            + MISSING_RANK_PENALTY * max_ranks.saturating_sub(group.ranks) as f64
    }
}

/// Bound session rank unless moving lowers the attention barrier enough; otherwise sticky cached
/// prefixes, then the least-loaded EP groups (within tolerance), then the rank with the least
/// context after admission.
fn pick_row(
    parameters: &Parameters,
    candidates: &[ScoredWorkerCandidate],
    load: &[WorkerLoadInput],
    overlap_blocks: &[f64],
    request_blocks: u64,
    affinity_target: Option<WorkerAffinityTarget>,
) -> Option<usize> {
    if candidates.is_empty()
        || candidates.len() != load.len()
        || candidates.len() != overlap_blocks.len()
    {
        return None;
    }
    let request_blocks_usize = request_blocks as usize;

    let affinity_threshold = parameters.cache_affinity_min_fraction * request_blocks as f64;
    let is_affine = |row: usize| {
        request_blocks > 0 && overlap_blocks[row] > 0.0 && overlap_blocks[row] >= affinity_threshold
    };
    let has_affine_row = (0..candidates.len()).any(is_affine);
    let allowed = |row: usize| !has_affine_row || is_affine(row);
    let state = Candidates::new(parameters, candidates, load, allowed);

    if let Some(target) = affinity_target
        && let Some(bound_row) = state.workers.iter().position(|worker| {
            worker.worker_id == target.worker_id
                && target.dp_rank.is_none_or(|rank| rank == worker.dp_rank)
        })
    {
        return Some(state.affinity_row(bound_row, request_blocks_usize));
    }

    let max_ranks = state.groups.iter().map(|g| g.ranks).max()?;
    let group_loads: Vec<f64> = state
        .groups
        .iter()
        .map(|g| state.group_load(g, max_ranks))
        .collect();
    let min_load = state
        .groups
        .iter()
        .zip(&group_loads)
        .filter(|(g, _)| g.has_allowed_row)
        .map(|(_, l)| *l)
        .fold(f64::INFINITY, f64::min);
    let load_cutoff = min_load * (1.0 + parameters.group_load_tolerance);

    (0..candidates.len())
        .filter(|row| allowed(*row) && group_loads[state.group_of[*row]] <= load_cutoff)
        .min_by(|l, r| {
            let key = |row: usize| {
                (
                    state.blocks[row] + request_blocks_usize,
                    group_loads[state.group_of[row]],
                    overlap_blocks[row],
                    state.workers[row],
                )
            };
            let (lb, ll, lo, lw) = key(*l);
            let (rb, rl, ro, rw) = key(*r);
            lb.cmp(&rb)
                .then_with(|| ll.total_cmp(&rl))
                .then_with(|| ro.total_cmp(&lo))
                .then_with(|| lw.cmp(&rw))
        })
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
            context.affinity_target(),
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
        select_with_affinity(groups, &[], None, 1, parameters)
    }

    fn select_with_affinity(
        groups: &[[(usize, usize, usize); 2]],
        device_overlap: &[(u64, u32, usize)],
        affinity_target: Option<(u64, u32)>,
        request_blocks: usize,
        parameters: Parameters,
    ) -> WorkerWithDpRank {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: request_blocks * BLOCK_SIZE as usize,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: affinity_target
                .map(|(worker_id, dp_rank)| WorkerAffinityTarget::new(worker_id, Some(dp_rank))),
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
        let selected = select_with_affinity(
            &[[(9, 90, 1), (0, 0, 1)], [(0, 0, 1), (0, 0, 1)]],
            &[(0, 0, 1)],
            None,
            1,
            Parameters::default(),
        );
        assert_eq!(selected, WorkerWithDpRank::new(0, 0));

        let below_threshold = select_with_affinity(
            &[[(9, 90, 1), (0, 0, 1)], [(0, 0, 1), (0, 0, 1)]],
            &[(0, 0, 1)],
            None,
            1,
            Parameters {
                cache_affinity_min_fraction: 2.0,
                ..Default::default()
            },
        );
        assert_ne!(below_threshold, WorkerWithDpRank::new(0, 0));
    }

    #[test]
    fn bound_session_moves_only_when_that_lowers_the_barrier() {
        let near_even = [[(4, 40, 1), (4, 40, 1)], [(4, 40, 1), (4, 40, 1)]];
        assert_eq!(
            select_with_affinity(&near_even, &[], Some((0, 1)), 8, Parameters::default()),
            WorkerWithDpRank::new(0, 1)
        );
        // Bound rank is the straggler; the other rank in its group absorbs the turn.
        let skewed = [[(4, 40, 1), (4, 120, 1)], [(4, 100, 1), (4, 100, 1)]];
        assert_eq!(
            select_with_affinity(&skewed, &[], Some((0, 1)), 40, Parameters::default()),
            WorkerWithDpRank::new(0, 0)
        );
        // A huge request makes any destination the new straggler: moving saves ~nothing.
        let huge = [[(4, 60, 1), (4, 40, 1)], [(4, 60, 1), (4, 60, 1)]];
        assert_eq!(
            select_with_affinity(&huge, &[], Some((0, 0)), 1000, Parameters::default()),
            WorkerWithDpRank::new(0, 0)
        );
        // A large threshold restores hard stickiness.
        assert_eq!(
            select_with_affinity(
                &skewed,
                &[],
                Some((0, 1)),
                40,
                Parameters {
                    affinity_min_imbalance_reduction: 1e9,
                    ..Parameters::default()
                }
            ),
            WorkerWithDpRank::new(0, 1)
        );
    }

    #[test]
    fn same_host_moves_are_cheaper() {
        // Bound rank 1 is the straggler.
        let skewed = [[(4, 100, 1), (4, 140, 1)], [(4, 100, 1), (4, 100, 1)]];
        let threshold = Parameters {
            affinity_min_imbalance_reduction: 1.5,
            ..Parameters::default()
        };
        // Saving 40 of a 40-block request (1.0) is below a full-cost threshold of 1.5.
        assert_eq!(
            select_with_affinity(&skewed, &[], Some((0, 1)), 40, threshold),
            WorkerWithDpRank::new(0, 1)
        );
        // With both ranks of worker 0 on one host the move costs 0.22 and goes ahead.
        assert_eq!(
            select_with_affinity(
                &skewed,
                &[],
                Some((0, 1)),
                40,
                Parameters {
                    dp_ranks_per_host: 2,
                    ..threshold
                }
            ),
            WorkerWithDpRank::new(0, 0)
        );
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
        for invalid in [
            Parameters {
                group_load_tolerance: f64::NAN,
                ..Default::default()
            },
            Parameters {
                same_host_move_cost: -1.0,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
