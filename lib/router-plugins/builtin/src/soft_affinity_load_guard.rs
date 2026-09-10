// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load guard for a soft session-affinity binding.
//!
//! Retains the bound worker while it holds at most `max_active_requests`, otherwise moves only to
//! a worker that is less loaded by at least `move_margin` requests. The margin is what keeps a
//! binding still: without it, load that alternates by a single request between workers relocates
//! the binding on every request and walks it across the pool (A -> B -> A -> C), losing the prefix
//! it exists to reuse. A swing larger than the margin still moves it, by design.
//!
//! Pair with `--router-session-affinity-mode soft --router-session-affinity-binding parent-group`
//! to co-locate the subagents of one parent session; the binding knob supplies the shared binding,
//! this policy supplies the load guard. The binding itself is owned by Dynamo's affinity
//! coordinator, which commits it only after a successful dispatch.
//!
//! **Experimental.** A picker sees only the workers eligible for the request in front of it, so it
//! cannot tell a request-local exclusion from a worker that left the pool: one constrained
//! subagent can move a whole group off its warm worker.

use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerAffinityTarget;
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-soft-affinity-load-guard";

/// Must clear ordinary per-worker concurrency or a binding moves on nearly every request. Chosen
/// to do that in practice, not derived from a model or topology.
const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 32;
/// One request of difference is ordinary jitter between workers; two is a real imbalance.
const DEFAULT_MOVE_MARGIN: usize = 2;

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional. Unknown keys are rejected at startup rather than ignored, so a
/// misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Active requests the bound worker may already hold before the binding may move, counting
    /// every request that worker serves. Compared inclusively.
    max_active_requests: usize,
    /// How many fewer active requests an alternative must hold before the binding moves to it.
    /// `1` moves on any strictly lower load; `0` is rejected because it allows equal-load moves.
    move_margin: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            max_active_requests: DEFAULT_MAX_ACTIVE_REQUESTS,
            move_margin: DEFAULT_MOVE_MARGIN,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.move_margin == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "move_margin must be at least 1",
            ));
        }
        Ok(())
    }
}

struct SoftAffinityLoadGuardPicker {
    parameters: Parameters,
}

impl SoftAffinityLoadGuardPicker {
    fn matches_target(candidate: &ScoredWorkerCandidate, target: WorkerAffinityTarget) -> bool {
        let worker = candidate.worker();
        worker.worker_id == target.worker_id
            && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
    }

    fn least_loaded_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        excluded_target: Option<WorkerAffinityTarget>,
    ) -> Option<usize> {
        candidates
            .iter()
            .zip(loads)
            .enumerate()
            .filter(|(_, (candidate, _))| {
                excluded_target.is_none_or(|target| !Self::matches_target(candidate, target))
            })
            .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
            .map(|(row, _)| row)
    }

    fn target_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        target: WorkerAffinityTarget,
    ) -> Option<usize> {
        candidates
            .iter()
            .zip(loads)
            .enumerate()
            .filter(|(_, (candidate, _))| Self::matches_target(candidate, target))
            .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
            .map(|(row, _)| row)
    }
}

impl WorkerPicker for SoftAffinityLoadGuardPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;

        if let Some(target) = context.affinity_target()
            && let Some(target_row) = Self::target_row(candidates, loads, target)
        {
            let target_load = loads[target_row].active_requests();
            if target_load <= self.parameters.max_active_requests {
                return Ok(target_row);
            }
            return Ok(Self::least_loaded_row(candidates, loads, Some(target))
                .filter(|&row| {
                    target_load.saturating_sub(loads[row].active_requests())
                        >= self.parameters.move_margin
                })
                .unwrap_or(target_row));
        }

        Self::least_loaded_row(candidates, loads, None)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
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
                Box::new(SoftAffinityLoadGuardPicker { parameters }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
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
            Some(1024)
        }
    }

    fn request(affinity_target: Option<WorkerAffinityTarget>) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target,
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
        }
    }

    fn set_active_requests(
        request: &mut SchedulingRequest,
        worker: WorkerWithDpRank,
        active_requests: usize,
    ) {
        request.worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_requests,
                ..Default::default()
            },
        );
    }

    fn policy(max_active_requests: usize) -> WorkerSelectionPolicy {
        policy_with_margin(max_active_requests, 1)
    }

    fn policy_with_margin(max_active_requests: usize, move_margin: usize) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(SoftAffinityLoadGuardPicker {
                parameters: Parameters {
                    max_active_requests,
                    move_margin,
                },
            }),
        )
    }

    fn select(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        request: &SchedulingRequest,
    ) -> WorkerWithDpRank {
        policy
            .select_worker(WorkerSelectionInput::configured(
                workers,
                request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
    }

    // Worker 29 wins every load tie because selection breaks ties on the lower worker id, so a
    // selection of worker 41 under a tie proves the affinity target was honored.
    fn workers() -> HashMap<u64, TestWorker> {
        HashMap::from([(29, TestWorker), (41, TestWorker)])
    }

    #[test]
    fn places_an_unbound_request_on_the_least_loaded_worker() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut unbound = request(None);
        set_active_requests(&mut unbound, worker_a, 2);
        set_active_requests(&mut unbound, worker_b, 0);

        assert_eq!(select(&policy(0), &workers(), &unbound), worker_b);
    }

    #[test]
    fn retains_the_target_at_or_below_the_threshold() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut bound = request(Some(worker_b.into()));
        set_active_requests(&mut bound, worker_a, 0);
        set_active_requests(&mut bound, worker_b, 4);

        assert_eq!(select(&policy(4), &workers(), &bound), worker_b);
    }

    #[test]
    fn moves_off_a_target_over_the_threshold() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut bound = request(Some(worker_b.into()));
        set_active_requests(&mut bound, worker_a, 0);
        set_active_requests(&mut bound, worker_b, 5);

        assert_eq!(select(&policy(4), &workers(), &bound), worker_a);
    }

    #[test]
    fn does_not_move_to_an_equally_loaded_worker() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut bound = request(Some(worker_b.into()));
        set_active_requests(&mut bound, worker_a, 1);
        set_active_requests(&mut bound, worker_b, 1);

        assert_eq!(select(&policy(0), &workers(), &bound), worker_b);
    }

    #[test]
    fn a_one_request_swing_does_not_walk_the_binding_across_the_pool() {
        let a = WorkerWithDpRank::from_worker_id(29);
        let b = WorkerWithDpRank::from_worker_id(41);
        let c = WorkerWithDpRank::from_worker_id(57);
        let pool = HashMap::from([(29, TestWorker), (41, TestWorker), (57, TestWorker)]);
        let policy = policy_with_margin(0, DEFAULT_MOVE_MARGIN);

        // Each step makes the bound worker one request busier than some other worker.
        for (la, lb, lc) in [(1, 0, 1), (1, 1, 0), (1, 0, 1)] {
            let mut bound = request(Some(a.into()));
            set_active_requests(&mut bound, a, la);
            set_active_requests(&mut bound, b, lb);
            set_active_requests(&mut bound, c, lc);
            assert_eq!(select(&policy, &pool, &bound), a);
        }

        let mut bound = request(Some(a.into()));
        set_active_requests(&mut bound, a, 3);
        set_active_requests(&mut bound, b, 0);
        set_active_requests(&mut bound, c, 1);
        assert_eq!(select(&policy, &pool, &bound), b);
    }

    #[test]
    fn a_maximal_move_margin_never_moves_and_never_overflows() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        // `alt + margin` overflowed here; the gap of 2 is below the margin, so it must stay.
        let mut bound = request(Some(worker_a.into()));
        set_active_requests(&mut bound, worker_a, 3);
        set_active_requests(&mut bound, worker_b, 1);

        assert_eq!(
            select(&policy_with_margin(0, usize::MAX), &workers(), &bound),
            worker_a
        );
    }

    #[test]
    fn rejects_a_zero_move_margin() {
        assert!(Parameters::default().validate().is_ok());
        assert!(
            Parameters {
                move_margin: 0,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn retains_the_only_eligible_target() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let only_worker_a = HashMap::from([(29, TestWorker)]);
        let mut bound = request(Some(worker_a.into()));
        set_active_requests(&mut bound, worker_a, 9);

        assert_eq!(select(&policy(0), &only_worker_a, &bound), worker_a);
    }

    #[test]
    fn falls_back_by_load_when_the_target_is_not_eligible() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_c = WorkerWithDpRank::from_worker_id(57);
        let remaining = HashMap::from([(29, TestWorker), (57, TestWorker)]);
        let mut bound = request(Some(WorkerWithDpRank::from_worker_id(41).into()));
        set_active_requests(&mut bound, worker_a, 3);
        set_active_requests(&mut bound, worker_c, 0);

        assert_eq!(select(&policy(0), &remaining, &bound), worker_c);
    }
}
