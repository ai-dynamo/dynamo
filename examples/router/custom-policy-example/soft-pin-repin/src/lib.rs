// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom worker-selection policy that demonstrates overload-aware soft affinity.
//!
//! The picker retains an eligible affinity target until its active request count
//! exceeds the configured threshold. It then selects the least-loaded alternative,
//! and Dynamo rebinds the session to the dispatched worker.

use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerAffinityTarget;
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

struct SoftPinRepinPicker {
    max_active_requests: usize,
}

impl SoftPinRepinPicker {
    fn matches_target(candidate: &ScoredWorkerCandidate, target: WorkerAffinityTarget) -> bool {
        let worker = candidate.worker();
        worker.worker_id == target.worker_id
            && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
    }

    fn least_loaded_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[dynamo_kv_router::WorkerLoadInput],
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
}

impl WorkerPicker for SoftPinRepinPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let loads = input.load().ok_or_else(|| {
            WorkerSelectionPolicyError::failed("active load input is unavailable")
        })?;

        if let Some(target) = context.affinity_target()
            && let Some(target_row) = candidates
                .iter()
                .zip(loads)
                .enumerate()
                .filter(|(_, (candidate, _))| Self::matches_target(candidate, target))
                .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
                .map(|(row, _)| row)
        {
            if loads[target_row].active_requests() <= self.max_active_requests {
                return Ok(target_row);
            }

            return Ok(
                Self::least_loaded_row(candidates, loads, Some(target)).unwrap_or(target_row)
            );
        }

        Self::least_loaded_row(candidates, loads, None)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    max_active_requests: usize,
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    let max_active_requests = parameters.max_active_requests;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SoftPinRepinPicker {
                    max_active_requests,
                }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("soft-pin-repin", Arc::new(provider))
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
            router_hint_candidates: None,
            retain_router_hint_chain: false,
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
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(SoftPinRepinPicker {
                max_active_requests,
            }),
        )
    }

    #[test]
    fn repins_only_after_the_affinity_target_exceeds_the_threshold() {
        let workers = HashMap::from([(41, TestWorker), (29, TestWorker)]);
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut unbound = request(None);
        set_active_requests(&mut unbound, worker_a, 0);
        set_active_requests(&mut unbound, worker_b, 2);
        let selected = policy(0)
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &unbound,
                unbound.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, worker_a);

        let mut bound = request(Some(selected.worker.into()));
        set_active_requests(&mut bound, worker_a, 0);
        set_active_requests(&mut bound, worker_b, 0);
        let retained = policy(0)
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &bound,
                bound.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(retained.worker, worker_a);

        set_active_requests(&mut bound, worker_a, 1);
        let repinned = policy(0)
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &bound,
                bound.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(repinned.worker, worker_b);
    }

    #[test]
    fn retains_the_only_eligible_affinity_target() {
        let workers = HashMap::from([(29, TestWorker)]);
        let mut request = request(Some(WorkerWithDpRank::from_worker_id(29).into()));
        set_active_requests(&mut request, WorkerWithDpRank::from_worker_id(29), 1);
        let selected = policy(0)
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, WorkerWithDpRank::from_worker_id(29));
    }
}
