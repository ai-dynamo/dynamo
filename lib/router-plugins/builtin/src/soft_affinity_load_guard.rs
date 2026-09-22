// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retain an eligible soft-affinity target until its active load warrants moving.
//!
//! Dynamo owns the affinity binding. This picker only selects an eligible worker; a successful
//! dispatch updates a soft binding through the normal coordinator path.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerAffinityTarget;
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

pub const POLICY_TYPE: &str = "dynamo-soft-affinity-load-guard";

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    max_active_requests: usize,
    move_margin: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            max_active_requests: 32,
            move_margin: 2,
        }
    }
}

impl Parameters {
    fn validate(self) -> Result<Self, WorkerSelectionPolicyProviderError> {
        if self.move_margin == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "move_margin must be at least 1",
            ));
        }
        Ok(self)
    }
}

struct SoftAffinityLoadGuardPicker {
    parameters: Parameters,
}

fn matches_target(candidate: &ScoredWorkerCandidate, target: WorkerAffinityTarget) -> bool {
    let worker = candidate.worker();
    worker.worker_id == target.worker_id && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
}

fn least_loaded(
    candidates: &[ScoredWorkerCandidate],
    loads: &[WorkerLoadInput],
    include: impl Fn(&ScoredWorkerCandidate) -> bool,
) -> Option<usize> {
    candidates
        .iter()
        .zip(loads)
        .enumerate()
        .filter(|(_, (candidate, _))| include(candidate))
        .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
        .map(|(row, _)| row)
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
            .ok_or_else(|| WorkerSelectionPolicyError::failed("active load input unavailable"))?;

        if let Some(target) = context.affinity_target()
            && let Some(target_row) = least_loaded(candidates, loads, |candidate| {
                matches_target(candidate, target)
            })
        {
            let target_load = loads[target_row].active_requests();
            if target_load <= self.parameters.max_active_requests {
                return Ok(target_row);
            }

            if let Some(alternative_row) = least_loaded(candidates, loads, |candidate| {
                !matches_target(candidate, target)
            }) {
                let alternative_load = loads[alternative_row].active_requests();
                if alternative_load < target_load
                    && target_load - alternative_load >= self.parameters.move_margin
                {
                    return Ok(alternative_row);
                }
            }
            return Ok(target_row);
        }

        least_loaded(candidates, loads, |_| true)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    let parameters = parameters.validate()?;

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
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
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

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    fn select(parameters: Parameters, target: Option<u64>, a_load: usize, b_load: usize) -> u64 {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: target.map(|id| worker(id).into()),
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
        for (id, active_requests) in [(7, a_load), (9, b_load)] {
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        let workers = HashMap::from([(7, TestWorker), (9, TestWorker)]);
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(SoftAffinityLoadGuardPicker { parameters }),
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ))
        .unwrap()
        .worker
        .worker_id
    }

    #[test]
    fn retains_the_group_until_both_load_conditions_hold() {
        let parameters = Parameters {
            max_active_requests: 7,
            move_margin: 2,
        };
        assert_eq!(select(parameters, Some(7), 7, 0), 7);
        assert_eq!(select(parameters, Some(7), 8, 7), 7);
        assert_eq!(select(parameters, Some(7), 8, 6), 9);
        assert_eq!(select(parameters, Some(9), 6, 8), 7);
        assert_eq!(select(parameters, Some(9), 7, 8), 9);
    }

    #[test]
    fn selects_an_eligible_worker_when_no_group_target_exists() {
        let parameters = Parameters::default();
        assert_eq!(select(parameters, None, 3, 0), 9);
        assert_eq!(select(parameters, Some(99), 3, 0), 9);
        assert_eq!(select(parameters, Some(7), 33, 33), 7);
        assert_eq!(select(parameters, Some(7), 33, 34), 7);
    }

    #[test]
    fn rejects_zero_move_margin() {
        assert!(
            Parameters {
                move_margin: 0,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
    }
}
