// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load-bounded cache-affinity worker selection.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy};
use picker::CacheAffinityBudgetPicker;
use scorer::ActiveLoadScorer;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    max_load_cost_delta_blocks: f64,
}

fn validate_cost_delta(value: f64) -> Result<(), WorkerSelectionPolicyProviderError> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(WorkerSelectionPolicyProviderError::new(
        "max_load_cost_delta_blocks must be a finite non-negative number",
    ))
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate_cost_delta(parameters.max_load_cost_delta_blocks)?;
    let max_load_cost_delta_blocks = parameters.max_load_cost_delta_blocks;

    Ok(Arc::new(move |config, worker_type, _partition| {
        create_policy(config, worker_type.as_str(), max_load_cost_delta_blocks)
    }))
}

fn create_policy(
    config: &KvRouterConfig,
    worker_label: &'static str,
    max_load_cost_delta_blocks: f64,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config.clone(),
        worker_label,
        vec![Box::new(ActiveLoadScorer {
            prefill_load_scale: config.prefill_load_scale,
            active_request_weight: config.decode_active_request_weight,
        })],
        Box::new(CacheAffinityBudgetPicker {
            max_load_cost_delta_blocks,
        }),
    )
}

/// Register the `cache-affinity-budget` policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("cache-affinity-budget", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{SchedulingRequest, WorkerSelectionInput, WorkerSelector};

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
            Some(4096)
        }
    }

    fn request(warm_load_blocks: usize) -> SchedulingRequest {
        request_with_overlap(80, 80.0, warm_load_blocks)
    }

    fn request_with_overlap(
        warm_device_blocks: usize,
        warm_effective_blocks: f64,
        warm_load_blocks: usize,
    ) -> SchedulingRequest {
        let warm = WorkerWithDpRank::from_worker_id(0);
        let cold = WorkerWithDpRank::from_worker_id(1);
        let mut overlap = OverlapSignals::default();
        overlap
            .tier_overlap_blocks
            .device
            .insert(warm, warm_device_blocks);
        overlap
            .effective_overlap_blocks
            .insert(warm, warm_effective_blocks);
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 1600,
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
            overlap,
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        request.worker_loads.insert(
            warm,
            dynamo_kv_router::sequences::WorkerLoadProjection {
                active_decode_blocks: warm_load_blocks,
                ..Default::default()
            },
        );
        request.worker_loads.insert(cold, Default::default());
        request
    }

    fn select(max_delta: f64, warm_load_blocks: usize) -> WorkerWithDpRank {
        let policy = create_policy(&KvRouterConfig::default(), "test", max_delta);
        let workers = HashMap::from([(0, TestWorker), (1, TestWorker)]);
        let request = request(warm_load_blocks);
        policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
    }

    #[test]
    fn cache_wins_when_load_penalty_is_within_budget() {
        assert_eq!(select(40.0, 40), WorkerWithDpRank::from_worker_id(0));
    }

    #[test]
    fn load_wins_when_cache_worker_exceeds_budget() {
        assert_eq!(select(39.0, 40), WorkerWithDpRank::from_worker_id(1));
    }

    #[test]
    fn lower_tier_cache_counts_within_load_budget() {
        let policy = create_policy(&KvRouterConfig::default(), "test", 40.0);
        let workers = HashMap::from([(0, TestWorker), (1, TestWorker)]);
        let request = request_with_overlap(0, 80.0, 40);
        assert_eq!(
            policy
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(0)
        );
    }

    #[test]
    fn validates_cost_delta() {
        assert!(validate_cost_delta(0.0).is_ok());
        assert!(validate_cost_delta(64.0).is_ok());
        assert!(validate_cost_delta(-1.0).is_err());
        assert!(validate_cost_delta(f64::NAN).is_err());
    }
}
