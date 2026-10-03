// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `fetch-to-idle` policy.
//!
//! The picker keeps a request on the worker that caches its prefix unless that worker is busy.
//! For a busy holder, it selects the least-loaded worker, and the experimental KV fetch policy
//! has that worker fetch the prefix instead of recomputing it.

mod picker;
mod transfer;

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::worker_selection::experimental::with_kv_transfer_policy;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerSelectionPolicy, WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
use picker::FetchToIdlePicker;
use transfer::FetchFromHolder;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    busy_active_requests: usize,
    #[serde(default)]
    min_fetch_blocks: u32,
}

fn validate_busy_active_requests(
    busy_active_requests: usize,
) -> Result<(), WorkerSelectionPolicyProviderError> {
    if busy_active_requests == 0 {
        return Err(WorkerSelectionPolicyProviderError::new(
            "busy_active_requests must be at least 1",
        ));
    }
    Ok(())
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let Parameters {
        busy_active_requests,
        min_fetch_blocks,
    } = parameters.deserialize()?;
    validate_busy_active_requests(busy_active_requests)?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            let policy = WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(FetchToIdlePicker {
                    busy_active_requests,
                }),
            );
            with_kv_transfer_policy(policy, Box::new(FetchFromHolder { min_fetch_blocks }))
        },
    ))
}

/// Register the `fetch-to-idle` policy type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("fetch-to-idle", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::transfer::choose_source;
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

    const HOLDER: WorkerWithDpRank = WorkerWithDpRank {
        worker_id: 1,
        dp_rank: 0,
    };
    const IDLE: WorkerWithDpRank = WorkerWithDpRank {
        worker_id: 2,
        dp_rank: 0,
    };

    /// The holder caches 8 blocks of a 128-token prompt and has `holder_active` requests.
    fn select(busy_active_requests: usize, holder_active: usize) -> WorkerWithDpRank {
        let workers = HashMap::from([(HOLDER.worker_id, TestWorker), (IDLE.worker_id, TestWorker)]);
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 128,
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
        request.overlap.tier_overlap_blocks.device.insert(HOLDER, 8);
        for (worker, active_requests) in [(HOLDER, holder_active), (IDLE, 0)] {
            request.worker_loads.insert(
                worker,
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(FetchToIdlePicker {
                busy_active_requests,
            }),
        );
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
    fn keeps_an_idle_cache_holder() {
        assert_eq!(select(1, 0), HOLDER);
        assert_eq!(select(2, 1), HOLDER);
    }

    #[test]
    fn moves_off_a_busy_cache_holder() {
        assert_eq!(select(1, 1), IDLE);
        assert_eq!(select(2, 3), IDLE);
    }

    #[test]
    fn fetches_only_when_the_source_adds_enough_blocks() {
        assert_eq!(choose_source(0, 64, [(63, true)]), None);
        // The source holds 70 blocks, but only 60 are beyond the 10 the target already holds.
        assert_eq!(choose_source(10, 64, [(70, false)]), None);
        assert_eq!(choose_source(0, 64, [(64, false)]), Some(0));
        assert_eq!(choose_source(0, 0, []), None);
    }

    #[test]
    fn prefers_a_kv_pool_among_the_longest_sources() {
        assert_eq!(
            choose_source(0, 0, [(9, false), (9, false), (9, true)]),
            Some(2)
        );
        // A shorter pool does not displace a longer worker source.
        assert_eq!(choose_source(0, 0, [(9, false), (8, true)]), Some(0));
    }

    #[test]
    fn validates_busy_active_requests() {
        assert!(validate_busy_active_requests(0).is_err());
        assert!(validate_busy_active_requests(1).is_ok());
    }
}
