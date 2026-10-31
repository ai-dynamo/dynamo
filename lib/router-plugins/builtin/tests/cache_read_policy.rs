// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "support/mod.rs"]
mod support;

use std::collections::HashMap;

use dynamo_custom_policy_builtin::default_factory;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::selector::WorkerSelector;
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerLoadProjection, WorkerType};

struct OneRank;
impl WorkerConfigLike for OneRank {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        1
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(2048)
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(1024)
    }
}

#[test]
fn default_scorer_does_not_credit_an_unreadable_warm_prefix() {
    let (_, mut request) = support::fixture(2, 65);
    let cold = WorkerWithDpRank::new(0, 0);
    let warm = WorkerWithDpRank::new(1, 0);
    let workers = HashMap::from([(0, OneRank), (1, OneRank)]);
    request.overlap = Default::default();
    request.overlap.tier_overlap_blocks.device.insert(warm, 4);
    request.overlap.effective_overlap_blocks.insert(warm, 4.0);
    request.overlap.effective_cached_tokens.insert(warm, 64);
    request.worker_loads = [
        (
            cold,
            WorkerLoadProjection {
                additional_active_blocks: 5,
                ..Default::default()
            },
        ),
        (
            warm,
            WorkerLoadProjection {
                active_prefill_tokens: 16,
                active_decode_blocks: 1,
                additional_active_blocks: 5,
                active_requests: 1,
            },
        ),
    ]
    .into_iter()
    .collect();
    let config = KvRouterConfig {
        router_temperature: 0.0,
        overlap_score_credit_decay: 0.0,
        ..Default::default()
    };
    let policy = default_factory()(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("test", "default"),
    );
    let normal = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(normal.worker, warm);
    request.prefix_cache_read_allowed = Some(false);
    let bypass = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(bypass.worker, cold);
    assert_eq!(bypass.cached_tokens, 0);
    request.prefix_cache_read_allowed = Some(true);
    let allowed = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(allowed.worker, warm);
    assert_eq!(request.overlap.tier_overlap_blocks.device[&warm], 4);
}
