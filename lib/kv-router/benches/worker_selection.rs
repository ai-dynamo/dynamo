// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default worker-selection scaling benchmark.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench worker_selection`

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::protocols::{
    RoutingConstraints, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
use dynamo_kv_router::{
    DefaultWorkerSelector, KvRouterConfig, SchedulingRequest, WorkerLoadProjection, WorkerSelector,
};
use rustc_hash::FxHashMap;

#[derive(Default)]
struct BenchWorkerConfig {
    taints: HashSet<String>,
}

impl WorkerConfigLike for BenchWorkerConfig {
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
        Some(131_072)
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}

fn fixture(worker_count: usize) -> (HashMap<WorkerId, BenchWorkerConfig>, SchedulingRequest) {
    let mut workers = HashMap::with_capacity(worker_count);
    let mut effective_overlap_blocks = HashMap::with_capacity(worker_count);
    let mut effective_cached_tokens = HashMap::with_capacity(worker_count);
    let mut worker_loads = FxHashMap::with_capacity_and_hasher(worker_count, Default::default());

    for worker_id in 0..worker_count as WorkerId {
        let worker = WorkerWithDpRank::from_worker_id(worker_id);
        let cached_tokens = (worker_id as usize % 32) * 64;
        workers.insert(worker_id, BenchWorkerConfig::default());
        effective_overlap_blocks.insert(worker, cached_tokens as f64 / 16.0);
        effective_cached_tokens.insert(worker, cached_tokens);
        worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_prefill_tokens: worker_id as usize * 16,
                active_decode_blocks: worker_id as usize % 127,
                active_requests: worker_id as usize % 17,
                ..Default::default()
            },
        );
    }

    let request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 2_048,
        lora_name: None,
        expected_output_tokens: Some(256),
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_id: None,
        overlap: OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens,
        },
        router_hint_candidates: None,
        retain_router_hint_chain: false,
        shared_cache_hits: None,
        worker_loads,
        resp_tx: None,
    };

    (workers, request)
}

fn worker_selection(c: &mut Criterion) {
    for temperature in [0.0, 0.7] {
        let mut group = c.benchmark_group(format!(
            "default_worker_selection/temperature_{temperature}"
        ));
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(50);

        for worker_count in [2, 32, 1_024, 10_000] {
            let (workers, request) = fixture(worker_count);
            let selector = DefaultWorkerSelector::new(
                Some(KvRouterConfig {
                    router_temperature: temperature,
                    ..Default::default()
                }),
                "prefill",
            );
            group.throughput(Throughput::Elements(worker_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(worker_count),
                &worker_count,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            selector
                                .select_worker(
                                    black_box(&workers),
                                    black_box(&request),
                                    request.eligibility(),
                                    black_box(16),
                                )
                                .unwrap(),
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, worker_selection);
criterion_main!(benches);
