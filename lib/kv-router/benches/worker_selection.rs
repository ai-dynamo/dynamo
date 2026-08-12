// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection scaling benchmark for built-in and custom policies.
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
    DefaultWorkerSelector, KvRouterConfig, SchedulingRequest, WorkerCandidate, WorkerFilter,
    WorkerInputView, WorkerInputs, WorkerLoadProjection, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
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

struct KeepAllFilter;

impl WorkerFilter for KeepAllFilter {
    fn keep(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        _candidate: &WorkerCandidate,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        Ok(true)
    }
}

struct BenchScorer;

impl WorkerScorer for BenchScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::ROUTING
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let routing = candidate
            .routing()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("routing input unavailable"))?;
        let uncached_blocks =
            (context.request_blocks() as f64 - cache.device_overlap_blocks()).max(0.0);
        let load_blocks = load.active_prefill_tokens() as f64 / context.block_size() as f64
            + load.decode_cost_blocks()
            + load.active_requests() as f64;
        Ok((uncached_blocks + load_blocks) * routing.preferred_taint_multiplier().unwrap_or(1.0))
    }
}

struct LowestCostPicker;

impl WorkerPicker for LowestCostPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
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
        session_context: None,
        overlap: OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens,
        },
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads,
        resp_tx: None,
    };

    (workers, request)
}

fn worker_selection(c: &mut Criterion) {
    for (scenario, overlap_score_credit_decay) in
        [("direct", 0.0), ("filtered_prefill_baseline", 1.0)]
    {
        for temperature in [0.0, 0.7] {
            let mut group = c.benchmark_group(format!(
                "default_worker_selection/{scenario}/temperature_{temperature}"
            ));
            group.warm_up_time(Duration::from_secs(2));
            group.measurement_time(Duration::from_secs(5));
            group.sample_size(50);

            for worker_count in [2, 32, 1_024, 10_000] {
                let (workers, request) = fixture(worker_count);
                let selector = DefaultWorkerSelector::new(
                    Some(KvRouterConfig {
                        router_temperature: temperature,
                        overlap_score_credit_decay,
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
}

fn custom_worker_selection(c: &mut Criterion) {
    const WORKER_COUNT: usize = 10_000;

    let (workers, request) = fixture(WORKER_COUNT);
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let custom_no_filter = WorkerSelectionPolicy::new(
        config.clone(),
        "prefill",
        vec![Box::new(BenchScorer)],
        Box::new(LowestCostPicker),
    );
    let custom_keep_all_filter = WorkerSelectionPolicy::new_with_filters(
        config,
        "prefill",
        vec![Box::new(KeepAllFilter)],
        vec![Box::new(BenchScorer)],
        Box::new(LowestCostPicker),
    );
    let mut group = c.benchmark_group("custom_worker_selection/10_000");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    group.throughput(Throughput::Elements(WORKER_COUNT as u64));

    group.bench_function("custom_no_filter", |b| {
        b.iter(|| {
            black_box(
                custom_no_filter
                    .select_worker(
                        black_box(&workers),
                        black_box(&request),
                        request.eligibility(),
                        black_box(16),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("custom_keep_all_filter", |b| {
        b.iter(|| {
            black_box(
                custom_keep_all_filter
                    .select_worker(
                        black_box(&workers),
                        black_box(&request),
                        request.eligibility(),
                        black_box(16),
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn default_policy_wrapper(c: &mut Criterion) {
    let config = KvRouterConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("default_policy_wrapper");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for worker_count in [2, 32, 1_024] {
        let (workers, request) = fixture(worker_count);
        let direct = DefaultWorkerSelector::new(Some(config.clone()), "prefill");
        let policy = WorkerSelectionPolicy::default(config.clone(), "prefill");
        group.throughput(Throughput::Elements(worker_count as u64));

        group.bench_with_input(
            BenchmarkId::new("direct", worker_count),
            &worker_count,
            |b, _| {
                b.iter(|| {
                    black_box(
                        direct
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
        group.bench_with_input(
            BenchmarkId::new("policy", worker_count),
            &worker_count,
            |b, _| {
                b.iter(|| {
                    black_box(
                        policy
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

criterion_group!(
    benches,
    worker_selection,
    custom_worker_selection,
    default_policy_wrapper
);
criterion_main!(benches);
