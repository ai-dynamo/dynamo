// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy queue lane-scaling benchmarks.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench policy_queue`

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::{
    PolicyProfile, PolicyQueue, QueueSnapshot, RouterPolicyConfig, WorkerPlacement,
};

#[derive(Debug, Clone, Copy)]
struct BenchRequest {
    dispatchable: bool,
}

fn profile() -> PolicyProfile {
    static PROFILE: OnceLock<PolicyProfile> = OnceLock::new();
    PROFILE
        .get_or_init(|| {
            RouterPolicyConfig::from_yaml(
                r#"
default_policy_family: bench
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: bench
    policy_family: bench
    cache_bucket: all
    quantum: 1000000
"#,
            )
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
        })
        .clone()
}

fn exact_queue(
    lanes: usize,
    requests: usize,
    blocked_modulus: Option<usize>,
) -> PolicyQueue<BenchRequest> {
    let mut queue = PolicyQueue::new(profile());
    for request_index in 0..requests {
        let lane = request_index % lanes;
        queue
            .enqueue(
                0,
                lanes,
                QueueSnapshot::new(1, 0),
                request_index as f64,
                0.0,
                0,
                WorkerPlacement::Exact(WorkerWithDpRank::new(lane as u64, 0)),
                BenchRequest {
                    dispatchable: blocked_modulus
                        .is_none_or(|modulus| !lane.is_multiple_of(modulus)),
                },
            )
            .unwrap();
    }
    queue
}

fn shared_queue(requests: usize) -> PolicyQueue<BenchRequest> {
    let mut queue = PolicyQueue::new(profile());
    for request_index in 0..requests {
        queue
            .enqueue(
                0,
                1,
                QueueSnapshot::new(1, 0),
                request_index as f64,
                0.0,
                0,
                WorkerPlacement::Any,
                BenchRequest { dispatchable: true },
            )
            .unwrap();
    }
    queue
}

fn drain(mut queue: PolicyQueue<BenchRequest>) -> usize {
    let mut count = 0;
    while queue
        .pop_next(|_, _, request| request.dispatchable)
        .is_some()
    {
        count += 1;
    }
    count
}

fn bench_pop_once(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/pop_once_exact");
    for lanes in [1, 8, 32, 128, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter_batched(
                || exact_queue(lanes, lanes, Some(2)),
                |mut queue| {
                    black_box(queue.pop_next(|_, _, request| request.dispatchable));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_drain_fixed_requests(c: &mut Criterion) {
    const REQUESTS: usize = 4096;
    let mut group = c.benchmark_group("policy_queue/drain_4096_exact");
    group.throughput(Throughput::Elements(REQUESTS as u64));
    for lanes in [1, 8, 32, 128, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter_batched(
                || exact_queue(lanes, REQUESTS, None),
                |queue| assert_eq!(black_box(drain(queue)), REQUESTS),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_drain_one_per_lane(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/drain_one_per_exact_lane");
    for lanes in [8, 32, 128, 512, 1024] {
        group.throughput(Throughput::Elements(lanes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter_batched(
                || exact_queue(lanes, lanes, None),
                |queue| assert_eq!(black_box(drain(queue)), lanes),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_drain_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/drain_shared");
    for requests in [128, 1024, 4096] {
        group.throughput(Throughput::Elements(requests as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(requests),
            &requests,
            |b, &requests| {
                b.iter_batched(
                    || shared_queue(requests),
                    |queue| assert_eq!(black_box(drain(queue)), requests),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03);
    targets = bench_pop_once, bench_drain_fixed_requests, bench_drain_one_per_lane, bench_drain_shared
}
criterion_main!(benches);
