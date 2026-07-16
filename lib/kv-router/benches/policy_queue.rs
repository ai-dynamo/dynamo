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
    PolicyProfile, PolicyQueue, QueueSnapshot, RequestProgress, RouterPolicyConfig, WorkerPlacement,
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
    is_dispatchable: impl Fn(usize) -> bool,
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
                    dispatchable: is_dispatchable(lane),
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
    for lanes in [1, 8, 32, 128, 512, 1024, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter_batched_ref(
                || exact_queue(lanes, lanes, |lane| !lane.is_multiple_of(2)),
                |queue| {
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
                || exact_queue(lanes, REQUESTS, |_| true),
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
                || exact_queue(lanes, lanes, |_| true),
                |queue| assert_eq!(black_box(drain(queue)), lanes),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_build_exact_lanes(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/build_one_per_exact_lane");
    for lanes in [128, 1024, 10_000] {
        group.throughput(Throughput::Elements(lanes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter(|| black_box(exact_queue(lanes, lanes, |_| true)));
        });
    }
    group.finish();
}

fn all(_: usize) -> bool {
    true
}

fn odd(lane: usize) -> bool {
    !lane.is_multiple_of(2)
}

fn every_tenth(lane: usize) -> bool {
    lane.is_multiple_of(10)
}

fn none(_: usize) -> bool {
    false
}

fn bench_blocked_fraction(c: &mut Criterion) {
    const LANES: usize = 10_000;
    let mut group = c.benchmark_group("policy_queue/drain_10000_exact_blocked");
    for (blocked, predicate, expected) in [
        ("0_percent", all as fn(usize) -> bool, LANES),
        ("50_percent", odd as fn(usize) -> bool, LANES / 2),
        ("90_percent", every_tenth as fn(usize) -> bool, LANES / 10),
        ("100_percent", none as fn(usize) -> bool, 0),
    ] {
        group.bench_with_input(
            BenchmarkId::new(blocked, LANES),
            &predicate,
            |b, &predicate| {
                b.iter_batched(
                    || exact_queue(LANES, LANES, predicate),
                    |queue| assert_eq!(black_box(drain(queue)), expected),
                    BatchSize::LargeInput,
                );
            },
        );
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

fn bench_request_progress(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_progress");
    let (progress, updater) = RequestProgress::new(0);
    let mut context_tokens = 0usize;
    group.bench_function("update", |b| {
        b.iter(|| {
            context_tokens += 1;
            updater.update_context_tokens(black_box(context_tokens));
        });
    });
    group.bench_function("read", |b| {
        b.iter(|| black_box(progress.context_tokens()));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03);
    targets = bench_pop_once, bench_drain_fixed_requests, bench_drain_one_per_lane, bench_build_exact_lanes, bench_blocked_fraction, bench_drain_shared, bench_request_progress
}
criterion_main!(benches);
