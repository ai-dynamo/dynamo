// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy queue scaling benchmarks.
//!
//! Each policy class owns exactly one due-time min-max heap, so these measure
//! enqueue/dispatch on that heap, cross-class deficit round robin, and the cost
//! of shedding heads whose class deadline has passed.
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
    PolicyProfile, PolicyQueue, QueueArrival, QueueSnapshot, RouterPolicyConfig, WorkerPlacement,
};
use tokio::time::Instant;

#[derive(Debug, Clone, Copy)]
struct BenchRequest {
    dispatchable: bool,
}

/// Long enough that nothing expires in the ordering benchmarks; the expiry
/// benchmark builds its own short-SLO profile.
const BENCH_SLO_MS: u64 = 3_600_000;

fn profile() -> PolicyProfile {
    static PROFILE: OnceLock<PolicyProfile> = OnceLock::new();
    PROFILE
        .get_or_init(|| {
            RouterPolicyConfig::from_yaml(&format!(
                r#"
default_policy_class: bench
policy_classes:
  - name: bench
    slo_ms: {BENCH_SLO_MS}
    quantum: 1000000
"#
            ))
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
        })
        .clone()
}

fn arrival<T>(queue: &PolicyQueue<T>, base: Instant, request_index: usize) -> QueueArrival {
    let at = base + Duration::from_micros(request_index as u64);
    QueueArrival::new(at, 0.0, 0.0, 0, queue.class_config(0))
}

/// One class holding `requests` entries, each pinned to one of `lanes` workers.
/// Placement no longer partitions storage; it only decides which entries the
/// dispatchability test rejects.
fn pinned_queue(
    lanes: usize,
    requests: usize,
    is_dispatchable: impl Fn(usize) -> bool,
) -> PolicyQueue<BenchRequest> {
    let base = Instant::now();
    let mut queue = PolicyQueue::new(profile());
    for request_index in 0..requests {
        let lane = request_index % lanes;
        let arrival = arrival(&queue, base, request_index);
        queue
            .enqueue(
                0,
                lanes,
                QueueSnapshot::new(1, 0),
                arrival,
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
    let base = Instant::now();
    let mut queue = PolicyQueue::new(profile());
    for request_index in 0..requests {
        let arrival = arrival(&queue, base, request_index);
        queue
            .enqueue(
                0,
                1,
                QueueSnapshot::new(1, 0),
                arrival,
                WorkerPlacement::Any,
                BenchRequest { dispatchable: true },
            )
            .unwrap();
    }
    queue
}

fn drain(mut queue: PolicyQueue<BenchRequest>) -> usize {
    let now = Instant::now();
    let mut expired = Vec::new();
    let mut count = 0;
    while queue
        .pop_next(now, &mut expired, |_, _, request: &BenchRequest| {
            request.dispatchable
        })
        .is_some()
    {
        count += 1;
    }
    assert!(expired.is_empty());
    count
}

fn bench_pop_once(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/pop_once");
    for requests in [1, 8, 32, 128, 512, 1024, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(requests),
            &requests,
            |b, &requests| {
                b.iter_batched_ref(
                    || shared_queue(requests),
                    |queue| {
                        let mut expired = Vec::new();
                        black_box(queue.pop_next(
                            Instant::now(),
                            &mut expired,
                            |_, _, request: &BenchRequest| request.dispatchable,
                        ));
                    },
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

fn bench_build_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/build");
    for requests in [128, 1024, 10_000] {
        group.throughput(Throughput::Elements(requests as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(requests),
            &requests,
            |b, &requests| {
                b.iter(|| black_box(shared_queue(requests)));
            },
        );
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

/// A blocked head holds its class's line, so this measures how quickly a drain
/// stalls as the fraction of undispatchable pinned requests grows.
fn bench_blocked_head_fraction(c: &mut Criterion) {
    const LANES: usize = 10_000;
    let mut group = c.benchmark_group("policy_queue/drain_10000_pinned_blocked");
    for (blocked, predicate) in [
        ("0_percent", all as fn(usize) -> bool),
        ("50_percent", odd as fn(usize) -> bool),
        ("90_percent", every_tenth as fn(usize) -> bool),
        ("100_percent", none as fn(usize) -> bool),
    ] {
        group.bench_with_input(
            BenchmarkId::new(blocked, LANES),
            &predicate,
            |b, &predicate| {
                b.iter_batched(
                    || pinned_queue(LANES, LANES, predicate),
                    |queue| black_box(drain(queue)),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Cost of shedding a run of expired heads before the first live one.
fn bench_dispatch_expiry(c: &mut Criterion) {
    const REQUESTS: usize = 4096;
    let expiring_profile = RouterPolicyConfig::from_yaml(
        r#"
default_policy_class: expiring
policy_classes:
  - name: expiring
    slo_ms: 1000
    quantum: 1000000
"#,
    )
    .unwrap()
    .resolve_profile(None, None, RouterQueuePolicy::Fcfs);

    let mut group = c.benchmark_group("policy_queue/dispatch_expiry");
    for expired_percent in [0, 50, 100] {
        group.throughput(Throughput::Elements(REQUESTS as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(expired_percent),
            &expired_percent,
            |b, &expired_percent| {
                b.iter_batched(
                    || {
                        let base = Instant::now();
                        let mut queue = PolicyQueue::new(expiring_profile.clone());
                        for request_index in 0..REQUESTS {
                            // Entries in the first microseconds after `base` are
                            // past their 1s SLO at poll time; the rest arrive
                            // 10s later and are still live.
                            let at = if request_index * 100 < REQUESTS * expired_percent {
                                base + Duration::from_micros(request_index as u64)
                            } else {
                                base + Duration::from_secs(10)
                                    + Duration::from_micros(request_index as u64)
                            };
                            let arrival = QueueArrival::new(at, 0.0, 0.0, 0, queue.class_config(0));
                            queue
                                .enqueue(
                                    0,
                                    1,
                                    QueueSnapshot::new(1, 0),
                                    arrival,
                                    WorkerPlacement::Any,
                                    BenchRequest { dispatchable: true },
                                )
                                .unwrap();
                        }
                        (queue, base)
                    },
                    |(mut queue, base)| {
                        let now = base + Duration::from_millis(1_001);
                        let mut expired = Vec::new();
                        let mut dispatched = 0;
                        while queue
                            .pop_next(now, &mut expired, |_, _, request: &BenchRequest| {
                                request.dispatchable
                            })
                            .is_some()
                        {
                            dispatched += 1;
                        }
                        black_box((dispatched, expired.len()));
                    },
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
    targets = bench_pop_once, bench_drain_shared, bench_build_queue, bench_blocked_head_fraction, bench_dispatch_expiry
}
criterion_main!(benches);
