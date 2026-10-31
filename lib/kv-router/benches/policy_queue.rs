// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy queue drain, head-of-line, due-time, and single-entry scaling benchmarks.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench policy_queue`

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use dynamo_kv_router::RouterQueuePolicy;
use dynamo_kv_router::scheduling::{
    PolicyProfile, PolicyQueue, QueueEntryHandle, QueueMetadata, QueueSnapshot, RouterPolicyConfig,
};
use tokio::time::Instant;

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
policy_classes:
  - name: bench
    quantum: 1000000
"#,
            )
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
        })
        .clone()
}

/// One class of `requests` entries in arrival order; request `index` can dispatch while
/// `is_dispatchable(index)` holds, standing in for the gate's verdict on the workers the
/// request may use. The queue never reads worker pins, so pinned and unpinned requests
/// build this same queue.
fn class_queue(
    requests: usize,
    is_dispatchable: impl Fn(usize) -> bool,
) -> PolicyQueue<BenchRequest> {
    let mut queue = PolicyQueue::new(&profile());
    for request_index in 0..requests {
        queue
            .enqueue(
                0,
                1,
                QueueSnapshot::new(1, 0),
                request_index as f64,
                0.0,
                0,
                BenchRequest {
                    dispatchable: is_dispatchable(request_index),
                },
            )
            .unwrap();
    }
    queue
}

fn shared_queue(requests: usize) -> PolicyQueue<BenchRequest> {
    class_queue(requests, |_| true)
}

/// Pops while the class head can dispatch and returns how many entries left the queue.
fn drain_ready(queue: &mut PolicyQueue<BenchRequest>) -> usize {
    let mut count = 0;
    while queue
        .pop_next(|_, _, request| request.dispatchable)
        .is_some()
    {
        count += 1;
    }
    count
}

fn drain(mut queue: PolicyQueue<BenchRequest>) -> usize {
    drain_ready(&mut queue)
}

/// Pops every entry, as once every worker has room again.
fn drain_all(queue: &mut PolicyQueue<BenchRequest>) -> usize {
    let mut count = 0;
    while queue.pop_next(|_, _, _| true).is_some() {
        count += 1;
    }
    count
}

/// Build plus drop: each built queue is dropped inside the timed loop.
fn bench_build_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/build_shared");
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

fn odd(request: usize) -> bool {
    !request.is_multiple_of(2)
}

fn every_tenth(request: usize) -> bool {
    request.is_multiple_of(10)
}

fn none(_: usize) -> bool {
    false
}

/// Drains while the given fraction of requests wait for busy workers, then drains again once
/// every worker has room. The class stops at its first blocked head, so the first phase
/// dispatches only the ready prefix; the two phases together dispatch every request.
fn bench_blocked_head_then_ready(c: &mut Criterion) {
    const REQUESTS: usize = 10_000;
    let mut group = c.benchmark_group("policy_queue/drain_10000_blocked_head_then_ready");
    group.throughput(Throughput::Elements(REQUESTS as u64));
    for (blocked, predicate, first_phase) in [
        ("0_percent", all as fn(usize) -> bool, REQUESTS),
        ("50_percent", odd as fn(usize) -> bool, 0),
        ("90_percent", every_tenth as fn(usize) -> bool, 1),
        ("100_percent", none as fn(usize) -> bool, 0),
    ] {
        group.bench_with_input(
            BenchmarkId::new(blocked, REQUESTS),
            &predicate,
            |b, &predicate| {
                b.iter_batched(
                    || class_queue(REQUESTS, predicate),
                    |mut queue| {
                        assert_eq!(black_box(drain_ready(&mut queue)), first_phase);
                        assert_eq!(black_box(drain_all(&mut queue)), REQUESTS - first_phase);
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

fn timed_profile(yaml: &'static str, cell: &'static OnceLock<PolicyProfile>) -> PolicyProfile {
    cell.get_or_init(|| {
        RouterPolicyConfig::from_yaml(yaml)
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Edf)
    })
    .clone()
}

/// An EDF class whose requests take their due time from the class budget.
fn budget_profile() -> PolicyProfile {
    static PROFILE: OnceLock<PolicyProfile> = OnceLock::new();
    timed_profile(
        r#"
policy_classes:
  - name: bench
    quantum: 1000000
    slo_ttft_ms: 1000
"#,
        &PROFILE,
    )
}

fn fcfs_profile() -> PolicyProfile {
    static PROFILE: OnceLock<PolicyProfile> = OnceLock::new();
    timed_profile(
        r#"
policy_classes:
  - name: bench
    quantum: 1000000
    queue_policy: fcfs
"#,
        &PROFILE,
    )
}

/// Arrival `index` microseconds after `origin`; every `explicit_every`-th request also
/// carries an explicit deadline, which enters the deadline index.
fn timed_metadata(
    profile: &PolicyProfile,
    origin: Instant,
    index: usize,
    explicit_every: Option<usize>,
) -> QueueMetadata {
    let ingress_at = origin + Duration::from_micros(index as u64);
    let mut metadata = QueueMetadata {
        class_index: 0,
        snapshot: QueueSnapshot::new(1, 0),
        due_at: explicit_every
            .filter(|every| index.is_multiple_of(*every))
            .map(|_| ingress_at + Duration::from_millis(500)),
        class_due_at: None,
        arrival_offset_secs: index as f64,
    };
    metadata
        .resolve_class_due_at(profile.class(0), ingress_at)
        .unwrap();
    metadata
}

fn timed_queue(
    profile: &PolicyProfile,
    origin: Instant,
    requests: usize,
    explicit_every: Option<usize>,
) -> (PolicyQueue<BenchRequest>, Vec<QueueEntryHandle>) {
    let mut queue = PolicyQueue::new(profile);
    let handles = (0..requests)
        .map(|index| {
            queue
                .enqueue_with_due_at(
                    timed_metadata(profile, origin, index, explicit_every),
                    1,
                    0.0,
                    0,
                    BenchRequest { dispatchable: true },
                )
                .unwrap()
        })
        .collect();
    (queue, handles)
}

fn bench_drain_shared_due_times(c: &mut Criterion) {
    let origin = Instant::now();
    for (group_name, profile, explicit_every) in [
        (
            "policy_queue/drain_shared_class_budget",
            budget_profile(),
            None,
        ),
        (
            "policy_queue/drain_shared_mixed_deadlines",
            budget_profile(),
            Some(4),
        ),
        ("policy_queue/drain_shared_fcfs", fcfs_profile(), None),
    ] {
        let mut group = c.benchmark_group(group_name);
        for requests in [128, 1024, 4096] {
            group.throughput(Throughput::Elements(requests as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(requests),
                &requests,
                |b, &requests| {
                    b.iter_batched(
                        || timed_queue(&profile, origin, requests, explicit_every).0,
                        |queue| assert_eq!(black_box(drain(queue)), requests),
                        BatchSize::LargeInput,
                    );
                },
            );
        }
        group.finish();
    }
}

const STEADY_SIZES: [usize; 3] = [1 << 10, 1 << 14, 1 << 17];

/// Inserts one entry whose due time falls in the middle of a queue of `size` entries,
/// then cancels it by handle, leaving the queue as it was.
fn bench_steady_insert_cancel(c: &mut Criterion) {
    let origin = Instant::now();
    let profile = budget_profile();
    for (group_name, explicit_every) in [
        ("policy_queue/steady_insert_cancel_shared", None),
        ("policy_queue/steady_insert_cancel_deadline", Some(1)),
    ] {
        let mut group = c.benchmark_group(group_name);
        for size in STEADY_SIZES {
            let (mut queue, _) = timed_queue(&profile, origin, size, explicit_every);
            let middle = timed_metadata(&profile, origin, size / 2, explicit_every);
            group.bench_function(BenchmarkId::from_parameter(size), |b| {
                b.iter(|| {
                    let handle = queue
                        .enqueue_with_due_at(
                            black_box(middle),
                            1,
                            0.0,
                            0,
                            BenchRequest { dispatchable: true },
                        )
                        .unwrap();
                    black_box(queue.remove(handle).unwrap());
                });
            });
            assert_eq!(queue.pending_count(), size);
        }
        group.finish();
    }
}

/// Dispatches the earliest entry of a queue of `size` entries, then enqueues a later one.
fn bench_steady_pop_insert(c: &mut Criterion) {
    let origin = Instant::now();
    let profile = budget_profile();
    let mut group = c.benchmark_group("policy_queue/steady_pop_insert_shared");
    for size in STEADY_SIZES {
        let (mut queue, _) = timed_queue(&profile, origin, size, None);
        let mut next = size;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                black_box(
                    queue
                        .pop_next(|_, _, request| request.dispatchable)
                        .unwrap(),
                );
                queue
                    .enqueue_with_due_at(
                        timed_metadata(&profile, origin, next, None),
                        1,
                        0.0,
                        0,
                        BenchRequest { dispatchable: true },
                    )
                    .unwrap();
                next += 1;
            });
        });
        assert_eq!(queue.pending_count(), size);
    }
    group.finish();
}

/// The bulk path for contrast: one middle insert removed by inspecting every entry.
fn bench_steady_insert_retain(c: &mut Criterion) {
    let origin = Instant::now();
    let profile = budget_profile();
    let mut group = c.benchmark_group("policy_queue/steady_insert_retain_shared");
    for size in STEADY_SIZES {
        let (mut queue, _) = timed_queue(&profile, origin, size, None);
        let middle = timed_metadata(&profile, origin, size / 2, None);
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                queue
                    .enqueue_with_due_at(
                        black_box(middle),
                        1,
                        0.0,
                        0,
                        BenchRequest {
                            dispatchable: false,
                        },
                    )
                    .unwrap();
                queue.retain(|request| request.dispatchable);
            });
        });
        assert_eq!(queue.pending_count(), size);
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
    targets = bench_build_shared, bench_blocked_head_then_ready, bench_drain_shared, bench_drain_shared_due_times, bench_steady_insert_cancel, bench_steady_pop_insert, bench_steady_insert_retain
}
criterion_main!(benches);
