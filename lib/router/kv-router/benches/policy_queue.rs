// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy queue lane-scaling benchmarks.
//!
//! The original groups use an explicit FCFS class, so they measure the same workload as
//! before policy-class YAML defaulted to EDF. The `edf_*` groups repeat the shared, exact-lane
//! and blocked-lane drains on an EDF class with class due times.
//!
//! EDF queues are built as the scheduler builds its queue, with
//! `PolicyQueue::with_class_budget_expiry`. The `edf_*` groups keep controlled delay off, its
//! default; the `edf_codel_*` groups repeat a matched subset with it on, which indexes every
//! lane root's due time for expiry. Both modes enqueue and drain the same requests in the same
//! order. Neither expires anything: `pop_next` never runs expiry and no actor timer runs, so a
//! one-second class due time that passes while a case runs changes nothing.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench policy_queue`

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use dynamo_admission::RouterQueuePolicy;
use dynamo_admission::{PolicyProfile, PolicyQueue, QueueMetadata, QueueSnapshot, WorkerPlacement};
use dynamo_kv_router::scheduling::policy_config::RouterPolicyConfig;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use tokio::time::Instant;

dynamo_allocator::jemalloc_global_allocator!();

#[derive(Debug, Clone, Copy)]
struct BenchRequest {
    dispatchable: bool,
}

fn resolve_profile(yaml: &str, queue_policy: RouterQueuePolicy) -> PolicyProfile {
    let profile = RouterPolicyConfig::from_yaml(yaml)
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
    assert_eq!(profile.default_class().queue_policy, queue_policy);
    profile
}

fn profile() -> PolicyProfile {
    static PROFILE: OnceLock<PolicyProfile> = OnceLock::new();
    PROFILE
        .get_or_init(|| {
            resolve_profile(
                r#"
policy_classes:
  - name: bench
    queue_policy: fcfs
    quantum: 1000000
"#,
                RouterQueuePolicy::Fcfs,
            )
        })
        .clone()
}

/// The EDF bench class, with controlled delay on or off.
fn edf_profile(controlled_delay: bool) -> PolicyProfile {
    static PROFILES: [OnceLock<PolicyProfile>; 2] = [OnceLock::new(), OnceLock::new()];
    PROFILES[usize::from(controlled_delay)]
        .get_or_init(|| {
            let profile = resolve_profile(
                &format!(
                    r#"
policy_classes:
  - name: bench
    queue_policy: edf
    quantum: 1000000
    slo_ttft_ms: 1000
    controlled_delay: {controlled_delay}
"#
                ),
                RouterQueuePolicy::Edf,
            );
            assert_eq!(profile.default_class().controlled_delay, controlled_delay);
            profile
        })
        .clone()
}

/// An EDF queue of `requests` entries, one per lane in turn (`None` for the shared heap), built
/// as the scheduler builds its queue. As a router would, each entry's due time is its ingress
/// plus the class budget.
fn edf_queue(
    controlled_delay: bool,
    lanes: Option<usize>,
    requests: usize,
    is_dispatchable: impl Fn(usize) -> bool,
) -> PolicyQueue<BenchRequest> {
    let profile = edf_profile(controlled_delay);
    let mut queue = PolicyQueue::with_class_budget_expiry(&profile);
    let config = queue.class_config(0);
    assert_eq!(config.queue_policy, RouterQueuePolicy::Edf);
    assert_eq!(config.controlled_delay, controlled_delay);
    let class = profile.default_class();
    let ingress_origin = Instant::now();
    for request_index in 0..requests {
        let (lane, placement) = match lanes {
            Some(lanes) => {
                let lane = request_index % lanes;
                (
                    lane,
                    WorkerPlacement::Exact(WorkerWithDpRank::new(lane as u64, 0)),
                )
            }
            None => (0, WorkerPlacement::Any),
        };
        let class_due_at =
            class.slo_ttft_due_at(ingress_origin + Duration::from_micros(request_index as u64));
        queue
            .enqueue_with_due_at(
                QueueMetadata {
                    class_index: 0,
                    snapshot: QueueSnapshot::new(1, 0),
                    due_at: None,
                    class_due_at,
                    arrival_offset_secs: request_index as f64,
                },
                lanes.unwrap_or(1),
                0.0,
                0,
                placement,
                BenchRequest {
                    dispatchable: is_dispatchable(lane),
                },
            )
            .unwrap();
    }
    queue
}

fn exact_queue(
    lanes: usize,
    requests: usize,
    is_dispatchable: impl Fn(usize) -> bool,
) -> PolicyQueue<BenchRequest> {
    let mut queue = PolicyQueue::new(&profile());
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

fn bench_edf_drain_shared(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_queue/edf_drain_shared");
    for requests in [128, 1024, 4096] {
        group.throughput(Throughput::Elements(requests as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(requests),
            &requests,
            |b, &requests| {
                b.iter_batched(
                    || edf_queue(false, None, requests, all),
                    |queue| assert_eq!(black_box(drain(queue)), requests),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_edf_drain_fixed_requests(c: &mut Criterion) {
    const REQUESTS: usize = 4096;
    let mut group = c.benchmark_group("policy_queue/edf_drain_4096_exact");
    group.throughput(Throughput::Elements(REQUESTS as u64));
    for lanes in [1, 8, 32, 128, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(lanes), &lanes, |b, &lanes| {
            b.iter_batched(
                || edf_queue(false, Some(lanes), REQUESTS, all),
                |queue| assert_eq!(black_box(drain(queue)), REQUESTS),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_edf_blocked_fraction(c: &mut Criterion) {
    const LANES: usize = 10_000;
    let mut group = c.benchmark_group("policy_queue/edf_drain_10000_exact_blocked");
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
                    || edf_queue(false, Some(LANES), LANES, predicate),
                    |queue| assert_eq!(black_box(drain(queue)), expected),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Controlled delay on, for a matched subset of the default-off EDF drains above: a shared heap,
/// 32 requests per exact lane, and 9,000 blocked lanes retained behind 1,000 dispatches.
fn bench_edf_codel_drains(c: &mut Criterion) {
    const REQUESTS: usize = 4096;
    const LANES: usize = 10_000;
    let mut group = c.benchmark_group("policy_queue/edf_codel_drain_shared");
    group.throughput(Throughput::Elements(REQUESTS as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(REQUESTS),
        &REQUESTS,
        |b, &requests| {
            b.iter_batched(
                || edf_queue(true, None, requests, all),
                |queue| assert_eq!(black_box(drain(queue)), requests),
                BatchSize::LargeInput,
            );
        },
    );
    group.finish();

    let mut group = c.benchmark_group("policy_queue/edf_codel_drain_4096_exact");
    group.throughput(Throughput::Elements(REQUESTS as u64));
    group.bench_with_input(BenchmarkId::from_parameter(128), &128, |b, &lanes| {
        b.iter_batched(
            || edf_queue(true, Some(lanes), REQUESTS, all),
            |queue| assert_eq!(black_box(drain(queue)), REQUESTS),
            BatchSize::LargeInput,
        );
    });
    group.finish();

    let mut group = c.benchmark_group("policy_queue/edf_codel_drain_10000_exact_blocked");
    group.bench_with_input(
        BenchmarkId::new("90_percent", LANES),
        &(every_tenth as fn(usize) -> bool),
        |b, &predicate| {
            b.iter_batched(
                || edf_queue(true, Some(LANES), LANES, predicate),
                |queue| assert_eq!(black_box(drain(queue)), LANES / 10),
                BatchSize::LargeInput,
            );
        },
    );
    group.finish();
}

/// Builds and drops an EDF queue of 10,000 dispatchable requests, one per exact lane, with
/// controlled delay off and on: the drains above leave construction out of the timed interval,
/// and this times the due-time index that enabling it adds.
fn bench_edf_build_exact_lanes(c: &mut Criterion) {
    const LANES: usize = 10_000;
    for (group, controlled_delay) in [
        ("policy_queue/edf_build_one_per_exact_lane", false),
        ("policy_queue/edf_codel_build_one_per_exact_lane", true),
    ] {
        let mut group = c.benchmark_group(group);
        group.throughput(Throughput::Elements(LANES as u64));
        group.bench_with_input(BenchmarkId::from_parameter(LANES), &LANES, |b, &lanes| {
            b.iter(|| black_box(edf_queue(controlled_delay, Some(lanes), lanes, all)));
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03);
    targets = bench_pop_once, bench_drain_fixed_requests, bench_drain_one_per_lane, bench_build_exact_lanes, bench_blocked_fraction, bench_drain_shared, bench_edf_drain_shared, bench_edf_drain_fixed_requests, bench_edf_blocked_fraction, bench_edf_codel_drains, bench_edf_build_exact_lanes
}
criterion_main!(benches);
