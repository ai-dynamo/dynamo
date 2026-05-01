// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmarks for the runtime mpsc channel wrapper.
//!
//! Default Tokio backend:
//! `cargo bench -p dynamo-runtime --bench channel_mpsc_perf`
//!
//! Flume backend:
//! `cargo bench -p dynamo-runtime --features flume-channels --bench channel_mpsc_perf`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_runtime::channels::mpsc;
use std::time::{Duration, Instant};

const MESSAGES_PER_ITER: u64 = 16 * 1024;
const MPSC_MESSAGES_PER_PRODUCER: u64 = 4 * 1024;
const WARM_UP_TIME: Duration = Duration::from_secs(10);
const MEASUREMENT_TIME: Duration = Duration::from_secs(30);

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("failed to build benchmark runtime")
}

async fn bounded_spsc(capacity: usize, messages: u64) {
    let (tx, mut rx) = mpsc::channel::<u64>(capacity);

    let receiver = tokio::spawn(async move {
        let mut checksum = 0_u64;
        for _ in 0..messages {
            checksum = checksum.wrapping_add(rx.recv().await.expect("channel closed"));
        }
        checksum
    });

    for i in 0..messages {
        tx.send(i).await.expect("receiver dropped");
    }

    black_box(receiver.await.expect("receiver task failed"));
}

async fn bounded_mpsc(capacity: usize, producers: u64, messages_per_producer: u64) {
    let (tx, mut rx) = mpsc::channel::<u64>(capacity);
    let total_messages = producers * messages_per_producer;

    let receiver = tokio::spawn(async move {
        let mut checksum = 0_u64;
        for _ in 0..total_messages {
            checksum = checksum.wrapping_add(rx.recv().await.expect("channel closed"));
        }
        checksum
    });

    let senders = (0..producers)
        .map(|producer| {
            let tx = tx.clone();
            tokio::spawn(async move {
                for i in 0..messages_per_producer {
                    tx.send((producer << 32) | i)
                        .await
                        .expect("receiver dropped");
                }
            })
        })
        .collect::<Vec<_>>();
    drop(tx);

    for sender in senders {
        sender.await.expect("sender task failed");
    }

    black_box(receiver.await.expect("receiver task failed"));
}

async fn unbounded_spsc(messages: u64) {
    let (tx, mut rx) = mpsc::unbounded_channel::<u64>();

    let receiver = tokio::spawn(async move {
        let mut checksum = 0_u64;
        for _ in 0..messages {
            checksum = checksum.wrapping_add(rx.recv().await.expect("channel closed"));
        }
        checksum
    });

    for i in 0..messages {
        tx.send(i).expect("receiver dropped");
    }

    black_box(receiver.await.expect("receiver task failed"));
}

fn bench_bounded_spsc(c: &mut Criterion) {
    let rt = runtime();
    let mut group = c.benchmark_group(format!("mpsc_bounded_spsc_{}", mpsc::backend_name()));
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(MESSAGES_PER_ITER));

    for capacity in [1, 2, 4, 64, 128, 256, 1024, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.to_async(&rt).iter_custom(|iters| async move {
                    let start = Instant::now();
                    for _ in 0..iters {
                        bounded_spsc(capacity, MESSAGES_PER_ITER).await;
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

fn bench_bounded_mpsc(c: &mut Criterion) {
    let rt = runtime();
    let producers = 4;
    let total_messages = producers * MPSC_MESSAGES_PER_PRODUCER;
    let mut group = c.benchmark_group(format!("mpsc_bounded_mpsc_{}", mpsc::backend_name()));
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(total_messages));

    for capacity in [64, 128, 256, 1024, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.to_async(&rt).iter_custom(|iters| async move {
                    let start = Instant::now();
                    for _ in 0..iters {
                        bounded_mpsc(capacity, producers, MPSC_MESSAGES_PER_PRODUCER).await;
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

fn bench_unbounded_spsc(c: &mut Criterion) {
    let rt = runtime();
    let mut group = c.benchmark_group(format!("mpsc_unbounded_spsc_{}", mpsc::backend_name()));
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(MESSAGES_PER_ITER));

    group.bench_function("send_recv", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let start = Instant::now();
            for _ in 0..iters {
                unbounded_spsc(MESSAGES_PER_ITER).await;
            }
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bounded_spsc,
    bench_bounded_mpsc,
    bench_unbounded_spsc
);
criterion_main!(benches);
