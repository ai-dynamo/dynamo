// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark comparing spawn_blocking vs loom's spawn_adaptive scheduling.
//!
//! Run with: cargo bench --features loom-runtime --bench loom_vs_current

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;

/// Simulated CPU-bound work of varying duration.
fn cpu_work(iterations: u64) -> u64 {
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(i.wrapping_mul(i));
        // Prevent optimization
        std::hint::black_box(&sum);
    }
    sum
}

/// Approximate iteration counts for target durations:
/// - 10μs:  ~1_000 iterations
/// - 100μs: ~10_000 iterations
/// - 1ms:   ~100_000 iterations
/// - 10ms:  ~1_000_000 iterations
const WORK_SIZES: [(u64, &str); 4] = [
    (1_000, "10us"),
    (10_000, "100us"),
    (100_000, "1ms"),
    (1_000_000, "10ms"),
];

fn bench_spawn_blocking(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_blocking");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(5));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    for (iterations, label) in WORK_SIZES {
        group.bench_with_input(
            BenchmarkId::new("tokio_spawn_blocking", label),
            &iterations,
            |b, &iters| {
                b.to_async(&rt).iter(|| async move {
                    let result = tokio::task::spawn_blocking(move || cpu_work(iters))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "loom-runtime")]
fn bench_loom_adaptive(c: &mut Criterion) {
    use loom_rs::LoomBuilder;

    let mut group = c.benchmark_group("loom_adaptive");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(5));

    let loom = LoomBuilder::new()
        .prefix("bench")
        .tokio_threads(4)
        .rayon_threads(4)
        .build()
        .expect("Failed to create loom runtime");

    for (iterations, label) in WORK_SIZES {
        group.bench_with_input(
            BenchmarkId::new("loom_spawn_adaptive", label),
            &iterations,
            |b, &iters| {
                b.iter(|| {
                    let i = iters; // Copy for move into closure
                    let result = loom.block_on(async { loom.spawn_adaptive(move || cpu_work(i)).await });
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("loom_spawn_compute", label),
            &iterations,
            |b, &iters| {
                b.iter(|| {
                    let i = iters; // Copy for move into closure
                    let result = loom.block_on(async { loom.spawn_compute(move || cpu_work(i)).await });
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "loom-runtime"))]
fn bench_loom_adaptive(_c: &mut Criterion) {
    // No-op when loom-runtime feature is not enabled
    eprintln!("Loom benchmarks require --features loom-runtime");
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocking_comparison");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    // Use 100μs work size for fair comparison
    let iterations = 10_000u64;

    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    // Tokio spawn_blocking
    group.bench_function("tokio_spawn_blocking_100us", |b| {
        b.to_async(&tokio_rt).iter(|| async move {
            let result = tokio::task::spawn_blocking(move || cpu_work(iterations))
                .await
                .unwrap();
            black_box(result)
        });
    });

    // Tokio-rayon (what loom uses internally)
    group.bench_function("tokio_rayon_100us", |b| {
        b.to_async(&tokio_rt).iter(|| async move {
            let result = tokio_rayon::spawn(move || cpu_work(iterations)).await;
            black_box(result)
        });
    });

    #[cfg(feature = "loom-runtime")]
    {
        use loom_rs::LoomBuilder;

        let loom = LoomBuilder::new()
            .prefix("bench")
            .tokio_threads(4)
            .rayon_threads(4)
            .build()
            .expect("Failed to create loom runtime");

        // Loom spawn_adaptive
        group.bench_function("loom_spawn_adaptive_100us", |b| {
            let iters = iterations;
            b.iter(|| {
                let result = loom.block_on(async { loom.spawn_adaptive(move || cpu_work(iters)).await });
                black_box(result)
            });
        });

        // Loom spawn_compute (always offload)
        group.bench_function("loom_spawn_compute_100us", |b| {
            let iters = iterations;
            b.iter(|| {
                let result = loom.block_on(async { loom.spawn_compute(move || cpu_work(iters)).await });
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark starvation scenario: many blocking tasks competing with async I/O.
fn bench_starvation_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("starvation_scenario");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    // Spawn many blocking tasks and measure how long an async "I/O" task takes
    group.bench_function("tokio_under_load", |b| {
        b.to_async(&tokio_rt).iter(|| async {
            // Spawn 20 blocking tasks
            let blocking_handles: Vec<_> = (0..20)
                .map(|_| tokio::task::spawn_blocking(move || cpu_work(100_000)))
                .collect();

            // Measure how long a simple async operation takes
            let start = std::time::Instant::now();
            tokio::time::sleep(Duration::from_micros(100)).await;
            let io_latency = start.elapsed();

            // Wait for blocking tasks
            for h in blocking_handles {
                let _ = h.await;
            }

            black_box(io_latency)
        });
    });

    #[cfg(feature = "loom-runtime")]
    {
        use loom_rs::LoomBuilder;

        let loom = LoomBuilder::new()
            .prefix("bench")
            .tokio_threads(4)
            .rayon_threads(4)
            .build()
            .expect("Failed to create loom runtime");

        group.bench_function("loom_under_load", |b| {
            b.iter(|| {
                loom.block_on(async {
                    // Spawn 20 compute tasks on rayon (doesn't block tokio)
                    let compute_handles: Vec<_> = (0..20)
                        .map(|_| {
                            let loom_ref = &loom;
                            async move { loom_ref.spawn_compute(|| cpu_work(100_000)).await }
                        })
                        .collect();

                    // Measure how long a simple async operation takes
                    let start = std::time::Instant::now();
                    tokio::time::sleep(Duration::from_micros(100)).await;
                    let io_latency = start.elapsed();

                    // Wait for compute tasks
                    for h in compute_handles {
                        let _ = h.await;
                    }

                    black_box(io_latency)
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_spawn_blocking,
    bench_loom_adaptive,
    bench_comparison,
    bench_starvation_scenario,
);
criterion_main!(benches);
