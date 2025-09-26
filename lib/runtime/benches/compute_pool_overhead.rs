// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_runtime::compute::ComputePool;
use std::sync::Arc;

/// Compute-intensive function: sum of all primes up to n
fn compute_primes_sum(n: u64) -> u64 {
    let mut sum = 0u64;
    for candidate in 2..=n {
        if is_prime(candidate) {
            sum += candidate;
        }
    }
    sum
}

fn is_prime(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (5..=sqrt_n).step_by(6) {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
    }
    true
}

fn bench_compute_overhead(c: &mut Criterion) {
    // Test 3 representative sizes: small, medium, large
    let test_sizes = [10, 1_000, 100_000];

    let mut group = c.benchmark_group("compute_overhead");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    // Setup runtimes
    let tokio_4thread = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let tokio_1thread = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();

    // Setup compute pool
    let compute_config = dynamo_runtime::compute::ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "bench".to_string(),
        pin_threads: false,
    };
    let compute_pool = Arc::new(ComputePool::new(compute_config).unwrap());

    for n in test_sizes {
        // Benchmark 1: Direct execution on Tokio (4 threads)
        group.bench_with_input(BenchmarkId::new("tokio_direct", n), &n, |b, &n| {
            b.to_async(&tokio_4thread)
                .iter(|| async move { black_box(compute_primes_sum(black_box(n))) });
        });

        // Benchmark 2: Rayon offload (1 Tokio thread + 4 Rayon threads)
        let pool = compute_pool.clone();
        group.bench_with_input(BenchmarkId::new("rayon_offload", n), &n, |b, &n| {
            b.to_async(&tokio_1thread).iter(|| {
                let pool = pool.clone();
                async move {
                    pool.execute(move || black_box(compute_primes_sum(black_box(n))))
                        .await
                        .unwrap()
                }
            });
        });

        // Benchmark 3: spawn_blocking (4 Tokio threads)
        group.bench_with_input(BenchmarkId::new("spawn_blocking", n), &n, |b, &n| {
            b.to_async(&tokio_4thread).iter(|| async move {
                tokio::task::spawn_blocking(move || black_box(compute_primes_sum(black_box(n))))
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_parallel_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_tasks");
    group.sample_size(10); // Even smaller sample for parallel benchmarks

    let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let compute_config = dynamo_runtime::compute::ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "bench".to_string(),
        pin_threads: false,
    };
    let compute_pool = Arc::new(ComputePool::new(compute_config).unwrap());

    // Test with different batch sizes
    for batch_size in [10, 100] {
        let n = 10_000; // Fixed compute size

        // Direct parallel execution on Tokio threads
        group.bench_with_input(
            BenchmarkId::new("tokio_direct_parallel", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&tokio_runtime).iter(|| async move {
                    let tasks = (0..batch_size)
                        .map(|_| tokio::spawn(async move { compute_primes_sum(n) }))
                        .collect::<Vec<_>>();

                    for task in tasks {
                        black_box(task.await.unwrap());
                    }
                });
            },
        );

        // Parallel execution with Rayon
        let pool = compute_pool.clone();
        group.bench_with_input(
            BenchmarkId::new("rayon_parallel", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&tokio_runtime).iter(|| {
                    let pool = pool.clone();
                    async move {
                        let tasks = (0..batch_size)
                            .map(|_| {
                                let pool = pool.clone();
                                tokio::spawn(async move {
                                    pool.execute(move || compute_primes_sum(n)).await.unwrap()
                                })
                            })
                            .collect::<Vec<_>>();

                        for task in tasks {
                            black_box(task.await.unwrap());
                        }
                    }
                });
            },
        );

        // Parallel execution with spawn_blocking
        group.bench_with_input(
            BenchmarkId::new("spawn_blocking_parallel", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&tokio_runtime).iter(|| async move {
                    let tasks = (0..batch_size)
                        .map(|_| {
                            tokio::spawn(async move {
                                tokio::task::spawn_blocking(move || compute_primes_sum(n))
                                    .await
                                    .unwrap()
                            })
                        })
                        .collect::<Vec<_>>();

                    for task in tasks {
                        black_box(task.await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_block_in_place_overhead(c: &mut Criterion) {
    // Test block_in_place overhead for medium-sized tasks
    let test_sizes = [10, 1_000, 100_000];

    let mut group = c.benchmark_group("block_in_place_overhead");
    group.sample_size(10);

    // Setup 4-thread runtime for testing
    let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();

    // Setup compute pool for comparison
    let compute_config = dynamo_runtime::compute::ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "bench".to_string(),
        pin_threads: false,
    };
    let compute_pool = Arc::new(ComputePool::new(compute_config).unwrap());

    for n in test_sizes {
        // Benchmark 1: Direct execution (baseline)
        group.bench_with_input(BenchmarkId::new("direct", n), &n, |b, &n| {
            b.to_async(&tokio_runtime)
                .iter(|| async move { black_box(compute_primes_sum(black_box(n))) });
        });

        // Benchmark 2: block_in_place (no semaphore)
        group.bench_with_input(BenchmarkId::new("block_in_place", n), &n, |b, &n| {
            b.to_async(&tokio_runtime).iter(|| async move {
                tokio::task::block_in_place(|| black_box(compute_primes_sum(black_box(n))))
            });
        });

        // Benchmark 3: spawn_blocking
        group.bench_with_input(BenchmarkId::new("spawn_blocking", n), &n, |b, &n| {
            b.to_async(&tokio_runtime).iter(|| async move {
                tokio::task::spawn_blocking(move || black_box(compute_primes_sum(black_box(n))))
                    .await
                    .unwrap()
            });
        });

        // Benchmark 4: Rayon offload
        let pool = compute_pool.clone();
        group.bench_with_input(BenchmarkId::new("rayon_offload", n), &n, |b, &n| {
            b.to_async(&tokio_runtime).iter(|| {
                let pool = pool.clone();
                async move {
                    pool.execute(move || black_box(compute_primes_sum(black_box(n))))
                        .await
                        .unwrap()
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_overhead,
    bench_parallel_tasks,
    bench_block_in_place_overhead
);
criterion_main!(benches);
