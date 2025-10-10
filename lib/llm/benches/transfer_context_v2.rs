// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "testing-cuda")]
mod benchmarks {
    use std::sync::Arc;

    use criterion::{BenchmarkId, Criterion, criterion_group};
    use cudarc::driver::{CudaContext, CudaStream};
    use nixl_sys;
    use tokio::runtime::Runtime;
    use tokio_util::task::TaskTracker;

    use dynamo_llm::block_manager::block::transfer::context;

    struct BenchmarkRuntime {
        _runtime: Runtime,
        handle: tokio::runtime::Handle,
        stream: Arc<CudaStream>,
        nixl_agent: Arc<Option<nixl_sys::Agent>>,
    }

    impl BenchmarkRuntime {
        fn new() -> Self {
            let runtime = Runtime::new().expect("Failed to create benchmark runtime");
            let handle = runtime.handle().clone();

            let cuda_ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
            let stream = cuda_ctx.default_stream();
            let nixl_agent = Arc::new(None);

            Self {
                _runtime: runtime,
                handle,
                stream,
                nixl_agent,
            }
        }

        fn create_transfer_context(&self) -> context::v2::TransferContext {
            context::v2::TransferContext::new(
                self.nixl_agent.clone(),
                self.stream.clone(),
                self.handle.clone(),
            )
        }
    }

    /// Benchmark blocking synchronization in tight loop
    /// This measures the baseline performance of direct CUDA event sync
    fn bench_blocking(c: &mut Criterion) {
        let runtime = BenchmarkRuntime::new();
        let ctx = runtime.create_transfer_context();

        let mut group = c.benchmark_group("blocking_sync");
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(3));

        group.bench_function("sync", |b| {
            b.iter(|| {
                let event = ctx.record_event().unwrap();
                event.synchronize_blocking().unwrap();
            })
        });

        group.finish();
    }

    /// Benchmark single-threaded async synchronization
    /// This measures only the tokio spawn_blocking overhead vs direct blocking
    fn bench_async_single(c: &mut Criterion) {
        let runtime = BenchmarkRuntime::new();
        let ctx = runtime.create_transfer_context();

        let mut group = c.benchmark_group("async_sync");
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(3));

        group.bench_function("sync", |b| {
            b.iter(|| {
                runtime._runtime.block_on(async {
                    let event = ctx.record_event().unwrap();
                    event.synchronize().await.unwrap();
                })
            })
        });

        group.finish();
    }

    /// Benchmark concurrent async synchronization at different scales
    /// This shows where async becomes beneficial due to parallelism
    fn bench_concurrent_async(c: &mut Criterion) {
        let runtime = BenchmarkRuntime::new();
        let mut group = c.benchmark_group("concurrent_async");
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(3));

        // Test different concurrency levels
        for concurrency in [1, 5, 10, 25, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new("concurrent", concurrency),
                concurrency,
                |b, &concurrency| {
                    let ctx = runtime.create_transfer_context();
                    b.iter(|| {
                        runtime._runtime.block_on(async {
                            // Spawn concurrent tasks using TaskTracker
                            let tracker = TaskTracker::new();

                            for _ in 0..concurrency {
                                let ctx_clone = ctx.clone();
                                tracker.spawn(async move {
                                    let event = ctx_clone.record_event().unwrap();
                                    event.synchronize().await.unwrap();
                                });
                            }

                            // Wait for all tasks to complete
                            tracker.close();
                            tracker.wait().await;
                        });
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark throughput: events per second at different concurrency levels
    fn bench_throughput(c: &mut Criterion) {
        let runtime = BenchmarkRuntime::new();
        let mut group = c.benchmark_group("throughput");
        group.sample_size(50); // Fewer samples for throughput tests
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(3));

        for concurrency in [1, 10, 50].iter() {
            let events_per_task = 10; // Process multiple events per task

            group.bench_with_input(
                BenchmarkId::new("events_per_sec", concurrency),
                concurrency,
                |b, &concurrency| {
                    let ctx = runtime.create_transfer_context();
                    b.iter(|| {
                        runtime._runtime.block_on(async {
                            let tracker = TaskTracker::new();

                            for _ in 0..concurrency {
                                let ctx_clone = ctx.clone();
                                tracker.spawn(async move {
                                    // Process multiple events per task
                                    for _ in 0..events_per_task {
                                        let event = ctx_clone.record_event().unwrap();
                                        event.synchronize().await.unwrap();
                                    }
                                });
                            }

                            tracker.close();
                            tracker.wait().await;
                        });
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group!(
        benches,
        // Core comparison benchmarks
        bench_blocking,
        bench_async_single,
        // Concurrency benchmarks
        bench_concurrent_async,
        bench_throughput
    );
}

#[cfg(feature = "testing-cuda")]
criterion::criterion_main!(benchmarks::benches);

#[cfg(not(feature = "testing-cuda"))]
fn main() {
    println!(
        "Benchmarks require 'testing-cuda' feature. Run with: cargo bench --features testing-cuda"
    );
}
