// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(feature = "testing-cuda")]
mod benchmarks {
    use std::sync::Arc;

    use criterion::{criterion_group, BenchmarkId, Criterion};
    use cudarc::driver::CudaContext;
    use rand;
    use tokio::runtime::Runtime;
    use tokio_util::task::TaskTracker;

    use dynamo_llm::block_manager::block::transfer::context;

    fn setup_transfer_context() -> context::v2::TransferContext {
        let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
        let stream = ctx.default_stream();
        let nixl_agent = Arc::new(None);
        let handle = tokio::runtime::Handle::try_current()
            .unwrap_or_else(|_| {
                // If no current runtime, create one for the benchmark setup
                Runtime::new().unwrap().handle().clone()
            });

        context::v2::TransferContext::new(nixl_agent, stream, handle)
    }

    /// Benchmark blocking synchronization in tight loop
    /// This measures the baseline performance of direct CUDA event sync
    fn bench_blocking(c: &mut Criterion) {
        let ctx = setup_transfer_context();

        c.bench_function("blocking_sync", |b| {
            b.iter(|| {
                let total_start = std::time::Instant::now();

                let event_start = std::time::Instant::now();
                let event = ctx.record_event().unwrap();
                let event_elapsed = event_start.elapsed();

                let sync_start = std::time::Instant::now();
                event.synchronize_blocking().unwrap();
                let sync_elapsed = sync_start.elapsed();

                let total_elapsed = total_start.elapsed();

                // Occasionally log timing breakdown (every 1000th iteration)
                if rand::random::<u32>() % 1000 == 0 {
                    eprintln!("[BLOCKING] Total: {:?}, Record: {:?}, Sync: {:?}",
                             total_elapsed, event_elapsed, sync_elapsed);
                }
            })
        });
    }


    /// Benchmark single-threaded async synchronization
    /// This measures only the tokio spawn_blocking overhead vs direct blocking
    fn bench_async_single(c: &mut Criterion) {
        let rt = Runtime::new().unwrap();

        // Create CUDA context ONCE outside the benchmark loop (same as blocking benchmark)
        let (_cuda_ctx, stream, nixl_agent) = rt.block_on(async {
            let cuda_ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
            let stream = cuda_ctx.default_stream();
            let nixl_agent = Arc::new(None);
            (cuda_ctx, stream, nixl_agent)
        });

        c.bench_function("async_sync", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let total_start = std::time::Instant::now();

                    // Only create TransferContext and do the actual work
                    let handle = tokio::runtime::Handle::current();
                    let ctx_start = std::time::Instant::now();
                    let ctx = context::v2::TransferContext::new(nixl_agent.clone(), stream.clone(), handle);
                    let ctx_elapsed = ctx_start.elapsed();

                    let event_start = std::time::Instant::now();
                    let event = ctx.record_event().unwrap();
                    let event_elapsed = event_start.elapsed();

                    let sync_start = std::time::Instant::now();
                    event.synchronize().await.unwrap();
                    let sync_elapsed = sync_start.elapsed();

                    let total_elapsed = total_start.elapsed();

                    // Occasionally log timing breakdown (every 1000th iteration)
                    if rand::random::<u32>() % 1000 == 0 {
                        eprintln!("[ASYNC] Total: {:?}, Context: {:?}, Record: {:?}, Sync: {:?}",
                                 total_elapsed, ctx_elapsed, event_elapsed, sync_elapsed);
                    }
                })
            })
        });
    }

    /// Benchmark concurrent async synchronization at different scales
    /// This shows where async becomes beneficial due to parallelism
    fn bench_concurrent_async(c: &mut Criterion) {
        let rt = Runtime::new().unwrap();
        let mut group = c.benchmark_group("concurrent_async");

        // Test different concurrency levels
        for concurrency in [1, 5, 10, 25, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new("concurrent", concurrency),
                concurrency,
                |b, &concurrency| {
                    b.iter(|| {
                        rt.block_on(async {
                            // Create context inside the runtime
                            let ctx_arc = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
                            let stream = ctx_arc.default_stream();
                            let nixl_agent = Arc::new(None);
                            let handle = tokio::runtime::Handle::current();
                            let ctx = context::v2::TransferContext::new(nixl_agent, stream, handle);

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
                }
            );
        }

        group.finish();
    }

    /// Benchmark throughput: events per second at different concurrency levels
    fn bench_throughput(c: &mut Criterion) {
        let rt = Runtime::new().unwrap();
        let mut group = c.benchmark_group("throughput");
        group.sample_size(50); // Fewer samples for throughput tests

        for concurrency in [1, 10, 50].iter() {
            let events_per_task = 10; // Process multiple events per task

            group.bench_with_input(
                BenchmarkId::new("events_per_sec", concurrency),
                concurrency,
                |b, &concurrency| {
                    b.iter(|| {
                        rt.block_on(async {
                            // Create context inside the runtime
                            let ctx_arc = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
                            let stream = ctx_arc.default_stream();
                            let nixl_agent = Arc::new(None);
                            let handle = tokio::runtime::Handle::current();
                            let ctx = context::v2::TransferContext::new(nixl_agent, stream, handle);

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
                }
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
    println!("Benchmarks require 'testing-cuda' feature. Run with: cargo bench --features testing-cuda");
}