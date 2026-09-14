// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_custom_policy_builtin::default_policy;
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionInput, WorkerSelector};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
// SAFETY: every operation delegates to System with the original pointer and layout.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
fn allocations(mut select: impl FnMut()) -> usize {
    select(); // Warm retained buffers and thread-local RNG.
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        select();
    }
    COUNTING.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn bench(c: &mut Criterion) {
    for temperature in [0.0, 0.7] {
        let mut group = c.benchmark_group(format!("default_selection/t{temperature}"));
        group
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(500))
            .sample_size(20);
        for count in [8, 64, 256, 1024] {
            let (workers, request) = support::fixture(count, 2048);
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let reference =
                dynamo_kv_router::DefaultWorkerSelector::new(Some(config.clone()), "prefill");
            let plugin = default_policy(config, "prefill");
            let reference_allocs = allocations(|| {
                black_box(
                    reference
                        .select_worker(WorkerSelectionInput::configured(
                            &workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap(),
                );
            });
            let plugin_allocs = allocations(|| {
                black_box(
                    plugin
                        .select_worker(WorkerSelectionInput::configured(
                            &workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap(),
                );
            });
            eprintln!(
                "allocations per 100 warm selections: temperature={temperature} workers={count} reference={reference_allocs} plugin={plugin_allocs}"
            );
            group.bench_function(BenchmarkId::new("reference", count), |b| {
                b.iter(|| {
                    black_box(
                        reference
                            .select_worker(WorkerSelectionInput::configured(
                                &workers,
                                &request,
                                request.eligibility(),
                                16,
                            ))
                            .unwrap(),
                    )
                })
            });
            group.bench_function(BenchmarkId::new("plugin", count), |b| {
                b.iter(|| {
                    black_box(
                        plugin
                            .select_worker(WorkerSelectionInput::configured(
                                &workers,
                                &request,
                                request.eligibility(),
                                16,
                            ))
                            .unwrap(),
                    )
                })
            });
        }
        group.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
