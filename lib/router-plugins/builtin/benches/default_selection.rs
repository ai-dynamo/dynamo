// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_custom_policy_builtin::default_policy;
use dynamo_kv_router::{KvRouterConfig, WorkerSelector};
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
    let decay: f64 = std::env::var("DYN_BENCH_OVERLAP_DECAY")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    for temperature in [0.0, 0.7] {
        let mut group = c.benchmark_group(format!("default_selection/t{temperature}"));
        group
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(500))
            .sample_size(30)
            .nresamples(1_000);
        for count in [8, 64, 256, 1024, 4096] {
            let (workers, request) = support::fixture(count, 2048);
            let config = KvRouterConfig {
                router_temperature: temperature,
                overlap_score_credit_decay: decay,
                ..Default::default()
            };
            let reference =
                dynamo_kv_router::DefaultWorkerSelector::new(Some(config.clone()), "prefill");
            let plugin = default_policy(config, "prefill");
            let reference_allocs = allocations(|| {
                black_box(
                    reference
                        .select_worker(support::selection_input(&workers, &request, 16))
                        .unwrap(),
                );
            });
            let plugin_allocs = allocations(|| {
                black_box(
                    plugin
                        .select_worker(support::selection_input(&workers, &request, 16))
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
                            .select_worker(support::selection_input(&workers, &request, 16))
                            .unwrap(),
                    )
                })
            });
            group.bench_function(BenchmarkId::new("plugin", count), |b| {
                b.iter(|| {
                    black_box(
                        plugin
                            .select_worker(support::selection_input(&workers, &request, 16))
                            .unwrap(),
                    )
                })
            });
        }
        group.finish();
    }
}

struct LoadScorer(f64);
impl dynamo_kv_router::WorkerScorer for LoadScorer {
    fn required_worker_inputs(&self) -> dynamo_kv_router::WorkerInputs {
        dynamo_kv_router::WorkerInputs::LOAD
    }
    fn score(
        &mut self,
        _: &dynamo_kv_router::WorkerSelectionContext<'_>,
        candidates: dynamo_kv_router::WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), dynamo_kv_router::WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = candidate.load().unwrap().active_requests() as f64 * self.0;
        }
        Ok(())
    }
}

struct MinimumPicker;
impl dynamo_kv_router::WorkerPicker for MinimumPicker {
    fn pick(
        &mut self,
        _: &dynamo_kv_router::WorkerSelectionContext<'_>,
        input: dynamo_kv_router::WorkerInputView<'_>,
    ) -> Result<usize, dynamo_kv_router::WorkerSelectionPolicyError> {
        Ok(input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost().total_cmp(&b.cost()))
            .unwrap()
            .0)
    }
}

fn stacked(c: &mut Criterion) {
    for scorer_count in [1, 2, 4] {
        let mut group = c.benchmark_group(format!("stacked_scoring/s{scorer_count}"));
        group
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(500))
            .sample_size(30)
            .nresamples(1_000);
        for count in [8, 64, 1024, 4096] {
            let (workers, request) = support::fixture(count, 2048);
            let scorers = (0..scorer_count)
                .map(|index| {
                    Box::new(LoadScorer((index + 1) as f64))
                        as Box<dyn dynamo_kv_router::WorkerScorer>
                })
                .collect();
            let policy = dynamo_kv_router::WorkerSelectionPolicy::new(
                KvRouterConfig::default(),
                "prefill",
                scorers,
                Box::new(MinimumPicker),
            );
            let mut select = || {
                black_box(
                    policy
                        .select_worker(support::selection_input(&workers, &request, 16))
                        .unwrap(),
                );
            };
            let count_allocs = allocations(&mut select);
            eprintln!(
                "stacked allocations per 100 warm selections: scorers={scorer_count} workers={count} allocations={count_allocs}"
            );
            group.bench_function(BenchmarkId::new("plugin", count), |b| b.iter(&mut select));
        }
        group.finish();
    }
}

// Shared ranges exercise both tier-aware scoring and the accounting-only fallback.
fn shared_cache(c: &mut Criterion) {
    for tier_matches in [false, true] {
        for range_count in [1, 5, 32] {
            for credit in [0.0, 0.6] {
                let mut group = c.benchmark_group(format!(
                    "shared_cache/tiers{tier_matches}/ranges{range_count}/credit{credit}"
                ));
                group
                    .warm_up_time(Duration::from_millis(200))
                    .measurement_time(Duration::from_millis(500))
                    .sample_size(30)
                    .nresamples(1_000);
                for count in [8, 1024, 4096] {
                    let (workers, mut request) = support::fixture(count, 2048);
                    if !tier_matches {
                        request.overlap.tier_overlap_blocks = Default::default();
                    }
                    request.shared_cache_hits =
                        Some(dynamo_kv_router::SharedCacheHits::from_ranges(
                            (0..range_count).map(|i| i * 4..i * 4 + 2).collect(),
                        ));
                    let policy = default_policy(
                        KvRouterConfig {
                            shared_cache_multiplier: credit,
                            overlap_score_credit_decay: 0.6,
                            ..Default::default()
                        },
                        "prefill",
                    );
                    let mut select = || {
                        black_box(
                            policy
                                .select_worker(support::selection_input(&workers, &request, 16))
                                .unwrap(),
                        );
                    };
                    let count_allocs = allocations(&mut select);
                    eprintln!(
                        "shared allocations per 100 warm selections: tiers={tier_matches} ranges={range_count} credit={credit} workers={count} allocations={count_allocs}"
                    );
                    group
                        .bench_function(BenchmarkId::new("plugin", count), |b| b.iter(&mut select));
                }
                group.finish();
            }
        }
    }
}

criterion_group!(benches, bench, stacked, shared_cache);
criterion_main!(benches);
