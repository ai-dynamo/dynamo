// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "../tests/support/mod.rs"]
mod support;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_custom_policy_builtin::default_policy;
use dynamo_kv_router::protocols::SelectionTraceSpec;
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

    // Selection-trace capture (`DYN_REQUEST_TRACE_RECORDS=router_selection`): the disabled arm
    // is the production default and must match `plugin` above; `top16` is the default capture.
    let mut group = c.benchmark_group("default_selection/trace");
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(30)
        .nresamples(1_000);
    for count in [64, 1024, 4096] {
        for (arm, spec) in [
            ("off", None),
            ("top16", Some(SelectionTraceSpec { top_k: 16 })),
            ("all", Some(SelectionTraceSpec { top_k: 0 })),
        ] {
            let (workers, request) = support::fixture(count, 2048);
            let plugin = default_policy(
                KvRouterConfig {
                    router_temperature: 0.0,
                    ..Default::default()
                },
                "prefill",
            );
            let allocs = allocations(|| {
                black_box(
                    plugin
                        .select_worker_traced(
                            support::selection_input(&workers, &request, 16),
                            spec,
                        )
                        .unwrap(),
                );
            });
            eprintln!(
                "allocations per 100 warm selections: trace={arm} workers={count} plugin={allocs}"
            );
            group.bench_function(BenchmarkId::new(arm, count), |b| {
                b.iter(|| {
                    black_box(
                        plugin
                            .select_worker_traced(
                                support::selection_input(&workers, &request, 16),
                                spec,
                            )
                            .unwrap(),
                    )
                })
            });
        }
    }
    group.finish();
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
                            shared_cache_multiplier: Some(credit),
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

/// A KVCR-style worker: hint capable, a source, and in an island of eight.
/// Like the selection service's config, it caches its hashed keys.
struct HintWorker {
    topology: std::collections::HashMap<String, String>,
    keys: dynamo_kv_router::kv_hints::WorkerKvTransferKeysCache,
}
impl dynamo_kv_router::protocols::WorkerConfigLike for HintWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        2
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(16384)
    }
    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        _dp_rank: u32,
    ) -> Option<dynamo_kv_router::protocols::KvHintTransferWorkerMetadata<'_>> {
        Some(dynamo_kv_router::protocols::KvHintTransferWorkerMetadata {
            worker_type: "prefill",
            source_control_endpoint: Some("tcp://worker:23280"),
        })
    }
    fn topology_domains(&self) -> Option<&std::collections::HashMap<String, String>> {
        Some(&self.topology)
    }
    fn kv_transfer_domain(&self) -> Option<&str> {
        Some("rdma-island")
    }
    fn kv_transfer_keys(
        &self,
    ) -> std::borrow::Cow<'_, dynamo_kv_router::kv_hints::WorkerKvTransferKeys> {
        std::borrow::Cow::Borrowed(self.keys.get_or_init(self))
    }
}

// Every worker rank is a fetch source (S = 2W). `none` carries no transfer
// candidates (no KVCR). `credit0.5` is the worst case: every source holds a
// different prefix, so the summary keeps nearly all of them. In
// `credit0.5_shared_prompt` every rank holds the same system prompt and one
// worker in eight also holds a longer session prefix, so sources no longer
// than every rank's own prefix are skipped.
fn kv_transfer(c: &mut Criterion) {
    for (name, credit, candidates, shared_prompt) in [
        ("none", 0.5, false, false),
        ("credit0", 0.0, true, false),
        ("credit0.5", 0.5, true, false),
        ("credit0.5_shared_prompt", 0.5, true, true),
    ] {
        let mut group = c.benchmark_group(format!("kv_transfer/{name}"));
        group
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(500))
            .sample_size(30)
            .nresamples(1_000);
        for count in [256, 1024, 4096] {
            let (_, mut request) = support::fixture(count, 2048);
            let workers: std::collections::HashMap<u64, HintWorker> = (0..count as u64)
                .map(|id| {
                    let island = format!("island-{}", id / 8);
                    let topology = [("rdma-island".to_string(), island)].into();
                    let keys = Default::default();
                    (id, HintWorker { topology, keys })
                })
                .collect();
            if shared_prompt {
                for blocks in request.overlap.tier_overlap_blocks.device.values_mut() {
                    *blocks = 16;
                }
            }
            if candidates {
                let mut sources: Vec<_> = request
                    .overlap
                    .tier_overlap_blocks
                    .device
                    .iter()
                    .map(|(&worker, &blocks)| {
                        let session = shared_prompt && worker.worker_id % 8 == 0;
                        (worker.into(), if session { 40 } else { blocks + 2 })
                    })
                    .collect();
                sources.sort_unstable();
                request.kv_transfer_candidates =
                    Some(dynamo_kv_router::kv_hints::KvTransferCandidates {
                        block_hashes: (0..128)
                            .map(dynamo_kv_router::protocols::ExternalSequenceBlockHash)
                            .collect(),
                        owner_prefix_blocks: sources,
                        routing_snapshot: None,
                    });
            }
            let policy = default_policy(
                KvRouterConfig {
                    shared_cache_multiplier: Some(credit),
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
            group.bench_function(BenchmarkId::new("plugin", count), |b| b.iter(&mut select));
        }
        group.finish();
    }
}

criterion_group!(benches, bench, stacked, shared_cache, kv_transfer);
criterion_main!(benches);
