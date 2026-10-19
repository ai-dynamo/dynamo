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

fn progress_group(c: &mut Criterion) {
    use dynamo_custom_policy_builtin::{default_registry, register};
    use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{
        OverlapSignals, SessionContext, decode_progress::DecodeProgressUpdate,
        queue::SchedulerBookingDescriptor,
    };
    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, NoopSequencePublisher, RoutingPartitionRef, SequenceRequest,
        WorkerLoadProjection, WorkerSelectionInput, WorkerType,
    };
    use std::collections::HashMap;

    struct SingleRankWorker;
    impl WorkerConfigLike for SingleRankWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(16384)
        }
    }
    let (_, mut request) = support::fixture(2, 16_385);
    let source = WorkerWithDpRank::from_worker_id(101);
    let other = WorkerWithDpRank::from_worker_id(103);
    request.overlap = OverlapSignals::default();
    request.worker_loads.clear();
    request.affinity_target = Some(WorkerAffinityTarget::from(source));
    request.session_context = Some(SessionContext::new(
        "child".into(),
        Some("parent".into()),
        None,
        None,
    ));
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(source, 256);
    request.overlap.tier_overlap_blocks.device.insert(other, 0);
    request.worker_loads.insert(
        source,
        WorkerLoadProjection {
            active_requests: 8,
            ..Default::default()
        },
    );
    request
        .worker_loads
        .insert(other, WorkerLoadProjection::default());
    let workers = HashMap::from([(101, SingleRankWorker), (103, SingleRankWorker)]);
    let slots = ActiveSequencesMultiWorker::new_without_expiry(
        NoopSequencePublisher,
        64,
        HashMap::from([(101, (0, 1)), (103, (0, 1))]),
        false,
        1,
        "aggregated",
    );
    for i in 0..8 {
        let request_id = format!("progress-bench-{i}");
        let now = tokio::time::Instant::now();
        let attempt_id = slots
            .add_request_if_registered_admitted(
                SequenceRequest {
                    request_id: request_id.clone(),
                    token_sequence: None,
                    expected_output_tokens: None,
                    track_prefill_tokens: false,
                    prefill_load_hint: None,
                    worker: source,
                    lora_name: None,
                },
                now,
            )
            .unwrap();
        slots.mark_prefill_completed(&request_id, now).unwrap();
        slots
            .update_decode_progress_for_benchmark(
                &SchedulerBookingDescriptor {
                    request_id,
                    worker: source,
                    attempt_id,
                },
                DecodeProgressUpdate::observed(128, 8),
            )
            .unwrap();
    }
    let provider = |worker| slots.decode_progress_for_benchmark(worker);
    let mut group = c.benchmark_group("progress_group_selection");
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(30)
        .nresamples(1_000);
    for enabled in [false, true] {
        let mut yaml = String::from(
            "worker_selection:\n  aggregated: progress-bench\n  instances:\n    - name: progress-bench\n      type: dynamo-two-tier-cost-fn\n",
        );
        // 60s keeps these unchanged CPU-only observations fresh while measuring selection.
        if enabled {
            yaml.push_str("      parameters:\n        progress_group:\n          source_active_requests: 8\n          min_prefix_tokens: 16384\n          min_remaining_tokens: 256\n          max_progress_age_ms: 60000\n");
        }
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            router_track_output_blocks: true,
            ..Default::default()
        };
        let mut registry = default_registry();
        register(&mut registry).unwrap();
        let factory = registry.resolve(&config).unwrap().unwrap();
        let policy = factory(
            &config,
            WorkerType::Aggregated,
            RoutingPartitionRef::new("bench", "default"),
        );
        let select = || {
            policy
                .select_worker_with_decode_progress(
                    WorkerSelectionInput::configured(&workers, &request, request.eligibility(), 64),
                    None,
                    Some(&provider),
                )
                .unwrap()
                .0
        };
        assert_eq!(select().worker, if enabled { other } else { source });
        let allocations = allocations(|| {
            black_box(select());
        });
        eprintln!(
            "progress-group allocations per 100 selections: enabled={enabled} allocations={allocations}"
        );
        group.bench_function(if enabled { "on" } else { "off" }, |b| {
            b.iter(|| black_box(select()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench, stacked, shared_cache, progress_group);
criterion_main!(benches);
