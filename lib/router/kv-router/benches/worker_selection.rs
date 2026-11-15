// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection scaling benchmark for built-in and custom policies.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench worker_selection`

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_events::event::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
    RouterEvent,
};
use dynamo_kv_hashing::routing::compute_seq_hash_for_block;
use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
use dynamo_kv_indexer::ResidentBlockCountsHandle;
use dynamo_kv_indexer::host::{
    ConcurrentRadixTreeCompressed, KvIndexerInterface, ThreadPoolIndexer,
};
use dynamo_router_protocol::columns::WorkerInputs;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::ScheduleMode;
use dynamo_scheduler::SchedulingRequest;
use dynamo_scheduler::overlap::OverlapSignals;
use dynamo_scheduler::protocols::{RoutingConstraints, WorkerConfigLike};
use dynamo_scheduler::selector::{DefaultWorkerSelector, WorkerSelectionInput, WorkerSelector};
use dynamo_slot_tracker::WorkerLoadProjection;
use dynamo_worker_selection::experimental::resident_blocks::{RESIDENT_BLOCKS, ResidentBlocks};
use dynamo_worker_selection::{
    SelectionConfig, WorkerCandidate, WorkerCandidates, WorkerFilter, WorkerInputView,
    WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};
use rustc_hash::FxHashMap;

#[derive(Default)]
struct BenchWorkerConfig {
    taints: HashSet<String>,
}

impl WorkerConfigLike for BenchWorkerConfig {
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
        Some(131_072)
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}

struct KeepAllFilter;

impl WorkerFilter for KeepAllFilter {
    fn keep(
        &self,
        _context: &WorkerSelectionContext<'_>,
        _candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        Ok(true)
    }
}

/// Reads every worker's static info and keeps the workers that report KV blocks.
struct CapacityFilter;

impl WorkerFilter for CapacityFilter {
    fn keep(
        &self,
        _context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let info = candidate.info();
        Ok(info.total_kv_blocks().is_some_and(|blocks| blocks > 0)
            && info.max_num_batched_tokens() != Some(0))
    }
}

struct BenchScorer;

impl WorkerScorer for BenchScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            let uncached_blocks =
                (context.request_blocks() as f64 - cache.device_overlap_blocks()).max(0.0);
            let load_blocks = load.active_prefill_tokens() as f64 / context.block_size() as f64
                + load.decode_cost_blocks()
                + load.active_requests() as f64;
            *cost = (uncached_blocks + load_blocks)
                * candidate.preferred_taint_multiplier().unwrap_or(1.0);
        }
        Ok(())
    }
}

/// Matches normal custom-policy cache/load work while deliberately not observing preferred-taint
/// metadata. This isolates the cost of materializing metadata that a policy does not use.
struct IgnoringPreferenceScorer;

impl WorkerScorer for IgnoringPreferenceScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            let uncached_blocks =
                (context.request_blocks() as f64 - cache.device_overlap_blocks()).max(0.0);
            let load_blocks = load.active_prefill_tokens() as f64 / context.block_size() as f64
                + load.decode_cost_blocks()
                + load.active_requests() as f64;
            *cost = uncached_blocks + load_blocks;
        }
        Ok(())
    }
}

/// `IgnoringPreferenceScorer` plus each candidate's resident block count.
struct ResidentBlocksScorer;

impl WorkerScorer for ResidentBlocksScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | RESIDENT_BLOCKS
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            // Real policies should treat None (no fresh counts) as neutral; failing keeps the bench honest.
            let resident_blocks = candidate.resident_blocks().ok_or_else(|| {
                WorkerSelectionPolicyError::failed("resident blocks input unavailable")
            })?;
            let uncached_blocks =
                (context.request_blocks() as f64 - cache.device_overlap_blocks()).max(0.0);
            let load_blocks = load.active_prefill_tokens() as f64 / context.block_size() as f64
                + load.decode_cost_blocks()
                + load.active_requests() as f64;
            *cost = uncached_blocks + load_blocks + resident_blocks as f64;
        }
        Ok(())
    }
}

struct LowestCostPicker;

impl WorkerPicker for LowestCostPicker {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn fixture(worker_count: usize) -> (HashMap<WorkerId, BenchWorkerConfig>, SchedulingRequest) {
    fixture_with_preferred_taints(worker_count, 0)
}

fn fixture_with_preferred_taints(
    worker_count: usize,
    preferred_taint_count: usize,
) -> (HashMap<WorkerId, BenchWorkerConfig>, SchedulingRequest) {
    const PREFERRED_TAINTS: [&str; 4] = ["rack-a", "zone-a", "gpu-a", "node-a"];
    assert!(preferred_taint_count <= PREFERRED_TAINTS.len());
    let mut workers = HashMap::with_capacity(worker_count);
    let mut effective_overlap_blocks =
        FxHashMap::with_capacity_and_hasher(worker_count, Default::default());
    let mut effective_cached_tokens =
        FxHashMap::with_capacity_and_hasher(worker_count, Default::default());
    let mut worker_loads = FxHashMap::with_capacity_and_hasher(worker_count, Default::default());

    for worker_id in 0..worker_count as WorkerId {
        let worker = WorkerWithDpRank::from_worker_id(worker_id);
        let cached_tokens = (worker_id as usize % 32) * 64;
        let taints = if worker_id % 2 == 0 {
            PREFERRED_TAINTS
                .iter()
                .take(preferred_taint_count)
                .map(|taint| (*taint).to_string())
                .collect()
        } else {
            HashSet::new()
        };
        workers.insert(worker_id, BenchWorkerConfig { taints });
        effective_overlap_blocks.insert(worker, cached_tokens as f64 / 16.0);
        effective_cached_tokens.insert(worker, cached_tokens);
        worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_prefill_tokens: worker_id as usize * 16,
                active_decode_blocks: worker_id as usize % 127,
                active_requests: worker_id as usize % 17,
                ..Default::default()
            },
        );
    }

    let request = SchedulingRequest {
        accounting_attribution: Default::default(),
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 2_048,
        lora_name: None,
        expected_output_tokens: Some(256),
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints {
            required_taints: HashSet::new(),
            preferred_taints: PREFERRED_TAINTS
                .iter()
                .take(preferred_taint_count)
                .map(|taint| ((*taint).to_string(), 0.5))
                .collect(),
        },
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens,
        },
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads,
        resp_tx: None,
    };

    (workers, request)
}

fn worker_selection(c: &mut Criterion) {
    for (scenario, overlap_score_credit_decay) in
        [("direct", 0.0), ("filtered_prefill_baseline", 1.0)]
    {
        for temperature in [0.0, 0.7] {
            let mut group = c.benchmark_group(format!(
                "default_worker_selection/{scenario}/temperature_{temperature}"
            ));
            group.warm_up_time(Duration::from_secs(2));
            group.measurement_time(Duration::from_secs(5));
            group.sample_size(50);

            for worker_count in [2, 32, 1_024, 2_048, 10_000] {
                let (workers, request) = fixture(worker_count);
                let selector = DefaultWorkerSelector::new(
                    Some(SelectionConfig {
                        router_temperature: temperature,
                        overlap_score_credit_decay,
                        ..Default::default()
                    }),
                    "prefill",
                );
                group.throughput(Throughput::Elements(worker_count as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(worker_count),
                    &worker_count,
                    |b, _| {
                        b.iter(|| {
                            black_box(
                                selector
                                    .select_worker(WorkerSelectionInput::configured(
                                        black_box(&workers),
                                        black_box(&request),
                                        request.eligibility(),
                                        black_box(16),
                                    ))
                                    .unwrap(),
                            )
                        })
                    },
                );
            }
            group.finish();
        }
    }
}

fn custom_worker_selection(c: &mut Criterion) {
    const WORKER_COUNT: usize = 10_000;

    let (workers, request) = fixture(WORKER_COUNT);
    let config = SelectionConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let custom_no_filter = WorkerSelectionPolicy::new(
        config.clone(),
        "prefill",
        vec![Box::new(BenchScorer)],
        Box::new(LowestCostPicker),
    );
    let custom_keep_all_filter = WorkerSelectionPolicy::new_with_filters(
        config.clone(),
        "prefill",
        vec![Box::new(KeepAllFilter)],
        vec![Box::new(BenchScorer)],
        Box::new(LowestCostPicker),
    );
    let custom_capacity_filter = WorkerSelectionPolicy::new_with_filters(
        config,
        "prefill",
        vec![Box::new(CapacityFilter)],
        vec![Box::new(BenchScorer)],
        Box::new(LowestCostPicker),
    );
    let mut group = c.benchmark_group("custom_worker_selection/10_000");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    group.throughput(Throughput::Elements(WORKER_COUNT as u64));

    group.bench_function("custom_no_filter", |b| {
        b.iter(|| {
            black_box(
                custom_no_filter
                    .select_worker(WorkerSelectionInput::configured(
                        black_box(&workers),
                        black_box(&request),
                        request.eligibility(),
                        black_box(16),
                    ))
                    .unwrap(),
            )
        })
    });
    group.bench_function("custom_keep_all_filter", |b| {
        b.iter(|| {
            black_box(
                custom_keep_all_filter
                    .select_worker(WorkerSelectionInput::configured(
                        black_box(&workers),
                        black_box(&request),
                        request.eligibility(),
                        black_box(16),
                    ))
                    .unwrap(),
            )
        })
    });
    group.bench_function("custom_capacity_filter", |b| {
        b.iter(|| {
            black_box(
                custom_capacity_filter
                    .select_worker(WorkerSelectionInput::configured(
                        black_box(&workers),
                        black_box(&request),
                        request.eligibility(),
                        black_box(16),
                    ))
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn unused_preferred_taint_metadata(c: &mut Criterion) {
    const WORKER_COUNT: usize = 10_000;
    let config = SelectionConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("unused_preferred_taint_metadata/10_000");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    group.throughput(Throughput::Elements(WORKER_COUNT as u64));

    for preferred_taint_count in [0, 1, 4] {
        let (workers, request) = fixture_with_preferred_taints(WORKER_COUNT, preferred_taint_count);
        let policy = WorkerSelectionPolicy::new(
            config.clone(),
            "prefill",
            vec![Box::new(IgnoringPreferenceScorer)],
            Box::new(LowestCostPicker),
        );
        group.bench_with_input(
            BenchmarkId::new("preferred_taints", preferred_taint_count),
            &preferred_taint_count,
            |b, _| {
                b.iter(|| {
                    black_box(
                        policy
                            .select_worker(WorkerSelectionInput::configured(
                                black_box(&workers),
                                black_box(&request),
                                request.eligibility(),
                                black_box(16),
                            ))
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn resident_blocks_read(c: &mut Criterion) {
    let config = SelectionConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("resident_blocks_read");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for worker_count in [2, 32, 1_024, 10_000] {
        let (workers, request) = fixture(worker_count);
        group.throughput(Throughput::Elements(worker_count as u64));
        let without = WorkerSelectionPolicy::new(
            config.clone(),
            "prefill",
            vec![Box::new(IgnoringPreferenceScorer)],
            Box::new(LowestCostPicker),
        );
        let with = WorkerSelectionPolicy::new(
            config.clone(),
            "prefill",
            vec![Box::new(ResidentBlocksScorer)],
            Box::new(LowestCostPicker),
        );
        // Fixed counts for every worker, lent to each selection the way the queue actor lends
        // the indexer's live counters: one load per selection.
        let counts = ResidentBlockCountsHandle::fixed(
            workers
                .keys()
                .map(|&worker_id| (WorkerWithDpRank::from_worker_id(worker_id), 4_096)),
        );
        for (name, policy, lend_counts, writing) in [
            ("without", &without, false, false),
            ("with", &with, true, false),
            // Another thread storing to every counter in a loop: the worst case for the
            // cross-core misses event-thread writes cause readers.
            ("with_writer", &with, true, true),
        ] {
            let stop = AtomicBool::new(false);
            std::thread::scope(|scope| {
                if writing {
                    scope.spawn(|| {
                        let mut blocks = 0;
                        while !stop.load(Ordering::Relaxed) {
                            counts.store_all(blocks);
                            blocks += 1;
                        }
                    });
                }
                group.bench_with_input(
                    BenchmarkId::new(name, worker_count),
                    &worker_count,
                    |b, _| {
                        b.iter(|| {
                            let counts = lend_counts.then(|| counts.load());
                            black_box(
                                policy
                                    .select_worker(
                                        WorkerSelectionInput::configured(
                                            black_box(&workers),
                                            black_box(&request),
                                            request.eligibility(),
                                            black_box(16),
                                        )
                                        .with_resident_blocks(
                                            counts.as_deref().map(|counts| &**counts),
                                        ),
                                    )
                                    .unwrap(),
                            )
                        })
                    },
                );
                stop.store(true, Ordering::Relaxed);
            });
        }
    }
    group.finish();
}

fn kv_event(worker: WorkerId, event_id: u64, data: KvCacheEventData) -> RouterEvent {
    RouterEvent::new(
        worker,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data,
        },
    )
}

/// Event-thread cost of keeping resident block counts, untracked and tracked: 128 workers on 8
/// event threads each store a 16-block chain one block per event, then remove it the same way.
/// Single-block events are the cheapest to apply, so the per-event tracking cost is the largest
/// share of them. Nothing reads the counters here; unpinned, a busy reader thread skews this
/// bench. The pinned Mooncake replay's `--resident-blocks-reader-cpu` covers a reader, at that
/// trace's larger events.
fn resident_blocks_write(c: &mut Criterion) {
    const WORKERS: u64 = 128;
    const BLOCKS: u64 = 16;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let events: Vec<_> = (0..WORKERS)
        .flat_map(|worker| {
            let local: Vec<_> = (0..BLOCKS)
                .map(|i| LocalBlockHash(worker * BLOCKS + i))
                .collect();
            let hashes: Vec<_> = compute_seq_hash_for_block(&local)
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect();
            let stores = (0..hashes.len()).map(|i| {
                let store = KvCacheStoreData {
                    parent_hash: i.checked_sub(1).map(|parent| hashes[parent]),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: hashes[i],
                        tokens_hash: local[i],
                        mm_extra_info: None,
                    }],
                };
                kv_event(worker, i as u64, KvCacheEventData::Stored(store))
            });
            let removes = (0..hashes.len()).rev().map(|i| {
                let remove = KvCacheRemoveData {
                    block_hashes: vec![hashes[i]],
                };
                kv_event(worker, BLOCKS + i as u64, KvCacheEventData::Removed(remove))
            });
            stores.chain(removes).collect::<Vec<_>>()
        })
        .collect();
    let mut group = c.benchmark_group("resident_blocks_write");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(events.len() as u64));
    for (name, tracked) in [("untracked", false), ("tracked", true)] {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 8, 32);
        if tracked {
            indexer.bench_track_resident_blocks();
        }
        group.bench_function(name, |b| {
            b.iter(|| {
                runtime.block_on(async {
                    for event in &events {
                        indexer.apply_event(event.clone()).await;
                    }
                    indexer.flush().await;
                })
            })
        });
        indexer.shutdown();
    }
    group.finish();
}

fn default_policy_wrapper(c: &mut Criterion) {
    let config = SelectionConfig {
        router_temperature: 0.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("default_policy_wrapper");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for worker_count in [2, 32, 1_024] {
        let (workers, request) = fixture(worker_count);
        let direct = DefaultWorkerSelector::new(Some(config.clone()), "prefill");
        let policy = WorkerSelectionPolicy::reference(config.clone(), "prefill");
        group.throughput(Throughput::Elements(worker_count as u64));

        group.bench_with_input(
            BenchmarkId::new("direct", worker_count),
            &worker_count,
            |b, _| {
                b.iter(|| {
                    black_box(
                        direct
                            .select_worker(WorkerSelectionInput::configured(
                                black_box(&workers),
                                black_box(&request),
                                request.eligibility(),
                                black_box(16),
                            ))
                            .unwrap(),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("policy", worker_count),
            &worker_count,
            |b, _| {
                b.iter(|| {
                    black_box(
                        policy
                            .select_worker(WorkerSelectionInput::configured(
                                black_box(&workers),
                                black_box(&request),
                                request.eligibility(),
                                black_box(16),
                            ))
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn worker_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_projection");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    for count in [32, 2_048] {
        let (workers, request) = fixture(count);
        let slots = dynamo_slot_tracker::ActiveSequencesMultiWorker::new(
            dynamo_slot_tracker::NoopSequencePublisher,
            16,
            workers.keys().map(|&id| (id, (0, 1))).collect(),
            false,
            0,
            "bench",
        );
        let now = tokio::time::Instant::now();
        group.bench_function(BenchmarkId::new("allocate", count), |b| {
            b.iter(|| black_box(slots.project_worker_loads(request.token_seq.as_deref(), now)))
        });
        #[cfg(feature = "bench")]
        {
            let mut projections = FxHashMap::default();
            group.bench_function(BenchmarkId::new("reuse", count), |b| {
                b.iter(|| {
                    slots.bench_project_worker_loads_into(
                        request.token_seq.as_deref(),
                        now,
                        &mut projections,
                    );
                    black_box(&projections);
                })
            });
        }
    }
    group.finish();
}

/// Load projection with active prompts: every rank holds requests behind one fleet-wide
/// shared prefix, as a common system prompt leaves it. `overlap` passes the request's blocks,
/// so the pass also finds each rank's overlap with its active prompts, which only `LOAD` rows
/// read; `no_overlap` reads each rank's load alone, as the scheduler does when no policy
/// component declares `LOAD`.
fn worker_projection_active_prompts(c: &mut Criterion) {
    const SHARED_BLOCKS: u64 = 64;
    const SUFFIX_BLOCKS: u64 = 64;
    const REQUESTS_PER_RANK: u64 = 8;

    let mut group = c.benchmark_group("worker_projection_active_prompts");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    for count in [2_048, 8_192] {
        let slots = dynamo_slot_tracker::ActiveSequencesMultiWorker::new(
            dynamo_slot_tracker::NoopSequencePublisher,
            16,
            (0..count).map(|id| (id, (0, 1))).collect(),
            false,
            0,
            "bench",
        );
        let now = tokio::time::Instant::now();
        let shared: Vec<u64> = (0..SHARED_BLOCKS).collect();
        let mut next_hash = SHARED_BLOCKS;
        let mut prompt = || {
            let suffix = next_hash..next_hash + SUFFIX_BLOCKS;
            next_hash += SUFFIX_BLOCKS;
            shared.iter().copied().chain(suffix).collect::<Vec<_>>()
        };
        for worker_id in 0..count {
            for request in 0..REQUESTS_PER_RANK {
                slots
                    .add_request(
                        dynamo_slot_tracker::SequenceRequest {
                            request_id: format!("{worker_id}-{request}"),
                            token_sequence: Some(prompt()),
                            track_prefill_tokens: true,
                            expected_output_tokens: None,
                            prefill_load_hint: None,
                            worker: WorkerWithDpRank::from_worker_id(worker_id),
                            lora_name: None,
                        },
                        now,
                    )
                    .unwrap();
            }
        }
        let query = prompt();
        let mut projections = FxHashMap::default();
        for (name, token_sequence) in [("overlap", Some(query.as_slice())), ("no_overlap", None)] {
            group.bench_function(BenchmarkId::new(name, count), |b| {
                b.iter(|| {
                    slots.project_worker_loads_into(token_sequence, now, &mut projections, None);
                    black_box(&projections);
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    worker_selection,
    custom_worker_selection,
    unused_preferred_taint_metadata,
    resident_blocks_read,
    resident_blocks_write,
    default_policy_wrapper,
    worker_projection,
    worker_projection_active_prompts
);
criterion_main!(benches);
