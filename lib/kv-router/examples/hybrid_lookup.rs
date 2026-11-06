// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reproducible public Indexer read-path microbenchmark; not a serving-QPS benchmark.
use std::hint::black_box;
use std::sync::{Arc, Barrier, OnceLock};
use std::time::{Duration, Instant};

use dynamo_kv_router::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::indexer::{LowerTierIndexers, LowerTierQueryOptions, ThreadPoolIndexer};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
    WorkerWithDpRank,
};
use dynamo_kv_router::services::indexer::backend::Indexer;

const WORKERS: usize = 64;
const BLOCKS: usize = 1024;
const CHECKPOINT: usize = 512;
const KV_BLOCK_SIZE: u32 = 576;

fn store(worker: u64, event_id: u64, positions: &[usize], group: Option<u32>) -> RouterEvent {
    let event = RouterEvent::with_storage_tier(
        worker,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: positions
                    .iter()
                    .map(|position| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(10_000 + *position as u64),
                        tokens_hash: LocalBlockHash(*position as u64),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        },
        StorageTier::Device,
    );
    match group {
        None => event,
        Some(group_idx) => event.with_kv_cache_group(KvCacheGroupTag {
            group_idx,
            kind: KvCacheGroupKind::Mamba,
            expected_group_count: Some(3),
            alignment_blocks: Some(1),
        }),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: hybrid_lookup CASE READERS DURATION_MS"
    );
    let case = args[1].as_str();
    assert!(["sparse", "incomplete", "dense", "populated_short"].contains(&case));
    let readers: usize = args[2].parse().unwrap();
    let duration = Duration::from_millis(args[3].parse().unwrap());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let primary = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        2,
        KV_BLOCK_SIZE,
    ));
    let indexer = Arc::new(Indexer::Concurrent {
        primary: Arc::clone(&primary),
        lower_tier: LowerTierIndexers::new(2, KV_BLOCK_SIZE),
        approx: None,
        primary_records_routing_decisions: false,
        session_updates: None,
    });
    let query_blocks = if case == "populated_short" { 8 } else { BLOCKS };
    let sequence: Arc<Vec<_>> = Arc::new(
        (1..=query_blocks)
            .map(|i| LocalBlockHash(i as u64))
            .collect(),
    );
    let ingest_started = Instant::now();
    runtime.block_on(async {
        let positions: Vec<_> = (1..=BLOCKS).collect();
        for worker in 0..WORKERS {
            indexer
                .apply_event_routed(store(worker as u64, 1, &positions, None))
                .await
                .unwrap();
            if case != "dense" {
                indexer.declare_recurrent_layout(
                    WorkerWithDpRank::new(worker as u64, 0),
                    &[0, 1, 2],
                    1,
                );
                for group in 0..if case == "incomplete" { 2 } else { 3 } {
                    let checkpoints = if case == "populated_short" {
                        positions.as_slice()
                    } else {
                        &[CHECKPOINT]
                    };
                    indexer
                        .apply_event_routed(store(
                            worker as u64,
                            2 + group as u64,
                            checkpoints,
                            Some(group),
                        ))
                        .await
                        .unwrap();
                }
            }
        }
        primary.flush_and_wait().await.unwrap();
    });
    let fixture_ingest_seconds = ingest_started.elapsed().as_secs_f64();
    let expected_depth = match case {
        "sparse" => CHECKPOINT,
        "incomplete" => 0,
        "populated_short" => 8,
        _ => BLOCKS,
    };
    let expected_workers = if expected_depth == 0 { 0 } else { WORKERS };
    let expected_sum = expected_depth as u64 * expected_workers as u64;
    let options = LowerTierQueryOptions {
        retain_kv_transfer_chain: false,
        max_reusable_blocks: if case == "populated_short" {
            None
        } else {
            Some(BLOCKS)
        },
    };
    runtime.block_on(async {
        let result = indexer
            .find_tiered_matches_ref_with_options(&sequence, options)
            .await
            .unwrap();
        assert_eq!(result.device.overlap_scores.scores.len(), expected_workers);
        assert!(
            result
                .device
                .overlap_scores
                .scores
                .values()
                .all(|v| *v as usize == expected_depth)
        );
        assert!(
            result.lower_tier.is_empty(),
            "fixture must contain device residency only"
        );
    });
    let barrier = Arc::new(Barrier::new(readers + 1));
    let ready = Arc::new(Barrier::new(readers + 1));
    let started = Arc::new(OnceLock::<Instant>::new());
    let mut handles = Vec::new();
    for _ in 0..readers {
        let (indexer, sequence, barrier, ready, started) = (
            Arc::clone(&indexer),
            Arc::clone(&sequence),
            Arc::clone(&barrier),
            Arc::clone(&ready),
            Arc::clone(&started),
        );
        let handle = runtime.handle().clone();
        handles.push(std::thread::spawn(move || {
            handle.block_on(async move {
                for _ in 0..32 {
                    black_box(
                        indexer
                            .find_tiered_matches_ref_with_options(&sequence, options)
                            .await
                            .unwrap(),
                    );
                }
                ready.wait();
                barrier.wait();
                let start = *started.get().unwrap();
                let mut count = 0u64;
                let mut checksum = 0u64;
                let mut sample_ns = Vec::new();
                loop {
                    if count.is_multiple_of(4) && start.elapsed() >= duration {
                        break;
                    }
                    let sample_start = (count.is_multiple_of(16)).then(Instant::now);
                    let result = indexer
                        .find_tiered_matches_ref_with_options(black_box(&sequence), options)
                        .await
                        .unwrap();
                    let scores = &result.device.overlap_scores.scores;
                    assert_eq!(scores.len(), expected_workers);
                    let sum: u64 = scores.values().map(|v| *v as u64).sum();
                    assert_eq!(sum, expected_sum);
                    checksum =
                        checksum.wrapping_add(black_box(sum).wrapping_add(scores.len() as u64));
                    drop(result);
                    if let Some(sample_start) = sample_start {
                        sample_ns.push(sample_start.elapsed().as_nanos() as u64);
                    }
                    count += 1;
                }
                (count, checksum, sample_ns, start.elapsed().as_secs_f64())
            })
        }));
    }
    ready.wait();
    started.set(Instant::now()).unwrap();
    barrier.wait();
    let mut total = 0u64;
    let mut checksum = 0u64;
    let mut samples = Vec::new();
    let mut wall_seconds = 0.0f64;
    for handle in handles {
        let (count, check, mut sample, elapsed) = handle.join().unwrap();
        total += count;
        checksum = checksum.wrapping_add(check);
        samples.append(&mut sample);
        wall_seconds = wall_seconds.max(elapsed);
    }
    samples.sort_unstable();
    assert!(!samples.is_empty());
    let percentile =
        |p: f64| samples[((samples.len() - 1) as f64 * p).round() as usize] as f64 / 1000.0;
    println!(
        "{}",
        serde_json::json!({
            "case": case, "readers": readers, "workers": WORKERS, "prefix_blocks": BLOCKS,
            "query_blocks": query_blocks, "fixture_ingest_seconds": fixture_ingest_seconds, "groups": if case == "dense" {0} else {3}, "expected_device_blocks": expected_depth,
            "lookups": total, "wall_seconds": wall_seconds, "lookups_per_second": total as f64 / wall_seconds,
            "sampled_latency_p50_us": percentile(0.50), "sampled_latency_p95_us": percentile(0.95),
            "latency_samples": samples.len(), "checksum": checksum,
        })
    );
}
