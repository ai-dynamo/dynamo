// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic corpus conversion: closed-loop capture through the shared merge and rescale into
//! the open-loop corpus, checked for determinism, per-worker ordering, count conservation,
//! and score agreement between the stack indexers and the independent reference.

#[allow(dead_code)]
#[path = "../kv_router/agentic_prep.rs"]
mod agentic_prep;
#[allow(dead_code, unused_imports)]
#[path = "../kv_router/mooncake_open_loop.rs"]
mod mooncake_open_loop;
#[allow(dead_code)]
#[path = "../kv_router/mooncake_shared.rs"]
mod mooncake_shared;
#[allow(dead_code)]
#[path = "../kv_router/scaling_diag.rs"]
mod scaling_diag;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agentic_prep::{
    AgenticEngine, check_worker_order, prepare_agentic_benchmark, prepare_agentic_benchmark_cached,
};
use dynamo_bench::kv_router_common::agentic::{
    AgenticCorpusConfig, AgenticPool, AgenticPrepReport,
};
use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_kv_router::{ConcurrentRadixTreeCompressed, PositionalIndexer, ThreadPoolIndexer};
use dynamo_mocker::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticSourceProvenance,
};
use mooncake_open_loop::{prepare_mooncake_corpus, run_correctness_check};
use mooncake_shared::{PreparedMooncakeBenchmark, WorkerTraceEntry};

const TRACE_BLOCK: usize = 16;
const ENGINE_BLOCK: u32 = 32;
const PLAYS: usize = 12;
const TURNS: usize = 6;
const WORKERS: usize = 4;

/// Multi-turn plays whose turns extend the previous prompt, plus one spawned side request
/// per play, so captured lookups reuse prefixes and the small cache evicts.
fn synthetic_pool() -> AgenticPool {
    let header = AgenticMooncakeHeader {
        schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
        version: AGENTIC_MOONCAKE_VERSION,
        block_size: TRACE_BLOCK,
        hash_id_scope: AgenticHashIdScope::Local,
        source: AgenticSourceProvenance {
            format: "synthetic".to_string(),
            digest: "synthetic-v1".to_string(),
        },
    };
    let mut rows = Vec::new();
    for play in 0..PLAYS {
        let play_id = format!("p{play}:play");
        let request = |turn: usize| format!("p{play}:request:{turn:02}");
        let mut hashes = Vec::new();
        for turn in 0..TURNS {
            let new_blocks = 4 + (play + turn) % 3;
            for _ in 0..new_blocks {
                hashes.push((play * 1_000 + hashes.len()) as u64);
            }
            // A partial tail block: 7 tokens past the last full block.
            let input = (hashes.len() - 1) * TRACE_BLOCK + 7;
            rows.push(AgenticMooncakeRow {
                request_id: request(turn),
                play_id: play_id.clone(),
                source_play_ordinal: Some(play),
                session_id: format!("p{play}:session:root"),
                model: "model".to_string(),
                input_length: Some(input),
                output_length: Some(8),
                hash_ids: Some(hashes.clone()),
                not_before_ms: 5_000.0 * turn as f64,
                dependencies: (turn > 0)
                    .then(|| AgenticDependency {
                        request_id: request(turn - 1),
                        trigger: AgenticDependencyTrigger::Completion,
                        delay_ms: 20.0 + turn as f64,
                        relation: AgenticDependencyRelation::Sequence,
                    })
                    .into_iter()
                    .collect(),
                ..AgenticMooncakeRow::default()
            });
            // The last full block becomes the next turn's partial tail predecessor.
        }
        let side_hashes = hashes[..3]
            .iter()
            .copied()
            .chain([(play * 1_000 + 900) as u64])
            .collect::<Vec<_>>();
        rows.push(AgenticMooncakeRow {
            request_id: format!("p{play}:request:side"),
            play_id: play_id.clone(),
            source_play_ordinal: Some(play),
            session_id: format!("p{play}:session:side"),
            model: "model".to_string(),
            input_length: Some(side_hashes.len() * TRACE_BLOCK),
            output_length: Some(4),
            hash_ids: Some(side_hashes),
            not_before_ms: 0.0,
            dependencies: vec![AgenticDependency {
                request_id: request(1),
                trigger: AgenticDependencyTrigger::Dispatch,
                delay_ms: 5.0,
                relation: AgenticDependencyRelation::Spawn,
            }],
            ..AgenticMooncakeRow::default()
        });
    }
    AgenticPool::from_rows(header, rows).unwrap()
}

fn config() -> AgenticCorpusConfig {
    AgenticCorpusConfig {
        workers: WORKERS,
        plays_per_worker: 6,
        lanes_per_worker: 2,
        sim_ms: 600_000,
        idle_cap_ms: 300_000.0,
        phase_spread: 0.05,
        length_factor: 1,
        seed: 42,
        allow_exhausted_lanes: true,
        collision_stats: true,
        warmup_sim_ms: 0,
    }
}

struct PoolFile(PathBuf);

impl PoolFile {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "agentic-corpus-{tag}-{}.msgpack",
            std::process::id()
        ));
        synthetic_pool().write(&path).unwrap();
        Self(path)
    }
}

impl Drop for PoolFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn prepare(
    pool: &Path,
    config: &AgenticCorpusConfig,
) -> (PreparedMooncakeBenchmark, AgenticPrepReport) {
    prepare_with(
        pool,
        config,
        AgenticEngine {
            num_gpu_blocks: 48,
            block_size: ENGINE_BLOCK,
            speedup_ratio: 1.0,
            sglang: false,
        },
    )
    .await
}

async fn prepare_with(
    pool: &Path,
    config: &AgenticCorpusConfig,
    engine: AgenticEngine,
) -> (PreparedMooncakeBenchmark, AgenticPrepReport) {
    let (prepared, _, report) = prepare_agentic_benchmark(pool, None, config, engine, 10_000)
        .await
        .unwrap();
    (prepared, report)
}

/// SGLang radix-cache engine at page size 1 with a small token capacity, so it evicts.
const SGLANG_PAGE1: AgenticEngine = AgenticEngine {
    num_gpu_blocks: 48 * ENGINE_BLOCK as usize,
    block_size: 1,
    speedup_ratio: 1.0,
    sglang: true,
};

#[tokio::test(flavor = "multi_thread")]
async fn sglang_warmup_prefix_conserves_events_and_keeps_scores_exact() {
    let pool = PoolFile::new("warmup");
    let (full, full_report) = prepare_with(&pool.0, &config(), SGLANG_PAGE1).await;
    assert_eq!(full_report.engine_type, "sglang");
    assert!(full.warmup_events.is_empty());
    assert!(full.totals.removed_blocks > 0, "the small cache must evict");

    let warm_config = AgenticCorpusConfig {
        warmup_sim_ms: 10_000,
        ..config()
    };
    let (warm, warm_report) = prepare_with(&pool.0, &warm_config, SGLANG_PAGE1).await;
    assert!(warm_report.warmup_events > 0);
    assert!(warm_report.warmup_requests_dropped > 0);
    assert!(warm.totals.requests > 0);

    // The cut moves events into the prefix and drops earlier lookups; nothing else changes.
    assert_eq!(warm.warmup_events.len() as u64, warm_report.warmup_events);
    assert_eq!(
        warm.totals.requests as u64 + warm_report.warmup_requests_dropped,
        full.totals.requests as u64
    );
    assert_eq!(
        warm.totals.stored_blocks as u64 + warm_report.warmup_stored_blocks,
        full.totals.stored_blocks as u64
    );
    assert_eq!(
        warm.totals.removed_blocks as u64 + warm_report.warmup_removed_blocks,
        full.totals.removed_blocks as u64
    );
    // Per worker, the prefix keeps source order (event IDs ascend).
    let mut last_id = [None; WORKERS];
    for warmup in &warm.warmup_events {
        let id = warmup.event.event_id;
        assert!(last_id[warmup.worker].is_none_or(|last| last < id));
        last_id[warmup.worker] = Some(id);
    }

    let crtc = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        2,
        1,
    ));
    let corpus = prepare_mooncake_corpus(warm, 1).unwrap();
    let report = run_correctness_check("crtc", crtc, corpus, 100_000, true)
        .await
        .unwrap();
    assert!(report.pass, "{report:?}");
    assert_eq!(report.checked_queries, report.total_queries);
    assert!(report.checked_matched_ranks.mean > 0.5);
}

#[tokio::test(flavor = "multi_thread")]
async fn agentic_prep_is_deterministic_ordered_and_conserves_counts() {
    let pool = PoolFile::new("determinism");
    let (first, report) = prepare(&pool.0, &config()).await;
    let (second, second_report) = prepare(&pool.0, &config()).await;
    assert_eq!(
        report.merged_corpus_digest,
        second_report.merged_corpus_digest
    );
    assert_eq!(report.pool_sha256, second_report.pool_sha256);
    assert_eq!(first.worker_traces.len(), WORKERS);
    assert_eq!(
        second.totals.total_block_ops(),
        first.totals.total_block_ops()
    );
    check_worker_order(&first).unwrap();

    // K = ceil(S * W / plays) copies; every instance is dealt exactly once.
    assert_eq!(report.copies, 2);
    assert_eq!(report.plays_assigned_per_worker.total, (2 * PLAYS) as u64);
    assert_eq!(report.shared_local_hashes_after_salt, Some(0));
    assert!(report.shared_local_hashes_before_salt.unwrap() > 0);

    // The merge and rescale conserve the captured queries and events per worker.
    let totals = first.totals;
    assert_eq!(
        totals.requests as u64,
        report.requests_captured_per_worker.total
    );
    assert_eq!(
        totals.request_blocks as u64,
        report.request_blocks_per_worker.total
    );
    assert_eq!(
        totals.stored_blocks as u64,
        report.stored_blocks_per_worker.total
    );
    assert_eq!(
        totals.removed_blocks as u64,
        report.removed_blocks_per_worker.total
    );
    assert!(totals.removed_blocks > 0, "the small cache must evict");
    let per_worker_requests = first
        .worker_traces
        .iter()
        .map(|trace| {
            trace
                .iter()
                .filter(|entry| matches!(entry.entry, WorkerTraceEntry::Request(_)))
                .count() as u64
        })
        .collect::<Vec<_>>();
    assert_eq!(
        per_worker_requests.iter().min().copied(),
        Some(report.requests_captured_per_worker.min)
    );
    assert_eq!(
        per_worker_requests.iter().max().copied(),
        Some(report.requests_captured_per_worker.max)
    );

    // Global rescale: the corpus spans the window, and workers keep distinct phases.
    let first_stamps = first
        .worker_traces
        .iter()
        .map(|trace| trace[0].timestamp_us)
        .collect::<Vec<_>>();
    assert_eq!(first_stamps.iter().min(), Some(&0));
    assert!(first_stamps.iter().any(|&stamp| stamp > 0));
    let last = first
        .worker_traces
        .iter()
        .filter_map(|trace| trace.last().map(|entry| entry.timestamp_us))
        .max();
    assert_eq!(last, Some(10_000_000));

    // Removals only name blocks the worker stored earlier.
    for trace in first.worker_traces.iter() {
        let mut stored = std::collections::HashSet::new();
        for entry in trace {
            let WorkerTraceEntry::Event { event, .. } = &entry.entry else {
                continue;
            };
            match &event.data {
                KvCacheEventData::Stored(store) => {
                    stored.extend(store.blocks.iter().map(|block| block.block_hash));
                }
                KvCacheEventData::Removed(remove) => {
                    assert!(remove.block_hashes.iter().all(|hash| stored.contains(hash)));
                }
                KvCacheEventData::Cleared => {}
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn agentic_scores_match_the_reference_for_crtc_and_nested_map() {
    let pool = PoolFile::new("scores");
    let (prepared, _) = prepare(&pool.0, &config()).await;

    let crtc = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        2,
        ENGINE_BLOCK,
    ));
    let corpus = prepare_mooncake_corpus(prepared.clone(), 1).unwrap();
    let report = run_correctness_check("crtc", crtc, corpus, 100_000, true)
        .await
        .unwrap();
    assert!(report.pass, "{report:?}");
    assert_eq!(report.checked_queries, report.total_queries);
    // Plays never share content, so a lookup matches at most its own worker.
    assert!(report.checked_matched_ranks.max <= 1);
    assert!(report.checked_matched_ranks.mean > 0.5);

    let nested = Arc::new(ThreadPoolIndexer::new(
        PositionalIndexer::new(8),
        2,
        ENGINE_BLOCK,
    ));
    let corpus = prepare_mooncake_corpus(prepared, 1).unwrap();
    let report = run_correctness_check("nested-map", nested, corpus, 100_000, true)
        .await
        .unwrap();
    assert!(report.pass, "{report:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn corpus_cache_round_trips_and_rejects_a_different_key() {
    let pool = PoolFile::new("cache");
    let cache = std::env::temp_dir().join(format!("agentic-cache-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&cache);
    let config = AgenticCorpusConfig {
        warmup_sim_ms: 10_000,
        ..config()
    };
    let key = serde_json::json!({"case": "round-trip"});
    let run = |window_ms: u64, key: serde_json::Value| {
        prepare_agentic_benchmark_cached(
            &pool.0,
            None,
            &config,
            SGLANG_PAGE1,
            window_ms,
            Some(&cache),
            key,
        )
    };
    let (written, _, written_report) = run(10_000, key.clone()).await.unwrap();
    let (loaded, _, loaded_report) = run(10_000, key.clone()).await.unwrap();
    assert_eq!(written_report["corpus_cache"]["loaded"], false);
    assert_eq!(loaded_report["corpus_cache"]["loaded"], true);
    assert_eq!(
        written_report["corpus_cache"]["digest"],
        loaded_report["corpus_cache"]["digest"]
    );
    assert_eq!(
        written_report["merged_corpus_digest"],
        loaded_report["merged_corpus_digest"]
    );
    assert_eq!(written.warmup_events.len(), loaded.warmup_events.len());
    assert!(!loaded.warmup_events.is_empty());
    assert_eq!(
        written.totals.total_block_ops(),
        loaded.totals.total_block_ops()
    );
    // A different window rescales the cached timeline instead of re-capturing.
    let (_, _, rescaled) = run(5_000, key).await.unwrap();
    assert_eq!(rescaled["corpus_cache"]["loaded"], true);
    assert_ne!(
        rescaled["merged_corpus_digest"],
        loaded_report["merged_corpus_digest"]
    );
    assert!(
        run(10_000, serde_json::json!({"case": "other"}))
            .await
            .is_err()
    );
    let _ = std::fs::remove_file(&cache);
}
