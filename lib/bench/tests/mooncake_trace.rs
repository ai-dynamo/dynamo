// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

#[path = "../kv_router/common/shared.rs"]
mod common;

#[path = "../kv_router/mooncake_shared.rs"]
mod mooncake_shared;

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{WorkerReplayArtifacts, generate_replay_artifacts, process_mooncake_trace};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::KvIndexerMetrics;
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, WorkerWithDpRank};
use dynamo_mocker::loadgen::{SessionTrace, Trace, TurnTrace};
use mooncake_shared::{MooncakeBenchmarkConfig, MooncakeIndexerConfig, run_benchmark};
use tempfile::NamedTempFile;

const BLOCK_SIZE: u32 = 128;
const NUM_GPU_BLOCKS: usize = 16384;
const NUM_UNIQUE_INFERENCE_WORKERS: usize = 10;
const BENCHMARK_DURATION_MS: u64 = 2000;
const NUM_EVENT_WORKERS: usize = 4;

type NormalizedOverlapScores = BTreeMap<WorkerWithDpRank, u32>;

#[derive(Clone)]
enum ReplayEntryKind {
    Request(Vec<LocalBlockHash>),
    Event(KvCacheEvent),
}

#[derive(Clone)]
struct ReplayEntry {
    timestamp_us: u64,
    worker_id: u64,
    kind_rank: u8,
    kind: ReplayEntryKind,
}

fn collect_replay_entries(artifacts: &[WorkerReplayArtifacts]) -> Vec<ReplayEntry> {
    let mut entries = Vec::new();
    for (worker_id, artifact) in artifacts.iter().enumerate() {
        entries.extend(artifact.requests.iter().map(|request| ReplayEntry {
            timestamp_us: request.timestamp_us,
            worker_id: worker_id as u64,
            kind_rank: 0,
            kind: ReplayEntryKind::Request(request.replay_hashes.local_block_hashes.clone()),
        }));
        entries.extend(artifact.kv_events.iter().map(|event| ReplayEntry {
            timestamp_us: event.timestamp_us,
            worker_id: worker_id as u64,
            kind_rank: 1,
            kind: ReplayEntryKind::Event(event.event.clone()),
        }));
    }
    entries.sort_by_key(|entry| (entry.timestamp_us, entry.kind_rank, entry.worker_id));
    entries
}

async fn collect_overlap_scores_for_replay(
    config: &MooncakeIndexerConfig,
    artifacts: &[WorkerReplayArtifacts],
) -> anyhow::Result<Vec<NormalizedOverlapScores>> {
    let indexer = config.build(BLOCK_SIZE, Arc::new(KvIndexerMetrics::new_unregistered()));
    let entries = collect_replay_entries(artifacts);
    let mut scores = Vec::new();
    let mut idx = 0;

    while idx < entries.len() {
        let timestamp_us = entries[idx].timestamp_us;
        while idx < entries.len() && entries[idx].timestamp_us == timestamp_us {
            match &entries[idx].kind {
                ReplayEntryKind::Request(request) => {
                    let overlap = indexer.find_matches(request.clone()).await?;
                    scores.push(overlap.scores.into_iter().collect());
                }
                ReplayEntryKind::Event(event) => {
                    indexer
                        .apply_event(RouterEvent::new(entries[idx].worker_id, event.clone()))
                        .await;
                }
            }
            idx += 1;
        }
        indexer.flush().await;
    }

    indexer.shutdown();
    Ok(scores)
}

async fn assert_overlap_score_parity(
    variants: &[MooncakeIndexerConfig],
    artifacts: &[WorkerReplayArtifacts],
) -> anyhow::Result<()> {
    let mut expected_name = None;
    let mut expected_scores = Vec::new();

    for config in variants {
        let actual_scores = collect_overlap_scores_for_replay(config, artifacts).await?;
        if expected_name.is_none() {
            expected_name = Some(config.short_name().to_string());
            expected_scores = actual_scores;
            continue;
        }

        assert_eq!(
            actual_scores.len(),
            expected_scores.len(),
            "{} produced a different number of request overlap results than {}",
            config.short_name(),
            expected_name.as_deref().unwrap()
        );

        for (request_idx, (actual, expected)) in
            actual_scores.iter().zip(expected_scores.iter()).enumerate()
        {
            assert_eq!(
                actual,
                expected,
                "{} overlap scores diverged from {} at replay request {request_idx}",
                config.short_name(),
                expected_name.as_deref().unwrap()
            );
        }
    }

    Ok(())
}

#[test]
fn process_mooncake_trace_expands_and_duplicates_hash_space() -> anyhow::Result<()> {
    let mut file = NamedTempFile::new()?;
    for (i, (hash_ids, output_length)) in [(&[0u64, 1, 2] as &[u64], 10u64), (&[0, 1, 3, 4], 10)]
        .iter()
        .enumerate()
    {
        writeln!(
            file,
            "{}",
            serde_json::json!({
                "timestamp": i as u64,
                "input_length": hash_ids.len(),
                "hash_ids": hash_ids,
                "output_length": output_length,
            })
        )?;
    }

    let traces = process_mooncake_trace(
        file.path().to_str().expect("temp path should be UTF-8"),
        512,
        2,
        2,
        2,
        42,
    )?;

    let mut all_hashes: Vec<Vec<u64>> = traces
        .into_iter()
        .flat_map(|worker| worker.sessions.into_iter())
        .flat_map(|session| session.turns.into_iter().map(|turn| turn.hash_ids))
        .collect();
    all_hashes.sort();

    let mut expected = vec![
        vec![0, 1, 2, 3, 4, 5],
        vec![10, 11, 12, 13, 14, 15],
        vec![0, 1, 2, 3, 6, 7, 8, 9],
        vec![10, 11, 12, 13, 16, 17, 18, 19],
    ];
    expected.sort();
    assert_eq!(all_hashes, expected, "hash_ids mismatch");

    let copy0: Vec<&Vec<u64>> = all_hashes.iter().filter(|hashes| hashes[0] == 0).collect();
    let copy1: Vec<&Vec<u64>> = all_hashes.iter().filter(|hashes| hashes[0] == 10).collect();
    assert_eq!(copy0.len(), 2);
    assert_eq!(copy1.len(), 2);
    assert_eq!(copy0[0][..4], copy0[1][..4], "copy 0 shared prefix broken");
    assert_eq!(copy1[0][..4], copy1[1][..4], "copy 1 shared prefix broken");

    let set0: HashSet<u64> = copy0
        .iter()
        .flat_map(|hashes| hashes.iter().copied())
        .collect();
    let set1: HashSet<u64> = copy1
        .iter()
        .flat_map(|hashes| hashes.iter().copied())
        .collect();
    assert!(set0.is_disjoint(&set1), "copies are not hash-disjoint");

    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn generate_replay_artifacts_waits_for_completion_delay() -> anyhow::Result<()> {
    let trace = Trace {
        block_size: 2,
        sessions: vec![SessionTrace {
            session_id: "session-a".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 2,
                    hash_ids: vec![1, 2],
                    delay_after_previous_ms: 0.0,
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 2,
                    hash_ids: vec![3, 4],
                    delay_after_previous_ms: 5.0,
                },
            ],
        }],
    };

    let artifacts = generate_replay_artifacts(&[trace], 1024, 2, None).await?;
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0].requests.len(), 2);

    let first_uuid = artifacts[0].requests[0].uuid;
    let first_completion_ms = artifacts[0]
        .output_signals
        .iter()
        .find(|signal| signal.signal.uuid == first_uuid && signal.signal.completed)
        .expect("first request must complete")
        .timestamp_us as f64
        / 1000.0;

    assert!(
        artifacts[0].requests[1].scheduled_ready_at_ms + 0.1 >= first_completion_ms + 5.0,
        "expected second request to wait for completion plus delay"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_trace_replays_without_warnings_across_indexer_variants() -> anyhow::Result<()> {
    let warning_count = support::warning_counter(&["dynamo_kv_router::indexer", "dynamo_mocker"]);

    let fixture = support::fixture_path("mooncake_trace_1000.jsonl")?;
    let traces =
        process_mooncake_trace(&fixture, BLOCK_SIZE, 1, 1, NUM_UNIQUE_INFERENCE_WORKERS, 42)?;
    let artifacts = generate_replay_artifacts(&traces, NUM_GPU_BLOCKS, BLOCK_SIZE, None).await?;

    let variants = [
        MooncakeIndexerConfig::radix_tree(),
        MooncakeIndexerConfig::nested_map(8, NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree(NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree_compressed(NUM_EVENT_WORKERS),
    ];

    for config in &variants {
        support::reset_warning_count(&warning_count);

        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let run = {
            let indexer = config.build(BLOCK_SIZE, Arc::clone(&metrics));
            run_benchmark(
                indexer,
                artifacts.clone(),
                MooncakeBenchmarkConfig {
                    benchmark_duration_ms: BENCHMARK_DURATION_MS,
                    inference_worker_duplication_factor: 1,
                    count_events: config.supports_remove(),
                    find_matches_concurrency: 0,
                },
            )
            .await?
        };

        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(
            run.kept_up,
            "{} replay fell behind in test profile; increase BENCHMARK_DURATION_MS if this becomes too tight",
            config.short_name()
        );
        assert!(
            run.results.ops_throughput > 0.0,
            "{} replay should record positive throughput",
            config.short_name()
        );
        assert_eq!(
            warning_count.load(Ordering::Relaxed),
            0,
            "{} emitted warn/error logs from dynamo_kv_router::indexer or dynamo_mocker",
            config.short_name()
        );
        assert_eq!(
            support::duplicate_store_warning_count(metrics.as_ref()),
            0,
            "{} recorded duplicate-store warning metrics",
            config.short_name()
        );
    }

    assert_overlap_score_parity(&variants, &artifacts).await?;

    Ok(())
}
