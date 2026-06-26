// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

#[cfg(feature = "mocker-kvbm-offload")]
#[path = "support/mooncake_g2_lower_tier.rs"]
mod g2_lower_tier;

#[path = "../kv_router/mooncake_shared.rs"]
mod mooncake_shared;

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

#[cfg(feature = "mocker-kvbm-offload")]
use dynamo_bench::kv_router_common::replay::generate_g2_replay_artifacts_with_capacity;
use dynamo_bench::kv_router_common::replay::{
    WorkerReplayArtifacts, generate_replay_artifacts,
    generate_replay_artifacts_with_args_and_visibility, process_mooncake_trace,
};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::pruning::PruneConfig;
use dynamo_kv_router::indexer::{KvIndexerInterface, KvIndexerMetrics};
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier, TokensWithHashes, WorkerWithDpRank,
};
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, SglangArgs};
use dynamo_mocker::loadgen::{SessionTrace, Trace, TurnTrace};
use dynamo_mocker::replay::ReplayKvEventVisibility;
use mooncake_shared::{
    MooncakeBenchmarkConfig, MooncakeBenchmarkInput, MooncakeIndexerConfig, run_benchmark,
};
use tempfile::NamedTempFile;

const BLOCK_SIZE: u32 = 128;
const NUM_GPU_BLOCKS: usize = 16384;
const NUM_UNIQUE_INFERENCE_WORKERS: usize = 10;
const BENCHMARK_DURATION_MS: u64 = 2000;
const NUM_EVENT_WORKERS: usize = 4;
const PARITY_NUM_GPU_BLOCKS: usize = NUM_GPU_BLOCKS;
const SGLANG_PARITY_PREFILL_TOKENS: usize = PARITY_NUM_GPU_BLOCKS * BLOCK_SIZE as usize;
#[cfg(feature = "mocker-kvbm-offload")]
const G2_TEST_NUM_GPU_BLOCKS: usize = 512;
#[cfg(feature = "mocker-kvbm-offload")]
const G2_TEST_NUM_G2_BLOCKS: usize = 16_384;

type NormalizedOverlapScores = BTreeMap<WorkerWithDpRank, u32>;

#[derive(Clone)]
struct MockEngineReplayArtifacts {
    engine_name: &'static str,
    artifacts: Vec<WorkerReplayArtifacts>,
}

#[derive(Clone, Copy, Debug)]
enum MockEngineParityKind {
    Vllm,
    Sglang,
    Trtllm,
}

impl MockEngineParityKind {
    fn name(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
            Self::Trtllm => "trtllm",
        }
    }

    fn engine_type(self) -> EngineType {
        match self {
            Self::Vllm => EngineType::Vllm,
            Self::Sglang => EngineType::Sglang,
            Self::Trtllm => EngineType::Trtllm,
        }
    }

    fn kv_event_visibility_override(self) -> Option<ReplayKvEventVisibility> {
        match self {
            Self::Sglang => Some(ReplayKvEventVisibility::PassStart),
            Self::Vllm | Self::Trtllm => None,
        }
    }

    fn mock_engine_args(self) -> anyhow::Result<MockEngineArgs> {
        let mut builder = MockEngineArgs::builder()
            .engine_type(self.engine_type())
            .num_gpu_blocks(PARITY_NUM_GPU_BLOCKS)
            .block_size(BLOCK_SIZE as usize)
            .speedup_ratio(10.0)
            .enable_prefix_caching(true)
            .max_num_batched_tokens(None)
            .max_num_seqs(None);

        if matches!(self, Self::Sglang) {
            builder = builder.sglang(Some(SglangArgs {
                page_size: Some(BLOCK_SIZE as usize),
                max_prefill_tokens: Some(SGLANG_PARITY_PREFILL_TOKENS),
                chunked_prefill_size: Some(SGLANG_PARITY_PREFILL_TOKENS),
                ..Default::default()
            }));
        }

        builder.build()?.normalized()
    }
}

#[derive(Clone)]
enum ReplayEntryKind {
    Request(Vec<LocalBlockHash>),
    Event {
        event: KvCacheEvent,
        storage_tier: StorageTier,
    },
    ApproxWrite(Vec<u32>),
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
            kind: ReplayEntryKind::Event {
                event: event.event.clone(),
                storage_tier: event.storage_tier,
            },
        }));
    }
    entries.sort_by_key(|entry| (entry.timestamp_us, entry.kind_rank, entry.worker_id));
    entries
}

fn collect_approx_replay_entries(traces: &[Trace]) -> anyhow::Result<Vec<ReplayEntry>> {
    let mut entries = Vec::new();
    for (worker_id, trace) in traces.iter().enumerate() {
        let trace_block_size = trace.block_size;
        for session in &trace.sessions {
            let mut timestamp_ms = session.first_arrival_timestamp_ms.unwrap_or(0.0);
            for (turn_idx, turn) in session.turns.iter().enumerate() {
                if turn_idx > 0 {
                    timestamp_ms += turn.delay_after_previous_ms;
                }
                let replay_hashes = turn.to_replay_hashes(trace_block_size, BLOCK_SIZE as usize)?;
                let timestamp_us = (timestamp_ms.max(0.0) * 1000.0) as u64;
                entries.push(ReplayEntry {
                    timestamp_us,
                    worker_id: worker_id as u64,
                    kind_rank: 0,
                    kind: ReplayEntryKind::Request(replay_hashes.local_block_hashes),
                });
                entries.push(ReplayEntry {
                    timestamp_us,
                    worker_id: worker_id as u64,
                    kind_rank: 1,
                    kind: ReplayEntryKind::ApproxWrite(turn.synthesize_tokens(trace_block_size)?),
                });
            }
        }
    }
    entries.sort_by_key(|entry| (entry.timestamp_us, entry.kind_rank, entry.worker_id));
    Ok(entries)
}

fn count_removed_kv_events(artifacts: &[WorkerReplayArtifacts]) -> usize {
    artifacts
        .iter()
        .flat_map(|artifact| artifact.kv_events.iter())
        .filter(|event| matches!(&event.event.data, KvCacheEventData::Removed(_)))
        .count()
}

fn assert_no_removed_kv_events(engine_name: &str, artifacts: &[WorkerReplayArtifacts]) {
    assert_eq!(
        count_removed_kv_events(artifacts),
        0,
        "{engine_name} parity artifacts should not contain Removed KV events; increase PARITY_NUM_GPU_BLOCKS if the fixture starts evicting cached blocks"
    );
}

async fn generate_mock_engine_parity_artifacts(
    traces: &[Trace],
) -> anyhow::Result<Vec<MockEngineReplayArtifacts>> {
    let mut artifact_sets = Vec::new();

    for engine in [
        MockEngineParityKind::Vllm,
        MockEngineParityKind::Sglang,
        MockEngineParityKind::Trtllm,
    ] {
        let artifacts = generate_replay_artifacts_with_args_and_visibility(
            traces,
            engine.mock_engine_args()?,
            None,
            engine.kv_event_visibility_override(),
        )
        .await?;
        assert_no_removed_kv_events(engine.name(), &artifacts);
        artifact_sets.push(MockEngineReplayArtifacts {
            engine_name: engine.name(),
            artifacts,
        });
    }

    Ok(artifact_sets)
}

#[derive(Clone, Copy, Debug)]
enum MooncakeReplayMode {
    KvEvents,
    Approx,
}

impl MooncakeReplayMode {
    fn name(self) -> &'static str {
        match self {
            MooncakeReplayMode::KvEvents => "kv-events",
            MooncakeReplayMode::Approx => "approx",
        }
    }
}

async fn collect_overlap_scores_for_replay(
    config: &MooncakeIndexerConfig,
    traces: &[Trace],
    artifacts: &[WorkerReplayArtifacts],
    mode: MooncakeReplayMode,
) -> anyhow::Result<Vec<NormalizedOverlapScores>> {
    let indexer = match mode {
        MooncakeReplayMode::KvEvents => {
            config.build(BLOCK_SIZE, Arc::new(KvIndexerMetrics::new_unregistered()))
        }
        MooncakeReplayMode::Approx => {
            config.build_approximate(BLOCK_SIZE, Arc::new(KvIndexerMetrics::new_unregistered()))?
        }
    };
    let entries = match mode {
        MooncakeReplayMode::KvEvents => collect_replay_entries(artifacts),
        MooncakeReplayMode::Approx => collect_approx_replay_entries(traces)?,
    };
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
                ReplayEntryKind::Event {
                    event,
                    storage_tier,
                } => {
                    if storage_tier.is_gpu() {
                        indexer
                            .apply_event(RouterEvent::with_storage_tier(
                                entries[idx].worker_id,
                                event.clone(),
                                *storage_tier,
                            ))
                            .await;
                    }
                }
                ReplayEntryKind::ApproxWrite(tokens) => {
                    let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), BLOCK_SIZE);
                    indexer
                        .process_routing_decision_for_request(
                            &mut tokens_with_hashes,
                            WorkerWithDpRank::from_worker_id(entries[idx].worker_id),
                        )
                        .await?;
                }
            }
            idx += 1;
        }
        indexer.flush().await;
    }

    indexer.shutdown();
    Ok(scores)
}

async fn collect_overlap_scores_for_supported_modes(
    config: &MooncakeIndexerConfig,
    traces: &[Trace],
    artifact_sets: &[MockEngineReplayArtifacts],
) -> anyhow::Result<Vec<(String, Vec<NormalizedOverlapScores>)>> {
    let mut results = Vec::new();
    for artifact_set in artifact_sets {
        results.push((
            format!(
                "{} {} ({})",
                config.short_name(),
                artifact_set.engine_name,
                MooncakeReplayMode::KvEvents.name()
            ),
            collect_overlap_scores_for_replay(
                config,
                traces,
                &artifact_set.artifacts,
                MooncakeReplayMode::KvEvents,
            )
            .await?,
        ));
    }
    if config.supports_approximate() {
        results.push((
            format!(
                "{} ({})",
                config.short_name(),
                MooncakeReplayMode::Approx.name()
            ),
            collect_overlap_scores_for_replay(config, traces, &[], MooncakeReplayMode::Approx)
                .await?,
        ));
    }
    Ok(results)
}

async fn assert_overlap_score_parity(
    variants: &[MooncakeIndexerConfig],
    traces: &[Trace],
    artifact_sets: &[MockEngineReplayArtifacts],
) -> anyhow::Result<()> {
    let mut expected_name = None;
    let mut expected_scores = Vec::new();

    for config in variants {
        for (actual_name, actual_scores) in
            collect_overlap_scores_for_supported_modes(config, traces, artifact_sets).await?
        {
            if expected_name.is_none() {
                expected_name = Some(actual_name);
                expected_scores = actual_scores;
                continue;
            }

            assert_eq!(
                actual_scores.len(),
                expected_scores.len(),
                "{} produced a different number of request overlap results than {}",
                actual_name,
                expected_name.as_deref().unwrap()
            );

            for (request_idx, (actual, expected)) in
                actual_scores.iter().zip(expected_scores.iter()).enumerate()
            {
                assert_eq!(
                    actual,
                    expected,
                    "{} overlap scores diverged from {} at replay request {request_idx}",
                    actual_name,
                    expected_name.as_deref().unwrap()
                );
            }
        }
    }

    Ok(())
}

fn approx_tokens(seed: u32, num_blocks: usize) -> Vec<u32> {
    (0..num_blocks)
        .flat_map(|block_idx| {
            let token = seed + block_idx as u32;
            (0..BLOCK_SIZE as usize).map(move |_| token)
        })
        .collect()
}

async fn route_approx_writes(
    indexer: &(dyn KvIndexerInterface + Send + Sync),
) -> anyhow::Result<()> {
    let writes = [
        (WorkerWithDpRank::new(0, 0), approx_tokens(10, 1)),
        (WorkerWithDpRank::new(1, 0), approx_tokens(20, 2)),
        (WorkerWithDpRank::new(0, 1), approx_tokens(30, 3)),
    ];

    for (worker, tokens) in writes {
        let mut tokens_with_hashes = TokensWithHashes::new(tokens, BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await?;
    }

    Ok(())
}

#[cfg(feature = "mocker-kvbm-offload")]
#[derive(Clone, Copy, Debug, Default)]
struct HostPinnedEventCounts {
    stored: usize,
    removed: usize,
}

#[cfg(feature = "mocker-kvbm-offload")]
fn count_host_pinned_events(artifacts: &[WorkerReplayArtifacts]) -> HostPinnedEventCounts {
    let mut counts = HostPinnedEventCounts::default();
    for event in artifacts
        .iter()
        .flat_map(|artifact| artifact.kv_events.iter())
        .filter(|event| event.storage_tier == StorageTier::HostPinned)
    {
        match &event.event.data {
            KvCacheEventData::Stored(_) => counts.stored += 1,
            KvCacheEventData::Removed(_) => counts.removed += 1,
            KvCacheEventData::Cleared => {}
        }
    }
    counts
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

#[test]
fn removed_legacy_branch_sharded_name_is_rejected() {
    let removed_name = format!("{}-{}-branch-sharded-crtc", "anchor", "aware");
    let error = MooncakeIndexerConfig::from_short_name(&removed_name, 4).unwrap_err();
    let message = error.to_string();

    assert!(message.contains("Unknown indexer"));
    assert!(message.contains("branch-sharded-crtc"));
    let valid_names = message.split("Valid names: ").nth(1).unwrap_or("");
    assert!(!valid_names.contains(&removed_name));
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
                    ..Default::default()
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 2,
                    hash_ids: vec![3, 4],
                    delay_after_previous_ms: 5.0,
                    ..Default::default()
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
async fn mooncake_approx_ttl_drain_leaves_indexer_dumps_empty() -> anyhow::Result<()> {
    let warning_count = support::warning_counter(&["dynamo_kv_router::indexer", "dynamo_mocker"]);
    let ttl = Duration::from_millis(250);
    let variants = [
        MooncakeIndexerConfig::radix_tree(),
        MooncakeIndexerConfig::nested_map(8, NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree(NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree_compressed(NUM_EVENT_WORKERS),
    ];

    for config in &variants {
        let label = config.short_name();
        support::reset_warning_count(&warning_count);
        let indexer = config.build_approximate_with_prune_config(
            BLOCK_SIZE,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            PruneConfig { ttl },
        )?;

        route_approx_writes(indexer.as_ref()).await?;
        indexer.flush().await;

        assert!(
            !indexer.dump_events().await?.is_empty(),
            "{label} test setup should populate approximate blocks"
        );

        tokio::time::sleep(ttl + Duration::from_millis(100)).await;
        indexer.flush().await;

        assert!(
            indexer.dump_events().await?.is_empty(),
            "{label} should not dump any visible store events after TTL drain"
        );

        indexer.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(
            warning_count.load(Ordering::Relaxed),
            0,
            "{label} emitted warn/error logs from dynamo_kv_router::indexer or dynamo_mocker"
        );
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_trace_replays_without_warnings_across_indexer_variants() -> anyhow::Result<()> {
    let warning_count = support::warning_counter(&["dynamo_kv_router::indexer", "dynamo_mocker"]);

    let fixture = support::fixture_path("mooncake_trace_1000.jsonl")?;
    let traces =
        process_mooncake_trace(&fixture, BLOCK_SIZE, 1, 1, NUM_UNIQUE_INFERENCE_WORKERS, 42)?;
    let artifact_sets = generate_mock_engine_parity_artifacts(&traces).await?;

    let variants = [
        MooncakeIndexerConfig::radix_tree(),
        MooncakeIndexerConfig::nested_map(8, NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree(NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::concurrent_radix_tree_compressed(NUM_EVENT_WORKERS),
        MooncakeIndexerConfig::branch_sharded_crtc(2, NUM_EVENT_WORKERS, 2),
    ];

    for config in &variants {
        let mut inputs = artifact_sets
            .iter()
            .map(|artifact_set| {
                (
                    format!(
                        "{} {} ({})",
                        config.short_name(),
                        artifact_set.engine_name,
                        MooncakeReplayMode::KvEvents.name()
                    ),
                    MooncakeBenchmarkInput::KvEvents(artifact_set.artifacts.clone()),
                    MooncakeReplayMode::KvEvents,
                )
            })
            .collect::<Vec<_>>();
        if config.supports_approximate() {
            inputs.push((
                format!(
                    "{} ({})",
                    config.short_name(),
                    MooncakeReplayMode::Approx.name()
                ),
                MooncakeBenchmarkInput::Approx(traces.clone()),
                MooncakeReplayMode::Approx,
            ));
        }

        for (label, input, mode) in inputs {
            support::reset_warning_count(&warning_count);

            let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
            let indexer = match mode {
                MooncakeReplayMode::KvEvents => config.build(BLOCK_SIZE, Arc::clone(&metrics)),
                MooncakeReplayMode::Approx => {
                    config.build_approximate(BLOCK_SIZE, Arc::clone(&metrics))?
                }
            };
            let run = run_benchmark(
                indexer,
                &input,
                MooncakeBenchmarkConfig {
                    benchmark_duration_ms: BENCHMARK_DURATION_MS,
                    inference_worker_duplication_factor: 1,
                    count_events: config.supports_remove(),
                    find_matches_concurrency: 0,
                    block_size: BLOCK_SIZE,
                },
            )
            .await?;

            tokio::time::sleep(Duration::from_millis(50)).await;

            assert!(
                run.kept_up,
                "{label} replay fell behind in test profile; increase BENCHMARK_DURATION_MS if this becomes too tight"
            );
            assert!(
                run.results.ops_throughput > 0.0,
                "{label} replay should record positive throughput"
            );
            assert_eq!(
                warning_count.load(Ordering::Relaxed),
                0,
                "{label} emitted warn/error logs from dynamo_kv_router::indexer or dynamo_mocker"
            );
            if matches!(mode, MooncakeReplayMode::KvEvents) {
                assert_eq!(
                    support::duplicate_store_warning_count(metrics.as_ref()),
                    0,
                    "{label} recorded duplicate-store warning metrics"
                );
            }
        }
    }

    assert_overlap_score_parity(&variants, &traces, &artifact_sets).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_trace_branch_sharded_depth4_matches_baseline() -> anyhow::Result<()> {
    let fixture = support::fixture_path("mooncake_trace_1000.jsonl")?;
    let traces =
        process_mooncake_trace(&fixture, BLOCK_SIZE, 1, 1, NUM_UNIQUE_INFERENCE_WORKERS, 42)?;
    let artifact_sets = generate_mock_engine_parity_artifacts(&traces).await?;
    let variants = [
        MooncakeIndexerConfig::radix_tree(),
        MooncakeIndexerConfig::branch_sharded_crtc(2, NUM_EVENT_WORKERS, 4),
    ];

    assert_overlap_score_parity(&variants, &traces, &artifact_sets).await
}

#[cfg(feature = "mocker-kvbm-offload")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_trace_g2_events_replay_through_host_pinned_lower_tier() -> anyhow::Result<()> {
    let warning_count = support::warning_counter(&["dynamo_kv_router::indexer", "dynamo_mocker"]);
    support::reset_warning_count(&warning_count);

    let fixture = support::fixture_path("mooncake_trace_1000.jsonl")?;
    let traces = process_mooncake_trace(&fixture, BLOCK_SIZE, 1, 1, 1, 42)?;
    let artifacts = generate_g2_replay_artifacts_with_capacity(
        &traces,
        G2_TEST_NUM_GPU_BLOCKS,
        G2_TEST_NUM_G2_BLOCKS,
        BLOCK_SIZE,
        None,
    )
    .await?;
    let counts = count_host_pinned_events(&artifacts);

    assert!(
        counts.stored > 0,
        "mooncake G2 artifact generation should capture HostPinned Stored events; counts={counts:?}"
    );
    let (reference_scores, crtc_scores, host_pinned_dumped_events) =
        g2_lower_tier::collect_tiered_replay_scores(&artifacts).await?;
    let device_only_scores = collect_overlap_scores_for_replay(
        &MooncakeIndexerConfig::concurrent_radix_tree_compressed(NUM_EVENT_WORKERS),
        &traces,
        &artifacts,
        MooncakeReplayMode::KvEvents,
    )
    .await?;

    assert!(
        host_pinned_dumped_events > 0,
        "HostPinned lower-tier indexer should retain replayed G2 state"
    );
    assert_eq!(
        crtc_scores.len(),
        reference_scores.len(),
        "CRTC lower-tier replay produced a different request count than the reference replay"
    );
    assert_eq!(
        crtc_scores.len(),
        device_only_scores.len(),
        "tiered replay produced a different request count than device-only replay"
    );
    assert!(
        crtc_scores
            .iter()
            .zip(device_only_scores.iter())
            .any(|(tiered, device_only)| {
                g2_lower_tier::score_sum(tiered) > g2_lower_tier::score_sum(device_only)
            }),
        "HostPinned lower-tier replay should improve at least one mooncake request over device-only replay"
    );

    for (request_idx, (actual, expected)) in
        crtc_scores.iter().zip(reference_scores.iter()).enumerate()
    {
        assert_eq!(
            actual, expected,
            "CRTC lower-tier additive overlap diverged from reference at replay request {request_idx}"
        );
    }

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        warning_count.load(Ordering::Relaxed),
        0,
        "G2 HostPinned lower-tier replay emitted warn/error logs from dynamo_kv_router::indexer or dynamo_mocker"
    );

    Ok(())
}
