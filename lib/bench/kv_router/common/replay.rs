// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::KvCacheEventData;
#[allow(unused_imports)]
pub use dynamo_kv_router::test_utils::NoopSequencePublisher;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::loadgen::{SessionPartitionSpec, Trace};
pub use dynamo_mocker::replay::ReplayWorkerArtifacts as WorkerReplayArtifacts;
use indicatif::ProgressBar;

use super::progress::make_progress_bar;

/// Load, transform, and partition the mooncake trace into per-worker request lists.
pub fn process_mooncake_trace(
    path: &str,
    block_size: u32,
    trace_length_factor: usize,
    trace_duplication_factor: usize,
    num_workers: usize,
    seed: u64,
) -> anyhow::Result<Vec<Trace>> {
    let trace = Trace::from_mooncake(std::path::Path::new(path), block_size as usize)?
        .expand_hash_prefix_depth(trace_length_factor)
        .duplicate_hash_space(trace_duplication_factor);
    Ok(trace.partition_by_session(SessionPartitionSpec::Random {
        num_partitions: num_workers,
        seed,
    }))
}

pub fn maybe_rescale_ready_span(
    trace: Trace,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Trace> {
    match trace_simulation_duration_ms {
        Some(duration_ms) => trace.rescale_ready_span(duration_ms),
        None => Ok(trace),
    }
}

/// Build default MockEngineArgs suitable for event generation.
pub fn default_mock_engine_args(
    num_gpu_blocks: usize,
    block_size: usize,
) -> anyhow::Result<MockEngineArgs> {
    Ok(MockEngineArgs::builder()
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(block_size)
        .speedup_ratio(10.0)
        .enable_prefix_caching(true)
        .max_num_batched_tokens(None)
        .max_num_seqs(None)
        .build()?)
}

#[cfg(feature = "mocker-kvbm-offload")]
#[allow(dead_code)]
pub fn g2_mock_engine_args(
    num_gpu_blocks: usize,
    block_size: usize,
    num_g2_blocks: usize,
) -> anyhow::Result<MockEngineArgs> {
    Ok(MockEngineArgs::builder()
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(block_size)
        .speedup_ratio(10.0)
        .enable_prefix_caching(true)
        .max_num_batched_tokens(None)
        .max_num_seqs(None)
        .num_g2_blocks(Some(num_g2_blocks))
        .kv_bytes_per_token(Some(1))
        .offload_batch_size(Some(32))
        .bandwidth_g1_to_g2_gbps(Some(14.0))
        .bandwidth_g2_to_g1_gbps(Some(14.0))
        .build()?)
}

fn replay_worker_trace(
    trace: Trace,
    sched_args: MockEngineArgs,
    trace_simulation_duration_ms: Option<u64>,
    progress: ProgressBar,
) -> anyhow::Result<WorkerReplayArtifacts> {
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum::<usize>();
    let artifacts = dynamo_mocker::replay::generate_trace_worker_artifacts_offline(
        sched_args,
        maybe_rescale_ready_span(trace, trace_simulation_duration_ms)?,
    )?;
    progress.inc(total_turns as u64);
    Ok(artifacts)
}

pub async fn generate_replay_artifacts_with_args(
    traces: &[Trace],
    sched_args: MockEngineArgs,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Vec<WorkerReplayArtifacts>> {
    println!("Generating events...");
    let progress = make_progress_bar(Some(
        traces
            .iter()
            .map(|trace| {
                trace
                    .sessions
                    .iter()
                    .map(|session| session.turns.len() as u64)
                    .sum::<u64>()
            })
            .sum::<u64>(),
    ));

    let mut tasks = Vec::new();
    for trace in traces.iter().cloned() {
        let sched_args = sched_args.clone();
        let progress = progress.clone();
        tasks.push(tokio::task::spawn_blocking(move || {
            replay_worker_trace(trace, sched_args, trace_simulation_duration_ms, progress)
        }));
    }

    let mut artifacts = Vec::new();
    for task in tasks {
        artifacts.push(task.await??);
    }

    for (worker_idx, worker_events) in artifacts
        .iter()
        .enumerate()
        .map(|(worker_idx, artifact)| (worker_idx, &artifact.kv_events))
    {
        for i in 1..worker_events.len() {
            assert!(
                worker_events[i].timestamp_us >= worker_events[i - 1].timestamp_us,
                "worker {worker_idx} non-monotonic kv_events at idx {i}: prev={}, curr={}",
                worker_events[i - 1].timestamp_us,
                worker_events[i].timestamp_us
            );
        }
    }

    println!(
        "Generated {} events. Processing...",
        artifacts
            .iter()
            .map(|artifact| artifact.kv_events.len())
            .sum::<usize>()
    );
    let mut num_stored_events = 0;
    let mut num_removed_events = 0;
    for event in artifacts
        .iter()
        .flat_map(|artifact| artifact.kv_events.iter())
    {
        match event.event.data {
            KvCacheEventData::Stored(_) => num_stored_events += 1,
            KvCacheEventData::Removed(_) => num_removed_events += 1,
            _ => (),
        }
    }

    println!("Store events: {}", num_stored_events);
    println!("Remove events: {}", num_removed_events);

    Ok(artifacts)
}

pub async fn generate_replay_artifacts(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Vec<WorkerReplayArtifacts>> {
    let sched_args = default_mock_engine_args(num_gpu_blocks, block_size as usize)?;
    generate_replay_artifacts_with_args(traces, sched_args, trace_simulation_duration_ms).await
}

#[cfg(feature = "mocker-kvbm-offload")]
#[allow(dead_code)]
pub async fn generate_g2_replay_artifacts_with_capacity(
    traces: &[Trace],
    num_gpu_blocks: usize,
    num_g2_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Vec<WorkerReplayArtifacts>> {
    let sched_args = g2_mock_engine_args(num_gpu_blocks, block_size as usize, num_g2_blocks)?;
    generate_replay_artifacts_with_args(traces, sched_args, trace_simulation_duration_ms).await
}
