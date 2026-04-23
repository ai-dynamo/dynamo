// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::KvCacheEventData;
pub use dynamo_kv_router::test_utils::NoopSequencePublisher;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::loadgen::{SessionPartitionSpec, Trace};
pub use dynamo_mocker::replay::ReplayWorkerArtifacts as WorkerReplayArtifacts;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

/// Create a styled progress bar, optionally with a known total length.
pub fn make_progress_bar(total: Option<u64>) -> ProgressBar {
    let progress = match total {
        Some(total) => ProgressBar::new(total),
        None => ProgressBar::no_length(),
    };

    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    progress
}

/// Results from a single benchmark run.
#[derive(Serialize)]
pub struct BenchmarkResults {
    pub offered_ops_throughput: f32,
    pub ops_throughput: f32,
    pub offered_block_throughput: f32,
    pub block_throughput: f32,
    pub latency_p99_us: f32,
}

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

fn replay_worker_trace(
    trace: Trace,
    sched_args: MockEngineArgs,
    trace_simulation_duration_ms: u64,
    progress: ProgressBar,
) -> anyhow::Result<WorkerReplayArtifacts> {
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum::<usize>();
    let artifacts = dynamo_mocker::replay::generate_trace_worker_artifacts_offline(
        sched_args,
        trace.rescale_ready_span(trace_simulation_duration_ms)?,
    )?;
    progress.inc(total_turns as u64);
    Ok(artifacts)
}

pub async fn generate_replay_artifacts(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: u64,
) -> anyhow::Result<Vec<WorkerReplayArtifacts>> {
    println!("Generating events...");
    let sched_args = default_mock_engine_args(num_gpu_blocks, block_size as usize)?;
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

    for worker_events in artifacts.iter().map(|artifact| &artifact.kv_events) {
        for i in 1..worker_events.len() {
            assert!(worker_events[i].timestamp_us >= worker_events[i - 1].timestamp_us);
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
