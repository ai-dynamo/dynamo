// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::KvCacheEventData;
#[allow(unused_imports)]
pub use dynamo_kv_router::test_utils::NoopSequencePublisher;
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::loadgen::{SessionPartitionSpec, Trace};
pub use dynamo_mocker::replay::ReplayWorkerArtifacts as WorkerReplayArtifacts;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::time::Duration;

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
#[derive(Clone, Copy, Serialize)]
pub struct BenchmarkResults {
    pub offered_ops_throughput: f32,
    pub ops_throughput: f32,
    pub offered_block_throughput: f32,
    pub block_throughput: f32,
    pub latency_p99_us: f32,
}

#[derive(Clone, Copy)]
pub struct BenchmarkRun {
    pub results: BenchmarkResults,
    pub kept_up: bool,
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

pub fn maybe_rescale_ready_span(
    trace: Trace,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Trace> {
    match trace_simulation_duration_ms {
        Some(duration_ms) => trace.rescale_ready_span(duration_ms),
        None => Ok(trace),
    }
}

pub fn rescale_trace_timestamps<T, GetTimestamp, WithTimestamp>(
    traces: &[Vec<T>],
    benchmark_duration_ms: u64,
    timestamp_of: GetTimestamp,
    with_timestamp: WithTimestamp,
) -> Vec<Vec<T>>
where
    GetTimestamp: Fn(&T) -> u64 + Copy,
    WithTimestamp: Fn(&T, u64) -> T + Copy,
{
    let target_us = u128::from(benchmark_duration_ms) * 1000;

    traces
        .iter()
        .map(|worker_trace| {
            if worker_trace.is_empty() {
                return Vec::new();
            }

            let max_timestamp_us = worker_trace.last().map(timestamp_of).unwrap_or(1).max(1);

            worker_trace
                .iter()
                .map(|entry| {
                    let scaled_timestamp =
                        u128::from(timestamp_of(entry)) * target_us / u128::from(max_timestamp_us);
                    with_timestamp(entry, scaled_timestamp.min(u128::from(u64::MAX)) as u64)
                })
                .collect()
        })
        .collect()
}

pub fn compute_benchmark_run(
    total_ops: usize,
    total_blocks: usize,
    benchmark_duration_ms: u64,
    total_duration: Duration,
    mut latencies_ns: Vec<u64>,
) -> BenchmarkRun {
    let kept_up = total_duration <= Duration::from_millis(benchmark_duration_ms * 11 / 10);
    let benchmark_duration_secs = (benchmark_duration_ms as f32 / 1000.0).max(1e-6);
    let total_duration_secs = total_duration.as_secs_f32().max(1e-6);
    let offered_ops_throughput = total_ops as f32 / benchmark_duration_secs;
    let ops_throughput = total_ops as f32 / total_duration_secs;
    let offered_block_throughput = total_blocks as f32 / benchmark_duration_secs;
    let block_throughput = total_blocks as f32 / total_duration_secs;

    latencies_ns.sort_unstable();
    let latency_p99_us = if latencies_ns.is_empty() {
        0.0
    } else {
        let p99_idx = latencies_ns.len().saturating_sub(1) * 99 / 100;
        latencies_ns[p99_idx] as f32 / 1000.0
    };

    BenchmarkRun {
        results: BenchmarkResults {
            offered_ops_throughput,
            ops_throughput,
            offered_block_throughput,
            block_throughput,
            latency_p99_us,
        },
        kept_up,
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

pub async fn generate_replay_artifacts(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: Option<u64>,
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
