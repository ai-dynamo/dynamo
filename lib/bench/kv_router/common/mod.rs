// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unused_imports)]

use std::time::Duration;

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, WorkerId, XXH3_SEED, compute_seq_hash_for_block,
};
pub use dynamo_kv_router::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::loadgen::{
    ArrivalSpec, DelaySpec, LengthSpec, ReplayRequestHashes, RouterSequence, SequenceHashMode,
    SessionPartitionSpec, SyntheticTraceSpec, Trace,
};
pub use dynamo_mocker::replay::{
    ReplayTimedKvEvent as TimedKvEvent, ReplayTimedOutputSignal as TimedOutputSignal,
    ReplayTimedRequest as TimedReplayRequest, ReplayWorkerArtifacts as WorkerReplayArtifacts,
};
use dynamo_tokens::compute_hash_v2;
use indicatif::{ProgressBar, ProgressStyle};
use plotters::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokio::task::JoinHandle;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

/// Shared CLI arguments for trace-based benchmarks.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Path to a JSONL mooncake trace file.
    pub mooncake_trace_path: Option<String>,

    /// Run built-in self-tests instead of the benchmark.
    #[clap(long)]
    pub test: bool,

    /// Number of GPU blocks available in the mock engine's KV cache.
    #[clap(long, default_value = "16384")]
    pub num_gpu_blocks: usize,

    /// Number of tokens per KV cache block.
    #[clap(long, default_value = "128")]
    pub block_size: u32,

    /// Wall-clock duration (ms) over which the trace is replayed during event generation.
    #[clap(long, default_value = "30000")]
    pub trace_simulation_duration_ms: u64,

    /// Wall-clock duration (ms) over which the benchmark replays operations.
    #[clap(long, default_value = "60000")]
    pub benchmark_duration_ms: u64,

    /// Number of unique simulated inference workers.
    #[clap(short, long, default_value = "1000")]
    pub num_unique_inference_workers: usize,

    /// How many times to duplicate unique workers during the benchmark phase.
    #[clap(short = 'd', long, default_value = "1")]
    pub inference_worker_duplication_factor: usize,

    /// Factor by which to stretch each request's hash sequence length.
    #[clap(long, default_value = "1")]
    pub trace_length_factor: usize,

    /// How many times to duplicate the raw trace data with offset hash_ids.
    #[clap(long, default_value = "1")]
    pub trace_duplication_factor: usize,

    /// RNG seed for reproducible worker-to-trace assignment.
    #[clap(long, default_value = "42")]
    pub seed: u64,

    /// Enable throughput vs p99 latency sweep mode.
    #[clap(long)]
    pub sweep: bool,

    /// Minimum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "1000")]
    pub sweep_min_ms: u64,

    /// Maximum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "50000")]
    pub sweep_max_ms: u64,

    /// Number of logarithmically spaced sweep steps between min and max.
    #[clap(long, default_value = "10")]
    pub sweep_steps: usize,

    /// Ignored - passed by cargo bench harness.
    #[arg(long, hide = true, global = true)]
    pub bench: bool,

    /// Opt in to runtime warn/error logs from the mocker and sequence tracker.
    #[clap(long)]
    pub sequence_logs: bool,
}

pub fn init_sequence_logging(enabled: bool) {
    if !enabled {
        return;
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "error,dynamo_kv_router::sequences=warn,dynamo_mocker=warn",
        ))
        .with_writer(std::io::stderr)
        .try_init();
}

/// A single request deserialized from the mooncake trace JSONL.
#[derive(Serialize, Deserialize, Clone)]
pub struct MooncakeRequest {
    #[serde(default = "Uuid::new_v4")]
    pub uuid: uuid::Uuid,
    pub timestamp: u64,
    #[serde(default)]
    pub input_length: usize,
    pub hash_ids: Vec<u64>,
    #[serde(alias = "output_length", alias = "osl")]
    pub output_length: u64,
}

/// Load the mooncake trace from disk into a flat list of requests.
///
/// Supports two JSONL formats:
///   - Legacy: every record has an integer `timestamp` field (absolute ms).
///   - aiperf: first record has `timestamp` (float), subsequent records have
///     `delay` (float ms since previous). Absolute timestamps are reconstructed
///     by accumulating delays.
pub fn load_mooncake_trace(path: &str) -> anyhow::Result<Vec<MooncakeRequest>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    println!("Loading trace...");
    let progress = make_progress_bar(None);

    let mut requests = Vec::new();
    let mut cursor_ms: f64 = 0.0;

    for line in reader.lines() {
        let raw: RawMooncakeRecord = serde_json::from_str(&line?)?;

        if let Some(ts) = raw.timestamp {
            cursor_ms = ts;
        } else if let Some(d) = raw.delay {
            cursor_ms += d;
        }

        requests.push(MooncakeRequest {
            uuid: Uuid::new_v4(),
            timestamp: cursor_ms as u64,
            input_length: 0,
            hash_ids: raw.hash_ids,
            output_length: raw.output_length,
        });
        progress.inc(1);
    }

    Ok(requests)
}

/// Randomly partition a flat request list across `num_workers` worker buckets.
pub fn partition_trace(
    requests: Vec<MooncakeRequest>,
    num_workers: usize,
    seed: u64,
) -> Vec<Vec<MooncakeRequest>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut traces: Vec<Vec<MooncakeRequest>> = (0..num_workers).map(|_| Vec::new()).collect();
    for request in requests {
        traces[rng.random_range(0..num_workers)].push(request);
    }
    // Sort each worker's trace by timestamp so that scale_mooncake_trace and
    // generate_kv_events see monotonically increasing timestamps.  Without this,
    // mixing requests from multiple sessions (each starting at timestamp=0) into
    // one worker produces non-monotonic sequences; u64 underflow in the delta
    // computation then creates sleep durations measured in centuries.
    for trace in &mut traces {
        trace.sort_by_key(|r| r.timestamp);
    }
    traces
}

/// Linearly rescale all timestamps in a worker's trace so the total span equals
/// `duration` milliseconds.
pub fn scale_mooncake_trace(trace: &[MooncakeRequest], duration: u64) -> Vec<MooncakeRequest> {
    let Some(first) = trace.first() else {
        return Vec::new();
    };
    let total_duration = trace.last().unwrap().timestamp - first.timestamp;
    if total_duration == 0 {
        return trace
            .iter()
            .map(|r| MooncakeRequest {
                timestamp: 0,
                ..r.clone()
            })
            .collect();
    }
    trace
        .iter()
        .map(|request| MooncakeRequest {
            timestamp: (request.timestamp - first.timestamp) * duration / total_duration,
            ..request.clone()
        })
        .collect()
}

/// Stretch each request's hash sequence by the given factor, simulating longer
/// prefix chains with the same tree structure.
///
/// Each hash `h` becomes `factor` consecutive hashes:
/// `h * factor`, `h * factor + 1`, ..., `h * factor + (factor - 1)`.
/// Two sequences that shared a k-block prefix now share a k*factor-block prefix.
pub fn expand_trace_lengths(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }

    println!("Expanding trace lengths by {}x", factor);

    requests
        .into_iter()
        .map(|mut request| {
            request.hash_ids = request
                .hash_ids
                .iter()
                .flat_map(|&h| {
                    let base = h * factor as u64;
                    (0..factor as u64).map(move |offset| base + offset)
                })
                .collect();
            request
        })
        .collect()
}

/// Duplicate all worker traces with offset hash_ids, creating `factor`
/// structurally identical copies of the prefix tree with disjoint hash spaces.
///
/// Copy `d` (1-indexed) offsets every hash_id by `(max_hash_id + 1) * d`.
/// The original traces (copy 0) are kept as-is.
pub fn duplicate_traces(requests: Vec<MooncakeRequest>, factor: usize) -> Vec<MooncakeRequest> {
    if factor <= 1 {
        return requests;
    }

    let max_hash_id = requests
        .iter()
        .flat_map(|r| r.hash_ids.iter().copied())
        .max()
        .unwrap_or(0);
    let offset_base = max_hash_id + 1;

    println!(
        "Duplicating traces: {}x (hash offset base: {})",
        factor, offset_base
    );

    let mut out = Vec::with_capacity(requests.len() * factor);
    for r in &requests {
        for d in 0..factor {
            let offset = offset_base * d as u64;
            out.push(MooncakeRequest {
                uuid: Uuid::new_v4(),
                hash_ids: r.hash_ids.iter().map(|&h| h + offset).collect(),
                ..r.clone()
            });
        }
    }
    out
}

/// Expand a request's block-level hash_ids into per-token IDs by repeating each
/// hash_id `block_size` times.
pub fn tokens_from_request(request: &MooncakeRequest, block_size: u32) -> Vec<u32> {
    let mut tokens = request
        .hash_ids
        .iter()
        .flat_map(|id| (0..block_size).map(|_| *id as u32))
        .collect::<Vec<_>>();
    if request.input_length > 0 && request.input_length < tokens.len() {
        tokens.truncate(request.input_length);
    }
    tokens
}

/// Compute the LocalBlockHash for a block-level hash_id the same way the mock
/// engine does: expand to `block_size` repeated u32 tokens, then XXH3 hash.
pub fn local_block_hash_from_id(hash_id: u64, block_size: u32) -> LocalBlockHash {
    let tokens: Vec<u32> = (0..block_size).map(|_| hash_id as u32).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
    LocalBlockHash(compute_hash_v2(bytes, XXH3_SEED))
}

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

    let mut tasks: Vec<JoinHandle<anyhow::Result<WorkerReplayArtifacts>>> = Vec::new();
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

pub async fn generate_kv_events(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: u64,
) -> anyhow::Result<Vec<Vec<TimedKvEvent>>> {
    Ok(generate_replay_artifacts(
        traces,
        num_gpu_blocks,
        block_size,
        trace_simulation_duration_ms,
    )
    .await?
    .into_iter()
    .map(|artifact| artifact.kv_events)
    .collect())
}

pub fn plot_sweep(
    all_results: &[(&str, Vec<(u64, BenchmarkResults)>)],
    output_path: &str,
) -> anyhow::Result<()> {
    use plotters::coord::combinators::IntoLogRange;
    use plotters::element::DashedPathElement;
    use plotters::style::ShapeStyle;

    let colors = [
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
    ];

    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;
    for (_, results) in all_results {
        for (_, r) in results {
            let offered = r.offered_block_throughput as f64;
            let achieved = r.block_throughput as f64;
            global_min = global_min.min(offered).min(achieved);
            global_max = global_max.max(offered).max(achieved);
        }
    }
    let axis_min = global_min * 0.9;
    let axis_max = global_max * 1.1;

    let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Achieved vs Offered Throughput",
            ("sans-serif", 22).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (axis_min..axis_max).log_scale(),
            (axis_min..axis_max).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("Offered Throughput (block ops/s)")
        .y_desc("Achieved Throughput (block ops/s)")
        .draw()?;

    let identity_style = ShapeStyle::from(&BLACK.mix(0.4)).stroke_width(1);
    chart.draw_series(std::iter::once(DashedPathElement::new(
        vec![(axis_min, axis_min), (axis_max, axis_max)],
        5,
        3,
        identity_style,
    )))?;

    for (i, (name, results)) in all_results.iter().enumerate() {
        let color = &colors[i % colors.len()];

        let points: Vec<(f64, f64)> = results
            .iter()
            .map(|(_, r)| (r.offered_block_throughput as f64, r.block_throughput as f64))
            .collect();

        let series_color = *color;
        chart
            .draw_series(LineSeries::new(
                points.iter().map(|&(x, y)| (x, y)),
                &series_color,
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    series_color.stroke_width(2),
                )
            });

        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 4, series_color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Sweep plot saved to {}", output_path);
    Ok(())
}

/// Compute logarithmically spaced benchmark durations for sweep mode.
pub fn compute_sweep_durations(min_ms: u64, max_ms: u64, steps: usize) -> Vec<u64> {
    let log_min = (min_ms as f64).ln();
    let log_max = (max_ms as f64).ln();
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            (log_max * (1.0 - t) + log_min * t).exp().round() as u64
        })
        .collect()
}

/// Print a formatted sweep summary table.
pub fn print_sweep_summary(name: &str, results: &[(u64, BenchmarkResults)]) {
    println!("\n=== Sweep Summary: {} ===", name);
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "duration_ms", "ops/s_off", "ops/s", "blk_ops/s_off", "blk_ops/s", "p99(us)"
    );
    for (dur, r) in results {
        println!(
            "{:>12} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>10.1}",
            dur,
            r.offered_ops_throughput,
            r.ops_throughput,
            r.offered_block_throughput,
            r.block_throughput,
            r.latency_p99_us,
        );
    }
}

// ---------------------------------------------------------------------------
// Sequence data generation (moved from src/bench_utils.rs)
// ---------------------------------------------------------------------------

/// Pre-generated sequence data for benchmarking.
#[derive(Clone)]
pub struct SequenceData {
    pub worker_id: WorkerId,
    pub local_hashes: Vec<LocalBlockHash>,
    pub external_hashes: Vec<ExternalSequenceBlockHash>,
}

impl From<RouterSequence> for SequenceData {
    fn from(sequence: RouterSequence) -> Self {
        Self {
            worker_id: sequence.worker_id,
            local_hashes: sequence.local_hashes,
            external_hashes: sequence.external_hashes,
        }
    }
}

impl SequenceData {
    /// Create a new sequence with synthetic hashes based on sequence ID.
    pub fn new(seq_id: u64, worker_id: WorkerId, depth: usize) -> Self {
        let local_hashes: Vec<LocalBlockHash> = (0..depth)
            .map(|block_idx| LocalBlockHash((seq_id << 32) | (block_idx as u64)))
            .collect();

        let external_hashes: Vec<ExternalSequenceBlockHash> = (0..depth)
            .map(|block_idx| ExternalSequenceBlockHash((seq_id << 32) | (block_idx as u64)))
            .collect();

        Self {
            worker_id,
            local_hashes,
            external_hashes,
        }
    }

    /// Create a sequence from local hashes, computing external hashes using cumulative hash.
    pub fn from_local_hashes(worker_id: WorkerId, local_hashes: Vec<LocalBlockHash>) -> Self {
        let seq_hashes = compute_seq_hash_for_block(&local_hashes);
        let external_hashes = seq_hashes
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect();

        Self {
            worker_id,
            local_hashes,
            external_hashes,
        }
    }

    /// Convert to a store event.
    pub fn to_store_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            self.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: self
                        .local_hashes
                        .iter()
                        .zip(self.external_hashes.iter())
                        .map(|(local, ext)| KvCacheStoredBlockData {
                            tokens_hash: *local,
                            block_hash: *ext,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    /// Convert to a remove event.
    pub fn to_remove_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            self.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: self.external_hashes.clone(),
                }),
                dp_rank: 0,
            },
        )
    }
}

/// Generate sequences with shared prefix prompts.
pub fn generate_sequences(
    num_sequences: usize,
    depth: usize,
    num_workers: usize,
    prefix_ratio: f64,
    num_prefix_groups: usize,
    seed: u64,
    use_cumulative_hash: bool,
) -> Vec<SequenceData> {
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: 1,
        num_sessions: num_sequences,
        turns_per_session: 1,
        input_tokens: LengthSpec {
            mean: depth,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 1,
            stddev: 0.0,
        },
        shared_prefix_ratio: prefix_ratio,
        num_prefix_groups,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::None,
        seed,
    })
    .expect("sequence generation spec must be valid");
    let hash_mode = if use_cumulative_hash {
        SequenceHashMode::Cumulative
    } else {
        SequenceHashMode::Raw
    };

    trace
        .partition_by_session(SessionPartitionSpec::RoundRobin {
            num_partitions: num_workers,
        })
        .into_iter()
        .enumerate()
        .flat_map(|(worker_idx, partition)| {
            partition
                .to_router_sequences(worker_idx as WorkerId, hash_mode)
                .expect("synthetic trace conversion must succeed")
                .into_iter()
                .map(SequenceData::from)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Compute median of durations.
pub fn median(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiturn_trace() -> Trace {
        Trace {
            block_size: 2,
            sessions: vec![dynamo_mocker::loadgen::SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    dynamo_mocker::loadgen::TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![1, 2],
                        delay_after_previous_ms: 0.0,
                    },
                    dynamo_mocker::loadgen::TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![3, 4],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            }],
        }
    }

    #[tokio::test]
    async fn test_replay_worker_trace_releases_follow_up_turn_after_completion_delay() {
        let artifacts = replay_worker_trace(
            multiturn_trace(),
            default_mock_engine_args(1024, 2).unwrap(),
            5,
            make_progress_bar(Some(2)),
        )
        .await
        .unwrap();

        assert_eq!(artifacts.requests.len(), 2);
        let first_uuid = artifacts.requests[0].uuid;
        let first_completion_ms = artifacts
            .output_signals
            .iter()
            .find(|signal| signal.signal.uuid == first_uuid && signal.signal.completed)
            .unwrap()
            .timestamp_us as f64
            / 1000.0;
        assert!(
            artifacts.requests[1].scheduled_ready_at_ms + 0.1 >= first_completion_ms + 5.0,
            "expected follow-up turn to wait for completion plus delay, got ready_at={} completion_at={}",
            artifacts.requests[1].scheduled_ready_at_ms,
            first_completion_ms
        );
    }
}
