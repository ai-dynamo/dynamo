// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unused_imports)]

use std::time::Duration;

#[path = "shared.rs"]
mod shared;

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::XXH3_SEED;
use dynamo_mocker::loadgen::{ReplayRequestHashes, Trace};
use dynamo_tokens::compute_hash_v2;
use plotters::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

pub use shared::{
    BenchmarkResults, BenchmarkRun, NoopSequencePublisher, WorkerReplayArtifacts,
    compute_benchmark_run, default_mock_engine_args, generate_replay_artifacts, make_progress_bar,
    process_mooncake_trace, rescale_trace_timestamps,
};

/// Shared CLI arguments for trace-based benchmarks.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Path to a JSONL mooncake trace file.
    pub mooncake_trace_path: Option<String>,

    /// Deprecated compatibility flag. Use `cargo test --package dynamo-bench --test ...`
    /// for the fixture-backed integration tests instead.
    #[clap(long)]
    pub test: bool,

    /// Number of GPU blocks available in the mock engine's KV cache.
    #[clap(long, default_value = "16384")]
    pub num_gpu_blocks: usize,

    /// Number of tokens per KV cache block.
    #[clap(long, default_value = "128")]
    pub block_size: u32,

    /// Optional wall-clock duration (ms) used to rescale the trace during event generation.
    /// Omit to preserve the original Mooncake timestamps.
    #[clap(long)]
    pub trace_simulation_duration_ms: Option<u64>,

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

#[derive(Deserialize)]
struct RawMooncakeRecord {
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    delay: Option<f64>,
    hash_ids: Vec<u64>,
    #[serde(alias = "output_length", alias = "osl")]
    output_length: u64,
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

/// Compute median of durations.
pub fn median(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
