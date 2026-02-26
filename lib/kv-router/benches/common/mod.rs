// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, XXH3_SEED};
use dynamo_mocker::common::protocols::{DirectRequest, KvCacheEventSink, MockEngineArgs};
use dynamo_mocker::scheduler::Scheduler;
use dynamo_tokens::compute_hash_v2;
use indicatif::{ProgressBar, ProgressStyle};
use plotters::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
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
    #[clap(long, default_value = "1048576")]
    pub num_gpu_blocks: usize,

    /// Number of tokens per KV cache block.
    #[clap(long, default_value = "512")]
    pub block_size: u32,

    /// Wall-clock duration (ms) over which the trace is replayed during event generation.
    #[clap(long, default_value = "30000")]
    pub trace_simulation_duration_ms: u64,

    /// Wall-clock duration (ms) over which the benchmark replays operations.
    #[clap(long, default_value = "60000")]
    pub benchmark_duration_ms: u64,

    /// Number of unique simulated inference workers.
    #[clap(short, long, default_value = "256")]
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
}

/// A single request deserialized from the mooncake trace JSONL.
#[derive(Serialize, Deserialize, Clone)]
pub struct MooncakeRequest {
    #[serde(default = "Uuid::new_v4")]
    pub uuid: uuid::Uuid,
    pub timestamp: u64,
    pub hash_ids: Vec<u64>,
    pub output_length: u64,
}

/// Collects KV cache events emitted by the mock engine during event generation,
/// tagging each with the wall-clock instant it was produced.
pub struct EventCollector {
    events: Mutex<Option<Vec<(KvCacheEvent, Instant)>>>,
}

impl EventCollector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Some(Vec::new())),
        })
    }

    pub fn get_events(self: Arc<Self>) -> Vec<(KvCacheEvent, Instant)> {
        self.events.lock().unwrap().take().unwrap()
    }
}

impl KvCacheEventSink for EventCollector {
    fn publish(
        &self,
        event: KvCacheEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        let timestamp = Instant::now();
        if let Some(events) = self.events.lock().unwrap().as_mut() {
            events.push((event, timestamp));
        }
        Ok(())
    }
}

/// Load the mooncake trace from disk into a flat list of requests.
pub fn load_mooncake_trace(path: &str) -> anyhow::Result<Vec<MooncakeRequest>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    println!("Loading trace...");
    let progress = make_progress_bar(None);

    let mut requests = Vec::new();
    for line in reader.lines() {
        requests.push(serde_json::from_str::<MooncakeRequest>(&line?)?);
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
    request
        .hash_ids
        .iter()
        .flat_map(|id| (0..block_size).map(|_| *id as u32))
        .collect()
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
    trace_length_factor: usize,
    trace_duplication_factor: usize,
    num_workers: usize,
    seed: u64,
) -> anyhow::Result<Vec<Vec<MooncakeRequest>>> {
    let requests = load_mooncake_trace(path)?;
    let requests = expand_trace_lengths(requests, trace_length_factor);
    let requests = duplicate_traces(requests, trace_duplication_factor);
    Ok(partition_trace(requests, num_workers, seed))
}

/// Build default MockEngineArgs suitable for event generation.
pub fn default_mock_engine_args(
    num_gpu_blocks: usize,
    block_size: usize,
) -> anyhow::Result<MockEngineArgs> {
    Ok(MockEngineArgs::builder()
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(block_size)
        .speedup_ratio(0.0)
        .enable_prefix_caching(true)
        .max_num_batched_tokens(None)
        .max_num_seqs(None)
        .build()?)
}

/// Replay each worker's request trace through a mock engine in real-time to
/// produce the KV cache events (store/remove/clear) that the engine would emit.
///
/// Returns one event list per worker, each entry paired with the wall-clock
/// instant it was produced.
pub async fn generate_kv_events(
    traces: &[Vec<MooncakeRequest>],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: u64,
) -> anyhow::Result<Vec<Vec<(KvCacheEvent, Instant)>>> {
    println!("Generating events...");
    let sched_args = default_mock_engine_args(num_gpu_blocks, block_size as usize)?;

    let scaled_traces = traces
        .iter()
        .map(|worker_trace| scale_mooncake_trace(worker_trace, trace_simulation_duration_ms));

    let progress = make_progress_bar(Some(
        traces.iter().map(|worker| worker.len() as u64).sum::<u64>(),
    ));

    let mut tasks: Vec<JoinHandle<Vec<(KvCacheEvent, Instant)>>> = Vec::new();
    for worker_trace in scaled_traces {
        let sched_args = sched_args.clone();
        let progress = progress.clone();
        tasks.push(tokio::spawn(async move {
            let collector = EventCollector::new();

            let scheduler = Scheduler::new(sched_args, 0, None, Some(collector.clone()), None);

            let mut i = 0;
            let mut target = Instant::now();

            while i < worker_trace.len() {
                let prev_i = i;
                scheduler
                    .receive(DirectRequest {
                        tokens: tokens_from_request(&worker_trace[i], block_size),
                        max_output_tokens: worker_trace[i].output_length as usize,
                        uuid: Some(worker_trace[i].uuid),
                        dp_rank: 0,
                    })
                    .await;
                i += 1;

                while i < worker_trace.len()
                    && worker_trace[i].timestamp == worker_trace[i - 1].timestamp
                {
                    scheduler
                        .receive(DirectRequest {
                            tokens: tokens_from_request(&worker_trace[i], block_size),
                            max_output_tokens: worker_trace[i].output_length as usize,
                            uuid: Some(worker_trace[i].uuid),
                            dp_rank: 0,
                        })
                        .await;
                    i += 1;
                }

                if i < worker_trace.len() {
                    target += Duration::from_millis(
                        worker_trace[i].timestamp - worker_trace[i - 1].timestamp,
                    );
                }

                tokio::time::sleep_until(target).await;
                progress.inc((i - prev_i) as u64);
            }

            collector.get_events()
        }));
    }

    let mut events = Vec::new();
    for task in tasks {
        events.push(task.await?);
    }

    for worker_events in &events {
        for i in 1..worker_events.len() {
            assert!(worker_events[i].1 >= worker_events[i - 1].1);
        }
    }

    println!(
        "Generated {} events. Processing...",
        events.iter().map(|e| e.len()).sum::<usize>()
    );

    if progress.elapsed() > Duration::from_millis(trace_simulation_duration_ms * 11 / 10) {
        eprintln!(
            "Warning: Generated events took significantly longer than the trace simulation duration. Inaccurate timing information has been produced. Rerun with a larger --trace-simulation-duration-ms."
        );
    }

    let mut num_stored_events = 0;
    let mut num_removed_events = 0;
    for event in events.iter().flatten() {
        match event.0.data {
            KvCacheEventData::Stored(_) => num_stored_events += 1,
            KvCacheEventData::Removed(_) => num_removed_events += 1,
            _ => (),
        }
    }

    println!("Store events: {}", num_stored_events);
    println!("Remove events: {}", num_removed_events);

    Ok(events)
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
