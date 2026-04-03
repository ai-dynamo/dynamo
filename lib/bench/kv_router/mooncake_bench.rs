// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/mod.rs"]
mod common;
use common::*;


use clap::{Parser, Subcommand};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded, ShardSizeSnapshot,
};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent};
use dynamo_kv_router::{
    BranchShardedIndexer, ConcurrentRadixTree, ConcurrentRadixTreeCompressed, PositionalIndexer,
    ThreadPoolIndexer,
};
use dynamo_mocker::loadgen::Trace;
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Indexer backend selection and its backend-specific parameters.
#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    /// Single-threaded radix tree indexer.
    RadixTree {},

    /// Position-based nested map indexer with jump search.
    NestedMap {
        /// Number of positions to skip during jump search before scanning back.
        #[clap(long, default_value = "8")]
        jump_size: usize,

        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Lock-based concurrent radix tree indexer.
    ConcurrentRadixTree {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Compressed concurrent radix tree indexer (compressed edges).
    ConcurrentRadixTreeCompressed {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Branch-sharded CRTC: N independent CRTC shards assigned via an explicit routing
    /// table keyed on the first K block hashes. New branches are assigned to the
    /// least-loaded shard. find_matches touches exactly ONE shard (no scatter-gather).
    /// Unknown branch keys return empty scores immediately without any dispatch.
    BranchShardedCrtc {
        /// Number of independent CRTC shards.
        #[clap(long, default_value = "2")]
        num_shards: usize,

        /// Number of OS event-worker threads per shard.
        #[clap(long, default_value = "4")]
        num_event_workers_per_shard: usize,

        /// Number of prefix blocks hashed to identify a branch. K=2 is the
        /// recommended default: depth=1 often produces too few distinct branch
        /// keys, while depth=2 gives a much larger set of distinguishable branches.
        #[clap(long, default_value = "2")]
        prefix_depth: usize,

        /// Number of OS threads per shard dedicated to find_matches (read isolation).
        /// 0 (default): reads run inline on the calling tokio thread.
        #[clap(long, default_value = "0")]
        num_read_threads_per_shard: usize,
    },
}

impl IndexerArgs {
    /// Construct the concrete indexer from the parsed CLI args.
    fn build(self, block_size: u32) -> Arc<dyn KvIndexerInterface + Send + Sync> {
        let cancel_token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        match self {
            IndexerArgs::RadixTree {} => {
                Arc::new(KvIndexer::new(cancel_token, block_size, metrics))
            }
            IndexerArgs::NestedMap {
                jump_size,
                num_event_workers,
            } => Arc::new(ThreadPoolIndexer::new(
                PositionalIndexer::new(jump_size),
                num_event_workers,
                block_size,
            )),
            IndexerArgs::ConcurrentRadixTree { num_event_workers } => Arc::new(
                ThreadPoolIndexer::new(ConcurrentRadixTree::new(), num_event_workers, block_size),
            ),
            IndexerArgs::ConcurrentRadixTreeCompressed { num_event_workers } => {
                Arc::new(ThreadPoolIndexer::new(
                    ConcurrentRadixTreeCompressed::new(),
                    num_event_workers,
                    block_size,
                ))
            }
            IndexerArgs::BranchShardedCrtc {
                num_shards,
                num_event_workers_per_shard,
                prefix_depth,
                num_read_threads_per_shard: _,
            } => {
                let shards = (0..num_shards)
                    .map(|_| {
                        ThreadPoolIndexer::new(
                            ConcurrentRadixTreeCompressed::new(),
                            num_event_workers_per_shard,
                            block_size,
                        )
                    })
                    .collect();
                Arc::new(BranchShardedIndexer::new_with_options(
                    shards,
                    prefix_depth,
                    block_size,
                ))
            }
        }
    }

    fn supports_remove(_name: &str) -> bool {
        true
    }

    fn is_multi_threaded(name: &str) -> bool {
        matches!(
            name,
            "nested-map"
                | "concurrent-radix-tree"
                | "concurrent-radix-tree-compressed"
                | "branch-sharded-crtc"
        )
    }

    /// Construct an indexer from a short name string.
    fn from_name(
        name: &str,
        block_size: u32,
        num_event_workers: usize,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let nw = num_event_workers;
        let indexer_args = match name {
            "radix-tree" => IndexerArgs::RadixTree {},
            "nested-map" => IndexerArgs::NestedMap {
                jump_size: 8,
                num_event_workers: nw,
            },
            "concurrent-radix-tree" => IndexerArgs::ConcurrentRadixTree {
                num_event_workers: nw,
            },
            "concurrent-radix-tree-compressed" => IndexerArgs::ConcurrentRadixTreeCompressed {
                num_event_workers: nw,
            },
            "branch-sharded-crtc" => IndexerArgs::BranchShardedCrtc {
                num_shards: 2,
                num_event_workers_per_shard: nw,
                prefix_depth: 2,
                num_read_threads_per_shard: 0,
            },
            _ => anyhow::bail!(
                "Unknown indexer '{}'. Valid names: radix-tree, radix-tree-sharded, \
                 nested-map, concurrent-radix-tree, concurrent-radix-tree-compressed, \
                 branch-sharded-crtc",
                name
            ),
        };
        Ok(indexer_args.build(block_size))
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Output path for the sweep plot SVG.
    #[clap(long, default_value = "sweep_plot.svg")]
    sweep_output: String,

    /// Comma-separated list of indexer names to benchmark and compare on the
    /// same plot. Overrides the subcommand indexer when present. Valid names:
    /// radix-tree, nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed.
    #[clap(long, value_delimiter = ',')]
    compare: Vec<String>,

    /// Number of OS threads for event processing in compare mode. Applies to
    /// indexers that use a thread pool (nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed).
    /// Ignored by radix-tree.
    #[clap(long, default_value = "16")]
    num_event_workers: usize,

    /// Number of additional concurrent tokio tasks that issue find_matches in a
    /// tight loop to stress the read path.  These tasks run alongside the normal
    /// trace-replay workers.  Set to 0 (default) to disable.
    #[clap(long, default_value = "0")]
    find_matches_concurrency: usize,

    /// Output path for the shard-size CSV produced when `shard-metrics` feature
    /// is enabled.  Rows: `elapsed_ms,shard_idx,worker_count,block_count,node_count`.
    /// An SVG plot is written alongside it (<path>.svg).
    /// Omit or leave empty to disable shard-size sampling.
    #[clap(long, default_value = "")]
    shard_metrics_csv: String,

    /// How often (ms) to sample shard sizes when `--shard-metrics-csv` is set.
    #[clap(long, default_value = "200")]
    shard_metrics_interval_ms: u64,

    /// Indexer backend to benchmark (defaults to radix-tree if not specified).
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,
}

impl Args {
    /// Return the indexer config, falling back to RadixTree if none was specified.
    fn get_indexer(&self) -> IndexerArgs {
        self.indexer.clone().unwrap_or(IndexerArgs::RadixTree {})
    }
}

/// A single entry in a worker's merged benchmark timeline.
#[derive(Clone)]
enum WorkerTraceEntry {
    /// A find_matches request with pre-computed block hashes.
    Request(Vec<LocalBlockHash>),
    /// A KV cache event (store/remove/clear) to apply to the indexer.
    Event(KvCacheEvent),
}

/// A timestamped entry in a worker's benchmark trace, used to replay requests
/// and events at the correct relative timing.
#[derive(Clone)]
struct WorkerTrace {
    entry: WorkerTraceEntry,
    timestamp_us: u64,
}

/// Merge each worker's request trace and event trace into a single
/// time-ordered sequence of `WorkerTrace` entries suitable for benchmark
/// replay.
///
/// Timestamps are rescaled from the original trace / simulation durations
/// into the benchmark duration (microseconds).
fn prepare_worker_traces(
    artifacts: Vec<WorkerReplayArtifacts>,
    benchmark_duration_ms: u64,
) -> Vec<Vec<WorkerTrace>> {
    artifacts
        .into_iter()
        .map(|artifact| {
            let mut merged = artifact
                .requests
                .into_iter()
                .map(|request| WorkerTrace {
                    timestamp_us: request.timestamp_us,
                    entry: WorkerTraceEntry::Request(request.replay_hashes.local_block_hashes),
                })
                .chain(artifact.kv_events.into_iter().map(|event| WorkerTrace {
                    timestamp_us: event.timestamp_us,
                    entry: WorkerTraceEntry::Event(event.event),
                }))
                .collect::<Vec<_>>();
            merged.sort_by_key(|entry| entry.timestamp_us);
            let max_timestamp_us = merged.last().map(|entry| entry.timestamp_us).unwrap_or(0);
            for entry in &mut merged {
                entry.timestamp_us = if max_timestamp_us == 0 {
                    0
                } else {
                    entry.timestamp_us * benchmark_duration_ms * 1000 / max_timestamp_us
                };
            }
            merged
        })
        .collect()
}

#[derive(Serialize)]
struct SweepStepResult {
    duration_ms: u64,
    #[serde(flatten)]
    results: BenchmarkResults,
}

// ---------------------------------------------------------------------------
// Shard-size sampling (always compiled; only called when a CSV path is given)
// ---------------------------------------------------------------------------

/// A single row in the shard-size time-series CSV.
#[derive(Clone)]
struct ShardSampleRow {
    elapsed_ms: u64,
    snapshot: ShardSizeSnapshot,
}

/// Spawn a background tokio task that samples `indexer.shard_sizes()` every
/// `interval_ms` milliseconds until `cancel` is triggered.
///
/// Returns a `JoinHandle` that resolves to all collected samples.
fn start_shard_sampler(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    interval_ms: u64,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<Vec<ShardSampleRow>> {
    tokio::spawn(async move {
        let mut rows = Vec::new();
        let start = Instant::now();
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let elapsed_ms = start.elapsed().as_millis() as u64;
                    for snap in indexer.shard_sizes() {
                        rows.push(ShardSampleRow { elapsed_ms, snapshot: snap });
                    }
                }
                _ = cancel.cancelled() => break,
            }
        }
        rows
    })
}

/// Write the collected shard-size samples to a CSV file.
fn write_shard_metrics_csv(rows: &[ShardSampleRow], path: &str) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "elapsed_ms,shard_idx,worker_count,block_count,node_count"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{}",
            r.elapsed_ms,
            r.snapshot.shard_idx,
            r.snapshot.worker_count,
            r.snapshot.block_count,
            r.snapshot.node_count,
        )?;
    }
    println!("Shard metrics CSV written to {path}");
    Ok(())
}

/// Plot per-shard `worker_count` and `block_count` over time and write an SVG.
///
/// Draws two panels stacked vertically:
/// - Top: workers per shard over time
/// - Bottom: blocks per shard over time
///
/// Each shard gets a distinct colour; shards are identified by their `shard_idx`.
fn plot_shard_metrics(rows: &[ShardSampleRow], svg_path: &str) -> anyhow::Result<()> {
    use plotters::prelude::*;

    if rows.is_empty() {
        return Ok(());
    }

    // Collect the set of shard indices present.
    let mut shard_indices: Vec<usize> = rows.iter().map(|r| r.snapshot.shard_idx).collect();
    shard_indices.sort_unstable();
    shard_indices.dedup();

    let max_elapsed = rows.iter().map(|r| r.elapsed_ms).max().unwrap_or(1);
    let max_workers = rows
        .iter()
        .map(|r| r.snapshot.worker_count)
        .max()
        .unwrap_or(1);
    let max_blocks = rows
        .iter()
        .map(|r| r.snapshot.block_count)
        .max()
        .unwrap_or(1);

    let colors: Vec<RGBColor> = vec![
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
    ];

    let root = SVGBackend::new(svg_path, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(350);

    // --- Top panel: workers per shard ---
    let mut chart = ChartBuilder::on(&upper)
        .caption("Workers per shard over time", ("sans-serif", 18))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0u64..max_elapsed, 0usize..max_workers + 1)?;
    chart
        .configure_mesh()
        .x_desc("Elapsed (ms)")
        .y_desc("Workers")
        .draw()?;

    for (i, &shard_idx) in shard_indices.iter().enumerate() {
        let color = colors[i % colors.len()];
        let points: Vec<(u64, usize)> = rows
            .iter()
            .filter(|r| r.snapshot.shard_idx == shard_idx)
            .map(|r| (r.elapsed_ms, r.snapshot.worker_count))
            .collect();
        let label = format!("shard {shard_idx}");
        chart
            .draw_series(LineSeries::new(points, &color))?
            .label(label)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    color.stroke_width(2),
                )
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // --- Bottom panel: blocks per shard ---
    let mut chart2 = ChartBuilder::on(&lower)
        .caption("Blocks per shard over time", ("sans-serif", 18))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0u64..max_elapsed, 0usize..max_blocks + 1)?;
    chart2
        .configure_mesh()
        .x_desc("Elapsed (ms)")
        .y_desc("Cached blocks")
        .draw()?;

    for (i, &shard_idx) in shard_indices.iter().enumerate() {
        let color = colors[i % colors.len()];
        let points: Vec<(u64, usize)> = rows
            .iter()
            .filter(|r| r.snapshot.shard_idx == shard_idx)
            .map(|r| (r.elapsed_ms, r.snapshot.block_count))
            .collect();
        let label = format!("shard {shard_idx}");
        chart2
            .draw_series(LineSeries::new(points, &color))?
            .label(label)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    color.stroke_width(2),
                )
            });
    }
    chart2
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Shard metrics plot written to {svg_path}");
    Ok(())
}

/// Run the benchmark: replay each worker's merged trace against the indexer,
/// measuring find_matches latency and event processing throughput.
///
/// Workers are spawned as tokio tasks, each replaying its trace at the
/// original inter-entry timing. After all workers finish, the event queue is
/// flushed and latency percentiles / throughput stats are printed.
async fn run_benchmark(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    artifacts: Vec<WorkerReplayArtifacts>,
    args: &Args,
    benchmark_duration_ms: u64,
    count_events: bool,
    find_matches_concurrency: usize,
) -> anyhow::Result<BenchmarkResults> {
    let worker_traces = prepare_worker_traces(artifacts, benchmark_duration_ms);
    let worker_traces = worker_traces.into_iter().map(Arc::new).collect::<Vec<_>>();

    let progress = make_progress_bar(Some(
        worker_traces
            .iter()
            .map(|trace| trace.len() as u64)
            .sum::<u64>()
            * args.common.inference_worker_duplication_factor as u64,
    ));

    let mut tasks = Vec::new();
    for replica in 0..args.common.inference_worker_duplication_factor {
        for (worker_id, worker_trace) in worker_traces.iter().enumerate() {
            let indexer = indexer.clone();
            let trace = worker_trace.clone();
            let progress = progress.clone();
            let worker_id = worker_id + replica * worker_traces.len();
            tasks.push(tokio::spawn(async move {
                let mut request_latencies = Vec::with_capacity(trace.len());

                let submit = |entry: WorkerTrace| async {
                    match entry.entry {
                        WorkerTraceEntry::Request(request) => {
                            let start = minstant::Instant::now();
                            indexer.find_matches(request).await?;
                            Ok::<Option<u64>, anyhow::Error>(
                                Some(start.elapsed().as_nanos() as u64),
                            )
                        }
                        WorkerTraceEntry::Event(event) => {
                            indexer
                                .apply_event(RouterEvent::new(worker_id as u64, event))
                                .await;
                            Ok(None)
                        }
                    }
                };

                let mut target = Instant::now();

                let mut trace = trace.iter().peekable();

                let mut local_count = 0;

                while let Some(entry) = trace.next() {
                    let mut processed = 1;
                    let entry_timestamp_us = entry.timestamp_us;

                    if let Some(latency) = submit(entry.clone()).await? {
                        request_latencies.push(latency);
                    }

                    while let Some(next) = trace.peek() {
                        if next.timestamp_us == entry_timestamp_us {
                            if let Some(latency) = submit(trace.next().unwrap().clone()).await? {
                                request_latencies.push(latency);
                            }
                            processed += 1;
                        } else {
                            break;
                        }
                    }

                    if let Some(next) = trace.peek() {
                        target += Duration::from_micros(next.timestamp_us - entry_timestamp_us);
                    }

                    if target > Instant::now() {
                        tokio::time::sleep_until(target).await;
                    }

                    local_count += processed;

                    if local_count > 100 {
                        progress.inc(local_count);
                        local_count = 0;
                    }
                }

                progress.inc(local_count);

                Ok::<_, anyhow::Error>(request_latencies)
            }));
        }
    }

    // Spawn additional concurrent find_matches callers if requested.
    // These tasks run alongside the trace-replay workers to stress the read path.
    let fm_stop = Arc::new(AtomicBool::new(false));
    let mut fm_tasks = Vec::new();
    if find_matches_concurrency > 0 {
        // Collect all request sequences as a pool for random selection.
        let seq_pool: Arc<Vec<Vec<LocalBlockHash>>> = Arc::new(
            worker_traces
                .iter()
                .flat_map(|t| t.iter())
                .filter_map(|entry| match &entry.entry {
                    WorkerTraceEntry::Request(hashes) => Some(hashes.clone()),
                    _ => None,
                })
                .collect(),
        );

        if !seq_pool.is_empty() {
            for task_id in 0..find_matches_concurrency {
                let indexer = indexer.clone();
                let pool = Arc::clone(&seq_pool);
                let stop = Arc::clone(&fm_stop);
                fm_tasks.push(tokio::spawn(async move {
                    let mut latencies = Vec::new();
                    let mut idx = task_id % pool.len();
                    while !stop.load(Ordering::Relaxed) {
                        let seq = pool[idx].clone();
                        let start = minstant::Instant::now();
                        let _ = indexer.find_matches(seq).await;
                        latencies.push(start.elapsed().as_nanos() as u64);
                        idx = (idx + 1) % pool.len();
                    }
                    latencies
                }));
            }
        }
    }

    let mut latencies = Vec::new();

    for task in tasks {
        latencies.extend(task.await??);
    }

    // Signal concurrent find_matches callers to stop and collect their latencies.
    fm_stop.store(true, Ordering::Relaxed);
    for task in fm_tasks {
        if let Ok(fm_latencies) = task.await {
            latencies.extend(fm_latencies);
        }
    }

    if progress.elapsed() > Duration::from_millis(benchmark_duration_ms * 11 / 10) {
        eprintln!(
            "WARNING: The benchmarker is unable to keep up with the request/event generation rate. Rerun with a larger --benchmark-duration-ms."
        )
    }

    let total_duration = progress.elapsed();

    let total_events = worker_traces
        .iter()
        .map(|trace| {
            trace
                .iter()
                .filter(|trace| matches!(trace.entry, WorkerTraceEntry::Event(_)))
                .count()
        })
        .sum::<usize>()
        * args.common.inference_worker_duplication_factor;

    let total_requests = worker_traces.iter().map(|trace| trace.len()).sum::<usize>()
        * args.common.inference_worker_duplication_factor
        - total_events;

    let total_request_blocks: usize = worker_traces
        .iter()
        .flat_map(|t| t.iter())
        .filter_map(|entry| match &entry.entry {
            WorkerTraceEntry::Request(hashes) => Some(hashes.len()),
            _ => None,
        })
        .sum::<usize>()
        * args.common.inference_worker_duplication_factor;

    let total_event_blocks: usize = worker_traces
        .iter()
        .flat_map(|t| t.iter())
        .filter_map(|entry| match &entry.entry {
            WorkerTraceEntry::Event(ev) => match &ev.data {
                KvCacheEventData::Stored(s) => Some(s.blocks.len()),
                _ => Some(0),
            },
            _ => None,
        })
        .sum::<usize>()
        * args.common.inference_worker_duplication_factor;

    let counted_events = if count_events { total_events } else { 0 };
    let counted_event_blocks = if count_events { total_event_blocks } else { 0 };

    let total_blocks = total_request_blocks + counted_event_blocks;
    let total_ops = total_requests + counted_events;
    let offered_ops_throughput = total_ops as f32 / benchmark_duration_ms as f32 * 1000.0;
    let ops_throughput = total_ops as f32 / total_duration.as_millis() as f32 * 1000.0;
    let offered_block_throughput = total_blocks as f32 / benchmark_duration_ms as f32 * 1000.0;
    let block_throughput = total_blocks as f32 / total_duration.as_millis() as f32 * 1000.0;

    latencies.sort_unstable();
    let latency_p99_us = if latencies.is_empty() {
        0.0
    } else {
        latencies[latencies.len() * 99 / 100] as f32 / 1000.0
    };

    println!(
        "Offered Ops Throughput: {} ops/s | Achieved: {} ops/s (requests + events)",
        offered_ops_throughput as u64, ops_throughput as u64,
    );
    println!(
        "Offered Block Throughput: {} block ops/s | Achieved: {} block ops/s",
        offered_block_throughput as u64, block_throughput as u64,
    );
    println!("Latency p99: {}us", latency_p99_us);

    Ok(BenchmarkResults {
        offered_ops_throughput,
        ops_throughput,
        offered_block_throughput,
        block_throughput,
        latency_p99_us,
    })
}

async fn run_tests() -> anyhow::Result<()> {
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::Write;

    let path =
        std::env::temp_dir().join(format!("mooncake_bench_test_{}.jsonl", std::process::id()));
    {
        let mut f = File::create(&path)?;
        for (i, (hash_ids, output_length)) in
            [(&[0u64, 1, 2] as &[u64], 10u64), (&[0, 1, 3, 4], 10)]
                .iter()
                .enumerate()
        {
            writeln!(
                f,
                "{}",
                serde_json::json!({
                    "timestamp": i as u64,
                    "input_length": hash_ids.len(),
                    "hash_ids": hash_ids,
                    "output_length": output_length,
                })
            )?;
        }
    }

    let traces = process_mooncake_trace(path.to_str().unwrap(), 512, 2, 2, 2, 42)?;
    std::fs::remove_file(&path).ok();

    let mut all_hashes: Vec<Vec<u64>> = traces
        .into_iter()
        .flat_map(|worker| worker.sessions.into_iter())
        .flat_map(|session| session.turns.into_iter().map(|turn| turn.hash_ids))
        .collect();
    all_hashes.sort();

    // expand(2): [0,1,2] → [0,1,2,3,4,5], [0,1,3,4] → [0,1,2,3,6,7,8,9]
    // duplicate(2): max=9, offset=10
    let mut expected = vec![
        vec![0, 1, 2, 3, 4, 5],
        vec![10, 11, 12, 13, 14, 15],
        vec![0, 1, 2, 3, 6, 7, 8, 9],
        vec![10, 11, 12, 13, 16, 17, 18, 19],
    ];
    expected.sort();
    assert_eq!(all_hashes, expected, "hash_ids mismatch");

    // Verify prefix structure within each copy.
    let copy0: Vec<&Vec<u64>> = all_hashes.iter().filter(|h| h[0] == 0).collect();
    let copy1: Vec<&Vec<u64>> = all_hashes.iter().filter(|h| h[0] == 10).collect();
    assert_eq!(copy0.len(), 2);
    assert_eq!(copy1.len(), 2);
    assert_eq!(copy0[0][..4], copy0[1][..4], "copy 0 shared prefix broken");
    assert_eq!(copy1[0][..4], copy1[1][..4], "copy 1 shared prefix broken");

    // Verify disjointness between copies.
    let set0: HashSet<u64> = copy0.iter().flat_map(|h| h.iter().copied()).collect();
    let set1: HashSet<u64> = copy1.iter().flat_map(|h| h.iter().copied()).collect();
    assert!(set0.is_disjoint(&set1), "copies are not hash-disjoint");

    let replay_trace = Trace {
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
    };
    let artifacts = generate_replay_artifacts(&[replay_trace], 1024, 2, 5).await?;
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

    println!("All tests passed.");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.common.test {
        return run_tests().await;
    }

    let path = match args.common.mooncake_trace_path.as_deref() {
        Some(p) => p,
        None => {
            eprintln!("No mooncake_trace_path provided, skipping benchmark");
            return Ok(());
        }
    };
    let traces = process_mooncake_trace(
        path,
        args.common.block_size,
        args.common.trace_length_factor,
        args.common.trace_duplication_factor,
        args.common.num_unique_inference_workers,
        args.common.seed,
    )?;
    let artifacts = generate_replay_artifacts(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;

    let indexer_names: Vec<String> = if args.compare.is_empty() {
        let name = match args.get_indexer() {
            IndexerArgs::RadixTree {} => "radix-tree",
            IndexerArgs::NestedMap { .. } => "nested-map",
            IndexerArgs::ConcurrentRadixTree { .. } => "concurrent-radix-tree",
            IndexerArgs::ConcurrentRadixTreeCompressed { .. } => "concurrent-radix-tree-compressed",
            IndexerArgs::BranchShardedCrtc { .. } => "branch-sharded-crtc",
        };
        vec![name.to_string()]
    } else {
        args.compare.clone()
    };

    if args.common.sweep {
        let durations_low_to_high = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        );
        let durations_high_to_low: Vec<u64> = durations_low_to_high.iter().copied().rev().collect();

        let mut all_results: Vec<(&str, Vec<(u64, BenchmarkResults)>)> = Vec::new();

        for name in &indexer_names {
            println!("\n{}", "=".repeat(60));
            println!("Benchmarking indexer: {}", name);
            println!("{}", "=".repeat(60));

            let multi_threaded = IndexerArgs::is_multi_threaded(name);
            let durations = if multi_threaded {
                &durations_high_to_low
            } else {
                &durations_low_to_high
            };

            let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();
            let mut consecutive_keeping_up = 0u32;

            for &dur_ms in durations {
                println!("\n=== Sweep: benchmark_duration_ms = {} ===", dur_ms);
                let indexer = if args.compare.is_empty() {
                    args.get_indexer().build(args.common.block_size)
                } else {
                    IndexerArgs::from_name(name, args.common.block_size, args.num_event_workers)?
                };
                let count_events = IndexerArgs::supports_remove(name);
                let result =
                    run_benchmark(indexer, artifacts.clone(), &args, dur_ms, count_events, args.find_matches_concurrency).await?;

                if multi_threaded {
                    if result.block_throughput >= result.offered_block_throughput * 0.95 {
                        consecutive_keeping_up += 1;
                    } else {
                        consecutive_keeping_up = 0;
                    }
                    results.push((dur_ms, result));
                    if consecutive_keeping_up >= 5 {
                        println!("Early stop: achieved >= 95% offered for 5 consecutive steps");
                        break;
                    }
                } else {
                    let saturated = result.offered_block_throughput > result.block_throughput * 5.0;
                    results.push((dur_ms, result));
                    if saturated {
                        println!("Early stop: offered throughput >5x achieved throughput");
                        break;
                    }
                }
            }

            results.sort_by_key(|(dur, _)| std::cmp::Reverse(*dur));
            print_sweep_summary(name, &results);

            all_results.push((name, results));
        }

        plot_sweep(&all_results, &args.sweep_output)?;

        let json_path = args
            .sweep_output
            .replace(".png", ".json")
            .replace(".svg", ".json");
        let json_map: std::collections::BTreeMap<&str, Vec<SweepStepResult>> = all_results
            .iter()
            .map(|(name, results)| {
                let steps = results
                    .iter()
                    .map(|(dur, r)| SweepStepResult {
                        duration_ms: *dur,
                        results: BenchmarkResults {
                            offered_ops_throughput: r.offered_ops_throughput,
                            ops_throughput: r.ops_throughput,
                            offered_block_throughput: r.offered_block_throughput,
                            block_throughput: r.block_throughput,
                            latency_p99_us: r.latency_p99_us,
                        },
                    })
                    .collect();
                (*name, steps)
            })
            .collect();
        std::fs::write(&json_path, serde_json::to_string_pretty(&json_map)?)?;
        println!("Sweep results saved to {}", json_path);
    } else {
        drop(traces);

        for name in &indexer_names {
            println!("\nBenchmarking indexer: {}", name);
            let indexer = if args.compare.is_empty() {
                args.get_indexer().build(args.common.block_size)
            } else {
                IndexerArgs::from_name(name, args.common.block_size, args.num_event_workers)?
            };
            let count_events = IndexerArgs::supports_remove(name);

            // Start shard-size sampler if a CSV path was provided.
            let shard_cancel = CancellationToken::new();
            let shard_sampler = if !args.shard_metrics_csv.is_empty() {
                Some(start_shard_sampler(
                    indexer.clone(),
                    args.shard_metrics_interval_ms,
                    shard_cancel.clone(),
                ))
            } else {
                None
            };

            run_benchmark(
                indexer.clone(),
                artifacts.clone(),
                &args,
                args.common.benchmark_duration_ms,
                count_events,
                args.find_matches_concurrency,
            )
            .await?;

            // Stop sampler and write CSV + plot.
            shard_cancel.cancel();
            if let Some(handle) = shard_sampler {
                let rows = handle.await?;
                // In compare mode, prefix the indexer name to distinguish outputs.
                let csv_path = if args.compare.len() > 1 {
                    let stem = args.shard_metrics_csv.trim_end_matches(".csv");
                    format!("{stem}_{name}.csv")
                } else {
                    args.shard_metrics_csv.clone()
                };
                write_shard_metrics_csv(&rows, &csv_path)?;
                let svg = format!("{}.svg", csv_path.trim_end_matches(".csv"));
                plot_shard_metrics(&rows, &svg)?;
            }

            let report = indexer.timing_report();
            if !report.is_empty() {
                println!("{}", report);
            }
            let sizes = indexer.shard_sizes();
            if sizes.len() > 1 {
                let total_blocks: usize = sizes.iter().map(|s| s.block_count).sum();
                let total_nodes: usize = sizes.iter().map(|s| s.node_count).sum();
                println!("Shard block distribution:");
                for s in &sizes {
                    let pct = if total_blocks > 0 {
                        100.0 * s.block_count as f64 / total_blocks as f64
                    } else {
                        0.0
                    };
                    println!(
                        "  shard {}: {} blocks ({:.1}%), {} workers, {} nodes",
                        s.shard_idx, s.block_count, pct, s.worker_count, s.node_count
                    );
                }
                if total_nodes > 0 {
                    println!("  total nodes across shards: {}", total_nodes);
                }
            }

            let mut edge_lengths = indexer.node_edge_lengths();
            if !edge_lengths.is_empty() {
                let avg = edge_lengths.iter().sum::<usize>() as f64 / edge_lengths.len() as f64;
                edge_lengths.sort_unstable();
                let p99 = edge_lengths[edge_lengths.len() * 99 / 100];
                println!(
                    "Node edge lengths ({} nodes): avg={:.1} hashes/node, p99={} hashes/node",
                    edge_lengths.len(),
                    avg,
                    p99,
                );
            }
        }
    }

    Ok(())
}
