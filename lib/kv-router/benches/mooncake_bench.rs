// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/mod.rs"]
mod common;
use common::*;

use clap::{Parser, Subcommand};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded,
};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent};
use dynamo_kv_router::{
    ConcurrentRadixTree, InvertedIndex, NaiveNestedMap, PositionalIndexer, ThreadPoolIndexer,
};
use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Indexer backend selection and its backend-specific parameters.
#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    /// Single-threaded radix tree indexer.
    RadixTree {},

    /// Sharded radix tree indexer that partitions workers across independent shards.
    RadixTreeSharded {
        /// Number of independent shards to split workers across.
        #[clap(long, default_value = "4")]
        num_shards: usize,
    },

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

    /// Naive per-worker nested HashMap indexer behind a single-threaded actor
    /// (blog section 2).
    NaiveNestedMap {},

    /// Inverted index keyed by local_hash (blog section 3).
    InvertedIndex {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },
}

impl IndexerArgs {
    /// Construct the concrete indexer from the parsed CLI args.
    fn build(self, args: &Args) -> Arc<dyn KvIndexerInterface + Send + Sync> {
        let cancel_token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        match self {
            IndexerArgs::RadixTree {} => {
                Arc::new(KvIndexer::new(cancel_token, args.block_size, metrics))
            }
            IndexerArgs::RadixTreeSharded { num_shards } => Arc::new(KvIndexerSharded::new(
                cancel_token,
                num_shards,
                args.block_size,
                metrics,
            )),
            IndexerArgs::NestedMap {
                jump_size,
                num_event_workers,
            } => Arc::new(ThreadPoolIndexer::new(
                PositionalIndexer::new(jump_size),
                num_event_workers,
                args.block_size,
            )),
            IndexerArgs::ConcurrentRadixTree { num_event_workers } => {
                Arc::new(ThreadPoolIndexer::new(
                    ConcurrentRadixTree::new(),
                    num_event_workers,
                    args.block_size,
                ))
            }
            IndexerArgs::NaiveNestedMap {} => Arc::new(NaiveNestedMap::new()),
            IndexerArgs::InvertedIndex { .. } => Arc::new(InvertedIndex::new()),
        }
    }

    /// Construct an indexer from a short name string, using `args.num_event_workers`.
    fn from_name(
        name: &str,
        args: &Args,
    ) -> anyhow::Result<Arc<dyn KvIndexerInterface + Send + Sync>> {
        let nw = args.num_event_workers;
        let indexer_args = match name {
            "radix-tree" => IndexerArgs::RadixTree {},
            "radix-tree-sharded" => IndexerArgs::RadixTreeSharded { num_shards: 4 },
            "nested-map" => IndexerArgs::NestedMap {
                jump_size: 8,
                num_event_workers: nw,
            },
            "concurrent-radix-tree" => IndexerArgs::ConcurrentRadixTree {
                num_event_workers: nw,
            },
            "naive-nested-map" => IndexerArgs::NaiveNestedMap {},
            "inverted-index" => IndexerArgs::InvertedIndex {
                num_event_workers: 0,
            },
            _ => anyhow::bail!(
                "Unknown indexer '{}'. Valid names: radix-tree, radix-tree-sharded, \
                 nested-map, concurrent-radix-tree, naive-nested-map, inverted-index",
                name
            ),
        };
        Ok(indexer_args.build(args))
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    /// Path to a JSONL mooncake trace file. Each line is a JSON object with
    /// fields: uuid, timestamp, hash_ids, output_length.
    /// Required unless --test is passed.
    mooncake_trace_path: Option<String>,

    /// Run built-in self-tests instead of the benchmark.
    #[clap(long)]
    test: bool,

    /// Number of GPU blocks available in the mock engine's KV cache.
    /// Smaller values force more evictions and produce more remove events.
    #[clap(long, default_value = "1048576")]
    num_gpu_blocks: usize,

    /// Number of tokens per KV cache block.
    #[clap(long, default_value = "512")]
    block_size: u32,

    /// Wall-clock duration (ms) over which the trace is replayed during event
    /// generation. Longer values produce more accurate inter-request timing but
    /// increase setup time.
    #[clap(long, default_value = "30000")]
    trace_simulation_duration_ms: u64,

    /// Wall-clock duration (ms) over which the benchmark replays requests and
    /// events against the indexer under test.
    #[clap(long, default_value = "60000")]
    benchmark_duration_ms: u64,

    /// Number of unique simulated inference workers. Each gets a random
    /// partition of the trace and its own mock engine for event generation.
    #[clap(short, long, default_value = "256")]
    num_unique_inference_workers: usize,

    /// How many times to duplicate the set of unique workers during the
    /// benchmark phase. Total workers = num_unique_inference_workers * factor.
    /// Duplicated workers replay identical traces with distinct worker IDs.
    #[clap(short = 'd', long, default_value = "1")]
    inference_worker_duplication_factor: usize,

    /// Factor by which to stretch each request's hash sequence length.
    /// Each original hash block becomes `factor` consecutive blocks.
    /// Applied before event generation and before trace duplication.
    #[clap(long, default_value = "1")]
    trace_length_factor: usize,

    /// How many times to duplicate the raw trace data with offset hash_ids
    /// before event generation. Each copy is a structurally identical prefix
    /// tree with disjoint hash values, increasing the number of unique
    /// prefix groups and workers.
    #[clap(long, default_value = "1")]
    trace_duplication_factor: usize,

    /// RNG seed for reproducible worker-to-trace assignment.
    #[clap(long, default_value = "42")]
    seed: u64,

    /// Enable throughput vs p99 latency sweep mode. Runs the benchmark at
    /// multiple benchmark_duration_ms values and plots the results.
    #[clap(long)]
    sweep: bool,

    /// Minimum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "1000")]
    sweep_min_ms: u64,

    /// Maximum benchmark duration (ms) for sweep mode.
    #[clap(long, default_value = "50000")]
    sweep_max_ms: u64,

    /// Number of logarithmically spaced sweep steps between min and max.
    #[clap(long, default_value = "10")]
    sweep_steps: usize,

    /// Output path for the sweep plot PNG.
    #[clap(long, default_value = "sweep_plot.svg")]
    sweep_output: String,

    /// Comma-separated list of indexer names to benchmark and compare on the
    /// same plot. Overrides the subcommand indexer when present. Valid names:
    /// radix-tree, radix-tree-sharded, nested-map, concurrent-radix-tree,
    /// naive-nested-map, inverted-index.
    #[clap(long, value_delimiter = ',')]
    compare: Vec<String>,

    /// Number of OS threads for event processing in compare mode. Applies to
    /// indexers that use a thread pool (nested-map, concurrent-radix-tree,
    /// inverted-index). Ignored by radix-tree, radix-tree-sharded, and
    /// naive-nested-map.
    #[clap(long, default_value = "16")]
    num_event_workers: usize,

    /// Indexer backend to benchmark (defaults to radix-tree if not specified).
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,

    /// Ignored - passed by cargo bench harness.
    #[arg(long, hide = true, global = true)]
    bench: bool,
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
    traces: Vec<Vec<MooncakeRequest>>,
    events: Vec<Vec<(KvCacheEvent, Instant)>>,
    block_size: u32,
    benchmark_duration_ms: u64,
    trace_simulation_duration_ms: u64,
) -> Vec<Vec<WorkerTrace>> {
    assert!(traces.len() == events.len());

    let scaled_request_traces: Vec<_> = traces
        .into_iter()
        .map(|trace| {
            let trace_duration_ms =
                trace.last().unwrap().timestamp - trace.first().unwrap().timestamp;
            trace
                .into_iter()
                .map(|request| WorkerTrace {
                    timestamp_us: request.timestamp * 1000 * benchmark_duration_ms
                        / trace_duration_ms,
                    entry: WorkerTraceEntry::Request(
                        request
                            .hash_ids
                            .iter()
                            .map(|id| local_block_hash_from_id(*id, block_size))
                            .collect(),
                    ),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let scaled_event_traces: Vec<_> = events
        .into_iter()
        .map(|worker_events| {
            let start_instant = worker_events.first().unwrap().1;
            worker_events
                .into_iter()
                .map(|(event, timestamp)| WorkerTrace {
                    timestamp_us: (timestamp - start_instant).as_micros() as u64
                        * benchmark_duration_ms
                        / trace_simulation_duration_ms,
                    entry: WorkerTraceEntry::Event(event),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    scaled_request_traces
        .into_iter()
        .zip(scaled_event_traces)
        .map(|(request_trace, event_trace)| {
            let mut merged: Vec<WorkerTrace> =
                request_trace.into_iter().chain(event_trace).collect();
            merged.sort_by_key(|entry| entry.timestamp_us);
            merged
        })
        .collect()
}

/// Run the benchmark: replay each worker's merged trace against the indexer,
/// measuring find_matches latency and event processing throughput.
///
/// Workers are spawned as tokio tasks, each replaying its trace at the
/// original inter-entry timing. After all workers finish, the event queue is
/// flushed and latency percentiles / throughput stats are printed.
async fn run_benchmark(
    indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    traces: Vec<Vec<MooncakeRequest>>,
    events: Vec<Vec<(KvCacheEvent, Instant)>>,
    args: &Args,
    benchmark_duration_ms: u64,
) -> anyhow::Result<BenchmarkResults> {
    let worker_traces = prepare_worker_traces(
        traces,
        events,
        args.block_size,
        benchmark_duration_ms,
        args.trace_simulation_duration_ms,
    );
    let worker_traces = worker_traces.into_iter().map(Arc::new).collect::<Vec<_>>();

    let progress = make_progress_bar(Some(
        worker_traces
            .iter()
            .map(|trace| trace.len() as u64)
            .sum::<u64>()
            * args.inference_worker_duplication_factor as u64,
    ));

    let mut tasks = Vec::new();
    for replica in 0..args.inference_worker_duplication_factor {
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
                                .apply_event(RouterEvent {
                                    worker_id: worker_id as u64,
                                    event,
                                })
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

    let mut latencies = Vec::new();

    for task in tasks {
        latencies.extend(task.await??);
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
        * args.inference_worker_duplication_factor;

    let total_requests = worker_traces.iter().map(|trace| trace.len()).sum::<usize>()
        * args.inference_worker_duplication_factor
        - total_events;

    let total_request_blocks: usize = worker_traces
        .iter()
        .flat_map(|t| t.iter())
        .filter_map(|entry| match &entry.entry {
            WorkerTraceEntry::Request(hashes) => Some(hashes.len()),
            _ => None,
        })
        .sum::<usize>()
        * args.inference_worker_duplication_factor;

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
        * args.inference_worker_duplication_factor;

    let total_blocks = total_request_blocks + total_event_blocks;

    let total_ops = total_requests + total_events;
    let offered_ops_throughput = total_ops as f32 / benchmark_duration_ms as f32 * 1000.0;
    let ops_throughput = total_ops as f32 / total_duration.as_millis() as f32 * 1000.0;
    let offered_block_throughput = total_blocks as f32 / benchmark_duration_ms as f32 * 1000.0;
    let block_throughput = total_blocks as f32 / total_duration.as_millis() as f32 * 1000.0;

    latencies.sort_unstable();
    let latency_p99_us = latencies[latencies.len() * 99 / 100] as f32 / 1000.0;

    println!(
        "Ops Throughput: {} ops/s (requests + events)",
        ops_throughput
    );
    println!("Block Throughput: {} block ops/s", block_throughput);
    println!("Latency p99: {}us", latency_p99_us);

    Ok(BenchmarkResults {
        offered_ops_throughput,
        ops_throughput,
        offered_block_throughput,
        block_throughput,
        latency_p99_us,
    })
}

fn run_tests() -> anyhow::Result<()> {
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
                    "hash_ids": hash_ids,
                    "output_length": output_length,
                })
            )?;
        }
    }

    let traces = process_mooncake_trace(path.to_str().unwrap(), 2, 2, 2, 42)?;
    std::fs::remove_file(&path).ok();

    let mut all_hashes: Vec<Vec<u64>> = traces
        .into_iter()
        .flat_map(|w| w.into_iter().map(|r| r.hash_ids))
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

    println!("All tests passed.");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.test {
        return run_tests();
    }

    let path = args
        .mooncake_trace_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("mooncake_trace_path is required for benchmarking"))?;
    let traces = process_mooncake_trace(
        path,
        args.trace_length_factor,
        args.trace_duplication_factor,
        args.num_unique_inference_workers,
        args.seed,
    )?;
    let events = generate_kv_events(
        &traces,
        args.num_gpu_blocks,
        args.block_size,
        args.trace_simulation_duration_ms,
    )
    .await?;

    let indexer_names: Vec<String> = if args.compare.is_empty() {
        let name = match args.get_indexer() {
            IndexerArgs::RadixTree {} => "radix-tree",
            IndexerArgs::RadixTreeSharded { .. } => "radix-tree-sharded",
            IndexerArgs::NestedMap { .. } => "nested-map",
            IndexerArgs::ConcurrentRadixTree { .. } => "concurrent-radix-tree",
            IndexerArgs::NaiveNestedMap {} => "naive-nested-map",
            IndexerArgs::InvertedIndex { .. } => "inverted-index",
        };
        vec![name.to_string()]
    } else {
        args.compare.clone()
    };

    if args.sweep {
        let log_min = (args.sweep_min_ms as f64).ln();
        let log_max = (args.sweep_max_ms as f64).ln();
        let n = args.sweep_steps;
        let durations: Vec<u64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (log_max * (1.0 - t) + log_min * t).exp().round() as u64
            })
            .collect();

        let mut all_results: Vec<(&str, Vec<(u64, BenchmarkResults)>)> = Vec::new();

        for name in &indexer_names {
            println!("\n{}", "=".repeat(60));
            println!("Benchmarking indexer: {}", name);
            println!("{}", "=".repeat(60));

            let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();

            for &dur_ms in &durations {
                println!("\n=== Sweep: benchmark_duration_ms = {} ===", dur_ms);
                let indexer = if args.compare.is_empty() {
                    args.get_indexer().build(&args)
                } else {
                    IndexerArgs::from_name(name, &args)?
                };
                let result =
                    run_benchmark(indexer, traces.clone(), events.clone(), &args, dur_ms).await?;
                results.push((dur_ms, result));
            }

            println!("\n=== Sweep Summary: {} ===", name);
            println!(
                "{:>12} {:>14} {:>14} {:>14} {:>14} {:>10}",
                "duration_ms", "ops/s_off", "ops/s", "blk_ops/s_off", "blk_ops/s", "p99(us)"
            );
            for (dur, r) in &results {
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

            all_results.push((name, results));
        }

        plot_sweep(&all_results, &args.sweep_output)?;
    } else {
        for name in &indexer_names {
            println!("\nBenchmarking indexer: {}", name);
            let indexer = if args.compare.is_empty() {
                args.get_indexer().build(&args)
            } else {
                IndexerArgs::from_name(name, &args)?
            };
            run_benchmark(
                indexer,
                traces.clone(),
                events.clone(),
                &args,
                args.benchmark_duration_ms,
            )
            .await?;
        }
    }

    Ok(())
}
