// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded,
};
use dynamo_kv_router::protocols::RouterEvent;
use dynamo_kv_router::protocols::XXH3_SEED;
use dynamo_kv_router::{
    ConcurrentRadixTree, InvertedIndex, NaiveNestedMap, PositionalIndexer, ThreadPoolIndexer,
};
use dynamo_tokens::compute_hash_v2;
use rand::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use uuid::Uuid;

use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use dynamo_mocker::Scheduler;
use dynamo_mocker::protocols::{DirectRequest, KvCacheEventSink, MockEngineArgs};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use serde::{Deserialize, Serialize};

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

    /// Naive per-worker nested HashMap indexer (blog section 2).
    NaiveNestedMap {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "1")]
        num_event_workers: usize,
    },

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
            IndexerArgs::NaiveNestedMap { num_event_workers } => Arc::new(ThreadPoolIndexer::new(
                NaiveNestedMap::new(),
                num_event_workers,
                args.block_size,
            )),
            IndexerArgs::InvertedIndex { num_event_workers } => Arc::new(ThreadPoolIndexer::new(
                InvertedIndex::new(),
                num_event_workers,
                args.block_size,
            )),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    /// Path to a JSONL mooncake trace file. Each line is a JSON object with
    /// fields: uuid, timestamp, hash_ids, output_length.
    mooncake_trace_path: String,

    /// Number of GPU blocks available in the mock engine's KV cache.
    /// Smaller values force more evictions and produce more remove events.
    #[clap(long, default_value = "2048")]
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
    #[clap(short, long, default_value = "64")]
    num_unique_inference_workers: usize,

    /// How many times to duplicate the set of unique workers during the
    /// benchmark phase. Total workers = num_unique_inference_workers * factor.
    /// Duplicated workers replay identical traces with distinct worker IDs.
    #[clap(short = 'd', long, default_value = "1")]
    inference_worker_duplication_factor: usize,

    /// RNG seed for reproducible worker-to-trace assignment.
    #[clap(long, default_value = "42")]
    seed: u64,

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

/// A single request deserialized from the mooncake trace JSONL.
#[derive(Serialize, Deserialize, Clone)]
struct MooncakeRequest {
    #[serde(default = "Uuid::new_v4")]
    uuid: uuid::Uuid,
    timestamp: u64,
    hash_ids: Vec<u64>,
    output_length: u64,
}

/// Collects KV cache events emitted by the mock engine during event generation,
/// tagging each with the wall-clock instant it was produced.
struct EventCollector {
    events: Mutex<Option<Vec<(KvCacheEvent, Instant)>>>,
}

impl EventCollector {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Some(Vec::new())),
        })
    }

    /// Take ownership of the collected events. Can only be called once.
    fn get_events(self: Arc<Self>) -> Vec<(KvCacheEvent, Instant)> {
        self.events.lock().unwrap().take().unwrap()
    }
}

impl KvCacheEventSink for EventCollector {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        let timestamp = Instant::now();
        if let Some(events) = self.events.lock().unwrap().as_mut() {
            events.push((event, timestamp));
        }
        Ok(())
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

/// Load the mooncake trace from disk and randomly partition requests across
/// `num_unique_inference_workers` worker buckets using the configured seed.
fn process_mooncake_trace(args: &Args) -> anyhow::Result<Vec<Vec<MooncakeRequest>>> {
    let mut traces: Vec<Vec<MooncakeRequest>> = Vec::new();
    for _ in 0..args.num_unique_inference_workers {
        traces.push(Vec::new());
    }

    let mut rng = StdRng::seed_from_u64(args.seed);

    let file = File::open(&args.mooncake_trace_path)?;
    let reader = BufReader::new(file);

    println!("Loading trace...");

    let progress = make_progress_bar(None);

    for line in reader.lines() {
        let request = serde_json::from_str::<MooncakeRequest>(&line?)?;
        traces[rng.random_range(0..args.num_unique_inference_workers)].push(request);
        progress.inc(1);
    }

    Ok(traces)
}

/// Linearly rescale all timestamps in a worker's trace so the total span equals
/// `duration` milliseconds.
fn scale_mooncake_trace(trace: &Vec<MooncakeRequest>, duration: u64) -> Vec<MooncakeRequest> {
    let total_duration = trace.last().unwrap().timestamp - trace.first().unwrap().timestamp;
    trace
        .iter()
        .map(|request| MooncakeRequest {
            timestamp: request.timestamp * duration / total_duration,
            ..request.clone()
        })
        .collect::<Vec<MooncakeRequest>>()
}

/// Expand a request's block-level hash_ids into per-token IDs by repeating each
/// hash_id `block_size` times.
fn tokens_from_request(request: &MooncakeRequest, block_size: u32) -> Vec<u32> {
    request
        .hash_ids
        .iter()
        .flat_map(|id| (0..block_size).map(|_| *id as u32))
        .collect()
}

/// Compute the LocalBlockHash for a block-level hash_id the same way the mock
/// engine does: expand to `block_size` repeated u32 tokens, then XXH3 hash.
fn local_block_hash_from_id(hash_id: u64, block_size: u32) -> LocalBlockHash {
    let tokens: Vec<u32> = (0..block_size).map(|_| hash_id as u32).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4) };
    LocalBlockHash(compute_hash_v2(bytes, XXH3_SEED))
}

/// Create a styled progress bar, optionally with a known total length.
fn make_progress_bar(total: Option<u64>) -> ProgressBar {
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

/// Replay each worker's request trace through a mock engine in real-time to
/// produce the KV cache events (store/remove/clear) that the engine would emit.
///
/// Returns one event list per worker, each entry paired with the wall-clock
/// instant it was produced. Event ordering within a worker is guaranteed
/// monotonically non-decreasing by timestamp.
async fn generate_events(
    traces: &Vec<Vec<MooncakeRequest>>,
    args: &Args,
) -> anyhow::Result<Vec<Vec<(KvCacheEvent, Instant)>>> {
    println!("Generating events...");
    let sched_args = MockEngineArgs::builder()
        .num_gpu_blocks(args.num_gpu_blocks)
        .block_size(args.block_size as usize)
        .speedup_ratio(0.0)
        .enable_prefix_caching(true)
        .max_num_batched_tokens(None)
        .max_num_seqs(None)
        .build()?;

    let scaled_traces = traces
        .iter()
        .map(|worker_trace| scale_mooncake_trace(worker_trace, args.trace_simulation_duration_ms));

    let progress = make_progress_bar(Some(
        traces.iter().map(|worker| worker.len() as u64).sum::<u64>(),
    ));

    let mut tasks: Vec<JoinHandle<Vec<(KvCacheEvent, Instant)>>> = Vec::new();
    for worker_trace in scaled_traces {
        let sched_args = sched_args.clone();
        let progress = progress.clone();
        let block_size = args.block_size;
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

                tokio::time::sleep_until(tokio::time::Instant::from(target)).await;
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

    if progress.elapsed() > Duration::from_millis(args.trace_simulation_duration_ms * 11 / 10) {
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

/// Merge each worker's request trace and event trace into a single
/// time-ordered sequence of `WorkerTrace` entries suitable for benchmark
/// replay.
///
/// Timestamps are rescaled from the original trace / simulation durations
/// into the benchmark duration (microseconds).
fn prepare_worker_traces(
    traces: Vec<Vec<MooncakeRequest>>,
    events: Vec<Vec<(KvCacheEvent, Instant)>>,
    args: &Args,
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
                    timestamp_us: request.timestamp * 1000 * args.benchmark_duration_ms
                        / trace_duration_ms,
                    entry: WorkerTraceEntry::Request(
                        request
                            .hash_ids
                            .iter()
                            .map(|id| local_block_hash_from_id(*id, args.block_size))
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
                        * args.benchmark_duration_ms
                        / args.trace_simulation_duration_ms,
                    entry: WorkerTraceEntry::Event(event),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    scaled_request_traces
        .into_iter()
        .zip(scaled_event_traces.into_iter())
        .map(|(request_trace, event_trace)| {
            let mut merged: Vec<WorkerTrace> = request_trace
                .into_iter()
                .chain(event_trace.into_iter())
                .collect();
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
) -> anyhow::Result<()> {
    let worker_traces = prepare_worker_traces(traces, events, args);
    let worker_traces = worker_traces
        .into_iter()
        .map(|trace| Arc::new(trace))
        .collect::<Vec<_>>();

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

    if progress.elapsed() > Duration::from_millis(args.benchmark_duration_ms * 11 / 10) {
        eprintln!(
            "WARNING: The benchmarker is unable to keep up with the request/event generation rate. Rerun with a larger --benchmark-duration-ms."
        )
    }

    println!("Flushing event queue...");

    let request_duration = progress.elapsed();

    let flush_start = Instant::now();
    let flush_size = indexer.flush().await;
    let flush_duration = flush_start.elapsed();

    let event_duration = progress.elapsed();

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

    let event_queue_flush_percentage = flush_size as f32 / total_events as f32 * 100.0;

    println!("Event queue flush duration: {:?}", flush_duration);
    println!(
        "Event queue flush size: {} ({}% of total events)",
        flush_size, event_queue_flush_percentage
    );

    if event_queue_flush_percentage > 5.0 {
        eprintln!(
            "ERROR: Over 5% of events were unable to be completed within the benchmark duration.
        Results are invalid. Rerun with a smaller trace or less worker duplication."
        );
    }

    println!(
        "Request Throughput: {} req/s",
        total_requests as f32 / request_duration.as_millis() as f32 * 1000.0
    );
    println!(
        "Event Throughput: {} events/s",
        total_events as f32 / event_duration.as_millis() as f32 * 1000.0
    );

    latencies.sort_unstable();
    println!(
        "Latency p50: {}us",
        latencies[latencies.len() / 2] as f32 / 1000.0
    );
    println!(
        "Latency p95: {}us",
        latencies[latencies.len() * 95 / 100] as f32 / 1000.0
    );
    println!(
        "Latency p99: {}us",
        latencies[latencies.len() * 99 / 100] as f32 / 1000.0
    );
    println!(
        "Latency max: {}us",
        *latencies.last().unwrap() as f32 / 1000.0
    );

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let traces = process_mooncake_trace(&args)?;

    let events = generate_events(&traces, &args).await?;

    let indexer = args.get_indexer().build(&args);

    run_benchmark(indexer, traces, events, &args).await?;

    Ok(())
}
