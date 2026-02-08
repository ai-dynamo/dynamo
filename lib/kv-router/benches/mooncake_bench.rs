use clap::{Parser, Subcommand};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded, PositionalIndexer,
};
use dynamo_kv_router::protocols::RouterEvent;
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

// TODO: Import your actual indexer interface and implementations
// use dynamo_kv_router::indexer::{KvIndexerInterface, HashTableIndexer, BTreeIndexer, ...};

#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    RadixTree {},

    RadixTreeSharded {
        #[clap(long, default_value = "4")]
        num_shards: usize,
    },

    NestedMap {
        #[clap(long, default_value = "8")]
        jump_size: usize,

        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },
}

impl IndexerArgs {
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
            } => Arc::new(PositionalIndexer::new(
                num_event_workers,
                args.block_size,
                jump_size,
            )),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    /// Path to mooncake trace.
    mooncake_trace_path: String,

    #[clap(long, default_value = "2048")]
    num_gpu_blocks: usize,

    #[clap(long, default_value = "512")]
    block_size: u32,

    #[clap(long, default_value = "30000")]
    trace_simulation_duration_ms: u64,

    #[clap(long, default_value = "60000")]
    benchmark_duration_ms: u64,

    #[clap(short, long, default_value = "64")]
    num_unique_inference_workers: usize,

    #[clap(short = 'd', long, default_value = "1")]
    inference_worker_duplication_factor: usize,

    #[clap(long, default_value = "42")]
    seed: u64,

    /// Indexer configuration (defaults to hash-table if not specified)
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,

    /// Ignored - passed by cargo bench harness
    #[arg(long, hide = true, global = true)]
    bench: bool,
}

impl Args {
    /// Get the indexer config, using a default if none was specified
    fn get_indexer(&self) -> IndexerArgs {
        self.indexer.clone().unwrap_or(IndexerArgs::RadixTree {})
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct MooncakeRequest {
    #[serde(default = "Uuid::new_v4")]
    uuid: uuid::Uuid,
    timestamp: u64,
    hash_ids: Vec<u64>,
    output_length: u64,
}

struct EventCollector {
    events: Mutex<Option<Vec<(KvCacheEvent, Instant)>>>,
}

impl EventCollector {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Some(Vec::new())),
        })
    }

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

#[derive(Clone)]
enum WorkerTraceEntry {
    Request(Vec<LocalBlockHash>),
    Event(KvCacheEvent),
}

#[derive(Clone)]
struct WorkerTrace {
    entry: WorkerTraceEntry,
    timestamp_us: u64,
}

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

fn tokens_from_request(request: &MooncakeRequest, block_size: u32) -> Vec<u32> {
    request
        .hash_ids
        .iter()
        .flat_map(|id| (0..block_size).map(|_| *id as u32))
        .collect()
}

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

async fn generate_events(
    traces: &Vec<Vec<MooncakeRequest>>,
    args: &Args,
) -> anyhow::Result<Vec<Vec<(KvCacheEvent, Instant)>>> {
    println!("Generating events...");
    let sched_args = MockEngineArgs::builder()
        .num_gpu_blocks(args.num_gpu_blocks) // small cache to force evictions
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
                            .map(|id| LocalBlockHash(*id))
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
            // Merge the request and event traces together per worker, sort by timestamp, and merge into 1 Vec
            let mut merged: Vec<WorkerTrace> = request_trace
                .into_iter()
                .chain(event_trace.into_iter())
                .collect();
            merged.sort_by_key(|entry| entry.timestamp_us);
            merged
        })
        .collect()
}

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
