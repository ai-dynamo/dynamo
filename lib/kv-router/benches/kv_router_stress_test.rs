// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue saturation stress test for the KV Router's radix tree indexer.
//!
//! Tests what happens when `find_matches` requests arrive faster than they can be processed.
//!
//! Run with: cargo run --package dynamo-llm --bin kv_router_stress_test --features kv-router-stress

use clap::Parser;
use dynamo_llm::kv_router::{
    indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, RouterEvent},
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, WorkerId,
    },
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
#[command(name = "kv_router_stress_test")]
#[command(about = "Queue saturation stress test for the KV Router radix tree indexer")]
struct Args {
    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "500000")]
    size: usize,

    /// Sequence depth in blocks (blocks per sequence)
    #[arg(long, default_value = "512")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Prefix sharing ratio (0.0 to 1.0) - fraction of sequences sharing a common prefix
    #[arg(long, default_value = "0.5")]
    prefix_share_ratio: f64,

    /// KV block size in tokens
    #[arg(long, default_value = "16")]
    kv_block_size: u32,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Requests per second to submit
    #[arg(long, default_value = "20.0")]
    arrival_rate: f64,

    /// Test duration in seconds
    #[arg(long, default_value = "10")]
    duration: u64,

    /// Seconds to wait for in-flight requests after test
    #[arg(long, default_value = "5")]
    in_flight_timeout: u64,

    /// Enable verbose output with per-request timings
    #[arg(short, long)]
    verbose: bool,
}

/// Pre-generated sequence data for benchmarking
#[derive(Clone)]
struct SequenceData {
    worker_id: WorkerId,
    local_hashes: Vec<LocalBlockHash>,
    external_hashes: Vec<ExternalSequenceBlockHash>,
}

impl SequenceData {
    fn new(seq_id: u64, worker_id: WorkerId, depth: usize) -> Self {
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

    fn to_store_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent {
            worker_id: self.worker_id,
            event: KvCacheEvent {
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
        }
    }
}

/// Generate sequences with optional prefix sharing
fn generate_sequences(
    num_sequences: usize,
    depth: usize,
    num_workers: usize,
    prefix_share_ratio: f64,
) -> Vec<SequenceData> {
    let mut sequences = Vec::with_capacity(num_sequences);
    let num_shared_prefix = (num_sequences as f64 * prefix_share_ratio) as usize;
    let prefix_length = depth / 4;

    for seq_id in 0..num_sequences {
        let worker_id = (seq_id % num_workers) as WorkerId;
        let mut seq = SequenceData::new(seq_id as u64, worker_id, depth);

        if seq_id < num_shared_prefix && prefix_length > 0 {
            for i in 0..prefix_length {
                seq.local_hashes[i] = LocalBlockHash(0xDEAD_BEEF_0000_0000 | (i as u64));
            }
        }

        sequences.push(seq);
    }

    sequences
}

/// Saturation test configuration
struct SaturationConfig {
    size: usize,
    depth: usize,
    num_workers: usize,
    prefix_share_ratio: f64,
    kv_block_size: u32,
    arrival_rate: f64,
    duration_secs: u64,
    in_flight_timeout_secs: u64,
    verbose: bool,
}

impl From<&Args> for SaturationConfig {
    fn from(args: &Args) -> Self {
        Self {
            size: args.size,
            depth: args.depth,
            num_workers: args.num_workers,
            prefix_share_ratio: args.prefix_share_ratio,
            kv_block_size: args.kv_block_size,
            arrival_rate: args.arrival_rate,
            duration_secs: args.duration,
            in_flight_timeout_secs: args.in_flight_timeout,
            verbose: args.verbose,
        }
    }
}

/// Result of a single request during saturation test
#[allow(dead_code)]
struct RequestResult {
    request_id: u64,
    submit_time: Instant,
    complete_time: Instant,
    success: bool,
}

/// Aggregated results from saturation test
struct SaturationResults {
    submitted: u64,
    completed: u64,
    timed_out: u64,
    latencies: Vec<Duration>,
    max_in_flight: u64,
    baseline_service_time: Duration,
    construction_time: Duration,
    construction_events: u64,
}

/// Statistics for latency measurements
struct LatencyStats {
    min: Duration,
    max: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
}

impl LatencyStats {
    fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();
        let n = sorted.len();

        Some(Self {
            min: sorted[0],
            max: sorted[n - 1],
            p50: sorted[n / 2],
            p95: sorted[n * 95 / 100],
            p99: sorted[n * 99 / 100],
        })
    }
}

fn create_indexer(token: &CancellationToken, kv_block_size: u32) -> KvIndexer {
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    KvIndexer::new(token.clone(), kv_block_size, metrics)
}

/// Compute median of durations
fn median(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

/// Run the saturation test
async fn run_saturation_test(
    config: &SaturationConfig,
    indexer: &KvIndexer,
    sequences: &[SequenceData],
) -> SaturationResults {
    // Phase 2: Baseline Measurement
    println!("\nPhase 2: Baseline Measurement");
    println!("  Running 10 sequential find_matches calls...");

    let mut baseline_durations = Vec::new();
    for seq in sequences.iter().take(10) {
        let start = Instant::now();
        let _ = indexer.find_matches(seq.local_hashes.clone()).await;
        baseline_durations.push(start.elapsed());
    }
    let baseline_service_time = median(&baseline_durations);
    let theoretical_max = 1.0 / baseline_service_time.as_secs_f64();

    println!(
        "  Baseline find_matches latency: {:?} (median of 10)",
        baseline_service_time
    );
    println!(
        "  Theoretical max throughput: {:.1} req/sec",
        theoretical_max
    );

    // Phase 3: Pre-generate Lookup Sequences
    println!("\nPhase 3: Pre-generating Lookup Sequences");
    let expected_requests =
        (config.arrival_rate * config.duration_secs as f64).ceil() as usize + 100;
    let lookup_sequences: Vec<Vec<LocalBlockHash>> = (0..expected_requests)
        .map(|i| {
            let seq = &sequences[i % sequences.len()];
            seq.local_hashes.clone()
        })
        .collect();
    println!(
        "  Pre-generated {} lookup sequences",
        lookup_sequences.len()
    );

    // Phase 4: Saturation Test
    println!("\nPhase 4: Saturation Test");
    println!("  Arrival rate: {:.1} req/sec", config.arrival_rate);
    println!("  Duration: {}s", config.duration_secs);

    let in_flight = Arc::new(AtomicU64::new(0));
    let max_in_flight = Arc::new(AtomicU64::new(0));
    let (result_tx, mut result_rx) = mpsc::channel::<RequestResult>(expected_requests);

    let start = Instant::now();
    let mut request_id = 0u64;
    let interval = Duration::from_secs_f64(1.0 / config.arrival_rate);

    while start.elapsed() < Duration::from_secs(config.duration_secs) {
        let submit_time = Instant::now();
        let seq = lookup_sequences[request_id as usize].clone();

        // Track in-flight
        let current = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        max_in_flight.fetch_max(current, Ordering::Relaxed);

        let indexer = indexer.clone();
        let result_tx = result_tx.clone();
        let in_flight_clone = in_flight.clone();
        let req_id = request_id;
        let verbose = config.verbose;

        tokio::spawn(async move {
            let result = indexer.find_matches(seq).await;
            let complete_time = Instant::now();
            in_flight_clone.fetch_sub(1, Ordering::Relaxed);

            if verbose {
                let latency = complete_time.duration_since(submit_time);
                println!("    Request {} completed in {:?}", req_id, latency);
            }

            let _ = result_tx
                .send(RequestResult {
                    request_id: req_id,
                    submit_time,
                    complete_time,
                    success: result.is_ok(),
                })
                .await;
        });

        request_id += 1;
        tokio::time::sleep(interval).await;
    }

    let submitted = request_id;
    println!("  Submitted {} requests", submitted);

    // Wait for in-flight requests with timeout
    println!("\nPhase 5: Draining In-flight Requests");
    let drain_start = Instant::now();
    let mut last_in_flight = in_flight.load(Ordering::Relaxed);
    println!(
        "  Waiting for {} in-flight requests (timeout: {}s)...",
        last_in_flight, config.in_flight_timeout_secs
    );

    while in_flight.load(Ordering::Relaxed) > 0
        && drain_start.elapsed() < Duration::from_secs(config.in_flight_timeout_secs)
    {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let current = in_flight.load(Ordering::Relaxed);
        if current != last_in_flight && config.verbose {
            println!("    In-flight: {}", current);
            last_in_flight = current;
        }
    }
    let timed_out = in_flight.load(Ordering::Relaxed);
    if timed_out > 0 {
        println!("  {} requests timed out", timed_out);
    } else {
        println!("  All requests completed");
    }

    // Collect results
    drop(result_tx);
    let mut results = Vec::new();
    while let Some(r) = result_rx.recv().await {
        results.push(r);
    }

    // Compute latencies
    let latencies: Vec<Duration> = results
        .iter()
        .map(|r| r.complete_time.duration_since(r.submit_time))
        .collect();

    SaturationResults {
        submitted,
        completed: results.len() as u64,
        timed_out,
        latencies,
        max_in_flight: max_in_flight.load(Ordering::Relaxed),
        baseline_service_time,
        construction_time: Duration::ZERO, // Set by caller
        construction_events: 0,            // Set by caller
    }
}

/// Print the final results report
fn print_results(config: &SaturationConfig, results: &SaturationResults) {
    let num_sequences = config.size / config.depth;

    println!("\n=====================");
    println!("Queue Saturation Test Results");
    println!("=====================\n");

    println!("Configuration:");
    println!(
        "  Tree size: {} blocks ({} sequences x {} depth)",
        config.size, num_sequences, config.depth
    );
    println!("  Workers: {}", config.num_workers);
    println!(
        "  Prefix share ratio: {:.1}%",
        config.prefix_share_ratio * 100.0
    );
    println!("  Arrival rate: {:.1} req/sec", config.arrival_rate);
    println!("  Duration: {}s", config.duration_secs);
    println!();

    println!("Tree Construction:");
    println!("  Time: {:.2?}", results.construction_time);
    println!("  Events: {}", results.construction_events);
    let throughput = results.construction_events as f64 / results.construction_time.as_secs_f64();
    println!("  Throughput: {:.0} events/sec", throughput);
    println!();

    println!("Baseline:");
    println!(
        "  find_matches latency: {:?} (median of 10)",
        results.baseline_service_time
    );
    let theoretical_max = 1.0 / results.baseline_service_time.as_secs_f64();
    println!(
        "  Theoretical max throughput: {:.1} req/sec",
        theoretical_max
    );
    println!();

    println!("Saturation Test Results:");
    println!("  Submitted: {} requests", results.submitted);
    println!("  Completed: {} requests", results.completed);
    println!(
        "  Timed out: {} requests (in-flight at end)",
        results.timed_out
    );
    println!();

    if !results.latencies.is_empty() {
        let test_duration = config.duration_secs as f64 + config.in_flight_timeout_secs as f64;
        let achieved_throughput = results.completed as f64 / test_duration;

        println!("  Throughput:");
        println!("    Requested: {:.1} req/sec", config.arrival_rate);
        println!("    Achieved: {:.1} req/sec", achieved_throughput);
        println!();

        if let Some(stats) = LatencyStats::from_durations(&results.latencies) {
            println!("  Latency (end-to-end, includes queue wait):");
            println!("    min:  {:>12?}", stats.min);
            println!("    p50:  {:>12?}", stats.p50);
            println!("    p95:  {:>12?}", stats.p95);
            println!("    p99:  {:>12?}", stats.p99);
            println!("    max:  {:>12?}", stats.max);
            println!();

            let estimated_queue_wait = if stats.p50 > results.baseline_service_time {
                stats.p50 - results.baseline_service_time
            } else {
                Duration::ZERO
            };

            println!("  Queue Analysis:");
            println!(
                "    Baseline service time: {:?}",
                results.baseline_service_time
            );
            println!("    Estimated queue wait (p50): {:?}", estimated_queue_wait);
            println!("    Max in-flight observed: {}", results.max_in_flight);
            println!();

            // Determine saturation status
            let is_saturated = achieved_throughput < config.arrival_rate * 0.95
                || results.timed_out > 0
                || stats.p50 > results.baseline_service_time * 2;

            if is_saturated {
                println!("  STATUS: SATURATED");
                if achieved_throughput < config.arrival_rate * 0.95 {
                    println!(
                        "    - Throughput ({:.1}) < Arrival rate ({:.1})",
                        achieved_throughput, config.arrival_rate
                    );
                }
                if results.timed_out > 0 {
                    println!("    - Requests timed out: {}", results.timed_out);
                }
                if stats.p50 > results.baseline_service_time * 2 {
                    println!(
                        "    - P50 latency ({:?}) > 2x baseline ({:?})",
                        stats.p50, results.baseline_service_time
                    );
                }
            } else {
                println!("  STATUS: NOT SATURATED");
                println!("    - Throughput matches arrival rate");
                println!("    - No requests timed out");
                println!("    - Latency within acceptable bounds");
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let config = SaturationConfig::from(&args);

    let num_sequences = config.size / config.depth;

    println!("Queue Saturation Test");
    println!("=====================\n");

    println!("Configuration:");
    println!(
        "  Tree size: {} blocks ({} sequences x {} depth)",
        config.size, num_sequences, config.depth
    );
    println!("  Workers: {}", config.num_workers);
    println!("  Block size: {} tokens", config.kv_block_size);
    println!(
        "  Prefix share ratio: {:.1}%",
        config.prefix_share_ratio * 100.0
    );
    println!("  Seed: {}", args.seed);
    println!("  Arrival rate: {:.1} req/sec", config.arrival_rate);
    println!("  Duration: {}s", config.duration_secs);
    println!("  In-flight timeout: {}s", config.in_flight_timeout_secs);

    let token = CancellationToken::new();
    let mut indexer = create_indexer(&token, config.kv_block_size);

    // Phase 1: Tree Construction
    println!("\nPhase 1: Tree Construction");
    println!("  Generating {} sequences...", num_sequences);

    let sequences = generate_sequences(
        num_sequences,
        config.depth,
        config.num_workers,
        config.prefix_share_ratio,
    );

    println!("  Applying {} store events...", sequences.len());
    let construction_start = Instant::now();

    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_store_event(event_id as u64);
        indexer.apply_event(event).await;

        if config.verbose && (event_id + 1) % 100 == 0 {
            println!("    Applied {}/{} events...", event_id + 1, sequences.len());
        }
    }

    let construction_time = construction_start.elapsed();
    let construction_events = sequences.len() as u64;

    println!("  Tree construction completed in {:?}", construction_time);
    println!(
        "  Throughput: {:.0} events/sec",
        construction_events as f64 / construction_time.as_secs_f64()
    );

    // Wait for events to be processed
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Run saturation test
    let mut results = run_saturation_test(&config, &indexer, &sequences).await;
    results.construction_time = construction_time;
    results.construction_events = construction_events;

    // Print final results
    print_results(&config, &results);

    token.cancel();
}
