// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark for KvIndexer and KvIndexerSharded.
//!
//! Tests the full async indexer interface (not just RadixTree in isolation),
//! allowing comparison between sharded and non-sharded implementations.
//!
//! Run with: cargo run --package dynamo-llm --bin kv_indexer_bench --features kv-router-stress -- --help

use clap::{Parser, ValueEnum};
use dynamo_llm::kv_router::{
    indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded, RouterEvent,
    },
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, WorkerId,
    },
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Indexer type to benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IndexerType {
    /// Non-sharded KvIndexer (single background thread)
    Single,
    /// Sharded KvIndexer (multiple shards with separate trees)
    Sharded,
    /// Run both and compare
    Both,
}

#[derive(Parser, Debug)]
#[command(name = "kv_indexer_bench")]
#[command(about = "Benchmark for KvIndexer vs KvIndexerSharded")]
struct Args {
    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "100000")]
    size: usize,

    /// Sequence depth in blocks (blocks per sequence)
    #[arg(long, default_value = "64")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Number of iterations per operation for timing
    #[arg(long, default_value = "1000")]
    iterations: usize,

    /// Prefix prompt ratio (0.0 to 1.0)
    #[arg(long, default_value = "0.25")]
    prefix_prompt_ratio: f64,

    /// Number of unique prefix prompt groups
    #[arg(long, default_value = "4")]
    num_prefix_prompts: usize,

    /// KV block size in tokens (for hash computation)
    #[arg(long, default_value = "16")]
    block_size: u32,

    /// Indexer type to benchmark
    #[arg(long, value_enum, default_value = "both")]
    indexer_type: IndexerType,

    /// Number of shards for sharded indexer
    #[arg(long, default_value = "4")]
    num_shards: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Run only specific benchmark (store, find_matches, remove, or all)
    #[arg(long, default_value = "all")]
    benchmark_type: String,

    /// Output format: "table" or "csv"
    #[arg(long, default_value = "table")]
    format: String,
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

    fn to_remove_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent {
            worker_id: self.worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: self.external_hashes.clone(),
                }),
                dp_rank: 0,
            },
        }
    }
}

/// Generate sequences with shared prefix prompts
fn generate_sequences(
    num_sequences: usize,
    depth: usize,
    num_workers: usize,
    prefix_prompt_ratio: f64,
    num_prefix_prompts: usize,
    seed: u64,
) -> Vec<SequenceData> {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut sequences = Vec::with_capacity(num_sequences);
    let prefix_length: usize = (depth as f64 * prefix_prompt_ratio).round() as usize;
    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    for seq_id in 0..num_sequences {
        let worker_id = (seq_id % num_workers) as WorkerId;
        let mut seq = SequenceData::new(seq_id as u64, worker_id, depth);

        if num_prefix_prompts > 0 && prefix_length > 0 {
            let group_id = rng.random_range(0..num_prefix_prompts);
            for i in 0..prefix_length {
                seq.local_hashes[i] =
                    LocalBlockHash(0xDEAD_BEEF_0000_0000 | ((group_id as u64) << 32) | (i as u64));
            }
        }

        sequences.push(seq);
    }

    sequences
}

/// Statistics for timing measurements
#[derive(Debug, Clone)]
struct LatencyStats {
    min: Duration,
    max: Duration,
    avg: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    throughput_ops_sec: f64,
}

impl LatencyStats {
    fn from_durations(mut durations: Vec<Duration>) -> Self {
        durations.sort();
        let n = durations.len();
        let total: Duration = durations.iter().sum();
        let avg = total / n as u32;

        Self {
            min: durations[0],
            max: durations[n - 1],
            avg,
            p50: durations[n / 2],
            p95: durations[n * 95 / 100],
            p99: durations[n * 99 / 100],
            throughput_ops_sec: n as f64 / total.as_secs_f64(),
        }
    }

    fn print(&self, operation: &str, blocks_per_op: usize) {
        println!("\n{} Latency Statistics:", operation);
        println!("  min:  {:>12?}", self.min);
        println!("  avg:  {:>12?}", self.avg);
        println!("  p50:  {:>12?}", self.p50);
        println!("  p95:  {:>12?}", self.p95);
        println!("  p99:  {:>12?}", self.p99);
        println!("  max:  {:>12?}", self.max);
        println!("  throughput: {:.2} ops/sec", self.throughput_ops_sec);
        println!(
            "  throughput: {:.2} blocks/sec",
            self.throughput_ops_sec * blocks_per_op as f64
        );
    }
}

/// Format duration for display
fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos() as u64;
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2}us", ns as f64 / 1_000.0)
    } else {
        format!("{}ns", ns)
    }
}

/// Results for a single indexer benchmark
#[derive(Debug)]
struct BenchmarkResults {
    indexer_name: String,
    construction_time: Duration,
    construction_events: usize,
    store_stats: Option<LatencyStats>,
    find_matches_hit_stats: Option<LatencyStats>,
    find_matches_miss_stats: Option<LatencyStats>,
    remove_stats: Option<LatencyStats>,
}

impl BenchmarkResults {
    fn print(&self, depth: usize) {
        println!("\n========================================");
        println!("Results for: {}", self.indexer_name);
        println!("========================================");

        println!("\nConstruction:");
        println!("  Time: {:?}", self.construction_time);
        println!("  Events: {}", self.construction_events);
        println!(
            "  Throughput: {:.0} events/sec",
            self.construction_events as f64 / self.construction_time.as_secs_f64()
        );

        if let Some(ref stats) = self.store_stats {
            stats.print("APPLY_EVENT (store)", depth);
        }
        if let Some(ref stats) = self.find_matches_hit_stats {
            stats.print("FIND_MATCHES (hit)", depth);
        }
        if let Some(ref stats) = self.find_matches_miss_stats {
            stats.print("FIND_MATCHES (miss)", depth);
        }
        if let Some(ref stats) = self.remove_stats {
            stats.print("APPLY_EVENT (remove)", depth);
        }
    }

    fn print_csv_header() {
        println!(
            "indexer,construction_ms,construction_events,construction_throughput,\
             store_avg_us,store_p50_us,store_p99_us,store_throughput,\
             find_hit_avg_us,find_hit_p50_us,find_hit_p99_us,find_hit_throughput,\
             find_miss_avg_us,find_miss_p50_us,find_miss_p99_us,find_miss_throughput,\
             remove_avg_us,remove_p50_us,remove_p99_us,remove_throughput"
        );
    }

    fn print_csv_row(&self) {
        let construction_throughput =
            self.construction_events as f64 / self.construction_time.as_secs_f64();

        let store = self.store_stats.as_ref();
        let find_hit = self.find_matches_hit_stats.as_ref();
        let find_miss = self.find_matches_miss_stats.as_ref();
        let remove = self.remove_stats.as_ref();

        println!(
            "{},{:.3},{},{:.0},{},{},{},{:.0},{},{},{},{:.0},{},{},{},{:.0},{},{},{},{:.0}",
            self.indexer_name,
            self.construction_time.as_secs_f64() * 1000.0,
            self.construction_events,
            construction_throughput,
            store.map(|s| s.avg.as_micros()).unwrap_or(0),
            store.map(|s| s.p50.as_micros()).unwrap_or(0),
            store.map(|s| s.p99.as_micros()).unwrap_or(0),
            store.map(|s| s.throughput_ops_sec).unwrap_or(0.0),
            find_hit.map(|s| s.avg.as_micros()).unwrap_or(0),
            find_hit.map(|s| s.p50.as_micros()).unwrap_or(0),
            find_hit.map(|s| s.p99.as_micros()).unwrap_or(0),
            find_hit.map(|s| s.throughput_ops_sec).unwrap_or(0.0),
            find_miss.map(|s| s.avg.as_micros()).unwrap_or(0),
            find_miss.map(|s| s.p50.as_micros()).unwrap_or(0),
            find_miss.map(|s| s.p99.as_micros()).unwrap_or(0),
            find_miss.map(|s| s.throughput_ops_sec).unwrap_or(0.0),
            remove.map(|s| s.avg.as_micros()).unwrap_or(0),
            remove.map(|s| s.p50.as_micros()).unwrap_or(0),
            remove.map(|s| s.p99.as_micros()).unwrap_or(0),
            remove.map(|s| s.throughput_ops_sec).unwrap_or(0.0),
        );
    }
}

/// Trait for abstracting over KvIndexer and KvIndexerSharded
#[async_trait::async_trait]
trait BenchableIndexer: Send + Sync {
    async fn apply_event(&mut self, event: RouterEvent);
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_llm::kv_router::indexer::KvRouterError>;
    fn name(&self) -> &str;
}

#[async_trait::async_trait]
impl BenchableIndexer for KvIndexer {
    async fn apply_event(&mut self, event: RouterEvent) {
        KvIndexerInterface::apply_event(self, event).await;
    }

    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_llm::kv_router::indexer::KvRouterError> {
        KvIndexerInterface::find_matches(self, sequence).await?;
        Ok(())
    }

    fn name(&self) -> &str {
        "KvIndexer (single)"
    }
}

#[async_trait::async_trait]
impl BenchableIndexer for KvIndexerSharded {
    async fn apply_event(&mut self, event: RouterEvent) {
        KvIndexerInterface::apply_event(self, event).await;
    }

    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_llm::kv_router::indexer::KvRouterError> {
        KvIndexerInterface::find_matches(self, sequence).await?;
        Ok(())
    }

    fn name(&self) -> &str {
        "KvIndexerSharded"
    }
}

/// Build a pre-populated indexer
async fn build_indexer<I: BenchableIndexer>(
    indexer: &mut I,
    sequences: &[SequenceData],
    verbose: bool,
) -> Duration {
    let num_blocks: usize = sequences.iter().map(|s| s.local_hashes.len()).sum();
    print!(
        "  Building {} with {} sequences ({} blocks)... ",
        indexer.name(),
        sequences.len(),
        num_blocks
    );
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let start = Instant::now();
    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_store_event(event_id as u64);
        indexer.apply_event(event).await;

        if verbose && (event_id + 1) % 1000 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    let elapsed = start.elapsed();

    // Allow background processing to complete
    tokio::time::sleep(Duration::from_millis(50)).await;

    println!(
        "done in {:.2?} ({:.2} events/sec)",
        elapsed,
        sequences.len() as f64 / elapsed.as_secs_f64()
    );

    elapsed
}

/// Benchmark apply_event (store) operation
async fn bench_store<I: BenchableIndexer>(
    indexer: &mut I,
    extra_sequences: &[SequenceData],
    iterations: usize,
    verbose: bool,
) -> LatencyStats {
    println!("\n  Benchmarking APPLY_EVENT (store)...");

    let mut durations = Vec::with_capacity(iterations);

    for (i, seq) in extra_sequences.iter().enumerate().take(iterations) {
        let event = seq.to_store_event((1_000_000 + i) as u64);

        let start = Instant::now();
        indexer.apply_event(event).await;
        durations.push(start.elapsed());

        // Remove to restore state (untimed)
        let remove_event = seq.to_remove_event((2_000_000 + i) as u64);
        indexer.apply_event(remove_event).await;

        if verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, iterations);
        }
    }

    LatencyStats::from_durations(durations)
}

/// Benchmark find_matches operation (hit case)
async fn bench_find_matches_hit<I: BenchableIndexer>(
    indexer: &I,
    sequences: &[SequenceData],
    iterations: usize,
    verbose: bool,
) -> LatencyStats {
    println!("\n  Benchmarking FIND_MATCHES (hit)...");

    let mut durations = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let seq = &sequences[i % sequences.len()];
        let hashes = seq.local_hashes.clone();

        let start = Instant::now();
        let _ = indexer.find_matches(hashes).await;
        durations.push(start.elapsed());

        if verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, iterations);
        }
    }

    LatencyStats::from_durations(durations)
}

/// Benchmark find_matches operation (miss case)
async fn bench_find_matches_miss<I: BenchableIndexer>(
    indexer: &I,
    depth: usize,
    iterations: usize,
    verbose: bool,
) -> LatencyStats {
    println!("\n  Benchmarking FIND_MATCHES (miss)...");

    let mut durations = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let miss_hashes: Vec<LocalBlockHash> = (0..depth)
            .map(|j| LocalBlockHash(0xBAD_C0DE_0000_0000 | ((i as u64) << 16) | (j as u64)))
            .collect();

        let start = Instant::now();
        let _ = indexer.find_matches(miss_hashes).await;
        durations.push(start.elapsed());

        if verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, iterations);
        }
    }

    LatencyStats::from_durations(durations)
}

/// Benchmark apply_event (remove) operation
async fn bench_remove<I: BenchableIndexer>(
    indexer: &mut I,
    sequences: &[SequenceData],
    iterations: usize,
    verbose: bool,
) -> LatencyStats {
    println!("\n  Benchmarking APPLY_EVENT (remove)...");

    let mut durations = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let seq = &sequences[i % sequences.len()];
        let remove_event = seq.to_remove_event((3_000_000 + i) as u64);

        let start = Instant::now();
        indexer.apply_event(remove_event).await;
        durations.push(start.elapsed());

        // Re-add to restore state (untimed)
        let store_event = seq.to_store_event((4_000_000 + i) as u64);
        indexer.apply_event(store_event).await;

        if verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, iterations);
        }
    }

    LatencyStats::from_durations(durations)
}

/// Run all benchmarks for an indexer
async fn run_benchmarks<I: BenchableIndexer>(
    indexer: &mut I,
    sequences: &[SequenceData],
    extra_sequences: &[SequenceData],
    args: &Args,
) -> BenchmarkResults {
    let indexer_name = indexer.name().to_string();
    println!("\n--- Benchmarking {} ---", indexer_name);

    // Build the indexer
    let construction_time = build_indexer(indexer, sequences, args.verbose).await;
    let construction_events = sequences.len();

    let run_all = args.benchmark_type == "all";

    let store_stats = if run_all || args.benchmark_type == "store" {
        Some(bench_store(indexer, extra_sequences, args.iterations, args.verbose).await)
    } else {
        None
    };

    let find_matches_hit_stats = if run_all || args.benchmark_type == "find_matches" {
        Some(bench_find_matches_hit(indexer, sequences, args.iterations, args.verbose).await)
    } else {
        None
    };

    let find_matches_miss_stats = if run_all || args.benchmark_type == "find_matches" {
        Some(bench_find_matches_miss(indexer, args.depth, args.iterations, args.verbose).await)
    } else {
        None
    };

    let remove_stats = if run_all || args.benchmark_type == "remove" {
        Some(bench_remove(indexer, sequences, args.iterations, args.verbose).await)
    } else {
        None
    };

    BenchmarkResults {
        indexer_name,
        construction_time,
        construction_events,
        store_stats,
        find_matches_hit_stats,
        find_matches_miss_stats,
        remove_stats,
    }
}

fn print_comparison(results: &[BenchmarkResults], depth: usize) {
    if results.len() < 2 {
        return;
    }

    println!("\n========================================");
    println!("COMPARISON SUMMARY");
    println!("========================================\n");

    let single = &results[0];
    let sharded = &results[1];

    println!(
        "{:<30} {:>15} {:>15} {:>10}",
        "Metric", "Single", "Sharded", "Ratio"
    );
    println!("{}", "-".repeat(72));

    // Construction
    let single_constr = single.construction_time.as_secs_f64() * 1000.0;
    let sharded_constr = sharded.construction_time.as_secs_f64() * 1000.0;
    println!(
        "{:<30} {:>12.2}ms {:>12.2}ms {:>9.2}x",
        "Construction time",
        single_constr,
        sharded_constr,
        single_constr / sharded_constr
    );

    // Store p50
    if let (Some(s1), Some(s2)) = (&single.store_stats, &sharded.store_stats) {
        let s1_us = s1.p50.as_nanos() as f64 / 1000.0;
        let s2_us = s2.p50.as_nanos() as f64 / 1000.0;
        println!(
            "{:<30} {:>12.2}us {:>12.2}us {:>9.2}x",
            "Store p50",
            s1_us,
            s2_us,
            s1_us / s2_us
        );
    }

    // Find matches hit p50
    if let (Some(s1), Some(s2)) = (
        &single.find_matches_hit_stats,
        &sharded.find_matches_hit_stats,
    ) {
        let s1_us = s1.p50.as_nanos() as f64 / 1000.0;
        let s2_us = s2.p50.as_nanos() as f64 / 1000.0;
        println!(
            "{:<30} {:>12.2}us {:>12.2}us {:>9.2}x",
            "Find matches (hit) p50",
            s1_us,
            s2_us,
            s1_us / s2_us
        );
    }

    // Find matches hit p99
    if let (Some(s1), Some(s2)) = (
        &single.find_matches_hit_stats,
        &sharded.find_matches_hit_stats,
    ) {
        let s1_us = s1.p99.as_nanos() as f64 / 1000.0;
        let s2_us = s2.p99.as_nanos() as f64 / 1000.0;
        println!(
            "{:<30} {:>12.2}us {:>12.2}us {:>9.2}x",
            "Find matches (hit) p99",
            s1_us,
            s2_us,
            s1_us / s2_us
        );
    }

    // Find matches miss p50
    if let (Some(s1), Some(s2)) = (
        &single.find_matches_miss_stats,
        &sharded.find_matches_miss_stats,
    ) {
        let s1_us = s1.p50.as_nanos() as f64 / 1000.0;
        let s2_us = s2.p50.as_nanos() as f64 / 1000.0;
        println!(
            "{:<30} {:>12.2}us {:>12.2}us {:>9.2}x",
            "Find matches (miss) p50",
            s1_us,
            s2_us,
            s1_us / s2_us
        );
    }

    // Remove p50
    if let (Some(s1), Some(s2)) = (&single.remove_stats, &sharded.remove_stats) {
        let s1_us = s1.p50.as_nanos() as f64 / 1000.0;
        let s2_us = s2.p50.as_nanos() as f64 / 1000.0;
        println!(
            "{:<30} {:>12.2}us {:>12.2}us {:>9.2}x",
            "Remove p50",
            s1_us,
            s2_us,
            s1_us / s2_us
        );
    }

    // Throughput comparison
    println!();
    println!(
        "{:<30} {:>15} {:>15} {:>10}",
        "Throughput (ops/sec)", "Single", "Sharded", "Ratio"
    );
    println!("{}", "-".repeat(72));

    if let (Some(s1), Some(s2)) = (
        &single.find_matches_hit_stats,
        &sharded.find_matches_hit_stats,
    ) {
        println!(
            "{:<30} {:>12.0}/s {:>12.0}/s {:>9.2}x",
            "Find matches (hit)",
            s1.throughput_ops_sec,
            s2.throughput_ops_sec,
            s2.throughput_ops_sec / s1.throughput_ops_sec
        );
    }

    println!("\nNote: Ratio > 1.0 means sharded is faster for that metric.");
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let num_sequences = args.size / args.depth;
    if num_sequences == 0 {
        eprintln!("Error: size must be >= depth");
        std::process::exit(1);
    }

    println!("KvIndexer Benchmark");
    println!("===================\n");
    println!("Configuration:");
    println!("  Target size: {} (worker, block) pairs", args.size);
    println!(
        "  Depth: {} blocks/sequence (= {} tokens with block_size={})",
        args.depth,
        args.depth * args.block_size as usize,
        args.block_size
    );
    println!("  Block size: {} tokens", args.block_size);
    println!("  Workers: {}", args.num_workers);
    println!("  Iterations: {}", args.iterations);
    println!(
        "  Prefix prompt ratio: {:.1}%",
        args.prefix_prompt_ratio * 100.0
    );
    println!("  Prefix prompt groups: {}", args.num_prefix_prompts);
    println!("  Num shards (for sharded): {}", args.num_shards);
    println!("  Indexer type: {:?}", args.indexer_type);
    println!("  Benchmark type: {}", args.benchmark_type);
    println!("\n  Derived: {} sequences to reach target size", num_sequences);

    // Generate sequences
    let extra_count = args.iterations;
    let all_sequences = generate_sequences(
        num_sequences + extra_count,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );
    let sequences = &all_sequences[..num_sequences];
    let extra_sequences = &all_sequences[num_sequences..];

    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let mut results = Vec::new();

    // Benchmark single indexer
    if matches!(args.indexer_type, IndexerType::Single | IndexerType::Both) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexer::new(token.clone(), args.block_size, metrics.clone());
        let result = run_benchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        token.cancel();
        // Allow cleanup
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Benchmark sharded indexer
    if matches!(args.indexer_type, IndexerType::Sharded | IndexerType::Both) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexerSharded::new(
            token.clone(),
            args.num_shards,
            args.block_size,
            metrics.clone(),
        );
        let result = run_benchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        token.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Print results
    if args.format == "csv" {
        BenchmarkResults::print_csv_header();
        for result in &results {
            result.print_csv_row();
        }
    } else {
        for result in &results {
            result.print(args.depth);
        }

        if results.len() == 2 {
            print_comparison(&results, args.depth);
        }
    }

    println!("\nBenchmark complete.");
}
