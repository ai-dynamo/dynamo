// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Combined benchmark for KvIndexer, KvIndexerSharded, and PositionalIndexer (nested).
//!
//! Provides two modes:
//! - `microbench`: Per-operation latency benchmarks comparing indexer implementations
//! - `stress`: Queue saturation stress test under load
//!
//! Supported indexer types: single, sharded, nested, all
//!
//! Run with:
//!   cargo bench --package dynamo-kv-router --bench kv_indexer_bench --features bench -- microbench --help
//!   cargo bench --package dynamo-kv-router --bench kv_indexer_bench --features bench -- stress --help

use clap::{Args, Parser, Subcommand, ValueEnum};
use dynamo_bench::common::LatencyStats;
use dynamo_kv_router::{
    ConcurrentRadixTree,
    bench_utils::{SequenceData, generate_sequences},
    indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvIndexerSharded, ThreadPoolIndexer,
    },
    nested_map::PositionalIndexer,
    protocols::{LocalBlockHash, RouterEvent},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

// ============================================================================
// CLI Definitions
// ============================================================================

#[derive(Parser)]
#[command(name = "kv_indexer_bench")]
#[command(about = "Combined benchmark for KvIndexer, KvIndexerSharded, and PositionalIndexer")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Ignored - passed by cargo bench harness
    #[arg(long, hide = true, global = true)]
    bench: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Per-operation latency benchmarks comparing indexer implementations
    Microbench(MicrobenchArgs),
    /// Queue saturation stress test under load
    Stress(StressArgs),
}

/// Indexer type to benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IndexerType {
    /// Non-sharded KvIndexer (single background thread)
    Single,
    /// Sharded KvIndexer (multiple shards with separate trees)
    Sharded,
    /// Nested PositionalIndexer (position-based HashMap with jump search)
    Nested,
    /// Concurrent radix tree (lock-per-node with DashMap lookup)
    Concurrent,
    /// Run all indexer types and compare
    All,
}

/// Common arguments shared between subcommands
#[derive(Args, Debug, Clone)]
struct CommonArgs {
    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "100000")]
    size: usize,

    /// Sequence depth in blocks (blocks per sequence)
    #[arg(long, default_value = "64")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// KV block size in tokens
    #[arg(long, default_value = "16")]
    block_size: u32,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args, Debug)]
struct MicrobenchArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Number of iterations per operation for timing
    #[arg(long, default_value = "1000")]
    iterations: usize,

    /// Prefix prompt ratio (0.0 to 1.0)
    #[arg(long, default_value = "0.25")]
    prefix_prompt_ratio: f64,

    /// Number of unique prefix prompt groups
    #[arg(long, default_value = "4")]
    num_prefix_prompts: usize,

    /// Indexer type to benchmark
    #[arg(long, value_enum, default_value = "all")]
    indexer_type: IndexerType,

    /// Number of shards for sharded indexer
    #[arg(long, default_value = "4")]
    num_shards: usize,

    /// Jump size for nested/positional indexer
    #[arg(long, default_value = "32")]
    jump_size: usize,

    /// Run only specific benchmark (store, find_matches, remove, or all)
    #[arg(long, default_value = "all")]
    benchmark_type: String,

    /// Output format: "table" or "csv"
    #[arg(long, default_value = "table")]
    format: String,
}

#[derive(Args, Debug)]
struct StressArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Prefix sharing ratio (0.0 to 1.0) - fraction of sequences sharing a common prefix
    #[arg(long, default_value = "0.5")]
    prefix_share_ratio: f64,

    /// Requests per second to submit
    #[arg(long, default_value = "20.0")]
    arrival_rate: f64,

    /// Test duration in seconds
    #[arg(long, default_value = "10")]
    duration: u64,

    /// Seconds to wait for in-flight requests after test
    #[arg(long, default_value = "5")]
    in_flight_timeout: u64,

    /// Indexer type to stress test
    #[arg(long, value_enum, default_value = "single")]
    indexer_type: IndexerType,

    /// Number of shards for sharded indexer
    #[arg(long, default_value = "4")]
    num_shards: usize,

    /// Jump size for nested/positional indexer
    #[arg(long, default_value = "32")]
    jump_size: usize,
}

// ============================================================================
// Benchable Indexer Trait
// ============================================================================

/// Trait for abstracting over KvIndexer and KvIndexerSharded
#[async_trait::async_trait]
trait BenchableIndexer: Send + Sync {
    async fn apply_event(&mut self, event: RouterEvent);
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError>;
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
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError> {
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
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError> {
        KvIndexerInterface::find_matches(self, sequence).await?;
        Ok(())
    }

    fn name(&self) -> &str {
        "KvIndexerSharded"
    }
}

#[async_trait::async_trait]
impl BenchableIndexer for ThreadPoolIndexer<PositionalIndexer> {
    async fn apply_event(&mut self, event: RouterEvent) {
        KvIndexerInterface::apply_event(self, event).await;
    }

    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError> {
        KvIndexerInterface::find_matches(self, sequence).await?;
        Ok(())
    }

    fn name(&self) -> &str {
        "PositionalIndexer (nested)"
    }
}

#[async_trait::async_trait]
impl BenchableIndexer for ThreadPoolIndexer<ConcurrentRadixTree> {
    async fn apply_event(&mut self, event: RouterEvent) {
        KvIndexerInterface::apply_event(self, event).await;
    }

    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError> {
        KvIndexerInterface::find_matches(self, sequence).await?;
        Ok(())
    }

    fn name(&self) -> &str {
        "ConcurrentRadixTree"
    }
}

// ============================================================================
// Microbench Mode
// ============================================================================

/// Results for a single indexer benchmark
#[derive(Debug)]
struct MicrobenchResults {
    indexer_name: String,
    construction_time: Duration,
    construction_events: usize,
    store_stats: Option<LatencyStats>,
    find_matches_hit_stats: Option<LatencyStats>,
    find_matches_miss_stats: Option<LatencyStats>,
    remove_stats: Option<LatencyStats>,
}

impl MicrobenchResults {
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

    LatencyStats::from_durations(&durations).unwrap()
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

    LatencyStats::from_durations(&durations).unwrap()
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

    LatencyStats::from_durations(&durations).unwrap()
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

    LatencyStats::from_durations(&durations).unwrap()
}

/// Run all microbenchmarks for an indexer
async fn run_microbenchmarks<I: BenchableIndexer>(
    indexer: &mut I,
    sequences: &[SequenceData],
    extra_sequences: &[SequenceData],
    args: &MicrobenchArgs,
) -> MicrobenchResults {
    let indexer_name = indexer.name().to_string();
    println!("\n--- Benchmarking {} ---", indexer_name);

    // Build the indexer
    let construction_time = build_indexer(indexer, sequences, args.common.verbose).await;
    let construction_events = sequences.len();

    let run_all = args.benchmark_type == "all";

    let store_stats = if run_all || args.benchmark_type == "store" {
        Some(
            bench_store(
                indexer,
                extra_sequences,
                args.iterations,
                args.common.verbose,
            )
            .await,
        )
    } else {
        None
    };

    let find_matches_hit_stats = if run_all || args.benchmark_type == "find_matches" {
        Some(bench_find_matches_hit(indexer, sequences, args.iterations, args.common.verbose).await)
    } else {
        None
    };

    let find_matches_miss_stats = if run_all || args.benchmark_type == "find_matches" {
        Some(
            bench_find_matches_miss(
                indexer,
                args.common.depth,
                args.iterations,
                args.common.verbose,
            )
            .await,
        )
    } else {
        None
    };

    let remove_stats = if run_all || args.benchmark_type == "remove" {
        Some(bench_remove(indexer, sequences, args.iterations, args.common.verbose).await)
    } else {
        None
    };

    MicrobenchResults {
        indexer_name,
        construction_time,
        construction_events,
        store_stats,
        find_matches_hit_stats,
        find_matches_miss_stats,
        remove_stats,
    }
}

fn print_microbench_comparison(results: &[MicrobenchResults], _depth: usize) {
    if results.len() < 2 {
        return;
    }

    println!("\n========================================");
    println!("COMPARISON SUMMARY");
    println!("========================================\n");

    // Build dynamic column headers
    let mut header = format!("{:<30}", "Metric");
    for result in results {
        header.push_str(&format!(
            " {:>15}",
            result
                .indexer_name
                .split_whitespace()
                .next()
                .unwrap_or(&result.indexer_name)
        ));
    }
    println!("{}", header);
    println!("{}", "-".repeat(30 + results.len() * 16));

    // Construction time
    let mut row = format!("{:<30}", "Construction time (ms)");
    for result in results {
        row.push_str(&format!(
            " {:>15.2}",
            result.construction_time.as_secs_f64() * 1000.0
        ));
    }
    println!("{}", row);

    // Store p50
    let mut row = format!("{:<30}", "Store p50 (us)");
    for result in results {
        if let Some(stats) = &result.store_stats {
            row.push_str(&format!(" {:>15.2}", stats.p50.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);

    // Find matches hit p50
    let mut row = format!("{:<30}", "Find hit p50 (us)");
    for result in results {
        if let Some(stats) = &result.find_matches_hit_stats {
            row.push_str(&format!(" {:>15.2}", stats.p50.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);

    // Find matches hit p99
    let mut row = format!("{:<30}", "Find hit p99 (us)");
    for result in results {
        if let Some(stats) = &result.find_matches_hit_stats {
            row.push_str(&format!(" {:>15.2}", stats.p99.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);

    // Find matches miss p50
    let mut row = format!("{:<30}", "Find miss p50 (us)");
    for result in results {
        if let Some(stats) = &result.find_matches_miss_stats {
            row.push_str(&format!(" {:>15.2}", stats.p50.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);

    // Remove p50
    let mut row = format!("{:<30}", "Remove p50 (us)");
    for result in results {
        if let Some(stats) = &result.remove_stats {
            row.push_str(&format!(" {:>15.2}", stats.p50.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);

    // Throughput
    println!();
    let mut header = format!("{:<30}", "Throughput (ops/sec)");
    for result in results {
        header.push_str(&format!(
            " {:>15}",
            result
                .indexer_name
                .split_whitespace()
                .next()
                .unwrap_or(&result.indexer_name)
        ));
    }
    println!("{}", header);
    println!("{}", "-".repeat(30 + results.len() * 16));

    let mut row = format!("{:<30}", "Find matches (hit)");
    for result in results {
        if let Some(stats) = &result.find_matches_hit_stats {
            row.push_str(&format!(" {:>15.0}", stats.throughput_ops_sec));
        } else {
            row.push_str(&format!(" {:>15}", "-"));
        }
    }
    println!("{}", row);
}

async fn run_microbench_mode(args: MicrobenchArgs) {
    let num_sequences = args.common.size / args.common.depth;
    if num_sequences == 0 {
        eprintln!("Error: size must be >= depth");
        std::process::exit(1);
    }

    println!("KvIndexer Microbenchmark");
    println!("========================\n");
    println!("Configuration:");
    println!("  Target size: {} (worker, block) pairs", args.common.size);
    println!(
        "  Depth: {} blocks/sequence (= {} tokens with block_size={})",
        args.common.depth,
        args.common.depth * args.common.block_size as usize,
        args.common.block_size
    );
    println!("  Block size: {} tokens", args.common.block_size);
    println!("  Workers: {}", args.common.num_workers);
    println!("  Iterations: {}", args.iterations);
    println!(
        "  Prefix prompt ratio: {:.1}%",
        args.prefix_prompt_ratio * 100.0
    );
    println!("  Prefix prompt groups: {}", args.num_prefix_prompts);
    println!("  Num shards (for sharded): {}", args.num_shards);
    println!("  Indexer type: {:?}", args.indexer_type);
    println!("  Benchmark type: {}", args.benchmark_type);
    println!(
        "\n  Derived: {} sequences to reach target size",
        num_sequences
    );

    // Generate sequences
    let extra_count = args.iterations;
    let all_sequences = generate_sequences(
        num_sequences + extra_count,
        args.common.depth,
        args.common.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.common.seed,
        false,
    );
    let sequences = &all_sequences[..num_sequences];
    let extra_sequences = &all_sequences[num_sequences..];

    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let mut results = Vec::new();

    // Benchmark single indexer
    if matches!(args.indexer_type, IndexerType::Single | IndexerType::All) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexer::new(token.clone(), args.common.block_size, metrics.clone());
        let result = run_microbenchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        token.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Benchmark sharded indexer
    if matches!(args.indexer_type, IndexerType::Sharded | IndexerType::All) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexerSharded::new(
            token.clone(),
            args.num_shards,
            args.common.block_size,
            metrics.clone(),
        );
        let result = run_microbenchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        token.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Benchmark nested indexer
    if matches!(args.indexer_type, IndexerType::Nested | IndexerType::All) {
        let mut indexer = ThreadPoolIndexer::new(
            PositionalIndexer::new(args.jump_size),
            args.num_shards,
            args.common.block_size,
        );
        let result = run_microbenchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        indexer.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Benchmark concurrent radix tree indexer
    if matches!(
        args.indexer_type,
        IndexerType::Concurrent | IndexerType::All
    ) {
        let mut indexer = ThreadPoolIndexer::new(
            ConcurrentRadixTree::new(),
            args.num_shards,
            args.common.block_size,
        );
        let result = run_microbenchmarks(&mut indexer, sequences, extra_sequences, &args).await;
        results.push(result);
        indexer.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Print results
    if args.format == "csv" {
        MicrobenchResults::print_csv_header();
        for result in &results {
            result.print_csv_row();
        }
    } else {
        for result in &results {
            result.print(args.common.depth);
        }

        if results.len() >= 2 {
            print_microbench_comparison(&results, args.common.depth);
        }
    }

    println!("\nMicrobenchmark complete.");
}

// ============================================================================
// Stress Test Mode
// ============================================================================

/// Result of a single request during stress test
#[allow(dead_code)]
struct RequestResult {
    request_id: u64,
    submit_time: Instant,
    complete_time: Instant,
    success: bool,
}

/// Aggregated results from stress test
struct StressResults {
    indexer_name: String,
    submitted: u64,
    completed: u64,
    timed_out: u64,
    latencies: Vec<Duration>,
    max_in_flight: u64,
    baseline_service_time: Duration,
    construction_time: Duration,
    construction_events: u64,
}

/// Run the stress test with a generic indexer
async fn run_stress_test<I: BenchableIndexer + 'static>(
    indexer: Arc<I>,
    sequences: &[SequenceData],
    args: &StressArgs,
) -> StressResults {
    let indexer_name = indexer.name().to_string();

    // Phase 2: Baseline Measurement
    println!("\nPhase 2: Baseline Measurement");
    println!("  Running 10 sequential find_matches calls...");

    let mut baseline_durations = Vec::new();
    for seq in sequences.iter().take(10) {
        let start = Instant::now();
        let _ = indexer.find_matches(seq.local_hashes.clone()).await;
        baseline_durations.push(start.elapsed());
    }
    let stats = LatencyStats::from_durations(&baseline_durations).unwrap();
    let baseline_service_time = stats.p50;
    let theoretical_max = stats.throughput_ops_sec;

    println!(
        "  Baseline find_matches latency: {:?} (p50 of 10)",
        baseline_service_time
    );
    println!(
        "  Theoretical max throughput: {:.1} req/sec",
        theoretical_max
    );

    // Phase 3: Pre-generate Lookup Sequences
    println!("\nPhase 3: Pre-generating Lookup Sequences");
    let expected_requests = (args.arrival_rate * args.duration as f64).ceil() as usize + 100;

    // Phase 4: Stress Test
    println!("\nPhase 4: Stress Test");
    println!("  Arrival rate: {:.1} req/sec", args.arrival_rate);
    println!("  Duration: {}s", args.duration);

    let in_flight = Arc::new(AtomicU64::new(0));
    let max_in_flight = Arc::new(AtomicU64::new(0));
    let (result_tx, mut result_rx) = mpsc::channel::<RequestResult>(expected_requests);

    let start = Instant::now();
    let mut request_id = 0u64;
    let mut interval = interval(Duration::from_secs_f64(1.0 / args.arrival_rate));

    while start.elapsed() < Duration::from_secs(args.duration) {
        let seq = sequences[request_id as usize % sequences.len()]
            .local_hashes
            .clone();

        // Track in-flight
        let current = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        max_in_flight.fetch_max(current, Ordering::Relaxed);

        let indexer = Arc::clone(&indexer);
        let result_tx = result_tx.clone();
        let in_flight_clone = in_flight.clone();
        let req_id = request_id;
        let verbose = args.common.verbose;

        tokio::spawn(async move {
            let submit_time = Instant::now();
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
        interval.tick().await;
    }

    let submitted = request_id;
    println!("  Submitted {} requests", submitted);

    // Wait for in-flight requests with timeout
    println!("\nPhase 5: Draining In-flight Requests");
    let drain_start = Instant::now();
    let mut last_in_flight = in_flight.load(Ordering::Relaxed);
    println!(
        "  Waiting for {} in-flight requests (timeout: {}s)...",
        last_in_flight, args.in_flight_timeout
    );

    while in_flight.load(Ordering::Relaxed) > 0
        && drain_start.elapsed() < Duration::from_secs(args.in_flight_timeout)
    {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let current = in_flight.load(Ordering::Relaxed);
        if current != last_in_flight && args.common.verbose {
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
    if timed_out > 0 {
        result_rx.close();
    }
    let mut results = Vec::new();
    while let Some(r) = result_rx.recv().await {
        results.push(r);
    }

    // Compute latencies
    let latencies: Vec<Duration> = results
        .iter()
        .map(|r| r.complete_time.duration_since(r.submit_time))
        .collect();

    StressResults {
        indexer_name,
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

/// Print the final stress test results report
fn print_stress_results(args: &StressArgs, results: &StressResults) {
    let num_sequences = args.common.size / args.common.depth;

    println!("\n=====================");
    println!("Queue Saturation Test Results: {}", results.indexer_name);
    println!("=====================\n");

    println!("Configuration:");
    println!(
        "  Tree size: {} blocks ({} sequences x {} depth)",
        args.common.size, num_sequences, args.common.depth
    );
    println!("  Workers: {}", args.common.num_workers);
    println!(
        "  Prefix share ratio: {:.1}%",
        args.prefix_share_ratio * 100.0
    );
    println!("  Arrival rate: {:.1} req/sec", args.arrival_rate);
    println!("  Duration: {}s", args.duration);
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
        let test_duration = args.duration as f64 + args.in_flight_timeout as f64;
        let achieved_throughput = results.completed as f64 / test_duration;

        println!("  Throughput:");
        println!("    Requested: {:.1} req/sec", args.arrival_rate);
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
            let is_saturated = achieved_throughput < args.arrival_rate * 0.95
                || results.timed_out > 0
                || stats.p50 > results.baseline_service_time * 2;

            if is_saturated {
                println!("  STATUS: SATURATED");
                if achieved_throughput < args.arrival_rate * 0.95 {
                    println!(
                        "    - Throughput ({:.1}) < Arrival rate ({:.1})",
                        achieved_throughput, args.arrival_rate
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

fn print_stress_comparison(results: &[StressResults], args: &StressArgs) {
    if results.len() < 2 {
        return;
    }

    println!("\n========================================");
    println!("STRESS TEST COMPARISON SUMMARY");
    println!("========================================\n");

    // Build dynamic column headers
    let mut header = format!("{:<35}", "Metric");
    for result in results {
        let short_name = result
            .indexer_name
            .split_whitespace()
            .next()
            .unwrap_or(&result.indexer_name);
        header.push_str(&format!(" {:>18}", short_name));
    }
    println!("{}", header);
    println!("{}", "-".repeat(35 + results.len() * 19));

    // Construction time
    let mut row = format!("{:<35}", "Construction time (ms)");
    for result in results {
        row.push_str(&format!(
            " {:>18.2}",
            result.construction_time.as_secs_f64() * 1000.0
        ));
    }
    println!("{}", row);

    // Baseline service time
    let mut row = format!("{:<35}", "Baseline service time (us)");
    for result in results {
        row.push_str(&format!(
            " {:>18.2}",
            result.baseline_service_time.as_nanos() as f64 / 1000.0
        ));
    }
    println!("{}", row);

    // Completed requests
    let mut row = format!("{:<35}", "Completed requests");
    for result in results {
        row.push_str(&format!(" {:>18}", result.completed));
    }
    println!("{}", row);

    // Max in-flight
    let mut row = format!("{:<35}", "Max in-flight");
    for result in results {
        row.push_str(&format!(" {:>18}", result.max_in_flight));
    }
    println!("{}", row);

    // Timed out
    let mut row = format!("{:<35}", "Timed out");
    for result in results {
        row.push_str(&format!(" {:>18}", result.timed_out));
    }
    println!("{}", row);

    // Latency p50
    let mut row = format!("{:<35}", "Latency p50 (us)");
    for result in results {
        if let Some(stats) = LatencyStats::from_durations(&result.latencies) {
            row.push_str(&format!(" {:>18.2}", stats.p50.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>18}", "-"));
        }
    }
    println!("{}", row);

    // Latency p99
    let mut row = format!("{:<35}", "Latency p99 (us)");
    for result in results {
        if let Some(stats) = LatencyStats::from_durations(&result.latencies) {
            row.push_str(&format!(" {:>18.2}", stats.p99.as_nanos() as f64 / 1000.0));
        } else {
            row.push_str(&format!(" {:>18}", "-"));
        }
    }
    println!("{}", row);

    // Achieved throughput
    let test_duration = args.duration as f64 + args.in_flight_timeout as f64;
    let mut row = format!("{:<35}", "Achieved throughput (req/s)");
    for result in results {
        let throughput = result.completed as f64 / test_duration;
        row.push_str(&format!(" {:>18.1}", throughput));
    }
    println!("{}", row);
}

async fn run_stress_mode(args: StressArgs) {
    // Validate inputs before proceeding
    if args.common.depth == 0 {
        eprintln!("Error: depth must be > 0");
        std::process::exit(1);
    }
    if args.common.num_workers == 0 {
        eprintln!("Error: num_workers must be > 0");
        std::process::exit(1);
    }
    if args.common.size < args.common.depth {
        eprintln!(
            "Error: size ({}) must be >= depth ({})",
            args.common.size, args.common.depth
        );
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&args.prefix_share_ratio) {
        eprintln!(
            "Error: prefix_share_ratio ({}) must be in range 0.0..=1.0",
            args.prefix_share_ratio
        );
        std::process::exit(1);
    }
    if args.arrival_rate <= 0.0 {
        eprintln!("Error: arrival_rate must be > 0.0");
        std::process::exit(1);
    }
    if matches!(
        args.indexer_type,
        IndexerType::Sharded | IndexerType::Nested | IndexerType::All
    ) && args.num_shards == 0
    {
        eprintln!("Error: num_shards must be > 0 when using Sharded, Nested, or All indexer type");
        std::process::exit(1);
    }

    let num_sequences = args.common.size / args.common.depth;

    println!("Queue Saturation Stress Test");
    println!("============================\n");

    println!("Configuration:");
    println!(
        "  Tree size: {} blocks ({} sequences x {} depth)",
        args.common.size, num_sequences, args.common.depth
    );
    println!("  Workers: {}", args.common.num_workers);
    println!("  Block size: {} tokens", args.common.block_size);
    println!(
        "  Prefix share ratio: {:.1}%",
        args.prefix_share_ratio * 100.0
    );
    println!("  Seed: {}", args.common.seed);
    println!("  Arrival rate: {:.1} req/sec", args.arrival_rate);
    println!("  Duration: {}s", args.duration);
    println!("  In-flight timeout: {}s", args.in_flight_timeout);
    println!("  Indexer type: {:?}", args.indexer_type);
    if matches!(args.indexer_type, IndexerType::Sharded | IndexerType::All) {
        println!("  Num shards (sharded): {}", args.num_shards);
    }
    if matches!(args.indexer_type, IndexerType::Nested | IndexerType::All) {
        println!("  Num workers (nested): {}", args.num_shards);
        println!("  Jump size (nested): {}", args.jump_size);
    }

    // Generate sequences
    println!("\nPhase 1: Tree Construction");
    println!("  Generating {} sequences...", num_sequences);

    // Use prefix_share_ratio as prefix_ratio and 1 group for stress test
    let sequences = generate_sequences(
        num_sequences,
        args.common.depth,
        args.common.num_workers,
        args.prefix_share_ratio,
        1, // Single prefix group for stress test
        args.common.seed,
        false, // use_cumulative_hash
    );

    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let mut all_results = Vec::new();

    // Test single indexer
    if matches!(args.indexer_type, IndexerType::Single | IndexerType::All) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexer::new(token.clone(), args.common.block_size, metrics.clone());

        println!(
            "\n  Applying {} store events to KvIndexer...",
            sequences.len()
        );
        let construction_start = Instant::now();

        for (event_id, seq) in sequences.iter().enumerate() {
            let event = seq.to_store_event(event_id as u64);
            KvIndexerInterface::apply_event(&mut indexer, event).await;

            if args.common.verbose && (event_id + 1) % 100 == 0 {
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

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut results = run_stress_test(Arc::new(indexer), &sequences, &args).await;
        results.construction_time = construction_time;
        results.construction_events = construction_events;

        print_stress_results(&args, &results);
        all_results.push(results);

        token.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Test sharded indexer
    if matches!(args.indexer_type, IndexerType::Sharded | IndexerType::All) {
        let token = CancellationToken::new();
        let mut indexer = KvIndexerSharded::new(
            token.clone(),
            args.num_shards,
            args.common.block_size,
            metrics.clone(),
        );

        println!(
            "\n  Applying {} store events to KvIndexerSharded...",
            sequences.len()
        );
        let construction_start = Instant::now();

        for (event_id, seq) in sequences.iter().enumerate() {
            let event = seq.to_store_event(event_id as u64);
            KvIndexerInterface::apply_event(&mut indexer, event).await;

            if args.common.verbose && (event_id + 1) % 100 == 0 {
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

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut results = run_stress_test(Arc::new(indexer), &sequences, &args).await;
        results.construction_time = construction_time;
        results.construction_events = construction_events;

        print_stress_results(&args, &results);
        all_results.push(results);

        token.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Test nested indexer
    if matches!(args.indexer_type, IndexerType::Nested | IndexerType::All) {
        let indexer = ThreadPoolIndexer::new(
            PositionalIndexer::new(args.jump_size),
            args.num_shards,
            args.common.block_size,
        );

        println!(
            "\n  Applying {} store events to PositionalIndexer...",
            sequences.len()
        );
        let construction_start = Instant::now();

        for (event_id, seq) in sequences.iter().enumerate() {
            let event = seq.to_store_event(event_id as u64);
            indexer.apply_event(event).await;

            if args.common.verbose && (event_id + 1) % 100 == 0 {
                println!("    Applied {}/{} events...", event_id + 1, sequences.len());
            }
        }

        indexer.flush().await;

        let construction_time = construction_start.elapsed();
        let construction_events = sequences.len() as u64;

        println!("  Tree construction completed in {:?}", construction_time);
        println!(
            "  Throughput: {:.0} events/sec",
            construction_events as f64 / construction_time.as_secs_f64()
        );

        tokio::time::sleep(Duration::from_millis(100)).await;

        let indexer = Arc::new(indexer);

        let mut results = run_stress_test(indexer.clone(), &sequences, &args).await;
        results.construction_time = construction_time;
        results.construction_events = construction_events;

        print_stress_results(&args, &results);
        all_results.push(results);

        indexer.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Test concurrent radix tree indexer
    if matches!(
        args.indexer_type,
        IndexerType::Concurrent | IndexerType::All
    ) {
        let indexer = ThreadPoolIndexer::new(
            ConcurrentRadixTree::new(),
            args.num_shards,
            args.common.block_size,
        );

        println!(
            "\n  Applying {} store events to ConcurrentRadixTree...",
            sequences.len()
        );
        let construction_start = Instant::now();

        for (event_id, seq) in sequences.iter().enumerate() {
            let event = seq.to_store_event(event_id as u64);
            indexer.apply_event(event).await;

            if args.common.verbose && (event_id + 1) % 100 == 0 {
                println!("    Applied {}/{} events...", event_id + 1, sequences.len());
            }
        }

        indexer.flush().await;

        let construction_time = construction_start.elapsed();
        let construction_events = sequences.len() as u64;

        println!("  Tree construction completed in {:?}", construction_time);
        println!(
            "  Throughput: {:.0} events/sec",
            construction_events as f64 / construction_time.as_secs_f64()
        );

        tokio::time::sleep(Duration::from_millis(100)).await;

        let indexer = Arc::new(indexer);

        let mut results = run_stress_test(indexer.clone(), &sequences, &args).await;
        results.construction_time = construction_time;
        results.construction_events = construction_events;

        print_stress_results(&args, &results);
        all_results.push(results);

        indexer.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Print comparison if multiple were run
    if all_results.len() >= 2 {
        print_stress_comparison(&all_results, &args);
    }

    println!("\nStress test complete.");
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Microbench(args) => run_microbench_mode(args).await,
        Command::Stress(args) => run_stress_mode(args).await,
    }
}
