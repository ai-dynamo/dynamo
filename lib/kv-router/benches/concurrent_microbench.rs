// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concurrent radix tree benchmark for measuring thread scaling and overhead.
//!
//! Measures:
//! - Phase 1: Single-threaded overhead of ConcurrentRadixTree vs RadixTree
//! - Phase 2: Thread scaling and comparison with RadixTree baseline
//!
//! Run with: cargo bench --package dynamo-kv-router --bench concurrent_microbench --features bench -- --help

use clap::{Parser, ValueEnum};
use dynamo_kv_router::{
    ConcurrentRadixTree, RadixTree, RouterEvent,
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, WorkerId, compute_seq_hash_for_block,
    },
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Query distribution mode for concurrent benchmarks
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum QueryMode {
    /// Each thread uses a different subset of sequences (no overlap)
    Disjoint,
    /// All threads query the same sequences (maximum contention on same paths)
    Shared,
    /// Random selection from the full set (realistic workload)
    Random,
}

/// Preset configurations for common benchmark scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Preset {
    /// Small tree with microsecond lookups
    Small,
    /// Large tree with millisecond lookups
    Large,
}

impl Preset {
    fn apply(&self, args: &mut Args) {
        match self {
            Preset::Small => {
                args.size = 10_000;
                args.depth = 32;
                args.num_prefix_prompts = 4;
            }
            Preset::Large => {
                args.size = 15_500_000;
                args.num_workers = 1000;
                args.depth = 2000;
                args.num_prefix_prompts = 100;
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "concurrent_microbench")]
#[command(about = "Concurrent radix tree benchmark for thread scaling")]
struct Args {
    /// Ignored: passed by cargo bench harness
    #[arg(long, hide = true)]
    bench: bool,

    /// Use preset configuration (small=us lookups, large=ms lookups)
    #[arg(long, value_enum)]
    preset: Option<Preset>,

    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "10000")]
    size: usize,

    /// Sequence depth in blocks
    #[arg(long, default_value = "32")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Prefix prompt ratio (0.0 to 1.0)
    #[arg(long, default_value = "0.25")]
    prefix_prompt_ratio: f64,

    /// Number of unique prefix prompt groups
    #[arg(long, default_value = "4")]
    num_prefix_prompts: usize,

    /// Number of concurrent reader threads (default: num_cpus)
    #[arg(long)]
    threads: Option<usize>,

    /// Duration in seconds for each benchmark phase
    #[arg(long, default_value = "5")]
    duration_secs: u64,

    /// Warmup duration in seconds before measurement
    #[arg(long, default_value = "1")]
    warmup_secs: u64,

    /// Query distribution mode
    #[arg(long, value_enum, default_value = "random")]
    query_mode: QueryMode,

    /// Run thread sweep with counts [1, 2, 4, 8, 16, ...] up to num_cpus
    #[arg(long)]
    thread_sweep: bool,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// KV block size in tokens
    #[arg(long, default_value = "16")]
    block_size: u32,

    /// Run only specific benchmark phase (overhead, scaling, or all)
    #[arg(long, default_value = "all")]
    phase: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Write ratio for mixed workload testing (0.0 = 100% reads, 0.05 = 95/5 read/write, 1.0 = 100% writes)
    #[arg(long, default_value = "0.0")]
    write_ratio: f64,
}


/// Pre-generated sequence data for benchmarking
#[derive(Clone)]
struct SequenceData {
    worker_id: WorkerId,
    local_hashes: Vec<LocalBlockHash>,
    external_hashes: Vec<ExternalSequenceBlockHash>,
}

impl SequenceData {
    fn from_local_hashes(worker_id: WorkerId, local_hashes: Vec<LocalBlockHash>) -> Self {
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

/// Generate sequences with shared prefix prompts
fn generate_sequences(
    num_sequences: usize,
    depth: usize,
    num_workers: usize,
    prefix_prompt_ratio: f64,
    num_prefix_prompts: usize,
    seed: u64,
) -> Vec<SequenceData> {
    let mut sequences = Vec::with_capacity(num_sequences);
    let prefix_length = (depth as f64 * prefix_prompt_ratio).round() as usize;
    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    for seq_id in 0..num_sequences {
        let seq_id_u64 = seq_id as u64;
        let worker_id = (seq_id % num_workers) as WorkerId;

        let group_id = if num_prefix_prompts > 0 && prefix_length > 0 {
            Some(rng.random_range(0..num_prefix_prompts) as u64)
        } else {
            None
        };

        let local_hashes: Vec<LocalBlockHash> = (0..depth)
            .map(|block_idx| {
                let block_idx_u64 = block_idx as u64;
                if let Some(gid) = group_id {
                    if block_idx < prefix_length {
                        return LocalBlockHash(0xDEAD_BEEF_0000_0000 | (gid << 32) | block_idx_u64);
                    }
                }
                LocalBlockHash((seq_id_u64 << 32) | block_idx_u64)
            })
            .collect();

        sequences.push(SequenceData::from_local_hashes(worker_id, local_hashes));
    }

    sequences
}

/// Latency statistics from benchmark runs
#[derive(Debug, Clone)]
struct LatencyStats {
    min: Duration,
    max: Duration,
    avg: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    total_ops: usize,
    total_duration: Duration,
    throughput_ops_sec: f64,
}

impl LatencyStats {
    fn from_durations(mut durations: Vec<Duration>, total_duration: Duration) -> Self {
        if durations.is_empty() {
            return Self {
                min: Duration::ZERO,
                max: Duration::ZERO,
                avg: Duration::ZERO,
                p50: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                total_ops: 0,
                total_duration,
                throughput_ops_sec: 0.0,
            };
        }

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
            total_ops: n,
            total_duration,
            throughput_ops_sec: n as f64 / total_duration.as_secs_f64(),
        }
    }

    fn print(&self, label: &str) {
        println!("\n{} Statistics:", label);
        println!("  Total ops:    {}", self.total_ops);
        println!("  Duration:     {:?}", self.total_duration);
        println!("  Throughput:   {:.2} ops/sec", self.throughput_ops_sec);
        println!("  Latency:");
        println!("    min:  {:>12?}", self.min);
        println!("    avg:  {:>12?}", self.avg);
        println!("    p50:  {:>12?}", self.p50);
        println!("    p95:  {:>12?}", self.p95);
        println!("    p99:  {:>12?}", self.p99);
        println!("    max:  {:>12?}", self.max);
    }
}

/// Result from a single thread's benchmark run
struct ThreadResult {
    latencies: Vec<Duration>,
    ops_count: usize,
}

/// Result from a single thread's mixed workload benchmark run
struct MixedThreadResult {
    read_latencies: Vec<Duration>,
    write_latencies: Vec<Duration>,
    read_count: usize,
    write_count: usize,
}

/// Aggregated results from concurrent benchmark
struct ConcurrentResult {
    throughput_ops_sec: f64,
    latency_stats: LatencyStats,
    num_threads: usize,
}

/// Aggregated results from mixed workload concurrent benchmark
struct MixedWorkloadResult {
    total_throughput_ops_sec: f64,
    read_throughput_ops_sec: f64,
    write_throughput_ops_sec: f64,
    read_latency_stats: LatencyStats,
    write_latency_stats: LatencyStats,
    num_threads: usize,
    write_ratio: f64,
    actual_write_ratio: f64,
}

/// Build a ConcurrentRadixTree from sequences
fn build_concurrent_tree(sequences: &[SequenceData]) -> ConcurrentRadixTree {
    let tree = ConcurrentRadixTree::new();
    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_store_event(event_id as u64);
        let _ = tree.apply_event(event);
    }
    tree
}

/// Build a RadixTree from sequences
fn build_radix_tree(sequences: &[SequenceData]) -> RadixTree {
    let mut tree = RadixTree::new();
    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_store_event(event_id as u64);
        let _ = tree.apply_event(event);
    }
    tree
}

/// Run single-threaded benchmark (works for both tree types via closure)
fn bench_single_threaded<F>(
    sequences: &[SequenceData],
    duration: Duration,
    warmup: Duration,
    mut find_matches: F,
) -> LatencyStats
where
    F: FnMut(&[LocalBlockHash]),
{
    let mut rng = StdRng::seed_from_u64(12345);

    // Warmup
    let warmup_start = Instant::now();
    while warmup_start.elapsed() < warmup {
        let seq = &sequences[rng.random_range(0..sequences.len())];
        find_matches(&seq.local_hashes);
    }

    // Measurement
    let mut latencies = Vec::with_capacity(100_000);
    let start = Instant::now();
    while start.elapsed() < duration {
        let seq = &sequences[rng.random_range(0..sequences.len())];
        let op_start = Instant::now();
        find_matches(&seq.local_hashes);
        latencies.push(op_start.elapsed());
    }
    let total_duration = start.elapsed();

    LatencyStats::from_durations(latencies, total_duration)
}

/// Get query sequence for a thread based on query mode
fn get_query_for_thread(
    thread_id: usize,
    iteration: usize,
    sequences: &[SequenceData],
    query_mode: QueryMode,
    num_threads: usize,
    rng: &mut StdRng,
) -> Vec<LocalBlockHash> {
    let seq_idx = match query_mode {
        QueryMode::Disjoint => {
            let chunk_size = sequences.len() / num_threads;
            let start = thread_id * chunk_size;
            let end = if thread_id == num_threads - 1 {
                sequences.len()
            } else {
                start + chunk_size
            };
            start + (iteration % (end - start).max(1))
        }
        QueryMode::Shared => iteration % sequences.len(),
        QueryMode::Random => rng.random_range(0..sequences.len()),
    };

    sequences[seq_idx].local_hashes.clone()
}

/// Run concurrent benchmark on ConcurrentRadixTree
fn bench_concurrent(
    tree: &Arc<ConcurrentRadixTree>,
    sequences: &Arc<Vec<SequenceData>>,
    num_threads: usize,
    duration: Duration,
    warmup: Duration,
    query_mode: QueryMode,
    seed: u64,
) -> ConcurrentResult {
    let barrier = Arc::new(Barrier::new(num_threads));

    let results: Vec<ThreadResult> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let tree = tree.clone();
                let sequences = sequences.clone();
                let barrier = barrier.clone();

                s.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed + thread_id as u64);
                    let mut latencies = Vec::with_capacity(50_000);

                    barrier.wait();

                    // Warmup
                    let warmup_start = Instant::now();
                    let mut warmup_iter = 0;
                    while warmup_start.elapsed() < warmup {
                        let query = get_query_for_thread(
                            thread_id, warmup_iter, &sequences, query_mode, num_threads, &mut rng,
                        );
                        let _ = tree.find_matches(query, false);
                        warmup_iter += 1;
                    }

                    barrier.wait();

                    // Measurement
                    let start = Instant::now();
                    let mut iter = 0;
                    while start.elapsed() < duration {
                        let query = get_query_for_thread(
                            thread_id, iter, &sequences, query_mode, num_threads, &mut rng,
                        );
                        let op_start = Instant::now();
                        let _ = tree.find_matches(query, false);
                        latencies.push(op_start.elapsed());
                        iter += 1;
                    }

                    ThreadResult { latencies, ops_count: iter }
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let total_ops: usize = results.iter().map(|r| r.ops_count).sum();
    let all_latencies: Vec<Duration> = results.into_iter().flat_map(|r| r.latencies).collect();
    let latency_stats = LatencyStats::from_durations(all_latencies, duration);

    ConcurrentResult {
        throughput_ops_sec: total_ops as f64 / duration.as_secs_f64(),
        latency_stats,
        num_threads,
    }
}

/// Run mixed workload benchmark with configurable read/write ratio
fn bench_mixed_workload(
    tree: &Arc<ConcurrentRadixTree>,
    read_sequences: &Arc<Vec<SequenceData>>,
    write_sequences: &Arc<Vec<SequenceData>>,
    num_threads: usize,
    duration: Duration,
    warmup: Duration,
    query_mode: QueryMode,
    write_ratio: f64,
    seed: u64,
) -> MixedWorkloadResult {
    use std::sync::atomic::{AtomicU64, Ordering};

    let barrier = Arc::new(Barrier::new(num_threads));
    let write_event_counter = Arc::new(AtomicU64::new(0));

    let results: Vec<MixedThreadResult> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let tree = tree.clone();
                let read_sequences = read_sequences.clone();
                let write_sequences = write_sequences.clone();
                let barrier = barrier.clone();
                let write_event_counter = write_event_counter.clone();

                s.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed + thread_id as u64);
                    let mut read_latencies = Vec::with_capacity(50_000);
                    let mut write_latencies = Vec::with_capacity(5_000);

                    barrier.wait();

                    // Warmup
                    let warmup_start = Instant::now();
                    let mut warmup_iter = 0;
                    while warmup_start.elapsed() < warmup {
                        let is_write = rng.random::<f64>() < write_ratio;
                        if is_write && !write_sequences.is_empty() {
                            let seq_idx = rng.random_range(0..write_sequences.len());
                            let event_id = write_event_counter.fetch_add(1, Ordering::Relaxed);
                            let event = write_sequences[seq_idx].to_store_event(event_id + 1_000_000);
                            let _ = tree.apply_event(event);
                        } else {
                            let query = get_query_for_thread(
                                thread_id, warmup_iter, &read_sequences, query_mode, num_threads, &mut rng,
                            );
                            let _ = tree.find_matches(query, false);
                        }
                        warmup_iter += 1;
                    }

                    barrier.wait();

                    // Measurement
                    let start = Instant::now();
                    let mut read_count = 0;
                    let mut write_count = 0;
                    let mut iter = 0;
                    while start.elapsed() < duration {
                        let is_write = rng.random::<f64>() < write_ratio;
                        let op_start = Instant::now();

                        if is_write && !write_sequences.is_empty() {
                            let seq_idx = rng.random_range(0..write_sequences.len());
                            let event_id = write_event_counter.fetch_add(1, Ordering::Relaxed);
                            let event = write_sequences[seq_idx].to_store_event(event_id + 1_000_000);
                            let _ = tree.apply_event(event);
                            write_latencies.push(op_start.elapsed());
                            write_count += 1;
                        } else {
                            let query = get_query_for_thread(
                                thread_id, iter, &read_sequences, query_mode, num_threads, &mut rng,
                            );
                            let _ = tree.find_matches(query, false);
                            read_latencies.push(op_start.elapsed());
                            read_count += 1;
                        }
                        iter += 1;
                    }

                    MixedThreadResult {
                        read_latencies,
                        write_latencies,
                        read_count,
                        write_count,
                    }
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let total_reads: usize = results.iter().map(|r| r.read_count).sum();
    let total_writes: usize = results.iter().map(|r| r.write_count).sum();
    let total_ops = total_reads + total_writes;

    let all_read_latencies: Vec<Duration> = results.iter().flat_map(|r| r.read_latencies.clone()).collect();
    let all_write_latencies: Vec<Duration> = results.into_iter().flat_map(|r| r.write_latencies).collect();

    let read_latency_stats = LatencyStats::from_durations(all_read_latencies, duration);
    let write_latency_stats = LatencyStats::from_durations(all_write_latencies, duration);

    let actual_write_ratio = if total_ops > 0 {
        total_writes as f64 / total_ops as f64
    } else {
        0.0
    };

    MixedWorkloadResult {
        total_throughput_ops_sec: total_ops as f64 / duration.as_secs_f64(),
        read_throughput_ops_sec: total_reads as f64 / duration.as_secs_f64(),
        write_throughput_ops_sec: total_writes as f64 / duration.as_secs_f64(),
        read_latency_stats,
        write_latency_stats,
        num_threads,
        write_ratio,
        actual_write_ratio,
    }
}

/// Generate thread counts for sweep: [1, 2, 4, 8, 16, ...] up to num_cpus * 2
fn get_thread_counts(args: &Args, num_cpus: usize) -> Vec<usize> {
    if args.thread_sweep {
        // No, step_by(2) increments by 2, producing [1, 3, 5, ...], not powers of two.
        // To double each time, use a sequence of powers of two, e.g. [1, 2, 4, 8, ...].
        let mut counts = vec![];
        let mut v = 1;
        while v <= num_cpus * 2 {
            counts.push(v);
            v *= 2;
        }
        counts
    } else {
        vec![args.threads.unwrap_or(num_cpus)]
    }
}

/// Phase 1: Measure overhead of ConcurrentRadixTree vs RadixTree (single-threaded)
fn phase_overhead(args: &Args, sequences: &[SequenceData]) {
    println!("\n{}", "=".repeat(60));
    println!("Phase 1: Overhead Measurement (Single-Threaded)");
    println!("{}", "=".repeat(60));

    let duration = Duration::from_secs(args.duration_secs);
    let warmup = Duration::from_secs(args.warmup_secs);

    // Build trees
    print!("Building RadixTree... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let radix_tree = build_radix_tree(sequences);
    println!("done ({} blocks)", radix_tree.current_size());

    print!("Building ConcurrentRadixTree... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let concurrent_tree = build_concurrent_tree(sequences);
    println!("done ({} blocks)", concurrent_tree.current_size());

    // Benchmark RadixTree
    println!("\nBenchmarking RadixTree (single-threaded)...");
    let radix_stats = bench_single_threaded(sequences, duration, warmup, |hashes| {
        let _ = radix_tree.find_matches(hashes.to_vec(), false);
    });
    radix_stats.print("RadixTree");

    // Benchmark ConcurrentRadixTree
    println!("\nBenchmarking ConcurrentRadixTree (single-threaded)...");
    let concurrent_stats = bench_single_threaded(sequences, duration, warmup, |hashes| {
        let _ = concurrent_tree.find_matches(hashes.to_vec(), false);
    });
    concurrent_stats.print("ConcurrentRadixTree");

    // Comparison
    println!("\n--- Overhead Analysis ---");
    let latency_ratio = concurrent_stats.avg.as_nanos() as f64 / radix_stats.avg.as_nanos() as f64;
    let throughput_ratio = radix_stats.throughput_ops_sec / concurrent_stats.throughput_ops_sec;

    println!(
        "  Latency overhead:     {:.1}% (ConcurrentRadixTree is {:.2}x slower)",
        (latency_ratio - 1.0) * 100.0,
        latency_ratio
    );
    println!(
        "  Throughput overhead:  {:.1}% (ConcurrentRadixTree has {:.2}x lower throughput)",
        (throughput_ratio - 1.0) * 100.0,
        throughput_ratio
    );
}

/// Phase 2: Thread scaling with RadixTree baseline comparison
/// If write_ratio > 0, uses mixed workload benchmark; otherwise read-only.
fn phase_scaling(args: &Args, read_sequences: &[SequenceData], write_sequences: &[SequenceData]) {
    println!("\n{}", "=".repeat(60));
    println!("Phase 2: Thread Scaling");
    println!("{}", "=".repeat(60));

    let duration = Duration::from_secs(args.duration_secs);
    let warmup = Duration::from_secs(args.warmup_secs);
    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let thread_counts = get_thread_counts(args, num_cpus);
    let use_mixed = args.write_ratio > 0.0 && !write_sequences.is_empty();

    // Build RadixTree for baseline
    print!("Building RadixTree... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let radix_tree = build_radix_tree(read_sequences);
    println!("done ({} blocks)", radix_tree.current_size());

    // Build ConcurrentRadixTree
    print!("Building ConcurrentRadixTree... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let concurrent_tree = Arc::new(build_concurrent_tree(read_sequences));
    println!("done ({} blocks)", concurrent_tree.current_size());

    let read_sequences_arc = Arc::new(read_sequences.to_vec());
    let write_sequences_arc = Arc::new(write_sequences.to_vec());

    println!("\nThread counts: {:?}", thread_counts);
    println!("Query mode: {:?}", args.query_mode);
    if use_mixed {
        println!("Write ratio: {:.1}%", args.write_ratio * 100.0);
    }
    println!("Duration per point: {}s", args.duration_secs);

    // Get RadixTree baseline (single-threaded, read-only)
    print!("\nBenchmarking RadixTree baseline... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let radix_stats = bench_single_threaded(read_sequences, duration, warmup, |hashes| {
        let _ = radix_tree.find_matches(hashes.to_vec(), false);
    });
    println!("{:.2} ops/sec", radix_stats.throughput_ops_sec);

    // Run concurrent benchmarks
    println!("\n--- Running thread scaling benchmark ---");

    if use_mixed {
        let mut results: Vec<MixedWorkloadResult> = Vec::new();

        for &num_threads in &thread_counts {
            // Rebuild tree for each thread count to start fresh
            let tree = Arc::new(build_concurrent_tree(read_sequences));

            print!("  threads={:>3}... ", num_threads);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let result = bench_mixed_workload(
                &tree,
                &read_sequences_arc,
                &write_sequences_arc,
                num_threads,
                duration,
                warmup,
                args.query_mode,
                args.write_ratio,
                args.seed,
            );

            println!(
                "total={:>10.2} ops/s, r_p50={:>10?}, w_p50={:>10?}",
                result.total_throughput_ops_sec,
                result.read_latency_stats.p50,
                result.write_latency_stats.p50
            );

            results.push(result);
        }

        // Get single-threaded baseline for efficiency calculation
        let baseline_throughput = results
            .iter()
            .find(|r| r.num_threads == 1)
            .map(|r| r.total_throughput_ops_sec)
            .unwrap_or(results[0].total_throughput_ops_sec);

        // Print combined analysis table
        println!("\n--- Scaling Analysis (Mixed Workload: {:.1}% writes) ---", args.write_ratio * 100.0);
        println!(
            "{:>10} | {:>15} | {:>12} | {:>12} | {:>10} | {:>8}",
            "threads", "throughput", "read_p50", "write_p50", "efficiency", "speedup"
        );
        println!("{}", "-".repeat(86));

        // RadixTree baseline row
        println!(
            "{:>10} | {:>12.2} ops/s | {:>12.3?} | {:>12} | {:>10} | {:>8}",
            "RadixTree",
            radix_stats.throughput_ops_sec,
            radix_stats.p50,
            "-",
            "-",
            "1.00x"
        );

        for result in &results {
            let efficiency = result.total_throughput_ops_sec / (result.num_threads as f64 * baseline_throughput);
            let speedup = result.total_throughput_ops_sec / radix_stats.throughput_ops_sec;
            println!(
                "{:>10} | {:>12.2} ops/s | {:>12.3?} | {:>12.3?} | {:>9.1}% | {:>7.2}x",
                result.num_threads,
                result.total_throughput_ops_sec,
                result.read_latency_stats.p50,
                result.write_latency_stats.p50,
                efficiency * 100.0,
                speedup
            );
        }

        // Breakeven analysis
        let breakeven = results
            .iter()
            .find(|r| r.total_throughput_ops_sec > radix_stats.throughput_ops_sec);

        println!("\n--- Breakeven Analysis ---");
        if let Some(result) = breakeven {
            println!(
                "  ConcurrentRadixTree beats RadixTree at {} threads ({:.2}x speedup)",
                result.num_threads,
                result.total_throughput_ops_sec / radix_stats.throughput_ops_sec
            );
        } else {
            println!("  ConcurrentRadixTree did not beat RadixTree in tested thread counts");
            if let Some(best) = results.iter().max_by(|a, b| {
                a.total_throughput_ops_sec.partial_cmp(&b.total_throughput_ops_sec).unwrap()
            }) {
                println!(
                    "  Best concurrent result: {} threads with {:.2}x relative throughput",
                    best.num_threads,
                    best.total_throughput_ops_sec / radix_stats.throughput_ops_sec
                );
            }
        }
    } else {
        // Read-only benchmark
        let mut results: Vec<ConcurrentResult> = Vec::new();

        for &num_threads in &thread_counts {
            print!("  threads={:>3}... ", num_threads);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let result = bench_concurrent(
                &concurrent_tree,
                &read_sequences_arc,
                num_threads,
                duration,
                warmup,
                args.query_mode,
                args.seed,
            );

            println!(
                "throughput={:>12.2} ops/sec, p50={:>10?}, p99={:>10?}",
                result.throughput_ops_sec, result.latency_stats.p50, result.latency_stats.p99
            );

            results.push(result);
        }

        // Get single-threaded concurrent baseline for efficiency calculation
        let baseline_throughput = results
            .iter()
            .find(|r| r.num_threads == 1)
            .map(|r| r.throughput_ops_sec)
            .unwrap_or(results[0].throughput_ops_sec);

        // Print combined analysis table
        println!("\n--- Scaling Analysis ---");
        println!(
            "{:>10} | {:>15} | {:>12} | {:>12} | {:>10} | {:>8}",
            "threads", "throughput", "p50", "p99", "efficiency", "speedup"
        );
        println!("{}", "-".repeat(86));

        // RadixTree baseline row
        println!(
            "{:>10} | {:>12.2} ops/s | {:>12.3?} | {:>12.3?} | {:>10} | {:>8}",
            "RadixTree",
            radix_stats.throughput_ops_sec,
            radix_stats.p50,
            radix_stats.p99,
            "-",
            "1.00x"
        );

        for result in &results {
            let efficiency = result.throughput_ops_sec / (result.num_threads as f64 * baseline_throughput);
            let speedup = result.throughput_ops_sec / radix_stats.throughput_ops_sec;
            println!(
                "{:>10} | {:>12.2} ops/s | {:>12.3?} | {:>12.3?} | {:>9.1}% | {:>7.2}x",
                result.num_threads,
                result.throughput_ops_sec,
                result.latency_stats.p50,
                result.latency_stats.p99,
                efficiency * 100.0,
                speedup
            );
        }

        // Breakeven analysis
        let breakeven = results
            .iter()
            .find(|r| r.throughput_ops_sec > radix_stats.throughput_ops_sec);

        println!("\n--- Breakeven Analysis ---");
        if let Some(result) = breakeven {
            println!(
                "  ConcurrentRadixTree beats RadixTree at {} threads ({:.2}x speedup)",
                result.num_threads,
                result.throughput_ops_sec / radix_stats.throughput_ops_sec
            );
        } else {
            println!("  ConcurrentRadixTree did not beat RadixTree in tested thread counts");
            if let Some(best) = results.iter().max_by(|a, b| {
                a.throughput_ops_sec.partial_cmp(&b.throughput_ops_sec).unwrap()
            }) {
                println!(
                    "  Best concurrent result: {} threads with {:.2}x relative throughput",
                    best.num_threads,
                    best.throughput_ops_sec / radix_stats.throughput_ops_sec
                );
            }
        }
    }
}

/// Phase 3: Mixed workload testing (sweeps thread counts and write ratios)
fn phase_mixed(args: &Args, read_sequences: &[SequenceData], write_sequences: &[SequenceData]) {
    println!("\n{}", "=".repeat(60));
    println!("Phase 3: Mixed Workload Testing");
    println!("{}", "=".repeat(60));

    if write_sequences.is_empty() {
        println!("No write sequences available. Skipping mixed workload phase.");
        return;
    }

    let duration = Duration::from_secs(args.duration_secs);
    let warmup = Duration::from_secs(args.warmup_secs);
    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let thread_counts = get_thread_counts(args, num_cpus);

    // Test different write ratios
    let write_ratios = if args.write_ratio > 0.0 {
        vec![args.write_ratio]
    } else {
        vec![0.0, 0.01, 0.05, 0.10, 0.25, 0.50]
    };

    println!("Thread counts: {:?}", thread_counts);
    println!("Write ratios: {:?}", write_ratios);
    println!("Read sequences: {}, Write sequences: {}", read_sequences.len(), write_sequences.len());
    println!("Duration per point: {}s", args.duration_secs);

    let read_sequences_arc = Arc::new(read_sequences.to_vec());
    let write_sequences_arc = Arc::new(write_sequences.to_vec());

    println!("\n--- Running mixed workload benchmark ---");

    // Store results indexed by (thread_count, write_ratio)
    let mut results: Vec<MixedWorkloadResult> = Vec::new();

    for &num_threads in &thread_counts {
        for &write_ratio in &write_ratios {
            // Rebuild tree for each configuration to start fresh
            let tree = Arc::new(build_concurrent_tree(read_sequences));

            print!("  threads={:>3}, write_ratio={:>5.1}%... ", num_threads, write_ratio * 100.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let result = bench_mixed_workload(
                &tree,
                &read_sequences_arc,
                &write_sequences_arc,
                num_threads,
                duration,
                warmup,
                args.query_mode,
                write_ratio,
                args.seed,
            );

            println!(
                "total={:>10.2} ops/s, r_p50={:>10?}, w_p50={:>10?}",
                result.total_throughput_ops_sec,
                result.read_latency_stats.p50,
                result.write_latency_stats.p50
            );

            results.push(result);
        }
    }

    // Print summary table
    println!("\n--- Mixed Workload Analysis ---");
    println!(
        "{:>8} | {:>12} | {:>14} | {:>14} | {:>12} | {:>12} | {:>12}",
        "threads", "write_ratio", "total_ops/s", "read_ops/s", "write_ops/s", "read_p50", "read_p99"
    );
    println!("{}", "-".repeat(100));

    for result in &results {
        println!(
            "{:>8} | {:>11.1}% | {:>14.2} | {:>14.2} | {:>12.2} | {:>12.3?} | {:>12.3?}",
            result.num_threads,
            result.write_ratio * 100.0,
            result.total_throughput_ops_sec,
            result.read_throughput_ops_sec,
            result.write_throughput_ops_sec,
            result.read_latency_stats.p50,
            result.read_latency_stats.p99,
        );
    }

    // Analyze write impact on read latency (compare across write ratios for each thread count)
    if write_ratios.len() >= 2 {
        println!("\n--- Write Impact on Read Latency ---");
        for &num_threads in &thread_counts {
            let thread_results: Vec<&MixedWorkloadResult> = results
                .iter()
                .filter(|r| r.num_threads == num_threads)
                .collect();

            if thread_results.len() >= 2 {
                let baseline = thread_results[0];
                if baseline.read_latency_stats.p50 > Duration::ZERO {
                    println!("  {} threads:", num_threads);
                    for result in thread_results.iter().skip(1) {
                        if result.read_latency_stats.p50 > Duration::ZERO {
                            let p50_ratio = result.read_latency_stats.p50.as_nanos() as f64
                                / baseline.read_latency_stats.p50.as_nanos() as f64;
                            let p99_ratio = result.read_latency_stats.p99.as_nanos() as f64
                                / baseline.read_latency_stats.p99.as_nanos() as f64;
                            println!(
                                "    {:>5.1}% writes: read p50 is {:.2}x baseline, read p99 is {:.2}x baseline",
                                result.write_ratio * 100.0,
                                p50_ratio,
                                p99_ratio
                            );
                        }
                    }
                }
            }
        }
    }

    // Analyze scaling efficiency per write ratio
    if thread_counts.len() >= 2 {
        println!("\n--- Scaling Efficiency by Write Ratio ---");
        for &write_ratio in &write_ratios {
            let ratio_results: Vec<&MixedWorkloadResult> = results
                .iter()
                .filter(|r| (r.write_ratio - write_ratio).abs() < 0.001)
                .collect();

            if ratio_results.len() >= 2 {
                let baseline = ratio_results
                    .iter()
                    .find(|r| r.num_threads == 1)
                    .unwrap_or(&ratio_results[0]);
                let best = ratio_results
                    .iter()
                    .max_by(|a, b| a.total_throughput_ops_sec.partial_cmp(&b.total_throughput_ops_sec).unwrap())
                    .unwrap();

                let max_threads = ratio_results.iter().map(|r| r.num_threads).max().unwrap();
                let max_result = ratio_results.iter().find(|r| r.num_threads == max_threads).unwrap();
                let efficiency = max_result.total_throughput_ops_sec
                    / (max_threads as f64 * baseline.total_throughput_ops_sec);

                println!(
                    "  {:>5.1}% writes: best={} threads ({:.2} ops/s), efficiency at {} threads: {:.1}%",
                    write_ratio * 100.0,
                    best.num_threads,
                    best.total_throughput_ops_sec,
                    max_threads,
                    efficiency * 100.0
                );
            }
        }
    }
}

fn main() {
    let mut args = Args::parse();

    // Apply preset if specified
    if let Some(preset) = args.preset {
        preset.apply(&mut args);
    }

    // Validate arguments
    if args.size == 0 || args.depth == 0 || args.num_workers == 0 {
        eprintln!("size, depth, and num_workers must be > 0");
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&args.prefix_prompt_ratio) {
        eprintln!("prefix_prompt_ratio must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&args.write_ratio) {
        eprintln!("write_ratio must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    let num_sequences = args.size / args.depth;
    if num_sequences == 0 {
        eprintln!("size must be >= depth to produce at least one sequence");
        std::process::exit(1);
    }

    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

    println!("Concurrent Radix Tree Benchmark");
    println!("================================\n");
    println!("Configuration:");
    if let Some(preset) = args.preset {
        println!("  Preset:               {:?}", preset);
    }
    println!("  Tree size:            {} (worker, block) pairs", args.size);
    println!("  Depth:                {} blocks/sequence", args.depth);
    println!("  Sequences:            {}", num_sequences);
    println!("  Workers:              {}", args.num_workers);
    println!("  Prefix prompt ratio:  {:.1}%", args.prefix_prompt_ratio * 100.0);
    println!("  Prefix prompt groups: {}", args.num_prefix_prompts);
    println!("  Duration:             {}s per phase", args.duration_secs);
    println!("  Warmup:               {}s", args.warmup_secs);
    println!("  Query mode:           {:?}", args.query_mode);
    println!("  Write ratio:          {:.1}%", args.write_ratio * 100.0);
    println!("  Seed:                 {}", args.seed);
    println!("  Available CPUs:       {}", num_cpus);
    if let Some(t) = args.threads {
        println!("  Threads (override):   {}", t);
    }

    // Generate read sequences (used for initial tree population and reads)
    print!("\nGenerating {} read sequences... ", num_sequences);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let read_sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );
    println!("done");

    // Generate write sequences (different from read sequences, used for inserts during mixed workload)
    // Use different seed to ensure different sequences
    // Generate if: mixed phase, all phases, or scaling phase with write_ratio > 0
    let needs_write_sequences = args.phase == "mixed"
        || args.phase == "all"
        || (args.phase == "scaling" && args.write_ratio > 0.0);

    let write_seq_count = num_sequences / 2; // Generate half as many write sequences
    let write_sequences = if needs_write_sequences {
        print!("Generating {} write sequences... ", write_seq_count);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let seqs = generate_sequences(
            write_seq_count,
            args.depth,
            args.num_workers,
            args.prefix_prompt_ratio,
            args.num_prefix_prompts,
            args.seed + 999_999, // Different seed for different sequences
        );
        println!("done");
        seqs
    } else {
        Vec::new()
    };

    // Run requested phases
    match args.phase.as_str() {
        "overhead" => phase_overhead(&args, &read_sequences),
        "scaling" => phase_scaling(&args, &read_sequences, &write_sequences),
        "mixed" => phase_mixed(&args, &read_sequences, &write_sequences),
        "all" => {
            phase_overhead(&args, &read_sequences);
            phase_scaling(&args, &read_sequences, &write_sequences);
            phase_mixed(&args, &read_sequences, &write_sequences);
        }
        _ => {
            eprintln!(
                "Unknown phase: {}. Use 'overhead', 'scaling', 'mixed', or 'all'",
                args.phase
            );
            std::process::exit(1);
        }
    }

    println!("\nBenchmark complete.");
}
