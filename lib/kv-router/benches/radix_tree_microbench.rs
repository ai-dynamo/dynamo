// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmark for radix tree operations with configurable size and depth.
//!
//! Measures latency and throughput of:
//! - store_block: Adding blocks to the tree
//! - remove_block: Removing blocks from the tree
//! - find_matches: Finding prefix matches in the tree
//!
//! Size is defined as total (worker, block) pairs in the tree.
//! Depth is the number of blocks per sequence (depth = (isl + osl) / block_size).
//!
//! Run with: cargo bench --package dynamo-kv-router --bench radix_tree_microbench --features bench -- --help

use clap::{Parser, ValueEnum};
use dynamo_kv_router::{
    compute_block_hash_for_seq,
    indexer::{RadixTree, RouterEvent},
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, WorkerId,
    },
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

/// Sweep benchmark mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SweepMode {
    /// Vary sequence/query length (query has exactly `depth` blocks, all matching)
    Depth,
    /// Vary match length (query has `max_depth` blocks, first `depth` match, rest garbage)
    MatchLength,
    /// Vary number of prefix prompt groups (width of shared prefixes)
    Width,
}

#[derive(Parser, Debug)]
#[command(name = "radix_tree_microbench")]
#[command(about = "Microbenchmark for radix tree operations")]
struct Args {
    /// Ignored: passed by cargo bench harness
    #[arg(long, hide = true)]
    bench: bool,

    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "10000")]
    size: usize,

    /// Sequence depth in blocks (depth = (isl + osl) / block_size, where block_size = 16)
    #[arg(long, default_value = "32")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Number of iterations per operation for timing
    #[arg(long, default_value = "1000")]
    iterations: usize,

    /// Prefix prompt ratio (0.0 to 1.0) - portion of sequence from the beginning that is a shared prefix
    #[arg(long, default_value = "0.25")]
    prefix_prompt_ratio: f64,

    /// Number of unique prefix prompt groups to randomly sample from
    #[arg(long, default_value = "4")]
    num_prefix_prompts: usize,

    /// Run only specific benchmark (hash, store, remove, find_matches, dump, sweep, or all)
    #[arg(long, default_value = "all")]
    benchmark_type: String,

    /// KV block size in tokens (for hash computation)
    #[arg(long, default_value = "16")]
    block_size: u32,

    /// Verbose output with per-iteration timings
    #[arg(short, long)]
    verbose: bool,

    /// Minimum depth for sweep mode
    #[arg(long, default_value = "1")]
    min_depth: usize,

    /// Maximum depth for sweep mode
    #[arg(long, default_value = "8000")]
    max_depth: usize,

    /// Number of depth points to sample in sweep mode (logarithmically spaced)
    #[arg(long, default_value = "20")]
    sweep_points: usize,

    /// Iterations per depth point in sweep mode
    #[arg(long, default_value = "100")]
    sweep_iterations: usize,

    /// Output format for sweep mode: "table" or "csv"
    #[arg(long, default_value = "table")]
    sweep_format: String,

    /// Sweep mode: what to vary during the sweep
    #[arg(long, value_enum, default_value = "depth")]
    sweep_mode: SweepMode,

    /// Minimum width (num_prefix_prompts) for width sweep mode
    #[arg(long, default_value = "1")]
    min_width: usize,

    /// Maximum width (num_prefix_prompts) for width sweep mode
    #[arg(long, default_value = "64")]
    max_width: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,
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

/// Build a pre-populated tree (prints timing info)
fn build_tree(sequences: &[SequenceData]) -> RadixTree {
    let num_blocks: usize = sequences.iter().map(|s| s.local_hashes.len()).sum();
    print!(
        "  Building tree with {} sequences ({} blocks)... ",
        sequences.len(),
        num_blocks
    );
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let start = Instant::now();
    let mut tree = RadixTree::new();
    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_store_event(event_id as u64);
        let _ = tree.apply_event(event);
    }
    let elapsed = start.elapsed();

    println!(
        "done in {:.2?} ({:.2} sequences/sec, {:.2} blocks/sec)",
        elapsed,
        sequences.len() as f64 / elapsed.as_secs_f64(),
        num_blocks as f64 / elapsed.as_secs_f64()
    );

    tree
}

/// Statistics for a set of timing measurements
#[derive(Debug)]
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

/// Benchmark compute_block_hash_for_seq operation
fn bench_hash(args: &Args) {
    println!("\n=== Benchmarking COMPUTE_BLOCK_HASH (per-request hot path) ===");

    let num_tokens = args.depth * args.block_size as usize;
    println!(
        "  Token sequence length: {} tokens ({} blocks)",
        num_tokens, args.depth
    );

    // Generate token sequences to hash
    let token_sequences: Vec<Vec<u32>> = (0..args.iterations)
        .map(|i| {
            (0..num_tokens)
                .map(|j| ((i * num_tokens + j) % 50000) as u32)
                .collect()
        })
        .collect();

    let mut durations = Vec::with_capacity(args.iterations);

    for (i, tokens) in token_sequences.iter().enumerate() {
        let start = Instant::now();
        let _ = compute_block_hash_for_seq(tokens, args.block_size, None);
        let elapsed = start.elapsed();

        durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("  Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let stats = LatencyStats::from_durations(durations);
    stats.print("COMPUTE_BLOCK_HASH", args.depth);
}

/// Benchmark store_block operation
fn bench_store(args: &Args) {
    println!("\n=== Benchmarking STORE_BLOCK ===");

    let num_sequences = args.size / args.depth;
    let bench_iters = args.iterations.min(num_sequences);

    let all_sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );
    let split_point = num_sequences.saturating_sub(bench_iters);
    let pre_sequences = &all_sequences[..split_point];
    let bench_sequences = &all_sequences[split_point..];

    // Build tree once, then store sequences sequentially
    // Tree grows from (size - iterations) to size over the benchmark
    let mut tree = build_tree(&pre_sequences);
    println!(
        "  Initial tree size: {} blocks, will grow to ~{} blocks",
        tree.current_size(),
        tree.current_size() + bench_iters * args.depth
    );

    let mut durations = Vec::with_capacity(bench_iters);

    for (i, seq) in bench_sequences.iter().enumerate() {
        let event = seq.to_store_event(i as u64);

        let start = Instant::now();
        let _ = tree.apply_event(event);
        let elapsed = start.elapsed();

        durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("  Completed {}/{} iterations", i + 1, bench_iters);
        }
    }

    let stats = LatencyStats::from_durations(durations);
    stats.print("STORE_BLOCK", args.depth);
}

/// Benchmark remove_block operation
fn bench_remove(args: &Args) {
    println!("\n=== Benchmarking REMOVE_BLOCK ===");

    let num_sequences = args.size / args.depth;
    let sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );

    // Build tree once, then remove/re-add to restore state after each timed removal
    let mut tree = build_tree(&sequences);
    println!("  Tree size: {} blocks", tree.current_size());

    let mut durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        // Remove a sequence (timed)
        let seq_to_remove = &sequences[i % sequences.len()];
        let remove_event = seq_to_remove.to_remove_event(i as u64);

        let start = Instant::now();
        let _ = tree.apply_event(remove_event);
        let elapsed = start.elapsed();

        durations.push(elapsed);

        // Re-add the sequence to restore tree state (untimed)
        let store_event = seq_to_remove.to_store_event(i as u64 + args.iterations as u64);
        let _ = tree.apply_event(store_event);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("  Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let stats = LatencyStats::from_durations(durations);
    stats.print("REMOVE_BLOCK", args.depth);
}

/// Benchmark find_matches operation
fn bench_find_matches(args: &Args) {
    println!("\n=== Benchmarking FIND_MATCHES ===");

    let num_sequences = args.size / args.depth;
    let sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );

    // Build tree once for all find_matches calls
    let tree = build_tree(&sequences);

    println!(
        "  Tree built with {} sequences, {} total blocks",
        sequences.len(),
        tree.current_size()
    );

    // Benchmark hit case (lookup existing sequences)
    println!("\n  --- HIT case (existing sequences) ---");
    let mut hit_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let seq = &sequences[i % sequences.len()];
        let hashes_copy = seq.local_hashes.clone();

        let start = Instant::now();
        let _ = tree.find_matches(hashes_copy, false);
        let elapsed = start.elapsed();

        hit_durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let hit_stats = LatencyStats::from_durations(hit_durations);
    hit_stats.print("FIND_MATCHES (HIT)", args.depth);

    // Benchmark miss case (find_matches on non-existing sequences)
    println!("\n  --- MISS case (non-existing sequences) ---");
    let mut miss_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        // Generate a sequence that won't match
        let miss_hashes: Vec<LocalBlockHash> = (0..args.depth)
            .map(|j| LocalBlockHash(0xBAD_C0DE_0000_0000 | ((i as u64) << 16) | (j as u64)))
            .collect();

        let start = Instant::now();
        let _ = tree.find_matches(miss_hashes, false);
        let elapsed = start.elapsed();

        miss_durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let miss_stats = LatencyStats::from_durations(miss_durations);
    miss_stats.print("FIND_MATCHES (MISS)", args.depth);

    // Benchmark partial match case
    println!("\n  --- PARTIAL case (prefix match only) ---");
    let mut partial_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let seq = &sequences[i % sequences.len()];
        // Use first half of real sequence, second half is garbage
        let half = args.depth / 2;
        let mut partial_hashes = seq.local_hashes[..half].to_vec();
        partial_hashes.extend(
            (0..half).map(|j| LocalBlockHash(0xDEAD_0000 | ((i as u64) << 16) | (j as u64))),
        );

        let start = Instant::now();
        let _ = tree.find_matches(partial_hashes, false);
        let elapsed = start.elapsed();

        partial_durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let partial_stats = LatencyStats::from_durations(partial_durations);
    partial_stats.print("FIND_MATCHES (PARTIAL)", args.depth);

    // Benchmark with early_exit=true
    println!("\n  --- EARLY_EXIT case ---");
    let mut early_exit_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let seq = &sequences[i % sequences.len()];

        let start = Instant::now();
        let _ = tree.find_matches(seq.local_hashes.clone(), true);
        let elapsed = start.elapsed();

        early_exit_durations.push(elapsed);
    }

    let early_exit_stats = LatencyStats::from_durations(early_exit_durations);
    early_exit_stats.print("FIND_MATCHES (EARLY_EXIT)", args.depth);
}

/// Generate logarithmically spaced values between min and max
fn generate_log_spaced_points(min_val: usize, max_val: usize, num_points: usize) -> Vec<usize> {
    if num_points <= 1 {
        return vec![max_val];
    }

    let log_min = (min_val as f64).ln();
    let log_max = (max_val as f64).ln();
    let step = (log_max - log_min) / (num_points - 1) as f64;

    let mut points: Vec<usize> = (0..num_points)
        .map(|i| (log_min + step * i as f64).exp().round() as usize)
        .map(|v| v.max(1)) // Ensure minimum value of 1
        .collect();

    // Deduplicate (logarithmic spacing can produce duplicates at low values)
    points.dedup();
    points
}

/// Latency statistics (avg, p50, p99) in nanoseconds
#[derive(Debug)]
struct DurationStats {
    avg_ns: u64,
    p50_ns: u64,
    p99_ns: u64,
}

impl DurationStats {
    /// Compute stats from durations. Sorts the input vector in place.
    fn from_durations(durations: &mut [Duration]) -> Self {
        durations.sort();
        let n = durations.len();
        let avg = durations.iter().sum::<Duration>() / n as u32;
        Self {
            avg_ns: avg.as_nanos() as u64,
            p50_ns: durations[n / 2].as_nanos() as u64,
            p99_ns: durations[n * 99 / 100].as_nanos() as u64,
        }
    }
}

/// Results for a single sweep point (depth or width)
#[derive(Debug)]
struct SweepResult {
    point: usize,
    point_label: &'static str,
    store: DurationStats,
    remove: DurationStats,
    find_matches: DurationStats,
}

impl SweepResult {
    fn csv_header(&self) -> String {
        format!(
            "{},store_avg_ns,store_p50_ns,store_p99_ns,remove_avg_ns,remove_p50_ns,remove_p99_ns,find_matches_avg_ns,find_matches_p50_ns,find_matches_p99_ns",
            self.point_label
        )
    }

    fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{}",
            self.point,
            self.store.avg_ns,
            self.store.p50_ns,
            self.store.p99_ns,
            self.remove.avg_ns,
            self.remove.p50_ns,
            self.remove.p99_ns,
            self.find_matches.avg_ns,
            self.find_matches.p50_ns,
            self.find_matches.p99_ns
        )
    }

    fn table_header(&self) -> String {
        format!(
            "{:>8} | store_avg    store_p50    store_p99 | remove_avg   remove_p50   remove_p99 |       fm_avg       fm_p50       fm_p99",
            self.point_label
        )
    }

    fn table_row(&self) -> String {
        format!(
            "{:>8} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}",
            self.point,
            format_duration_ns(self.store.avg_ns),
            format_duration_ns(self.store.p50_ns),
            format_duration_ns(self.store.p99_ns),
            format_duration_ns(self.remove.avg_ns),
            format_duration_ns(self.remove.p50_ns),
            format_duration_ns(self.remove.p99_ns),
            format_duration_ns(self.find_matches.avg_ns),
            format_duration_ns(self.find_matches.p50_ns),
            format_duration_ns(self.find_matches.p99_ns)
        )
    }
}

fn print_sweep_results_dynamic(results: &[SweepResult], format: &str) {
    if results.is_empty() {
        return;
    }
    println!();
    if format == "csv" {
        println!("{}", results[0].csv_header());
        for r in results {
            println!("{}", r.csv_row());
        }
    } else {
        println!("{}", results[0].table_header());
        println!("{}", "-".repeat(130));
        for r in results {
            println!("{}", r.table_row());
        }
    }
}

/// Benchmark store/remove/find_matches across a range of depths or widths.
///
/// For each sweep point, the tree is rebuilt.
///
/// With `--sweep_mode match_length`, find_matches queries have `max_depth` blocks
/// where only the first `depth` blocks match (rest are garbage). With `--sweep_mode depth`,
/// queries have exactly `depth` blocks (all matching). With `--sweep_mode width`,
/// the number of prefix prompt groups is varied.
fn bench_sweep(args: &Args) {
    let seq_length = args.max_depth;
    let num_sequences = args.size / seq_length;

    if num_sequences < 2 {
        eprintln!(
            "Error: size {} / max_depth {} = {} sequences (need at least 2). \
             Increase --size or decrease --max-depth.",
            args.size, seq_length, num_sequences
        );
        std::process::exit(1);
    }

    let (mode_name, point_label, sweep_points) = match args.sweep_mode {
        SweepMode::Depth => (
            "Depth",
            "depth",
            generate_log_spaced_points(args.min_depth, args.max_depth, args.sweep_points),
        ),
        SweepMode::MatchLength => (
            "Match Length",
            "depth",
            generate_log_spaced_points(args.min_depth, args.max_depth, args.sweep_points),
        ),
        SweepMode::Width => (
            "Width",
            "width",
            generate_log_spaced_points(args.min_width, args.max_width, args.sweep_points),
        ),
    };

    println!("\n=== {} Sweep Benchmark ===", mode_name);
    println!("  Sequence length: {} blocks (fixed)", seq_length);
    match args.sweep_mode {
        SweepMode::Depth | SweepMode::MatchLength => {
            println!(
                "  Sweep range: {} to {} ({} points, log-spaced)",
                args.min_depth, args.max_depth, args.sweep_points
            );
        }
        SweepMode::Width => {
            println!(
                "  Width range: {} to {} ({} points, log-spaced)",
                args.min_width, args.max_width, args.sweep_points
            );
            println!(
                "  Prefix prompt ratio: {:.1}%",
                args.prefix_prompt_ratio * 100.0
            );
        }
    }
    println!("  Iterations per point: {}", args.sweep_iterations);
    println!(
        "  Tree: {} sequences, {} total blocks",
        num_sequences,
        num_sequences * seq_length
    );
    println!("  Workers: {}", args.num_workers);
    match args.sweep_mode {
        SweepMode::MatchLength => {
            println!("  Mode: find_matches queries padded with garbage to max_depth");
        }
        SweepMode::Depth => {
            println!("  Mode: find_matches queries truncated to depth");
        }
        SweepMode::Width => {
            println!("  Mode: varying num_prefix_prompts, full-depth operations");
        }
    }
    println!();

    let mut results: Vec<SweepResult> = Vec::with_capacity(sweep_points.len());

    for (idx, &point) in sweep_points.iter().enumerate() {
        print!(
            "[{}/{}] {}={}... ",
            idx + 1,
            sweep_points.len(),
            point_label,
            point
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Determine depth and num_prefix_prompts for this sweep point
        let (depth, num_prefix_prompts) = match args.sweep_mode {
            SweepMode::Depth | SweepMode::MatchLength => (point, args.num_prefix_prompts),
            SweepMode::Width => (seq_length, point),
        };

        // Generate sequences and rebuild tree for this point
        let extra_count = args.sweep_iterations;
        let all_sequences = generate_sequences(
            num_sequences + extra_count,
            seq_length,
            args.num_workers,
            args.prefix_prompt_ratio,
            num_prefix_prompts,
            args.seed,
        );
        let tree_sequences = &all_sequences[..num_sequences];
        let extra_sequences = &all_sequences[num_sequences..];

        let mut tree = build_tree(tree_sequences);

        // --- STORE benchmark ---
        let mut store_durations = Vec::with_capacity(args.sweep_iterations);
        for (i, seq) in extra_sequences
            .iter()
            .enumerate()
            .take(args.sweep_iterations)
        {
            let truncated = SequenceData {
                worker_id: seq.worker_id,
                local_hashes: seq.local_hashes[..depth].to_vec(),
                external_hashes: seq.external_hashes[..depth].to_vec(),
            };
            let store_event = truncated.to_store_event(i as u64);

            let start = Instant::now();
            let _ = tree.apply_event(store_event);
            store_durations.push(start.elapsed());

            // Remove to restore tree state (untimed)
            let remove_event = truncated.to_remove_event(i as u64);
            let _ = tree.apply_event(remove_event);
        }

        // --- REMOVE benchmark ---
        let mut remove_durations = Vec::with_capacity(args.sweep_iterations);
        for i in 0..args.sweep_iterations.min(num_sequences) {
            let seq = &tree_sequences[i % tree_sequences.len()];
            let truncated = SequenceData {
                worker_id: seq.worker_id,
                local_hashes: seq.local_hashes[..depth].to_vec(),
                external_hashes: seq.external_hashes[..depth].to_vec(),
            };
            let remove_event = truncated.to_remove_event(i as u64);

            let start = Instant::now();
            let _ = tree.apply_event(remove_event);
            remove_durations.push(start.elapsed());

            // Re-add to restore state (untimed)
            let store_event = truncated.to_store_event(i as u64 + 1000000);
            let _ = tree.apply_event(store_event);
        }

        // --- FIND_MATCHES benchmark ---
        let mut find_matches_durations = Vec::with_capacity(args.sweep_iterations);
        for i in 0..args.sweep_iterations {
            let seq = &tree_sequences[i % tree_sequences.len()];

            let query = match args.sweep_mode {
                SweepMode::MatchLength => {
                    // Match length mode: first `depth` blocks match, rest are garbage
                    let mut q = seq.local_hashes[..depth].to_vec();
                    let garbage_len = seq_length - depth;
                    q.extend((0..garbage_len).map(|j| {
                        LocalBlockHash(0xBAD_C0DE_0000_0000 | ((i as u64) << 16) | (j as u64))
                    }));
                    q
                }
                SweepMode::Depth | SweepMode::Width => {
                    // Depth/width mode: query has exactly `depth` blocks
                    seq.local_hashes[..depth].to_vec()
                }
            };

            let start = Instant::now();
            let _ = tree.find_matches(query, false);
            find_matches_durations.push(start.elapsed());
        }

        // Compute stats
        let store = DurationStats::from_durations(&mut store_durations);
        let remove = DurationStats::from_durations(&mut remove_durations);
        let find_matches = DurationStats::from_durations(&mut find_matches_durations);

        println!(
            "store={:.2}us, remove={:.2}us, find_matches={:.2}us",
            store.avg_ns as f64 / 1000.0,
            remove.avg_ns as f64 / 1000.0,
            find_matches.avg_ns as f64 / 1000.0
        );

        results.push(SweepResult {
            point,
            point_label,
            store,
            remove,
            find_matches,
        });
    }

    print_sweep_results_dynamic(&results, &args.sweep_format);
}

/// Benchmark dump_tree_as_events (BFS dump)
fn bench_dump(args: &Args) {
    println!("\n=== Benchmarking DUMP_TREE_AS_EVENTS (BFS dump) ===");

    let num_sequences = args.size / args.depth;
    let sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_prompt_ratio,
        args.num_prefix_prompts,
        args.seed,
    );

    let tree = build_tree(&sequences);
    println!(
        "  Tree built with {} sequences, {} total blocks",
        sequences.len(),
        tree.current_size()
    );

    // Single iteration timing
    let start = Instant::now();
    let events = tree.dump_tree_as_events();
    let elapsed = start.elapsed();

    println!("\nDUMP_TREE_AS_EVENTS Results:");
    println!("  Time:        {:?}", elapsed);
    println!("  Events:      {}", events.len());
    println!(
        "  Throughput:  {:.2} events/sec",
        events.len() as f64 / elapsed.as_secs_f64()
    );
}

/// Format nanoseconds as human-readable string
fn format_duration_ns(ns: u64) -> String {
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

fn main() {
    let args = Args::parse();

    // Validate arguments to prevent panics
    if args.size == 0
        || args.depth == 0
        || args.num_workers == 0
        || args.iterations == 0
        || args.block_size == 0
        || args.min_depth == 0
        || args.max_depth == 0
        || args.min_width == 0
        || args.max_width == 0
        || args.sweep_iterations == 0
    {
        eprintln!(
            "size, depth, num_workers, iterations, block_size, min_depth, max_depth, min_width, max_width, and sweep_iterations must be > 0"
        );
        std::process::exit(1);
    }
    if args.min_depth > args.max_depth {
        eprintln!("min_depth must be <= max_depth");
        std::process::exit(1);
    }
    if args.min_width > args.max_width {
        eprintln!("min_width must be <= max_width");
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&args.prefix_prompt_ratio) {
        eprintln!("prefix_prompt_ratio must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    let num_sequences = args.size / args.depth;
    if matches!(
        args.benchmark_type.as_str(),
        "store" | "remove" | "lookup" | "sweep" | "all"
    ) && num_sequences == 0
    {
        eprintln!(
            "size must be >= depth to produce at least one sequence for {}",
            args.benchmark_type
        );
        std::process::exit(1);
    }

    println!("Radix Tree Microbenchmark");
    println!("=========================\n");
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
        "  Prefix prompt ratio: {:.1}% ({} blocks at depth {})",
        args.prefix_prompt_ratio * 100.0,
        (args.depth as f64 * args.prefix_prompt_ratio).round() as usize,
        args.depth
    );
    println!("  Prefix prompt groups: {}", args.num_prefix_prompts);

    println!(
        "\n  Derived: {} sequences to reach target size",
        num_sequences
    );

    match args.benchmark_type.as_str() {
        "hash" => bench_hash(&args),
        "store" => bench_store(&args),
        "remove" => bench_remove(&args),
        "find_matches" => bench_find_matches(&args),
        "dump" => bench_dump(&args),
        "sweep" => bench_sweep(&args),
        "all" => {
            bench_hash(&args);
            bench_store(&args);
            bench_remove(&args);
            bench_find_matches(&args);
            bench_dump(&args);
        }
        _ => {
            eprintln!(
                "Unknown benchmark type: {}. Use 'hash', 'store', 'remove', 'find_matches', 'dump', 'sweep', or 'all'",
                args.benchmark_type
            );
            std::process::exit(1);
        }
    }

    println!("\nBenchmark complete.");
}
