// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmark for radix tree operations with configurable size and depth.
//!
//! Measures latency and throughput of:
//! - store_block: Adding blocks to the tree
//! - remove_block: Removing blocks from the tree
//! - lookup_input: Finding prefix matches (find_matches)
//!
//! Size is defined as total (worker, block) pairs in the tree.
//! Depth is the number of blocks per sequence (depth = (isl + osl) / block_size).
//!
//! Run with: cargo run --package dynamo-llm --bin radix_tree_microbench --features kv-router-stress -- --help

use clap::Parser;
use dynamo_llm::kv_router::{
    indexer::{RadixTree, RouterEvent},
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, WorkerId,
        compute_block_hash_for_seq,
    },
};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "radix_tree_microbench")]
#[command(about = "Microbenchmark for radix tree operations")]
struct Args {
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

    /// Prefix sharing ratio (0.0 to 1.0) - fraction of sequences sharing a common prefix
    #[arg(long, default_value = "0.5")]
    prefix_share_ratio: f64,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Run only specific benchmark (hash, store, remove, lookup, sweep, or all)
    #[arg(long, default_value = "all")]
    bench: String,

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

    /// Sweep match length instead of sequence length.
    /// When set, tree sequences are fixed at max_depth length,
    /// and depth controls how many blocks of the lookup query match.
    #[arg(long)]
    sweep_match_length: bool,

    /// Fixed sequence length for sweep_match_length mode (defaults to max_depth)
    #[arg(long)]
    seq_length: Option<usize>,
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
    println!("  Token sequence length: {} tokens ({} blocks)", num_tokens, args.depth);

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
    let pre_sequences = generate_sequences(
        num_sequences.saturating_sub(args.iterations),
        args.depth,
        args.num_workers,
        args.prefix_share_ratio,
    );

    // Generate sequences to insert during benchmark
    let bench_sequences: Vec<SequenceData> = ((num_sequences - args.iterations)..num_sequences)
        .map(|seq_id| {
            SequenceData::new(
                seq_id as u64,
                (seq_id % args.num_workers) as WorkerId,
                args.depth,
            )
        })
        .collect();

    // Build tree once, then store sequences sequentially
    // Tree grows from (size - iterations) to size over the benchmark
    let mut tree = build_tree(&pre_sequences);
    println!(
        "  Initial tree size: {} blocks, will grow to ~{} blocks",
        tree.current_size(),
        tree.current_size() + args.iterations * args.depth
    );

    let mut durations = Vec::with_capacity(args.iterations);

    for (i, seq) in bench_sequences.iter().enumerate() {
        let event = seq.to_store_event(i as u64);

        let start = Instant::now();
        let _ = tree.apply_event(event);
        let elapsed = start.elapsed();

        durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("  Completed {}/{} iterations", i + 1, args.iterations);
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
        args.prefix_share_ratio,
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

/// Benchmark lookup_input (find_matches) operation
fn bench_lookup(args: &Args) {
    println!("\n=== Benchmarking LOOKUP_INPUT (find_matches) ===");

    let num_sequences = args.size / args.depth;
    let sequences = generate_sequences(
        num_sequences,
        args.depth,
        args.num_workers,
        args.prefix_share_ratio,
    );

    // Build tree once for all lookups
    let tree = build_tree(&sequences);

    println!("  Tree built with {} sequences, {} total blocks",
             sequences.len(),
             tree.current_size());

    // Benchmark hit case (lookup existing sequences)
    println!("\n  --- HIT case (existing sequences) ---");
    let mut hit_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let seq = &sequences[i % sequences.len()];

        let start = Instant::now();
        let _ = tree.find_matches(seq.local_hashes.clone(), false);
        let elapsed = start.elapsed();

        hit_durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let hit_stats = LatencyStats::from_durations(hit_durations);
    hit_stats.print("LOOKUP_INPUT (HIT)", args.depth);

    // Benchmark miss case (lookup non-existing sequences)
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
    miss_stats.print("LOOKUP_INPUT (MISS)", args.depth);

    // Benchmark partial match case
    println!("\n  --- PARTIAL case (prefix match only) ---");
    let mut partial_durations = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let seq = &sequences[i % sequences.len()];
        // Use first half of real sequence, second half is garbage
        let half = args.depth / 2;
        let mut partial_hashes = seq.local_hashes[..half].to_vec();
        partial_hashes
            .extend((0..half).map(|j| LocalBlockHash(0xDEAD_0000 | ((i as u64) << 16) | (j as u64))));

        let start = Instant::now();
        let _ = tree.find_matches(partial_hashes, false);
        let elapsed = start.elapsed();

        partial_durations.push(elapsed);

        if args.verbose && (i + 1) % 100 == 0 {
            println!("    Completed {}/{} iterations", i + 1, args.iterations);
        }
    }

    let partial_stats = LatencyStats::from_durations(partial_durations);
    partial_stats.print("LOOKUP_INPUT (PARTIAL)", args.depth);

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
    early_exit_stats.print("LOOKUP_INPUT (EARLY_EXIT)", args.depth);
}

/// Generate logarithmically spaced depth values
fn generate_depth_points(min_depth: usize, max_depth: usize, num_points: usize) -> Vec<usize> {
    if num_points <= 1 {
        return vec![max_depth];
    }

    let log_min = (min_depth as f64).ln();
    let log_max = (max_depth as f64).ln();
    let step = (log_max - log_min) / (num_points - 1) as f64;

    let mut depths: Vec<usize> = (0..num_points)
        .map(|i| (log_min + step * i as f64).exp().round() as usize)
        .map(|d| d.max(1)) // Ensure minimum depth of 1
        .collect();

    // Deduplicate (logarithmic spacing can produce duplicates at low values)
    depths.dedup();
    depths
}

/// Results for a single depth point
#[derive(Debug)]
struct DepthResult {
    depth: usize,
    store_avg_ns: u64,
    store_p50_ns: u64,
    store_p99_ns: u64,
    remove_avg_ns: u64,
    remove_p50_ns: u64,
    remove_p99_ns: u64,
    lookup_avg_ns: u64,
    lookup_p50_ns: u64,
    lookup_p99_ns: u64,
}

/// Benchmark store/remove/lookup across a range of depths
fn bench_sweep(args: &Args) {
    println!("\n=== Depth Sweep Benchmark ===");
    println!("  Depths: {} to {} ({} points, log-spaced)",
             args.min_depth, args.max_depth, args.sweep_points);
    println!("  Iterations per depth: {}", args.sweep_iterations);
    println!("  Tree size: {} blocks", args.size);
    println!("  Workers: {}", args.num_workers);
    println!();

    let depths = generate_depth_points(args.min_depth, args.max_depth, args.sweep_points);
    let mut results: Vec<DepthResult> = Vec::with_capacity(depths.len());

    for (idx, &depth) in depths.iter().enumerate() {
        // Skip if depth is larger than size (would result in 0 sequences)
        if depth > args.size {
            continue;
        }

        let num_sequences = args.size / depth;
        if num_sequences < 2 {
            continue; // Need at least 2 sequences for meaningful benchmark
        }

        print!("[{}/{}] depth={}, sequences={}... ",
               idx + 1, depths.len(), depth, num_sequences);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Generate sequences for this depth
        let sequences = generate_sequences(
            num_sequences,
            depth,
            args.num_workers,
            args.prefix_share_ratio,
        );

        // --- STORE benchmark ---
        // Build tree once with fewer sequences, then store additional ones
        let mut store_durations = Vec::with_capacity(args.sweep_iterations);
        let pre_count = num_sequences.saturating_sub(args.sweep_iterations).max(num_sequences / 2);
        let pre_sequences = &sequences[..pre_count];
        let store_sequences = &sequences[pre_count..];

        let mut store_tree = build_tree(pre_sequences);
        for (i, seq_to_store) in store_sequences.iter().enumerate().take(args.sweep_iterations) {
            let event = seq_to_store.to_store_event(i as u64);

            let start = Instant::now();
            let _ = store_tree.apply_event(event);
            store_durations.push(start.elapsed());
        }

        // --- REMOVE benchmark ---
        // Build tree once, remove/re-add to restore state
        let mut remove_tree = build_tree(&sequences);
        let mut remove_durations = Vec::with_capacity(args.sweep_iterations);
        for i in 0..args.sweep_iterations.min(num_sequences) {
            let seq_to_remove = &sequences[i % sequences.len()];
            let remove_event = seq_to_remove.to_remove_event(i as u64);

            let start = Instant::now();
            let _ = remove_tree.apply_event(remove_event);
            remove_durations.push(start.elapsed());

            // Re-add to restore state (untimed)
            let store_event = seq_to_remove.to_store_event(i as u64 + 1000000);
            let _ = remove_tree.apply_event(store_event);
        }

        // --- LOOKUP benchmark ---
        let tree = build_tree(&sequences);
        let mut lookup_durations = Vec::with_capacity(args.sweep_iterations);
        for i in 0..args.sweep_iterations {
            let seq = &sequences[i % sequences.len()];

            let start = Instant::now();
            let _ = tree.find_matches(seq.local_hashes.clone(), false);
            lookup_durations.push(start.elapsed());
        }

        // Compute stats
        store_durations.sort();
        remove_durations.sort();
        lookup_durations.sort();

        let store_avg = store_durations.iter().sum::<Duration>() / store_durations.len() as u32;
        let remove_avg = remove_durations.iter().sum::<Duration>() / remove_durations.len() as u32;
        let lookup_avg = lookup_durations.iter().sum::<Duration>() / lookup_durations.len() as u32;

        let result = DepthResult {
            depth,
            store_avg_ns: store_avg.as_nanos() as u64,
            store_p50_ns: store_durations[store_durations.len() / 2].as_nanos() as u64,
            store_p99_ns: store_durations[store_durations.len() * 99 / 100].as_nanos() as u64,
            remove_avg_ns: remove_avg.as_nanos() as u64,
            remove_p50_ns: remove_durations[remove_durations.len() / 2].as_nanos() as u64,
            remove_p99_ns: remove_durations[remove_durations.len() * 99 / 100].as_nanos() as u64,
            lookup_avg_ns: lookup_avg.as_nanos() as u64,
            lookup_p50_ns: lookup_durations[lookup_durations.len() / 2].as_nanos() as u64,
            lookup_p99_ns: lookup_durations[lookup_durations.len() * 99 / 100].as_nanos() as u64,
        };

        println!("store={:.2}us, remove={:.2}us, lookup={:.2}us",
                 result.store_avg_ns as f64 / 1000.0,
                 result.remove_avg_ns as f64 / 1000.0,
                 result.lookup_avg_ns as f64 / 1000.0);

        results.push(result);
    }

    // Print results in requested format
    println!();
    if args.sweep_format == "csv" {
        println!("depth,store_avg_ns,store_p50_ns,store_p99_ns,remove_avg_ns,remove_p50_ns,remove_p99_ns,lookup_avg_ns,lookup_p50_ns,lookup_p99_ns");
        for r in &results {
            println!("{},{},{},{},{},{},{},{},{},{}",
                     r.depth,
                     r.store_avg_ns, r.store_p50_ns, r.store_p99_ns,
                     r.remove_avg_ns, r.remove_p50_ns, r.remove_p99_ns,
                     r.lookup_avg_ns, r.lookup_p50_ns, r.lookup_p99_ns);
        }
    } else {
        // Table format
        println!("{:>8} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}",
                 "depth", "store_avg", "store_p50", "store_p99",
                 "remove_avg", "remove_p50", "remove_p99",
                 "lookup_avg", "lookup_p50", "lookup_p99");
        println!("{}", "-".repeat(130));
        for r in &results {
            println!("{:>8} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}",
                     r.depth,
                     format_duration_ns(r.store_avg_ns),
                     format_duration_ns(r.store_p50_ns),
                     format_duration_ns(r.store_p99_ns),
                     format_duration_ns(r.remove_avg_ns),
                     format_duration_ns(r.remove_p50_ns),
                     format_duration_ns(r.remove_p99_ns),
                     format_duration_ns(r.lookup_avg_ns),
                     format_duration_ns(r.lookup_p50_ns),
                     format_duration_ns(r.lookup_p99_ns));
        }
    }
}

/// Results for a single match-length point (lookup only)
#[derive(Debug)]
struct MatchLengthResult {
    match_length: usize,
    seq_length: usize,
    lookup_avg_ns: u64,
    lookup_p50_ns: u64,
    lookup_p99_ns: u64,
}

/// Benchmark lookup with varying match lengths (fixed tree structure)
fn bench_sweep_match_length(args: &Args) {
    let seq_length = args.seq_length.unwrap_or(args.max_depth);
    let num_sequences = args.size / seq_length;

    if num_sequences < 1 {
        eprintln!("Error: size {} / seq_length {} = 0 sequences. Increase --size or decrease sequence length.",
                  args.size, seq_length);
        std::process::exit(1);
    }

    println!("\n=== Match Length Sweep Benchmark ===");
    println!("  Sequence length: {} blocks (fixed)", seq_length);
    println!("  Match lengths: {} to {} ({} points, log-spaced)",
             args.min_depth, args.max_depth.min(seq_length), args.sweep_points);
    println!("  Iterations per match length: {}", args.sweep_iterations);
    println!("  Tree: {} sequences, {} total blocks", num_sequences, num_sequences * seq_length);
    println!("  Workers: {}", args.num_workers);
    println!();

    // Build tree once with fixed-length sequences
    let sequences = generate_sequences(
        num_sequences,
        seq_length,
        args.num_workers,
        args.prefix_share_ratio,
    );
    let tree = build_tree(&sequences);

    // Generate match length points (capped at seq_length)
    let max_match = args.max_depth.min(seq_length);
    let match_lengths = generate_depth_points(args.min_depth, max_match, args.sweep_points);
    let mut results: Vec<MatchLengthResult> = Vec::with_capacity(match_lengths.len());

    for (idx, &match_len) in match_lengths.iter().enumerate() {
        print!("[{}/{}] match_length={}... ",
               idx + 1, match_lengths.len(), match_len);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Generate lookup queries: first match_len blocks match, rest are garbage
        let mut lookup_durations = Vec::with_capacity(args.sweep_iterations);

        for i in 0..args.sweep_iterations {
            let seq = &sequences[i % sequences.len()];

            // Build query: match_len blocks from real sequence, then garbage
            let mut query_hashes = seq.local_hashes[..match_len].to_vec();
            // Add garbage blocks to reach full seq_length
            let garbage_len = seq_length - match_len;
            query_hashes.extend(
                (0..garbage_len).map(|j| LocalBlockHash(0xBAD_C0DE_0000_0000 | ((i as u64) << 16) | (j as u64)))
            );

            let start = Instant::now();
            let _ = tree.find_matches(query_hashes, false);
            lookup_durations.push(start.elapsed());
        }

        lookup_durations.sort();
        let lookup_avg = lookup_durations.iter().sum::<Duration>() / lookup_durations.len() as u32;

        let result = MatchLengthResult {
            match_length: match_len,
            seq_length,
            lookup_avg_ns: lookup_avg.as_nanos() as u64,
            lookup_p50_ns: lookup_durations[lookup_durations.len() / 2].as_nanos() as u64,
            lookup_p99_ns: lookup_durations[lookup_durations.len() * 99 / 100].as_nanos() as u64,
        };

        println!("lookup={:.2}us",
                 result.lookup_avg_ns as f64 / 1000.0);

        results.push(result);
    }

    // Print results
    println!();
    if args.sweep_format == "csv" {
        println!("match_length,seq_length,lookup_avg_ns,lookup_p50_ns,lookup_p99_ns");
        for r in &results {
            println!("{},{},{},{},{}",
                     r.match_length, r.seq_length,
                     r.lookup_avg_ns, r.lookup_p50_ns, r.lookup_p99_ns);
        }
    } else {
        println!("{:>12} | {:>10} | {:>12} {:>12} {:>12}",
                 "match_length", "seq_length", "lookup_avg", "lookup_p50", "lookup_p99");
        println!("{}", "-".repeat(70));
        for r in &results {
            println!("{:>12} | {:>10} | {:>12} {:>12} {:>12}",
                     r.match_length, r.seq_length,
                     format_duration_ns(r.lookup_avg_ns),
                     format_duration_ns(r.lookup_p50_ns),
                     format_duration_ns(r.lookup_p99_ns));
        }
    }
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

    println!("Radix Tree Microbenchmark");
    println!("=========================\n");
    println!("Configuration:");
    println!("  Target size: {} (worker, block) pairs", args.size);
    println!("  Depth: {} blocks/sequence (= {} tokens with block_size={})",
             args.depth, args.depth * args.block_size as usize, args.block_size);
    println!("  Block size: {} tokens", args.block_size);
    println!("  Workers: {}", args.num_workers);
    println!("  Iterations: {}", args.iterations);
    println!("  Prefix share ratio: {:.1}%", args.prefix_share_ratio * 100.0);
    println!("  Seed: {}", args.seed);

    let num_sequences = args.size / args.depth;
    println!("\n  Derived: {} sequences to reach target size", num_sequences);

    match args.bench.as_str() {
        "hash" => bench_hash(&args),
        "store" => bench_store(&args),
        "remove" => bench_remove(&args),
        "lookup" => bench_lookup(&args),
        "sweep" => {
            if args.sweep_match_length {
                bench_sweep_match_length(&args);
            } else {
                bench_sweep(&args);
            }
        }
        "all" => {
            bench_hash(&args);
            bench_store(&args);
            bench_remove(&args);
            bench_lookup(&args);
        }
        _ => {
            eprintln!("Unknown benchmark type: {}. Use 'hash', 'store', 'remove', 'lookup', 'sweep', or 'all'", args.bench);
            std::process::exit(1);
        }
    }

    println!("\nBenchmark complete.");
}
