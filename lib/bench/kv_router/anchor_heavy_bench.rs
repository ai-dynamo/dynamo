// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;
use std::sync::Arc;

use clap::Parser;
use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, compute_seq_hash_for_block,
};
use dynamo_kv_router::{
    BranchShardedIndexer, ConcurrentRadixTreeCompressed, LocalBlockHash, ThreadPoolIndexer,
};
use tokio::sync::Barrier;

type Bsi = BranchShardedIndexer<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>;

#[derive(Parser, Debug, Clone)]
#[clap(
    version,
    about = "Anchor-heavy BSI steady-state Stored-event benchmark"
)]
struct Args {
    /// Number of stable workers to warm and reuse during the measured phase.
    #[clap(long, default_value = "1024")]
    workers: usize,

    /// Number of Stored events measured after warm-up.
    #[clap(long, default_value = "100000")]
    events: usize,

    /// Number of concurrent submitter tasks issuing Stored events.
    #[clap(long, default_value = "4")]
    submit_tasks: usize,

    /// Number of independent CRTC shards.
    #[clap(long, default_value = "2")]
    num_shards: usize,

    /// Number of OS event-worker threads per shard.
    #[clap(long, default_value = "4")]
    num_event_workers_per_shard: usize,

    /// Routing depth before suffixes are dispatched to a shard.
    #[clap(long, default_value = "2")]
    prefix_depth: usize,

    /// KV block size used by the indexer.
    #[clap(long, default_value = "32")]
    block_size: u32,

    /// Number of suffix blocks beyond prefix_depth in each Stored event.
    #[clap(long, default_value = "4")]
    suffix_blocks: usize,

    /// Ignored - passed by cargo bench harness.
    #[arg(long, hide = true)]
    bench: bool,
}

impl Args {
    fn validate(&self) -> anyhow::Result<()> {
        if self.workers == 0 {
            anyhow::bail!("--workers must be at least 1");
        }
        if self.events == 0 {
            anyhow::bail!("--events must be at least 1");
        }
        if self.submit_tasks == 0 {
            anyhow::bail!("--submit-tasks must be at least 1");
        }
        if self.num_shards == 0 {
            anyhow::bail!("--num-shards must be at least 1");
        }
        if self.num_event_workers_per_shard == 0 {
            anyhow::bail!("--num-event-workers-per-shard must be at least 1");
        }
        if self.prefix_depth == 0 {
            anyhow::bail!("--prefix-depth must be at least 1");
        }
        if self.suffix_blocks == 0 {
            anyhow::bail!("--suffix-blocks must be at least 1");
        }
        if self.block_size == 0 {
            anyhow::bail!("--block-size must be at least 1");
        }
        Ok(())
    }

    fn blocks_per_event(&self) -> usize {
        self.prefix_depth + self.suffix_blocks
    }
}

fn make_indexer(args: &Args) -> Arc<Bsi> {
    let shards = (0..args.num_shards)
        .map(|_| {
            ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                args.num_event_workers_per_shard,
                args.block_size,
            )
        })
        .collect();
    Arc::new(BranchShardedIndexer::new_with_options(
        shards,
        args.prefix_depth,
        args.block_size,
    ))
}

fn sequence_hashes(args: &Args, worker_id: u64, event_idx: u64) -> Vec<LocalBlockHash> {
    let mut hashes = Vec::with_capacity(args.blocks_per_event());

    for depth in 0..args.prefix_depth {
        hashes.push(LocalBlockHash(46 + depth as u64));
    }

    let suffix_seed = 1_000_000_000u64
        .wrapping_add(worker_id.wrapping_mul(65_537))
        .wrapping_add(event_idx.wrapping_mul(1_048_573));
    for suffix_idx in 0..args.suffix_blocks {
        hashes.push(LocalBlockHash(suffix_seed.wrapping_add(suffix_idx as u64)));
    }

    hashes
}

fn stored_event(args: &Args, worker_id: u64, event_id: u64, event_idx: u64) -> RouterEvent {
    let local_hashes = sequence_hashes(args, worker_id, event_idx);
    let seq_hashes = compute_seq_hash_for_block(&local_hashes);
    let blocks = local_hashes
        .iter()
        .zip(seq_hashes.iter())
        .map(|(&tokens_hash, &block_hash)| KvCacheStoredBlockData {
            tokens_hash,
            block_hash: ExternalSequenceBlockHash(block_hash),
            mm_extra_info: None,
        })
        .collect();

    RouterEvent::new(
        worker_id,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks,
            }),
        },
    )
}

fn task_range(events: usize, task_idx: usize, tasks: usize) -> Range<usize> {
    let base = events / tasks;
    let extra = events % tasks;
    let start = task_idx * base + task_idx.min(extra);
    let len = base + usize::from(task_idx < extra);
    start..start + len
}

fn percentile_us(mut latencies_ns: Vec<u64>, pct: usize) -> f64 {
    if latencies_ns.is_empty() {
        return 0.0;
    }
    latencies_ns.sort_unstable();
    let idx = (latencies_ns.len().saturating_sub(1) * pct) / 100;
    latencies_ns[idx] as f64 / 1000.0
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.validate()?;

    let indexer = make_indexer(&args);

    println!("Warming {} stable workers...", args.workers);
    for worker_id in 0..args.workers {
        indexer
            .apply_event(stored_event(&args, worker_id as u64, worker_id as u64, 0))
            .await;
    }
    indexer.flush().await;

    let barrier = Arc::new(Barrier::new(args.submit_tasks + 1));
    let mut tasks = Vec::with_capacity(args.submit_tasks);
    for task_idx in 0..args.submit_tasks {
        let args = args.clone();
        let indexer = Arc::clone(&indexer);
        let barrier = Arc::clone(&barrier);
        let range = task_range(args.events, task_idx, args.submit_tasks);
        tasks.push(tokio::spawn(async move {
            let mut latencies_ns = Vec::with_capacity(range.len());
            barrier.wait().await;
            for event_idx in range {
                let worker_id = (event_idx % args.workers) as u64;
                let event_id = args.workers as u64 + event_idx as u64;
                let start = minstant::Instant::now();
                indexer
                    .apply_event(stored_event(
                        &args,
                        worker_id,
                        event_id,
                        event_idx as u64 + 1,
                    ))
                    .await;
                latencies_ns.push(start.elapsed().as_nanos() as u64);
            }
            latencies_ns
        }));
    }

    barrier.wait().await;
    let started_at = minstant::Instant::now();

    let mut latencies_ns = Vec::with_capacity(args.events);
    for task in tasks {
        latencies_ns.extend(task.await?);
    }
    indexer.flush().await;

    let elapsed = started_at.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(1e-9);
    let event_throughput = args.events as f64 / elapsed_secs;
    let block_throughput = (args.events * args.blocks_per_event()) as f64 / elapsed_secs;
    let p99_us = percentile_us(latencies_ns, 99);
    let metrics = indexer.metrics_snapshot();

    println!();
    println!("Anchor-heavy BSI Stored-event benchmark");
    println!("workers: {}", args.workers);
    println!("stored events: {}", args.events);
    println!("blocks per event: {}", args.blocks_per_event());
    println!("submit tasks: {}", args.submit_tasks);
    println!(
        "config: {} shards x {} event workers, prefix_depth={}",
        args.num_shards, args.num_event_workers_per_shard, args.prefix_depth
    );
    println!("elapsed: {:.3}s", elapsed_secs);
    println!("stored-event throughput: {:.0} events/s", event_throughput);
    println!(
        "stored-block throughput: {:.0} block events/s",
        block_throughput
    );
    println!("apply_event p99: {:.1}us", p99_us);
    println!("anchor installs: {}", metrics.anchor_installs);
    println!("anchor reuses: {}", metrics.anchor_reuses);
    println!("remove broadcasts: {}", metrics.remove_broadcasts);

    Ok(())
}
