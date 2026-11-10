// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's routing for the pool benchmark: deliberately minimal, so
//! the KV index and its ingest are what the benchmark measures.
//!
//! Per request: hash the prompt, look it up in the per-worker index, score each
//! pool by its best member's matched blocks, and take the best pool, breaking
//! ties round-robin. No booking, no load, no serial stage.

use std::collections::hash_map::Entry;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, WorkerId, compute_block_hash_for_seq,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use rustc_hash::FxHashMap;
use serde::Serialize;

/// Which pool each worker belongs to. Workers only join during a run.
pub struct PoolMap {
    workers: RwLock<FxHashMap<WorkerId, u32>>,
    pools: usize,
}

impl PoolMap {
    pub fn new(pools: usize) -> Self {
        Self {
            workers: RwLock::new(FxHashMap::default()),
            pools,
        }
    }

    pub fn pools(&self) -> usize {
        self.pools
    }

    /// Record `worker` in `pool`; a worker already mapped keeps its first pool.
    pub fn insert(&self, worker: WorkerId, pool: u32) {
        assert!((pool as usize) < self.pools, "pool {pool} out of range");
        if self.workers.read().unwrap().contains_key(&worker) {
            return;
        }
        if let Entry::Vacant(slot) = self.workers.write().unwrap().entry(worker) {
            slot.insert(pool);
        }
    }

    /// Workers mapped to each pool.
    pub fn sizes(&self) -> Vec<u32> {
        let mut sizes = vec![0u32; self.pools];
        for &pool in self.workers.read().unwrap().values() {
            sizes[pool as usize] += 1;
        }
        sizes
    }

    /// Each pool's best member score into `best` (one slot per pool). Workers not
    /// yet mapped are ignored.
    pub fn best_per_pool(&self, scores: &OverlapScores, best: &mut [u32]) {
        best.fill(0);
        let workers = self.workers.read().unwrap();
        for (worker, &score) in &scores.scores {
            if let Some(&pool) = workers.get(&worker.worker_id) {
                let slot = &mut best[pool as usize];
                *slot = (*slot).max(score);
            }
        }
    }
}

/// The highest-scoring pool; among `n` pools tied at the top, the `rr % n`-th.
/// Returns `(pool, matched blocks, tied pools)`.
pub fn pick(best: &[u32], rr: u64) -> (u32, u32, u32) {
    let top = best.iter().copied().max().unwrap_or(0);
    let tied = best.iter().filter(|&&score| score == top).count() as u64;
    let nth = rr % tied.max(1);
    let pool = best
        .iter()
        .enumerate()
        .filter(|&(_, &score)| score == top)
        .nth(nth as usize)
        .map_or(0, |(pool, _)| pool as u32);
    (pool, top, tied as u32)
}

#[derive(Debug, Serialize)]
pub struct RouteReply {
    pub pool: u32,
    pub matched_blocks: u32,
    pub tied: u32,
}

/// Latency buckets are powers of two microseconds: bucket `i` holds [2^i, 2^(i+1)).
pub const LATENCY_BUCKETS: usize = 32;

#[derive(Default)]
pub struct RouteStats {
    pub routes: AtomicU64,
    pub errors: AtomicU64,
    pub prompt_blocks: AtomicU64,
    pub matched_blocks: AtomicU64,
    pub latency_us_sum: AtomicU64,
    pub latency: [AtomicU64; LATENCY_BUCKETS],
}

impl RouteStats {
    pub fn record(&self, elapsed: Duration) {
        let us = elapsed.as_micros().min(u64::MAX as u128) as u64;
        self.latency_us_sum.fetch_add(us, Ordering::Relaxed);
        let bucket = (63 - us.max(1).leading_zeros() as usize).min(LATENCY_BUCKETS - 1);
        self.latency[bucket].fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "routes": self.routes.load(Ordering::Relaxed),
            "errors": self.errors.load(Ordering::Relaxed),
            "prompt_blocks": self.prompt_blocks.load(Ordering::Relaxed),
            "matched_blocks": self.matched_blocks.load(Ordering::Relaxed),
            "latency_us_sum": self.latency_us_sum.load(Ordering::Relaxed),
            "latency_log2_us": self.latency.iter().map(|b| b.load(Ordering::Relaxed)).collect::<Vec<_>>(),
        })
    }
}

pub struct Router {
    pub indexer: Indexer,
    pub pools: std::sync::Arc<PoolMap>,
    pub block_size: u32,
    pub stats: RouteStats,
    rr: AtomicU64,
}

impl Router {
    pub fn new(indexer: Indexer, pools: std::sync::Arc<PoolMap>, block_size: u32) -> Self {
        Self {
            indexer,
            pools,
            block_size,
            stats: RouteStats::default(),
            rr: AtomicU64::new(0),
        }
    }

    pub async fn route(&self, tokens: &[u32]) -> anyhow::Result<RouteReply> {
        let hashes =
            compute_block_hash_for_seq(tokens, self.block_size, BlockHashOptions::default());
        let blocks = hashes.len() as u64;
        let scores = self.indexer.find_matches(hashes).await?;
        let mut best = vec![0u32; self.pools.pools()];
        self.pools.best_per_pool(&scores, &mut best);
        let (pool, matched_blocks, tied) = pick(&best, self.rr.fetch_add(1, Ordering::Relaxed));
        self.stats
            .prompt_blocks
            .fetch_add(blocks, Ordering::Relaxed);
        self.stats
            .matched_blocks
            .fetch_add(u64::from(matched_blocks), Ordering::Relaxed);
        Ok(RouteReply {
            pool,
            matched_blocks,
            tied,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::WorkerWithDpRank;

    fn scores(entries: &[(WorkerId, u32)]) -> OverlapScores {
        let mut scores = OverlapScores::new();
        for &(worker, score) in entries {
            scores
                .scores
                .insert(WorkerWithDpRank::new(worker, 0), score);
        }
        scores
    }

    #[test]
    fn pool_score_is_best_member_and_ties_rotate() {
        let map = PoolMap::new(3);
        for (worker, pool) in [(1, 0), (2, 0), (3, 1), (4, 2)] {
            map.insert(worker, pool);
        }
        map.insert(1, 2); // a worker keeps its first pool
        assert_eq!(map.sizes(), vec![2, 1, 1]);

        let mut best = vec![0; 3];
        map.best_per_pool(&scores(&[(1, 3), (2, 7), (3, 5), (9, 50)]), &mut best);
        assert_eq!(
            best,
            vec![7, 5, 0],
            "best member per pool; unmapped workers ignored"
        );
        assert_eq!(pick(&best, 0), (0, 7, 1));
        assert_eq!(pick(&best, 5), (0, 7, 1));

        map.best_per_pool(&scores(&[(1, 4), (3, 4)]), &mut best);
        assert_eq!(best, vec![4, 4, 0]);
        let picks: Vec<u32> = (0..4).map(|rr| pick(&best, rr).0).collect();
        assert_eq!(
            picks,
            vec![0, 1, 0, 1],
            "ties rotate over the tied pools only"
        );

        best.fill(0);
        let picks: Vec<u32> = (0..3).map(|rr| pick(&best, rr).0).collect();
        assert_eq!(
            picks,
            vec![0, 1, 2],
            "no overlap anywhere rotates over every pool"
        );
    }

    #[test]
    fn latency_buckets_are_log2_microseconds() {
        let stats = RouteStats::default();
        stats.record(Duration::from_micros(1));
        stats.record(Duration::from_micros(3));
        stats.record(Duration::from_micros(1500));
        assert_eq!(stats.latency[0].load(Ordering::Relaxed), 1);
        assert_eq!(stats.latency[1].load(Ordering::Relaxed), 1);
        assert_eq!(stats.latency[10].load(Ordering::Relaxed), 1);
    }
}
