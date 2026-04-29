// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic logical sharding with stateful virtual-shard ownership.
//!
//! [`VirtualShardShardedIndexer`] splits routing into two layers:
//!
//! - `branch_key -> virtual_shard` is deterministic
//! - `virtual_shard -> physical_shard` is an explicit ownership table
//!
//! Unlike [`PrefixShardedIndexer`], this implementation can change ownership at
//! runtime. A background rebalancer detects hot physical shards and migrates the
//! hottest virtual shard to a cooler owner using a replay -> dual-write ->
//! ownership-switch protocol similar to the branch-level rebalancer.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use rustc_hash::FxBuildHasher;

use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

#[derive(Clone, Copy)]
struct DualWriteEntry {
    old_shard: usize,
    new_shard: usize,
}

pub struct VirtualShardShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    num_shards: usize,
    num_virtual_shards: usize,
    prefix_depth: usize,
    inherit_parent_shard: bool,

    /// Tracks which branch keys have actually been observed so that unknown
    /// branches can still early-exit even though logical routing is deterministic.
    known_branch_keys: DashMap<u64, (), FxBuildHasher>,
    /// Content-addressed mapping for continuation routing / remove routing.
    block_to_virtual_shard: DashMap<u64, usize, FxBuildHasher>,
    /// Ownership table: virtual shard index -> physical shard index.
    virtual_to_physical: Vec<AtomicUsize>,

    /// Query-rate counters used by the rebalancer.
    physical_shard_query_counts: Vec<AtomicU64>,
    virtual_shard_query_counts: Vec<AtomicU64>,
    virtual_shard_block_counts: Vec<AtomicUsize>,

    /// Migration state.
    replaying_virtual_shards: DashSet<usize, FxBuildHasher>,
    dualwrite_virtual_shards: DashMap<usize, DualWriteEntry, FxBuildHasher>,
    rebalancer_handle: Mutex<Option<tokio::task::AbortHandle>>,
    migrations_completed: AtomicU64,

    kv_block_size: u32,

    timing_calls: AtomicU64,
    timing_sum_routing_ns: AtomicU64,
    timing_sum_shard_ns: AtomicU64,
    timing_apply_events: AtomicU64,
    timing_sum_shadow_tree_ns: AtomicU64,
    find_matches_miss_count: AtomicU64,
}

impl<T: SyncIndexer> VirtualShardShardedIndexer<T> {
    pub fn new(
        shards: Vec<ThreadPoolIndexer<T>>,
        num_virtual_shards: usize,
        prefix_depth: usize,
        kv_block_size: u32,
    ) -> Self {
        Self::new_with_options(
            shards,
            num_virtual_shards,
            prefix_depth,
            kv_block_size,
            true,
        )
    }

    pub fn new_with_options(
        shards: Vec<ThreadPoolIndexer<T>>,
        num_virtual_shards: usize,
        prefix_depth: usize,
        kv_block_size: u32,
        inherit_parent_shard: bool,
    ) -> Self {
        assert!(
            !shards.is_empty(),
            "Must provide at least one physical shard"
        );
        assert!(
            num_virtual_shards > 0,
            "Must provide at least one virtual shard"
        );

        let num_shards = shards.len();
        let shards: Vec<Arc<ThreadPoolIndexer<T>>> = shards.into_iter().map(Arc::new).collect();
        let virtual_to_physical = (0..num_virtual_shards)
            .map(|virtual_idx| AtomicUsize::new(virtual_idx % num_shards))
            .collect();

        Self {
            shards,
            num_shards,
            num_virtual_shards,
            prefix_depth: prefix_depth.max(1),
            inherit_parent_shard,
            known_branch_keys: DashMap::with_hasher(FxBuildHasher),
            block_to_virtual_shard: DashMap::with_hasher(FxBuildHasher),
            virtual_to_physical,
            physical_shard_query_counts: (0..num_shards).map(|_| AtomicU64::new(0)).collect(),
            virtual_shard_query_counts: (0..num_virtual_shards)
                .map(|_| AtomicU64::new(0))
                .collect(),
            virtual_shard_block_counts: (0..num_virtual_shards)
                .map(|_| AtomicUsize::new(0))
                .collect(),
            replaying_virtual_shards: DashSet::with_hasher(FxBuildHasher),
            dualwrite_virtual_shards: DashMap::with_hasher(FxBuildHasher),
            rebalancer_handle: Mutex::new(None),
            migrations_completed: AtomicU64::new(0),
            kv_block_size,
            timing_calls: AtomicU64::new(0),
            timing_sum_routing_ns: AtomicU64::new(0),
            timing_sum_shard_ns: AtomicU64::new(0),
            timing_apply_events: AtomicU64::new(0),
            timing_sum_shadow_tree_ns: AtomicU64::new(0),
            find_matches_miss_count: AtomicU64::new(0),
        }
    }

    pub fn start_rebalancer(
        self: Arc<Self>,
        interval: std::time::Duration,
        imbalance_threshold: f64,
        dual_write_window: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(&self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                match weak.upgrade() {
                    Some(idx) => {
                        idx.rebalance_once(imbalance_threshold, dual_write_window)
                            .await
                    }
                    None => break,
                }
            }
        });
        *self.rebalancer_handle.lock().unwrap() = Some(handle.abort_handle());
        handle
    }

    pub fn num_virtual_shards(&self) -> usize {
        self.num_virtual_shards
    }

    pub fn virtual_owner(&self, virtual_shard: usize) -> usize {
        self.virtual_to_physical[virtual_shard].load(Ordering::Relaxed)
    }

    fn physical_shard_for_virtual(&self, virtual_shard: usize) -> usize {
        self.virtual_owner(virtual_shard) % self.num_shards
    }

    fn branch_key_for_local_hashes(&self, hashes: &[LocalBlockHash]) -> u64 {
        if hashes.is_empty() {
            return 0;
        }
        let k = self.prefix_depth.min(hashes.len());
        let mut h = FNV_OFFSET_BASIS;
        for block in &hashes[..k] {
            for b in block.0.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    fn branch_key_for_stored_blocks(&self, blocks: &[KvCacheStoredBlockData]) -> u64 {
        if blocks.is_empty() {
            return 0;
        }
        let k = self.prefix_depth.min(blocks.len());
        let mut h = FNV_OFFSET_BASIS;
        for block in &blocks[..k] {
            for b in block.tokens_hash.0.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    fn virtual_shard_for_branch_key(&self, branch_key: u64) -> usize {
        branch_key as usize % self.num_virtual_shards
    }

    fn dispatch_find_matches(
        &self,
        shard_idx: usize,
        sequence: Vec<LocalBlockHash>,
    ) -> impl std::future::Future<Output = Result<OverlapScores, KvRouterError>> + '_ {
        self.shards[shard_idx].find_matches(sequence)
    }

    fn merge_overlap_scores(mut a: OverlapScores, b: OverlapScores) -> OverlapScores {
        for (worker, score) in b.scores {
            let entry = a.scores.entry(worker).or_insert(0);
            *entry = (*entry).max(score);
        }
        for (worker, size) in b.tree_sizes {
            let entry = a.tree_sizes.entry(worker).or_insert(0);
            *entry = (*entry).max(size);
        }
        if b.frequencies.len() > a.frequencies.len() {
            a.frequencies.resize(b.frequencies.len(), 0);
        }
        for (i, &freq) in b.frequencies.iter().enumerate() {
            a.frequencies[i] = a.frequencies[i].max(freq);
        }
        a
    }

    fn saturating_sub_usize(counter: &AtomicUsize, value: usize) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(value))
        });
    }

    pub async fn rebalance_once(
        &self,
        imbalance_threshold: f64,
        dual_write_window: std::time::Duration,
    ) {
        if self.num_shards < 2 {
            return;
        }
        if !self.replaying_virtual_shards.is_empty() || !self.dualwrite_virtual_shards.is_empty() {
            return;
        }

        let counts: Vec<usize> = self
            .shards
            .iter()
            .map(|shard| {
                shard
                    .shard_sizes()
                    .into_iter()
                    .map(|snapshot| snapshot.block_count)
                    .sum()
            })
            .collect();
        let total: usize = counts.iter().sum();
        if total == 0 {
            return;
        }

        let avg = total as f64 / self.num_shards as f64;
        let (hot_shard, &max_count) = counts.iter().enumerate().max_by_key(|&(_, c)| c).unwrap();
        if max_count as f64 / avg < imbalance_threshold {
            return;
        }

        let cool_shard = counts
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != hot_shard)
            .min_by_key(|&(_, c)| c)
            .unwrap()
            .0;

        let hot_virtual = (0..self.num_virtual_shards)
            .filter(|&virtual_shard| self.virtual_owner(virtual_shard) == hot_shard)
            .filter(|virtual_shard| !self.replaying_virtual_shards.contains(virtual_shard))
            .filter(|virtual_shard| !self.dualwrite_virtual_shards.contains_key(virtual_shard))
            .map(|virtual_shard| {
                (
                    virtual_shard,
                    self.virtual_shard_block_counts[virtual_shard].load(Ordering::Relaxed),
                )
            })
            .max_by_key(|&(_, hits)| hits);

        let Some((virtual_shard, virtual_hits)) = hot_virtual else {
            return;
        };
        if virtual_hits == 0 {
            return;
        }

        self.migrate_virtual_shard(virtual_shard, hot_shard, cool_shard, dual_write_window)
            .await;
    }

    async fn migrate_virtual_shard(
        &self,
        virtual_shard: usize,
        old_shard: usize,
        new_shard: usize,
        dual_write_window: std::time::Duration,
    ) {
        if !self.replaying_virtual_shards.insert(virtual_shard) {
            return;
        }

        let all_events = match self.shards[old_shard].dump_events().await {
            Ok(events) => events,
            Err(_) => {
                self.replaying_virtual_shards.remove(&virtual_shard);
                return;
            }
        };

        let replay_events: Vec<RouterEvent> = all_events
            .into_iter()
            .filter(|event| match &event.event.data {
                KvCacheEventData::Stored(store_data) => store_data
                    .blocks
                    .first()
                    .and_then(|block| self.block_to_virtual_shard.get(&block.block_hash.0))
                    .map(|v| *v == virtual_shard)
                    .unwrap_or(false),
                _ => false,
            })
            .collect();

        for event in replay_events {
            self.shards[new_shard].apply_event(event).await;
        }

        self.dualwrite_virtual_shards.insert(
            virtual_shard,
            DualWriteEntry {
                old_shard,
                new_shard,
            },
        );
        self.replaying_virtual_shards.remove(&virtual_shard);

        <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(&*self.shards[new_shard]).await;
        tokio::time::sleep(dual_write_window).await;

        self.virtual_to_physical[virtual_shard].store(new_shard, Ordering::Relaxed);
        let hits = self.virtual_shard_query_counts[virtual_shard].swap(0, Ordering::Relaxed);
        let old_total = self.physical_shard_query_counts[old_shard].load(Ordering::Relaxed);
        self.physical_shard_query_counts[old_shard]
            .fetch_sub(hits.min(old_total), Ordering::Relaxed);
        self.physical_shard_query_counts[new_shard].fetch_add(hits, Ordering::Relaxed);
        self.dualwrite_virtual_shards.remove(&virtual_shard);
        self.migrations_completed.fetch_add(1, Ordering::Relaxed);
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for VirtualShardShardedIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let t_routing = std::time::Instant::now();
        let branch_key = self.branch_key_for_local_hashes(&sequence);
        if self.known_branch_keys.get(&branch_key).is_none() {
            self.find_matches_miss_count.fetch_add(1, Ordering::Relaxed);
            return Ok(OverlapScores::new());
        }

        let virtual_shard = self.virtual_shard_for_branch_key(branch_key);
        if let Some(entry) = self.dualwrite_virtual_shards.get(&virtual_shard) {
            let DualWriteEntry {
                old_shard,
                new_shard,
            } = *entry;
            drop(entry);
            let routing_ns = t_routing.elapsed().as_nanos() as u64;
            self.virtual_shard_query_counts[virtual_shard].fetch_add(1, Ordering::Relaxed);
            let (r_old, r_new) = tokio::join!(
                self.dispatch_find_matches(old_shard, sequence.clone()),
                self.dispatch_find_matches(new_shard, sequence),
            );
            self.timing_calls.fetch_add(1, Ordering::Relaxed);
            self.timing_sum_routing_ns
                .fetch_add(routing_ns, Ordering::Relaxed);
            return Ok(Self::merge_overlap_scores(r_old?, r_new?));
        }

        let physical_shard = self.physical_shard_for_virtual(virtual_shard);
        let routing_ns = t_routing.elapsed().as_nanos() as u64;
        self.physical_shard_query_counts[physical_shard].fetch_add(1, Ordering::Relaxed);
        self.virtual_shard_query_counts[virtual_shard].fetch_add(1, Ordering::Relaxed);

        let t_shard = std::time::Instant::now();
        let result = self.shards[physical_shard].find_matches(sequence).await;
        let shard_ns = t_shard.elapsed().as_nanos() as u64;

        self.timing_calls.fetch_add(1, Ordering::Relaxed);
        self.timing_sum_routing_ns
            .fetch_add(routing_ns, Ordering::Relaxed);
        self.timing_sum_shard_ns
            .fetch_add(shard_ns, Ordering::Relaxed);
        result
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.kv_block_size,
            BlockHashOptions {
                lora_name,
                is_eagle,
                ..Default::default()
            },
        );
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let t_shadow = std::time::Instant::now();
                let branch_key = self.branch_key_for_stored_blocks(&store_data.blocks);
                self.known_branch_keys.insert(branch_key, ());

                let virtual_shard = if self.inherit_parent_shard {
                    if let Some(parent_hash) = &store_data.parent_hash {
                        self.block_to_virtual_shard
                            .get(&parent_hash.0)
                            .map(|v| *v)
                            .unwrap_or_else(|| self.virtual_shard_for_branch_key(branch_key))
                    } else {
                        self.virtual_shard_for_branch_key(branch_key)
                    }
                } else {
                    self.virtual_shard_for_branch_key(branch_key)
                };

                for block in &store_data.blocks {
                    self.block_to_virtual_shard
                        .insert(block.block_hash.0, virtual_shard);
                }
                self.virtual_shard_block_counts[virtual_shard]
                    .fetch_add(store_data.blocks.len(), Ordering::Relaxed);

                let shadow_ns = t_shadow.elapsed().as_nanos() as u64;
                self.timing_apply_events.fetch_add(1, Ordering::Relaxed);
                self.timing_sum_shadow_tree_ns
                    .fetch_add(shadow_ns, Ordering::Relaxed);

                let dualwrite_target = if !self.replaying_virtual_shards.contains(&virtual_shard) {
                    self.dualwrite_virtual_shards
                        .get(&virtual_shard)
                        .map(|entry| *entry)
                } else {
                    None
                };

                if let Some(DualWriteEntry {
                    old_shard,
                    new_shard,
                }) = dualwrite_target
                {
                    self.shards[new_shard].apply_event(event.clone()).await;
                    self.shards[old_shard].apply_event(event).await;
                } else {
                    let physical_shard = self.physical_shard_for_virtual(virtual_shard);
                    self.shards[physical_shard].apply_event(event).await;
                }
            }
            KvCacheEventData::Removed(remove_data) => {
                let t_shadow = std::time::Instant::now();
                let mut per_shard_blocks = vec![Vec::new(); self.num_shards];

                for block_hash in &remove_data.block_hashes {
                    if let Some((_, virtual_shard)) =
                        self.block_to_virtual_shard.remove(&block_hash.0)
                    {
                        Self::saturating_sub_usize(
                            &self.virtual_shard_block_counts[virtual_shard],
                            1,
                        );
                        if let Some(entry) = self.dualwrite_virtual_shards.get(&virtual_shard) {
                            per_shard_blocks[entry.old_shard].push(block_hash.clone());
                            per_shard_blocks[entry.new_shard].push(block_hash.clone());
                        } else {
                            let physical_shard = self.physical_shard_for_virtual(virtual_shard);
                            per_shard_blocks[physical_shard].push(block_hash.clone());
                        }
                    } else {
                        for blocks in &mut per_shard_blocks {
                            blocks.push(block_hash.clone());
                        }
                    }
                }

                let shadow_ns = t_shadow.elapsed().as_nanos() as u64;
                self.timing_apply_events.fetch_add(1, Ordering::Relaxed);
                self.timing_sum_shadow_tree_ns
                    .fetch_add(shadow_ns, Ordering::Relaxed);

                for (physical_shard, blocks) in per_shard_blocks.into_iter().enumerate() {
                    if blocks.is_empty() {
                        continue;
                    }
                    let routed = RouterEvent {
                        worker_id: event.worker_id,
                        storage_tier: event.storage_tier,
                        event: KvCacheEvent {
                            event_id: event.event.event_id,
                            dp_rank: event.event.dp_rank,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: blocks,
                            }),
                        },
                    };
                    self.shards[physical_shard].apply_event(routed).await;
                }
            }
            KvCacheEventData::Cleared => {
                self.known_branch_keys.clear();
                self.block_to_virtual_shard.clear();
                for counter in &self.physical_shard_query_counts {
                    counter.store(0, Ordering::Relaxed);
                }
                for counter in &self.virtual_shard_query_counts {
                    counter.store(0, Ordering::Relaxed);
                }
                for counter in &self.virtual_shard_block_counts {
                    counter.store(0, Ordering::Relaxed);
                }
                self.replaying_virtual_shards.clear();
                self.dualwrite_virtual_shards.clear();
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker: WorkerId) {
        for shard in &self.shards {
            shard.remove_worker(worker).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        for shard in &self.shards {
            shard.remove_worker_dp_rank(worker, dp_rank).await;
        }
    }

    fn shutdown(&self) {
        if let Some(handle) = self.rebalancer_handle.lock().unwrap().take() {
            handle.abort();
        }
        for shard in &self.shards {
            shard.shutdown();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all = Vec::new();
        for shard in &self.shards {
            all.extend(shard.dump_events().await?);
        }
        Ok(all)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let idx = worker.worker_id as usize % self.num_shards;
        self.shards[idx]
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }

    async fn flush(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            total += <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(shard).await;
        }
        total
    }

    fn timing_report(&self) -> String {
        let dispatched = self.timing_calls.load(Ordering::Relaxed);
        let misses = self.find_matches_miss_count.load(Ordering::Relaxed);
        let total_calls = dispatched + misses;
        if total_calls == 0 {
            return String::new();
        }

        let miss_pct = 100.0 * misses as f64 / total_calls as f64;
        let avg_routing_ns = if dispatched > 0 {
            self.timing_sum_routing_ns.load(Ordering::Relaxed) / dispatched
        } else {
            0
        };
        let avg_shard_us = if dispatched > 0 {
            self.timing_sum_shard_ns.load(Ordering::Relaxed) / dispatched / 1000
        } else {
            0
        };
        let owner_dist: Vec<String> = (0..self.num_shards)
            .map(|physical_shard| {
                let owned = (0..self.num_virtual_shards)
                    .filter(|&virtual_shard| self.virtual_owner(virtual_shard) == physical_shard)
                    .count();
                format!("shard[{physical_shard}]={owned}")
            })
            .collect();

        format!(
            "VirtualShardShardedIndexer find_matches ({total_calls} total: {dispatched} dispatched, \
             {misses} early-exit / {miss_pct:.1}% miss):\n  \
             avg routing        = {avg_routing_ns}ns  (virtual-shard lookup + owner table)\n  \
             avg shard          = {avg_shard_us}µs  (CRTC traversal, inline on caller thread)\n  \
             branches known     = {}\n  \
             virtual shards     = {}  ({})\n  \
             migrations done    = {}",
            self.known_branch_keys.len(),
            self.num_virtual_shards,
            owner_dist.join(", "),
            self.migrations_completed.load(Ordering::Relaxed)
        )
    }

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                shard.shard_sizes().into_iter().map(move |mut s| {
                    s.shard_idx = idx;
                    s
                })
            })
            .collect()
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.node_edge_lengths())
            .collect()
    }
}
