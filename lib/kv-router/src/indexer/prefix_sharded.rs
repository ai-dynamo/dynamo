// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefix-based sharding over `ThreadPoolIndexer<T>`.
//!
//! [`PrefixShardedIndexer`] partitions the prefix space across N independent
//! [`ThreadPoolIndexer`] instances (shards).  The shard for a given prefix is
//! determined by hashing the first `prefix_depth` block hashes of the sequence.
//!
//! ## Sharding strategy
//!
//! - **find_matches**: hashes `sequence[0..min(prefix_depth, len)]` to select
//!   exactly one shard — no scatter-gather, so throughput scales linearly with
//!   shard count.
//! - **Stored events** (root, `parent_hash = None`): shard is computed by
//!   hashing the first `min(prefix_depth, blocks.len())` `tokens_hash` values.
//!   All new `block_hash → shard` mappings are recorded in a global index.
//! - **Stored events** (continuation, `parent_hash = Some(h)`): shard is
//!   looked up from the `block_hash → shard` index using the parent hash,
//!   ensuring the entire chain lives in one shard.
//! - **Removed events**: each `ExternalSequenceBlockHash` is looked up in the
//!   index and the Remove is routed to the owning shard only.  The mapping is
//!   removed from the index after routing.
//! - **Cleared events**: broadcast to all shards (a worker may have blocks on
//!   any shard after reusing different prefixes over its lifetime).
//! - **remove_worker**: broadcast to all shards for the same reason.
//!
//! ## Consistency guarantee
//!
//! A root `Stored` event for worker W that has fewer than `prefix_depth` blocks
//! will be hashed over however many blocks are available.  The corresponding
//! `find_matches` query uses the same `min(prefix_depth, len)` logic, so both
//! agree as long as no query issues a shorter sequence than the root event.  In
//! practice set `prefix_depth` to a value ≤ the typical prefill batch size.
//!
//! ## Load balance note
//!
//! If most requests share the same system-prompt prefix, all load concentrates
//! on one shard (hot prefix skew).  For diverse workloads, load is near-uniform.
//! See the sharding design doc for mitigation strategies (consistent hashing,
//! power-of-two-choices).

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

// ---------------------------------------------------------------------------
// PrefixShardedIndexer
// ---------------------------------------------------------------------------

/// Prefix-sharded wrapper over N [`ThreadPoolIndexer<T>`] instances.
///
/// Construct with [`PrefixShardedIndexer::new`]; for CRTC specifically,
/// use [`PrefixShardedIndexer::new_crtc`].
pub struct PrefixShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    num_shards: usize,
    /// Number of prefix blocks hashed for shard selection.  Must be >= 1.
    prefix_depth: usize,
    /// When `true` (default), a Stored event with a known parent block inherits
    /// the parent's shard instead of re-hashing the prefix.  This keeps all
    /// descendants of a sequence on the same shard (correct for CRTC tree
    /// consistency) but can cause severe imbalance on traces with a long shared
    /// prefix (e.g. system prompts): the first event wins the shard for the
    /// entire shared subtree.  Set to `false` to always route by prefix hash,
    /// which gives the expected statistical balance at the cost of potentially
    /// splitting a prefix tree across shards.
    inherit_parent_shard: bool,
    /// Global `ExternalSequenceBlockHash (as u64) → shard index` index.
    /// Written on Stored, read+deleted on Removed.
    block_to_shard: DashMap<u64, usize, FxBuildHasher>,
    kv_block_size: u32,
    /// Number of completed `find_matches` calls (for timing averages).
    timing_calls: AtomicU64,
    /// Sum of FNV routing hash time (ns): `shard_for_local_hashes()` only.
    timing_sum_routing_ns: AtomicU64,
    /// Sum of delegated `ThreadPoolIndexer::find_matches` time (ns).
    timing_sum_shard_ns: AtomicU64,
    /// Number of `apply_event` calls that touched the shadow tree (Stored + Removed).
    timing_apply_events: AtomicU64,
    /// Sum of shadow-tree time (ns) in `apply_event`: DashMap inserts (Stored) and
    /// DashMap lookups+removes (Removed).  Cleared events are excluded because they
    /// do not touch `block_to_shard`.
    timing_sum_shadow_tree_ns: AtomicU64,
}

impl<T: SyncIndexer> PrefixShardedIndexer<T> {
    /// Create a prefix-sharded indexer from pre-built [`ThreadPoolIndexer`] shards.
    ///
    /// # Arguments
    ///
    /// * `shards` - One `ThreadPoolIndexer` per shard.
    /// * `prefix_depth` - Number of prefix blocks to hash for routing.  Clamped
    ///   to ≥ 1.  Larger values give better balance when prefixes are diverse;
    ///   value of 1 (hash only the first block) is always safe and consistent.
    /// * `kv_block_size` - Block size for KV cache.
    ///
    /// # Panics
    ///
    /// Panics if `shards` is empty.
    pub fn new(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
    ) -> Self {
        Self::new_with_options(shards, prefix_depth, kv_block_size, true)
    }

    pub fn new_with_options(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
        inherit_parent_shard: bool,
    ) -> Self {
        assert!(!shards.is_empty(), "Must provide at least one shard");
        let num_shards = shards.len();

        let shards: Vec<Arc<ThreadPoolIndexer<T>>> =
            shards.into_iter().map(Arc::new).collect();

        Self {
            shards,
            num_shards,
            prefix_depth: prefix_depth.max(1),
            inherit_parent_shard,
            block_to_shard: DashMap::with_hasher(FxBuildHasher),
            kv_block_size,
            timing_calls: AtomicU64::new(0),
            timing_sum_routing_ns: AtomicU64::new(0),
            timing_sum_shard_ns: AtomicU64::new(0),
            timing_apply_events: AtomicU64::new(0),
            timing_sum_shadow_tree_ns: AtomicU64::new(0),
        }
    }

    /// Hash the first `min(prefix_depth, k)` `LocalBlockHash` values to a shard index.
    ///
    /// Returns 0 for an empty slice.
    fn shard_for_local_hashes(&self, hashes: &[LocalBlockHash]) -> usize {
        if hashes.is_empty() {
            return 0;
        }
        let k = self.prefix_depth.min(hashes.len());
        let mut h: u64 = 14695981039346656037; // FNV-1a 64-bit offset basis
        for block in &hashes[..k] {
            let bytes = block.0.to_le_bytes();
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211); // FNV prime
            }
        }
        h as usize % self.num_shards
    }

    /// Hash the first `min(prefix_depth, blocks.len())` `tokens_hash` values to a shard index.
    ///
    /// Returns 0 for an empty slice.
    fn shard_for_stored_blocks(&self, blocks: &[KvCacheStoredBlockData]) -> usize {
        if blocks.is_empty() {
            return 0;
        }
        let k = self.prefix_depth.min(blocks.len());
        let mut h: u64 = 14695981039346656037;
        for block in &blocks[..k] {
            let bytes = block.tokens_hash.0.to_le_bytes();
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h as usize % self.num_shards
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for PrefixShardedIndexer<T> {
    /// Route to a single shard determined by the first `prefix_depth` block hashes.
    ///
    /// This is the primary performance advantage over worker-based sharding:
    /// no scatter-gather, so throughput scales linearly with shard count.
    ///
    /// Timing breakdown recorded per call:
    /// - **routing** — `shard_for_local_hashes()`: FNV-1a over `prefix_depth`
    ///   blocks, typically < 100 ns regardless of tree size.
    /// - **shard** — the delegated `find_matches()` call (inline CRTC traversal
    ///   on the calling tokio thread).
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let t_routing = std::time::Instant::now();
        let shard_idx = self.shard_for_local_hashes(&sequence);
        let routing_ns = t_routing.elapsed().as_nanos() as u64;

        let t_shard = std::time::Instant::now();
        let result = self.shards[shard_idx].find_matches(sequence).await;
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
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None, lora_name);
        let shard_idx = self.shard_for_local_hashes(&sequence);
        self.shards[shard_idx].find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(store_data) => {
                // --- Shadow-tree phase: DashMap lookup (parent) + inserts (all blocks) ---
                let t_shadow = std::time::Instant::now();

                // Determine target shard: inherit from parent if available, else hash prefix.
                let shard_idx = if self.inherit_parent_shard {
                    if let Some(parent_hash) = &store_data.parent_hash {
                        self.block_to_shard
                            .get(&parent_hash.0)
                            .map(|v| *v)
                            .unwrap_or_else(|| {
                                // Parent not recorded (out-of-order or evicted from index).
                                // Fall back to hashing this event's own blocks.
                                self.shard_for_stored_blocks(&store_data.blocks)
                            })
                    } else {
                        // Root event: hash the first prefix_depth blocks.
                        self.shard_for_stored_blocks(&store_data.blocks)
                    }
                } else {
                    // Always route by prefix hash regardless of parent.
                    self.shard_for_stored_blocks(&store_data.blocks)
                };

                // Record every new block → shard mapping before dispatching, so that
                // a fast continuation event can find these entries immediately.
                for block in &store_data.blocks {
                    self.block_to_shard.insert(block.block_hash.0, shard_idx);
                }

                let shadow_ns = t_shadow.elapsed().as_nanos() as u64;
                self.timing_apply_events.fetch_add(1, Ordering::Relaxed);
                self.timing_sum_shadow_tree_ns
                    .fetch_add(shadow_ns, Ordering::Relaxed);

                self.shards[shard_idx].apply_event(event).await;
            }

            KvCacheEventData::Removed(remove_data) => {
                // --- Shadow-tree phase: DashMap remove for each block ---
                let t_shadow = std::time::Instant::now();

                // Route each block to its recorded shard.  Group into per-shard
                // Remove events so each shard receives one batched Remove.
                let mut shard_blocks: Vec<Vec<ExternalSequenceBlockHash>> =
                    vec![Vec::new(); self.num_shards];

                for &block_hash in &remove_data.block_hashes {
                    let shard_idx = self
                        .block_to_shard
                        .remove(&block_hash.0)
                        .map(|(_, s)| s)
                        // Unknown block (already removed or never seen): default to shard 0.
                        // The CRTC will handle the missing block as a no-op.
                        .unwrap_or(0);
                    shard_blocks[shard_idx].push(block_hash);
                }

                let shadow_ns = t_shadow.elapsed().as_nanos() as u64;
                self.timing_apply_events.fetch_add(1, Ordering::Relaxed);
                self.timing_sum_shadow_tree_ns
                    .fetch_add(shadow_ns, Ordering::Relaxed);

                for (shard_idx, blocks) in shard_blocks.into_iter().enumerate() {
                    if blocks.is_empty() {
                        continue;
                    }
                    let shard_event = RouterEvent {
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
                    self.shards[shard_idx].apply_event(shard_event).await;
                }
            }

            KvCacheEventData::Cleared => {
                // A Cleared event wipes all blocks for this worker.  Since a
                // worker may have blocks in multiple shards (different prefixes
                // stored over its lifetime), broadcast to all shards.
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        // Broadcast: a worker may have blocks spread across multiple shards.
        for shard in &self.shards {
            shard.remove_worker(worker_id).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        for shard in &self.shards {
            shard.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
    }

    fn shutdown(&self) {
        for shard in &self.shards {
            shard.shutdown();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();
        for shard in &self.shards {
            all_events.extend(shard.dump_events().await?);
        }
        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // No-op: pruning not supported in PrefixShardedIndexer.
        Ok(())
    }

    async fn flush(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            // Explicitly call the trait method to avoid shadowing by the inherent
            // `ThreadPoolIndexer::flush() -> ()`.
            total += <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(shard).await;
        }
        total
    }

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                let node_count = shard.node_edge_lengths().len();
                shard.shard_sizes().into_iter().map(move |mut s| {
                    s.shard_idx = idx;
                    s.node_count = node_count;
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

    /// Break down average `find_matches` time into:
    /// - **routing** — FNV-1a hash over `prefix_depth` blocks to select a shard
    ///   (nanoseconds; negligible at all scales)
    /// - **shard** — CRTC traversal on the selected shard, either inline on the
    ///   caller's thread (no read pools) or via a per-shard dedicated OS thread pool
    ///
    /// If routing overhead is confirmed negligible but p99 is still higher than
    /// the single-CRTC baseline, the cause is write-path contention: the
    /// `block_to_shard` DashMap inserts in `apply_event` add cache pressure that
    /// slightly worsens CRTC lock tail latency.
    fn timing_report(&self) -> String {
        let calls = self.timing_calls.load(Ordering::Relaxed);
        if calls == 0 {
            return String::new();
        }
        let avg_routing_ns = self.timing_sum_routing_ns.load(Ordering::Relaxed) / calls;
        let avg_shard_us = self.timing_sum_shard_ns.load(Ordering::Relaxed) / calls / 1000;
        let avg_outer_us = avg_shard_us + avg_routing_ns / 1000;
        let routing_pct = if avg_outer_us > 0 {
            100.0 * (avg_routing_ns / 1000) as f64 / avg_outer_us as f64
        } else {
            0.0
        };
        let mode = "inline on caller thread";

        let apply_events = self.timing_apply_events.load(Ordering::Relaxed);
        let shadow_tree_size = self.block_to_shard.len();
        let shadow_section = if apply_events > 0 {
            let avg_shadow_ns =
                self.timing_sum_shadow_tree_ns.load(Ordering::Relaxed) / apply_events;
            format!(
                "\nShadow-tree (block_to_shard DashMap, {apply_events} events):\n  \
                 avg orchestration = {avg_shadow_ns}ns  (DashMap inserts/removes per apply_event)\n  \
                 current size      = {shadow_tree_size} entries  (live blocks tracked across all shards)"
            )
        } else {
            format!(
                "\nShadow-tree (block_to_shard DashMap): 0 events seen\n  \
                 current size = {shadow_tree_size} entries"
            )
        };

        format!(
            "PrefixShardedIndexer find_matches ({calls} calls):\n  \
             avg routing = {avg_routing_ns}ns  (FNV hash, {routing_pct:.2}% of outer)\n  \
             avg shard   = {avg_shard_us}µs  (CRTC traversal, {mode})\n  \
             [any p99 gap vs baseline comes from write-path DashMap contention, not routing]\
             {shadow_section}"
        )
    }
}
