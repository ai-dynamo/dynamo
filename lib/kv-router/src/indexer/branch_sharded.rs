// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Branch-based prefix sharding over `ThreadPoolIndexer<T>`.
//!
//! [`BranchShardedIndexer`] partitions the prefix space by building an explicit
//! routing table that maps branch keys (FNV-1a hash of first `prefix_depth`
//! block hashes) to shard indices.  Unlike [`PrefixShardedIndexer`] which uses
//! `hash % N`, new branches are assigned to the **least-loaded shard** at first
//! insertion time, so load is balanced regardless of hash distribution.
//!
//! ## Key properties
//!
//! - **Single-shard `find_matches`**: a query routes to exactly one shard — no
//!   scatter-gather.  Read throughput scales linearly with shard count.
//! - **Least-loaded branch assignment**: each new branch key is assigned to the
//!   shard with the fewest branches, ensuring balanced distribution even when
//!   the underlying hash values cluster.
//! - **Stable shard assignment**: once a branch is assigned, it never migrates.
//!   CRTC-internal splits stay within the owning shard — no migration protocol
//!   needed.  The shard assignment is keyed on the *sequence prefix* (first K
//!   blocks), not on tree nodes, so splits are transparent to this layer.
//! - **Unknown-branch fast path**: if a query's branch key is not in the routing
//!   table, no worker has ever stored that prefix.  `find_matches` returns empty
//!   scores immediately without dispatching to any shard.
//!
//! ## Remove routing
//!
//! Two strategies are used in combination:
//!
//! 1. **Mapping (primary)**: each `block_hash` is looked up in a
//!    `block_to_shard` index (populated at Stored time) and routed to its
//!    owning shard only.
//! 2. **Broadcast fallback**: if a block hash is absent from the index (evicted,
//!    out-of-order event, or index overflow), the Remove is broadcast to all
//!    shards.  Each shard's CRTC handles a missing block as a no-op. Shared
//!    sharded-indexer metrics track how often this occurs.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "bench")]
use std::time::Instant;

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

#[cfg(feature = "bench")]
use super::ShardedIndexerMetrics;
use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

// ---------------------------------------------------------------------------
// Per-shard read thread pool (kept for potential future use)
// ---------------------------------------------------------------------------

/// A bounded pool of OS threads dedicated to `find_matches` requests for one
/// shard.  Mirrors the equivalent struct in `prefix_sharded.rs`.
///
/// Not currently used by [`BranchShardedIndexer`] — reads run inline on the
/// caller's thread.  Retained here as a building block if dedicated read
/// isolation is needed in the future.
#[allow(dead_code)]
struct ShardReadPool {
    sender: flume::Sender<(
        Vec<LocalBlockHash>,
        tokio::sync::oneshot::Sender<OverlapScores>,
    )>,
    _threads: Vec<std::thread::JoinHandle<()>>,
}

#[allow(dead_code)]
impl ShardReadPool {
    fn new<T: SyncIndexer>(backend: Arc<T>, num_threads: usize) -> Self {
        let (tx, rx) = flume::unbounded();
        let mut threads = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let backend = Arc::clone(&backend);
            let rx: flume::Receiver<(
                Vec<LocalBlockHash>,
                tokio::sync::oneshot::Sender<OverlapScores>,
            )> = rx.clone();
            threads.push(std::thread::spawn(move || {
                while let Ok((seq, resp_tx)) = rx.recv() {
                    let result = backend.find_matches(&seq, false);
                    let _ = resp_tx.send(result);
                }
            }));
        }
        Self {
            sender: tx,
            _threads: threads,
        }
    }
}

// ---------------------------------------------------------------------------
// FNV-1a constants
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

/// Fold one `u64` value into an FNV-1a accumulator.
#[inline(always)]
fn fnv_fold(state: u64, value: u64) -> u64 {
    let mut h = state;
    for b in value.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// BranchShardedIndexer
// ---------------------------------------------------------------------------

/// Branch-sharded wrapper over N [`ThreadPoolIndexer<T>`] instances.
///
/// Construct with [`BranchShardedIndexer::new`].
pub struct BranchShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    num_shards: usize,

    /// Number of leading blocks used to identify a branch.  Default: 2.
    prefix_depth: usize,

    /// Routing table: FNV-1a(first `prefix_depth` `LocalBlockHash`) → shard index.
    ///
    /// Populated lazily at first `Stored` event for each distinct branch.
    branch_to_shard: DashMap<u64, usize, FxBuildHasher>,

    /// Number of branches assigned to each shard (for observability).
    branch_counts: Mutex<Vec<usize>>,

    /// Eagerly-updated block count per shard.
    ///
    /// Incremented synchronously in `apply_event` (before the event is dispatched
    /// to the async worker thread) so that `assign_shard` always sees an up-to-date
    /// load estimate even when the CRTC backend has not yet processed the event.
    /// This prevents every branch from being assigned to the same shard during
    /// burst startup, when all CRTC node counts are still zero.
    shard_block_counts: Vec<AtomicUsize>,

    /// Remove index: `ExternalSequenceBlockHash.0` → `(shard_index, ref_count)`.
    ///
    /// Written on `Stored` (ref_count incremented), decremented on `Removed`.
    /// The entry is deleted only when ref_count reaches zero — i.e. every worker
    /// that stored the block has since evicted it.
    ///
    /// Note: `block_to_shard` entries are content-addressed — the same
    /// `ExternalSequenceBlockHash` can be shared by multiple workers (identical
    /// token sequences).  Without ref-counting, the first worker to evict a
    /// shared block would delete the entry, causing all subsequent workers'
    /// Removed events for that block to fall through to broadcast.  Ref-counting
    /// keeps the entry alive until the last holder evicts it.
    ///
    /// A `Cleared` event does NOT touch this map because doing so would break
    /// routing for other workers whose continuations reference the same parent
    /// hashes.  Only `Removed` events (which carry explicit block hashes)
    /// decrement the ref-count.
    ///
    /// Note: parent-hash inheritance via this map is only used once a chain tail
    /// has reached `prefix_depth` blocks (depth ≥ prefix_depth).  Shallower
    /// tails are tracked in `block_to_fnv_state` and route by FNV accumulation.
    block_to_shard: DashMap<u64, (usize, usize), FxBuildHasher>,

    /// FNV accumulator for chain tails that have not yet reached `prefix_depth` blocks.
    ///
    /// Maps the `ExternalSequenceBlockHash.0` of the **last stored block** in a
    /// shallow chain to `(accumulated_fnv, depth)`, where `depth < prefix_depth`.
    ///
    /// # Why this exists
    ///
    /// For workloads with a shared prefix shorter than `prefix_depth` (e.g. a
    /// 15-block system prompt with `prefix_depth = 17`), all root events produce
    /// the **same** partial FNV hash, collapsing every conversation onto a single
    /// shard.  By carrying the accumulated FNV forward into continuation events,
    /// each conversation extends the hash with its own unique blocks (positions
    /// 15 and 16) and thereby receives a distinct, balanced shard assignment.
    ///
    /// # CRTC chain / lookup notes
    ///
    /// When a continuation's finalized FNV routes it to a different shard than its
    /// parent, the CRTC on the new shard will not find the parent and will drop the
    /// event.  Fixing this fully requires replaying the shallow prefix to the new
    /// shard ("shallow chain replay"), which is left as a future improvement.  For
    /// now the routing table is correct — `find_matches` routes to the right shard —
    /// but the underlying CRTC may have no data there until replay is implemented.
    ///
    /// Separately, `find_matches` hashes only the available prefix
    /// (`min(prefix_depth, len)`). A query shorter than `prefix_depth` would
    /// therefore probe with a shorter key than the key registered by a root
    /// `Stored` event. This is addressed by `register_prefix_keys`, which
    /// records intermediate FNV values (depths 1 … prefix_depth) at store time
    /// so short queries route to the correct shard instead of receiving a false
    /// miss.
    ///
    /// Like `block_to_shard`, entries are content-addressed and are NOT removed by
    /// `Cleared` events; only `Removed` events prune them.
    block_to_fnv_state: DashMap<u64, (u64, usize), FxBuildHasher>,

    kv_block_size: u32,

    #[cfg(feature = "bench")]
    metrics: ShardedIndexerMetrics,
}

impl<T: SyncIndexer> BranchShardedIndexer<T> {
    /// Create a branch-sharded indexer from pre-built [`ThreadPoolIndexer`] shards.
    ///
    /// # Arguments
    ///
    /// * `shards` - One `ThreadPoolIndexer` per shard.
    /// * `prefix_depth` - Number of prefix blocks to hash for routing.  Clamped
    ///   to ≥ 1.  K=2 is the recommended default (depth=1 gives too few distinct
    ///   branch keys on many workloads).
    /// * `kv_block_size` - Block size for KV cache.
    ///
    /// # Panics
    ///
    /// Panics if `shards` is empty.
    pub fn new(shards: Vec<ThreadPoolIndexer<T>>, prefix_depth: usize, kv_block_size: u32) -> Self {
        assert!(!shards.is_empty(), "Must provide at least one shard");
        let num_shards = shards.len();

        let shards: Vec<Arc<ThreadPoolIndexer<T>>> = shards.into_iter().map(Arc::new).collect();

        Self {
            shards,
            num_shards,
            prefix_depth: prefix_depth.max(1),
            branch_to_shard: DashMap::with_hasher(FxBuildHasher),
            branch_counts: Mutex::new(vec![0usize; num_shards]),
            shard_block_counts: (0..num_shards).map(|_| AtomicUsize::new(0)).collect(),
            block_to_shard: DashMap::with_hasher(FxBuildHasher),
            block_to_fnv_state: DashMap::with_hasher(FxBuildHasher),
            kv_block_size,
            #[cfg(feature = "bench")]
            metrics: ShardedIndexerMetrics::new(),
        }
    }

    /// Alias for [`BranchShardedIndexer::new`], kept for call-site compatibility.
    pub fn new_with_options(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
    ) -> Self {
        Self::new(shards, prefix_depth, kv_block_size)
    }

    // --- branch key computation ---

    /// FNV-1a hash of the first `min(prefix_depth, len)` `LocalBlockHash` values.
    ///
    /// Used by `find_matches` to compute the branch key for an incoming query.
    fn branch_key_for_local_hashes(&self, hashes: &[LocalBlockHash]) -> u64 {
        let k = self.prefix_depth.min(hashes.len());
        hashes[..k]
            .iter()
            .fold(FNV_OFFSET_BASIS, |h, block| fnv_fold(h, block.0))
    }

    /// FNV-1a hash of the first `min(prefix_depth, len)` `tokens_hash` values
    /// from a `Stored` event's block list.
    fn branch_key_for_stored_blocks(&self, blocks: &[KvCacheStoredBlockData]) -> u64 {
        let k = self.prefix_depth.min(blocks.len());
        blocks[..k].iter().fold(FNV_OFFSET_BASIS, |h, block| {
            fnv_fold(h, block.tokens_hash.0)
        })
    }

    // --- routing table operations ---

    fn lookup_shard(&self, branch_key: u64) -> Option<usize> {
        self.branch_to_shard.get(&branch_key).map(|v| *v)
    }

    /// Get or create a shard assignment for a branch key.
    ///
    /// Fast path if already assigned; otherwise acquires the lock, picks the
    /// least-loaded shard, and inserts atomically.
    ///
    /// Load is measured by **live block count** in each shard (an O(1) atomic
    /// read).  Block count is a better proxy than branch count when conversation
    /// lengths vary widely — long conversations contribute many more blocks than
    /// short ones even though both count as one branch.  Branch count is used as
    /// a tiebreaker when block counts are equal (e.g. at startup before any
    /// events have been processed).
    fn assign_shard(&self, branch_key: u64) -> usize {
        if let Some(shard_idx) = self.branch_to_shard.get(&branch_key).map(|v| *v) {
            return shard_idx;
        }
        let mut counts = self.branch_counts.lock().unwrap();
        if let Some(shard_idx) = self.branch_to_shard.get(&branch_key).map(|v| *v) {
            return shard_idx;
        }
        let selected = self
            .shard_block_counts
            .iter()
            .enumerate()
            .min_by(|(i, a), (j, b)| {
                a.load(Ordering::Relaxed)
                    .cmp(&b.load(Ordering::Relaxed))
                    .then(counts[*i].cmp(&counts[*j]))
            })
            .unwrap()
            .0;
        counts[selected] += 1;
        drop(counts);
        self.branch_to_shard.insert(branch_key, selected);
        selected
    }

    /// Register shorter prefix keys for the same shard without affecting
    /// branch-count accounting.
    ///
    /// This supports short queries (`len < prefix_depth`) that should still
    /// route to the shard containing a longer cached prefix and obtain a
    /// shallower overlap score instead of an immediate false miss.
    fn register_prefix_keys<I>(&self, shard_idx: usize, branch_keys: I)
    where
        I: IntoIterator<Item = u64>,
    {
        for branch_key in branch_keys {
            self.branch_to_shard.entry(branch_key).or_insert(shard_idx);
        }
    }

    /// Compute cumulative FNV prefix keys for the first `limit` stored blocks,
    /// starting from `initial_state`.
    fn prefix_keys_for_stored_blocks_from_state(
        initial_state: u64,
        blocks: &[KvCacheStoredBlockData],
        limit: usize,
    ) -> Vec<u64> {
        let mut keys = Vec::with_capacity(limit);
        let mut state = initial_state;
        for block in blocks.iter().take(limit) {
            state = fnv_fold(state, block.tokens_hash.0);
            keys.push(state);
        }
        keys
    }

    // -----------------------------------------------------------------------
    // Private event handlers (called from apply_event)
    // -----------------------------------------------------------------------

    /// Compute the target shard and (if still shallow) the updated FNV
    /// accumulator state for a `Stored` event.
    ///
    /// Shard assignment uses accumulated FNV until the chain reaches
    /// `prefix_depth` blocks, then switches to parent-hash inheritance.
    ///
    /// Three cases:
    ///
    /// A. Parent tail found in `block_to_fnv_state` (depth < prefix_depth):
    ///    Extend the FNV accumulator with leading blocks from this batch.
    ///    Once the accumulated depth reaches `prefix_depth`, call
    ///    `assign_shard` with the finalized key so that distinct
    ///    continuations receive distinct shard assignments.
    ///    Record the updated state on the last block of this batch if the
    ///    chain is still shallow after processing.
    ///
    /// B. Parent tail found in `block_to_shard` (depth >= prefix_depth):
    ///    Inherit the shard — the branch was already decided.
    ///
    /// C. No parent (root) or OOO (parent not in either map):
    ///    Compute FNV from this batch's own blocks.  For root events
    ///    shorter than `prefix_depth` this is a partial key; a future
    ///    continuation in case A will extend it to the full depth.
    ///
    /// Returns `(shard_idx, Option<(fnv, depth)>)`.  A `Some` state means
    /// the chain has not yet reached `prefix_depth` blocks; the caller should
    /// record it on the last block of the batch so the next continuation can
    /// extend it.
    fn compute_stored_routing(
        &self,
        store_data: &KvCacheStoreData,
    ) -> (usize, Option<(u64, usize)>) {
        if let Some(parent_hash) = &store_data.parent_hash {
            if let Some(entry) = self.block_to_fnv_state.get(&parent_hash.0) {
                // Case A: parent is shallow — extend FNV accumulator.
                let (parent_fnv, parent_depth) = *entry;
                drop(entry);
                let remaining = self.prefix_depth - parent_depth;
                let to_process = remaining.min(store_data.blocks.len());
                let prefix_keys = Self::prefix_keys_for_stored_blocks_from_state(
                    parent_fnv,
                    &store_data.blocks,
                    to_process,
                );
                let fnv = prefix_keys.last().copied().unwrap_or(parent_fnv);
                let new_depth = parent_depth + to_process;
                let shard = self.assign_shard(fnv);
                if new_depth >= self.prefix_depth {
                    self.register_prefix_keys(shard, prefix_keys);
                }
                let state = (new_depth < self.prefix_depth).then_some((fnv, new_depth));
                (shard, state)
            } else if let Some(shard) = self.block_to_shard.get(&parent_hash.0).map(|v| v.0) {
                // Case B: deep chain — inherit shard.
                (shard, None)
            } else {
                // Case C (OOO): parent not in either map; best-effort key from this batch.
                let key = self.branch_key_for_stored_blocks(&store_data.blocks);
                (self.assign_shard(key), None)
            }
        } else {
            // Case C (root): start FNV accumulation from scratch.
            let to_process = self.prefix_depth.min(store_data.blocks.len());
            let prefix_keys = Self::prefix_keys_for_stored_blocks_from_state(
                FNV_OFFSET_BASIS,
                &store_data.blocks,
                to_process,
            );
            let fnv = prefix_keys.last().copied().unwrap_or(FNV_OFFSET_BASIS);
            let depth = to_process;
            let shard = self.assign_shard(fnv);
            self.register_prefix_keys(shard, prefix_keys);
            let state = (depth < self.prefix_depth).then_some((fnv, depth));
            (shard, state)
        }
    }

    async fn apply_stored(&self, event: RouterEvent) {
        let KvCacheEventData::Stored(store_data) = &event.event.data else {
            return;
        };

        let (shard_idx, new_fnv_state) = self.compute_stored_routing(store_data);

        // Update eager block count before dispatching.
        self.shard_block_counts[shard_idx].fetch_add(store_data.blocks.len(), Ordering::Relaxed);

        // Record block → shard before dispatching so a fast continuation
        // can find entries immediately.
        for block in &store_data.blocks {
            self.block_to_shard
                .entry(block.block_hash.0)
                .and_modify(|e| e.1 += 1)
                .or_insert((shard_idx, 1));
        }

        // Propagate partial FNV state on the last block of this batch.
        if let (Some(fnv_state), Some(last_block)) = (new_fnv_state, store_data.blocks.last()) {
            self.block_to_fnv_state
                .insert(last_block.block_hash.0, fnv_state);
        }

        self.shards[shard_idx].apply_event(event).await;
    }

    async fn apply_removed(&self, event: RouterEvent) {
        // Copy metadata before borrowing event.event.data.
        let worker_id = event.worker_id;
        let storage_tier = event.storage_tier;
        let event_id = event.event.event_id;
        let dp_rank = event.event.dp_rank;

        let KvCacheEventData::Removed(remove_data) = &event.event.data else {
            return;
        };

        // --- Plan: classify each block as mapped-to-shard or broadcast ---
        let mut shard_blocks: Vec<Vec<ExternalSequenceBlockHash>> =
            vec![Vec::new(); self.num_shards];
        let mut broadcast_blocks: Vec<ExternalSequenceBlockHash> = Vec::new();

        for &block_hash in &remove_data.block_hashes {
            self.block_to_fnv_state.remove(&block_hash.0);
            let found_shard = self.block_to_shard.get_mut(&block_hash.0).map(|mut e| {
                let shard_idx = e.0;
                e.1 = e.1.saturating_sub(1);
                shard_idx
            });
            match found_shard {
                Some(shard_idx) => {
                    self.block_to_shard
                        .remove_if(&block_hash.0, |_, v| v.1 == 0);
                    shard_blocks[shard_idx].push(block_hash);
                }
                None => {
                    #[cfg(feature = "bench")]
                    self.metrics
                        .counters
                        .remove_broadcasts
                        .fetch_add(1, Ordering::Relaxed);
                    broadcast_blocks.push(block_hash);
                }
            }
        }

        // --- Dispatch: route mapped removes to their owning shards ---
        for (shard_idx, blocks) in shard_blocks.into_iter().enumerate() {
            if blocks.is_empty() {
                continue;
            }
            self.shard_block_counts[shard_idx]
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                    Some(count.saturating_sub(blocks.len()))
                })
                .ok();
            let shard_event = RouterEvent {
                worker_id,
                storage_tier,
                event: KvCacheEvent {
                    event_id,
                    dp_rank,
                    data: KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: blocks,
                    }),
                },
            };
            self.shards[shard_idx].apply_event(shard_event).await;
        }

        // Broadcast unknown blocks to all shards; each CRTC treats a missing
        // block as a no-op so correctness is maintained.
        if !broadcast_blocks.is_empty() {
            for shard in &self.shards {
                let broadcast_event = RouterEvent {
                    worker_id,
                    storage_tier,
                    event: KvCacheEvent {
                        event_id,
                        dp_rank,
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: broadcast_blocks.clone(),
                        }),
                    },
                };
                shard.apply_event(broadcast_event).await;
            }
        }
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for BranchShardedIndexer<T> {
    /// Route to a single shard determined by the first `prefix_depth` block hashes.
    ///
    /// If the branch key is not in the routing table, no worker has ever stored
    /// that prefix, so the result would be empty regardless of which shard is
    /// queried.  We return `OverlapScores::new()` immediately without dispatching.
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let t_routing = Instant::now();
        let branch_key = self.branch_key_for_local_hashes(&sequence);
        let shard_idx = match self.lookup_shard(branch_key) {
            Some(idx) => idx,
            None => {
                #[cfg(feature = "bench")]
                self.metrics
                    .counters
                    .find_match_early_returns
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(OverlapScores::new());
            }
        };
        #[cfg(feature = "bench")]
        self.metrics
            .counters
            .find_match_dispatches
            .fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "bench")]
        let routing_ns = t_routing.elapsed().as_nanos() as u64;

        #[cfg(feature = "bench")]
        let t_shard = Instant::now();
        let result = self.shards[shard_idx].find_matches(sequence).await;
        #[cfg(feature = "bench")]
        {
            let shard_ns = t_shard.elapsed().as_nanos() as u64;
            self.metrics.timing.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .timing
                .routing_ns
                .fetch_add(routing_ns, Ordering::Relaxed);
            self.metrics
                .timing
                .shard_ns
                .fetch_add(shard_ns, Ordering::Relaxed);
        }

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
                block_mm_infos: None,
            },
        );
        let branch_key = self.branch_key_for_local_hashes(&sequence);
        match self.lookup_shard(branch_key) {
            Some(idx) => self.shards[idx].find_matches(sequence).await,
            None => Ok(OverlapScores::new()),
        }
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(_) => self.apply_stored(event).await,
            KvCacheEventData::Removed(_) => self.apply_removed(event).await,
            KvCacheEventData::Cleared => {
                // A worker may have blocks across multiple shards (different
                // branches stored over its lifetime) — broadcast to all.
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        // A worker may have blocks on any shard — broadcast.
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
        Ok(())
    }

    async fn flush(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            total += <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(shard).await;
        }
        total
    }

    async fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        let mut sizes = Vec::new();
        for (idx, shard) in self.shards.iter().enumerate() {
            // ThreadPoolIndexer::shard_sizes() already populates node_count
            // via backend.node_count() (O(1)).  No need to call
            // node_edge_lengths().len() which allocates an O(N) Vec.
            sizes.extend(shard.shard_sizes().await.into_iter().map(move |mut s| {
                s.shard_idx = idx;
                s
            }));
        }
        sizes
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.node_edge_lengths())
            .collect()
    }

    fn timing_report(&self) -> String {
        #[cfg(not(feature = "bench"))]
        {
            String::new()
        }

        #[cfg(feature = "bench")]
        {
            let dispatched = self
                .metrics
                .counters
                .find_match_dispatches
                .load(Ordering::Relaxed);
            let misses = self
                .metrics
                .counters
                .find_match_early_returns
                .load(Ordering::Relaxed);
            let total_calls = dispatched + misses;
            let broadcasts = self
                .metrics
                .counters
                .remove_broadcasts
                .load(Ordering::Relaxed);
            if total_calls == 0 {
                return String::new();
            }
            let miss_pct = 100.0 * misses as f64 / total_calls as f64;

            let timing = {
                let timing_calls = self.metrics.timing.calls.load(Ordering::Relaxed);
                let avg_routing_ns = if timing_calls > 0 {
                    self.metrics.timing.routing_ns.load(Ordering::Relaxed) / timing_calls
                } else {
                    0
                };
                let avg_shard_us = if timing_calls > 0 {
                    self.metrics.timing.shard_ns.load(Ordering::Relaxed) / timing_calls / 1000
                } else {
                    0
                };
                format!(
                    "\n  avg routing    = {avg_routing_ns}ns  (routing table lookup)\n  \
                 avg shard      = {avg_shard_us}µs  (CRTC traversal, inline on caller thread)"
                )
            };

            let branch_counts = self.branch_counts.lock().unwrap();
            let total_branches: usize = branch_counts.iter().sum();
            let branch_dist: Vec<String> = branch_counts
                .iter()
                .enumerate()
                .map(|(i, c)| format!("shard[{i}]={c}"))
                .collect();
            drop(branch_counts);
            format!(
                "BranchShardedIndexer find_matches ({total_calls} total: {dispatched} dispatched, \
             {misses} early-exit / {miss_pct:.1}% miss):{timing}\n  \
             branches known = {total_branches}  ({})\n  \
             remove broadcasts = {broadcasts}  (fallback for blocks absent from index)",
                branch_dist.join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;

    fn block(block_hash: u64, tokens_hash: u64) -> KvCacheStoredBlockData {
        KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block_hash),
            tokens_hash: LocalBlockHash(tokens_hash),
            mm_extra_info: None,
        }
    }

    fn stored_event(
        worker_id: WorkerId,
        parent_hash: Option<u64>,
        blocks: Vec<KvCacheStoredBlockData>,
    ) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: worker_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
        )
    }

    fn make_indexer(
        num_shards: usize,
        prefix_depth: usize,
    ) -> BranchShardedIndexer<ConcurrentRadixTreeCompressed> {
        let shards = (0..num_shards)
            .map(|_| ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32))
            .collect();
        BranchShardedIndexer::new_with_options(shards, prefix_depth, 32)
    }

    #[tokio::test]
    async fn short_query_hits_after_full_root_batch_registers_intermediate_keys() {
        let indexer = make_indexer(2, 2);
        indexer
            .apply_event(stored_event(
                1,
                None,
                vec![block(101, 11), block(102, 22), block(103, 33)],
            ))
            .await;
        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![LocalBlockHash(11)])
            .await
            .expect("query should succeed");
        let best = overlap.scores.values().copied().max().unwrap_or(0);
        assert_eq!(
            best, 1,
            "short query should get shallow overlap instead of miss"
        );
    }

    #[tokio::test]
    async fn short_query_hits_after_shallow_root_crosses_prefix_depth() {
        let indexer = make_indexer(1, 4);
        indexer
            .apply_event(stored_event(1, None, vec![block(201, 11)]))
            .await;
        indexer
            .apply_event(stored_event(
                1,
                Some(201),
                vec![
                    block(202, 22),
                    block(203, 33),
                    block(204, 44),
                    block(205, 55),
                ],
            ))
            .await;
        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![LocalBlockHash(11), LocalBlockHash(22)])
            .await
            .expect("query should succeed");
        let best = overlap.scores.values().copied().max().unwrap_or(0);
        assert_eq!(
            best, 2,
            "prefix keys added during depth crossing should route short queries correctly"
        );
    }
}
