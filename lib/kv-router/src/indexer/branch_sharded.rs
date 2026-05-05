// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Branch-based prefix sharding over `ThreadPoolIndexer<T>`.
//!
//! [`BranchShardedIndexer`] partitions the prefix space by building an explicit
//! routing table that maps FNV-1a prefix keys to shard indices. Unlike
//! [`PrefixShardedIndexer`] which uses `hash % N`, new independent branches are
//! assigned to the least-loaded shard at first insertion time; continuations and
//! overlapping prefix aliases reuse the existing shard so one logical chain does
//! not cross shards.
//!
//! ## Key properties
//!
//! - **Single-shard `find_matches`**: a query routes to the shard for the
//!   deepest registered prefix alias — no scatter-gather.
//! - **Load-aware branch assignment**: independent new branches are assigned by
//!   live shard block count, with branch count as a tiebreaker.
//! - **Stable shard assignment**: once a branch is assigned, it never migrates.
//!   CRTC-internal splits stay within the owning shard — no migration protocol
//!   needed.  The shard assignment is keyed on the *sequence prefix* (first K
//!   blocks), not on tree nodes, so splits are transparent to this layer.
//! - **Unknown-branch fast path**: if none of a query's prefix keys are in the
//!   routing table, no worker has ever stored that routed prefix. `find_matches`
//!   returns empty scores immediately without dispatching to any shard.
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
// FNV-1a constants
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;
const INLINE_PREFIX_KEY_CAP: usize = 16;

type PartialFnvState = (u64, usize, usize);

struct StoredRoutingDecision {
    shard_idx: usize,
    new_fnv_state: Option<PartialFnvState>,
    parent_found: bool,
    branch_keys: Vec<u64>,
}

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

    /// Routing table: FNV-1a prefix key → shard index.
    ///
    /// Populated lazily at first `Stored` event for each distinct branch.  A
    /// branch can register multiple aliases, one for each prefix depth up to
    /// `prefix_depth`, so shallow-overlap queries can route to a useful shard.
    branch_to_shard: DashMap<u64, usize, FxBuildHasher>,

    /// Number of canonical branch assignments per shard (for observability and
    /// load-selection tiebreaking).  Prefix aliases are not counted separately.
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
    /// shallow chain to `(accumulated_fnv, depth, shard_idx)`, where `depth < prefix_depth`.
    /// The shard is carried forward so continuations inherit the root's shard (sticky
    /// routing) and never cross to a shard that lacks the parent chain.
    ///
    /// # Why this exists
    ///
    /// For workloads with a shared prefix shorter than `prefix_depth` (e.g. a
    /// 15-block system prompt with `prefix_depth = 17`), all root events produce
    /// the **same** partial FNV hash.  `block_to_fnv_state` carries both that
    /// accumulated FNV and the root shard into continuation events, so routing is
    /// sticky: the continuation finalizes prefix aliases on the root's shard
    /// instead of crossing to a shard that lacks the parent chain.
    ///
    /// `find_matches` computes prefix keys up to `prefix_depth` and routes by
    /// the deepest registered key.  `register_prefix_keys` records intermediate
    /// FNV values (depths 1 … prefix_depth) at store time so queries that only
    /// overlap a shallow prefix still probe the shard that can return that score.
    ///
    /// Like `block_to_shard`, entries are content-addressed and are NOT removed by
    /// `Cleared` events; only `Removed` events prune them.
    block_to_fnv_state: DashMap<u64, PartialFnvState, FxBuildHasher>,

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
    /// Test helper for checking routing-table aliases.
    #[cfg(test)]
    fn branch_key_for_local_hashes(&self, hashes: &[LocalBlockHash]) -> u64 {
        Self::prefix_keys_for_local_hashes(hashes, self.prefix_depth)
            .last()
            .copied()
            .unwrap_or(FNV_OFFSET_BASIS)
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

    fn lookup_deepest_prefix_shard(&self, branch_keys: &[u64]) -> Option<(usize, bool)> {
        let exact_depth = branch_keys.len();
        branch_keys
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, branch_key)| {
                self.lookup_shard(*branch_key)
                    .map(|shard| (shard, idx + 1 < exact_depth))
            })
    }

    fn lookup_deepest_prefix_shard_for_hashes(
        &self,
        hashes: &[LocalBlockHash],
    ) -> Option<(usize, bool)> {
        let limit = self.prefix_depth.min(hashes.len());
        if limit == 0 {
            return None;
        }

        if limit <= INLINE_PREFIX_KEY_CAP {
            let mut branch_keys = [0; INLINE_PREFIX_KEY_CAP];
            let mut state = FNV_OFFSET_BASIS;
            for (idx, block) in hashes.iter().take(limit).enumerate() {
                state = fnv_fold(state, block.0);
                branch_keys[idx] = state;
            }
            return self.lookup_deepest_prefix_shard(&branch_keys[..limit]);
        }

        let branch_keys = Self::prefix_keys_for_local_hashes(hashes, limit);
        self.lookup_deepest_prefix_shard(&branch_keys)
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

    /// Get or create one shard assignment for a complete set of prefix aliases.
    ///
    /// A root batch that already has a shorter prefix alias (for example `[A]`)
    /// must reuse that shard when it also registers the full prefix (`[A, B]`).
    /// Serializing the alias check and insertion with the same lock used by
    /// `assign_shard` prevents the shallow-root and full-root batch shapes from
    /// assigning the same logical prefix to different shards.
    fn assign_shard_for_prefix_keys(&self, branch_keys: &[u64]) -> usize {
        if branch_keys.is_empty() {
            return self.assign_shard(FNV_OFFSET_BASIS);
        }

        let mut counts = self.branch_counts.lock().unwrap();
        let selected = branch_keys
            .iter()
            .find_map(|key| self.branch_to_shard.get(key).map(|v| *v))
            .unwrap_or_else(|| {
                self.shard_block_counts
                    .iter()
                    .enumerate()
                    .min_by(|(i, a), (j, b)| {
                        a.load(Ordering::Relaxed)
                            .cmp(&b.load(Ordering::Relaxed))
                            .then(counts[*i].cmp(&counts[*j]))
                    })
                    .unwrap()
                    .0
            });

        let canonical_key = *branch_keys.last().unwrap();
        let canonical_was_known = self.branch_to_shard.contains_key(&canonical_key);
        for &branch_key in branch_keys {
            self.branch_to_shard.entry(branch_key).or_insert(selected);
        }
        if !canonical_was_known
            && self.branch_to_shard.get(&canonical_key).map(|v| *v) == Some(selected)
        {
            counts[selected] += 1;
        }

        selected
    }

    /// Register prefix keys for an already-selected shard.
    ///
    /// This supports short queries (`len < prefix_depth`) that should still
    /// route to the shard containing a longer cached prefix and obtain a
    /// shallower overlap score instead of an immediate false miss.
    fn register_prefix_keys<I>(&self, shard_idx: usize, branch_keys: I)
    where
        I: IntoIterator<Item = u64>,
    {
        let branch_keys: Vec<u64> = branch_keys.into_iter().collect();
        if branch_keys.is_empty() {
            return;
        }

        let mut counts = self.branch_counts.lock().unwrap();
        let canonical_key = *branch_keys.last().unwrap();
        let canonical_was_known = self.branch_to_shard.contains_key(&canonical_key);
        for branch_key in branch_keys {
            self.branch_to_shard.entry(branch_key).or_insert(shard_idx);
        }
        if !canonical_was_known
            && self.branch_to_shard.get(&canonical_key).map(|v| *v) == Some(shard_idx)
        {
            counts[shard_idx] += 1;
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

    fn prefix_keys_for_local_hashes(hashes: &[LocalBlockHash], limit: usize) -> Vec<u64> {
        let limit = limit.min(hashes.len());
        let mut keys = Vec::with_capacity(limit);
        let mut state = FNV_OFFSET_BASIS;
        for block in hashes.iter().take(limit) {
            state = fnv_fold(state, block.0);
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
    ///    Inherit the parent's shard (sticky routing) so the chain stays
    ///    on one shard.  Once the accumulated depth reaches `prefix_depth`,
    ///    call `register_prefix_keys` to record finalized prefix aliases and
    ///    increment `branch_counts` for the inherited shard.
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
    /// Returns a [`StoredRoutingDecision`].
    ///
    /// `new_fnv_state` is `Some` only while the chain has not yet reached
    /// `prefix_depth`; the caller records it on the last block of the batch so
    /// the next continuation can extend it.  The shard in the state tuple is the
    /// sticky shard so continuations never cross shards mid-chain.
    ///
    /// `parent_found` is `false` only for Case C OOO (continuation whose parent
    /// is absent from both routing maps).  The caller uses this to strip the
    /// parent hash before dispatching so the CRTC stores the blocks as an orphan
    /// root rather than dropping them with a "parent not found" warning.
    ///
    /// `branch_keys` contains every key written (or updated) in `branch_to_shard`
    /// during this call.  The caller uses this list to correct `branch_to_shard`
    /// when the canonical shard (`actual_shard`, determined after inspecting
    /// `block_to_shard`) differs from `shard_idx`.
    fn compute_stored_routing(&self, store_data: &KvCacheStoreData) -> StoredRoutingDecision {
        if let Some(parent_hash) = &store_data.parent_hash {
            if let Some(entry) = self.block_to_fnv_state.get(&parent_hash.0) {
                // Case A: parent is shallow — extend FNV accumulator.
                // Inherit the parent's shard (sticky routing) so the full chain
                // lands on one shard and the CRTC never has to cross.
                let (parent_fnv, parent_depth, parent_shard) = *entry;
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
                let shard = parent_shard;
                let branch_keys = if new_depth >= self.prefix_depth {
                    self.register_prefix_keys(shard, prefix_keys.iter().copied());
                    prefix_keys
                } else {
                    vec![]
                };
                let state = (new_depth < self.prefix_depth).then_some((fnv, new_depth, shard));
                StoredRoutingDecision {
                    shard_idx: shard,
                    new_fnv_state: state,
                    parent_found: true,
                    branch_keys,
                }
            } else if let Some(shard) = self.block_to_shard.get(&parent_hash.0).map(|v| v.0) {
                // Case B: deep chain — inherit shard.
                StoredRoutingDecision {
                    shard_idx: shard,
                    new_fnv_state: None,
                    parent_found: true,
                    branch_keys: vec![],
                }
            } else {
                // Case C (OOO): parent not in either map; best-effort key from this batch.
                let key = self.branch_key_for_stored_blocks(&store_data.blocks);
                StoredRoutingDecision {
                    shard_idx: self.assign_shard(key),
                    new_fnv_state: None,
                    parent_found: false,
                    branch_keys: vec![key],
                }
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
            let shard = self.assign_shard_for_prefix_keys(&prefix_keys);
            let state = (depth < self.prefix_depth).then_some((fnv, depth, shard));
            StoredRoutingDecision {
                shard_idx: shard,
                new_fnv_state: state,
                parent_found: true,
                branch_keys: prefix_keys,
            }
        }
    }

    async fn apply_stored(&self, mut event: RouterEvent) {
        let KvCacheEventData::Stored(store_data) = &event.event.data else {
            return;
        };

        let StoredRoutingDecision {
            shard_idx,
            new_fnv_state,
            parent_found,
            branch_keys,
        } = self.compute_stored_routing(store_data);

        // Resolve the canonical shard for Case C (root or OOO) by atomically
        // inserting the first block into `block_to_shard`.  Two concurrent
        // workers whose root events share the same first
        // `ExternalSequenceBlockHash` can race through `assign_shard` with
        // different `shard_block_counts` snapshots and compute different
        // `shard_idx` values.  Only one `or_insert` wins; the loser's
        // `shard_idx` then diverges from what `block_to_shard` records, so
        // continuations (Case B) land on the wrong shard.  Reading back the
        // winning entry's shard as `actual_shard` and correcting
        // `branch_to_shard` keeps root, continuations, and `find_matches` on
        // the same shard.
        //
        // Scoped to Case C only: Case A/B already inherit a deterministic
        // parent shard via sticky routing, and applying the reconciliation
        // there corrupts sequential-mode overlap scoring.
        let is_case_c = store_data.parent_hash.is_none() || !parent_found;
        let actual_shard = if is_case_c {
            if let Some(first_block) = store_data.blocks.first() {
                let entry = self
                    .block_to_shard
                    .entry(first_block.block_hash.0)
                    .and_modify(|e| e.1 += 1)
                    .or_insert((shard_idx, 1));
                let s = entry.0;
                drop(entry);
                s
            } else {
                shard_idx
            }
        } else {
            shard_idx
        };

        if is_case_c && actual_shard != shard_idx {
            let canonical_key = branch_keys.last().copied();
            let mut moved_canonical = false;
            let mut counts = self.branch_counts.lock().unwrap();
            for key in &branch_keys {
                self.branch_to_shard.entry(*key).and_modify(|v| {
                    if *v == shard_idx {
                        *v = actual_shard;
                        moved_canonical |= Some(*key) == canonical_key;
                    }
                });
            }
            if moved_canonical {
                counts[shard_idx] = counts[shard_idx].saturating_sub(1);
                counts[actual_shard] += 1;
            }
        }

        // Update eager block count for the canonical shard.
        self.shard_block_counts[actual_shard].fetch_add(store_data.blocks.len(), Ordering::Relaxed);

        // Register blocks in block_to_shard.
        // Case C: the first block was already registered when computing
        // actual_shard above; register the rest starting at index 1.
        // Case A/B: register all blocks (shard_idx == actual_shard).
        let block_skip = usize::from(is_case_c);
        for block in store_data.blocks.iter().skip(block_skip) {
            self.block_to_shard
                .entry(block.block_hash.0)
                .and_modify(|e| e.1 += 1)
                .or_insert((actual_shard, 1));
        }

        // Propagate partial FNV state with the canonical shard on the last block.
        let new_fnv_state = new_fnv_state.map(|(fnv, depth, _)| (fnv, depth, actual_shard));
        if let (Some(fnv_state), Some(last_block)) = (new_fnv_state, store_data.blocks.last()) {
            self.block_to_fnv_state
                .insert(last_block.block_hash.0, fnv_state);
        }
        // store_data borrow ends here.

        // OOO event: parent is unknown so the target CRTC cannot chain these
        // blocks.  Strip parent_hash so they are stored as an orphan root
        // rather than being silently dropped by the CRTC with a warning.
        if let (false, KvCacheEventData::Stored(data)) = (parent_found, &mut event.event.data) {
            data.parent_hash = None;
        }

        self.shards[actual_shard].apply_event(event).await;
    }

    async fn find_matches_for_sequence(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let t_routing = Instant::now();
        let (shard_idx, used_shallow_fallback) =
            match self.lookup_deepest_prefix_shard_for_hashes(&sequence) {
                Some(route) => route,
                None => {
                    #[cfg(feature = "bench")]
                    self.metrics
                        .counters
                        .find_match_early_returns
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(OverlapScores::new());
                }
            };
        #[cfg(not(feature = "bench"))]
        let _ = used_shallow_fallback;
        #[cfg(feature = "bench")]
        {
            if used_shallow_fallback {
                self.metrics
                    .counters
                    .shallow_fallback
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.metrics
                .counters
                .find_match_dispatches
                .fetch_add(1, Ordering::Relaxed);
        }

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
    /// Route to a single shard determined by the deepest registered query prefix.
    ///
    /// If none of the query's prefix keys are in the routing table, no worker
    /// has ever stored an overlapping routed prefix, so the result would be
    /// empty regardless of which shard is queried.  We return
    /// `OverlapScores::new()` immediately without dispatching.
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.find_matches_for_sequence(sequence).await
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
        self.find_matches_for_sequence(sequence).await
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

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                // ThreadPoolIndexer::shard_sizes() already populates node_count
                // via backend.node_count() (O(1)).  No need to call
                // node_edge_lengths().len() which allocates an O(N) Vec.
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
            let shallow_fallbacks = self
                .metrics
                .counters
                .shallow_fallback
                .load(Ordering::Relaxed);
            let misses = self
                .metrics
                .counters
                .find_match_early_returns
                .load(Ordering::Relaxed);
            let exact_dispatches = dispatched.saturating_sub(shallow_fallbacks);
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
            let fallback_pct = 100.0 * shallow_fallbacks as f64 / total_calls as f64;

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
                "BranchShardedIndexer find_matches ({total_calls} total: {exact_dispatches} exact dispatch, \
             {shallow_fallbacks} shallow-fallback / {fallback_pct:.1}%, \
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
    #[cfg(feature = "bench")]
    use std::sync::atomic::Ordering;

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

    fn local_hashes(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn score(scores: &OverlapScores, worker_id: WorkerId) -> Option<u32> {
        scores
            .scores
            .get(&WorkerWithDpRank::new(worker_id, 0))
            .copied()
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
    async fn shallow_overlap_fallback_avoids_false_miss_for_diverging_query() {
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
            .find_matches(vec![LocalBlockHash(11), LocalBlockHash(99)])
            .await
            .expect("query should succeed");
        assert_eq!(score(&overlap, 1), Some(1));
        #[cfg(feature = "bench")]
        assert_eq!(
            indexer
                .metrics
                .counters
                .shallow_fallback
                .load(Ordering::Relaxed),
            1
        );
        #[cfg(feature = "bench")]
        assert_eq!(
            indexer
                .metrics
                .counters
                .find_match_early_returns
                .load(Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn shallow_overlap_fallback_uses_deepest_matching_prefix() {
        let indexer = make_indexer(2, 3);
        indexer
            .apply_event(stored_event(
                1,
                None,
                vec![
                    block(101, 11),
                    block(102, 22),
                    block(103, 33),
                    block(104, 44),
                ],
            ))
            .await;
        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![
                LocalBlockHash(11),
                LocalBlockHash(22),
                LocalBlockHash(99),
            ])
            .await
            .expect("query should succeed");
        assert_eq!(score(&overlap, 1), Some(2));
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

    #[tokio::test]
    async fn case_a_intermediate_query_routes_via_fallback_before_crossing_prefix_depth() {
        let indexer = make_indexer(1, 4);
        indexer
            .apply_event(stored_event(1, None, vec![block(100, 10)]))
            .await;
        indexer
            .apply_event(stored_event(
                1,
                Some(100),
                vec![block(101, 11), block(102, 22)],
            ))
            .await;
        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![LocalBlockHash(10), LocalBlockHash(11)])
            .await
            .expect("query should succeed");
        assert_eq!(score(&overlap, 1), Some(2));
        #[cfg(feature = "bench")]
        assert_eq!(
            indexer
                .metrics
                .counters
                .shallow_fallback
                .load(Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn shallow_root_continuation_inherits_parent_shard() {
        // prefix_depth=2, 2 shards.  Root has 1 block (< prefix_depth), so its
        // FNV state is carried forward in block_to_fnv_state.  After storing the
        // root, shard_block_counts = [1, 0].  Without sticky routing the
        // continuation would call assign_shard on the finalized FNV key and land
        // on shard 1 (lower load), while the parent block lives on shard 0 —
        // a shard crossing.  With sticky routing the continuation inherits shard 0
        // from the stored state, the CRTC on shard 0 receives the full chain, and
        // a full-prefix query returns score 2 instead of ≤1.
        let indexer = make_indexer(2, 2);

        // Root: 1 block — shallow, stores FNV state on block_hash=301.
        indexer
            .apply_event(stored_event(1, None, vec![block(301, 31)]))
            .await;

        // Continuation: finalises the prefix and adds more blocks.
        indexer
            .apply_event(stored_event(
                1,
                Some(301),
                vec![block(302, 32), block(303, 33)],
            ))
            .await;

        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![LocalBlockHash(31), LocalBlockHash(32)])
            .await
            .expect("query should succeed");
        let best = overlap.scores.values().copied().max().unwrap_or(0);
        assert_eq!(
            best, 2,
            "continuation must land on the same shard as its root so the CRTC has the full chain"
        );
    }

    #[tokio::test]
    async fn full_root_batch_reuses_existing_shallow_prefix_assignment() {
        // The same logical [A, B] prefix can arrive as a shallow root [A]
        // followed by continuation [B], or as one full root batch [A, B].
        // Once [A] has claimed a shard, the full root batch must reuse that
        // prefix alias instead of assigning [A, B] to a different shard.
        let indexer = make_indexer(2, 2);

        indexer
            .apply_event(stored_event(1, None, vec![block(501, 51)]))
            .await;
        indexer
            .apply_event(stored_event(2, None, vec![block(501, 51), block(502, 52)]))
            .await;
        indexer
            .apply_event(stored_event(1, Some(501), vec![block(502, 52)]))
            .await;

        indexer.flush().await;

        let prefix_a = indexer.branch_key_for_local_hashes(&local_hashes(&[51]));
        let prefix_ab = indexer.branch_key_for_local_hashes(&local_hashes(&[51, 52]));
        assert_eq!(
            indexer.lookup_shard(prefix_a),
            indexer.lookup_shard(prefix_ab)
        );

        let overlap = indexer
            .find_matches(local_hashes(&[51, 52]))
            .await
            .expect("query should succeed");
        assert_eq!(score(&overlap, 1), Some(2));
        assert_eq!(score(&overlap, 2), Some(2));
    }

    #[tokio::test]
    async fn multi_hop_sticky_routing_preserves_shard_through_intermediate_lookups() {
        // prefix_depth=4, 2 shards.  The chain is built in three hops:
        //   hop 0 (root):     1 block  → depth 1, stored in block_to_fnv_state
        //   hop 1 (cont):     1 block  → depth 2, still in block_to_fnv_state
        //   hop 2 (cont):     1 block  → depth 3, still in block_to_fnv_state
        //   hop 3 (cont):     1 block  → depth 4, finalized → block_to_shard
        //
        // Each hop is a Case A lookup.  A bug that drops the shard field on any
        // intermediate lookup would cause the finalization hop to call assign_shard
        // instead of inheriting, potentially landing on a different shard than the root.
        let indexer = make_indexer(2, 4);

        indexer
            .apply_event(stored_event(1, None, vec![block(401, 41)]))
            .await;
        indexer
            .apply_event(stored_event(1, Some(401), vec![block(402, 42)]))
            .await;
        indexer
            .apply_event(stored_event(1, Some(402), vec![block(403, 43)]))
            .await;
        indexer
            .apply_event(stored_event(1, Some(403), vec![block(404, 44)]))
            .await;

        indexer.flush().await;

        let overlap = indexer
            .find_matches(vec![
                LocalBlockHash(41),
                LocalBlockHash(42),
                LocalBlockHash(43),
                LocalBlockHash(44),
            ])
            .await
            .expect("query should succeed");
        let best = overlap.scores.values().copied().max().unwrap_or(0);
        assert_eq!(
            best, 4,
            "all hops must land on the same shard so the full 4-block chain is visible"
        );
    }
}
