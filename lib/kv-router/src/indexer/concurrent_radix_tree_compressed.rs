// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concurrent Radix Tree (compressed trie) implementation for KV cache routing.
//!
//! This module provides a thread-safe radix tree data structure that enables concurrent
//! `find_matches` operations while maintaining correctness for write operations.
//!
//! Unlike a regular trie where each node holds a single hash, each node here holds
//! a compressed edge: a `Vec` of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
//! Per-worker validity within each edge is tracked as a match index (cutoff) rather than
//! a simple present/absent flag. Nodes support splitting (when a partial match requires
//! divergent paths) but not merging.
//!
//! # Key Data Structures
//!
//! Each node contains:
//! - `edge`: the sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs
//! - `edge_index`: reverse lookup from `ExternalSequenceBlockHash` to position in `edge`,
//!   enabling O(1) position queries during removal.
//! - `full_edge_workers`: workers with full edge coverage (fast path set)
//! - `worker_cutoffs`: workers with partial coverage, mapping to their match index `k`,
//!   meaning the worker has cached blocks `edge[0..k]` with `0 < k < edge.len()`.
//! - `children`: child nodes keyed by the first `LocalBlockHash` of the child's edge
//!
//! # Removal Semantics
//!
//! When a remove event arrives for worker `w` at edge position `i`:
//! - current_cutoff = `edge.len()` if `w` is in `full_edge_workers`, else `worker_cutoffs[w]`
//! - If `i >= current_cutoff`: **no-op** (block is already beyond the worker's coverage)
//! - If `i < current_cutoff`: new_cutoff = `i`
//!   - If new_cutoff == 0: remove worker entirely from this node
//!   - Else: move worker to `worker_cutoffs[w] = new_cutoff`
//! - Worker lookup entries for the newly uncovered suffix are scrubbed eagerly
//!
//! Removal does NOT perform structural splits. Multiple workers can independently reduce
//! their match indices without fragmenting the tree, accurately tracking each worker's
//! individual eviction patterns.
//!
//! # Split Semantics (during store only)
//!
//! When a new store requires splitting an edge at position `pos`:
//! - `full_edge_workers`: full in both prefix (unchanged) and suffix
//! - `worker_cutoffs[w] = k` where `k >= pos`: promoted to full in prefix;
//!   in suffix with `adj = k - pos` (partial if `adj > 0`, absent if `adj == 0`)
//! - `worker_cutoffs[w] = k` where `k < pos`: unchanged in prefix, absent from suffix
//!
//! # Concurrency Model
//!
//! - Multiple `find_matches` can run in parallel (read locks only)
//! - Write operations (`apply_event`, `remove_worker`) acquire write locks
//! - Each worker thread owns its own `WorkerLookup`; no cross-thread lookup contention
//! - Deadlock prevention: always lock parent before child (hand-over-hand)
//! - Cross-thread splits: stale lookup entries are resolved lazily via `resolve_lookup`
//!
//! # Limitations vs RadixTree
//!
//! - Does NOT support `expiration_duration` / frequency tracking
//! - `new_with_frequency()` is not provided
//! - `find_matches` does not populate `OverlapScores.frequencies`

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{SyncIndexer, WorkerTask};
use crate::protocols::*;

macro_rules! read_lock {
    ($self:expr, $lock:expr) => {
        $lock.read()
    };
}

/// Thread-safe shared reference to a Node.
type SharedNode = Arc<RwLock<Node>>;

/// Per-worker block-hash → node map.
///
/// Maps each `ExternalSequenceBlockHash` to the node whose `edge` contains it.
/// Position within the edge is resolved via `Node::edge_index` (O(1)) rather than
/// stored here, keeping the map compact and correct across concurrent splits.
type WorkerLookup = FxHashMap<ExternalSequenceBlockHash, SharedNode>;

/// A node in the concurrent radix tree.
///
/// Stores a compressed edge with per-worker match indices. Workers with full coverage
/// live in `full_edge_workers` for O(1) set membership tests on the common fast path.
/// Workers with partial coverage live in `worker_cutoffs`.
#[derive(Debug)]
struct Node {
    /// Compressed edge: sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
    /// Empty for the root node; non-empty for all other nodes.
    edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` → position in `edge`.
    /// Provides O(1) position lookup during removal, avoiding a linear scan.
    edge_index: FxHashMap<ExternalSequenceBlockHash, usize>,
    /// Workers with partial edge coverage. `worker_cutoffs[w] = k` means worker `w`
    /// has cached `edge[0..k]`, where `0 < k < edge.len()`.
    worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    /// Workers with full edge coverage (match index == edge.len()).
    full_edge_workers: FxHashSet<WorkerWithDpRank>,
    /// Child nodes, keyed by the first `LocalBlockHash` of the child's edge.
    children: FxHashMap<LocalBlockHash, SharedNode>,
}

impl Node {
    fn new() -> Self {
        Self {
            edge: Vec::new(),
            edge_index: FxHashMap::default(),
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers: FxHashSet::default(),
            children: FxHashMap::default(),
        }
    }

    fn has_any_workers(&self) -> bool {
        !self.full_edge_workers.is_empty() || !self.worker_cutoffs.is_empty()
    }

    #[inline]
    fn current_cutoff(&self, worker: WorkerWithDpRank) -> usize {
        if self.full_edge_workers.contains(&worker) {
            self.edge.len()
        } else {
            self.worker_cutoffs.get(&worker).copied().unwrap_or(0)
        }
    }

    #[inline]
    fn covers_pos(&self, worker: WorkerWithDpRank, pos: usize) -> bool {
        self.full_edge_workers.contains(&worker)
            || matches!(self.worker_cutoffs.get(&worker), Some(&cutoff) if pos < cutoff)
    }

    // Descendants are only reachable through full-edge coverage; partial workers stop in this node.
    fn clear_children_if_unreachable(&mut self) {
        if self.full_edge_workers.is_empty() {
            self.children.clear();
        }
    }

    // These hashes are no longer covered after a cutoff shrink and must be scrubbed from lookup.
    fn uncovered_suffix_hashes(&self, cutoff: usize) -> Vec<ExternalSequenceBlockHash> {
        debug_assert!(cutoff <= self.edge.len());
        self.edge[cutoff..].iter().map(|&(_, hash)| hash).collect()
    }

    #[inline]
    fn drop_worker(&mut self, worker: WorkerWithDpRank) {
        self.full_edge_workers.remove(&worker);
        self.worker_cutoffs.remove(&worker);
        self.clear_children_if_unreachable();
    }

    #[inline]
    fn promote_to_full(&mut self, worker: WorkerWithDpRank) {
        if !self.full_edge_workers.contains(&worker) {
            self.worker_cutoffs.remove(&worker);
            self.full_edge_workers.insert(worker);
        }
    }

    #[inline]
    fn remove_worker_at_pos(
        &mut self,
        worker: WorkerWithDpRank,
        pos: usize,
        removed_hash: ExternalSequenceBlockHash,
    ) -> RemoveOutcome {
        let current_cutoff = self.current_cutoff(worker);
        if pos >= current_cutoff {
            // Duplicate remove for an already-uncovered hash: just scrub this lookup entry.
            return RemoveOutcome {
                removed: 0,
                stale_hashes: vec![removed_hash],
            };
        }

        let new_cutoff = pos;
        let removed = current_cutoff - new_cutoff;
        let stale_hashes = self.uncovered_suffix_hashes(new_cutoff);

        if new_cutoff == 0 {
            self.drop_worker(worker);
        } else {
            self.full_edge_workers.remove(&worker);
            self.worker_cutoffs.insert(worker, new_cutoff);
            self.clear_children_if_unreachable();
        }

        RemoveOutcome {
            removed,
            stale_hashes,
        }
    }

    // Used by dump/restore to ignore dead child pointers that may still exist in the live tree.
    fn live_children(&self) -> Vec<SharedNode> {
        self.children
            .values()
            .filter(|child| {
                let guard = child.read();
                guard.has_any_workers() || !guard.children.is_empty()
            })
            .cloned()
            .collect()
    }

    // Dump-time merge for passthrough nodes with identical full-coverage worker sets.
    fn can_merge_with_only_child(&self, live_children: &[SharedNode]) -> bool {
        self.worker_cutoffs.is_empty() && live_children.len() == 1 && {
            let child_guard = live_children[0].read();
            child_guard.full_edge_workers == self.full_edge_workers
                && child_guard.worker_cutoffs.is_empty()
                && child_guard.has_any_workers()
        }
    }
}

/// Data returned by [`ConcurrentRadixTreeCompressed::split_node`] for deferred lookup updates.
///
/// Callers must call [`ConcurrentRadixTreeCompressed::apply_split_lookup`] **after**
/// dropping the write guard to avoid holding the write lock during O(workers × edge_len)
/// HashMap insertions.
struct SplitLookupData {
    suffix: SharedNode,
}

struct RemoveOutcome {
    removed: usize,
    stale_hashes: Vec<ExternalSequenceBlockHash>,
}

/// Thread-safe radix tree (compressed trie) for concurrent KV cache lookups.
pub struct ConcurrentRadixTreeCompressed {
    /// The root of the radix tree. Has an empty edge and only contains children.
    root: SharedNode,

    tree_sizes: DashMap<WorkerWithDpRank, AtomicUsize, FxBuildHasher>,
}

impl Default for ConcurrentRadixTreeCompressed {
    fn default() -> Self {
        Self::new()
    }
}

// Dropping nodes can cause a cascade of drops that overflow the stack.
// This custom drop uses an iterative approach.
impl Drop for ConcurrentRadixTreeCompressed {
    fn drop(&mut self) {
        let mut stack: Vec<SharedNode> = Vec::new();
        {
            let mut root = self.root.write();
            stack.extend(root.children.drain().map(|(_, v)| v));
        }
        while let Some(node) = stack.pop() {
            if let Ok(rwlock) = Arc::try_unwrap(node) {
                let mut inner = rwlock.into_inner();
                stack.extend(inner.children.drain().map(|(_, v)| v));
            }
        }
    }
}

impl ConcurrentRadixTreeCompressed {
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(Node::new())),
            tree_sizes: DashMap::with_hasher(FxBuildHasher),
        }
    }

    // ------------------------------------------------------------------
    // Lookup resolution helpers
    // ------------------------------------------------------------------

    /// Search a node's subtree for the node whose edge contains `hash`.
    /// Used to resolve stale lookup entries caused by cross-thread splits.
    fn find_in_subtree(start: &SharedNode, hash: ExternalSequenceBlockHash) -> Option<SharedNode> {
        let mut stack = Vec::new();
        {
            let guard = start.read();
            stack.extend(guard.children.values().cloned());
        }
        while let Some(node) = stack.pop() {
            let guard = node.read();
            if guard.edge_index.contains_key(&hash) {
                drop(guard);
                return Some(node);
            }
            stack.extend(guard.children.values().cloned());
        }
        None
    }

    /// Look up `hash` in a worker's lookup, resolving stale entries caused by
    /// cross-thread splits. Returns the `SharedNode` whose edge contains `hash`.
    fn resolve_lookup(
        worker_lookup: &mut WorkerLookup,
        hash: ExternalSequenceBlockHash,
    ) -> Option<SharedNode> {
        let node = worker_lookup.get(&hash)?.clone();

        // Fast path: hash is still in this node's edge_index.
        let found = {
            let guard = node.read();
            guard.edge_index.contains_key(&hash)
        };
        if found {
            return Some(node);
        }

        // Slow path: hash was moved to a descendant by a cross-thread split.
        let resolved = Self::find_in_subtree(&node, hash)?;
        worker_lookup.insert(hash, resolved.clone());
        Some(resolved)
    }

    // ------------------------------------------------------------------
    // Split helpers
    // ------------------------------------------------------------------

    /// Split a node's edge at position `pos` (caller holds the node's write lock).
    ///
    /// Splits `node.edge` into prefix `edge[..pos]` (stays in `node`) and suffix
    /// `edge[pos..]` (moved to a new child node). Updates `edge_index` for both
    /// halves and distributes workers according to their match indices.
    ///
    /// Worker distribution:
    /// - `full_edge_workers`: full in both prefix (unchanged) and suffix
    /// - `worker_cutoffs[w] = k`, `k >= pos`: promoted to full in prefix;
    ///   suffix gets `adj = k - pos` (partial if > 0, absent if == 0)
    /// - `worker_cutoffs[w] = k`, `k < pos`: unchanged in prefix, absent from suffix
    ///
    /// Returns `SplitLookupData`; caller must call `apply_split_lookup` after releasing
    /// the write guard.
    ///
    /// `pos` must satisfy `0 < pos < node.edge.len()`.
    fn split_node(node: &mut Node, pos: usize) -> SplitLookupData {
        debug_assert!(
            pos > 0 && pos < node.edge.len(),
            "split position {pos} out of range for edge length {}",
            node.edge.len()
        );

        let suffix_edge = node.edge.split_off(pos);
        let suffix_first_local = suffix_edge[0].0;
        let prefix_len = pos;

        // Build suffix edge_index (positions reindexed from 0).
        let mut suffix_edge_index =
            FxHashMap::with_capacity_and_hasher(suffix_edge.len(), FxBuildHasher);
        for (i, &(_, h)) in suffix_edge.iter().enumerate() {
            suffix_edge_index.insert(h, i);
        }
        // Remove suffix hashes from the prefix edge_index.
        for &(_, h) in &suffix_edge {
            node.edge_index.remove(&h);
        }

        // Distribute workers: full stays full in both; partial workers may be promoted.
        let mut suffix_full =
            FxHashSet::with_capacity_and_hasher(node.full_edge_workers.len(), FxBuildHasher);
        let mut suffix_cutoffs =
            FxHashMap::with_capacity_and_hasher(node.worker_cutoffs.len(), FxBuildHasher);
        let mut to_promote: Vec<WorkerWithDpRank> = Vec::new();

        for &w in &node.full_edge_workers {
            suffix_full.insert(w);
        }
        for (&w, &k) in &node.worker_cutoffs {
            if k >= prefix_len {
                // Covers the full prefix → promote to full in prefix.
                to_promote.push(w);
                let adj = k - prefix_len;
                if adj > 0 {
                    suffix_cutoffs.insert(w, adj);
                }
                // adj == 0: exact split point, absent from suffix.
            }
            // k < prefix_len: stays partial in prefix (same k), absent from suffix.
        }
        for w in &to_promote {
            node.worker_cutoffs.remove(w);
            node.full_edge_workers.insert(*w);
        }

        let suffix_children = std::mem::take(&mut node.children);
        let suffix = Arc::new(RwLock::new(Node {
            edge: suffix_edge,
            edge_index: suffix_edge_index,
            worker_cutoffs: suffix_cutoffs,
            full_edge_workers: suffix_full,
            children: suffix_children,
        }));
        node.children.insert(suffix_first_local, suffix.clone());

        SplitLookupData { suffix }
    }

    /// Apply deferred lookup updates after `split_node`.
    ///
    /// Updates worker lookup maps so entries for blocks that moved to the suffix now
    /// point to the suffix node. Must be called **after** the write guard is dropped.
    fn apply_split_lookup(
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        split: SplitLookupData,
    ) {
        let guard = split.suffix.read();
        for &w in &guard.full_edge_workers {
            if let Some(wl) = lookup.get_mut(&w) {
                for &(_, h) in &guard.edge {
                    wl.insert(h, split.suffix.clone());
                }
            }
        }
        for (&w, &k) in &guard.worker_cutoffs {
            if let Some(wl) = lookup.get_mut(&w) {
                for &(_, h) in &guard.edge[..k] {
                    wl.insert(h, split.suffix.clone());
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // find_matches
    // ------------------------------------------------------------------

    /// Traverse the radix tree to find the best match for a given sequence of
    /// [`LocalBlockHash`]es.
    ///
    /// Workers in `full_edge_workers` are tracked in the `active` set and continue
    /// into children. Workers in `worker_cutoffs` are scored at the node where their
    /// cutoff falls short and are never propagated into children.
    pub fn find_matches_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        let mut active: FxHashSet<WorkerWithDpRank> = FxHashSet::default();
        let mut active_count: usize = 0;
        let mut matched_depth: u32 = 0;
        let mut seq_pos: usize = 0;
        let mut first_node = true;

        let mut next_child = {
            let root_guard = read_lock!(self, self.root);
            root_guard.children.get(&sequence[0]).cloned()
        };

        loop {
            if seq_pos >= sequence.len() {
                break;
            }
            let child = match next_child.take() {
                Some(c) => c,
                None => break,
            };

            let edge_len;
            let edge_match_len;
            {
                let guard = read_lock!(self, child);
                edge_len = guard.edge.len();
                let walk_len = edge_len.min(sequence.len() - seq_pos);

                // First element is guaranteed by the parent's children HashMap lookup.
                let mut match_len = 1;
                for i in 1..walk_len {
                    if guard.edge[i].0 != sequence[seq_pos + i] {
                        break;
                    }
                    match_len += 1;
                }
                edge_match_len = match_len;

                let prev_depth = matched_depth;

                if first_node {
                    // Seed active set from full-edge workers (they can continue to children).
                    // Score partial workers immediately; they never continue into children.
                    active = guard.full_edge_workers.clone();
                    active_count = active.len();
                    for (&w, &k) in &guard.worker_cutoffs {
                        let contribution = k.min(edge_match_len) as u32;
                        if contribution > 0 {
                            scores.scores.insert(w, contribution);
                        }
                    }
                    first_node = false;
                } else {
                    let has_partial = !guard.worker_cutoffs.is_empty();
                    if has_partial {
                        // Slow path: check each active worker against both maps.
                        active.retain(|w| {
                            if guard.full_edge_workers.contains(w) {
                                true
                            } else if let Some(&k) = guard.worker_cutoffs.get(w) {
                                let effective = k.min(edge_match_len) as u32;
                                scores.scores.insert(*w, prev_depth + effective);
                                false
                            } else {
                                scores.scores.insert(*w, prev_depth);
                                false
                            }
                        });
                    } else {
                        // Fast path: no partial workers — all coverage is full or absent.
                        let full_count = guard.full_edge_workers.len();
                        if full_count != active_count {
                            active.retain(|w| {
                                if guard.full_edge_workers.contains(w) {
                                    true
                                } else {
                                    scores.scores.insert(*w, prev_depth);
                                    false
                                }
                            });
                        }
                        // full_count == active_count: sets are identical (fast path).
                    }
                    active_count = active.len();
                }

                next_child = if edge_match_len == edge_len
                    && active_count > 0
                    && seq_pos + edge_match_len < sequence.len()
                {
                    guard
                        .children
                        .get(&sequence[seq_pos + edge_match_len])
                        .cloned()
                } else {
                    None
                };
            }

            if active_count == 0 {
                break;
            }
            matched_depth += edge_match_len as u32;
            if edge_match_len < edge_len {
                break;
            }
            seq_pos += edge_match_len;
            if early_exit && active_count == 1 {
                break;
            }
        }

        for worker in &active {
            scores.scores.insert(*worker, matched_depth);
        }
        for worker in scores.scores.keys() {
            if let Some(s) = self.tree_sizes.get(worker) {
                scores.tree_sizes.insert(*worker, s.load(Ordering::Relaxed));
            }
        }
        scores
    }

    // ------------------------------------------------------------------
    // apply_event dispatch
    // ------------------------------------------------------------------

    fn apply_event(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);
        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        match op {
            KvCacheEventData::Stored(op) => self.apply_stored(lookup, worker, op, id),
            KvCacheEventData::Removed(op) => self.apply_removed(lookup, worker, op, id),
            KvCacheEventData::Cleared => {
                lookup.entry(worker).or_default();
                self.tree_sizes
                    .entry(worker)
                    .or_insert_with(|| AtomicUsize::new(0));
                self.clear_all_blocks(lookup, worker.worker_id);
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------------
    // apply_stored
    // ------------------------------------------------------------------

    fn apply_stored(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        lookup.entry(worker).or_default();

        let parent = match op.parent_hash {
            Some(parent_hash) => {
                // Retry loop: re-resolve if a concurrent split moves parent_hash
                // into a descendant between resolve_lookup and the write lock below.
                loop {
                    let node = {
                        let wl = lookup.get_mut(&worker).unwrap();
                        match Self::resolve_lookup(wl, parent_hash) {
                            Some(n) => n,
                            None => {
                                tracing::warn!(
                                    worker_id = worker.worker_id.to_string(),
                                    dp_rank = worker.dp_rank,
                                    id,
                                    parent_hash = ?op.parent_hash,
                                    num_blocks = op.blocks.len(),
                                    "Failed to find parent block; skipping store operation"
                                );
                                return Err(KvCacheEventError::ParentBlockNotFound);
                            }
                        }
                    };

                    // Verify the worker still covers parent_hash. A prior removal may
                    // have reduced the worker's cutoff past this position, leaving a
                    // stale entry in the lookup map.
                    {
                        let guard = node.read();
                        if let Some(&pos) = guard.edge_index.get(&parent_hash)
                            && !guard.covers_pos(worker, pos)
                        {
                            let cutoff = guard.current_cutoff(worker);
                            tracing::warn!(
                                worker_id = worker.worker_id.to_string(),
                                dp_rank = worker.dp_rank,
                                id,
                                parent_hash = ?parent_hash,
                                pos,
                                cutoff,
                                "Stale parent: worker no longer covers parent_hash; rejecting store"
                            );
                            drop(guard);
                            let wl = lookup.get_mut(&worker).unwrap();
                            wl.remove(&parent_hash);
                            return Err(KvCacheEventError::ParentBlockNotFound);
                        }
                    }

                    // If parent_hash is not the tail of the node's edge, split so it becomes tail.
                    // We check edge_index inside the write lock: if parent_hash is absent, a
                    // concurrent split moved it to a descendant — retry resolve from the top.
                    let split_data = {
                        let mut guard = node.write();
                        if !guard.edge_index.contains_key(&parent_hash) {
                            // Concurrent split moved parent_hash; retry resolve.
                            continue;
                        }
                        if !guard.edge.is_empty() && guard.edge.last().unwrap().1 != parent_hash {
                            guard
                                .edge
                                .iter()
                                .position(|&(_, h)| h == parent_hash)
                                .map(|pos| Self::split_node(&mut guard, pos + 1))
                        } else {
                            None
                        }
                    };
                    if let Some(split) = split_data {
                        Self::apply_split_lookup(lookup, split);
                    }

                    break node;
                }
            }
            None => self.root.clone(),
        };

        let num_blocks_added =
            self.insert_blocks_from(lookup, worker, &parent, op.parent_hash, &op.blocks);

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_add(num_blocks_added, Ordering::Relaxed);
            }
            None => {
                self.tree_sizes
                    .insert(worker, AtomicUsize::new(num_blocks_added));
            }
        }

        Ok(())
    }

    fn insert_blocks_from(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        parent: &SharedNode,
        seed_hash: Option<ExternalSequenceBlockHash>,
        blocks: &[KvCacheStoredBlockData],
    ) -> usize {
        let mut current_parent = parent.clone();
        let mut remaining = blocks;
        let mut num_blocks_added = 0usize;
        // Track the last ExternalSequenceBlockHash we matched to detect if
        // `current_parent` was split by a concurrent thread between iterations.
        // A split shortens `current_parent`'s edge and moves our last-matched
        // hash into a new suffix child. We detect this cheaply inside the write
        // lock we already take on `current_parent`, so no extra lock is needed
        // in the common case.
        //
        // Seeded with parent_hash so the very first iteration detects a split
        // that occurred after apply_stored released its write lock but before
        // we acquired ours here.
        let mut last_ext_hash: Option<ExternalSequenceBlockHash> = seed_hash;

        while !remaining.is_empty() {
            let first_local = remaining[0].tokens_hash;

            let child = {
                let mut parent_guard = current_parent.write();

                // Detect concurrent split: if last_ext_hash is no longer in
                // this node's edge_index, another thread shortened this edge.
                // Drop the lock, re-resolve to the correct suffix node, retry.
                if let Some(hash) = last_ext_hash
                    && !parent_guard.edge_index.contains_key(&hash)
                {
                    drop(parent_guard);
                    let wl = lookup.get_mut(&worker).unwrap();
                    if let Some(resolved) = Self::resolve_lookup(wl, hash) {
                        current_parent = resolved;
                    }
                    continue;
                }

                match parent_guard.children.get(&first_local).cloned() {
                    Some(existing) => existing,
                    None => {
                        // No existing child — create a new node for all remaining blocks.
                        let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = remaining
                            .iter()
                            .map(|b| (b.tokens_hash, b.block_hash))
                            .collect();
                        let mut edge_index =
                            FxHashMap::with_capacity_and_hasher(edge.len(), FxBuildHasher);
                        for (i, &(_, h)) in edge.iter().enumerate() {
                            edge_index.insert(h, i);
                        }
                        let mut full_edge_workers =
                            FxHashSet::with_capacity_and_hasher(1, FxBuildHasher);
                        full_edge_workers.insert(worker);

                        let new_node = Arc::new(RwLock::new(Node {
                            edge,
                            edge_index,
                            worker_cutoffs: FxHashMap::default(),
                            full_edge_workers,
                            children: FxHashMap::default(),
                        }));
                        parent_guard.children.insert(first_local, new_node.clone());
                        drop(parent_guard);

                        let wl = lookup.get_mut(&worker).unwrap();
                        for b in remaining {
                            if wl.insert(b.block_hash, new_node.clone()).is_none() {
                                num_blocks_added += 1;
                            }
                        }
                        return num_blocks_added;
                    }
                }
            };

            {
                let mut child_guard = child.write();
                let edge_len = child_guard.edge.len();

                let mut match_len = 0;
                for (edge_elem, rem_elem) in child_guard.edge.iter().zip(remaining.iter()) {
                    if edge_elem.0 != rem_elem.tokens_hash {
                        break;
                    }
                    if edge_elem.1 != rem_elem.block_hash {
                        tracing::warn!(
                            expected = ?rem_elem.block_hash,
                            actual = ?edge_elem.1,
                            "block_hash mismatch: sequence hashes should be uniform across workers"
                        );
                    }
                    match_len += 1;
                }

                debug_assert!(
                    match_len >= 1,
                    "first hash must match since child was found by it"
                );

                if match_len < edge_len {
                    // Partial edge match: split at match_len, add worker to prefix.
                    let split = Self::split_node(&mut child_guard, match_len);

                    // Ensure worker has full coverage of the prefix.
                    child_guard.promote_to_full(worker);

                    let tail = &remaining[match_len..];
                    if !tail.is_empty() {
                        // Create new tail node for the worker's additional blocks.
                        let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> =
                            tail.iter().map(|b| (b.tokens_hash, b.block_hash)).collect();
                        let mut edge_index =
                            FxHashMap::with_capacity_and_hasher(edge.len(), FxBuildHasher);
                        for (i, &(_, h)) in edge.iter().enumerate() {
                            edge_index.insert(h, i);
                        }
                        let mut full_edge_workers =
                            FxHashSet::with_capacity_and_hasher(1, FxBuildHasher);
                        full_edge_workers.insert(worker);
                        let tail_first_local = tail[0].tokens_hash;

                        let new_node = Arc::new(RwLock::new(Node {
                            edge,
                            edge_index,
                            worker_cutoffs: FxHashMap::default(),
                            full_edge_workers,
                            children: FxHashMap::default(),
                        }));
                        child_guard
                            .children
                            .insert(tail_first_local, new_node.clone());
                        drop(child_guard);

                        Self::apply_split_lookup(lookup, split);

                        let wl = lookup.get_mut(&worker).unwrap();
                        for b in &remaining[..match_len] {
                            if wl.insert(b.block_hash, child.clone()).is_none() {
                                num_blocks_added += 1;
                            }
                        }
                        for b in tail {
                            if wl.insert(b.block_hash, new_node.clone()).is_none() {
                                num_blocks_added += 1;
                            }
                        }
                    } else {
                        drop(child_guard);
                        Self::apply_split_lookup(lookup, split);

                        let wl = lookup.get_mut(&worker).unwrap();
                        for b in &remaining[..match_len] {
                            if wl.insert(b.block_hash, child.clone()).is_none() {
                                num_blocks_added += 1;
                            }
                        }
                    }
                    return num_blocks_added;
                }

                // Full edge match: upgrade worker to full coverage if necessary.
                child_guard.promote_to_full(worker);
                drop(child_guard);

                let wl = lookup.get_mut(&worker).unwrap();
                for b in &remaining[..edge_len] {
                    if wl.insert(b.block_hash, child.clone()).is_none() {
                        num_blocks_added += 1;
                    }
                }

                last_ext_hash = Some(remaining[edge_len - 1].block_hash);
                remaining = &remaining[edge_len..];
                current_parent = child;
            }
        }

        num_blocks_added
    }

    // ------------------------------------------------------------------
    // apply_removed
    // ------------------------------------------------------------------

    /// Apply a remove operation (eviction).
    ///
    /// For each evicted block hash, finds its position in the node via `edge_index` (O(1)).
    /// Updates the worker's match index without splitting the tree:
    /// - `pos >= current_cutoff`: no-op (already beyond coverage)
    /// - `pos < current_cutoff`: `new_cutoff = pos`; moves worker to `worker_cutoffs`
    ///   or removes entirely if `new_cutoff == 0`.
    ///
    /// Lookup entries for the newly uncovered suffix are removed eagerly so
    /// later duplicate remove events fast-path through the missing-hash case.
    fn apply_removed(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        if !lookup.contains_key(&worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }

        let mut total_removed = 0usize;

        'outer: for block_hash in op.block_hashes {
            let mut cur_node = {
                let Some(wl) = lookup.get_mut(&worker) else {
                    continue;
                };
                match Self::resolve_lookup(wl, block_hash) {
                    Some(n) => n,
                    None => {
                        tracing::debug!(
                            worker_id = worker.worker_id.to_string(),
                            dp_rank = worker.dp_rank,
                            id,
                            block_hash = ?block_hash,
                            "Block not found during remove; skipping"
                        );
                        continue;
                    }
                }
            };

            loop {
                // Returns Some(remove_outcome) on success, None if the node is stale
                // (hash was moved to a descendant by a concurrent split).
                let update: Option<RemoveOutcome> = {
                    let mut guard = cur_node.write();

                    guard
                        .edge_index
                        .get(&block_hash)
                        .copied()
                        .map(|pos| guard.remove_worker_at_pos(worker, pos, block_hash))
                };

                match update {
                    Some(outcome) => {
                        total_removed += outcome.removed;
                        if let Some(wl) = lookup.get_mut(&worker) {
                            for hash in outcome.stale_hashes {
                                wl.remove(&hash);
                            }
                        }
                        continue 'outer;
                    }
                    None => {
                        // Hash was moved to a descendant by a concurrent split.
                        match Self::find_in_subtree(&cur_node, block_hash) {
                            Some(resolved) => {
                                if let Some(wl) = lookup.get_mut(&worker) {
                                    wl.insert(block_hash, resolved.clone());
                                }
                                cur_node = resolved;
                                // Retry the inner loop with the resolved node.
                            }
                            None => {
                                // Hash not found anywhere — evicted by a concurrent clear.
                                tracing::debug!(
                                    worker_id = worker.worker_id.to_string(),
                                    dp_rank = worker.dp_rank,
                                    id,
                                    block_hash = ?block_hash,
                                    "Block not found in subtree during remove; skipping"
                                );
                                if let Some(wl) = lookup.get_mut(&worker) {
                                    wl.remove(&block_hash);
                                }
                                continue 'outer;
                            }
                        }
                    }
                }
            }
        }

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(total_removed))
                })
                .ok();
            }
            None => {
                self.tree_sizes.insert(worker, AtomicUsize::new(0));
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Worker removal / clearing
    // ------------------------------------------------------------------

    fn remove_or_clear_worker_blocks(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker_id: WorkerId,
        keep_worker: bool,
    ) {
        let workers: Vec<WorkerWithDpRank> = lookup
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        for worker in workers {
            if let Some(worker_lookup) = lookup.remove(&worker) {
                let mut seen = FxHashSet::<usize>::default();
                for (_, node) in worker_lookup.into_iter() {
                    let ptr = Arc::as_ptr(&node) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }
                    let mut guard = node.write();
                    guard.drop_worker(worker);
                }

                if keep_worker {
                    lookup.insert(worker, FxHashMap::default());
                    if let Some(size) = self.tree_sizes.get(&worker) {
                        size.store(0, Ordering::Relaxed);
                    }
                } else {
                    self.tree_sizes.remove(&worker);
                }
            }
        }
    }

    fn remove_worker_dp_rank(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) {
        let key = WorkerWithDpRank { worker_id, dp_rank };
        if let Some(worker_lookup) = lookup.remove(&key) {
            let mut seen = FxHashSet::<usize>::default();
            for (_, node) in worker_lookup.into_iter() {
                let ptr = Arc::as_ptr(&node) as usize;
                if !seen.insert(ptr) {
                    continue;
                }
                let mut guard = node.write();
                guard.drop_worker(key);
            }
            self.tree_sizes.remove(&key);
        }
    }

    fn clear_all_blocks(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker_id: WorkerId,
    ) {
        self.remove_or_clear_worker_blocks(lookup, worker_id, true);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn get_workers(&self) -> Vec<WorkerId> {
        let mut worker_ids: Vec<WorkerId> = self
            .tree_sizes
            .iter()
            .map(|entry| entry.key().worker_id)
            .collect();
        worker_ids.sort_unstable();
        worker_ids.dedup();
        worker_ids
    }

    // ------------------------------------------------------------------
    // Tree dump
    // ------------------------------------------------------------------

    fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!("Dumping concurrent radix tree as events");

        let mut events = Vec::new();
        let mut event_id = 0u64;
        let mut queue = VecDeque::new();

        {
            let root_guard = self.root.read();
            for child_node in root_guard.children.values() {
                queue.push_back((child_node.clone(), None::<ExternalSequenceBlockHash>));
            }
        }

        while let Some((start_node, parent_hash)) = queue.pop_front() {
            let mut merged_edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = Vec::new();
            let mut current = start_node;

            loop {
                let guard = current.read();

                if !guard.has_any_workers() && guard.children.is_empty() {
                    break;
                }

                merged_edge.extend_from_slice(&guard.edge);

                let live_children = guard.live_children();

                // Merge condition: this node is a pure passthrough that can be
                // collapsed with its single child. Requires identical worker sets
                // and no partial-coverage cutoffs on either side.
                let can_merge = guard.can_merge_with_only_child(&live_children);

                if can_merge {
                    let next = live_children[0].clone();
                    drop(guard);
                    current = next;
                    continue;
                }

                if merged_edge.is_empty() {
                    drop(guard);
                    break;
                }

                let full_blocks: Vec<KvCacheStoredBlockData> = merged_edge
                    .iter()
                    .map(|&(local, ext)| KvCacheStoredBlockData {
                        tokens_hash: local,
                        block_hash: ext,
                        mm_extra_info: None,
                    })
                    .collect();
                let last_ext = merged_edge.last().unwrap().1;

                for &worker in &guard.full_edge_workers {
                    events.push(RouterEvent::new(
                        worker.worker_id,
                        KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash,
                                blocks: full_blocks.clone(),
                            }),
                            dp_rank: worker.dp_rank,
                        },
                    ));
                    event_id += 1;
                }
                for (&worker, &k) in &guard.worker_cutoffs {
                    events.push(RouterEvent::new(
                        worker.worker_id,
                        KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash,
                                blocks: full_blocks[..k].to_vec(),
                            }),
                            dp_rank: worker.dp_rank,
                        },
                    ));
                    event_id += 1;
                }

                for child in live_children {
                    queue.push_back((child, Some(last_ext)));
                }

                drop(guard);
                break;
            }
        }

        events
    }
}

// ============================================================================
// SyncIndexer implementation for ConcurrentRadixTreeCompressed
// ============================================================================

impl SyncIndexer for ConcurrentRadixTreeCompressed {
    fn worker(&self, event_receiver: flume::Receiver<WorkerTask>) -> anyhow::Result<()> {
        let mut lookup = FxHashMap::default();

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    if let Err(e) = self.apply_event(&mut lookup, event) {
                        tracing::warn!("Failed to apply event: {:?}", e);
                    }
                }
                WorkerTask::RemoveWorker(worker_id) => {
                    self.remove_or_clear_worker_blocks(&mut lookup, worker_id, false);
                }
                WorkerTask::RemoveWorkerDpRank(worker_id, dp_rank) => {
                    self.remove_worker_dp_rank(&mut lookup, worker_id, dp_rank);
                }
                WorkerTask::DumpEvents(_sender) => {
                    let _ = _sender.send(Ok(Vec::new()));
                }
                WorkerTask::Terminate => {
                    break;
                }
            }
        }

        tracing::debug!("ConcurrentRadixTreeCompressed worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.find_matches_impl(sequence, early_exit)
    }

    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        Some(self.dump_tree_as_events())
    }
}
