// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concurrent Radix Tree (compressed trie) implementation for KV cache routing.
//!
//! This module provides a thread-safe radix tree data structure that enables concurrent
//! `find_matches` operations while maintaining correctness for write operations.
//!
//! Unlike a regular trie where each node holds a single hash, each node here holds
//! a compressed edge: a `Vec` of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs
//! with a single worker set for the whole node. Nodes support splitting (when a
//! partial match requires divergent paths) but not merging.
//!
//! # Approximations
//!
//! - `apply_removed` splits a node when the removed hash is at position `i > 0`
//!   within the edge, keeping the prefix `[0..i)` valid and removing the worker
//!   from the suffix `[i..)`. When the first hash (position 0) is removed, the
//!   worker is removed from the entire node. This does NOT cascade to descendants;
//!   stale worker entries in deeper nodes are handled by `find_matches`.
//! - `find_matches` checks worker sets at node boundaries. Within a compressed
//!   edge, per-hash worker differences are not visible.
//!
//! # Limitations vs RadixTree
//!
//! - Does NOT support `expiration_duration` / frequency tracking
//! - `new_with_frequency()` is not provided
//! - `find_matches` does not populate `OverlapScores.frequencies`
//!
//! # Concurrency Model
//!
//! - Multiple `find_matches` can run in parallel (read locks only)
//! - Write operations (`apply_event`, `remove_worker`) acquire write locks
//! - Outer `DashMap` provides shard-level locking for per-worker access.
//!   Inner `RwLock` per worker allows per-worker write concurrency.
//! - Deadlock prevention: always lock parent before child, hand-over-hand locking
//! - Cross-thread splits: when one thread splits a node, other threads' lookup
//!   entries for the suffix hashes become stale. These are resolved lazily via
//!   `resolve_lookup` on next access.

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{SyncIndexer, WorkerTask};
use crate::protocols::*;

/// Thread-safe shared reference to a Block.
type SharedBlock = Arc<RwLock<Block>>;

/// Per-worker block-hash map. Inner RwLock allows concurrent reads of different workers.
type WorkerLookup = FxHashMap<ExternalSequenceBlockHash, SharedBlock>;

/// A node in the concurrent radix tree.
///
/// Each node holds a compressed edge (a sequence of hash pairs) rather than a
/// single hash. The worker set applies to the entire edge: all listed workers
/// have cached every block in this edge.
#[derive(Debug)]
struct Block {
    /// The compressed edge: a sequence of `(LocalBlockHash, ExternalSequenceBlockHash)`
    /// pairs. Empty for the root node; non-empty for all other nodes.
    edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Child nodes, keyed by the first `LocalBlockHash` of the child's edge.
    children: FxHashMap<LocalBlockHash, SharedBlock>,
    /// The set of workers that have cached all blocks in this edge.
    workers: FxHashSet<WorkerWithDpRank>,
}

impl Block {
    /// Create a new root `Block` with an empty edge.
    fn new() -> Self {
        Self {
            edge: Vec::new(),
            children: FxHashMap::default(),
            workers: FxHashSet::default(),
        }
    }
}

/// Thread-safe radix tree (compressed trie) for concurrent KV cache lookups.
///
/// Each node holds a compressed edge of multiple `(LocalBlockHash,
/// ExternalSequenceBlockHash)` pairs with a single worker set. Nodes support
/// splitting but not merging. Blocks from a single store event are placed into
/// at most one new node; separate store events do not get merged into the same
/// node even if they are contiguous.
///
/// # Concurrency Model
///
/// - Multiple `find_matches` can run in parallel (read locks only)
/// - Write operations (`apply_event`, `remove_worker`) acquire write locks
/// - Outer `DashMap` provides shard-level locking for per-worker access.
/// - Inner `RwLock` per worker allows per-worker write concurrency.
/// - Deadlock prevention: always lock parent before child, hand-over-hand locking
pub struct ConcurrentRadixTreeCompressed {
    /// The root of the radix tree. Has an empty edge and only contains children.
    root: SharedBlock,

    tree_sizes: DashMap<WorkerWithDpRank, AtomicUsize, FxBuildHasher>,
}

impl Default for ConcurrentRadixTreeCompressed {
    fn default() -> Self {
        Self::new()
    }
}

// Dropping blocks can cause a cascade of drops that can overflow the stack.
// This custom drop implementation avoids this using an iterative approach.
impl Drop for ConcurrentRadixTreeCompressed {
    fn drop(&mut self) {
        let mut stack: Vec<SharedBlock> = Vec::new();

        // Break root -> children edge up front
        {
            let mut root = self.root.write();
            stack.extend(root.children.drain().map(|(_, v)| v));
        }

        // Iteratively drop blocks to avoid stack overflow on deep trees.
        // Without this loop, dropping `stack` would recursively drop each
        // Arc<RwLock<Block>> through its `children` map.
        while let Some(block) = stack.pop() {
            if let Ok(rwlock) = Arc::try_unwrap(block) {
                let mut inner = rwlock.into_inner();
                stack.extend(inner.children.drain().map(|(_, v)| v));
            }
        }
    }
}

impl ConcurrentRadixTreeCompressed {
    /// Create a new `ConcurrentRadixTreeCompressed`.
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(Block::new())),
            tree_sizes: DashMap::with_hasher(FxBuildHasher),
        }
    }

    // ------------------------------------------------------------------
    // Lookup resolution helpers
    // ------------------------------------------------------------------

    /// Search a node's subtree for a block whose edge contains `hash`.
    /// Used to resolve stale lookup entries caused by cross-thread splits.
    fn find_in_subtree(
        start: &SharedBlock,
        hash: ExternalSequenceBlockHash,
    ) -> Option<SharedBlock> {
        let mut stack = Vec::new();
        {
            let guard = start.read();
            stack.extend(guard.children.values().cloned());
        }
        while let Some(node) = stack.pop() {
            let guard = node.read();
            if guard.edge.iter().any(|&(_, h)| h == hash) {
                drop(guard);
                return Some(node);
            }
            stack.extend(guard.children.values().cloned());
        }
        None
    }

    /// Look up `hash` in a worker's lookup, resolving stale entries caused by
    /// cross-thread splits. If the node that the lookup points to no longer
    /// contains `hash` in its edge (because another thread split the node and
    /// moved `hash` to a descendant), this function walks the subtree to find
    /// the correct node and updates the lookup entry.
    fn resolve_lookup(
        worker_lookup: &mut WorkerLookup,
        hash: ExternalSequenceBlockHash,
    ) -> Option<SharedBlock> {
        let block = worker_lookup.get(&hash)?.clone();

        // Fast path: hash is where the lookup says it is.
        let found = {
            let guard = block.read();
            guard.edge.iter().any(|&(_, h)| h == hash)
        };
        if found {
            return Some(block);
        }

        // Slow path: hash was moved to a descendant by a cross-thread split.
        let resolved = Self::find_in_subtree(&block, hash)?;
        worker_lookup.insert(hash, resolved.clone());
        Some(resolved)
    }

    // ------------------------------------------------------------------
    // Split helpers
    // ------------------------------------------------------------------

    /// Split a block's edge at position `pos`, given a mutable reference to
    /// the block (caller already holds its write lock).
    ///
    /// The edge is divided into:
    /// - prefix `edge[..pos]` -- stays in `block`
    /// - suffix `edge[pos..]` -- placed in a newly allocated child node
    ///
    /// Returns the newly created suffix node.
    ///
    /// `pos` must satisfy `0 < pos < block.edge.len()`.
    fn split_block(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        block: &mut Block,
        pos: usize,
    ) -> SharedBlock {
        debug_assert!(
            pos > 0 && pos < block.edge.len(),
            "split position {pos} out of range for edge length {}",
            block.edge.len()
        );

        let suffix_edge = block.edge.split_off(pos);
        let suffix_first_local = suffix_edge[0].0;

        let suffix_ext_hashes: Vec<ExternalSequenceBlockHash> =
            suffix_edge.iter().map(|&(_, h)| h).collect();

        let suffix_children = std::mem::take(&mut block.children);
        let suffix_workers = block.workers.clone();

        let suffix = Arc::new(RwLock::new(Block {
            edge: suffix_edge,
            children: suffix_children,
            workers: suffix_workers,
        }));

        block.children.insert(suffix_first_local, suffix.clone());

        // Update lookups for workers that are in this thread's lookup.
        // Workers on other threads will resolve stale entries lazily.
        for worker in &block.workers {
            if let Some(wl) = lookup.get_mut(worker) {
                for &ext_hash in &suffix_ext_hashes {
                    wl.insert(ext_hash, suffix.clone());
                }
            }
        }

        suffix
    }

    /// Split a node's edge at position `pos`.
    ///
    /// Convenience wrapper around `split_block` that acquires the node's write
    /// lock. Returns the newly created suffix node.
    fn split_node(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        node: &SharedBlock,
        pos: usize,
    ) -> SharedBlock {
        let mut guard = node.write();
        self.split_block(lookup, &mut guard, pos)
    }

    // ------------------------------------------------------------------
    // find_matches
    // ------------------------------------------------------------------

    /// Traverse the radix tree to find the best match for a given sequence of [`LocalBlockHash`]es.
    ///
    /// Worker sets are checked at node boundaries. Within a compressed edge,
    /// each hash is compared to determine match length, but per-hash worker
    /// membership is not checked. This is faster than a regular trie because
    /// there are fewer nodes to lock and inspect.
    ///
    /// ### Arguments
    ///
    /// * `sequence` - A slice of `LocalBlockHash` representing the sequence to match.
    /// * `early_exit` - A boolean indicating whether to exit early if a single match is found.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    /// Note: `frequencies` field will be empty since frequency tracking is not supported.
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
        let mut current = self.root.clone();
        let mut first_node = true;

        loop {
            if seq_pos >= sequence.len() {
                break;
            }

            // Find child of current matching sequence[seq_pos].
            let child = {
                let guard = current.read();
                guard.children.get(&sequence[seq_pos]).cloned()
            };

            let Some(child) = child else {
                break;
            };

            // Walk child's edge against sequence[seq_pos..] and check workers
            // at this node boundary.
            let edge_len;
            let edge_match_len;
            {
                let guard = child.read();
                edge_len = guard.edge.len();

                let walk_len = edge_len.min(sequence.len() - seq_pos);
                let mut match_len = 0;
                for i in 0..walk_len {
                    if guard.edge[i].0 != sequence[seq_pos + i] {
                        break;
                    }
                    match_len += 1;
                }
                edge_match_len = match_len;

                // Check workers at this node boundary.
                //
                // In a clean tree, workers at a child node are always a subset
                // of the parent (along the same path), so:
                //   - workers can only drop out, never join, as we descend
                //   - if child.workers.len() == active_count, the sets are identical
                //
                // However, because apply_removed does NOT cascade to descendants,
                // a child may transiently have MORE workers than its parent
                // (stale entries from an ancestor remove whose descendant remove
                // events haven't arrived yet). We detect this via
                // child_count > active_count and fall back to a full membership
                // check.
                if first_node {
                    active = guard.workers.clone();
                    active_count = active.len();
                    first_node = false;
                } else {
                    let child_count = guard.workers.len();
                    if child_count != active_count {
                        // Workers changed: either dropped out (child < active)
                        // or stale entries exist (child > active). Retain only
                        // workers present in the child, scoring dropouts.
                        let prev_depth = matched_depth;
                        active.retain(|w| {
                            if guard.workers.contains(w) {
                                true
                            } else {
                                scores.scores.insert(*w, prev_depth);
                                false
                            }
                        });
                        active_count = active.len();
                    }
                    // child_count == active_count: fast path, sets are identical
                    // (or, in the rare edge case, different membership with same
                    // cardinality -- accepted as a transient routing quality
                    // degradation that resolves once pending remove events arrive).
                }
            }

            if active_count == 0 {
                break;
            }

            matched_depth += edge_match_len as u32;

            if edge_match_len < edge_len {
                // Partial edge match: can't continue to children.
                break;
            }

            // Full edge match.
            seq_pos += edge_match_len;

            if early_exit && active_count == 1 {
                break;
            }

            current = child;
        }

        // Record scores for workers that survived through the deepest matched level.
        for worker in &active {
            scores.scores.insert(*worker, matched_depth);
        }

        // Get tree sizes from lookup.
        for worker in scores.scores.keys() {
            if let Some(worker_tree_size) = self.tree_sizes.get(worker) {
                scores
                    .tree_sizes
                    .insert(*worker, worker_tree_size.load(Ordering::Relaxed));
            }
        }

        scores
    }

    // ------------------------------------------------------------------
    // apply_event dispatch
    // ------------------------------------------------------------------

    /// Apply a [`RouterEvent`] to the radix tree.
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

    /// Apply a store operation.
    fn apply_stored(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        lookup.entry(worker).or_default();

        // Find parent block.
        let parent = match op.parent_hash {
            Some(parent_hash) => {
                // Resolve the lookup (may fix a stale entry from a cross-thread split).
                let block = {
                    let wl = lookup.get_mut(&worker).unwrap();
                    match Self::resolve_lookup(wl, parent_hash) {
                        Some(b) => b,
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
                // `lookup` borrow released here.

                // If parent_hash is not the last element in the block's edge,
                // split so that parent_hash becomes the last element.
                let needs_split = {
                    let guard = block.read();
                    if guard.edge.is_empty() || guard.edge.last().unwrap().1 == parent_hash {
                        None
                    } else {
                        guard
                            .edge
                            .iter()
                            .position(|&(_, h)| h == parent_hash)
                            .map(|pos| pos + 1)
                    }
                };

                if let Some(split_pos) = needs_split {
                    self.split_node(lookup, &block, split_pos);
                }

                block
            }
            None => self.root.clone(),
        };

        let num_blocks = op.blocks.len();
        self.insert_blocks_from(lookup, worker, &parent, &op.blocks);

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_add(num_blocks, Ordering::Relaxed);
            }
            None => {
                self.tree_sizes.insert(worker, AtomicUsize::new(num_blocks));
            }
        }

        Ok(())
    }

    /// Insert a sequence of blocks starting from a parent node.
    ///
    /// Uses write locks on parent nodes for atomic check-and-insert of children,
    /// preventing TOCTOU races between concurrent threads. Child nodes are
    /// processed under their own write locks for atomic edge-walk + split.
    fn insert_blocks_from(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        parent: &SharedBlock,
        blocks: &[KvCacheStoredBlockData],
    ) {
        let mut current_parent = parent.clone();
        let mut remaining = blocks;

        while !remaining.is_empty() {
            let first_local = remaining[0].tokens_hash;

            // Atomically check for an existing child under a write lock on the
            // parent. This prevents a race where two threads both see "no child"
            // and both try to insert, with one overwriting the other.
            let child = {
                let mut parent_guard = current_parent.write();
                match parent_guard.children.get(&first_local).cloned() {
                    Some(existing) => existing,
                    None => {
                        // No matching child: create a new node with all
                        // remaining blocks and insert under the same lock.
                        let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = remaining
                            .iter()
                            .map(|b| (b.tokens_hash, b.block_hash))
                            .collect();

                        let mut workers = FxHashSet::default();
                        workers.insert(worker);
                        let new_node = Arc::new(RwLock::new(Block {
                            edge,
                            children: FxHashMap::default(),
                            workers,
                        }));

                        parent_guard.children.insert(first_local, new_node.clone());
                        drop(parent_guard);

                        let worker_lookup = lookup.get_mut(&worker).unwrap();
                        for b in remaining {
                            worker_lookup.insert(b.block_hash, new_node.clone());
                        }
                        return;
                    }
                }
            };
            // parent write lock released here.

            // Process the existing child under its own write lock. This is safe
            // because we already released the parent lock (no nested locks).
            // The write lock on the child ensures that concurrent threads
            // processing the same child are serialized.
            {
                let mut child_guard = child.write();
                let edge_len = child_guard.edge.len();
                let min_len = edge_len.min(remaining.len());

                let mut match_len = 0;
                for i in 0..min_len {
                    if child_guard.edge[i].0 != remaining[i].tokens_hash {
                        break;
                    }
                    if child_guard.edge[i].1 != remaining[i].block_hash {
                        tracing::warn!(
                            expected = ?remaining[i].block_hash,
                            actual = ?child_guard.edge[i].1,
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
                    // Partial edge match: split inline under the held write lock.
                    self.split_block(lookup, &mut child_guard, match_len);

                    // Add worker to the prefix (the child node after split).
                    child_guard.workers.insert(worker);

                    if match_len < remaining.len() {
                        // Create a new sibling for the unmatched tail.
                        let tail = &remaining[match_len..];
                        let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> =
                            tail.iter().map(|b| (b.tokens_hash, b.block_hash)).collect();
                        let tail_first_local = tail[0].tokens_hash;

                        let mut workers = FxHashSet::default();
                        workers.insert(worker);
                        let new_node = Arc::new(RwLock::new(Block {
                            edge,
                            children: FxHashMap::default(),
                            workers,
                        }));

                        child_guard
                            .children
                            .insert(tail_first_local, new_node.clone());
                        drop(child_guard);

                        let worker_lookup = lookup.get_mut(&worker).unwrap();
                        for b in &remaining[..match_len] {
                            worker_lookup.insert(b.block_hash, child.clone());
                        }
                        for b in tail {
                            worker_lookup.insert(b.block_hash, new_node.clone());
                        }
                    } else {
                        drop(child_guard);

                        let worker_lookup = lookup.get_mut(&worker).unwrap();
                        for b in &remaining[..match_len] {
                            worker_lookup.insert(b.block_hash, child.clone());
                        }
                    }
                    return;
                }

                // Full edge match: add worker and continue deeper.
                child_guard.workers.insert(worker);
                drop(child_guard);

                let worker_lookup = lookup.get_mut(&worker).unwrap();
                for b in &remaining[..edge_len] {
                    worker_lookup.insert(b.block_hash, child.clone());
                }

                remaining = &remaining[edge_len..];
                current_parent = child;
            }
        }
    }

    // ------------------------------------------------------------------
    // apply_removed
    // ------------------------------------------------------------------

    /// Apply a remove operation.
    ///
    /// When a hash at position `i > 0` within a compressed edge is removed,
    /// the node is split at `i`: the prefix `[0..i)` keeps the worker
    /// (those blocks are still cached), while the suffix `[i..)` has the
    /// worker removed (block `i` was evicted, invalidating the chain).
    ///
    /// When the removed hash is at position 0, the worker is removed from
    /// the entire node (the first block in the chain is gone, so every
    /// subsequent block is also invalid).
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

        let mut total_removed = 0;

        for block_hash in op.block_hashes {
            // Resolve and remove block_hash from worker's lookup.
            let block = {
                let Some(wl) = lookup.get_mut(&worker) else {
                    continue;
                };
                // Resolve stale lookup before removing.
                let block = match Self::resolve_lookup(wl, block_hash) {
                    Some(b) => b,
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
                };
                wl.remove(&block_hash);
                block
            };
            // `lookup` borrow released here.

            // Find position of block_hash in the block's edge.
            let (pos, edge_len) = {
                let guard = block.read();
                (
                    guard.edge.iter().position(|&(_, h)| h == block_hash),
                    guard.edge.len(),
                )
            };

            match pos {
                Some(pos) if pos > 0 => {
                    // block_hash is in the middle/end of the edge. Split the
                    // node so the prefix [0..pos) keeps the worker and the
                    // suffix [pos..) has the worker removed.
                    let suffix = self.split_node(lookup, &block, pos);

                    // Remove worker from the suffix.
                    let suffix_hashes: Vec<ExternalSequenceBlockHash>;
                    {
                        let mut guard = suffix.write();
                        guard.workers.remove(&worker);
                        suffix_hashes = guard.edge.iter().map(|&(_, h)| h).collect();
                        if guard.workers.is_empty() {
                            guard.children.clear();
                        }
                    }

                    // Remove all suffix hashes from the worker's lookup.
                    // (split_node re-inserted them for this worker)
                    if let Some(wl) = lookup.get_mut(&worker) {
                        for h in &suffix_hashes {
                            wl.remove(h);
                        }
                    }

                    total_removed += edge_len - pos;
                }
                _ => {
                    // pos == 0 or not found in edge: the first block in the
                    // chain was removed (or something unexpected), so remove
                    // the worker from the entire node.
                    let sibling_hashes: Vec<ExternalSequenceBlockHash>;
                    {
                        let mut guard = block.write();
                        guard.workers.remove(&worker);
                        sibling_hashes = guard
                            .edge
                            .iter()
                            .map(|&(_, h)| h)
                            .filter(|&h| h != block_hash)
                            .collect();
                        if guard.workers.is_empty() {
                            guard.children.clear();
                        }
                    }

                    // Remove sibling hashes from worker's lookup.
                    if let Some(wl) = lookup.get_mut(&worker) {
                        for h in &sibling_hashes {
                            wl.remove(h);
                        }
                    }

                    total_removed += edge_len;
                }
            }
        }

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_sub(total_removed, Ordering::Relaxed);
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

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains in lookup with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed from lookup.
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
                // Deduplicate: in a radix tree, multiple ext hashes can map to
                // the same node. Track visited nodes by Arc pointer to avoid
                // redundant operations.
                let mut seen = FxHashSet::<usize>::default();
                for (_, block) in worker_lookup.into_iter() {
                    let ptr = Arc::as_ptr(&block) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }
                    let mut guard = block.write();
                    guard.workers.remove(&worker);
                    if guard.workers.is_empty() {
                        guard.children.clear();
                    }
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
            for (_, block) in worker_lookup.into_iter() {
                let ptr = Arc::as_ptr(&block) as usize;
                if !seen.insert(ptr) {
                    continue;
                }
                let mut guard = block.write();
                guard.workers.remove(&key);
                if guard.workers.is_empty() {
                    guard.children.clear();
                }
            }
            self.tree_sizes.remove(&key);
        }
    }

    /// Clear all blocks for a worker but keep the worker tracked.
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

    /// Get all worker IDs currently tracked in the radix tree.
    /// Returns unique worker_ids (ignoring dp_rank differences).
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

    /// Dump the radix tree as a series of RouterEvents that can reconstruct the tree.
    /// Uses BFS traversal over the shared tree. Each node's compressed edge is
    /// emitted as a single store event per worker. Since all worker/block
    /// membership is stored in the tree nodes themselves, this can be called
    /// from any thread without needing per-thread lookup state.
    fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!("Dumping concurrent radix tree as events");

        let mut events = Vec::new();
        let mut event_id = 0u64;

        let mut queue = VecDeque::new();

        {
            let root_guard = self.root.read();
            for (_, child_block) in &root_guard.children {
                queue.push_back((child_block.clone(), None::<ExternalSequenceBlockHash>));
            }
        }

        while let Some((current_block, parent_hash)) = queue.pop_front() {
            let current_guard = current_block.read();

            debug_assert!(
                !current_guard.edge.is_empty(),
                "non-root block must have non-empty edge"
            );

            let blocks: Vec<KvCacheStoredBlockData> = current_guard
                .edge
                .iter()
                .map(|&(local, ext)| KvCacheStoredBlockData {
                    tokens_hash: local,
                    block_hash: ext,
                    mm_extra_info: None,
                })
                .collect();

            let last_ext = current_guard.edge.last().unwrap().1;

            for worker in &current_guard.workers {
                let event = RouterEvent {
                    worker_id: worker.worker_id,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            blocks: blocks.clone(),
                        }),
                        dp_rank: worker.dp_rank,
                    },
                };
                events.push(event);
                event_id += 1;
            }

            for (_, child_block) in &current_guard.children {
                queue.push_back((child_block.clone(), Some(last_ext)));
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
                    // Handled directly via dump_events() on the shared tree.
                    // Should not be reached, but respond with empty to avoid blocking.
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
