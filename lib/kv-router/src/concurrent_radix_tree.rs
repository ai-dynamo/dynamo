// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concurrent Radix Tree implementation for KV cache routing.
//!
//! This module provides a thread-safe radix tree data structure that enables concurrent
//! `find_matches` operations while maintaining correctness for write operations.
//!
//! Unlike `RadixTree` which uses `Rc<RefCell<>>` and requires single-threaded access,
//! `ConcurrentRadixTree` uses `Arc<RwLock<>>` per node and a
//! `RwLock<FxHashMap<..., RwLock<FxHashMap<...>>>>` for the lookup table.
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
//! - Outer `RwLock` is read-locked on the hot path; structural mutations
//!   (adding/removing workers) are rare. Inner `RwLock` per worker allows
//!   per-worker write concurrency.
//! - Deadlock prevention: always lock parent before child, hand-over-hand locking

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::indexer::{SyncIndexer, WorkerTask};
use crate::protocols::*;

/// Thread-safe shared reference to a Block.
type SharedBlock = Arc<RwLock<Block>>;

/// Per-worker block-hash map. Inner RwLock allows concurrent reads of different workers.
type WorkerLookup = FxHashMap<ExternalSequenceBlockHash, SharedBlock>;

/// A block in the concurrent radix tree.
#[derive(Debug)]
struct Block {
    /// A map of child blocks, keyed by their local block hash.
    children: FxHashMap<LocalBlockHash, SharedBlock>,
    /// The set of workers that have this block cached.
    workers: FxHashSet<WorkerWithDpRank>,
    /// The external sequence block hash for this block (None for root).
    block_hash: Option<ExternalSequenceBlockHash>,
    // NOTE: No recent_uses field.
    // Frequency tracking is not supported - keeps find_matches fully read-only.
}

impl Block {
    /// Create a new `Block` (used for root node).
    fn new() -> Self {
        Self {
            children: FxHashMap::default(),
            workers: FxHashSet::default(),
            block_hash: None,
        }
    }

    /// Create a new `Block` with a specific block hash.
    fn with_hash(block_hash: ExternalSequenceBlockHash) -> Self {
        Self {
            children: FxHashMap::default(),
            workers: FxHashSet::default(),
            block_hash: Some(block_hash),
        }
    }
}

/// Thread-safe radix tree for concurrent KV cache lookups.
///
/// Unlike `RadixTree` which uses `Rc<RefCell<>>` and requires single-threaded access,
/// `ConcurrentRadixTree` uses `Arc<RwLock<>>` per node and a
/// `RwLock<FxHashMap<..., RwLock<FxHashMap<...>>>>` for the lookup table,
/// enabling concurrent `find_matches` operations.
///
/// # Limitations vs RadixTree
///
/// - Does NOT support `expiration_duration` / frequency tracking
/// - `new_with_frequency()` is not provided
/// - `find_matches` does not populate `OverlapScores.frequencies`
///
/// # Concurrency Model
///
/// - Multiple `find_matches` can run in parallel (read locks only)
/// - Write operations (`apply_event`, `remove_worker`) acquire write locks
/// - Outer RwLock is read-locked on the hot path; structural mutations
///   (adding/removing workers) are rare and take a write lock.
/// - Inner `RwLock` per worker allows per-worker write concurrency.
/// - Deadlock prevention: always lock parent before child, hand-over-hand locking
pub struct ConcurrentRadixTree {
    /// This is the root of the radix/prefix tree.
    /// This will only contain root blocks.
    root: SharedBlock,

    tree_sizes: DashMap<WorkerWithDpRank, AtomicUsize, FxBuildHasher>,
}

impl Default for ConcurrentRadixTree {
    fn default() -> Self {
        Self::new()
    }
}

// Dropping blocks can cause a cascade of drops that can overflow the stack.
// This custom drop implementation avoids this using an iterative approach.
impl Drop for ConcurrentRadixTree {
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

impl ConcurrentRadixTree {
    /// Create a new `ConcurrentRadixTree`.
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(Block::new())),
            tree_sizes: DashMap::with_hasher(FxBuildHasher),
        }
    }

    /// Traverse the radix tree to find the best match for a given sequence of [`LocalBlockHash`]es.
    ///
    /// This operation is thread-safe and can run concurrently with other `find_matches` calls.
    /// Uses hand-over-hand read locking to minimize lock contention.
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

        // Get first child from root.
        let first_child = {
            let guard = self.root.read();
            guard.children.get(&sequence[0]).cloned()
        };

        let Some(first_child) = first_child else {
            return scores;
        };

        // Initialize active worker set from first child.
        let (mut active, mut active_count) = {
            let guard = first_child.read();
            (guard.workers.clone(), guard.workers.len())
        };

        if active.is_empty() {
            return scores;
        }

        if early_exit && active_count == 1 {
            for worker in &active {
                scores.scores.insert(*worker, 1);
            }
            for worker in scores.scores.keys() {
                if let Some(worker_tree_size) = self.tree_sizes.get(worker) {
                    scores
                        .tree_sizes
                        .insert(*worker, worker_tree_size.load(Ordering::Relaxed));
                }
            }
            return scores;
        }

        let mut current = first_child;
        let mut matched_depth = 1u32;

        // Traverse remaining levels. In a clean tree, workers at a child node
        // are always a subset of the parent (along the same path), so:
        //   - workers can only drop out, never join, as we descend
        //   - if child.workers.len() == active_count, the sets are identical
        //
        // However, because apply_removed does NOT cascade to descendants, a
        // child may transiently have MORE workers than its parent (stale
        // entries from an ancestor remove whose descendant remove events
        // haven't arrived yet). We detect this via child_count > active_count
        // and fall back to a full membership check.
        for (idx, local_hash) in sequence.iter().enumerate().skip(1) {
            let next_block = {
                let guard = current.read();
                guard.children.get(local_hash).cloned()
            };

            let Some(block) = next_block else {
                break;
            };

            {
                let guard = block.read();
                let child_count = guard.workers.len();

                if child_count != active_count {
                    // Workers changed: either dropped out (child < active) or
                    // stale entries exist (child > active). In both cases,
                    // retain only workers present in the child, scoring dropouts.
                    active.retain(|w| {
                        if guard.workers.contains(w) {
                            true
                        } else {
                            scores.scores.insert(*w, matched_depth);
                            false
                        }
                    });
                    active_count = active.len();

                    if active_count == 0 {
                        break;
                    }
                }
                // child_count == active_count: fast path, sets are identical
                // (or, in the rare edge case, different membership with same
                // cardinality -- accepted as a transient routing quality
                // degradation that resolves once pending remove events arrive).

                if early_exit && active_count == 1 {
                    matched_depth = (idx + 1) as u32;
                    break;
                }
            }

            current = block;
            matched_depth = (idx + 1) as u32;
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

    /// Apply a [`RouterEvent`] to the radix tree.
    ///
    /// This operation is thread-safe. Interior mutability via locks allows
    /// `&self` instead of `&mut self`.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    fn apply_event(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);

        // Construct WorkerWithDpRank from worker_id and dp_rank from the event
        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        match op {
            KvCacheEventData::Stored(op) => self.apply_stored(lookup, worker, op, id),
            KvCacheEventData::Removed(op) => self.apply_removed(lookup, worker, op, id),
            KvCacheEventData::Cleared => {
                // Ensure the worker is tracked in lookup before clearing,
                // matching RadixTree behavior where `lookup.entry(worker).or_default()`
                // fires before the match arm.
                lookup.entry(worker).or_default();
                self.tree_sizes
                    .entry(worker)
                    .or_insert_with(|| AtomicUsize::new(0));
                self.clear_all_blocks(lookup, worker.worker_id);
                Ok(())
            }
        }
    }

    /// Apply a store operation.
    fn apply_stored(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        // Ensure this worker has an entry in the outer map.
        let worker_lookup = lookup.entry(worker).or_default();

        // Find parent block
        let mut current = match op.parent_hash {
            Some(parent) => match worker_lookup.get(&parent) {
                Some(block) => block.clone(),
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
            },
            None => self.root.clone(),
        };

        let mut needs_worker_insert = false;

        let num_blocks_added = op.blocks.len();

        // In each iteration, we lock the parent block and insert the worker into it from
        // the previous iteration. This avoids locking a block twice.
        for block_data in op.blocks {
            let child = {
                let mut parent_guard = current.write();

                // Insert worker into this node if it was the child from the
                // previous iteration (skip for the initial parent, which is
                // not one of the blocks being stored).
                if needs_worker_insert {
                    parent_guard.workers.insert(worker);
                }
                needs_worker_insert = true;

                // parent_guard is dropped at the end of this block
                match parent_guard.children.get(&block_data.tokens_hash) {
                    Some(existing) => {
                        // Verify our simplifying assumption: block_hash is uniform across workers
                        {
                            let existing_guard = existing.read();
                            if existing_guard.block_hash != Some(block_data.block_hash) {
                                tracing::warn!(
                                    expected = ?block_data.block_hash,
                                    actual = ?existing_guard.block_hash,
                                    "block_hash mismatch: sequence hashes should be uniform across workers"
                                );
                            }
                        }
                        existing.clone()
                    }
                    None => {
                        // Reuse from lookup or create new
                        let new_block = worker_lookup
                            .get(&block_data.block_hash)
                            .cloned()
                            .unwrap_or_else(|| {
                                Arc::new(RwLock::new(Block::with_hash(block_data.block_hash)))
                            });

                        parent_guard
                            .children
                            .insert(block_data.tokens_hash, new_block.clone());
                        new_block
                    }
                }
            };

            // Update lookup
            worker_lookup.insert(block_data.block_hash, child.clone());

            current = child;
        }

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_add(num_blocks_added, Ordering::Relaxed);
            }
            None => {
                self.tree_sizes
                    .insert(worker, AtomicUsize::new(num_blocks_added));
            }
        }

        // Insert worker into the last child (not yet handled since there is
        // no subsequent iteration to pick it up).
        if needs_worker_insert {
            current.write().workers.insert(worker);
        }

        Ok(())
    }

    /// Apply a remove operation.
    ///
    /// This method does NOT cascade to descendants. Each block hash in the event
    /// is removed individually in O(1). Descendant blocks may transiently retain
    /// the worker in their `workers` set until their own explicit remove events
    /// arrive. `find_matches_impl` handles this by detecting stale entries when
    /// `child_count > active_count`.
    fn apply_removed(
        &self,
        lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        let Some(worker_lookup) = lookup.get_mut(&worker) else {
            return Err(KvCacheEventError::BlockNotFound);
        };

        let mut num_removed = 0;

        for block_hash in op.block_hashes {
            let Some(block) = worker_lookup.remove(&block_hash) else {
                tracing::debug!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found during remove; skipping"
                );
                continue;
            };

            // Remove the worker from this block's worker set.
            let mut guard = block.write();
            guard.workers.remove(&worker);
            if guard.workers.is_empty() {
                guard.children.clear();
            }

            num_removed += 1;
        }

        match self.tree_sizes.get(&worker) {
            Some(size) => {
                size.fetch_sub(num_removed, Ordering::Relaxed);
            }
            None => {
                self.tree_sizes
                    .insert(worker, AtomicUsize::new(num_removed));
            }
        }

        Ok(())
    }

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
                for (_, block) in worker_lookup.into_iter() {
                    let mut guard = block.write();
                    guard.workers.remove(&worker);
                    if guard.workers.is_empty() {
                        guard.children.clear();
                    }
                }

                if keep_worker {
                    lookup.insert(worker, FxHashMap::default());
                    // Reset tree size to 0 but keep the entry so get_workers()
                    // still returns this worker (matches RadixTree::clear_all_blocks behavior).
                    if let Some(size) = self.tree_sizes.get(&worker) {
                        size.store(0, Ordering::Relaxed);
                    }
                } else {
                    // Fully remove the worker from tree_sizes so get_workers()
                    // no longer returns it (matches RadixTree::remove_worker behavior).
                    self.tree_sizes.remove(&worker);
                }
            }
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

    /// Dump the radix tree as a series of RouterEvents that can reconstruct the tree.
    /// Uses BFS traversal over the shared tree. Since all worker/block membership is
    /// stored in the tree nodes themselves, this can be called from any thread without
    /// needing per-thread lookup state.
    fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!("Dumping concurrent radix tree as events");

        let mut events = Vec::new();
        let mut event_id = 0u64;

        // Queue entries: (current_block, parent_hash, tokens_hash)
        let mut queue = VecDeque::new();

        // Process root's children first
        {
            let root_guard = self.root.read();
            for (tokens_hash, child_block) in &root_guard.children {
                queue.push_back((child_block.clone(), None, *tokens_hash));
            }
        }

        while let Some((current_block, parent_hash, tokens_hash)) = queue.pop_front() {
            let current_guard = current_block.read();

            // Get this block's hash (same for all workers)
            let block_hash = current_guard
                .block_hash
                .expect("non-root block must have block_hash");

            // For each worker that has this block
            for worker in &current_guard.workers {
                // Create a store event for this worker
                let event = RouterEvent {
                    worker_id: worker.worker_id,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash,
                                mm_extra_info: None,
                                tokens_hash,
                            }],
                        }),
                        dp_rank: worker.dp_rank,
                    },
                };
                events.push(event);
                event_id += 1;
            }

            // Enqueue children with this block's hash as their parent
            for (child_tokens_hash, child_block) in &current_guard.children {
                queue.push_back((child_block.clone(), Some(block_hash), *child_tokens_hash));
            }
        }

        events
    }
}

// ============================================================================
// SyncIndexer implementation for ConcurrentRadixTree
// ============================================================================

impl SyncIndexer for ConcurrentRadixTree {
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

        tracing::debug!("ConcurrentRadixTree worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.find_matches_impl(sequence, early_exit)
    }

    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        Some(self.dump_tree_as_events())
    }
}
