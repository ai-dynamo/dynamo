// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concurrent Radix Tree implementation for KV cache routing.
//!
//! This module provides a thread-safe radix tree data structure that enables concurrent
//! `find_matches` operations while maintaining correctness for write operations.
//!
//! Unlike `RadixTree` which uses `Rc<RefCell<>>` and requires single-threaded access,
//! `ConcurrentRadixTree` uses `Arc<RwLock<>>` per node and a nested
//! `RwLock<HashMap<..., RwLock<HashMap<...>>>>` for the lookup table.
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
//! - The nested `RwLock` lookup allows per-worker write concurrency: writers
//!   targeting different workers only contend on the outer read lock.
//! - Deadlock prevention: always lock parent before child, hand-over-hand locking

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, RwLock},
};

use crate::indexer::SyncIndexer;
use crate::protocols::*;

/// Thread-safe shared reference to a Block.
type SharedBlock = Arc<RwLock<Block>>;

/// A block in the concurrent radix tree.
#[derive(Debug)]
struct Block {
    /// A map of child blocks, keyed by their local block hash.
    children: HashMap<LocalBlockHash, SharedBlock>,
    /// The set of workers that have this block cached.
    workers: HashSet<WorkerWithDpRank>,
    /// The external sequence block hash for this block (None for root).
    block_hash: Option<ExternalSequenceBlockHash>,
    // NOTE: No recent_uses field.
    // Frequency tracking is not supported - keeps find_matches fully read-only.
}

impl Block {
    /// Create a new `Block` (used for root node).
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            workers: HashSet::new(),
            block_hash: None,
        }
    }

    /// Create a new `Block` with a specific block hash.
    fn with_hash(block_hash: ExternalSequenceBlockHash) -> Self {
        Self {
            children: HashMap::new(),
            workers: HashSet::new(),
            block_hash: Some(block_hash),
        }
    }
}

/// Thread-safe radix tree for concurrent KV cache lookups.
///
/// Unlike `RadixTree` which uses `Rc<RefCell<>>` and requires single-threaded access,
/// `ConcurrentRadixTree` uses `Arc<RwLock<>>` per node and a nested
/// `RwLock<HashMap<..., RwLock<HashMap<...>>>>` for the lookup table,
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
/// - The nested `RwLock` lookup allows per-worker write concurrency: writers
///   targeting different workers only contend on the outer read lock.
/// - Deadlock prevention: always lock parent before child, hand-over-hand locking
pub struct ConcurrentRadixTree {
    /// This is the root of the radix/prefix tree.
    /// This will only contain root blocks.
    root: SharedBlock,

    /// Per-worker lookup table for O(1) block access.
    /// Outer `RwLock` protects the worker-level map; inner `RwLock` per worker
    /// protects that worker's block-hash map, so writers targeting different
    /// workers only contend on the outer read lock.
    lookup:
        RwLock<HashMap<WorkerWithDpRank, RwLock<HashMap<ExternalSequenceBlockHash, SharedBlock>>>>,
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
        if let Ok(mut root) = self.root.write() {
            stack.extend(root.children.drain().map(|(_, v)| v));
        }

        // Remove all lookup references (they may include blocks not reachable from root).
        // &mut self lets us bypass the outer RwLock; drain() gives us owned inner locks.
        let lookup = self.lookup.get_mut().unwrap();
        for (_, inner_lock) in lookup.drain() {
            if let Ok(inner) = inner_lock.into_inner() {
                stack.extend(inner.into_values());
            }
        }

        // Iteratively free any uniquely-owned blocks without recursion
        while let Some(block) = stack.pop() {
            if let Ok(rwlock) = Arc::try_unwrap(block)
                && let Ok(mut inner) = rwlock.into_inner()
            {
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
            lookup: RwLock::new(HashMap::new()),
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
            let guard = self.root.read().unwrap();
            guard.children.get(&sequence[0]).cloned()
        };

        let Some(first_child) = first_child else {
            return scores;
        };

        // Initialize active worker set from first child.
        let (mut active, mut active_count) = {
            let guard = first_child.read().unwrap();
            (guard.workers.clone(), guard.workers.len())
        };

        if active.is_empty() {
            return scores;
        }

        if early_exit && active_count == 1 {
            for worker in &active {
                scores.scores.insert(*worker, 1);
            }
            let lookup = self.lookup.read().unwrap();
            for worker in scores.scores.keys() {
                if let Some(inner_lock) = lookup.get(worker) {
                    scores
                        .tree_sizes
                        .insert(*worker, inner_lock.read().unwrap().len());
                }
            }
            return scores;
        }

        let mut current = first_child;
        let mut matched_depth = 1u32;

        // Traverse remaining levels. Workers at a child node are always a subset
        // of workers at the parent (along the same path), so:
        //   - workers can only drop out, never join, as we descend
        //   - if child.workers.len() == active_count, the sets are identical
        //
        // This lets us replace O(W) update_scores work at every level with an
        // O(1) length check. Per-worker iteration only happens at dropout
        // boundaries, where the count decreases.
        for (idx, item) in sequence.iter().enumerate().skip(1) {
            let next_block = {
                let guard = current.read().unwrap();
                guard.children.get(item).cloned()
            };

            let Some(block) = next_block else {
                break;
            };

            {
                let guard = block.read().unwrap();
                let child_count = guard.workers.len();

                if child_count < active_count {
                    // Workers dropped out. Record scores for those that left.
                    // Score = matched_depth (number of nodes they were present at).
                    for worker in &active {
                        if !guard.workers.contains(worker) {
                            scores.scores.insert(*worker, matched_depth);
                        }
                    }
                    active.clone_from(&guard.workers);
                    active_count = child_count;

                    if active_count == 0 {
                        break;
                    }
                }

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
        {
            let lookup = self.lookup.read().unwrap();
            for worker in scores.scores.keys() {
                if let Some(inner_lock) = lookup.get(worker) {
                    scores
                        .tree_sizes
                        .insert(*worker, inner_lock.read().unwrap().len());
                }
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
    pub fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);

        // Construct WorkerWithDpRank from worker_id and dp_rank from the event
        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        match op {
            KvCacheEventData::Stored(op) => self.apply_stored(worker, op, id),
            KvCacheEventData::Removed(op) => self.apply_removed(worker, op, id),
            KvCacheEventData::Cleared => {
                self.clear_all_blocks(worker.worker_id);
                Ok(())
            }
        }
    }

    /// Apply a store operation.
    fn apply_stored(
        &self,
        worker: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        // Ensure this worker has an entry in the outer map.
        // Double-checked locking: try read first, promote to write only if needed.
        {
            let lookup = self.lookup.read().unwrap();
            if !lookup.contains_key(&worker) {
                drop(lookup);
                self.lookup
                    .write()
                    .unwrap()
                    .entry(worker)
                    .or_insert_with(|| RwLock::new(HashMap::new()));
            }
        }

        // Hold the outer read lock for the remainder.  Other workers can still
        // be looked up / mutated concurrently because we only write-lock the
        // inner map for *this* worker.
        let lookup = self.lookup.read().unwrap();
        let inner_lock = lookup.get(&worker).unwrap();
        let mut worker_lookup = inner_lock.write().unwrap();

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

        // In each iteration, we lock the parent block and insert the worker into it from
        // the previous iteration. This avoids locking a block twice.
        for block_data in op.blocks {
            let child = {
                let mut parent_guard = current.write().unwrap();

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
                        if let Ok(existing_guard) = existing.read()
                            && existing_guard.block_hash != Some(block_data.block_hash)
                        {
                            tracing::warn!(
                                expected = ?block_data.block_hash,
                                actual = ?existing_guard.block_hash,
                                "block_hash mismatch: sequence hashes should be uniform across workers"
                            );
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

        // Insert worker into the last child (not yet handled since there is
        // no subsequent iteration to pick it up).
        if needs_worker_insert {
            current.write().unwrap().workers.insert(worker);
        }

        Ok(())
    }

    /// Apply a remove operation.
    fn apply_removed(
        &self,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        let lookup = self.lookup.read().unwrap();
        let Some(inner_lock) = lookup.get(&worker) else {
            return Err(KvCacheEventError::BlockNotFound);
        };
        let mut worker_lookup = inner_lock.write().unwrap();

        for block_hash in op.block_hashes {
            let Some(block) = worker_lookup.get(&block_hash).cloned() else {
                // Block not found - this is expected if:
                // 1. A parent block was removed earlier in this batch, cascade-removing this block
                // 2. Events arrived out of order
                // 3. Block was never stored for this worker
                // In all cases, we skip silently since the end state is correct.
                tracing::debug!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    id,
                    block_hash = ?block_hash,
                    "Block not found during remove; likely already cascade-removed"
                );
                continue;
            };

            // Collect descendant hashes before modifying anything.
            let descendants = self.collect_descendant_hashes(&block);

            // Write-lock block to remove worker
            {
                let mut guard = block.write().unwrap();
                guard.workers.remove(&worker);
                if guard.workers.is_empty() {
                    // Clear children from tree structure
                    guard.children.clear();
                }
            }

            // Remove the block from the worker's lookup table
            worker_lookup.remove(&block_hash);

            // Cascade: remove all descendants from this worker's lookup table.
            for descendant_hash in descendants {
                if let Some(desc_block) = worker_lookup.remove(&descendant_hash) {
                    let mut desc_guard = desc_block.write().unwrap();
                    desc_guard.workers.remove(&worker);
                    if desc_guard.workers.is_empty() {
                        desc_guard.children.clear();
                    }
                }
            }
        }

        Ok(())
    }

    /// Collect all descendant block hashes from a given block.
    /// Uses iterative DFS to avoid stack overflow on deep trees.
    fn collect_descendant_hashes(&self, block: &SharedBlock) -> Vec<ExternalSequenceBlockHash> {
        let mut result = Vec::new();
        let mut stack = vec![block.clone()];

        while let Some(current) = stack.pop() {
            let guard = current.read().unwrap();
            for child in guard.children.values() {
                if let Ok(child_guard) = child.read()
                    && let Some(hash) = child_guard.block_hash
                {
                    result.push(hash);
                    stack.push(child.clone());
                }
            }
        }
        result
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains in lookup with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed from lookup.
    fn remove_or_clear_worker_blocks(&self, worker_id: WorkerId, keep_worker: bool) {
        let mut lookup = self.lookup.write().unwrap();

        // Collect all WorkerWithDpRank keys that match this worker_id
        let workers: Vec<WorkerWithDpRank> = lookup
            .keys()
            .filter(|k| k.worker_id == worker_id)
            .cloned()
            .collect();

        for worker in workers {
            if let Some(inner_lock) = lookup.remove(&worker) {
                // We now own the inner RwLock; extract the HashMap.
                let blocks = inner_lock.into_inner().unwrap();
                for (_, block) in blocks {
                    let mut guard = block.write().unwrap();
                    guard.workers.remove(&worker);
                    // If no workers are using this block, that is true for all children
                    if guard.workers.is_empty() {
                        guard.children.clear();
                    }
                }

                if keep_worker {
                    // Re-insert worker with empty blocks map to keep it tracked
                    lookup.insert(worker, RwLock::new(HashMap::new()));
                }
            }
        }
    }

    /// Remove a worker and all their blocks from the tree.
    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, false);
    }

    /// Clear all blocks for a worker but keep the worker tracked.
    pub fn clear_all_blocks(&self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, true);
    }

    /// Get all worker IDs currently tracked in the radix tree.
    /// Returns unique worker_ids (ignoring dp_rank differences).
    pub fn get_workers(&self) -> Vec<WorkerId> {
        let lookup = self.lookup.read().unwrap();
        let mut worker_ids: Vec<WorkerId> = lookup
            .keys()
            .map(|k| k.worker_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        worker_ids.sort_unstable();
        worker_ids
    }

    /// Dump the radix tree as a series of RouterEvents that can reconstruct the tree.
    /// Uses BFS traversal to ensure that the tree reconstruction is unique,
    /// though the exact event ordering will be lost.
    pub fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!(
            "Dumping concurrent radix tree as events (contains information about {:?} workers)",
            self.lookup.read().unwrap().len()
        );

        let mut events = Vec::new();
        let mut event_id = 0u64;

        // Queue entries: (current_block, parent_hash, tokens_hash)
        let mut queue = VecDeque::new();

        // Process root's children first
        {
            let root_guard = self.root.read().unwrap();
            for (tokens_hash, child_block) in &root_guard.children {
                queue.push_back((child_block.clone(), None, *tokens_hash));
            }
        }

        while let Some((current_block, parent_hash, tokens_hash)) = queue.pop_front() {
            let current_guard = current_block.read().unwrap();

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

    /// Get total number of blocks across all workers.
    pub fn current_size(&self) -> usize {
        let lookup = self.lookup.read().unwrap();
        lookup
            .values()
            .map(|inner_lock| inner_lock.read().unwrap().len())
            .sum()
    }
}

// ============================================================================
// SyncIndexer implementation for ConcurrentRadixTree
// ============================================================================

impl SyncIndexer for ConcurrentRadixTree {
    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        // Delegate to the existing find_matches method
        self.find_matches_impl(sequence, early_exit)
    }

    fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        self.apply_event(event)
    }

    fn remove_worker(&self, worker_id: WorkerId) {
        self.remove_worker(worker_id);
    }

    fn dump_events(&self) -> Vec<RouterEvent> {
        self.dump_tree_as_events()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{create_remove_event, create_store_event};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_concurrent_radix_tree_basic() {
        let trie = ConcurrentRadixTree::new();

        let worker_1 = 0;
        let worker_2 = 1;

        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();

        let scores = trie.find_matches_impl(
            &[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
            false,
        );
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap(),
            &3
        );

        assert_eq!(trie.lookup.read().unwrap().len(), 1);
        assert_eq!(
            trie.lookup
                .read()
                .unwrap()
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .read()
                .unwrap()
                .len(),
            3
        );

        trie.apply_event(create_store_event(worker_2, 1, vec![1, 4, 5], None))
            .unwrap();

        let scores = trie.find_matches_impl(
            &[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
            false,
        );
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap(),
            &3
        );
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap(),
            &1
        );

        assert_eq!(trie.lookup.read().unwrap().len(), 2);
    }

    #[test]
    fn test_concurrent_radix_tree_remove() {
        let trie = ConcurrentRadixTree::new();

        let worker_1 = 0;
        let worker_2 = 1;

        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_2, 1, vec![1, 4, 5], None))
            .unwrap();

        trie.apply_event(create_remove_event(worker_2, 2, vec![5]))
            .unwrap();

        assert_eq!(
            trie.lookup
                .read()
                .unwrap()
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .read()
                .unwrap()
                .len(),
            2
        );

        trie.apply_event(create_remove_event(worker_2, 3, vec![4]))
            .unwrap();

        assert_eq!(
            trie.lookup
                .read()
                .unwrap()
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .read()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_concurrent_radix_tree_apply_event_errors() {
        let trie = ConcurrentRadixTree::new();
        let worker_0 = 0;

        // Parent block not found
        let result = trie.apply_event(create_store_event(
            worker_0,
            0,
            vec![1, 2, 3],
            Some(ExternalSequenceBlockHash(12345)),
        ));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KvCacheEventError::ParentBlockNotFound
        ));
    }

    #[test]
    fn test_clear_all_blocks() {
        let trie = ConcurrentRadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        trie.apply_event(create_store_event(worker_0, 0, vec![0, 1, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![0, 2, 3], None))
            .unwrap();

        let result = trie.find_matches_impl(&[LocalBlockHash(0)], false).scores;
        assert_eq!(result.len(), 2);

        trie.clear_all_blocks(worker_0);

        assert!(
            trie.lookup
                .read()
                .unwrap()
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            trie.lookup
                .read()
                .unwrap()
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .unwrap()
                .read()
                .unwrap()
                .is_empty()
        );

        let result = trie
            .find_matches_impl(&[LocalBlockHash(0), LocalBlockHash(2)], false)
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_1)], 2);
    }

    #[test]
    fn test_remove_worker() {
        let trie = ConcurrentRadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        trie.apply_event(create_store_event(worker_0, 0, vec![1, 2, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![1, 2, 3], None))
            .unwrap();

        assert_eq!(trie.lookup.read().unwrap().len(), 2);

        trie.remove_worker(worker_0);

        assert!(
            !trie
                .lookup
                .read()
                .unwrap()
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert_eq!(trie.lookup.read().unwrap().len(), 1);

        let result = trie
            .find_matches_impl(
                &[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
                false,
            )
            .scores;
        assert_eq!(result.len(), 1);
        assert!(!result.contains_key(&WorkerWithDpRank::from_worker_id(worker_0)));
        assert!(result.contains_key(&WorkerWithDpRank::from_worker_id(worker_1)));
    }

    #[test]
    fn test_concurrent_radix_tree_default() {
        let trie: ConcurrentRadixTree = Default::default();
        assert!(trie.root.read().unwrap().children.is_empty());
        assert!(trie.root.read().unwrap().workers.is_empty());
        assert!(trie.lookup.read().unwrap().is_empty());
    }

    #[test]
    fn test_concurrent_find_matches() {
        let trie = Arc::new(ConcurrentRadixTree::new());

        // Populate tree
        trie.apply_event(create_store_event(0, 0, vec![1, 2, 3, 4, 5], None))
            .unwrap();
        trie.apply_event(create_store_event(1, 0, vec![1, 2, 6, 7, 8], None))
            .unwrap();

        let sequence = vec![
            LocalBlockHash(1),
            LocalBlockHash(2),
            LocalBlockHash(3),
            LocalBlockHash(4),
            LocalBlockHash(5),
        ];

        // Spawn multiple threads doing concurrent find_matches
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let tree = trie.clone();
                let seq = sequence.clone();
                thread::spawn(move || tree.find_matches_impl(&seq, false))
            })
            .collect();

        // All should return the same result
        let expected_worker_0_score = 5;
        let expected_worker_1_score = 2;

        for h in handles {
            let result = h.join().unwrap();
            assert_eq!(
                result
                    .scores
                    .get(&WorkerWithDpRank::from_worker_id(0))
                    .unwrap(),
                &expected_worker_0_score
            );
            assert_eq!(
                result
                    .scores
                    .get(&WorkerWithDpRank::from_worker_id(1))
                    .unwrap(),
                &expected_worker_1_score
            );
        }
    }

    #[test]
    fn test_concurrent_read_write() {
        let trie = Arc::new(ConcurrentRadixTree::new());

        // Pre-populate
        for i in 0..5 {
            trie.apply_event(create_store_event(i, 0, vec![1, 2, 3], None))
                .unwrap();
        }

        let sequence = vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)];

        // Spawn readers
        let reader_handles: Vec<_> = (0..5)
            .map(|_| {
                let tree = trie.clone();
                let seq = sequence.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = tree.find_matches_impl(&seq, false);
                    }
                })
            })
            .collect();

        // Spawn writers (adding more workers)
        let writer_handles: Vec<_> = (5..10)
            .map(|i| {
                let tree = trie.clone();
                thread::spawn(move || {
                    for j in 0..10 {
                        let _ =
                            tree.apply_event(create_store_event(i, j, vec![1, 2, 3, 4 + j], None));
                    }
                })
            })
            .collect();

        // Wait for all threads
        for h in reader_handles {
            h.join().unwrap();
        }
        for h in writer_handles {
            h.join().unwrap();
        }

        // Tree should have 10 workers now
        assert_eq!(trie.get_workers().len(), 10);
    }

    #[test]
    fn test_remove_parent_cascades_to_children() {
        let trie = ConcurrentRadixTree::new();
        let worker_1 = 0;

        // Create a chain: root -> block1 -> block2 -> block3
        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();

        let worker_key = WorkerWithDpRank::from_worker_id(worker_1);
        {
            let lookup = trie.lookup.read().unwrap();
            assert_eq!(lookup.get(&worker_key).unwrap().read().unwrap().len(), 3);
        }

        // Remove ONLY block1 - children should be cascade-removed
        trie.apply_event(create_remove_event(worker_1, 2, vec![1]))
            .unwrap();

        // All blocks should be removed (cascade cleanup)
        let lookup = trie.lookup.read().unwrap();
        let worker_lookup = lookup.get(&worker_key).unwrap().read().unwrap();
        assert!(!worker_lookup.contains_key(&ExternalSequenceBlockHash(100)));
        assert!(
            !worker_lookup.contains_key(&ExternalSequenceBlockHash(200)),
            "block2 should be cascade-removed with parent"
        );
        assert!(
            !worker_lookup.contains_key(&ExternalSequenceBlockHash(300)),
            "block3 should be cascade-removed with parent"
        );
        assert_eq!(worker_lookup.len(), 0);
    }

    // ========================================================================
    // ThreadPoolIndexer<ConcurrentRadixTree> Tests
    // ========================================================================

    mod thread_pool_indexer_tests {
        use tokio::time::Duration;

        use super::*;
        use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};

        fn make_indexer(
            num_workers: usize,
            kv_block_size: u32,
        ) -> ThreadPoolIndexer<ConcurrentRadixTree> {
            ThreadPoolIndexer::new(ConcurrentRadixTree::new(), num_workers, kv_block_size)
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_basic() {
            let indexer = make_indexer(4, 16);

            let worker_1 = 0;
            let worker_2 = 1;

            indexer
                .apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
                .await;
            indexer
                .apply_event(create_store_event(worker_2, 1, vec![1, 4, 5], None))
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            let scores = indexer
                .find_matches(vec![
                    LocalBlockHash(1),
                    LocalBlockHash(2),
                    LocalBlockHash(3),
                ])
                .await
                .unwrap();

            assert_eq!(
                scores
                    .scores
                    .get(&WorkerWithDpRank::from_worker_id(worker_1))
                    .unwrap(),
                &3
            );
            assert_eq!(
                scores
                    .scores
                    .get(&WorkerWithDpRank::from_worker_id(worker_2))
                    .unwrap(),
                &1
            );

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_remove_worker() {
            let indexer = make_indexer(2, 16);

            let worker_0 = 0;
            let worker_1 = 1;

            indexer
                .apply_event(create_store_event(worker_0, 1, vec![1, 2, 3], None))
                .await;
            indexer
                .apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            assert_eq!(indexer.backend().get_workers().len(), 2);

            indexer.remove_worker(worker_0).await;

            let workers = indexer.backend().get_workers();
            assert_eq!(workers.len(), 1);
            assert!(!workers.contains(&worker_0));
            assert!(workers.contains(&worker_1));

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_dump_events() {
            let indexer = make_indexer(2, 16);

            indexer
                .apply_event(create_store_event(0, 1, vec![1, 2, 3], None))
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            let events = indexer.dump_events().await.unwrap();
            assert_eq!(events.len(), 3);

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_find_matches_for_request() {
            let indexer = make_indexer(2, 1);

            indexer
                .apply_event(create_store_event(0, 1, vec![100, 200, 300], None))
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            let scores = indexer.find_matches_for_request(&[100, 200, 300]).await;
            assert!(scores.is_ok());

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_sticky_routing() {
            let indexer = make_indexer(4, 16);

            for i in 0..10 {
                indexer
                    .apply_event(create_store_event(0, i, vec![i as u64], None))
                    .await;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;

            assert_eq!(indexer.backend().current_size(), 10);

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_multiple_workers() {
            let indexer = make_indexer(4, 16);

            for worker_id in 0..8 {
                indexer
                    .apply_event(create_store_event(
                        worker_id,
                        1,
                        vec![1, 2, worker_id as u64 + 10],
                        None,
                    ))
                    .await;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;

            assert_eq!(indexer.backend().get_workers().len(), 8);

            let scores = indexer
                .find_matches(vec![LocalBlockHash(1), LocalBlockHash(2)])
                .await
                .unwrap();

            assert_eq!(scores.scores.len(), 8);
            for (_, score) in scores.scores.iter() {
                assert_eq!(*score, 2);
            }

            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_shutdown_idempotent() {
            let indexer = make_indexer(2, 16);

            indexer
                .apply_event(create_store_event(0, 1, vec![1, 2, 3], None))
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            indexer.shutdown();
            indexer.shutdown();
        }

        #[tokio::test]
        async fn test_thread_pool_indexer_concurrent_operations() {
            use std::sync::Arc;

            let indexer = Arc::new(make_indexer(4, 16));

            for worker_id in 0..4 {
                indexer
                    .apply_event(create_store_event(worker_id, 1, vec![1, 2, 3, 4, 5], None))
                    .await;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;

            let sequence = vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)];

            let mut handles = Vec::new();
            for _ in 0..10 {
                let idx = indexer.clone();
                let seq = sequence.clone();
                handles.push(tokio::spawn(
                    async move { idx.find_matches(seq).await.unwrap() },
                ));
            }

            for handle in handles {
                let scores = handle.await.unwrap();
                assert_eq!(scores.scores.len(), 4);
            }

            indexer.shutdown();
        }
    }
}
