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

use std::{collections::VecDeque, sync::Arc};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::indexer::SyncIndexer;
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

    /// Per-worker lookup table for O(1) block access.
    /// Outer RwLock protects the worker map structure (rarely mutated);
    /// inner RwLock per worker protects that worker's block-hash map.
    lookup: RwLock<FxHashMap<WorkerWithDpRank, RwLock<WorkerLookup>>>,
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

        // Remove all lookup references (they may include blocks not reachable from root).
        // We have &mut self so no concurrent access; drain the map.
        let lookup = self.lookup.get_mut();
        for (_, inner_lock) in lookup.drain() {
            stack.extend(inner_lock.into_inner().into_values());
        }

        // Iteratively free any uniquely-owned blocks without recursion
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
            lookup: RwLock::new(FxHashMap::default()),
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
            let lk = self.lookup.read();
            for worker in scores.scores.keys() {
                if let Some(inner_lock) = lk.get(worker) {
                    scores.tree_sizes.insert(*worker, inner_lock.read().len());
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
        let lk = self.lookup.read();
        for worker in scores.scores.keys() {
            if let Some(inner_lock) = lk.get(worker) {
                scores.tree_sizes.insert(*worker, inner_lock.read().len());
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
        if !self.lookup.read().contains_key(&worker) {
            self.lookup
                .write()
                .entry(worker)
                .or_insert_with(|| RwLock::new(FxHashMap::default()));
        }

        let lk = self.lookup.read();
        let mut worker_lookup = lk.get(&worker).unwrap().write();

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
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        let lk = self.lookup.read();
        let Some(inner_ref) = lk.get(&worker) else {
            return Err(KvCacheEventError::BlockNotFound);
        };
        let mut worker_lookup = inner_ref.write();

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
        }

        Ok(())
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains in lookup with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed from lookup.
    fn remove_or_clear_worker_blocks(&self, worker_id: WorkerId, keep_worker: bool) {
        let workers: Vec<WorkerWithDpRank> = self
            .lookup
            .read()
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        let mut lk = self.lookup.write();
        for worker in workers {
            if let Some(inner_lock) = lk.remove(&worker) {
                let blocks = inner_lock.into_inner();
                for (_, block) in blocks {
                    let mut guard = block.write();
                    guard.workers.remove(&worker);
                    if guard.workers.is_empty() {
                        guard.children.clear();
                    }
                }

                if keep_worker {
                    lk.insert(worker, RwLock::new(FxHashMap::default()));
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
        let mut worker_ids: Vec<WorkerId> = self
            .lookup
            .read()
            .keys()
            .map(|w| w.worker_id)
            .collect::<FxHashSet<_>>()
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
            self.lookup.read().len()
        );

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

    /// Get total number of blocks across all workers.
    pub fn current_size(&self) -> usize {
        self.lookup
            .read()
            .values()
            .map(|inner| inner.read().len())
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

        assert_eq!(trie.lookup.read().len(), 1);
        assert_eq!(
            trie.lookup
                .read()
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .read()
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

        assert_eq!(trie.lookup.read().len(), 2);
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
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .read()
                .len(),
            2
        );

        trie.apply_event(create_remove_event(worker_2, 3, vec![4]))
            .unwrap();

        assert_eq!(
            trie.lookup
                .read()
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .read()
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
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            trie.lookup
                .read()
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .unwrap()
                .read()
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

        assert_eq!(trie.lookup.read().len(), 2);

        trie.remove_worker(worker_0);

        assert!(
            !trie
                .lookup
                .read()
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert_eq!(trie.lookup.read().len(), 1);

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
        assert!(trie.root.read().children.is_empty());
        assert!(trie.root.read().workers.is_empty());
        assert!(trie.lookup.read().is_empty());
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
    fn test_remove_parent_does_not_cascade() {
        let trie = ConcurrentRadixTree::new();
        let worker_1 = 0;

        // Create a chain: root -> block1 -> block2 -> block3
        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();

        let worker_key = WorkerWithDpRank::from_worker_id(worker_1);
        assert_eq!(trie.lookup.read().get(&worker_key).unwrap().read().len(), 3);

        // Remove ONLY block1 -- descendants should NOT be cascade-removed
        trie.apply_event(create_remove_event(worker_1, 2, vec![1]))
            .unwrap();

        let lk = trie.lookup.read();
        let worker_lookup = lk.get(&worker_key).unwrap().read();
        assert!(
            !worker_lookup.contains_key(&ExternalSequenceBlockHash(100)),
            "block1 should be removed"
        );
        assert!(
            worker_lookup.contains_key(&ExternalSequenceBlockHash(200)),
            "block2 should remain (no cascade)"
        );
        assert!(
            worker_lookup.contains_key(&ExternalSequenceBlockHash(300)),
            "block3 should remain (no cascade)"
        );
        assert_eq!(worker_lookup.len(), 2);
    }

    #[test]
    fn test_remove_all_blocks_individually() {
        // Verifies that explicitly removing all blocks (as the engine would)
        // cleans up fully, even without cascade.
        let trie = ConcurrentRadixTree::new();
        let worker_1 = 0;

        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();

        let worker_key = WorkerWithDpRank::from_worker_id(worker_1);

        // Remove all three blocks explicitly in one event
        trie.apply_event(create_remove_event(worker_1, 2, vec![1, 2, 3]))
            .unwrap();

        let lk = trie.lookup.read();
        let worker_lookup = lk.get(&worker_key).unwrap().read();
        assert_eq!(worker_lookup.len(), 0, "all blocks should be removed");
    }

    #[test]
    fn test_find_matches_with_stale_entries() {
        // Two workers share a full path. Remove worker_1 from the root block
        // only (simulating a partial remove). find_matches should still
        // produce correct scores for worker_2, and worker_1 should score at
        // the stale descendant depth (transiently inflated but not a crash).
        let trie = ConcurrentRadixTree::new();
        let worker_1 = 0;
        let worker_2 = 1;

        // Both workers have blocks 1 -> 2 -> 3
        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_2, 2, vec![1, 2, 3], None))
            .unwrap();

        // Remove worker_1 from block 1 only (no cascade to 2,3)
        trie.apply_event(create_remove_event(worker_1, 3, vec![1]))
            .unwrap();

        let scores = trie.find_matches_impl(
            &[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
            false,
        );

        // worker_2 was never removed, should have full depth
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_2)),
            Some(&3),
            "worker_2 should score 3 (fully present)"
        );

        // worker_1 was removed from block 1 so it drops out at depth 1.
        // But because blocks 2 and 3 still have worker_1 (stale), the
        // child_count > active_count path fires and detects the dropout.
        // The exact score depends on the detection logic: worker_1 is absent
        // from block 1's workers, so it should be scored at depth 0 from the
        // first child initialization (it won't appear in `active` at all).
        // So worker_1 should NOT appear in scores (it was never in active).
        assert!(
            !scores
                .scores
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_1)),
            "worker_1 should not appear in scores (removed from root-level block)"
        );
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
                    .apply_event(create_store_event(0, i, vec![i], None))
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
                        vec![1, 2, worker_id + 10],
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
