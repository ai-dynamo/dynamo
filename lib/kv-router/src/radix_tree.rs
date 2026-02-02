// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Radix Tree implementation for KV cache routing.
//!
//! This module provides a radix tree (prefix tree) data structure optimized for
//! efficient KV cache block lookup and management in distributed LLM inference.
//!
//! # Overview
//!
//! The main components include:
//!
//! - **RadixTree**: The main data structure with nodes (`RadixBlock`) containing
//!   children and associated worker IDs. Allows efficient storage and retrieval
//!   of data blocks based on their hashes.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::protocols::*;

/// A shared reference to a [`RadixBlock`].
pub(crate) type SharedRadixBlock = Rc<RefCell<RadixBlock>>;

/// A block in the Radix Tree.
#[derive(Debug)]
pub(crate) struct RadixBlock {
    /// A map of child blocks, keyed by their local block hash.
    pub(crate) children: HashMap<LocalBlockHash, SharedRadixBlock>,
    /// The set of workers that have this block cached.
    pub(crate) workers: HashSet<WorkerWithDpRank>,
    /// The external sequence block hash for this block (None for root).
    /// This is the same for all workers under the simplifying assumption.
    pub(crate) block_hash: Option<ExternalSequenceBlockHash>,
    /// A buffer of times that this block was last traversed
    pub(crate) recent_uses: VecDeque<Instant>,
}

impl RadixBlock {
    /// Create a new `RadixBlock` (used for root node).
    ///
    /// ### Returns
    ///
    /// A new `RadixBlock` with no block_hash.
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            workers: HashSet::new(),
            block_hash: None,
            recent_uses: VecDeque::new(),
        }
    }

    /// Create a new `RadixBlock` with a specific block hash.
    ///
    /// ### Returns
    ///
    /// A new `RadixBlock` with the given block_hash.
    pub fn with_hash(block_hash: ExternalSequenceBlockHash) -> Self {
        Self {
            children: HashMap::new(),
            workers: HashSet::new(),
            block_hash: Some(block_hash),
            recent_uses: VecDeque::new(),
        }
    }
}

pub struct RadixTree {
    /// This is the root of the radix/prefix tree
    /// This will only contain root blocks
    pub(crate) root: SharedRadixBlock,

    /// Per-worker lookup table for O(1) block access.
    /// Maps worker -> (block_hash -> block).
    pub(crate) lookup:
        HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, SharedRadixBlock>>,

    /// The time buffer the radix tree should check when considering frequence of block accesses
    pub(crate) expiration_duration: Option<Duration>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

// Dropping Radix blocks can cause a cascade of drops that can overflow the stack.
// This custom drop implementation avoids this using an iterative approach.
impl Drop for RadixTree {
    fn drop(&mut self) {
        let mut stack: Vec<SharedRadixBlock> = Vec::new();
        // Break root -> children edge up front
        {
            let mut root = self.root.borrow_mut();
            stack.extend(root.children.drain().map(|(_, v)| v));
        }

        // Remove all lookup references (they may include blocks not reachable from root)
        for (_, worker_blocks) in self.lookup.drain() {
            stack.extend(worker_blocks.into_values());
        }

        // Iteratively free any uniquely-owned blocks without recursion
        while let Some(block) = stack.pop() {
            match Rc::try_unwrap(block) {
                Ok(cell) => {
                    // We own the cell, so we can take inner and it will drop after this block.
                    let mut inner: RadixBlock = cell.into_inner();
                    stack.extend(inner.children.drain().map(|(_, v)| v));
                }
                Err(rc) => {
                    // We don't own the cell, just call drop on it.
                    drop(rc);
                }
            }
        }
    }
}

impl RadixTree {
    /// Create a new `RadixTree`.
    ///
    /// ### Returns
    ///
    /// A new `RadixTree`.
    pub fn new_with_frequency(expiration_duration: Option<Duration>) -> Self {
        Self {
            root: Rc::new(RefCell::new(RadixBlock::new())),
            lookup: HashMap::new(),
            expiration_duration,
        }
    }

    pub fn new() -> Self {
        Self::new_with_frequency(None)
    }

    /// Traverse the radix tree to find the best match for a given sequence of [`LocalBlockHash`]es.
    ///
    /// ### Arguments
    ///
    /// * `sequence` - A vector of `LocalBlockHash` representing the sequence to match.
    /// * `early_exit` - A boolean indicating whether to exit early if a single match is found.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    pub fn find_matches(&self, sequence: Vec<LocalBlockHash>, early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();
        let mut current = self.root.clone();
        let now = Instant::now();

        tracing::trace!(
            "RadixTree::find_matches: looking for sequence={:?}",
            sequence.iter().map(|h| h.0).collect::<Vec<_>>()
        );

        for (idx, block_hash) in sequence.iter().enumerate() {
            let next_block = {
                let current_borrow = current.borrow();
                current_borrow.children.get(block_hash).cloned()
            };
            if let Some(block) = next_block {
                scores.update_scores(block.borrow().workers.iter());

                if let Some(expiration_duration) = self.expiration_duration {
                    let mut block_mut = block.borrow_mut();

                    while let Some(access_time) = block_mut.recent_uses.front() {
                        if now.duration_since(*access_time) > expiration_duration {
                            block_mut.recent_uses.pop_front();
                        } else {
                            break;
                        }
                    }
                    scores.add_frequency(block_mut.recent_uses.len());
                    block_mut.recent_uses.push_back(now);
                }

                if early_exit && block.borrow().workers.len() == 1 {
                    break;
                }

                current = block;
            } else {
                tracing::trace!(
                    "RadixTree::find_matches: block not found at index {} for hash {}",
                    idx,
                    block_hash.0
                );
                break;
            }
        }

        tracing::trace!("RadixTree::find_matches: final scores={:?}", scores.scores);

        // Populate tree sizes for all workers that have scores
        for worker in scores.scores.keys() {
            let tree_size = self
                .lookup
                .get(worker)
                .expect("worker in scores must exist in lookup table")
                .len();
            scores.tree_sizes.insert(*worker, tree_size);
        }

        scores
    }

    /// Apply a [`RouterEvent`] to the radix tree.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    pub fn apply_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);

        // Construct WorkerWithDpRank from worker_id and dp_rank from the event
        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        tracing::trace!(id, "RadixTree::apply_event: Store operation: {:?}", op);

        let worker_lookup = self.lookup.entry(worker).or_default();

        match op {
            KvCacheEventData::Stored(op) => {
                // find the parent block from this worker's lookup
                let mut current = match op.parent_hash {
                    Some(parent) => match worker_lookup.get(&parent) {
                        Some(current) => current.clone(),
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

                for block_data in op.blocks {
                    let mut parent_mut = current.borrow_mut();
                    let child = match parent_mut.children.get(&block_data.tokens_hash) {
                        Some(block) => {
                            // Verify our simplifying assumption: block_hash is uniform across workers
                            if block.borrow().block_hash != Some(block_data.block_hash) {
                                tracing::warn!(
                                    expected = ?block_data.block_hash,
                                    actual = ?block.borrow().block_hash,
                                    "block_hash mismatch: sequence hashes should be uniform across workers"
                                );
                            }
                            block.clone()
                        }
                        None => {
                            // create new block or reuse existing from worker's lookup
                            let new_block = worker_lookup
                                .get(&block_data.block_hash)
                                .cloned()
                                .unwrap_or_else(|| {
                                    Rc::new(RefCell::new(RadixBlock::with_hash(
                                        block_data.block_hash,
                                    )))
                                });

                            // insert into radix tree
                            parent_mut
                                .children
                                .insert(block_data.tokens_hash, new_block.clone());

                            new_block
                        }
                    };

                    // Update child and check for self referential blocks
                    {
                        // Try to borrow the child mutably - if it fails, it's already borrowed
                        // which means a self referencing block.
                        let mut child_mut = match child.try_borrow_mut() {
                            Ok(b) => b,
                            Err(_) => {
                                tracing::warn!(
                                    worker_id = worker.worker_id.to_string(),
                                    dp_rank = worker.dp_rank,
                                    id,
                                    block_hash = ?block_data.block_hash,
                                    "Detected self referencing block in store event; rejecting sequence"
                                );
                                return Err(KvCacheEventError::InvalidBlockSequence);
                            }
                        };

                        // add our worker to the block
                        child_mut.workers.insert(worker);
                    }

                    // add the block to the worker's lookup table
                    worker_lookup.insert(block_data.block_hash, child.clone());

                    // drop child so we can shift current to this block
                    drop(parent_mut);

                    current = child;
                }
                Ok(())
            }
            KvCacheEventData::Removed(remove) => {
                let mut kv_cache_err: Option<KvCacheEventError> = None;
                for block in remove.block_hashes {
                    // lookup block in worker's table
                    let entry = match worker_lookup.get(&block) {
                        Some(entry) => entry.clone(),
                        None => {
                            tracing::warn!(
                                worker_id = worker.worker_id.to_string(),
                                dp_rank = worker.dp_rank,
                                id,
                                block_hash = ?block,
                                "Failed to find block to remove; skipping remove operation"
                            );
                            // Kv cache removed events may be batched; we should try to apply all
                            // operations in the batch before returning an error. Return the first
                            // error.
                            if kv_cache_err.is_none() {
                                kv_cache_err = Some(KvCacheEventError::BlockNotFound);
                            }
                            continue;
                        }
                    };

                    let mut guard = entry.borrow_mut();
                    guard.workers.remove(&worker);
                    if guard.workers.is_empty() {
                        // if no workers are using this block, that is true for all children
                        guard.children.clear();
                    }
                    // remove the block from the worker's lookup table
                    worker_lookup.remove(&block);
                }
                kv_cache_err.map_or(Ok(()), Err)
            }
            KvCacheEventData::Cleared => {
                self.clear_all_blocks(worker.worker_id);
                Ok(())
            }
        }
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains in lookup with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed from lookup.
    fn remove_or_clear_worker_blocks(&mut self, worker_id: WorkerId, keep_worker: bool) {
        // Collect all WorkerWithDpRank keys that match this worker_id
        let workers: Vec<WorkerWithDpRank> = self
            .lookup
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        for worker in workers {
            if let Some((worker_key, blocks)) = self.lookup.remove_entry(&worker) {
                for (_, block) in blocks {
                    block.borrow_mut().workers.remove(&worker);
                    // If no workers are using this block, that is true for all children
                    if block.borrow().workers.is_empty() {
                        block.borrow_mut().children.clear();
                    }
                }

                if keep_worker {
                    // Re-insert worker with empty blocks map to keep it tracked
                    self.lookup.insert(worker_key, HashMap::new());
                }
            }
        }
    }

    pub fn remove_worker(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, false);
    }

    pub fn clear_all_blocks(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, true);
    }

    /// Get all worker IDs currently tracked in the radix tree.
    /// Returns unique worker_ids (ignoring dp_rank differences).
    pub fn get_workers(&self) -> Vec<WorkerId> {
        let mut worker_ids: Vec<WorkerId> = self.lookup.keys().map(|w| w.worker_id).collect();
        worker_ids.sort_unstable();
        worker_ids.dedup();
        worker_ids
    }

    /// Dump the radix tree as a series of RouterEvents that can reconstruct the tree.
    /// Uses BFS traversal to ensure that the tree reconstruction is unique,
    /// though the exact event ordering will be lost.
    pub fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!(
            "Dumping radix tree as events (contains information about {:?} workers)",
            self.lookup.len()
        );

        let mut events = Vec::new();
        let mut event_id = 0u64;

        // Queue entries: (current_block, parent_hash, tokens_hash)
        let mut queue = VecDeque::new();

        // Process root's children first
        let root_borrow = self.root.borrow();
        for (tokens_hash, child_block) in &root_borrow.children {
            queue.push_back((child_block.clone(), None, *tokens_hash));
        }
        drop(root_borrow);

        while let Some((current_block, parent_hash, tokens_hash)) = queue.pop_front() {
            let current_borrow = current_block.borrow();

            // Get this block's hash (same for all workers)
            let block_hash = current_borrow
                .block_hash
                .expect("non-root block must have block_hash");

            // For each worker that has this block
            for worker in &current_borrow.workers {
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
            for (child_tokens_hash, child_block) in &current_borrow.children {
                queue.push_back((child_block.clone(), Some(block_hash), *child_tokens_hash));
            }
        }

        events
    }

    pub fn current_size(&self) -> usize {
        self.lookup.values().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, WorkerId,
    };

    /// Creates blocks with artificial hash mapping (hash * 100) for testing RadixTree internals.
    fn make_blocks(hashes: Vec<u64>) -> Vec<KvCacheStoredBlockData> {
        hashes
            .iter()
            .map(|i| KvCacheStoredBlockData {
                tokens_hash: LocalBlockHash(*i),
                block_hash: ExternalSequenceBlockHash(*i * 100),
                mm_extra_info: None,
            })
            .collect()
    }

    fn add_blocks(
        hashes: Vec<u64>,
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) -> KvCacheEventData {
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks: make_blocks(hashes),
        })
    }

    fn create_store_event(
        worker_id: WorkerId,
        event_id: u64,
        hashes: Vec<u64>,
        parent: Option<ExternalSequenceBlockHash>,
    ) -> RouterEvent {
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: add_blocks(hashes, parent),
                dp_rank: 0,
            },
        }
    }

    fn create_remove_event(worker_id: WorkerId, event_id: u64, hashes: Vec<u64>) -> RouterEvent {
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .iter()
                        .map(|i| ExternalSequenceBlockHash(*i * 100))
                        .collect(),
                }),
                dp_rank: 0,
            },
        }
    }

    #[test]
    fn test_radix_tree() {
        let mut trie = RadixTree::new();

        let worker_1 = 0;
        let worker_2 = 1;

        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .unwrap();

        let scores = trie.find_matches(
            vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
            false,
        );
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap(),
            &3
        );

        assert_eq!(trie.lookup.len(), 1);
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(trie.root.borrow().workers.len(), 0);
        assert_eq!(trie.root.borrow().children.len(), 1);
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .workers
                .len(),
            1
        );
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .children
                .len(),
            1
        );

        trie.apply_event(create_store_event(worker_2, 1, vec![1, 4, 5], None))
            .unwrap();

        let scores = trie.find_matches(
            vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
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

        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(trie.root.borrow().workers.len(), 0);
        assert_eq!(trie.root.borrow().children.len(), 1);
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .workers
                .len(),
            2
        );
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .children
                .len(),
            2
        );

        trie.apply_event(create_remove_event(worker_2, 2, vec![5]))
            .unwrap();
        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .len(),
            2
        );
        assert_eq!(trie.root.borrow().workers.len(), 0);
        assert_eq!(trie.root.borrow().children.len(), 1);
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .workers
                .len(),
            2
        );
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .children
                .len(),
            2
        );

        trie.apply_event(create_remove_event(worker_2, 3, vec![4]))
            .unwrap();

        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(trie.root.borrow().workers.len(), 0);
        assert_eq!(trie.root.borrow().children.len(), 1);
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .workers
                .len(),
            2
        );
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .children
                .len(),
            2
        );

        trie.apply_event(create_store_event(
            worker_2,
            4,
            vec![2, 6, 7],
            Some(ExternalSequenceBlockHash(100)),
        ))
        .unwrap();

        let scores = trie.find_matches(
            vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
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
            &2
        );

        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
                .unwrap()
                .len(),
            4
        );
        assert_eq!(trie.root.borrow().workers.len(), 0);
        assert_eq!(trie.root.borrow().children.len(), 1);
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .workers
                .len(),
            2
        );
        assert_eq!(
            trie.root
                .borrow()
                .children
                .get(&LocalBlockHash(1))
                .unwrap()
                .borrow()
                .children
                .len(),
            2
        );
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .get(&ExternalSequenceBlockHash(200))
                .unwrap()
                .borrow()
                .workers
                .len(),
            2
        );
    }

    #[test]
    fn test_radix_tree_apply_event_errors() {
        let mut trie = RadixTree::new();
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

        // Block not found for remove event.
        let result = trie.apply_event(create_remove_event(worker_0, 0, vec![1, 2, 3]));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KvCacheEventError::BlockNotFound
        ));

        // Parent appears in blocks: parent=1, blocks=[1, 2, 3]
        // This should be rejected as block 1 (hash 100) is the parent - this is
        // a self referencing block.
        trie.apply_event(create_store_event(worker_0, 4, vec![1], None))
            .unwrap();
        let result = trie.apply_event(create_store_event(
            worker_0,
            5,
            vec![1, 2, 3],
            Some(ExternalSequenceBlockHash(100)),
        ));
        assert!(matches!(
            result.unwrap_err(),
            KvCacheEventError::InvalidBlockSequence
        ));
    }

    #[test]
    fn test_clear_all_blocks() {
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        assert!(
            trie.find_matches(vec![LocalBlockHash(0)], false)
                .scores
                .is_empty()
        );

        // Test clearing an empty worker
        trie.clear_all_blocks(worker_0);
        assert!(
            !trie
                .lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );

        // Test clearing a worker with shared blocks
        trie.apply_event(create_store_event(worker_0, 0, vec![0, 1, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![0, 2, 3], None))
            .unwrap();

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(
            result.len() == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_0)] == 1
                && result[&WorkerWithDpRank::from_worker_id(worker_1)] == 1
        );

        trie.clear_all_blocks(worker_0);

        assert!(
            trie.lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .unwrap()
                .is_empty()
        );
        let result = trie
            .find_matches(vec![LocalBlockHash(0), LocalBlockHash(2)], false)
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_1)], 2);
        let result = trie
            .find_matches(
                vec![LocalBlockHash(0), LocalBlockHash(1), LocalBlockHash(3)],
                false,
            )
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_1)], 1);

        // Test re-adding blocks after clearing worker
        trie.apply_event(create_store_event(worker_0, 0, vec![4, 5], None))
            .unwrap();
        let result = trie
            .find_matches(vec![LocalBlockHash(4), LocalBlockHash(5)], false)
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_0)], 2);

        // Test multiple clears
        trie.clear_all_blocks(worker_0);
        trie.clear_all_blocks(worker_0);
        assert!(
            trie.lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );

        // Test clearing all workers
        trie.clear_all_blocks(worker_0);
        trie.clear_all_blocks(worker_1);
        assert!(!trie.lookup.is_empty());
        assert!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .unwrap()
                .is_empty()
        );
        assert!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_1))
                .unwrap()
                .is_empty()
        );

        // Test clearing a worker that has been removed
        trie.apply_event(create_store_event(worker_0, 0, vec![6], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![6], None))
            .unwrap();
        trie.remove_worker(worker_0);
        trie.clear_all_blocks(worker_0);
        assert!(
            !trie
                .lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        let result = trie.find_matches(vec![LocalBlockHash(6)], false).scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_1)], 1);

        // Test clearing a worker that doesn't exist
        let worker_fake = 2;
        assert!(
            !trie
                .lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_fake))
        );
        trie.clear_all_blocks(worker_fake);
        assert!(
            !trie
                .lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_fake))
        );
        assert!(
            trie.lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_1))
        );
        let result = trie.find_matches(vec![LocalBlockHash(6)], false).scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&WorkerWithDpRank::from_worker_id(worker_1)], 1);
    }

    #[test]
    fn test_radix_tree_default() {
        let radix_tree: RadixTree = Default::default();
        assert!(radix_tree.root.borrow().children.is_empty());
        assert!(radix_tree.root.borrow().workers.is_empty());
        assert!(radix_tree.lookup.is_empty());
    }

    #[test]
    fn test_remove_worker_verifies_hash_removal() {
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;
        let worker_2 = 2;

        // Add blocks for multiple workers
        trie.apply_event(create_store_event(worker_0, 0, vec![1, 2, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![1, 2, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_2, 0, vec![1, 4, 5], None))
            .unwrap();

        // Verify worker_0 has 3 blocks in lookup
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .unwrap()
                .len(),
            3
        );

        // Verify that blocks have the correct workers
        let block_1 = trie
            .lookup
            .get(&WorkerWithDpRank::from_worker_id(worker_0))
            .unwrap()
            .get(&ExternalSequenceBlockHash(100))
            .unwrap();
        assert_eq!(block_1.borrow().workers.len(), 3); // worker_0, worker_1, and worker_2 (all have hash 1)
        assert!(
            block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_1))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_2))
        );

        // Remove worker_0
        trie.remove_worker(worker_0);

        // Verify worker_0 is completely removed from lookup table
        assert!(
            !trie
                .lookup
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert_eq!(trie.lookup.len(), 2);

        // Verify that worker_0's hash is removed from the workers set
        let block_1 = trie
            .lookup
            .get(&WorkerWithDpRank::from_worker_id(worker_1))
            .unwrap()
            .get(&ExternalSequenceBlockHash(100))
            .unwrap();
        assert_eq!(block_1.borrow().workers.len(), 2); // worker_1 and worker_2 remain
        assert!(
            !block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_1))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_2))
        );

        // Verify that blocks with no remaining workers have their children cleared
        // This tests the optimization where empty blocks clear their children
        let block_2 = trie
            .lookup
            .get(&WorkerWithDpRank::from_worker_id(worker_1))
            .unwrap()
            .get(&ExternalSequenceBlockHash(200))
            .unwrap();
        assert_eq!(block_2.borrow().workers.len(), 1); // only worker_1
        assert!(
            block_2
                .borrow()
                .workers
                .contains(&WorkerWithDpRank::from_worker_id(worker_1))
        );

        // Verify match results no longer include worker_0
        let result = trie
            .find_matches(
                vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
                false,
            )
            .scores;
        assert_eq!(result.len(), 2);
        assert!(!result.contains_key(&WorkerWithDpRank::from_worker_id(worker_0)));
        assert!(result.contains_key(&WorkerWithDpRank::from_worker_id(worker_1)));
        assert!(result.contains_key(&WorkerWithDpRank::from_worker_id(worker_2)));
    }
}
