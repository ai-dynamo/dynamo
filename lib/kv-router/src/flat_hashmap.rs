// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flat HashMap baseline for benchmarking comparison with RadixTree.
//!
//! This module provides a `FlatHashMap` structure that has full feature parity with `RadixTree`
//! but uses flat HashMaps instead of a tree structure. This isolates the overhead of
//! tree traversal (pointer chasing) from pure HashMap operations.
//!
//! The `find_matches` API matches RadixTree exactly: it takes `LocalBlockHash` values
//! and internally computes the cumulative sequence hashes for lookup.

use std::collections::{HashMap, HashSet};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
    compute_seq_hash_for_block,
};

/// A flat HashMap-based structure for KV cache indexing.
///
/// Unlike RadixTree which uses a tree of nodes connected by pointers,
/// FlatHashMap uses bidirectional HashMaps. This provides the same
/// find_matches semantics but with better cache locality.
///
/// # Structure
///
/// - `block_to_workers`: Maps ExternalSequenceBlockHash -> Set of workers that have this block.
///   Used for efficient find_matches lookups.
/// - `worker_to_blocks`: Maps Worker -> Set of ExternalSequenceBlockHash they have.
///   Used for remove operations and current_size.
pub struct FlatHashMap {
    /// Primary index: block -> workers (for find_matches)
    block_to_workers: HashMap<ExternalSequenceBlockHash, HashSet<WorkerWithDpRank>>,

    /// Secondary index: worker -> blocks (for remove and current_size)
    worker_to_blocks: HashMap<WorkerWithDpRank, HashSet<ExternalSequenceBlockHash>>,
}

impl FlatHashMap {
    /// Create a new empty FlatHashMap.
    pub fn new() -> Self {
        Self {
            block_to_workers: HashMap::new(),
            worker_to_blocks: HashMap::new(),
        }
    }

    /// Store blocks for a worker.
    ///
    /// Updates both indexes for each block.
    pub fn store(&mut self, worker: WorkerWithDpRank, block_hashes: &[ExternalSequenceBlockHash]) {
        let worker_blocks = self.worker_to_blocks.entry(worker).or_default();

        for &block_hash in block_hashes {
            // Add to block -> workers index
            self.block_to_workers
                .entry(block_hash)
                .or_default()
                .insert(worker);

            // Add to worker -> blocks index
            worker_blocks.insert(block_hash);
        }
    }

    /// Remove blocks for a worker.
    ///
    /// Updates both indexes for each block.
    pub fn remove(&mut self, worker: WorkerWithDpRank, block_hashes: &[ExternalSequenceBlockHash]) {
        let Some(worker_blocks) = self.worker_to_blocks.get_mut(&worker) else {
            return;
        };

        for &block_hash in block_hashes {
            // Remove from worker -> blocks index
            worker_blocks.remove(&block_hash);

            // Remove from block -> workers index
            if let Some(workers) = self.block_to_workers.get_mut(&block_hash) {
                workers.remove(&worker);
                if workers.is_empty() {
                    self.block_to_workers.remove(&block_hash);
                }
            }
        }

        // Clean up empty worker entry
        if worker_blocks.is_empty() {
            self.worker_to_blocks.remove(&worker);
        }
    }

    /// Find matches for a sequence of local block hashes.
    ///
    /// This has the same signature as `RadixTree::find_matches`: it takes `LocalBlockHash`
    /// values and internally computes the cumulative sequence hashes for lookup.
    ///
    /// Returns OverlapScores showing which workers have matching blocks.
    /// Stops at first non-match (same semantics as RadixTree).
    ///
    /// # Algorithm
    ///
    /// 1. Compute cumulative sequence hashes from local block hashes
    /// 2. For each sequence hash:
    ///    - Look up which workers have this block
    ///    - Intersect with previously matching workers (in place)
    ///    - Track depth for scoring
    ///    - Stop if no workers remain
    ///
    /// This is O(depth) HashMap lookups + O(num_workers) set operations per level.
    pub fn find_matches(&self, sequence: Vec<LocalBlockHash>, early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();

        if sequence.is_empty() {
            return scores;
        }

        // Compute cumulative sequence hashes from local block hashes
        let seq_hashes = compute_seq_hash_for_block(&sequence);

        // Track active workers and their match depth
        // Workers drop out when they miss a block; their final score is the depth they reached
        let mut active_workers: Option<HashSet<WorkerWithDpRank>> = None;
        let mut depth = 0u32;

        for seq_hash in seq_hashes {
            let block_hash = ExternalSequenceBlockHash(seq_hash);

            // Look up workers that have this block
            let Some(workers) = self.block_to_workers.get(&block_hash) else {
                break; // No workers have this block, stop
            };

            // Intersect with previously active workers (or initialize on first block)
            match &mut active_workers {
                None => {
                    // First block: initialize with workers that have it
                    active_workers = Some(workers.clone());
                }
                Some(active) => {
                    // Record score for workers about to drop out (they matched up to current depth)
                    for &worker in active.iter() {
                        if !workers.contains(&worker) {
                            scores.scores.insert(worker, depth);
                        }
                    }
                    // Keep only workers that have this block (in-place, no allocation)
                    active.retain(|w| workers.contains(w));
                }
            }

            depth += 1;

            let active = active_workers.as_ref().unwrap();
            if active.is_empty() {
                break;
            }

            // Early exit if only one worker matches
            if early_exit && active.len() == 1 {
                break;
            }
        }

        // Record final scores for workers that matched all blocks (or until early exit)
        if let Some(active) = active_workers {
            for worker in active {
                scores.scores.insert(worker, depth);
            }
        }

        // Populate tree sizes for workers with scores
        for &worker in scores.scores.keys() {
            if let Some(blocks) = self.worker_to_blocks.get(&worker) {
                scores.tree_sizes.insert(worker, blocks.len());
            }
        }

        scores
    }

    /// Apply a RouterEvent (for API compatibility with RadixTree).
    pub fn apply_event(&mut self, event: RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let hashes: Vec<_> = store_data.blocks.iter().map(|b| b.block_hash).collect();
                self.store(worker, &hashes);
            }
            KvCacheEventData::Removed(remove_data) => {
                self.remove(worker, &remove_data.block_hashes);
            }
            KvCacheEventData::Cleared => {
                self.clear_all_blocks(worker.worker_id);
            }
        }
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains in lookup with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed from lookup.
    fn remove_or_clear_worker_blocks(&mut self, worker_id: WorkerId, keep_worker: bool) {
        // Collect all WorkerWithDpRank keys that match this worker_id
        let workers: Vec<WorkerWithDpRank> = self
            .worker_to_blocks
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        for worker in workers {
            if let Some(blocks) = self.worker_to_blocks.remove(&worker) {
                for block_hash in blocks {
                    if let Some(workers_set) = self.block_to_workers.get_mut(&block_hash) {
                        workers_set.remove(&worker);
                        if workers_set.is_empty() {
                            self.block_to_workers.remove(&block_hash);
                        }
                    }
                }

                if keep_worker {
                    // Re-insert worker with empty blocks set to keep it tracked
                    self.worker_to_blocks.insert(worker, HashSet::new());
                }
            }
        }
    }

    /// Remove a worker and all their blocks from the index.
    pub fn remove_worker(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, false);
    }

    /// Clear all blocks for a worker but keep the worker tracked.
    pub fn clear_all_blocks(&mut self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, true);
    }

    /// Get all worker IDs currently tracked in the index.
    /// Returns unique worker_ids sorted (ignoring dp_rank differences).
    pub fn get_workers(&self) -> Vec<WorkerId> {
        let mut worker_ids: Vec<WorkerId> = self
            .worker_to_blocks
            .keys()
            .map(|w| w.worker_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        worker_ids.sort_unstable();
        worker_ids
    }

    /// Dump the index as a series of RouterEvents that can reconstruct the state.
    /// For API compatibility with RadixTree.
    pub fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        let mut events = Vec::new();
        let mut event_id = 0u64;

        for (&worker, blocks) in &self.worker_to_blocks {
            for &block_hash in blocks {
                let event = RouterEvent {
                    worker_id: worker.worker_id,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None, // FlatHashMap doesn't track parent relationships
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash,
                                mm_extra_info: None,
                                // We don't have the original tokens_hash, use a placeholder
                                tokens_hash: LocalBlockHash(0),
                            }],
                        }),
                        dp_rank: worker.dp_rank,
                    },
                };
                events.push(event);
                event_id += 1;
            }
        }

        events
    }

    /// Returns the total number of (worker, block) pairs stored.
    pub fn current_size(&self) -> usize {
        self.worker_to_blocks.values().map(|s| s.len()).sum()
    }
}

impl Default for FlatHashMap {
    fn default() -> Self {
        Self::new()
    }
}
