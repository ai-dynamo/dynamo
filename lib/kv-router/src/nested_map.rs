// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Positional HashMap-based KV cache index with nested structure.
//!
//! This module provides a `NestedMap` structure that uses nested HashMaps
//! keyed by position for better cache locality and enables jump/binary-search
//! optimizations in find_matches.
//!
//! # Structure
//!
//! - `index`: position -> local_hash -> seq_hash -> workers
//!   The main lookup structure. Position-first nesting enables O(1) position access.
//! - `worker_blocks`: worker -> seq_hash -> (position, local_hash)
//!   Per-worker reverse lookup for efficient remove operations.

use std::collections::{HashMap, HashSet};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores,
    RouterEvent, WorkerId, WorkerWithDpRank, compute_hash,
};

/// Entry for the innermost level of the index.
///
/// Optimizes for the common case where there's only one sequence hash
/// at a given (position, local_hash) pair, avoiding HashMap allocation.
#[derive(Debug, Clone)]
enum SeqEntry {
    /// Single seq_hash -> workers mapping (common case, no HashMap allocation)
    Single(ExternalSequenceBlockHash, HashSet<WorkerWithDpRank>),
    /// Multiple seq_hash -> workers mappings (rare case, different prefixes)
    Multi(HashMap<ExternalSequenceBlockHash, HashSet<WorkerWithDpRank>>),
}

impl SeqEntry {
    /// Create a new entry with a single worker.
    fn new(seq_hash: ExternalSequenceBlockHash, worker: WorkerWithDpRank) -> Self {
        let mut workers = HashSet::new();
        workers.insert(worker);
        Self::Single(seq_hash, workers)
    }

    /// Insert a worker for a given seq_hash, upgrading to Multi if needed.
    fn insert(&mut self, seq_hash: ExternalSequenceBlockHash, worker: WorkerWithDpRank) {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.insert(worker);
            }
            Self::Single(existing_hash, existing_workers) => {
                // Upgrade to Multi
                let mut map = HashMap::with_capacity(2);
                map.insert(*existing_hash, std::mem::take(existing_workers));
                map.entry(seq_hash).or_default().insert(worker);
                *self = Self::Multi(map);
            }
            Self::Multi(map) => {
                map.entry(seq_hash).or_default().insert(worker);
            }
        }
    }

    /// Remove a worker from a given seq_hash.
    /// Returns true if the entry is now completely empty and should be removed.
    fn remove(&mut self, seq_hash: ExternalSequenceBlockHash, worker: WorkerWithDpRank) -> bool {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.remove(&worker);
                workers.is_empty()
            }
            Self::Single(_, _) => false, // Different hash, nothing to remove
            Self::Multi(map) => {
                if let Some(workers) = map.get_mut(&seq_hash) {
                    workers.remove(&worker);
                    if workers.is_empty() {
                        map.remove(&seq_hash);
                    }
                }
                map.is_empty()
            }
        }
    }

    /// Get workers for a specific seq_hash.
    fn get(&self, seq_hash: ExternalSequenceBlockHash) -> Option<&HashSet<WorkerWithDpRank>> {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => Some(workers),
            Self::Single(_, _) => None,
            Self::Multi(map) => map.get(&seq_hash),
        }
    }

    /// Get workers when there's only one entry (lazy optimization).
    /// Returns None if there are multiple seq_hash entries.
    fn get_single(&self) -> Option<&HashSet<WorkerWithDpRank>> {
        match self {
            Self::Single(_, workers) => Some(workers),
            Self::Multi(map) if map.len() == 1 => map.values().next(),
            Self::Multi(_) => None,
        }
    }

    /// Check if this entry has only one seq_hash (lazy optimization applies).
    #[cfg(test)]
    fn is_single(&self) -> bool {
        match self {
            Self::Single(_, _) => true,
            Self::Multi(map) => map.len() == 1,
        }
    }

    /// Get the number of distinct seq_hash entries.
    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::Single(_, _) => 1,
            Self::Multi(map) => map.len(),
        }
    }

    /// Check if a seq_hash exists in this entry.
    #[cfg(test)]
    fn contains(&self, seq_hash: ExternalSequenceBlockHash) -> bool {
        match self {
            Self::Single(existing_hash, _) => *existing_hash == seq_hash,
            Self::Multi(map) => map.contains_key(&seq_hash),
        }
    }

    /// Iterate over all worker sets (for testing/debugging).
    #[cfg(test)]
    fn iter_workers(&self) -> impl Iterator<Item = &HashSet<WorkerWithDpRank>> {
        let single_iter = match self {
            Self::Single(_, workers) => Some(workers),
            Self::Multi(_) => None,
        };
        let multi_iter = match self {
            Self::Multi(map) => Some(map.values()),
            Self::Single(_, _) => None,
        };
        single_iter
            .into_iter()
            .chain(multi_iter.into_iter().flatten())
    }
}

/// A positional HashMap-based structure for KV cache indexing.
///
/// Uses nested HashMaps with position as the primary key to enable:
/// - O(1) position-based access
/// - Jump/binary-search optimization for find_matches
/// - Lazy sequence hash computation (only when disambiguation needed)
///
/// # Structure
///
/// - `index`: position -> local_hash -> seq_hash -> workers
/// - `worker_blocks`: worker -> seq_hash -> (position, local_hash)
pub struct NestedMap {
    /// Main index: position -> local_hash -> SeqEntry
    /// Uses SeqEntry to optimize for the common single-seq-hash case.
    index: HashMap<u64, HashMap<LocalBlockHash, SeqEntry>>,

    /// Per-worker reverse lookup: worker -> seq_hash -> (position, local_hash)
    /// Enables efficient remove operations without global flat reverse map.
    worker_blocks:
        HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, (u64, LocalBlockHash)>>,

    /// Jump size for find_matches optimization (default 32)
    jump_size: u64,
}

impl NestedMap {
    /// Create a new empty NestedMap with default jump size of 32.
    pub fn new() -> Self {
        Self::new_with_jump_size(32)
    }

    /// Create a new empty NestedMap with specified jump size.
    pub fn new_with_jump_size(jump_size: u64) -> Self {
        Self {
            index: HashMap::new(),
            worker_blocks: HashMap::new(),
            jump_size,
        }
    }

    /// Store blocks for a worker starting from a parent position.
    ///
    /// # Arguments
    /// * `worker` - The worker storing the blocks
    /// * `parent_hash` - Optional parent block hash (None for root)
    /// * `blocks` - List of (seq_hash, local_hash) pairs to store
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(KvCacheEventError::ParentBlockNotFound)` if parent_hash is Some but not found
    fn store_blocks(
        &mut self,
        worker: WorkerWithDpRank,
        parent_hash: Option<ExternalSequenceBlockHash>,
        blocks: &[(ExternalSequenceBlockHash, LocalBlockHash)],
    ) -> Result<(), KvCacheEventError> {
        // Determine starting position
        let start_pos = match parent_hash {
            None => 0,
            Some(parent) => {
                let worker_map = self.worker_blocks.get(&worker);
                match worker_map.and_then(|m| m.get(&parent)) {
                    Some(&(parent_pos, _)) => parent_pos + 1,
                    None => {
                        tracing::warn!(
                            worker_id = worker.worker_id.to_string(),
                            dp_rank = worker.dp_rank,
                            parent_hash = ?parent,
                            num_blocks = blocks.len(),
                            "Failed to find parent block; skipping store operation"
                        );
                        return Err(KvCacheEventError::ParentBlockNotFound);
                    }
                }
            }
        };

        // Get or create worker's block map
        let worker_block_map = self.worker_blocks.entry(worker).or_default();

        // Insert each block
        for (i, &(seq_hash, local_hash)) in blocks.iter().enumerate() {
            let position = start_pos + i as u64;

            // Insert into main index using SeqEntry
            let local_map = self.index.entry(position).or_default();
            match local_map.get_mut(&local_hash) {
                Some(entry) => entry.insert(seq_hash, worker),
                None => {
                    local_map.insert(local_hash, SeqEntry::new(seq_hash, worker));
                }
            }

            // Insert into worker's reverse lookup
            worker_block_map.insert(seq_hash, (position, local_hash));
        }

        Ok(())
    }

    /// Remove blocks for a worker.
    ///
    /// Uses per-worker reverse lookup to find position and local_hash for each block.
    fn remove_blocks(
        &mut self,
        worker: WorkerWithDpRank,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), KvCacheEventError> {
        let Some(worker_block_map) = self.worker_blocks.get_mut(&worker) else {
            return Ok(());
        };

        // Collect blocks to remove (need position and local_hash from reverse lookup)
        let mut to_remove: Vec<(ExternalSequenceBlockHash, u64, LocalBlockHash)> = Vec::new();
        let mut any_not_found = false;

        for &seq_hash in block_hashes {
            let Some(&(position, local_hash)) = worker_block_map.get(&seq_hash) else {
                tracing::warn!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    block_hash = ?seq_hash,
                    "Block not found in worker's lookup; skipping"
                );
                any_not_found = true;
                continue;
            };
            to_remove.push((seq_hash, position, local_hash));
        }

        // Remove from worker's reverse lookup
        for &(seq_hash, _, _) in &to_remove {
            worker_block_map.remove(&seq_hash);
        }

        // Clean up empty worker entry before we lose the mutable reference
        let worker_empty = worker_block_map.is_empty();
        if worker_empty {
            self.worker_blocks.remove(&worker);
        }

        // Remove from main index
        for (seq_hash, position, local_hash) in to_remove {
            self.remove_worker_from_index(worker, position, local_hash, seq_hash);
        }

        if any_not_found {
            Err(KvCacheEventError::BlockNotFound)
        } else {
            Ok(())
        }
    }

    /// Get workers at a specific position and local_hash with lazy hash computation.
    ///
    /// If `seq_hash` is provided, uses it for exact lookup.
    /// Otherwise, if there's only one entry at (position, local_hash), uses it directly.
    /// Returns None if disambiguation is needed but seq_hash wasn't provided.
    fn get_workers_at_position(
        &self,
        position: u64,
        local_hash: LocalBlockHash,
        seq_hash: Option<ExternalSequenceBlockHash>,
    ) -> Option<&HashSet<WorkerWithDpRank>> {
        let local_map = self.index.get(&position)?;
        let entry = local_map.get(&local_hash)?;

        if let Some(hash) = seq_hash {
            return entry.get(hash);
        }

        // Lazy optimization: if only one entry, use it directly
        entry.get_single()
    }

    /// Get workers at position with lazy hash, computing seq_hash only if needed.
    ///
    /// This is a convenience wrapper that handles the common pattern of:
    /// 1. Check if position/local_hash exists
    /// 2. If single entry, use directly (skip hash computation)
    /// 3. If multiple entries, compute hash and disambiguate
    fn get_workers_lazy(
        &self,
        position: u64,
        local_hash: LocalBlockHash,
        seq_hashes: &mut Vec<u64>,
        sequence: &[LocalBlockHash],
    ) -> Option<&HashSet<WorkerWithDpRank>> {
        let local_map = self.index.get(&position)?;
        let entry = local_map.get(&local_hash)?;

        // Lazy optimization: if single entry, skip hash computation
        if let Some(workers) = entry.get_single() {
            return Some(workers);
        }

        // Multiple entries - compute seq_hash for disambiguation
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        let hash = ExternalSequenceBlockHash(seq_hashes[position as usize]);
        entry.get(hash)
    }

    /// Ensure seq_hashes is computed up to and including target_pos.
    fn ensure_seq_hash_computed(
        seq_hashes: &mut Vec<u64>,
        target_pos: u64,
        sequence: &[LocalBlockHash],
    ) {
        while seq_hashes.len() <= target_pos as usize {
            let pos = seq_hashes.len();
            let prev = seq_hashes[pos - 1];
            let local = sequence[pos].0;
            seq_hashes.push(Self::compute_next_seq_hash(prev, local));
        }
    }

    /// Compute sequence hash incrementally from previous hash and current local hash.
    fn compute_next_seq_hash(prev_seq_hash: u64, current_local_hash: u64) -> u64 {
        let combined = [prev_seq_hash, current_local_hash];
        let bytes: Vec<u8> = combined.iter().flat_map(|&num| num.to_le_bytes()).collect();
        compute_hash(&bytes)
    }

    /// Find matches for a sequence of local block hashes.
    ///
    /// Uses jump optimization to skip common prefixes efficiently.
    /// Lazy sequence hash computation - only computes when disambiguation is needed.
    ///
    /// # Algorithm
    ///
    /// 1. Start at position 0, jump by `jump_size` positions
    /// 2. At each jump, check if workers drained:
    ///    - Same workers: continue jumping
    ///    - Fewer workers or missing position: scan range to find drain points
    /// 3. Continue until sequence exhausted or no workers remain
    pub fn find_matches(&self, sequence: Vec<LocalBlockHash>, early_exit: bool) -> OverlapScores {
        let mut scores = OverlapScores::new();

        if sequence.is_empty() {
            return scores;
        }

        let seq_len = sequence.len() as u64;

        // Check first position - first block's seq_hash equals its local_hash
        let first_local = sequence[0];
        let first_seq_hash = ExternalSequenceBlockHash(first_local.0);
        let Some(first_workers) =
            self.get_workers_at_position(0, first_local, Some(first_seq_hash))
        else {
            return scores;
        };

        if first_workers.is_empty() {
            return scores;
        }

        let mut active = first_workers.clone();
        let mut current_pos: u64 = 0;
        let mut seq_hashes: Vec<u64> = vec![first_local.0];

        // Process remaining positions with jump optimization
        while current_pos + 1 < seq_len {
            let next_pos = std::cmp::min(current_pos + self.jump_size, seq_len - 1);
            let jumped = next_pos > current_pos + 1;
            let next_local = sequence[next_pos as usize];

            let Some(workers) =
                self.get_workers_lazy(next_pos, next_local, &mut seq_hashes, &sequence)
            else {
                // Position doesn't exist or no match
                if !jumped {
                    break; // Sequential move - we're done
                }
                // Scan to find boundary
                let prev_pos = current_pos;
                current_pos = self.linear_scan_drain(
                    &sequence,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos,
                    next_pos,
                    early_exit,
                );
                // If no progress made (first position in range had no match), we're done
                if current_pos == prev_pos || active.is_empty() || (early_exit && active.len() == 1)
                {
                    break;
                }
                continue;
            };

            // Check drain status
            let all_match = active.iter().all(|w| workers.contains(w));
            let any_match = active.iter().any(|w| workers.contains(w));

            if all_match {
                // No drain - continue (skip intermediate positions)
                current_pos = next_pos;
            } else if !any_match {
                // All workers drained
                if !jumped {
                    break; // Sequential - all drained at this position
                }
                let prev_pos = current_pos;
                current_pos = self.linear_scan_drain(
                    &sequence,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos,
                    next_pos,
                    early_exit,
                );
                if current_pos == prev_pos || active.is_empty() || (early_exit && active.len() == 1)
                {
                    break;
                }
            } else if jumped {
                // Some workers drained after a jump - scan for exact drain points
                let prev_pos = current_pos;
                current_pos = self.linear_scan_drain(
                    &sequence,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos,
                    next_pos,
                    early_exit,
                );
                if current_pos == prev_pos || active.is_empty() || (early_exit && active.len() == 1)
                {
                    break;
                }
            } else {
                // Sequential move with partial drain - record drains directly
                let depth = next_pos as u32; // depth at previous position
                active.retain(|worker| {
                    if workers.contains(worker) {
                        return true;
                    }
                    scores.scores.insert(*worker, depth);
                    false
                });
                current_pos = next_pos;
            }

            if early_exit && active.len() == 1 {
                break;
            }
        }

        // Drain remaining active workers with final depth
        let final_depth = (current_pos + 1) as u32;
        for worker in active {
            scores.scores.insert(worker, final_depth);
        }

        // Populate tree sizes
        for &worker in scores.scores.keys() {
            let Some(blocks) = self.worker_blocks.get(&worker) else {
                continue;
            };
            scores.tree_sizes.insert(worker, blocks.len());
        }

        scores
    }

    /// Scan positions sequentially in (lo, hi] range, updating active set and scores.
    ///
    /// Returns the highest position where workers remain.
    ///
    /// TODO: Optimize to use binary search for O(log J) instead of O(J) per range.
    #[allow(clippy::too_many_arguments)]
    fn linear_scan_drain(
        &self,
        sequence: &[LocalBlockHash],
        seq_hashes: &mut Vec<u64>,
        active: &mut HashSet<WorkerWithDpRank>,
        scores: &mut OverlapScores,
        lo: u64,
        hi: u64,
        early_exit: bool,
    ) -> u64 {
        let mut current_pos = lo;

        for pos in (lo + 1)..=hi {
            let local_hash = sequence[pos as usize];

            let Some(worker_set) = self.get_workers_lazy(pos, local_hash, seq_hashes, sequence)
            else {
                return current_pos; // No workers at this position
            };

            let depth = pos as u32;
            active.retain(|worker| {
                if worker_set.contains(worker) {
                    return true;
                }
                scores.scores.insert(*worker, depth);
                false
            });

            if active.is_empty() {
                return current_pos;
            }

            current_pos = pos;

            // Early exit if only one worker remains
            if early_exit && active.len() == 1 {
                return current_pos;
            }
        }

        current_pos
    }

    /// Apply a RouterEvent to the index.
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(KvCacheEventError)` on error (parent not found, block not found, etc.)
    pub fn apply_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let blocks: Vec<_> = store_data
                    .blocks
                    .iter()
                    .map(|b| (b.block_hash, b.tokens_hash))
                    .collect();
                self.store_blocks(worker, store_data.parent_hash, &blocks)
            }
            KvCacheEventData::Removed(remove_data) => {
                self.remove_blocks(worker, &remove_data.block_hashes)
            }
            KvCacheEventData::Cleared => {
                self.clear_all_blocks(worker.worker_id);
                Ok(())
            }
        }
    }

    /// Helper function to remove or clear blocks for a worker.
    fn remove_or_clear_worker_blocks(&mut self, worker_id: WorkerId, keep_worker: bool) {
        let workers: Vec<WorkerWithDpRank> = self
            .worker_blocks
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        for worker in workers {
            let Some(block_map) = self.worker_blocks.remove(&worker) else {
                continue;
            };

            // Remove worker from all blocks in main index
            for (seq_hash, (position, local_hash)) in block_map {
                self.remove_worker_from_index(worker, position, local_hash, seq_hash);
            }

            if keep_worker {
                self.worker_blocks.insert(worker, HashMap::new());
            }
        }
    }

    /// Remove a worker from a specific index entry and cleanup empty maps.
    fn remove_worker_from_index(
        &mut self,
        worker: WorkerWithDpRank,
        position: u64,
        local_hash: LocalBlockHash,
        seq_hash: ExternalSequenceBlockHash,
    ) {
        let Some(local_map) = self.index.get_mut(&position) else {
            return;
        };

        let Some(entry) = local_map.get_mut(&local_hash) else {
            return;
        };

        // Remove worker from entry; if entry becomes empty, remove it
        if entry.remove(seq_hash, worker) {
            local_map.remove(&local_hash);
        }

        if local_map.is_empty() {
            self.index.remove(&position);
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
    pub fn get_workers(&self) -> Vec<WorkerId> {
        let mut worker_ids: Vec<WorkerId> = self
            .worker_blocks
            .keys()
            .map(|w| w.worker_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        worker_ids.sort_unstable();
        worker_ids
    }

    /// Dump the index as a series of RouterEvents that can reconstruct the state.
    ///
    /// NOTE: Not implemented for PositionalIndex - use serialization directly if needed.
    pub fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        unimplemented!(
            "dump_tree_as_events not supported for PositionalIndex; serialize directly if needed"
        )
    }

    /// Returns the total number of (worker, block) pairs stored.
    pub fn current_size(&self) -> usize {
        self.worker_blocks.values().map(|m| m.len()).sum()
    }
}

impl Default for NestedMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{
        KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
    };

    /// Helper to create store event with proper cumulative hashes.
    fn make_store_event(
        worker_id: WorkerId,
        event_id: u64,
        local_hashes: &[u64],
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) -> RouterEvent {
        // Compute cumulative sequence hashes
        let local_blocks: Vec<LocalBlockHash> =
            local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let seq_hashes = crate::protocols::compute_seq_hash_for_block(&local_blocks);

        let blocks: Vec<KvCacheStoredBlockData> = local_hashes
            .iter()
            .zip(seq_hashes.iter())
            .map(|(&local, &seq)| KvCacheStoredBlockData {
                tokens_hash: LocalBlockHash(local),
                block_hash: ExternalSequenceBlockHash(seq),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    blocks,
                }),
                dp_rank: 0,
            },
        }
    }

    fn make_remove_event(worker_id: WorkerId, event_id: u64, seq_hashes: &[u64]) -> RouterEvent {
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: seq_hashes
                        .iter()
                        .map(|&h| ExternalSequenceBlockHash(h))
                        .collect(),
                }),
                dp_rank: 0,
            },
        }
    }

    #[test]
    fn test_index_structure_after_store() {
        let mut index = NestedMap::new();

        // Store [1, 2, 3] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();

        // Verify index has 3 positions
        assert_eq!(index.index.len(), 3, "Should have 3 positions");
        assert!(index.index.contains_key(&0), "Position 0 should exist");
        assert!(index.index.contains_key(&1), "Position 1 should exist");
        assert!(index.index.contains_key(&2), "Position 2 should exist");

        // Verify each position has the correct local_hash entry
        let pos0 = index.index.get(&0).unwrap();
        assert!(
            pos0.contains_key(&LocalBlockHash(1)),
            "Position 0 should have local_hash 1"
        );

        let pos1 = index.index.get(&1).unwrap();
        assert!(
            pos1.contains_key(&LocalBlockHash(2)),
            "Position 1 should have local_hash 2"
        );

        let pos2 = index.index.get(&2).unwrap();
        assert!(
            pos2.contains_key(&LocalBlockHash(3)),
            "Position 2 should have local_hash 3"
        );

        // Verify worker is in each position's worker set
        let worker = WorkerWithDpRank::new(0, 0);
        for pos in 0..3 {
            let local_map = index.index.get(&pos).unwrap();
            let has_worker = local_map.values().any(|entry| {
                entry
                    .iter_workers()
                    .any(|workers| workers.contains(&worker))
            });
            assert!(has_worker, "Worker should be at position {}", pos);
        }
    }

    #[test]
    fn test_worker_blocks_reverse_lookup() {
        let mut index = NestedMap::new();

        // Store [10, 20, 30] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[10, 20, 30], None))
            .unwrap();

        let worker = WorkerWithDpRank::new(0, 0);
        let worker_map = index.worker_blocks.get(&worker).unwrap();

        // Should have 3 entries
        assert_eq!(worker_map.len(), 3, "Worker should have 3 blocks");

        // Verify each entry has correct (position, local_hash)
        for (seq_hash, &(position, local_hash)) in worker_map {
            // Verify position matches expected
            assert!(position < 3, "Position should be 0, 1, or 2");

            // Verify we can navigate to the correct place in main index
            let local_map = index.index.get(&position).unwrap();
            let entry = local_map.get(&local_hash).unwrap();
            assert!(entry.contains(*seq_hash), "Should find seq_hash in index");
        }
    }

    #[test]
    fn test_multiple_workers_same_prefix() {
        let mut index = NestedMap::new();

        // Both workers store [100, 200]
        index
            .apply_event(make_store_event(0, 0, &[100, 200], None))
            .unwrap();
        index
            .apply_event(make_store_event(1, 1, &[100, 200], None))
            .unwrap();

        // Position 0, local_hash 100 should have 1 seq_hash entry with 2 workers
        let pos0 = index.index.get(&0).unwrap();
        let entry = pos0.get(&LocalBlockHash(100)).unwrap();
        assert_eq!(
            entry.len(),
            1,
            "Should have exactly 1 seq_hash (no collision)"
        );

        let workers = entry.get_single().unwrap();
        assert_eq!(workers.len(), 2, "Should have 2 workers");
        assert!(workers.contains(&WorkerWithDpRank::new(0, 0)));
        assert!(workers.contains(&WorkerWithDpRank::new(1, 0)));
    }

    #[test]
    fn test_different_prefixes_same_local_at_position() {
        let mut index = NestedMap::new();

        // Worker 0: [1, 2, SHARED]
        // Worker 1: [3, 4, SHARED]
        // At position 2, both have local_hash SHARED but different seq_hash (different prefixes)
        let shared = 999u64;
        index
            .apply_event(make_store_event(0, 0, &[1, 2, shared], None))
            .unwrap();
        index
            .apply_event(make_store_event(1, 1, &[3, 4, shared], None))
            .unwrap();

        // Position 2 should have local_hash SHARED
        let pos2 = index.index.get(&2).unwrap();
        let entry = pos2.get(&LocalBlockHash(shared)).unwrap();

        // Should have 2 different seq_hash entries (different prefixes)
        assert_eq!(
            entry.len(),
            2,
            "Should have 2 seq_hash entries for different prefixes"
        );

        // Each seq_hash should map to exactly 1 worker
        for workers in entry.iter_workers() {
            assert_eq!(workers.len(), 1, "Each seq_hash should have 1 worker");
        }
    }

    #[test]
    fn test_remove_cleans_up_nested_maps() {
        let mut index = NestedMap::new();

        // Store [1, 2, 3] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();
        assert_eq!(index.index.len(), 3);

        // Compute the seq_hashes for removal
        let local_blocks = vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)];
        let seq_hashes = crate::protocols::compute_seq_hash_for_block(&local_blocks);

        // Remove all blocks
        index
            .apply_event(make_remove_event(0, 1, &seq_hashes))
            .unwrap();

        // Index should be completely empty
        assert!(
            index.index.is_empty(),
            "Index should be empty after removing all blocks"
        );
        assert!(
            index.worker_blocks.is_empty(),
            "Worker blocks should be empty"
        );
    }

    #[test]
    fn test_remove_partial_keeps_other_positions() {
        let mut index = NestedMap::new();

        // Store [1, 2, 3] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();

        // Compute seq_hashes
        let local_blocks = vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)];
        let seq_hashes = crate::protocols::compute_seq_hash_for_block(&local_blocks);

        // Remove only the last block
        index
            .apply_event(make_remove_event(0, 1, &[seq_hashes[2]]))
            .unwrap();

        // Should have 2 positions left
        assert_eq!(
            index.index.len(),
            2,
            "Should have 2 positions after partial remove"
        );
        assert!(index.index.contains_key(&0));
        assert!(index.index.contains_key(&1));
        assert!(
            !index.index.contains_key(&2),
            "Position 2 should be removed"
        );
    }

    #[test]
    fn test_jump_optimization_skips_positions() {
        // Use small jump size to test the jump behavior
        let mut index = NestedMap::new_with_jump_size(4);

        // Store a long sequence [0, 1, 2, ..., 15] for worker 0
        let seq: Vec<u64> = (0..16).collect();
        index
            .apply_event(make_store_event(0, 0, &seq, None))
            .unwrap();

        // Store same sequence for worker 1 but only first 8 blocks
        let partial: Vec<u64> = (0..8).collect();
        index
            .apply_event(make_store_event(1, 1, &partial, None))
            .unwrap();

        // Query the full sequence
        let query: Vec<LocalBlockHash> = seq.iter().map(|&h| LocalBlockHash(h)).collect();
        let scores = index.find_matches(query, false);

        // Worker 0 should have depth 16, worker 1 should have depth 8
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            16
        );
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(), 8);
    }

    #[test]
    fn test_lazy_hash_single_entry_optimization() {
        let mut index = NestedMap::new();

        // Store [1, 2, 3] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();

        // Each (position, local_hash) should have exactly 1 seq_hash entry (Single variant)
        for (pos, local_map) in &index.index {
            for (local_hash, entry) in local_map {
                assert!(
                    entry.is_single(),
                    "Position {}, local_hash {:?} should be Single variant (lazy optimization applies)",
                    pos,
                    local_hash
                );
            }
        }
    }

    #[test]
    fn test_chained_stores_with_parent() {
        let mut index = NestedMap::new();

        // Store [1, 2] for worker 0
        index
            .apply_event(make_store_event(0, 0, &[1, 2], None))
            .unwrap();

        // Compute parent hash (seq_hash at position 1)
        let local_blocks = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let seq_hashes = crate::protocols::compute_seq_hash_for_block(&local_blocks);
        let parent = ExternalSequenceBlockHash(seq_hashes[1]);

        // Store [3, 4] continuing from parent for worker 0
        index
            .apply_event(make_store_event(0, 1, &[3, 4], Some(parent)))
            .unwrap();

        // Should have 4 positions now
        assert_eq!(
            index.index.len(),
            4,
            "Should have 4 positions after chained store"
        );
        assert!(index.index.contains_key(&0));
        assert!(index.index.contains_key(&1));
        assert!(index.index.contains_key(&2));
        assert!(index.index.contains_key(&3));

        // Worker should have 4 blocks total
        let worker = WorkerWithDpRank::new(0, 0);
        assert_eq!(index.worker_blocks.get(&worker).unwrap().len(), 4);
    }

    #[test]
    fn test_parent_not_found_error() {
        let mut index = NestedMap::new();

        // Try to store with non-existent parent
        let fake_parent = ExternalSequenceBlockHash(0xDEADBEEF);
        let result = index.apply_event(make_store_event(0, 0, &[1, 2], Some(fake_parent)));

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            KvCacheEventError::ParentBlockNotFound
        ));
    }

    #[test]
    fn test_find_matches_empty_on_miss() {
        let mut index = NestedMap::new();

        // Store [1, 2, 3]
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();

        // Query for completely different sequence
        let query = vec![LocalBlockHash(999), LocalBlockHash(998)];
        let scores = index.find_matches(query, false);

        assert!(
            scores.scores.is_empty(),
            "Should have no matches for miss query"
        );
    }

    #[test]
    fn test_tree_sizes_in_overlap_scores() {
        let mut index = NestedMap::new();

        // Worker 0 has 5 blocks, worker 1 has 3 blocks
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3, 4, 5], None))
            .unwrap();
        index
            .apply_event(make_store_event(1, 1, &[1, 2, 3], None))
            .unwrap();

        // Query [1, 2]
        let query = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let scores = index.find_matches(query, false);

        // Both workers should appear with correct tree sizes
        assert_eq!(
            *scores.tree_sizes.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            5
        );
        assert_eq!(
            *scores.tree_sizes.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            3
        );
    }

    #[test]
    fn test_early_exit_stops_at_single_worker() {
        let mut index = NestedMap::new_with_jump_size(2);

        // Worker 0: [1, 2, 3, 4, 5]
        // Worker 1: [1] only
        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3, 4, 5], None))
            .unwrap();
        index
            .apply_event(make_store_event(1, 1, &[1], None))
            .unwrap();

        // Query [1, 2, 3, 4, 5] with early_exit=true
        let query: Vec<LocalBlockHash> = (1..=5).map(LocalBlockHash).collect();
        let scores = index.find_matches(query, true);

        // Worker 1 drops at position 1 (after block 1), leaving worker 0 alone
        // With early_exit, we stop when only 1 worker remains
        // Worker 0's score should be 2 (blocks 1 and 2), not 5
        let worker0_score = *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap();
        assert!(
            worker0_score <= 2,
            "Early exit should stop early, got {}",
            worker0_score
        );
    }

    #[test]
    fn test_current_size() {
        let mut index = NestedMap::new();

        assert_eq!(index.current_size(), 0);

        index
            .apply_event(make_store_event(0, 0, &[1, 2, 3], None))
            .unwrap();
        assert_eq!(index.current_size(), 3);

        index
            .apply_event(make_store_event(1, 1, &[1, 2], None))
            .unwrap();
        assert_eq!(index.current_size(), 5);

        index.remove_worker(0);
        assert_eq!(index.current_size(), 2);
    }

    #[test]
    fn test_get_workers() {
        let mut index = NestedMap::new();

        index
            .apply_event(make_store_event(2, 0, &[1], None))
            .unwrap();
        index
            .apply_event(make_store_event(0, 1, &[1], None))
            .unwrap();
        index
            .apply_event(make_store_event(1, 2, &[1], None))
            .unwrap();

        let workers = index.get_workers();
        assert_eq!(workers, vec![0, 1, 2], "Workers should be sorted");
    }
}
