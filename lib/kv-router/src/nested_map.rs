// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Positional HashMap-based KV cache index with nested structure.
//!
//! This module provides a `PositionalIndexer` that uses nested HashMaps
//! keyed by position for better cache locality and enables jump/binary-search
//! optimizations in find_matches.
//!
//! # Structure
//!
//! - `index`: position -> local_hash -> seq_hash -> workers
//!   The main lookup structure. Position-first nesting enables O(1) position access.
//! - `worker_blocks`: worker -> seq_hash -> (position, local_hash)
//!   Per-worker reverse lookup for efficient remove operations.
//!
//! # Threading
//!
//! `PositionalIndexer` implements `SyncIndexer`, meaning all its methods are
//! synchronous and thread-safe (via `DashMap` and `RwLock`). To get the full
//! `KvIndexerInterface` with sticky event routing and worker threads, wrap it
//! in a `ThreadPoolIndexer`.
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::indexer::SyncIndexer;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheStoreData,
    LocalBlockHash, OverlapScores, RouterEvent, WorkerId, WorkerWithDpRank,
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
}

type LevelIndex = RwLock<HashMap<ExternalSequenceBlockHash, (usize, LocalBlockHash)>>;

/// Positional HashMap-based KV cache index.
///
/// Implements [`SyncIndexer`] for use with [`ThreadPoolIndexer`](crate::indexer::ThreadPoolIndexer).
/// All methods are synchronous and thread-safe.
pub struct PositionalIndexer {
    index: DashMap<(usize, LocalBlockHash), SeqEntry>,
    /// Per-worker reverse lookup: worker -> seq_hash -> (position, local_hash)
    /// Enables efficient remove operations without global flat reverse map.
    worker_blocks: DashMap<WorkerWithDpRank, LevelIndex>,

    jump_size: usize,
}

impl PositionalIndexer {
    /// Create a new PositionalIndexer.
    ///
    /// # Arguments
    /// * `jump_size` - Jump size for find_matches optimization (e.g., 32).
    ///   The algorithm jumps by this many positions at a time, only scanning
    ///   intermediate positions when workers drain (stop matching).
    pub fn new(jump_size: usize) -> Self {
        assert!(jump_size > 0, "jump_size must be greater than 0");

        Self {
            index: DashMap::new(),
            worker_blocks: DashMap::new(),
            jump_size,
        }
    }
}

// ============================================================================
// SyncIndexer implementation
// ============================================================================

impl SyncIndexer for PositionalIndexer {
    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.jump_search_matches(sequence, early_exit)
    }

    fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        Self::apply_event_impl(&self.index, &self.worker_blocks, event)
    }

    fn remove_worker(&self, worker_id: WorkerId) {
        Self::remove_or_clear_worker_blocks_impl(
            &self.index,
            &self.worker_blocks,
            worker_id,
            false,
        );
    }

    fn dump_events(&self) -> Vec<RouterEvent> {
        // Not implemented for positional indexer - would require reconstructing
        // the tree structure from the flat position-based index.
        unimplemented!("dump_events is not supported for PositionalIndexer");
    }
}

// ============================================================================
// Event processing (write operations)
// ============================================================================

impl PositionalIndexer {
    /// Process an event using the provided index and worker_blocks.
    /// This is called from worker threads.
    fn apply_event_impl(
        index: &DashMap<(usize, LocalBlockHash), SeqEntry>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);

        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        tracing::trace!(
            id,
            "PositionalIndexer::apply_event_impl: operation: {:?}",
            op
        );

        match op {
            KvCacheEventData::Stored(store_data) => {
                Self::store_blocks_impl(index, worker_blocks, worker, store_data, id)?;

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                Self::remove_blocks_impl(
                    index,
                    worker_blocks,
                    worker,
                    &remove_data.block_hashes,
                    id,
                )?;
                Ok(())
            }
            KvCacheEventData::Cleared => {
                Self::clear_worker_blocks_impl(index, worker_blocks, worker_id);
                Ok(())
            }
        }
    }

    fn store_blocks_impl(
        index: &DashMap<(usize, LocalBlockHash), SeqEntry>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker: WorkerWithDpRank,
        store_data: KvCacheStoreData,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        // Determine starting position based on parent_hash
        let start_pos = match store_data.parent_hash {
            Some(parent_hash) => {
                // Find parent position from worker_blocks

                let Some(worker_map) = worker_blocks.get(&worker) else {
                    tracing::warn!(
                        worker_id = worker.worker_id.to_string(),
                        dp_rank = worker.dp_rank,
                        event_id,
                        parent_hash = ?parent_hash,
                    );
                    return Err(KvCacheEventError::ParentBlockNotFound);
                };

                let worker_map = worker_map.read().unwrap();

                let Some(entry) = worker_map.get(&parent_hash) else {
                    tracing::warn!(
                        worker_id = worker.worker_id.to_string(),
                        dp_rank = worker.dp_rank,
                        event_id,
                        parent_hash = ?parent_hash,
                    );
                    return Err(KvCacheEventError::ParentBlockNotFound);
                };

                entry.0 + 1 // parent position + 1
            }
            None => 0, // Start from position 0
        };

        if !worker_blocks.contains_key(&worker) {
            worker_blocks.insert(worker, RwLock::new(HashMap::new()));
        }

        let worker_blocks_entry = worker_blocks.get(&worker).unwrap();
        let mut worker_map = worker_blocks_entry.write().unwrap();

        for (i, block_data) in store_data.blocks.into_iter().enumerate() {
            let position = start_pos + i;
            let local_hash = block_data.tokens_hash;
            let seq_hash = block_data.block_hash;

            index
                .entry((position, local_hash))
                .and_modify(|entry| entry.insert(seq_hash, worker))
                .or_insert_with(|| SeqEntry::new(seq_hash, worker));

            // Insert into worker_blocks: worker -> seq_hash -> (position, local_hash)
            worker_map.insert(seq_hash, (position, local_hash));
        }

        Ok(())
    }

    fn remove_blocks_impl(
        index: &DashMap<(usize, LocalBlockHash), SeqEntry>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker: WorkerWithDpRank,
        seq_hashes: &Vec<ExternalSequenceBlockHash>,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        let worker_map = worker_blocks.get(&worker).ok_or_else(|| {
            tracing::warn!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                event_id,
                block_hashes = ?seq_hashes,
                "Failed to find worker blocks to remove"
            );
            KvCacheEventError::BlockNotFound
        })?;

        let mut worker_map = worker_map.write().unwrap();

        for seq_hash in seq_hashes {
            let Some((position, local_hash)) = worker_map.remove(seq_hash) else {
                tracing::warn!(
                    worker_id = worker.worker_id.to_string(),
                    dp_rank = worker.dp_rank,
                    event_id,
                    block_hash = ?seq_hash,
                    "Failed to find block to remove; skipping remove operation"
                );
                return Err(KvCacheEventError::BlockNotFound);
            };

            // Remove from index
            if let Some(mut entry) = index.get_mut(&(position, local_hash)) {
                let _ = entry.remove(*seq_hash, worker);
            }
        }

        Ok(())
    }

    /// Clear all blocks for a specific worker_id (all dp_ranks), but keep worker tracked.
    /// Static version for use in worker threads.
    fn clear_worker_blocks_impl(
        index: &DashMap<(usize, LocalBlockHash), SeqEntry>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
    ) {
        Self::remove_or_clear_worker_blocks_impl(index, worker_blocks, worker_id, true);
    }

    /// Remove a worker and all their blocks completely from the index.
    #[allow(dead_code)]
    fn remove_worker_blocks(&self, worker_id: WorkerId) {
        Self::remove_or_clear_worker_blocks_impl(
            &self.index,
            &self.worker_blocks,
            worker_id,
            false,
        );
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains tracked with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed.
    fn remove_or_clear_worker_blocks_impl(
        index: &DashMap<(usize, LocalBlockHash), SeqEntry>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
        keep_worker: bool,
    ) {
        // Collect all WorkerWithDpRank keys that match this worker_id
        let workers: Vec<WorkerWithDpRank> = worker_blocks
            .iter()
            .filter(|entry| entry.key().worker_id == worker_id)
            .map(|entry| *entry.key())
            .collect();

        for worker in workers {
            if let Some((_, worker_map)) = worker_blocks.remove(&worker) {
                // Remove each block from the index
                for entry in worker_map.read().unwrap().iter() {
                    let seq_hash = *entry.0;
                    let (position, local_hash) = *entry.1;

                    if let Some(mut entry) = index.get_mut(&(position, local_hash)) {
                        let _ = entry.remove(seq_hash, worker);
                    }
                }
            }

            if keep_worker {
                // Re-insert worker with empty map to keep it tracked
                worker_blocks.insert(worker, RwLock::new(HashMap::new()));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Jump-based search methods (associated functions for use in worker threads)
// -----------------------------------------------------------------------------

impl PositionalIndexer {
    /// Compute sequence hash incrementally from previous hash and current local hash.
    #[inline]
    fn compute_next_seq_hash(prev_seq_hash: u64, current_local_hash: u64) -> u64 {
        let mut bytes = [0u8; 16];

        bytes[..8].copy_from_slice(&prev_seq_hash.to_le_bytes());
        bytes[8..].copy_from_slice(&current_local_hash.to_le_bytes());

        crate::protocols::compute_hash(&bytes)
    }

    /// Ensure seq_hashes is computed up to and including target_pos.
    /// Lazily extends the seq_hashes vector as needed.
    #[inline]
    fn ensure_seq_hash_computed(
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        target_pos: usize,
        sequence: &[LocalBlockHash],
    ) {
        while seq_hashes.len() <= target_pos {
            let pos = seq_hashes.len();
            if pos == 0 {
                // First block's seq_hash equals its local_hash
                seq_hashes.push(ExternalSequenceBlockHash::from(sequence[0].0));
            } else {
                let prev_seq_hash = seq_hashes[pos - 1].0;
                let current_local_hash = sequence[pos].0;
                let next_hash = Self::compute_next_seq_hash(prev_seq_hash, current_local_hash);
                seq_hashes.push(ExternalSequenceBlockHash::from(next_hash));
            }
        }
    }

    /// Get workers at a position by verifying both local_hash and seq_hash match.
    ///
    /// Returns None if no workers match at this position.
    /// Always computes and verifies the seq_hash to ensure correctness when
    /// the query may have diverged from stored sequences at earlier positions.
    fn get_workers_lazy(
        &self,
        position: usize,
        local_hash: LocalBlockHash,
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        sequence: &[LocalBlockHash],
    ) -> Option<HashSet<WorkerWithDpRank>> {
        let entry = self.index.get(&(position, local_hash))?;

        // Always compute and verify seq_hash to handle divergent queries correctly.
        // Even if there's only one seq_hash entry, the query's seq_hash might differ
        // if the query diverged from the stored sequence at an earlier position.
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        let seq_hash = seq_hashes[position];
        entry.get(seq_hash).cloned()
    }

    fn count_workers_at(
        &self,
        position: usize,
        local_hash: LocalBlockHash,
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        sequence: &[LocalBlockHash],
    ) -> Option<usize> {
        let entry = self.index.get(&(position, local_hash))?;

        // Always compute and verify seq_hash to handle divergent queries correctly.
        // Even if there's only one seq_hash entry, the query's seq_hash might differ
        // if the query diverged from the stored sequence at an earlier position.
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        let seq_hash = seq_hashes[position];
        Some(
            entry
                .get(seq_hash)
                .map(|workers| workers.len())
                .unwrap_or(0),
        )
    }

    /// Scan positions sequentially, updating active set and recording drain scores.
    ///
    /// Returns the highest position where workers remain active (or lo-1 if none match at lo).
    #[allow(clippy::too_many_arguments)]
    fn linear_scan_drain(
        &self,
        sequence: &[LocalBlockHash],
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        active: &mut HashSet<WorkerWithDpRank>,
        scores: &mut OverlapScores,
        lo: usize,
        hi: usize,
        early_exit: bool,
    ) {
        for pos in lo..hi {
            if active.is_empty() {
                break;
            }

            let workers_at_pos = self.get_workers_lazy(pos, sequence[pos], seq_hashes, sequence);

            match workers_at_pos {
                Some(workers) => {
                    active.retain(|w| {
                        if workers.contains(w) {
                            true
                        } else {
                            // Score is the position where they stopped matching (i.e., pos)
                            // which represents they matched positions 0..pos
                            scores.scores.insert(*w, pos as u32);
                            false
                        }
                    });
                    if early_exit && !active.is_empty() {
                        // Found at least one match, can exit early
                        break;
                    }
                }
                None => {
                    for worker in active.iter() {
                        scores.scores.insert(*worker, pos as u32);
                    }
                    active.clear();
                }
            }
        }
    }

    /// Jump-based search to find matches for a sequence of block hashes.
    ///
    /// # Algorithm
    ///
    /// 1. Check first position - initialize active set with matching workers
    /// 2. Initialize seq_hashes with first block's hash (seq_hash[0] = local_hash[0])
    /// 3. Loop: jump by jump_size positions
    ///    - At each jump, check if active workers still match:
    ///      - All match: Continue jumping (skip intermediate positions)
    ///      - None match: Scan range with linear_scan_drain
    ///      - Partial match: Scan range to find exact drain points
    /// 4. Record final scores for remaining active workers
    /// 5. Populate tree_sizes from worker_blocks
    ///
    /// # Arguments
    /// * `index` - The position -> local_hash -> SeqEntry index
    /// * `worker_blocks` - Per-worker reverse lookup for tree sizes
    /// * `local_hashes` - Sequence of LocalBlockHash to match
    /// * `jump_size` - Number of positions to jump at a time
    /// * `early_exit` - If true, stop after finding any match
    fn jump_search_matches(
        &self,
        local_hashes: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();

        if local_hashes.is_empty() {
            return scores;
        }

        // Lazily computed sequence hashes
        let mut seq_hashes: Vec<ExternalSequenceBlockHash> = Vec::new();

        // Check first position to initialize active set
        let Some(initial_workers) =
            self.get_workers_lazy(0, local_hashes[0], &mut seq_hashes, local_hashes)
        else {
            return scores;
        };

        let mut active = initial_workers;

        if active.is_empty() {
            return scores;
        }

        if early_exit {
            // For early exit, just record that these workers matched at least position 0
            for worker in &active {
                scores.scores.insert(*worker, 1);
            }
            // Populate tree_sizes
            for worker in scores.scores.keys() {
                if let Some(worker_map) = self.worker_blocks.get(worker) {
                    let worker_map = worker_map.read().unwrap();
                    scores.tree_sizes.insert(*worker, worker_map.len());
                }
            }
            return scores;
        }

        let len = local_hashes.len();
        let mut current_pos = 0;

        // Jump through positions
        while current_pos < len - 1 && !active.is_empty() {
            let next_pos = (current_pos + self.jump_size).min(len - 1);

            // Check workers at jump destination
            let num_workers_at_next = self
                .count_workers_at(
                    next_pos,
                    local_hashes[next_pos],
                    &mut seq_hashes,
                    local_hashes,
                )
                .unwrap_or(0);

            if num_workers_at_next == active.len() {
                current_pos = next_pos;
            } else {
                // No active workers match at jump destination
                // Scan the range to find where each worker drained
                self.linear_scan_drain(
                    local_hashes,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos + 1,
                    next_pos + 1,
                    false,
                );
                current_pos = next_pos;
            }
        }

        // Record final scores for remaining active workers
        // They matched all positions through the end
        let final_score = len as u32;
        for worker in active {
            scores.scores.insert(worker, final_score);
        }

        // Populate tree_sizes from worker_blocks
        for worker in scores.scores.keys() {
            if let Some(worker_map) = self.worker_blocks.get(worker) {
                let worker_map = worker_map.read().unwrap();
                scores.tree_sizes.insert(*worker, worker_map.len());
            }
        }

        scores
    }
}
