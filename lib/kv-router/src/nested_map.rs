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
use async_trait::async_trait;
use dashmap::DashMap;
use flume::unbounded;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId,
    WorkerWithDpRank, compute_seq_hash_for_block,
};

use crate::compute_block_hash_for_seq;
use crate::indexer::{KvIndexerInterface, KvRouterError, MatchRequest};

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

type LevelIndex = DashMap<ExternalSequenceBlockHash, (usize, LocalBlockHash)>;

pub(crate) struct PositionalIndexer {
    index: Arc<DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>>,
    /// Per-worker reverse lookup: worker -> seq_hash -> (position, local_hash)
    /// Enables efficient remove operations without global flat reverse map.
    worker_blocks: Arc<DashMap<WorkerWithDpRank, LevelIndex>>,

    worker_assignments: Arc<DashMap<WorkerId, usize>>,
    worker_assignment_count: AtomicUsize,

    worker_event_channels: Arc<Vec<flume::Sender<RouterEvent>>>,
    worker_request_channel: flume::Sender<MatchRequest>,

    num_workers: usize,
    kv_block_size: u32,

    /// Weight for binary search starting position (0.0 to 1.0).
    /// With weight=0.2 and 100 blocks, search starts at position 20.
    /// Lower values are better when expecting shorter prefix matches.
    search_weight: f64,

    /// Cancellation token to signal worker threads to shut down.
    cancel_token: CancellationToken,
    /// Handles to worker threads for joining on shutdown.
    thread_handles: Mutex<Vec<JoinHandle<()>>>,
}

impl PositionalIndexer {
    /// Create a new PositionalIndexer.
    ///
    /// # Arguments
    /// * `num_workers` - Number of worker threads for event processing
    /// * `kv_block_size` - Block size for KV cache
    /// * `search_weight` - Weight for binary search starting position (0.0 to 1.0).
    ///   With weight=0.2 and 100 blocks, search starts at position 20.
    ///   Lower values are better when expecting shorter prefix matches.
    pub fn new(num_workers: usize, kv_block_size: u32, search_weight: f64) -> Self {
        assert!(num_workers > 0, "Number of workers must be greater than 0");
        assert!(
            (0.0..=1.0).contains(&search_weight),
            "search_weight must be between 0.0 and 1.0"
        );

        let index = Arc::new(DashMap::new());
        let worker_blocks = Arc::new(DashMap::new());
        let mut worker_event_senders = Vec::new();
        let mut thread_handles = Vec::new();
        let cancel_token = CancellationToken::new();

        let (worker_request_channel_tx, worker_request_channel_rx) = unbounded::<MatchRequest>();

        for _ in 0..num_workers {
            let (event_sender, event_receiver) = unbounded::<RouterEvent>();
            worker_event_senders.push(event_sender);

            let worker_request_channel_rx = worker_request_channel_rx.clone();
            let index = Arc::clone(&index);
            let cancel_token = cancel_token.clone();

            let handle = std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        loop {
                            tokio::select! {
                                biased;

                                _ = cancel_token.cancelled() => {
                                    break;
                                }

                                Ok(_event) = event_receiver.recv_async() => {
                                    // Event processing handled elsewhere
                                }

                                Ok(request) = worker_request_channel_rx.recv_async() => {
                                    let scores = PositionalIndexer::weighted_binary_search_matches(
                                        &index,
                                        &request.sequence,
                                        search_weight,
                                        request.early_exit,
                                    );
                                    let _ = request.resp.send(scores);
                                }
                            }
                        }
                    });
            });
            thread_handles.push(handle);
        }

        Self {
            index,
            worker_blocks,
            worker_assignments: Arc::new(DashMap::new()),
            worker_assignment_count: AtomicUsize::new(0),
            worker_event_channels: Arc::new(worker_event_senders),
            worker_request_channel: worker_request_channel_tx,
            num_workers,
            kv_block_size,
            search_weight,
            cancel_token,
            thread_handles: Mutex::new(thread_handles),
        }
    }
}

#[async_trait]
impl KvIndexerInterface for PositionalIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = MatchRequest {
            sequence,
            early_exit: false,
            resp: resp_tx,
        };

        if let Err(e) = self.worker_request_channel.send(req) {
            tracing::error!(
                "Failed to send match request: {:?}; the indexer maybe offline",
                e
            );
            return Err(KvRouterError::IndexerOffline);
        }

        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        // Process events synchronously since DashMap is thread-safe
        if let Err(e) = self.apply_event_sync(event) {
            tracing::warn!("Failed to apply event: {:?}", e);
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        self.remove_worker_blocks(worker_id);
    }

    fn shutdown(&self) {
        // Signal all worker threads to stop
        self.cancel_token.cancel();

        // Take ownership of thread handles and join them
        let handles = std::mem::take(
            &mut *self
                .thread_handles
                .lock()
                .expect("thread_handles mutex poisoned"),
        );
        for handle in handles {
            if let Err(e) = handle.join() {
                tracing::error!("Worker thread panicked during shutdown: {:?}", e);
            }
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        Ok(self.dump_as_events())
    }

    async fn process_routing_decision_for_request(
        &self,
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // TODO(jthomson04): Nothing to do here, right?
        Ok(())
    }
}

impl PositionalIndexer {
    fn apply_event_sync(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let (worker_id, kv_event) = (event.worker_id, event.event);
        let (id, op) = (kv_event.event_id, kv_event.data);

        let worker = WorkerWithDpRank::new(worker_id, kv_event.dp_rank);

        tracing::trace!(
            id,
            "PositionalIndexer::apply_event_sync: operation: {:?}",
            op
        );

        match op {
            KvCacheEventData::Stored(store_data) => {
                // Determine starting position based on parent_hash
                let start_pos = match store_data.parent_hash {
                    Some(parent_hash) => {
                        // Find parent position from worker_blocks

                        let Some(worker_map) = self.worker_blocks.get(&worker) else {
                            tracing::warn!(
                                worker_id = worker.worker_id.to_string(),
                                dp_rank = worker.dp_rank,
                                id,
                                parent_hash = ?parent_hash,
                            );
                            return Err(KvCacheEventError::ParentBlockNotFound);
                        };

                        let Some(entry) = worker_map.get(&parent_hash) else {
                            tracing::warn!(
                                worker_id = worker.worker_id.to_string(),
                                dp_rank = worker.dp_rank,
                                id,
                                parent_hash = ?parent_hash,
                            );
                            return Err(KvCacheEventError::ParentBlockNotFound);
                        };

                        entry.0 + 1 // parent position + 1
                    }
                    None => 0, // Start from position 0
                };

                for (i, block_data) in store_data.blocks.into_iter().enumerate() {
                    let position = start_pos + i;
                    let local_hash = block_data.tokens_hash;
                    let seq_hash = block_data.block_hash;

                    // Insert into index: position -> local_hash -> seq_hash -> worker
                    let pos_map = self.index.entry(position).or_default();
                    pos_map
                        .entry(local_hash)
                        .and_modify(|entry| entry.insert(seq_hash, worker))
                        .or_insert_with(|| SeqEntry::new(seq_hash, worker));

                    // Insert into worker_blocks: worker -> seq_hash -> (position, local_hash)
                    let worker_map = self.worker_blocks.entry(worker).or_default();
                    worker_map.insert(seq_hash, (position, local_hash));
                }

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                let mut first_err: Option<KvCacheEventError> = None;
                for seq_hash in remove_data.block_hashes {
                    if let Err(e) = self.remove_single_block(worker, seq_hash, id) {
                        first_err.get_or_insert(e);
                    }
                }
                first_err.map_or(Ok(()), Err)
            }
            KvCacheEventData::Cleared => {
                self.clear_worker_blocks(worker_id);
                Ok(())
            }
        }
    }

    /// Remove a single block from the index for a given worker.
    fn remove_single_block(
        &self,
        worker: WorkerWithDpRank,
        seq_hash: ExternalSequenceBlockHash,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        let worker_map = self.worker_blocks.get(&worker).ok_or_else(|| {
            tracing::warn!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                event_id,
                block_hash = ?seq_hash,
                "Failed to find worker blocks to remove"
            );
            KvCacheEventError::BlockNotFound
        })?;

        let (_, (position, local_hash)) = worker_map.remove(&seq_hash).ok_or_else(|| {
            tracing::warn!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                event_id,
                block_hash = ?seq_hash,
                "Failed to find block to remove; skipping remove operation"
            );
            KvCacheEventError::BlockNotFound
        })?;

        // Remove from index
        if let Some(pos_map) = self.index.get(&position)
            && let Some(mut entry) = pos_map.get_mut(&local_hash)
            && entry.remove(seq_hash, worker)
        {
            drop(entry);
            pos_map.remove(&local_hash);
        }

        Ok(())
    }

    /// Clear all blocks for a specific worker_id (all dp_ranks), but keep worker tracked.
    fn clear_worker_blocks(&self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, true);
    }

    /// Remove a worker and all their blocks completely from the index.
    fn remove_worker_blocks(&self, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(worker_id, false);
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains tracked with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed.
    fn remove_or_clear_worker_blocks(&self, worker_id: WorkerId, keep_worker: bool) {
        // Collect all WorkerWithDpRank keys that match this worker_id
        let workers: Vec<WorkerWithDpRank> = self
            .worker_blocks
            .iter()
            .filter(|entry| entry.key().worker_id == worker_id)
            .map(|entry| *entry.key())
            .collect();

        for worker in workers {
            if let Some((_, worker_map)) = self.worker_blocks.remove(&worker) {
                // Remove each block from the index
                for entry in worker_map.iter() {
                    let seq_hash = *entry.key();
                    let (position, local_hash) = *entry.value();

                    if let Some(pos_map) = self.index.get(&position)
                        && let Some(mut seq_entry) = pos_map.get_mut(&local_hash)
                        && seq_entry.remove(seq_hash, worker)
                    {
                        // Entry is empty, remove it
                        drop(seq_entry);
                        pos_map.remove(&local_hash);
                    }
                }
            }

            if keep_worker {
                // Re-insert worker with empty map to keep it tracked
                self.worker_blocks.insert(worker, DashMap::new());
            }
        }
    }

    /// Dump the index as a series of RouterEvents that can reconstruct the index.
    /// Each worker gets one event per block they have stored.
    fn dump_as_events(&self) -> Vec<RouterEvent> {
        unimplemented!();
    }
}

// -----------------------------------------------------------------------------
// Weighted binary search methods (associated functions for use in worker threads)
// -----------------------------------------------------------------------------

impl PositionalIndexer {
    /// Check if there's a match at the given position for the (local_hash, seq_hash) pair.
    #[inline]
    fn has_match_at_position(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        position: usize,
        local_hash: LocalBlockHash,
        seq_hash: ExternalSequenceBlockHash,
    ) -> bool {
        let Some(pos_map) = index.get(&position) else {
            return false;
        };
        let Some(entry) = pos_map.get(&local_hash) else {
            return false;
        };
        entry.get(seq_hash).is_some()
    }

    /// Update scores with workers at a specific position for a (local_hash, seq_hash) pair.
    /// Returns true if workers were found and scores were updated.
    #[inline]
    fn update_scores_at_position(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        position: usize,
        local_hash: LocalBlockHash,
        seq_hash: ExternalSequenceBlockHash,
        scores: &mut OverlapScores,
        record_frequency: bool,
    ) -> bool {
        let Some(pos_map) = index.get(&position) else {
            return false;
        };
        let Some(entry) = pos_map.get(&local_hash) else {
            return false;
        };
        let Some(workers) = entry.get(seq_hash) else {
            return false;
        };
        scores.update_scores(workers.iter());
        if record_frequency {
            scores.add_frequency(workers.len());
        }
        true
    }

    /// Find the boundary between matching and non-matching positions using weighted binary search.
    ///
    /// Returns the first position with no match (i.e., all positions in `0..boundary` match).
    /// The weight is applied at each step of the binary search, biasing toward lower positions.
    ///
    /// # Arguments
    /// * `index` - The position -> local_hash -> SeqEntry index
    /// * `local_hashes` - Sequence of LocalBlockHash to match
    /// * `seq_hashes` - Corresponding sequence hashes for exact matching
    /// * `search_weight` - Weight for midpoint calculation (0.0 to 1.0) applied at each step
    fn find_match_boundary(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        local_hashes: &[LocalBlockHash],
        seq_hashes: &[ExternalSequenceBlockHash],
        search_weight: f64,
    ) -> usize {
        let len = local_hashes.len();
        if len == 0 {
            return 0;
        }

        let mut low = 0;
        let mut high = len;

        while low < high {
            let range = high - low;
            // Apply weight to calculate midpoint, clamped to ensure progress
            let offset = (((range as f64) * search_weight).floor() as usize).min(range - 1);
            let mid = low + offset;

            if Self::has_match_at_position(index, mid, local_hashes[mid], seq_hashes[mid]) {
                // Match found, boundary is above mid
                low = mid + 1;
            } else {
                // No match, boundary is at or below mid
                high = mid;
            }
        }

        low
    }

    /// Perform weighted binary search to find matches for a sequence of block hashes.
    ///
    /// # Algorithm
    ///
    /// 1. Compute sequence hashes from local block hashes for exact prefix matching
    /// 2. Use weighted binary search to find the boundary (first non-matching position)
    /// 3. The weight biases each step toward lower positions (useful when short matches are common)
    /// 4. Scan all matching positions (0..boundary) to collect worker scores
    ///
    /// # Arguments
    /// * `index` - The position -> local_hash -> SeqEntry index
    /// * `local_hashes` - Sequence of LocalBlockHash to match
    /// * `search_weight` - Weight for midpoint calculation (0.0 to 1.0) applied at each step
    /// * `early_exit` - If true, stop after finding any match
    fn weighted_binary_search_matches(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        local_hashes: &[LocalBlockHash],
        search_weight: f64,
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();

        if local_hashes.is_empty() {
            return scores;
        }

        // Compute sequence hashes for exact prefix matching
        let seq_hashes: Vec<ExternalSequenceBlockHash> = compute_seq_hash_for_block(local_hashes)
            .into_iter()
            .map(ExternalSequenceBlockHash::from)
            .collect();

        let boundary = Self::find_match_boundary(index, local_hashes, &seq_hashes, search_weight);

        if boundary == 0 {
            return scores;
        }

        // Collect workers from all matching positions (0..boundary)
        if early_exit {
            // For early exit, just check position 0
            Self::update_scores_at_position(
                index,
                0,
                local_hashes[0],
                seq_hashes[0],
                &mut scores,
                false,
            );
        } else {
            // Scan all matching positions
            for pos in 0..boundary {
                Self::update_scores_at_position(
                    index,
                    pos,
                    local_hashes[pos],
                    seq_hashes[pos],
                    &mut scores,
                    true,
                );
            }
        }

        scores
    }
}
