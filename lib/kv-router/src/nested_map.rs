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

use std::time::Duration;
#[cfg(feature = "bench")]
use std::time::Instant;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores,
    RouterEvent, TokensWithHashes, WorkerId, WorkerWithDpRank,
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

pub struct PositionalIndexer {
    index: Arc<DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>>,
    /// Per-worker reverse lookup: worker -> seq_hash -> (position, local_hash)
    /// Enables efficient remove operations without global flat reverse map.
    worker_blocks: Arc<DashMap<WorkerWithDpRank, LevelIndex>>,

    /// Maps WorkerId to worker thread index for sticky routing.
    worker_assignments: Arc<DashMap<WorkerId, usize>>,
    /// Counter for round-robin assignment of new WorkerIds.
    worker_assignment_count: AtomicUsize,

    /// Channels to send events to worker threads.
    worker_event_channels: Arc<Vec<flume::Sender<RouterEvent>>>,
    worker_request_channel: flume::Sender<MatchRequest>,

    num_workers: usize,
    kv_block_size: u32,

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
    /// * `jump_size` - Jump size for find_matches optimization (e.g., 32).
    ///   The algorithm jumps by this many positions at a time, only scanning
    ///   intermediate positions when workers drain (stop matching).
    pub fn new(num_workers: usize, kv_block_size: u32, jump_size: usize) -> Self {
        assert!(num_workers > 0, "Number of workers must be greater than 0");
        assert!(jump_size > 0, "jump_size must be greater than 0");

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
            let worker_blocks = Arc::clone(&worker_blocks);
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

                                Ok(event) = event_receiver.recv_async() => {
                                    if let Err(e) = Self::apply_event_impl(&index, &worker_blocks, event) {
                                        tracing::warn!("Failed to apply event: {:?}", e);
                                    }
                                }

                                Ok(request) = worker_request_channel_rx.recv_async() => {
                                    let scores = PositionalIndexer::jump_search_matches(
                                        &index,
                                        &worker_blocks,
                                        &request.sequence,
                                        jump_size,
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
            cancel_token,
            thread_handles: Mutex::new(thread_handles),
        }
    }

    /// Wait for all pending events and requests to be processed. Used primarily for debugging and benchmarking.
    pub async fn flush(&self) {
        loop {
            let mut all_empty = true;

            for worker_event_channel in self.worker_event_channels.iter() {
                if !worker_event_channel.is_empty() {
                    all_empty = false;
                    break;
                }
            }

            if all_empty && self.worker_request_channel.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
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
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
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
        let worker_id = event.worker_id;

        // Get or assign worker thread index using sticky round-robin
        let thread_idx = *self.worker_assignments.entry(worker_id).or_insert_with(|| {
            let idx = self
                .worker_assignment_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            idx % self.num_workers
        });

        // Send event to the assigned worker thread
        if let Err(e) = self.worker_event_channels[thread_idx].send(event) {
            tracing::error!(
                "Failed to send event to worker thread {}: {:?}",
                thread_idx,
                e
            );
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
        unimplemented!();
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
    /// Process an event using the provided index and worker_blocks.
    /// This is called from worker threads.
    fn apply_event_impl(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
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
                // Determine starting position based on parent_hash
                let start_pos = match store_data.parent_hash {
                    Some(parent_hash) => {
                        // Find parent position from worker_blocks

                        let Some(worker_map) = worker_blocks.get(&worker) else {
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
                    let pos_map = index.entry(position).or_default();
                    pos_map
                        .entry(local_hash)
                        .and_modify(|entry| entry.insert(seq_hash, worker))
                        .or_insert_with(|| SeqEntry::new(seq_hash, worker));

                    // Insert into worker_blocks: worker -> seq_hash -> (position, local_hash)
                    let worker_map = worker_blocks.entry(worker).or_default();
                    worker_map.insert(seq_hash, (position, local_hash));
                }

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                let mut first_err: Option<KvCacheEventError> = None;
                for seq_hash in remove_data.block_hashes {
                    if let Err(e) =
                        Self::remove_single_block_impl(index, worker_blocks, worker, seq_hash, id)
                    {
                        first_err.get_or_insert(e);
                    }
                }
                first_err.map_or(Ok(()), Err)
            }
            KvCacheEventData::Cleared => {
                Self::clear_worker_blocks_impl(index, worker_blocks, worker_id);
                Ok(())
            }
        }
    }

    /// Remove a single block from the index for a given worker.
    fn remove_single_block_impl(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker: WorkerWithDpRank,
        seq_hash: ExternalSequenceBlockHash,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        let worker_map = worker_blocks.get(&worker).ok_or_else(|| {
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
        if let Some(pos_map) = index.get(&position)
            && let Some(mut entry) = pos_map.get_mut(&local_hash)
            && entry.remove(seq_hash, worker)
        {
            drop(entry);
            pos_map.remove(&local_hash);
        }

        Ok(())
    }

    /// Clear all blocks for a specific worker_id (all dp_ranks), but keep worker tracked.
    /// Static version for use in worker threads.
    fn clear_worker_blocks_impl(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
    ) {
        Self::remove_or_clear_worker_blocks_impl(index, worker_blocks, worker_id, true);
    }

    /// Remove a worker and all their blocks completely from the index.
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
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
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
                for entry in worker_map.iter() {
                    let seq_hash = *entry.key();
                    let (position, local_hash) = *entry.value();

                    if let Some(pos_map) = index.get(&position)
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
                worker_blocks.insert(worker, DashMap::new());
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
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        position: usize,
        local_hash: LocalBlockHash,
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        sequence: &[LocalBlockHash],
    ) -> Option<HashSet<WorkerWithDpRank>> {
        let pos_map = index.get(&position)?;
        let entry = pos_map.get(&local_hash)?;

        // Always compute and verify seq_hash to handle divergent queries correctly.
        // Even if there's only one seq_hash entry, the query's seq_hash might differ
        // if the query diverged from the stored sequence at an earlier position.
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        let seq_hash = seq_hashes[position];
        entry.get(seq_hash).cloned()
    }

    /// Scan positions sequentially, updating active set and recording drain scores.
    ///
    /// Returns the highest position where workers remain active (or lo-1 if none match at lo).
    #[allow(clippy::too_many_arguments)]
    fn linear_scan_drain(
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
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

            let workers_at_pos =
                Self::get_workers_lazy(index, pos, sequence[pos], seq_hashes, sequence);

            match workers_at_pos {
                Some(workers) => {
                    for worker in active.difference(&workers) {
                        // Score is the position where they stopped matching (i.e., pos)
                        // which represents they matched positions 0..pos
                        scores.scores.insert(*worker, pos as u32);
                    }
                    *active = workers;
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
        index: &DashMap<usize, DashMap<LocalBlockHash, SeqEntry>>,
        worker_blocks: &DashMap<WorkerWithDpRank, LevelIndex>,
        local_hashes: &[LocalBlockHash],
        jump_size: usize,
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
            Self::get_workers_lazy(index, 0, local_hashes[0], &mut seq_hashes, local_hashes)
        else {
            return scores;
        };

        let mut active = initial_workers;

        if active.is_empty() {
            return scores;
        }

        // Record frequency for position 0
        scores.add_frequency(active.len());

        if early_exit {
            // For early exit, just record that these workers matched at least position 0
            for worker in &active {
                scores.scores.insert(*worker, 1);
            }
            // Populate tree_sizes
            for worker in scores.scores.keys() {
                if let Some(worker_map) = worker_blocks.get(worker) {
                    scores.tree_sizes.insert(*worker, worker_map.len());
                }
            }
            return scores;
        }

        let len = local_hashes.len();
        let mut current_pos = 0;

        // Jump through positions
        while current_pos < len - 1 && !active.is_empty() {
            let next_pos = (current_pos + jump_size).min(len - 1);

            // Check workers at jump destination
            let workers_at_next = Self::get_workers_lazy(
                index,
                next_pos,
                local_hashes[next_pos],
                &mut seq_hashes,
                local_hashes,
            );

            let still_active_at_next: HashSet<WorkerWithDpRank> = match &workers_at_next {
                Some(workers) => active.intersection(workers).cloned().collect(),
                None => HashSet::new(),
            };

            if still_active_at_next == active {
                // All active workers still match at jump destination
                // Record frequency for skipped positions (we know all active workers match)
                for _pos in (current_pos + 1)..=next_pos {
                    scores.add_frequency(active.len());
                }
                current_pos = next_pos;
            } else if still_active_at_next.is_empty() {
                // No active workers match at jump destination
                // Scan the range to find where each worker drained
                Self::linear_scan_drain(
                    index,
                    local_hashes,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    current_pos + 1,
                    next_pos + 1,
                    false,
                );
                current_pos = next_pos;
            } else {
                // Partial match - some workers drained in between
                // Scan the range to find exact drain points
                Self::linear_scan_drain(
                    index,
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
            if let Some(worker_map) = worker_blocks.get(worker) {
                scores.tree_sizes.insert(*worker, worker_map.len());
            }
        }

        scores
    }
}
