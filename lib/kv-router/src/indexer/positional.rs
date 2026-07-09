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
use dashmap::mapref::entry::Entry;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::sync::Arc;

#[cfg(feature = "bench")]
use super::WorkerObservationState;
use super::{
    EventKind, EventWarningKind, KvIndexerMetrics, PreBoundEventCounters, SyncIndexer,
    WorkerLookupStats, WorkerTask,
};
use crate::active_set::reconcile_active_workers;
use crate::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheEventError,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    WorkerWithDpRank,
};

/// Environment variable selecting the default algorithm used by
/// [`PositionalIndexer::find_matches`]. Values: `"strided"` (default, current behavior) or
/// `"binary"`. Only consulted by [`PositionalIndexer::new`]; explicit constructors override it.
pub const DYN_ROUTER_POSITIONAL_SEARCH_MODE: &str = "DYN_ROUTER_POSITIONAL_SEARCH_MODE";

/// Selects which algorithm [`PositionalIndexer::find_matches`] uses.
///
/// `Strided` jumps by `jump_size`; `Binary` binary-searches the monotone "all active workers
/// still match" predicate (with a linear-scan base case for tight windows). The two produce
/// identical [`OverlapScores`] whenever the worker set matching the query is monotonically
/// non-increasing with position — i.e. for contiguous-from-zero stores and tail removals. Both
/// modes rely on this `count == active.len() ⟹ set equality` property (`Strided` via its
/// per-`jump_size` count check); sparse absolute-position stores (e.g. front eviction or a
/// partial snapshot restore that leaves a worker at a later position without its prefix) can
/// violate it, and there `Binary`'s coarser probing can diverge from `Strided`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Strided jump-search (the original behavior).
    #[default]
    Strided,
    /// Monotone-predicate binary search with a linear-scan base case.
    Binary,
}

impl SearchMode {
    /// Read the default search mode from [`DYN_ROUTER_POSITIONAL_SEARCH_MODE`], falling back to
    /// [`SearchMode::Strided`]. Follows the `DYN_ROUTER_*` pattern: parse, default, warn on bad value.
    fn from_env() -> Self {
        match std::env::var(DYN_ROUTER_POSITIONAL_SEARCH_MODE) {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "strided" => SearchMode::Strided,
                "binary" => SearchMode::Binary,
                other => {
                    tracing::warn!(
                        value = %other,
                        "invalid {DYN_ROUTER_POSITIONAL_SEARCH_MODE}, expected 'strided' or 'binary'; falling back to strided"
                    );
                    SearchMode::Strided
                }
            },
            Err(_) => SearchMode::Strided,
        }
    }
}

/// Entry for the innermost level of the index.
///
/// Optimizes for the common case where there's only one sequence hash
/// at a given (position, local_hash) pair, avoiding HashMap allocation.
#[derive(Debug, Clone)]
enum SeqEntry {
    /// Single seq_hash -> workers mapping (common case, no HashMap allocation)
    Single(ExternalSequenceBlockHash, FxHashSet<WorkerWithDpRank>),
    /// Multiple seq_hash -> workers mappings (rare case, different prefixes)
    Multi(FxHashMap<ExternalSequenceBlockHash, FxHashSet<WorkerWithDpRank>>),
}

impl SeqEntry {
    /// Create a new entry with a single worker.
    fn new(seq_hash: ExternalSequenceBlockHash, worker: WorkerWithDpRank) -> Self {
        let mut workers = FxHashSet::default();
        workers.insert(worker);
        Self::Single(seq_hash, workers)
    }

    /// Insert a worker for a given seq_hash, upgrading to Multi if needed.
    fn insert(&mut self, seq_hash: ExternalSequenceBlockHash, worker: WorkerWithDpRank) -> bool {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.insert(worker)
            }
            Self::Single(existing_hash, existing_workers) => {
                // Upgrade to Multi
                let mut map = FxHashMap::with_capacity_and_hasher(2, FxBuildHasher);
                map.insert(*existing_hash, std::mem::take(existing_workers));
                map.entry(seq_hash).or_default().insert(worker);
                *self = Self::Multi(map);
                true
            }
            Self::Multi(map) => map.entry(seq_hash).or_default().insert(worker),
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
    fn get(&self, seq_hash: ExternalSequenceBlockHash) -> Option<&FxHashSet<WorkerWithDpRank>> {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => Some(workers),
            Self::Single(_, _) => None,
            Self::Multi(map) => map.get(&seq_hash),
        }
    }
}

pub type LevelIndex = FxHashMap<ExternalSequenceBlockHash, (usize, LocalBlockHash)>;

/// Positional HashMap-based KV cache index.
///
/// Implements [`SyncIndexer`] for use with [`ThreadPoolIndexer`](crate::indexer::ThreadPoolIndexer).
/// All methods are synchronous and thread-safe.
pub struct PositionalIndexer {
    index: DashMap<(usize, LocalBlockHash), SeqEntry, FxBuildHasher>,

    jump_size: usize,

    search_mode: SearchMode,
}

impl PositionalIndexer {
    /// Create a new PositionalIndexer.
    ///
    /// The search mode defaults to whatever [`DYN_ROUTER_POSITIONAL_SEARCH_MODE`] selects
    /// (or [`SearchMode::Strided`] when unset). Use [`PositionalIndexer::new_with_mode`] or
    /// [`PositionalIndexer::with_search_mode`] to pin a mode regardless of the environment.
    ///
    /// # Arguments
    /// * `jump_size` - Jump size for find_matches optimization (e.g., 32).
    ///   The algorithm jumps by this many positions at a time, only scanning
    ///   intermediate positions when workers drain (stop matching). In `Binary` mode it also
    ///   serves as the window threshold below which the search falls back to a linear scan.
    pub fn new(jump_size: usize) -> Self {
        Self::new_with_mode(jump_size, SearchMode::from_env())
    }

    /// Create a new PositionalIndexer with an explicit [`SearchMode`].
    ///
    /// The explicit mode always wins over the environment variable.
    pub fn new_with_mode(jump_size: usize, search_mode: SearchMode) -> Self {
        assert!(jump_size > 0, "jump_size must be greater than 0");

        Self {
            index: DashMap::with_hasher(FxBuildHasher),
            jump_size,
            search_mode,
        }
    }

    /// Builder-style override of the [`SearchMode`] (chiefly for tests / benchmarks).
    pub fn with_search_mode(mut self, search_mode: SearchMode) -> Self {
        self.search_mode = search_mode;
        self
    }
}

// ============================================================================
// SyncIndexer implementation
// ============================================================================

impl SyncIndexer for PositionalIndexer {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut worker_blocks = FxHashMap::default();
        let counters = metrics.as_ref().map(|m| m.prebind());
        #[cfg(feature = "bench")]
        let mut observation = WorkerObservationState::default();

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut worker_blocks, event, counters.as_ref());
                    if result.is_err() {
                        tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
                    }
                    if let Some(ref c) = counters {
                        c.inc(kind, result);
                    }
                }
                WorkerTask::EventWithAck { event, resp } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut worker_blocks, event, counters.as_ref());
                    let applied = result.is_ok();
                    if result.is_err() {
                        tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
                    }
                    if let Some(ref c) = counters {
                        c.inc(kind, result);
                    }
                    let _ = resp.send(applied);
                }
                #[cfg(feature = "bench")]
                WorkerTask::InstallObservation { writer, resp } => {
                    observation.install(writer, resp);
                }
                #[cfg(feature = "bench")]
                WorkerTask::ObservedEvent {
                    event,
                    correlation_id,
                } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut worker_blocks, event, counters.as_ref());
                    observation.record(correlation_id, result.is_ok());
                    if result.is_err() {
                        tracing::warn!("Failed to apply event: {:?}", result.as_ref().err());
                    }
                    if let Some(ref c) = counters {
                        c.inc(kind, result);
                    }
                }
                #[cfg(feature = "bench")]
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                #[cfg(feature = "bench")]
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { worker, anchor } => {
                    if let Err(error) = self.apply_anchor(worker, anchor) {
                        tracing::warn!(?error, "Failed to apply anchor");
                    }
                }
                WorkerTask::RemoveWorker { worker_id, .. } => {
                    self.remove_or_clear_worker_blocks_impl(&mut worker_blocks, worker_id, false);
                }
                WorkerTask::RemoveWorkerDpRank {
                    worker_id, dp_rank, ..
                } => {
                    self.remove_worker_dp_rank_impl(&mut worker_blocks, worker_id, dp_rank);
                }
                WorkerTask::CleanupStaleChildren => {
                    self.run_cleanup_task();
                }
                WorkerTask::DumpEvents(sender) => {
                    let events = self.dump_events(&worker_blocks);
                    if let Err(e) = sender.send(Ok(events)) {
                        tracing::warn!("Failed to send events: {:?}", e);
                    }
                }
                WorkerTask::Stats(sender) => {
                    let stats = WorkerLookupStats::from_worker_block_counts(
                        worker_blocks
                            .iter()
                            .map(|(worker, worker_map)| (*worker, worker_map.len())),
                    );
                    let _ = sender.send(stats);
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => {
                    break;
                }
            }
        }

        tracing::debug!("PositionalIndexer worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        match self.search_mode {
            SearchMode::Strided => self.jump_search_matches(sequence, early_exit),
            SearchMode::Binary => self.binary_search_matches(sequence, early_exit),
        }
    }
}

// ============================================================================
// Event processing (write operations)
// ============================================================================

impl PositionalIndexer {
    /// Process an event using the provided index and worker_blocks.
    /// This is called from worker threads.
    pub fn apply_event(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
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
                self.store_blocks_impl(worker_blocks, worker, store_data, id, counters)?;

                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                self.remove_blocks_impl(worker_blocks, worker, &remove_data.block_hashes, id)?;
                Ok(())
            }
            KvCacheEventData::Cleared => {
                self.clear_worker_blocks_impl(worker_blocks, worker_id);
                Ok(())
            }
        }
    }

    fn store_blocks_impl(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        worker: WorkerWithDpRank,
        store_data: KvCacheStoreData,
        event_id: u64,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let KvCacheStoreData {
            parent_hash,
            start_position,
            blocks,
        } = store_data;
        let worker_map = worker_blocks.entry(worker).or_default();
        let start_pos = match start_position {
            Some(start_position) => start_position as usize,
            None => match parent_hash {
                Some(parent_hash) => {
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
            },
        };

        let worker_blocks_entry = worker_blocks.entry(worker).or_default();

        let mut duplicate_store = !blocks.is_empty();

        for (i, block_data) in blocks.into_iter().enumerate() {
            let position = start_pos + i;
            let local_hash = block_data.tokens_hash;
            let seq_hash = block_data.block_hash;

            match self.index.entry((position, local_hash)) {
                Entry::Occupied(mut entry) => {
                    if entry.get_mut().insert(seq_hash, worker) {
                        duplicate_store = false;
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(SeqEntry::new(seq_hash, worker));
                    duplicate_store = false;
                }
            }

            // Insert into worker_blocks: worker -> seq_hash -> (position, local_hash)
            match worker_blocks_entry.insert(seq_hash, (position, local_hash)) {
                Some(existing) if existing == (position, local_hash) => {}
                Some(_) => duplicate_store = false,
                None => {
                    duplicate_store = false;
                }
            }
        }

        if duplicate_store && let Some(counters) = counters {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }

        Ok(())
    }

    fn remove_blocks_impl(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        worker: WorkerWithDpRank,
        seq_hashes: &Vec<ExternalSequenceBlockHash>,
        event_id: u64,
    ) -> Result<(), KvCacheEventError> {
        let worker_map = worker_blocks.get_mut(&worker).ok_or_else(|| {
            tracing::warn!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                event_id,
                block_hashes = ?seq_hashes,
                "Failed to find worker blocks to remove"
            );
            KvCacheEventError::BlockNotFound
        })?;

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

            if let Some(mut entry) = self.index.get_mut(&(position, local_hash)) {
                let _ = entry.remove(*seq_hash, worker);
            }
        }

        Ok(())
    }

    /// Clear all blocks for a specific worker_id (all dp_ranks), but keep worker tracked.
    /// Static version for use in worker threads.
    fn clear_worker_blocks_impl(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
    ) {
        self.remove_or_clear_worker_blocks_impl(worker_blocks, worker_id, true);
    }

    fn remove_worker_dp_rank_impl(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) {
        let key = WorkerWithDpRank { worker_id, dp_rank };
        if let Some(worker_map) = worker_blocks.remove(&key) {
            for (seq_hash, (position, local_hash)) in worker_map.iter() {
                if let Some(mut entry) = self.index.get_mut(&(*position, *local_hash)) {
                    let _ = entry.remove(*seq_hash, key);
                }
            }
        }
    }

    /// Helper function to remove or clear blocks for a worker.
    /// If `keep_worker` is true, the worker remains tracked with empty blocks.
    /// If `keep_worker` is false, the worker is completely removed.
    fn remove_or_clear_worker_blocks_impl(
        &self,
        worker_blocks: &mut FxHashMap<WorkerWithDpRank, LevelIndex>,
        worker_id: WorkerId,
        keep_worker: bool,
    ) {
        let workers: Vec<WorkerWithDpRank> = worker_blocks
            .iter()
            .filter(|entry| entry.0.worker_id == worker_id)
            .map(|entry| *entry.0)
            .collect();

        for worker in workers {
            if let Some(worker_map) = worker_blocks.remove(&worker) {
                for (seq_hash, (position, local_hash)) in worker_map.iter() {
                    if let Some(mut entry) = self.index.get_mut(&(*position, *local_hash)) {
                        let _ = entry.remove(*seq_hash, worker);
                    }
                }
            }

            if keep_worker {
                // Re-insert worker with empty map to keep it tracked
                worker_blocks.insert(worker, FxHashMap::default());
            }
        }
    }

    fn dump_events(
        &self,
        worker_blocks: &FxHashMap<WorkerWithDpRank, LevelIndex>,
    ) -> Vec<RouterEvent> {
        let mut events = Vec::new();
        let mut event_id = 0u64;

        for (worker, worker_map) in worker_blocks.iter() {
            // Collect (position, local_hash, seq_hash) and sort by position.
            let mut blocks: Vec<_> = worker_map
                .iter()
                .map(|(seq_hash, (pos, local_hash))| (*pos, *local_hash, *seq_hash))
                .collect();
            blocks.sort_unstable_by_key(|(pos, _, _)| *pos);

            for (pos, local_hash, seq_hash) in blocks {
                events.push(RouterEvent {
                    worker_id: worker.worker_id,
                    storage_tier: crate::protocols::StorageTier::Device,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: Some(pos as u32),
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash: seq_hash,
                                tokens_hash: local_hash,
                                mm_extra_info: None,
                            }],
                        }),
                        dp_rank: worker.dp_rank,
                    },
                });
                event_id += 1;
            }
        }

        events
    }
}

// -----------------------------------------------------------------------------
// Jump-based search methods (associated functions for use in worker threads)
// -----------------------------------------------------------------------------

impl PositionalIndexer {
    /// Compute sequence hash incrementally from previous hash and current local hash.
    /// Delegates to [`dynamo_tokens::compute_next_sequence_hash`] so the request-side
    /// chain agrees with whatever produced the event stream.
    #[inline]
    fn compute_next_seq_hash(prev_seq_hash: u64, current_local_hash: u64) -> u64 {
        dynamo_tokens::compute_next_sequence_hash(prev_seq_hash, current_local_hash)
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
    ) -> Option<FxHashSet<WorkerWithDpRank>> {
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
    #[expect(clippy::too_many_arguments)]
    fn linear_scan_drain(
        &self,
        sequence: &[LocalBlockHash],
        seq_hashes: &mut Vec<ExternalSequenceBlockHash>,
        active: &mut FxHashSet<WorkerWithDpRank>,
        scores: &mut OverlapScores,
        lo: usize,
        hi: usize,
        early_exit: bool,
    ) {
        if active.is_empty() {
            return;
        }
        for pos in lo..hi {
            if active.is_empty() {
                break;
            }

            let Some(entry) = self.index.get(&(pos, sequence[pos])) else {
                for worker in active.drain() {
                    scores.scores.insert(worker, pos as u32);
                }
                break;
            };

            Self::ensure_seq_hash_computed(seq_hashes, pos, sequence);
            let Some(workers) = entry.get(seq_hashes[pos]) else {
                for worker in active.drain() {
                    scores.scores.insert(worker, pos as u32);
                }
                break;
            };

            if workers.len() != active.len() {
                reconcile_active_workers(active, workers, |worker| {
                    scores.scores.insert(worker, pos as u32);
                });
            }

            if early_exit && !active.is_empty() {
                break;
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
    ///
    /// # Arguments
    /// * `index` - The position -> local_hash -> SeqEntry index
    /// * `worker_blocks` - Per-worker reverse lookup for event removals
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
        let mut seq_hashes: Vec<ExternalSequenceBlockHash> = Vec::with_capacity(local_hashes.len());

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

        scores
    }

    /// Binary-search variant of [`Self::jump_search_matches`].
    ///
    /// Produces output identical to the strided search for any query whose matching-worker set is
    /// monotonically non-increasing with position (contiguous-from-zero stores and tail removals),
    /// but the number of index probes scales with the number of distinct drain points
    /// (`O(W · log L)`, `W` = workers at position 0) rather than `O(len / jump_size)`.
    ///
    /// # Preconditions
    ///
    /// Both this and [`Self::jump_search_matches`] use a count-only predicate
    /// (`count_workers_at(p) == active.len()`) that assumes `workers(p) ⊆ active` for `p` past the
    /// frontier. Sparse absolute-position stores (`start_position` writes, e.g. front eviction or a
    /// partial snapshot restore that leaves a worker at a later position without its prefix) can
    /// break that subset property. Because this search probes far fewer positions than the strided
    /// scan (notably the `len - 1` fast path below), a non-active worker that only exists late in
    /// the sequence can short-circuit it to a full match where the strided scan would still drain
    /// an active worker earlier — so the two can diverge in that regime. The bench/test event
    /// streams that drive [`PositionalIndexer`] today are all contiguous, so the property holds.
    ///
    /// # Algorithm
    ///
    /// The set of workers still matching the query is monotonically non-increasing as position
    /// grows (a worker can leave the `active` set but never rejoin it), so
    /// `P(p) := (workers matching at p) == active` is a monotone-decreasing predicate. It is
    /// checked cheaply as `count_workers_at(p) == active.len()`, which is sound because
    /// `workers(p) ⊆ active` for any position past the current frontier (count equality therefore
    /// implies set equality).
    ///
    /// Each round binary-searches for the largest position where the whole `active` set still
    /// matches; the next position is where one or more workers drain. Because every active worker
    /// matched at that last-true position, every worker missing one step later drained exactly
    /// there. Those workers are scored, the survivors become the new `active`, and the search
    /// continues rightward from there. Each round drops at least one worker, so the loop runs at
    /// most once per worker.
    ///
    /// For tight windows (`<= jump_size`) it defers to [`Self::linear_scan_drain`], which is
    /// already proven and records the same exact drain positions, bounding the worst case while
    /// reusing validated code.
    fn binary_search_matches(
        &self,
        local_hashes: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();

        if local_hashes.is_empty() {
            return scores;
        }

        // Lazily computed sequence hashes. `ensure_seq_hash_computed` only ever extends this
        // vector from its current length up to the requested position, so probing positions out
        // of order (as binary search does) is safe and deterministic.
        let mut seq_hashes: Vec<ExternalSequenceBlockHash> = Vec::with_capacity(local_hashes.len());

        // Check first position to initialize active set.
        let Some(mut active) =
            self.get_workers_lazy(0, local_hashes[0], &mut seq_hashes, local_hashes)
        else {
            return scores;
        };

        if active.is_empty() {
            return scores;
        }

        if early_exit {
            // Mirror jump_search_matches exactly: record that these workers matched position 0.
            for worker in &active {
                scores.scores.insert(*worker, 1);
            }
            return scores;
        }

        let len = local_hashes.len();

        // Invariant: every worker in `active` matches all positions through `frontier`, and
        // `active == workers(frontier)`, so `P(frontier)` is always true at the loop head.
        let mut frontier = 0;

        while frontier < len - 1 && !active.is_empty() {
            // Fast path: do all active workers survive to the final position? If so, they all
            // matched through the end and are scored `len` by the final loop below.
            if self
                .count_workers_at(
                    len - 1,
                    local_hashes[len - 1],
                    &mut seq_hashes,
                    local_hashes,
                )
                .unwrap_or(0)
                == active.len()
            {
                break;
            }

            // P(frontier) is true and P(len - 1) is false. Find the last-true boundary.
            let mut lo = frontier;
            let mut hi = len - 1;

            // Base case: once the window is small, defer to the proven linear scan over
            // (lo, hi]. It records exact drain positions (matching the bisection path) and
            // leaves the survivors in `active`. Since `hi == len - 1` here, this resolves the
            // remaining suffix and the outer loop then terminates.
            if hi - lo <= self.jump_size {
                self.linear_scan_drain(
                    local_hashes,
                    &mut seq_hashes,
                    &mut active,
                    &mut scores,
                    lo + 1,
                    hi + 1,
                    false,
                );
                frontier = hi;
                continue;
            }

            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self
                    .count_workers_at(mid, local_hashes[mid], &mut seq_hashes, local_hashes)
                    .unwrap_or(0)
                    == active.len()
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            // `hi` is the first position where at least one active worker drains. Because every
            // active worker matched at `lo == hi - 1`, each worker missing at `hi` drained exactly
            // here. Reconcile `active` against workers(hi) IN PLACE (no clone): cloning the
            // survivor set here is a per-drain heap allocation that scales with fan-out (workers
            // sharing the prefix) and becomes an allocator bottleneck under high concurrency.
            // A missing index entry (or a seq_hash mismatch) means every active worker drains here.
            let drain_pos = hi;
            Self::ensure_seq_hash_computed(&mut seq_hashes, drain_pos, local_hashes);
            let drain_seq_hash = seq_hashes[drain_pos];

            match self
                .index
                .get(&(drain_pos, local_hashes[drain_pos]))
                .as_ref()
                .and_then(|entry| entry.get(drain_seq_hash))
            {
                Some(workers) => {
                    reconcile_active_workers(&mut active, workers, |worker| {
                        scores.scores.insert(worker, drain_pos as u32);
                    });
                }
                None => {
                    for worker in active.drain() {
                        scores.scores.insert(worker, drain_pos as u32);
                    }
                }
            }

            frontier = drain_pos;
        }

        // Record final scores for remaining active workers; they matched through the end.
        let final_score = len as u32;
        for worker in active {
            scores.scores.insert(worker, final_score);
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::{LevelIndex, PositionalIndexer, SearchMode};
    use crate::protocols::{LocalBlockHash, RouterEvent, WorkerWithDpRank};
    use crate::test_utils::{assert_overlap_scores_eq, make_store_event};
    use rustc_hash::FxHashMap;

    fn local_hashes(hashes: &[u64]) -> Vec<LocalBlockHash> {
        hashes.iter().copied().map(LocalBlockHash).collect()
    }

    /// Populate an indexer's shared index by applying events against a local `worker_blocks`
    /// map, exactly as the worker thread does.
    fn populate(indexer: &PositionalIndexer, events: &[RouterEvent]) {
        let mut worker_blocks: FxHashMap<WorkerWithDpRank, LevelIndex> = FxHashMap::default();
        for ev in events {
            indexer
                .apply_event(&mut worker_blocks, ev.clone(), None)
                .expect("apply_event should succeed");
        }
    }

    /// The strided and binary searches must produce identical results for both `early_exit`
    /// settings. The public async surface always passes `early_exit = false`, so the
    /// `early_exit = true` path is exercised here by calling the two methods directly.
    #[test]
    fn binary_matches_strided_both_early_exit_settings() {
        // jump_size = 8 keeps sequences long enough relative to the stride to exercise both the
        // bisection and the linear base case; the search-mode field is irrelevant because we call
        // the two algorithms directly rather than via find_matches dispatch.
        let indexer = PositionalIndexer::new_with_mode(8, SearchMode::Strided);

        populate(
            &indexer,
            &[
                make_store_event(0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                make_store_event(1, &[1, 2, 3, 99, 100]),
                make_store_event(2, &[1, 2, 3, 4, 5, 6, 7, 8, 200, 201]),
                make_store_event(3, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            ],
        );

        let queries: &[&[u64]] = &[
            &[],
            &[1],
            &[42], // miss
            &[1, 2, 3],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[1, 2, 3, 99, 100],
            &[1, 2, 3, 4, 5, 6, 7, 8, 200, 201],
        ];

        for q in queries {
            let seq = local_hashes(q);
            for early_exit in [false, true] {
                let strided = indexer.jump_search_matches(&seq, early_exit);
                let binary = indexer.binary_search_matches(&seq, early_exit);
                assert_overlap_scores_eq(
                    &strided,
                    &binary,
                    &format!("q={q:?} early_exit={early_exit}"),
                );
            }
        }
    }
}
