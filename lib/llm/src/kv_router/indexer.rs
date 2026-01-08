// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV RadixTree
//!
//! This module implements a key-value (KV) store using a Radix Tree structure to efficiently manage and retrieve data blocks.
//! It is designed to support LLM (Large Language Model) inference by re-using a global KV cache.
//!
//! # Overview
//!
//! The main components of this module include:
//!
//! - **Radix Tree Structure**:
//!   - The `RadixTree` struct represents the main data structure, with nodes (`RadixBlock`) containing children and associated worker IDs.
//!   - It allows efficient storage and retrieval of data blocks based on their hashes.
//!
//! - **Event Handling**:
//!   - The `RouterEvent` struct represents events emitted by LLM workers, which can be applied to the Radix Tree to update its state.
//!   - The `KvIndexer` struct manages these events and match requests asynchronously using Tokio channels.
//!
//! - **Hash Computation**:
//!   - Functions like `compute_block_hash` and `compute_block_hash_for_seq` compute hashes for data blocks and sequences of tokens, facilitating quick lookups.
//!
//! - **Concurrency and Asynchronous Operations**:
//!   - The `KvIndexer` uses a single-threaded Tokio runtime to handle events and match requests concurrently, ensuring efficient processing without blocking.
//!
//! - **Match Requests**:
//!   - The `MatchRequest` struct represents requests to find matches in the Radix Tree, returning overlap scores indicating the best matches.
//!
//! # Purpose
//!
//! This module provides a scalable and efficient way to manage and retrieve data blocks for LLM inference, leveraging a global KV cache to optimize performance.

use async_trait::async_trait;
use dynamo_runtime::{
    component::Component,
    metrics::{MetricsHierarchy, prometheus_names::kvrouter},
};
use prometheus::{IntCounterVec, Opts};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    iter,
    rc::Rc,
    sync::{Arc, Mutex, OnceLock},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kv_router::approx::{BlockEntry, PruneConfig, PruneManager};
use crate::kv_router::protocols::*;
use crate::tokens::SequenceHash;

/// Errors that can occur in the KV Router.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer is dropped request")]
    IndexerDroppedRequest,

    #[error("Prune operation failed: {0}")]
    PruneFailed(String),
}

/// Errors that can occur during KV Cache Event processing.
#[derive(Debug, thiserror::Error)]
pub enum KvCacheEventError {
    #[error("Failed to find parent block")]
    ParentBlockNotFound,

    #[error("Failed to find block")]
    BlockNotFound,

    #[error("Invalid block sequence")]
    InvalidBlockSequence,
}

/// A shared reference to a [`RadixBlock`].
type SharedRadixBlock = Rc<RefCell<RadixBlock>>;

/// A [`KvCacheEvent`] on a specific LLM worker denoted by [`WorkerId`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterEvent {
    /// The ID of the worker emitting the event.
    pub worker_id: WorkerId,
    /// The cache event associated with the worker.
    pub event: KvCacheEvent,
}

impl RouterEvent {
    /// Create a new `RouterEvent`.
    ///
    /// ### Arguments
    ///
    /// * `worker_id` - The ID of the worker emitting the event.
    /// * `event` - The cache event.
    ///
    /// ### Returns
    ///
    /// A new `RouterEvent`.
    pub fn new(worker_id: WorkerId, event: KvCacheEvent) -> Self {
        Self { worker_id, event }
    }
}

// -------
// Distributed router - Worker KV Query types
// -------

/// Request to query a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerKvQueryRequest {
    /// The worker ID of the worker to query.
    pub worker_id: WorkerId,

    /// Start event ID (inclusive). If `None`, dumps entire tree.
    pub start_event_id: Option<u64>,
    /// End event ID (inclusive). If `None`, returns up to newest available.
    pub end_event_id: Option<u64>,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerKvQueryResponse {
    /// Events served from the circular buffer (with original event IDs)
    Events(Vec<RouterEvent>),
    /// Full tree dump (with synthetic 0-indexed event IDs)
    TreeDump(Vec<RouterEvent>),
    /// Requested range is newer than available data
    TooNew {
        requested_start: Option<u64>,
        requested_end: Option<u64>,
        newest_available: u64,
    },
    /// Invalid range: end_id < start_id
    InvalidRange { start_id: u64, end_id: u64 },
}

/// A block in the Radix Tree.
#[derive(Debug)]
struct RadixBlock {
    /// A map of child blocks, keyed by their local block hash.
    children: HashMap<LocalBlockHash, SharedRadixBlock>,
    /// A map of workers (with dp_rank) to their external sequence block hash for this block.
    /// The external hash is preserved to speed up snapshotting.
    workers: HashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    /// A buffer of times that this block was last traversed
    recent_uses: VecDeque<Instant>,
}

impl RadixBlock {
    /// Create a new `RadixBlock`.
    ///
    /// ### Returns
    ///
    /// A new `RadixBlock`.
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            workers: HashMap::new(),
            recent_uses: VecDeque::new(),
        }
    }
}

pub struct RadixTree {
    /// This is the root of the radix/prefix tree
    /// This will only contain root blocks
    root: SharedRadixBlock,

    /// This is a global lookup table for all blocks which will let you jump into
    /// the radix tree at any point
    /// Lookup is best case O(1) and worst case O(N); however, even constant in-time
    /// could be expensive if N is large
    /// We should monitor the size of this table and consider using a proper radix tree.
    /// Transitioning to a radix tree only would require a change in the messaging structure
    /// as the entire prefix would need to be sent. Alternatively, we could use block_depth
    /// integers to indicate how many blocks to skip and use a radix/prefix tree at each level.
    lookup: HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, SharedRadixBlock>>,
    /// The time buffer the radix tree should check when considering frequence of block accesses
    expiration_duration: Option<Duration>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
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
                scores.update_scores(block.borrow().workers.keys());

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
                // find the parent block - if the parent exists it must be on our worker, if not,
                // we check the radix tree's root to find it.
                // this is the single most expensive lookup
                let current = match op.parent_hash {
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

                fn process_blocks(
                    parent: SharedRadixBlock,
                    blocks: &[KvCacheStoredBlockData],
                    worker: WorkerWithDpRank,
                    worker_lookup: &mut HashMap<ExternalSequenceBlockHash, SharedRadixBlock>,
                    id: u64,
                ) -> Result<(), KvCacheEventError> {
                    if blocks.is_empty() {
                        return Ok(());
                    }

                    let mut parent_mut = parent.borrow_mut();
                    let block_data = &blocks[0];

                    let child = match parent_mut.children.get(&block_data.tokens_hash) {
                        Some(block) => block.clone(),
                        None => {
                            // create new block - automatically added to the lookup table
                            let new_block = worker_lookup
                                .get(&block_data.block_hash)
                                .cloned()
                                .unwrap_or_else(|| Rc::new(RefCell::new(RadixBlock::new())));

                            // insert into radix tree
                            parent_mut
                                .children
                                .insert(block_data.tokens_hash, new_block.clone());

                            new_block
                        }
                    };

                    // Update child and check for cycles
                    {
                        // Try to borrow the child mutably - if it fails, it's already borrowed
                        // in the ancestor chain (parent_mut is alive + all ancestors in recursive stack)
                        let mut child_mut = match child.try_borrow_mut() {
                            Ok(b) => b,
                            Err(_) => {
                                tracing::warn!(
                                    worker_id = worker.worker_id.to_string(),
                                    dp_rank = worker.dp_rank,
                                    id,
                                    block_hash = ?block_data.block_hash,
                                    "Detected cycle in store event (block already in parent chain); rejecting sequence"
                                );
                                return Err(KvCacheEventError::InvalidBlockSequence);
                            }
                        };

                        // add our worker to the block with its external hash
                        child_mut.workers.insert(worker, block_data.block_hash);
                    }

                    // add the block to the worker_id lookup table
                    worker_lookup.insert(block_data.block_hash, child.clone());

                    // Recurse with the child and remaining blocks
                    process_blocks(child, &blocks[1..], worker, worker_lookup, id)
                }

                process_blocks(current, &op.blocks, worker, worker_lookup, id)
            }
            KvCacheEventData::Removed(remove) => {
                // tracing::trace!(id, "KV Remove Operation: {:?}", op);
                // let mut worker_lookup = self.lookup.get(&worker_id).expect("Worker not found");

                for block in remove.block_hashes {
                    // entry in radix tree
                    // a small optimization would be to get the next block from the reduced set of children
                    // in order to apply this optimization, we would need to know the list of blocks is always sorted
                    // by parent -> child relationship
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
                            return Err(KvCacheEventError::BlockNotFound);
                        }
                    };

                    let mut guard = entry.borrow_mut();
                    guard.workers.remove(&worker);
                    if guard.workers.is_empty() {
                        // if no workers are using this block, that is true for all children
                        guard.children.clear();
                    }
                    // remove the block from the lookup table
                    worker_lookup.remove(&block);
                }
                Ok(())
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
                blocks.iter().for_each(|(_, block)| {
                    block.borrow_mut().workers.remove(&worker);
                    // If no workers are using this block, that is true for all children
                    if block.borrow().workers.is_empty() {
                        block.borrow_mut().children.clear();
                    }
                });

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

        // BFS queue: (current_block, parent_hashes_per_worker, tokens_hash)
        // parent_hashes_per_worker maps WorkerWithDpRank -> ExternalSequenceBlockHash
        let mut queue: VecDeque<(
            SharedRadixBlock,
            HashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
            LocalBlockHash,
        )> = VecDeque::new();

        // Process root's children first
        let root_borrow = self.root.borrow();
        for (tokens_hash, child_block) in &root_borrow.children {
            queue.push_back((child_block.clone(), HashMap::new(), *tokens_hash));
        }
        drop(root_borrow);

        while let Some((current_block, parent_hashes, tokens_hash)) = queue.pop_front() {
            let current_borrow = current_block.borrow();

            // Map of this block's external hashes per worker (for children to use as parent)
            let mut current_external_hashes = HashMap::new();

            // For each worker that has this block
            for (worker_id, external_hash) in &current_borrow.workers {
                // Get the correct parent hash for this worker
                let parent_hash = parent_hashes.get(worker_id).copied();

                // Create a store event for this worker
                let event = RouterEvent {
                    worker_id: worker_id.worker_id,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash: *external_hash,
                                mm_extra_info: None,
                                tokens_hash,
                            }],
                        }),
                        dp_rank: worker_id.dp_rank,
                    },
                };
                events.push(event);
                event_id += 1;

                // Track this block's external hash for this worker
                current_external_hashes.insert(*worker_id, *external_hash);
            }

            // Enqueue children with per-worker parent hashes
            for (child_tokens_hash, child_block) in &current_borrow.children {
                queue.push_back((
                    child_block.clone(),
                    current_external_hashes.clone(),
                    *child_tokens_hash,
                ));
            }
        }

        events
    }

    pub fn current_size(&self) -> usize {
        self.lookup.values().map(|m| m.len()).sum()
    }
}

/// Metrics for the KV Indexer.
#[derive(Clone)]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    pub kv_cache_events_applied: IntCounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    fn new(kv_cache_events_applied: IntCounterVec) -> Self {
        Self {
            kv_cache_events_applied,
        }
    }

    /// Creates a new KvIndexerMetrics from a Component, memoizing the result in
    /// KV_INDEXER_METRICS to avoid duplicate registration issues.
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_INDEXER_METRICS.get_or_init(|| {
            match component.metrics().create_intcountervec(
                kvrouter::KV_CACHE_EVENTS_APPLIED,
                "Total number of KV cache events applied to index",
                &["event_type", "status"],
                &[],
            ) {
                Ok(kv_cache_events_applied) => Arc::new(Self::new(kv_cache_events_applied)),
                Err(e) => {
                    tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                    Arc::new(Self::new_unregistered())
                }
            }
        }).clone()
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    pub fn new_unregistered() -> Self {
        Self {
            kv_cache_events_applied: IntCounterVec::new(
                Opts::new(
                    kvrouter::KV_CACHE_EVENTS_APPLIED,
                    "Total number of KV cache events applied to index",
                ),
                &["event_type", "status"],
            )
            .unwrap(),
        }
    }

    pub fn get_event_type(event_data: &KvCacheEventData) -> &'static str {
        match event_data {
            KvCacheEventData::Stored(_) => METRIC_EVENT_STORED,
            KvCacheEventData::Removed(_) => METRIC_EVENT_REMOVED,
            KvCacheEventData::Cleared => METRIC_EVENT_CLEARED,
        }
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        match result {
            Ok(_) => {
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, METRIC_STATUS_OK])
                    .inc_by(1);
            }
            Err(e) => {
                let error_label = match e {
                    KvCacheEventError::ParentBlockNotFound => METRIC_STATUS_PARENT_NOT_FOUND,
                    KvCacheEventError::BlockNotFound => METRIC_STATUS_BLOCK_NOT_FOUND,
                    KvCacheEventError::InvalidBlockSequence => METRIC_STATUS_INVALID_BLOCK,
                };
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, error_label])
                    .inc_by(1);
            }
        }
    }
}

/// Scores representing the overlap of workers (with their dp_rank).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapScores {
    // map of worker (with dp_rank) to score
    pub scores: HashMap<WorkerWithDpRank, u32>,
    // List of frequencies that the blocks have been accessed. Entries with value 0 are omitted.
    pub frequencies: Vec<usize>,
    // Map of worker to their tree size (number of blocks in the tree for that worker)
    pub tree_sizes: HashMap<WorkerWithDpRank, usize>,
}

impl Default for OverlapScores {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapScores {
    /// Create a new `OverlapScores`.
    ///
    /// ### Returns
    ///
    /// A new `OverlapScores`.
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            frequencies: Vec::with_capacity(32),
            tree_sizes: HashMap::new(),
        }
    }

    /// Update the scores with a set of workers.
    ///
    /// ### Arguments
    ///
    /// * `workers` - An iterator over `WorkerWithDpRank` references.
    pub fn update_scores<'a, I>(&mut self, workers: I)
    where
        I: IntoIterator<Item = &'a WorkerWithDpRank>,
    {
        for worker in workers {
            let score = self.scores.entry(*worker).or_insert(0);
            *score += 1;
        }
    }

    /// Add an entry in the frequency list.
    pub fn add_frequency(&mut self, frequency: usize) {
        if frequency != 0 {
            self.frequencies
                .last()
                .inspect(|elem| debug_assert!(**elem >= frequency));
            self.frequencies.push(frequency);
        }
    }
}

/// A request to find matches in the Radix Tree.
pub struct MatchRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    early_exit: bool,
    /// A channel sender to send the `OverlapScores` response.
    resp: oneshot::Sender<OverlapScores>,
}

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

#[async_trait]
pub trait KvIndexerInterface {
    /// Find matches for a given sequence of `LocalBlockHash`es.
    ///
    /// ### Arguments
    ///
    /// * `sequence` - A vector of `LocalBlockHash` representing the sequence to match.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError>;

    /// Find matches for a given sequence of tokens.
    ///
    /// ### Arguments
    ///
    /// * `tokens` - A vector of `u32` tokens.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError>;

    /// Apply a `RouterEvent` to the KV store.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    async fn apply_event(&mut self, event: RouterEvent);

    /// Remove a worker's entries from the trie.
    ///
    /// ### Arguments
    ///
    /// * `worker` - The worker to remove from the trie.
    async fn remove_worker(&mut self, worker: WorkerId);

    /// Shutdown the KV Indexer.
    fn shutdown(&mut self);

    /// Dump the entire tree as RouterEvents.
    ///
    /// ### Returns
    ///
    /// A vector of RouterEvents representing the current state of the tree.
    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError>;

    /// Process a routing decision for a request with tokens.
    ///
    /// Uses TokensWithHashes for lazy hash computation - if hashes were already
    /// computed (e.g., by find_best_match), they will be reused.
    ///
    /// ### Arguments
    ///
    /// * `tokens_with_hashes` - Tokens with lazily computed hashes.
    /// * `worker` - The worker (with dp_rank) that was selected.
    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError>;
}

/// A request to process a routing decision.
struct RoutingDecisionRequest {
    worker: WorkerWithDpRank,
    local_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
}

/// The KV Indexer, managing the KV store and handling events and match requests.
#[derive(Clone)]
pub struct KvIndexer {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// A sender for `RouterEvent`s.
    event_tx: mpsc::Sender<RouterEvent>,
    /// A sender for `MatchRequest`s.
    match_tx: mpsc::Sender<MatchRequest>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for get workers requests.
    get_workers_tx: mpsc::Sender<GetWorkersRequest>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A sender for routing decision requests.
    routing_tx: mpsc::Sender<RoutingDecisionRequest>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    /// Reference counter for Clone-aware Drop.
    /// Only the last clone should cancel the token on drop.
    _ref_count: Arc<()>,
}

impl KvIndexer {
    /// Create a new `KvIndexer`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(2048);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (get_workers_tx, get_workers_rx) = mpsc::channel::<GetWorkersRequest>(16);
        let (dump_tx, dump_rx) = mpsc::channel::<DumpRequest>(16);
        let (routing_tx, mut routing_rx) = mpsc::channel::<RoutingDecisionRequest>(2048);
        let (prune_tx, mut prune_rx) = mpsc::channel::<()>(1);

        let cancel_clone = token.clone();

        std::thread::spawn(move || {
            // Create a single-threaded tokio runtime
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let cancel = cancel_clone;
                let mut match_rx = match_rx;
                let mut event_rx = event_rx;
                let mut remove_worker_rx = remove_worker_rx;
                let mut get_workers_rx = get_workers_rx;
                let mut dump_rx = dump_rx;
                let mut trie = RadixTree::new_with_frequency(expiration_duration);

                // Create PruneManager if prune_config is specified
                let mut prune_manager = prune_config.map(|config| {
                    PruneManager::<BlockEntry>::new(50, config)
                });
                let mut event_id_counter = 0u64;

                loop {
                    // Create a future that sleeps until the next expiration time
                    let expiry_fut = if let Some(ref pm) = prune_manager
                        && let Some(next_expiry) = pm.peek_next_expiry() {
                        tokio::time::sleep_until(next_expiry)
                    } else {
                        tokio::time::sleep(Duration::MAX)
                    };

                    tokio::select! {
                        biased;

                        _ = cancel.cancelled() => {
                            tracing::debug!("KvCacheIndexer progress loop shutting down");
                            return;
                        }

                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                        }

                        Some(get_workers_req) = get_workers_rx.recv() => {
                            let workers = trie.get_workers();
                            let _ = get_workers_req.resp.send(workers);
                        }

                        Some(_) = prune_rx.recv() => {
                            // Tree size-based pruning triggered
                            let Some(ref mut pm) = prune_manager else { continue };
                            let Ok(pruned) = pm.prune(trie.current_size()) else { continue };

                            for p in pruned {
                                event_id_counter += 1;
                                let event = RouterEvent::new(
                                    p.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![p.key],
                                        }),
                                        dp_rank: p.worker.dp_rank,
                                    }
                                );
                                let _ = trie.apply_event(event);
                            }
                        }

                        Some(event) = event_rx.recv() => {
                            let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                            let result = trie.apply_event(event.clone());
                            let result_is_ok = result.is_ok();
                            metrics.increment_event_applied(event_type, result);

                            // Track blocks in PruneManager if TTL is enabled and event was stored successfully
                            let Some(ref mut pm) = prune_manager else { continue };
                            if !result_is_ok { continue };
                            let KvCacheEventData::Stored(ref store_data) = event.event.data else { continue };

                            let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                            let block_entries: Vec<BlockEntry> = store_data.blocks.iter().enumerate().map(|(idx, block)| {
                                BlockEntry {
                                    key: block.block_hash,
                                    worker,
                                    seq_position: idx,
                                }
                            }).collect();
                            pm.insert(block_entries);

                            // Check if we need to prune due to tree size
                            let Some(ref pc) = pm.prune_config else { continue };
                            let current_size = trie.current_size();
                            if current_size > pc.max_tree_size {
                                tracing::info!(
                                    "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                    current_size,
                                    pc.max_tree_size
                                );
                                let _ = prune_tx.try_send(());
                            }
                        }

                        Some(dump_req) = dump_rx.recv() => {
                            let events = trie.dump_tree_as_events();
                            let _ = dump_req.resp.send(events);
                        }

                        Some(routing_req) = routing_rx.recv() => {
                            // Process routing decisions when TTL/pruning is enabled
                            let Some(ref mut pm) = prune_manager else { continue };

                            event_id_counter += 1;

                            let hashes = routing_req.local_hashes.iter().zip(routing_req.sequence_hashes.iter());
                            let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash: None,
                                blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                    tokens_hash: *local_hash,
                                    block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                mm_extra_info: None,
                                }).collect(),
                            });

                            let event = RouterEvent::new(
                                routing_req.worker.worker_id,
                                KvCacheEvent {
                                    event_id: event_id_counter,
                                    data: stored_event,
                                    dp_rank: routing_req.worker.dp_rank,
                                }
                            );

                            if trie.apply_event(event).is_err() {
                                continue;
                            }

                            let block_entries: Vec<BlockEntry> = routing_req.sequence_hashes.iter().enumerate().map(|(idx, h)| {
                                BlockEntry {
                                    key: ExternalSequenceBlockHash(*h),
                                    worker: routing_req.worker,
                                    seq_position: idx,
                                }
                            }).collect();
                            pm.insert(block_entries);

                            // Check if we need to prune due to tree size
                            let Some(ref pc) = pm.prune_config else { continue };
                            let current_size = trie.current_size();
                            if current_size > pc.max_tree_size {
                                tracing::info!(
                                    "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                    current_size,
                                    pc.max_tree_size
                                );
                                let _ = prune_tx.try_send(());
                            }
                        }

                        Some(req) = match_rx.recv() => {
                            let matches = trie.find_matches(req.sequence, req.early_exit);
                            let _ = req.resp.send(matches);
                        }

                        _ = expiry_fut => {
                            // TTL-based expiry triggered
                            let Some(ref mut pm) = prune_manager else { continue };

                            let expired = pm.pop_expired();
                            for e in expired {
                                event_id_counter += 1;
                                let event = RouterEvent::new(
                                    e.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![e.key],
                                        }),
                                        dp_rank: e.worker.dp_rank,
                                    }
                                );
                                let _ = trie.apply_event(event);
                            }
                        }
                    }
                }
            });

            tracing::debug!("KvCacheIndexer task completed");
        });

        Self {
            cancel: token,
            event_tx,
            match_tx,
            remove_worker_tx,
            get_workers_tx,
            dump_tx,
            routing_tx,
            kv_block_size,
            _ref_count: Arc::new(()),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, None, kv_block_size, metrics, None)
    }

    /// Get a sender for `RouterEvent`s.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.event_tx.clone()
    }

    /// Get a sender for dump requests (snapshot events).
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `DumpRequest`s.
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.dump_tx.clone()
    }

    /// Get a sender for worker removal requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `WorkerId`s.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.remove_worker_tx.clone()
    }

    /// Get a sender for get workers requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `GetWorkersRequest`s.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.get_workers_tx.clone()
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexer {
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

        if let Err(e) = self.match_tx.send(req).await {
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
        tracing::debug!(
            "Finding matches for request tokens: {:?} / len: {}",
            tokens,
            tokens.len()
        );
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
        tracing::debug!("Computed sequence: {:?}", sequence);
        self.find_matches(sequence).await
    }

    async fn apply_event(&mut self, event: RouterEvent) {
        self.event_tx.send(event).await.unwrap();
    }

    async fn remove_worker(&mut self, worker: WorkerId) {
        self.remove_worker_tx.send(worker).await.unwrap();
    }

    fn shutdown(&mut self) {
        self.cancel.cancel();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let dump_req = DumpRequest { resp: resp_tx };

        if let Err(e) = self.dump_tx.send(dump_req).await {
            tracing::error!("Failed to send dump request: {:?}", e);
            return Err(KvRouterError::IndexerOffline);
        }

        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_internal(worker, local_hashes, sequence_hashes)
            .await
    }
}

impl KvIndexer {
    /// Internal method to process a routing decision with pre-computed hashes.
    async fn process_routing_decision_internal(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.routing_tx
            .send(RoutingDecisionRequest {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        Ok(())
    }
}

impl Drop for KvIndexer {
    fn drop(&mut self) {
        // Only cancel the token if we're the last reference.
        // This allows clones to be dropped without killing the background task.
        if Arc::strong_count(&self._ref_count) == 1 {
            self.shutdown();
        }
    }
}

// -------------------------------------------------
// Decentralized router: LocalKvIndexer for workers
// -------------------------------------------------

/// A thin wrapper around KvIndexer that buffers recent events
/// (e.g. which may be queued by router upon startup)
///
pub struct LocalKvIndexer {
    /// The underlying indexer
    indexer: KvIndexer,
    /// Circular buffer of recent events
    event_buffer: Mutex<VecDeque<RouterEvent>>,
    /// Maximum number of events to keep in buffer
    max_buffer_size: usize, // Router sets this to WORKER_KV_INDEXER_BUFFER_SIZE
}

impl LocalKvIndexer {
    /// create a new LocalKvIndexer pointing to a KvIndexer.
    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            indexer: KvIndexer::new(token, kv_block_size, metrics),
            event_buffer: Mutex::new(VecDeque::with_capacity(max_buffer_size)),
            max_buffer_size,
        }
    }

    /// Get all buffered events (oldest first).
    pub fn get_all_events_in_buffer(&self) -> Vec<RouterEvent> {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.iter().cloned().collect()
    }

    /// Query events by ID range, returning events in `[start_id, end_id]` (both inclusive).
    ///
    /// ### Arguments
    ///
    /// * `start_id` - Starting event ID (inclusive). If `None`, dumps entire tree.
    /// * `end_id` - Ending event ID (inclusive). If `None`, returns up to newest available.
    ///
    /// ### Returns
    ///
    /// - `Events`: Buffered events with original IDs (when range is within buffer)
    /// - `TreeDump`: Full tree dump with synthetic IDs (when range is too old or unspecified)
    /// - `TooNew`: Error when requested range is newer than available data
    /// - `InvalidRange`: Error when end_id < start_id
    pub async fn get_events_in_id_range(
        &self,
        start_id: Option<u64>,
        end_id: Option<u64>,
    ) -> WorkerKvQueryResponse {
        // Validate range if both specified
        if let (Some(s), Some(e)) = (start_id, end_id)
            && e < s
        {
            tracing::warn!(start_id = s, end_id = e, "Invalid range: end_id < start_id");
            return WorkerKvQueryResponse::InvalidRange {
                start_id: s,
                end_id: e,
            };
        }

        // Get buffer state
        let (first_id, last_id) = {
            let buffer = self.event_buffer.lock().unwrap();
            if buffer.is_empty() {
                (None, None)
            } else {
                (
                    Some(buffer.front().unwrap().event.event_id),
                    Some(buffer.back().unwrap().event.event_id),
                )
            }
        };

        // If no start_id specified, dump entire tree
        if start_id.is_none() {
            tracing::debug!("No start_id specified, dumping entire tree");
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        }

        let start_id = start_id.unwrap();
        let end_id = end_id.unwrap_or_else(|| last_id.unwrap_or(start_id));

        // Check for empty buffer
        let Some(first_buffered) = first_id else {
            tracing::debug!("Buffer empty, dumping entire tree");
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        };
        let last_buffered = last_id.unwrap();

        // Check if request is too new
        if start_id > last_buffered {
            tracing::warn!(
                start_id,
                last_buffered,
                "Requested start_id is newer than buffer"
            );
            return WorkerKvQueryResponse::TooNew {
                requested_start: Some(start_id),
                requested_end: Some(end_id),
                newest_available: last_buffered,
            };
        }

        // Check if start_id is too old (before buffer) -> tree dump
        if start_id < first_buffered {
            tracing::info!(
                start_id,
                first_buffered,
                "Requested start_id is older than buffer, dumping entire tree"
            );
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        }

        // Serve from buffer
        let buffer = self.event_buffer.lock().unwrap();

        let start_idx = match buffer.binary_search_by_key(&start_id, |e| e.event.event_id) {
            Ok(idx) => idx,
            Err(insertion_point) => insertion_point,
        };

        // Clamp end_id to buffer bounds
        let clamped_end_id = end_id.min(last_buffered);
        let end_idx = match buffer.binary_search_by_key(&clamped_end_id, |e| e.event.event_id) {
            Ok(idx) => idx + 1, // Include the matched element
            Err(insertion_point) => insertion_point,
        };

        let events: Vec<RouterEvent> = buffer
            .iter()
            .skip(start_idx)
            .take(end_idx.saturating_sub(start_idx))
            .cloned()
            .collect();

        WorkerKvQueryResponse::Events(events)
    }

    /// Record an event in the buffer
    fn record_event(&self, event: RouterEvent) {
        let mut buffer = self.event_buffer.lock().unwrap();

        // Check that event id is consecutive to last one
        if let Some(last_event) = buffer.back()
            && event.event.event_id != last_event.event.event_id + 1
        {
            let expected = last_event.event.event_id + 1;
            tracing::error!(
                worker_id = event.worker_id,
                expected,
                got = event.event.event_id,
                "Non-consecutive KV event id; buffer may have gaps"
            );
        }
        tracing::debug!(
            "Recorded event {:?} in buffer, now size is {}",
            event,
            buffer.len()
        );

        // Add to back
        buffer.push_back(event);

        // Remove from front if over capacity (circular buffer behavior)
        while buffer.len() > self.max_buffer_size {
            buffer.pop_front();
        }
    }

    /// Apply event with buffering.
    ///
    /// This records the event in the buffer and forwards it to the underlying indexer.
    pub async fn apply_event_with_buffer(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        // Record in buffer
        self.record_event(event.clone());

        // Forward to underlying indexer
        self.indexer
            .event_sender()
            .send(event)
            .await
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    /// Clear the event buffer.
    pub fn clear_buffer(&self) {
        let mut buffer = self.event_buffer.lock().unwrap();
        buffer.clear();
    }

    /// Get the current buffer size.
    pub fn buffer_len(&self) -> usize {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.len()
    }

    // Delegation methods to underlying KvIndexer
    /// Get a sender for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.indexer.event_sender()
    }

    /// Get a sender for dump requests (snapshot events).
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.indexer.snapshot_event_sender()
    }

    /// Get a sender for worker removal requests.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.indexer.remove_worker_sender()
    }

    /// Get a sender for get workers requests.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.indexer.get_workers_sender()
    }

    /// Get the KV block size.
    pub fn block_size(&self) -> u32 {
        self.indexer.block_size()
    }
}

// Implement KvIndexerInterface by delegating to the underlying indexer
#[async_trait]
impl KvIndexerInterface for LocalKvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer.find_matches(sequence).await
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer.find_matches_for_request(tokens).await
    }

    async fn apply_event(&mut self, event: RouterEvent) {
        // Use the buffering version
        let _ = self.apply_event_with_buffer(event).await;
    }

    async fn remove_worker(&mut self, worker: WorkerId) {
        let _ = self.indexer.remove_worker_sender().send(worker).await;
    }

    fn shutdown(&mut self) {
        // Note: Since indexer is Arc<KvIndexer>, we can't call mutable methods directly.
        // The indexer will be shut down when the CancellationToken is cancelled
        // or when the last Arc reference is dropped.
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // TODO I guess the local kvindexers have little use for this method?
        // Keeping it here now to implement the trait fully
        self.indexer
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }
}

#[derive(Debug, Clone)]
pub struct ShardedMatchRequest {
    sequence: Vec<LocalBlockHash>,
    early_exit: bool,
    resp: mpsc::Sender<OverlapScores>,
}

/// A sharded KV Indexer that partitions the RadixTree across multiple independent shards.
///
/// ## Sharding Strategy
/// - Each worker is **permanently assigned** to a single shard on first event
/// - All KV blocks from a worker exist only in that worker's assigned shard
/// - New workers are assigned to the shard with the fewest workers (load balancing)
///
/// ## Operation
/// - **Events**: Routed directly to the worker's assigned shard
/// - **Match requests**: Broadcast to all shards (scatter-gather pattern)
/// - **Threading**: Each shard runs in its own thread with a single-threaded runtime
///
/// This design ensures no cross-shard synchronization for writes while enabling
/// parallel processing and better scalability.
pub struct KvIndexerSharded {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    worker_assignments: HashMap<WorkerId, usize>,
    worker_counts: Vec<usize>,

    event_tx: Vec<mpsc::Sender<RouterEvent>>,
    request_broadcast_tx: broadcast::Sender<ShardedMatchRequest>,
    remove_worker_tx: Vec<mpsc::Sender<WorkerId>>,
    dump_tx: Vec<mpsc::Sender<DumpRequest>>,
    routing_tx: Vec<mpsc::Sender<RoutingDecisionRequest>>,
    tasks: Vec<JoinHandle<()>>,
}

impl KvIndexerSharded {
    /// Create a new `KvIndexerSharded`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `shards` - A list of kvindexer shards.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        num_shards: usize,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let worker_assignments: HashMap<WorkerId, usize> = HashMap::new();
        let worker_counts: Vec<usize> = vec![0; num_shards];

        let mut event_tx = Vec::new();
        let mut remove_worker_tx = Vec::new();
        let mut get_workers_tx = Vec::new();
        let mut dump_tx = Vec::new();
        let mut routing_tx = Vec::new();
        let mut tasks = Vec::new();

        let (request_broadcast_tx, _) = broadcast::channel::<ShardedMatchRequest>(1048576);

        for _ in 0..num_shards {
            let (shard_event_tx, mut shard_event_rx) = mpsc::channel::<RouterEvent>(2048);
            let (shard_remove_worker_tx, mut shard_remove_worker_rx) =
                mpsc::channel::<WorkerId>(16);
            let (shard_get_workers_tx, mut shard_get_workers_rx) =
                mpsc::channel::<GetWorkersRequest>(16);
            let (shard_dump_tx, mut shard_dump_rx) = mpsc::channel::<DumpRequest>(16);
            let (shard_routing_tx, mut shard_routing_rx) =
                mpsc::channel::<RoutingDecisionRequest>(2048);
            let (shard_prune_tx, mut shard_prune_rx) = mpsc::channel::<()>(1);
            let mut shard_broadcast_rx = request_broadcast_tx.subscribe();
            let cancel = token.clone();
            let metrics = metrics.clone();
            let prune_config_clone = prune_config.clone();

            event_tx.push(shard_event_tx);
            remove_worker_tx.push(shard_remove_worker_tx);
            get_workers_tx.push(shard_get_workers_tx);
            dump_tx.push(shard_dump_tx);
            routing_tx.push(shard_routing_tx);

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            tasks.push(std::thread::spawn(move || {
                runtime.block_on(async move {
                    let mut trie = RadixTree::new_with_frequency(expiration_duration);

                    // Create PruneManager if prune_config is specified
                    let mut prune_manager = prune_config_clone.map(|config| {
                        PruneManager::<BlockEntry>::new(50, config)
                    });
                    let mut event_id_counter = 0u64;

                    loop {
                        // Create a future that sleeps until the next expiration time
                        let expiry_fut = if let Some(ref pm) = prune_manager
                            && let Some(next_expiry) = pm.peek_next_expiry() {
                            tokio::time::sleep_until(next_expiry)
                        } else {
                            tokio::time::sleep(Duration::MAX)
                        };

                        tokio::select! {
                            biased;

                            _ = cancel.cancelled() => {
                                tracing::trace!("KvCacheIndexer progress loop shutting down");
                                return;
                            }

                            Some(worker) = shard_remove_worker_rx.recv() => {
                                trie.remove_worker(worker);
                            }

                            Some(get_workers_req) = shard_get_workers_rx.recv() => {
                                let workers = trie.get_workers();
                                let _ = get_workers_req.resp.send(workers);
                            }

                            Some(_) = shard_prune_rx.recv() => {
                                // Tree size-based pruning triggered
                                let Some(ref mut pm) = prune_manager else { continue };
                                let Ok(pruned) = pm.prune(trie.current_size()) else { continue };

                                for p in pruned {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        p.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![p.key],
                                            }),
                                            dp_rank: p.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }

                            Some(event) = shard_event_rx.recv() => {
                                let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                                let result = trie.apply_event(event.clone());
                                let result_is_ok = result.is_ok();
                                metrics.increment_event_applied(event_type, result);

                                // Track blocks in PruneManager if TTL is enabled and event was stored successfully
                                let Some(ref mut pm) = prune_manager else { continue };
                                if !result_is_ok { continue };
                                let KvCacheEventData::Stored(ref store_data) = event.event.data else { continue };

                                let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                                let block_entries: Vec<BlockEntry> = store_data.blocks.iter().enumerate().map(|(idx, block)| {
                                    BlockEntry {
                                        key: block.block_hash,
                                        worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(routing_req) = shard_routing_rx.recv() => {
                                // Process routing decisions when TTL/pruning is enabled
                                let Some(ref mut pm) = prune_manager else { continue };

                                event_id_counter += 1;

                                let hashes = routing_req.local_hashes.iter().zip(routing_req.sequence_hashes.iter());
                                let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                    parent_hash: None,
                                    blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                        tokens_hash: *local_hash,
                                        block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                mm_extra_info: None,
                                    }).collect(),
                                });

                                let event = RouterEvent::new(
                                    routing_req.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: stored_event,
                                        dp_rank: routing_req.worker.dp_rank,
                                    }
                                );

                                if trie.apply_event(event).is_err() {
                                    continue;
                                }

                                let block_entries: Vec<BlockEntry> = routing_req.sequence_hashes.iter().enumerate().map(|(idx, h)| {
                                    BlockEntry {
                                        key: ExternalSequenceBlockHash(*h),
                                        worker: routing_req.worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(dump_req) = shard_dump_rx.recv() => {
                                let events = trie.dump_tree_as_events();
                                let _ = dump_req.resp.send(events);
                            }

                            Ok(req) = shard_broadcast_rx.recv() => {
                                let matches = trie.find_matches(req.sequence, req.early_exit);
                                if let Err(e) = req.resp.send(matches).await {
                                    tracing::trace!("Failed to send match response: {:?}", e);
                                }
                            }

                            _ = expiry_fut => {
                                // TTL-based expiry triggered
                                let Some(ref mut pm) = prune_manager else { continue };

                                let expired = pm.pop_expired();
                                for e in expired {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        e.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![e.key],
                                            }),
                                            dp_rank: e.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }
                        }
                    }
                });

                tracing::debug!("KvCacheIndexer task completed");
            }));
        }

        Self {
            cancel: token,
            kv_block_size,
            worker_assignments,
            worker_counts,
            event_tx,
            request_broadcast_tx,
            remove_worker_tx,
            dump_tx,
            routing_tx,
            tasks,
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        num_shards: usize,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, num_shards, None, kv_block_size, metrics, None)
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexerSharded {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        'match_loop: loop {
            let (match_tx, mut match_rx) = mpsc::channel(self.event_tx.len());
            self.request_broadcast_tx
                .send(ShardedMatchRequest {
                    sequence: sequence.clone(),
                    early_exit: false,
                    resp: match_tx,
                })
                .map_err(|_| KvRouterError::IndexerOffline)?;

            let mut scores = OverlapScores::new();

            for response_num in 0..self.event_tx.len() {
                match match_rx.recv().await {
                    Some(response) => {
                        scores.scores.extend(response.scores);
                        scores.tree_sizes.extend(response.tree_sizes);

                        if response_num == 0 {
                            scores.frequencies = response.frequencies;
                        } else {
                            let diff = (response.frequencies.len() as i64)
                                - (scores.frequencies.len() as i64);

                            if diff > 0 {
                                scores.frequencies.extend(iter::repeat_n(0, diff as usize));
                            }

                            for i in 0..response.frequencies.len() {
                                scores.frequencies[i] += response.frequencies[i];
                            }
                        }
                    }
                    None => {
                        // This can only happen if the broadcast channel overflows.
                        // In this case, we don't want to recursively call find_matches again. Otherwise, we could overflow the stack.
                        continue 'match_loop;
                    }
                }
            }
            return Ok(scores);
        }
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
        self.find_matches(sequence).await
    }

    async fn apply_event(&mut self, event: RouterEvent) {
        #[allow(clippy::map_entry)]
        if !self.worker_assignments.contains_key(&event.worker_id) {
            // Get the shard with the smallest amount of workers.
            let selected_shard = self
                .worker_counts
                .iter()
                .enumerate()
                .min_by_key(|&(_, value)| value)
                .unwrap()
                .0;

            self.worker_assignments
                .insert(event.worker_id, selected_shard);
            self.worker_counts[selected_shard] += 1;
        }

        self.event_tx[self.worker_assignments[&event.worker_id]]
            .send(event)
            .await
            .unwrap();
    }

    async fn remove_worker(&mut self, worker: WorkerId) {
        if let Some((_, shard)) = self.worker_assignments.remove_entry(&worker) {
            self.worker_counts[shard] -= 1;
            self.remove_worker_tx[shard].send(worker).await.unwrap();
        }
    }

    /// Shutdown the KV Indexer.
    fn shutdown(&mut self) {
        self.cancel.cancel();
        while !self.tasks.is_empty() {
            self.tasks.pop().unwrap().join().unwrap();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();

        // Create channels for each shard
        let mut receivers = Vec::new();

        for shard_dump_tx in &self.dump_tx {
            let (resp_tx, resp_rx) = oneshot::channel();
            let dump_req = DumpRequest { resp: resp_tx };

            if let Err(e) = shard_dump_tx.send(dump_req).await {
                tracing::error!("Failed to send dump request to shard: {:?}", e);
                return Err(KvRouterError::IndexerOffline);
            }

            receivers.push(resp_rx);
        }

        // Collect results from all shards
        for resp_rx in receivers {
            match resp_rx.await {
                Ok(events) => all_events.extend(events),
                Err(_) => return Err(KvRouterError::IndexerDroppedRequest),
            }
        }

        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_internal(worker, local_hashes, sequence_hashes)
            .await
    }
}

impl KvIndexerSharded {
    /// Internal method to process a routing decision with pre-computed hashes.
    async fn process_routing_decision_internal(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        // Route to the appropriate shard based on worker assignment
        let shard_idx = self
            .worker_assignments
            .get(&worker.worker_id)
            .copied()
            .unwrap_or(0);

        self.routing_tx[shard_idx]
            .send(RoutingDecisionRequest {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        Ok(())
    }
}

impl Drop for KvIndexerSharded {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::protocols::{ExternalSequenceBlockHash, LocalBlockHash};
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use tokio::time;
    use tokio_util::sync::CancellationToken;

    fn setup() {
        dynamo_runtime::logging::init();
    }

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
        setup();

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
        assert_eq!(
            trie.lookup
                .get(&WorkerWithDpRank::from_worker_id(worker_2))
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
        // This should be rejected as block 1 (hash 100) is the parent
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

        // Block appears twice in sequence: parent=1, blocks=[2, 3, 2]
        // Block 2 appears at positions 0 and 2, creating a cycle
        let result = trie.apply_event(create_store_event(
            worker_0,
            6,
            vec![2, 3, 2],
            Some(ExternalSequenceBlockHash(100)),
        ));
        assert!(matches!(
            result.unwrap_err(),
            KvCacheEventError::InvalidBlockSequence
        ));
    }

    #[test]
    fn test_remove_worker() {
        setup();
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        assert!(
            trie.find_matches(vec![LocalBlockHash(0)], false)
                .scores
                .is_empty()
        );

        trie.apply_event(create_store_event(worker_0, 0, vec![0], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![0], None))
            .unwrap();

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(
            result.len() == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_0)] == 1
                && result[&WorkerWithDpRank::from_worker_id(worker_1)] == 1
        );

        trie.remove_worker(worker_0);

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(result.len() == 1 && result[&WorkerWithDpRank::from_worker_id(worker_1)] == 1);
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
    fn test_early_stopping() {
        setup();
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        trie.apply_event(create_store_event(worker_0, 0, vec![0, 1, 2], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![0], None))
            .unwrap();

        let result = trie
            .find_matches(
                vec![LocalBlockHash(0), LocalBlockHash(1), LocalBlockHash(2)],
                true,
            )
            .scores;

        assert!(
            result.len() == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_0)] == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_1)] == 1
        );

        let result = trie
            .find_matches(vec![LocalBlockHash(0), LocalBlockHash(1)], true)
            .scores;
        assert!(
            result.len() == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_0)] == 2
                && result[&WorkerWithDpRank::from_worker_id(worker_1)] == 1
        );
    }

    #[rstest]
    #[case(11)]
    #[case(32)]
    #[case(64)]
    fn test_compute_block_hash_for_seq(#[case] kv_block_size: u32) {
        setup();
        // create a sequence of 64 elements
        let sequence = (0..kv_block_size).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 65 elements
        let sequence = (0..(kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 129 elements
        let sequence = (0..(2 * kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 2);
    }

    fn make_indexer(
        token: &CancellationToken,
        num_shards: usize,
        kv_block_size: u32,
    ) -> Box<dyn KvIndexerInterface> {
        let metrics = KvIndexerMetrics::new_unregistered();
        if num_shards == 1 {
            Box::new(KvIndexer::new(token.clone(), kv_block_size, metrics.into()))
        } else {
            Box::new(KvIndexerSharded::new(
                token.clone(),
                num_shards,
                kv_block_size,
                metrics.into(),
            ))
        }
    }

    #[template]
    #[rstest]
    fn indexer_template(
        #[values(1, 3, 8)] num_shards: usize,
        #[values(11, 32, 64)] kv_block_size: usize,
    ) {
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_kv_indexer_new(num_shards: usize, kv_block_size: u32) {
        setup();
        let token: CancellationToken = CancellationToken::new();
        let _ = make_indexer(&token, num_shards, kv_block_size);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_find_matches(num_shards: usize, kv_block_size: u32) {
        setup();
        let token = CancellationToken::new();
        let kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let sequence = vec![compute_block_hash(b"test data")];
        let scores = kv_indexer.find_matches(sequence).await;

        assert!(scores.unwrap().scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_find_matches_for_request(num_shards: usize, kv_block_size: u32) {
        setup();
        let token = CancellationToken::new();
        let kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let tokens = vec![1, 2, 3, 4];
        let scores = kv_indexer.find_matches_for_request(&tokens).await;

        assert!(scores.unwrap().scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_apply_event(num_shards: usize, kv_block_size: u32) {
        setup();
        let worker_id = 0;

        let token = CancellationToken::new();
        let mut kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let event = create_store_event(worker_id, 1, vec![1, 2, 3], None);
        kv_indexer.apply_event(event).await;

        // No assertion here, just ensuring it runs without panic
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_shutdown(num_shards: usize, kv_block_size: u32) {
        setup();
        let token = CancellationToken::new();
        let mut kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        kv_indexer.shutdown();
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_frequency(num_shards: usize, kv_block_size: u32) {
        const ONE_MILLIS: Duration = Duration::from_millis(1);

        setup();
        let mut kv_indexer: Box<dyn KvIndexerInterface>;
        let token = CancellationToken::new();
        let expiration = Duration::from_millis(50);
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());

        if num_shards == 1 {
            kv_indexer = Box::new(KvIndexer::new_with_frequency(
                token,
                Some(expiration),
                kv_block_size,
                metrics,
                None,
            ));
        } else {
            kv_indexer = Box::new(KvIndexerSharded::new_with_frequency(
                token,
                num_shards,
                Some(expiration),
                kv_block_size,
                metrics,
                None,
            ));
        }

        // The blocks
        let block_hashes = vec![
            LocalBlockHash(1),
            LocalBlockHash(2),
            LocalBlockHash(3),
            LocalBlockHash(4),
        ];

        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Should be no cached blocks yet"
        );

        // Blocks go in cache
        let worker_id = 0;
        let event = create_store_event(worker_id, 0, vec![1, 2, 3, 4], None);
        kv_indexer.apply_event(event).await;

        // First access
        // The store event is applied async so poll briefly
        let mut overlap = OverlapScores::default();
        let timeout = Duration::from_millis(10);
        let start = Instant::now();
        while overlap.scores.is_empty() && Instant::now().duration_since(start) < timeout {
            time::sleep(ONE_MILLIS).await;
            overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        }
        assert_eq!(
            overlap.scores.len(),
            1,
            "One worker has these blocks cached"
        );
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Blocks have not previously been accessed"
        );

        // Second access
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(overlap.scores.len(), 1, "Still one worker matches");
        assert_eq!(
            overlap.frequencies,
            vec![1, 1, 1, 1],
            "We should see the first access now"
        );

        // Let those two accesses expire
        time::sleep(expiration + Duration::from_millis(10)).await;

        // New first access
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Blocks were accessed too long ago"
        );

        // New second access
        let _ = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();

        // Access only the first three blocks
        let overlap = kv_indexer
            .find_matches(block_hashes[0..3].to_vec())
            .await
            .unwrap();
        // We see the previous two new accesses
        assert_eq!(overlap.frequencies, vec![2, 2, 2]);

        // The third access did not touch the last block
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(overlap.frequencies, vec![3, 3, 3, 2]);
    }

    #[test]
    fn test_router_event_new() {
        setup();
        let worker_id = 0;
        let kv_cache_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(0),
                    mm_extra_info: None,
                    tokens_hash: LocalBlockHash(13226331709069118873),
                }],
            }),
            dp_rank: 0,
        };
        let router_event = RouterEvent::new(worker_id, kv_cache_event);

        assert_eq!(router_event.worker_id, worker_id);
        assert_eq!(router_event.event.event_id, 1);
        if let KvCacheEventData::Stored(store_op) = &router_event.event.data {
            assert_eq!(store_op.blocks.len(), 1);
            assert_eq!(
                store_op.blocks[0].tokens_hash,
                compute_block_hash(b"test data")
            );
            assert_eq!(store_op.blocks[0].block_hash, ExternalSequenceBlockHash(0));
        } else {
            panic!("Expected KvCacheEventData::Stored");
        }
    }

    #[test]
    fn test_radix_tree_default() {
        setup();
        let radix_tree: RadixTree = Default::default();
        assert!(radix_tree.root.borrow().children.is_empty());
        assert!(radix_tree.root.borrow().workers.is_empty());
        assert!(radix_tree.lookup.is_empty());
    }

    #[test]
    fn test_overlap_scores_default() {
        setup();
        let overlap_scores: OverlapScores = Default::default();
        assert!(overlap_scores.scores.is_empty());
    }

    #[tokio::test]
    async fn test_dump_tree_as_events_round_trip() {
        setup();

        // Configuration
        let kv_block_size = 32;
        let num_shards = 2;
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());

        // Build a non-trivial indexer with events
        let token1 = CancellationToken::new();
        let mut original_indexer =
            KvIndexerSharded::new(token1.clone(), num_shards, kv_block_size, metrics.clone());

        let worker_0 = 0;
        let worker_1 = 1;
        let worker_2 = 2;

        // Apply events to the original indexer
        original_indexer
            .apply_event(create_store_event(worker_0, 0, vec![1, 2, 3], None))
            .await;

        original_indexer
            .apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None))
            .await;
        original_indexer
            .apply_event(create_store_event(
                worker_1,
                2,
                vec![4, 5],
                Some(ExternalSequenceBlockHash(100)),
            ))
            .await;

        original_indexer
            .apply_event(create_store_event(worker_2, 3, vec![6, 7], None))
            .await;

        original_indexer
            .apply_event(create_store_event(
                worker_0,
                4,
                vec![4],
                Some(ExternalSequenceBlockHash(100)),
            ))
            .await;

        // Allow some time for events to be processed
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Dump the original indexer
        let dump1 = original_indexer.dump_events().await.unwrap();
        println!("Dumped {} events", dump1.len());

        // Create a new indexer and apply all dumped events
        let token2 = CancellationToken::new();
        let mut reconstructed_indexer =
            KvIndexerSharded::new(token2.clone(), num_shards, kv_block_size, metrics);

        for event in &dump1 {
            reconstructed_indexer.apply_event(event.clone()).await;
        }

        // Allow some time for events to be processed
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Dump the reconstructed indexer
        let dump2 = reconstructed_indexer.dump_events().await.unwrap();

        // Sort both dumps for comparison (order might differ due to HashMap iteration and sharding)
        let mut sorted_dump1 = dump1.clone();
        let mut sorted_dump2 = dump2.clone();

        // Sort by (worker_id, tokens_hash, parent_hash)
        let sort_key = |event: &RouterEvent| {
            if let KvCacheEventData::Stored(ref data) = event.event.data {
                (
                    event.worker_id,
                    data.blocks.first().map(|b| b.tokens_hash.0).unwrap_or(0),
                    data.parent_hash.map(|h| h.0).unwrap_or(0),
                )
            } else {
                (event.worker_id, 0, 0)
            }
        };

        sorted_dump1.sort_by_key(sort_key);
        sorted_dump2.sort_by_key(sort_key);

        // Verify the dumps have the same length
        assert_eq!(
            sorted_dump1.len(),
            sorted_dump2.len(),
            "Dumps have different lengths: {} vs {}",
            sorted_dump1.len(),
            sorted_dump2.len()
        );

        // Verify each event matches
        for (i, (event1, event2)) in sorted_dump1.iter().zip(sorted_dump2.iter()).enumerate() {
            assert_eq!(
                event1.worker_id, event2.worker_id,
                "Event {} worker_id mismatch",
                i
            );

            if let (KvCacheEventData::Stored(data1), KvCacheEventData::Stored(data2)) =
                (&event1.event.data, &event2.event.data)
            {
                assert_eq!(
                    data1.parent_hash, data2.parent_hash,
                    "Event {} parent_hash mismatch",
                    i
                );
                assert_eq!(
                    data1.blocks.len(),
                    data2.blocks.len(),
                    "Event {} blocks length mismatch",
                    i
                );

                for (j, (block1, block2)) in
                    data1.blocks.iter().zip(data2.blocks.iter()).enumerate()
                {
                    assert_eq!(
                        block1.tokens_hash, block2.tokens_hash,
                        "Event {} block {} tokens_hash mismatch",
                        i, j
                    );
                    assert_eq!(
                        block1.block_hash, block2.block_hash,
                        "Event {} block {} block_hash mismatch",
                        i, j
                    );
                }
            } else {
                panic!("Expected Stored events in both dumps");
            }
        }

        // Also verify that both indexers produce the same match results
        for test_seq in [
            vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
            vec![LocalBlockHash(1), LocalBlockHash(4), LocalBlockHash(5)],
            vec![LocalBlockHash(6), LocalBlockHash(7)],
            vec![LocalBlockHash(1)],
        ] {
            let scores1 = original_indexer
                .find_matches(test_seq.clone())
                .await
                .unwrap();
            let scores2 = reconstructed_indexer
                .find_matches(test_seq.clone())
                .await
                .unwrap();

            // Sort the scores to compare
            let mut scores1_sorted: Vec<_> = scores1.scores.iter().collect();
            let mut scores2_sorted: Vec<_> = scores2.scores.iter().collect();
            scores1_sorted.sort_by_key(|(k, _)| *k);
            scores2_sorted.sort_by_key(|(k, _)| *k);

            assert_eq!(
                scores1_sorted, scores2_sorted,
                "Match scores differ for sequence {:?}",
                test_seq
            );
        }

        // Clean up
        original_indexer.shutdown();
        reconstructed_indexer.shutdown();
    }

    #[test]
    fn test_increment_event_applied() {
        let metrics = KvIndexerMetrics::new_unregistered();

        metrics.increment_event_applied(METRIC_EVENT_STORED, Ok(()));
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_OK])
                .unwrap()
                .get(),
            1
        );

        metrics.increment_event_applied(
            METRIC_EVENT_STORED,
            Err(KvCacheEventError::ParentBlockNotFound),
        );
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[
                    METRIC_EVENT_STORED,
                    METRIC_STATUS_PARENT_NOT_FOUND
                ])
                .unwrap()
                .get(),
            1
        );

        metrics
            .increment_event_applied(METRIC_EVENT_REMOVED, Err(KvCacheEventError::BlockNotFound));
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[
                    METRIC_EVENT_REMOVED,
                    METRIC_STATUS_BLOCK_NOT_FOUND
                ])
                .unwrap()
                .get(),
            1
        );
    }

    #[test]
    fn test_remove_worker_verifies_hash_removal() {
        setup();
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
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_1))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_2))
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
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_1))
        );
        assert!(
            block_1
                .borrow()
                .workers
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_2))
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
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_1))
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

    // LocalKvIndexer tests
    fn make_indexer_with_events(ids: &[u64]) -> LocalKvIndexer {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            32,
        );
        {
            let mut buffer = indexer.event_buffer.lock().unwrap();
            for &id in ids {
                buffer.push_back(RouterEvent::new(
                    0,
                    KvCacheEvent {
                        event_id: id,
                        data: KvCacheEventData::Cleared,
                        dp_rank: 0,
                    },
                ));
            }
        }
        indexer
    }

    #[tokio::test]
    async fn returns_slice_within_range() {
        let indexer = make_indexer_with_events(&[1, 2, 3, 4, 5]);

        // Helper to extract events from response
        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events(e) => e,
                WorkerKvQueryResponse::TreeDump(e) => e,
                _ => panic!("Unexpected response type"),
            }
        };

        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Test get_events_in_id_range (buffer queries)
        // Range is [start, end] inclusive
        let result = indexer.get_events_in_id_range(Some(2), Some(4)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![2, 3, 4]); // inclusive range [2, 4]

        let result = indexer.get_events_in_id_range(Some(2), Some(6)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![2, 3, 4, 5]); // clamp end to buffer max

        // start_id=0 is before buffer (first is 1), so should trigger tree dump
        let result = indexer.get_events_in_id_range(Some(0), Some(4)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));

        let result = indexer.get_events_in_id_range(Some(3), Some(3)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![3]); // single element when start == end

        // Invalid range: end < start
        let result = indexer.get_events_in_id_range(Some(5), Some(2)).await;
        assert!(matches!(result, WorkerKvQueryResponse::InvalidRange { .. }));
    }

    #[tokio::test]
    async fn test_get_events_in_id_range_all_cases() {
        // Create indexer with small buffer (5 events max)
        // This way older events will only be in the tree, not the buffer
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4, // block_size
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5, // max_buffer_size - only keeps 5 most recent events
        );

        // Helper to create a test event
        let make_event = |id: u64| {
            RouterEvent::new(
                0, // worker_id
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(id * 100),
                            tokens_hash: LocalBlockHash(id * 200),
                            mm_extra_info: None,
                        }],
                    }),
                    dp_rank: 0,
                },
            )
        };

        // Add 10 events (IDs 5-14)
        // Buffer will only keep the last 5: events 10-14
        // Tree will have all blocks
        for id in 5..15 {
            indexer
                .apply_event_with_buffer(make_event(id))
                .await
                .unwrap();
        }

        // Wait for events to be processed by the tree
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Helper to extract events from response
        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events(e) => e,
                WorkerKvQueryResponse::TreeDump(e) => e,
                _ => panic!("Unexpected response type: {:?}", resp),
            }
        };

        // Helper to extract event IDs from result
        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Verify buffer state: should have events 10-14 (last 5)
        let buffer_events = indexer.get_all_events_in_buffer();
        assert_eq!(
            get_ids(buffer_events),
            vec![10, 11, 12, 13, 14],
            "Buffer should have events 10-14"
        );

        // ========== BUFFER PATH TESTS (start_id >= first_buffered) ==========
        // Range is [start, end] inclusive

        // Test: start_id within buffer, no end
        let result = indexer.get_events_in_id_range(Some(11), None).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![11, 12, 13, 14],
            "start_id=11 (in buffer) should return [11, 14]"
        );

        // Test: start_id at buffer boundary
        let result = indexer.get_events_in_id_range(Some(10), None).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![10, 11, 12, 13, 14],
            "start_id=10 (buffer start) should return [10, 14]"
        );

        // Test: both start and end within buffer (inclusive)
        let result = indexer.get_events_in_id_range(Some(11), Some(13)).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![11, 12, 13],
            "range [11, 13] inclusive should return 3 events"
        );

        let result = indexer.get_events_in_id_range(Some(10), Some(14)).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![10, 11, 12, 13, 14],
            "range [10, 14] should return all buffer events"
        );

        // ========== TREE DUMP PATH TESTS (range extends before buffer) ==========
        // Note: Tree dumps return synthetic 0-indexed event IDs, so we just check
        // that we get events back (the IDs won't match original IDs)

        // Test: (None, None) dumps entire tree
        let result = indexer.get_events_in_id_range(None, None).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));
        assert_eq!(
            extract_events(result).len(),
            10,
            "(None, None) should dump entire tree (10 events)"
        );

        // Test: (None, Some(_)) dumps entire tree
        let result = indexer.get_events_in_id_range(None, Some(8)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));
        assert_eq!(
            extract_events(result).len(),
            10,
            "(None, Some(_)) dumps entire tree - end_id is ignored for tree dumps"
        );

        // Test: start_id before buffer triggers tree dump
        let result = indexer.get_events_in_id_range(Some(7), None).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));
        assert_eq!(
            extract_events(result).len(),
            10,
            "start_id=7 (before buffer) should dump entire tree"
        );

        let result = indexer.get_events_in_id_range(Some(5), Some(12)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));
        assert_eq!(
            extract_events(result).len(),
            10,
            "range [5, 12] extending before buffer should dump entire tree"
        );

        // ========== EDGE CASES ==========

        // Single element when start == end (inclusive range)
        let result = indexer.get_events_in_id_range(Some(12), Some(12)).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![12],
            "start == end should return single event"
        );

        // InvalidRange when start > end
        let result = indexer.get_events_in_id_range(Some(15), Some(10)).await;
        assert!(
            matches!(result, WorkerKvQueryResponse::InvalidRange { .. }),
            "start > end should return InvalidRange"
        );

        // TooNew when start_id is beyond buffer
        let result = indexer.get_events_in_id_range(Some(100), Some(200)).await;
        assert!(
            matches!(result, WorkerKvQueryResponse::TooNew { .. }),
            "start_id beyond buffer should return TooNew"
        );

        // Request with end beyond buffer but valid start -> buffer returns what it has
        let result = indexer.get_events_in_id_range(Some(12), Some(100)).await;
        assert!(matches!(result, WorkerKvQueryResponse::Events(_)));
        assert_eq!(
            get_ids(extract_events(result)),
            vec![12, 13, 14],
            "range with end beyond buffer should return available buffer events"
        );
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_and_serialization() {
        // Tests components of the LocalKvIndexer query without using nats

        let worker_id = 42u64;

        // Create a local indexer
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Add events to local indexer's buffer
        let test_event_1 = RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        // Apply events with buffer
        local_indexer
            .apply_event_with_buffer(test_event_1)
            .await
            .unwrap();

        // Wait for events to be processed
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Get buffered events (what the query service would return)
        let buffered_events = local_indexer.get_all_events_in_buffer();

        // Verify buffer contents
        assert_eq!(buffered_events.len(), 1, "Buffer should have 1 event");
        assert_eq!(buffered_events[0].worker_id, worker_id);
        assert_eq!(buffered_events[0].event.event_id, 1);

        // Build the response that would be sent (Events variant)
        let response = WorkerKvQueryResponse::Events(buffered_events.clone());

        // Test serialization/deserialization (simulating NATS round-trip)
        let serialized = serde_json::to_vec(&response).unwrap();
        let deserialized: WorkerKvQueryResponse = serde_json::from_slice(&serialized).unwrap();

        // Verify response correctness
        let events = match deserialized {
            WorkerKvQueryResponse::Events(e) => e,
            _ => panic!("Expected Events variant"),
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].worker_id, worker_id);
        assert_eq!(events[0].event.event_id, 1);

        // Verify event data
        match &events[0].event.data {
            KvCacheEventData::Stored(store_data) => {
                assert_eq!(store_data.blocks.len(), 1);
                assert_eq!(store_data.blocks[0].block_hash.0, 100);
                assert_eq!(store_data.blocks[0].tokens_hash.0, 200);
            }
            _ => panic!("Expected Stored event"),
        }
    }
}
