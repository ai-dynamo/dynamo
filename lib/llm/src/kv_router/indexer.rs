// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use bytes::Bytes;
use dynamo_runtime::{
    component::Component,
    metrics::{MetricsRegistry, prometheus_names::kvrouter},
};
use prometheus::{IntCounterVec, Opts};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    iter,
    rc::Rc,
    sync::{Arc, OnceLock},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use xxhash_rust::xxh3;

pub const XXH3_SEED: u64 = 1337;

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
}

/// Errors that can occur during KV Cache Event processing.
#[derive(Debug, thiserror::Error)]
pub enum KvCacheEventError {
    #[error("Failed to find parent block")]
    ParentBlockNotFound,

    #[error("Failed to find block")]
    BlockNotFound,
}

/// Identifier of a LLM worker which emits events to the router.
pub type WorkerId = i64;

/// A shared reference to a [`RadixBlock`].
type SharedRadixBlock = Rc<RefCell<RadixBlock>>;

pub fn compute_hash(data: &[u8]) -> u64 {
    xxh3::xxh3_64_with_seed(data, XXH3_SEED)
}

/// Compute the hash of a local block.
///
/// ### Arguments
///
/// * `data` - A byte slice representing the data to hash.
///
/// ### Returns
///
/// A `LocalBlockHash` representing the computed hash.
pub fn compute_block_hash(data: &[u8]) -> LocalBlockHash {
    LocalBlockHash(compute_hash(data))
}

// /// Updated version of the `compute_block_hash` function that included the lora_id
// pub fn compute_block_hash_v2(token_id: &[u32], lora_id: u64) {
//     let mut bytes = Vec::new();
//     for token in token_id {
//         bytes.extend_from_slice(&token.to_le_bytes());
//     }
//     bytes.extend_from_slice(&lora_id.to_le_bytes());
//     let hash = xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED);
// }

/// Compute the hash for a sequence of tokens.
///
/// ### Arguments
///
/// * `tokens` - A vector of `u32` tokens.
///
/// ### Returns
///
/// A vector of `LocalBlockHash` representing the computed hashes for each chunk of tokens.
pub fn compute_block_hash_for_seq(tokens: &[u32], kv_block_size: u32) -> Vec<LocalBlockHash> {
    tokens
        .chunks_exact(kv_block_size as usize) // Split into chunks of kv_block_size elements
        .map(|chunk| {
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|&num| num.to_le_bytes()) // Convert each i32 to its little-endian bytes
                .collect();

            compute_block_hash(&Bytes::from(bytes)) // Convert the byte Vec to Bytes
        })
        .collect()
}

/// Compute rolling sequence hashes for a vector of block hashes.
///
/// This mirrors the behavior in tokens.rs where:
/// - The first block's sequence hash equals its block hash
/// - Subsequent blocks' sequence hash = hash([parent_sequence_hash, current_block_hash], seed)
///
/// ### Arguments
///
/// * `block_hashes` - A vector of `LocalBlockHash` values representing the block hashes.
///
/// ### Returns
///
/// A vector of u64 values representing the sequence hashes for each block.
pub fn compute_seq_hash_for_block(block_hashes: &[LocalBlockHash]) -> Vec<SequenceHash> {
    if block_hashes.is_empty() {
        return Vec::new();
    }

    let mut sequence_hashes = Vec::with_capacity(block_hashes.len());
    sequence_hashes.push(block_hashes[0].0);

    for i in 1..block_hashes.len() {
        let parent_seq_hash = sequence_hashes[i - 1];
        let current_block_hash = block_hashes[i].0;

        let combined = [parent_seq_hash, current_block_hash];
        let bytes: Vec<u8> = combined.iter().flat_map(|&num| num.to_le_bytes()).collect();
        let seq_hash = compute_hash(&bytes);
        sequence_hashes.push(seq_hash);
    }

    sequence_hashes
}

/// A [`KvCacheEvent`] on a specific LLM worker denoted by [`WorkerId`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEvent {
    /// The ID of the worker emitting the event.
    worker_id: WorkerId,
    /// The cache event associated with the worker.
    event: KvCacheEvent,
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

/// A block in the Radix Tree.
#[derive(Debug)]
struct RadixBlock {
    /// A map of child blocks, keyed by their local block hash.
    children: HashMap<LocalBlockHash, SharedRadixBlock>,
    /// A map of worker IDs to their external sequence block hash for this block.
    /// The external hash is preserved to speed up snapshotting.
    workers: HashMap<WorkerId, ExternalSequenceBlockHash>,
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
    lookup: HashMap<WorkerId, HashMap<ExternalSequenceBlockHash, SharedRadixBlock>>,
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

        scores
    }

    /// Apply a [`RouterEvent`] to the radix tree.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    pub fn apply_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let (worker_id, event) = (event.worker_id, event.event);
        let (id, op) = (event.event_id, event.data);
        tracing::trace!(id, "RadixTree::apply_event: Store operation: {:?}", op);

        let worker_lookup = self.lookup.entry(worker_id).or_default();

        match op {
            KvCacheEventData::Stored(op) => {
                // find the parent block - if the parent exists it must be on our worker, if not,
                // we check the radix tree's root to find it.
                // this is the single most expensive lookup
                let current = match op.parent_hash {
                    Some(parent) => worker_lookup.get(&parent),
                    None => Some(&self.root),
                };

                let mut current = match current {
                    Some(current) => current.clone(),
                    None => {
                        tracing::warn!(
                            worker_id = worker_id.to_string(),
                            id,
                            parent_hash = ?op.parent_hash,
                            "Failed to find parent block; skipping store operation"
                        );
                        return Err(KvCacheEventError::ParentBlockNotFound);
                    }
                };

                for block_id in op.blocks {
                    let mut inner = current.borrow_mut();
                    let block = match inner.children.get(&block_id.tokens_hash) {
                        Some(block) => block.clone(),
                        None => {
                            // create new block - automatically added to the lookup table
                            let new_block = worker_lookup
                                .get(&block_id.block_hash)
                                .cloned()
                                .unwrap_or_else(|| Rc::new(RefCell::new(RadixBlock::new())));

                            // insert into radix tree
                            inner
                                .children
                                .insert(block_id.tokens_hash, new_block.clone());

                            new_block
                        }
                    };

                    // add our worker_id to the block with its external hash
                    block
                        .borrow_mut()
                        .workers
                        .insert(worker_id, block_id.block_hash);

                    // add the block to the worker_id lookup table
                    worker_lookup.insert(block_id.block_hash, block.clone());

                    // drop inner so we can shift current to this block
                    drop(inner);

                    current = block;
                }
                Ok(())
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
                                worker_id = worker_id.to_string(),
                                id,
                                "Failed to find block to remove; skipping remove operation"
                            );
                            return Err(KvCacheEventError::BlockNotFound);
                        }
                    };

                    let mut guard = entry.borrow_mut();
                    guard.workers.remove(&worker_id);
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
                self.clear_all_blocks(worker_id);
                Ok(())
            }
        }
    }

    pub fn remove_worker(&mut self, worker: WorkerId) {
        if let Some((_, blocks)) = self.lookup.remove_entry(&worker) {
            blocks.iter().for_each(|(_, block)| {
                block.borrow_mut().workers.remove(&worker);
                // If no workers are using this block, that is true for all children
                if block.borrow().workers.is_empty() {
                    block.borrow_mut().children.clear();
                }
            });
        }
    }

    pub fn clear_all_blocks(&mut self, worker: WorkerId) {
        // Check if the worker has any blocks to clear
        if let Some(blocks) = self.lookup.get(&worker) {
            let blocks_to_clear: Vec<_> = blocks.values().collect();

            // Remove the worker from each block's workers map
            blocks_to_clear.iter().for_each(|block| {
                block.borrow_mut().workers.remove(&worker);
                // If no workers are using this block, that is true for all children
                if block.borrow().workers.is_empty() {
                    block.borrow_mut().children.clear();
                }
            });

            // Clear the worker's blocks
            if let Some(worker_lookup) = self.lookup.get_mut(&worker) {
                worker_lookup.clear();
            }
        }
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
        // parent_hashes_per_worker maps WorkerId -> ExternalSequenceBlockHash
        let mut queue: VecDeque<(
            SharedRadixBlock,
            HashMap<WorkerId, ExternalSequenceBlockHash>,
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
                    worker_id: *worker_id,
                    event: KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash: *external_hash,
                                tokens_hash,
                            }],
                        }),
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
            match component.create_intcountervec(
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
                };
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, error_label])
                    .inc_by(1);
            }
        }
    }
}

/// Scores representing the overlap of workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapScores {
    // map of worker_id to score
    pub scores: HashMap<WorkerId, u32>,
    // List of frequencies that the blocks have been accessed. Entries with value 0 are omitted.
    pub frequencies: Vec<usize>,
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
        }
    }

    /// Update the scores with a set of workers.
    ///
    /// ### Arguments
    ///
    /// * `workers` - An iterator over `WorkerId` references.
    pub fn update_scores<'a, I>(&mut self, workers: I)
    where
        I: IntoIterator<Item = &'a WorkerId>,
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
}

/// The KV Indexer, managing the KV store and handling events and match requests.
pub struct KvIndexer {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// A sender for `RouterEvent`s.
    event_tx: mpsc::Sender<RouterEvent>,
    /// A sender for `MatchRequest`s.
    match_tx: mpsc::Sender<MatchRequest>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A handle to the background task managing the KV store.
    task: OnceLock<std::thread::JoinHandle<()>>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
}

impl KvIndexer {
    /// Create a new `KvIndexer`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(2048);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (dump_tx, dump_rx) = mpsc::channel::<DumpRequest>(16);
        let cancel_clone = token.clone();

        let task = std::thread::spawn(move || {
            // create a new tokio runtime which will only perform work on a single thread
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1) // Single-threaded environment
                .enable_all()
                .build()
                .unwrap();

            let local_set = tokio::task::LocalSet::new();

            runtime.block_on(local_set.run_until(async move {
                tokio::task::spawn_local(async move {
                    let cancel = cancel_clone;
                    let mut match_rx = match_rx;
                    let mut event_rx = event_rx;
                    let mut remove_worker_rx = remove_worker_rx;
                    let mut dump_rx = dump_rx;
                    let mut trie = RadixTree::new_with_frequency(expiration_duration);
                    loop {
                        tokio::select! {
                            biased;

                            _ = cancel.cancelled() => {
                                tracing::debug!("KvCacheIndexer progress loop shutting down");
                                return;
                            }

                            Some(worker) = remove_worker_rx.recv() => {
                                trie.remove_worker(worker);
                            }

                            Some(event) = event_rx.recv() => {
                                let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                                let result = trie.apply_event(event);
                                metrics.increment_event_applied(event_type, result);
                            }

                            Some(dump_req) = dump_rx.recv() => {
                                let events = trie.dump_tree_as_events();
                                let _ = dump_req.resp.send(events);
                            }

                            Some(req) = match_rx.recv() => {
                                let matches = trie.find_matches(req.sequence, req.early_exit);
                                let _ = req.resp.send(matches);
                            }
                        }
                    }
                })
                .await
                .unwrap()
            }));

            tracing::debug!("KvCacheIndexer task completed");
        });

        let once = OnceLock::new();
        once.set(task).unwrap();

        Self {
            cancel: token,
            event_tx,
            match_tx,
            remove_worker_tx,
            dump_tx,
            task: once,
            kv_block_size,
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
        Self::new_with_frequency(token, None, kv_block_size, metrics)
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
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size);
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
        if let Some(task) = self.task.take() {
            task.join().expect("Failed to join kv indexer task");
        }
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
    ) -> Self {
        let worker_assignments: HashMap<WorkerId, usize> = HashMap::new();
        let worker_counts: Vec<usize> = vec![0; num_shards];

        let mut event_tx = Vec::new();
        let mut remove_worker_tx = Vec::new();
        let mut dump_tx = Vec::new(); // Add dump channels
        let mut tasks = Vec::new();

        let (request_broadcast_tx, _) = broadcast::channel::<ShardedMatchRequest>(1048576);

        for _ in 0..num_shards {
            let (shard_event_tx, mut shard_event_rx) = mpsc::channel::<RouterEvent>(2048);
            let (shard_remove_worker_tx, mut shard_remove_worker_rx) =
                mpsc::channel::<WorkerId>(16);
            let (shard_dump_tx, mut shard_dump_rx) = mpsc::channel::<DumpRequest>(16); // Add dump channel
            let mut shard_broadcast_rx = request_broadcast_tx.subscribe();
            let cancel = token.clone();
            let metrics = metrics.clone();

            event_tx.push(shard_event_tx);
            remove_worker_tx.push(shard_remove_worker_tx);
            dump_tx.push(shard_dump_tx); // Store dump sender

            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();

            tasks.push(std::thread::spawn(move || {
                let local_set = tokio::task::LocalSet::new();

                runtime.block_on(local_set.run_until(async move {
                    tokio::task::spawn_local(async move {
                        let mut trie = RadixTree::new_with_frequency(expiration_duration);
                        loop {
                            tokio::select! {
                                biased;

                                _ = cancel.cancelled() => {
                                    tracing::trace!("KvCacheIndexer progress loop shutting down");
                                    return;
                                }

                                Some(worker) = shard_remove_worker_rx.recv() => {
                                    trie.remove_worker(worker);
                                }

                                Some(event) = shard_event_rx.recv() => {
                                    let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                                    let result = trie.apply_event(event);
                                    metrics.increment_event_applied(event_type, result);
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
                            }
                        }
                    })
                    .await
                    .unwrap()
                }));

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
            dump_tx, // Add dump_tx field
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
        Self::new_with_frequency(token, num_shards, None, kv_block_size, metrics)
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
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size);
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
}

#[cfg(test)]
mod tests {

    use super::*;
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
        assert_eq!(scores.scores.get(&worker_1).unwrap(), &3);

        assert_eq!(trie.lookup.len(), 1);
        assert_eq!(trie.lookup.get(&worker_1).unwrap().len(), 3);
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
        assert_eq!(scores.scores.get(&worker_1).unwrap(), &3);
        assert_eq!(scores.scores.get(&worker_2).unwrap(), &1);

        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(trie.lookup.get(&worker_1).unwrap().len(), 3);
        assert_eq!(trie.lookup.get(&worker_2).unwrap().len(), 3);
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
        assert_eq!(trie.lookup.get(&worker_1).unwrap().len(), 3);
        assert_eq!(trie.lookup.get(&worker_2).unwrap().len(), 2);
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
        assert_eq!(trie.lookup.get(&worker_1).unwrap().len(), 3);
        assert_eq!(trie.lookup.get(&worker_2).unwrap().len(), 1);
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
        assert_eq!(scores.scores.get(&worker_1).unwrap(), &3);
        assert_eq!(scores.scores.get(&worker_2).unwrap(), &2);

        assert_eq!(trie.lookup.len(), 2);
        assert_eq!(trie.lookup.get(&worker_1).unwrap().len(), 3);
        assert_eq!(trie.lookup.get(&worker_2).unwrap().len(), 4);
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
                .get(&worker_1)
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
                .get(&worker_2)
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
        assert!(result.len() == 2 && result[&worker_0] == 1 && result[&worker_1] == 1);

        trie.remove_worker(worker_0);

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(result.len() == 1 && result[&worker_1] == 1);
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
        assert!(!trie.lookup.contains_key(&worker_0));

        // Test clearing a worker with shared blocks
        trie.apply_event(create_store_event(worker_0, 0, vec![0, 1, 3], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![0, 2, 3], None))
            .unwrap();

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(result.len() == 2 && result[&worker_0] == 1 && result[&worker_1] == 1);

        trie.clear_all_blocks(worker_0);

        assert!(trie.lookup.contains_key(&worker_0));
        assert!(trie.lookup.get(&worker_0).unwrap().is_empty());
        let result = trie
            .find_matches(vec![LocalBlockHash(0), LocalBlockHash(2)], false)
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&worker_1], 2);
        let result = trie
            .find_matches(
                vec![LocalBlockHash(0), LocalBlockHash(1), LocalBlockHash(3)],
                false,
            )
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&worker_1], 1);

        // Test re-adding blocks after clearing worker
        trie.apply_event(create_store_event(worker_0, 0, vec![4, 5], None))
            .unwrap();
        let result = trie
            .find_matches(vec![LocalBlockHash(4), LocalBlockHash(5)], false)
            .scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&worker_0], 2);

        // Test multiple clears
        trie.clear_all_blocks(worker_0);
        trie.clear_all_blocks(worker_0);
        assert!(trie.lookup.contains_key(&worker_0));

        // Test clearing all workers
        trie.clear_all_blocks(worker_0);
        trie.clear_all_blocks(worker_1);
        assert!(!trie.lookup.is_empty());
        assert!(trie.lookup.get(&worker_0).unwrap().is_empty());
        assert!(trie.lookup.get(&worker_1).unwrap().is_empty());

        // Test clearing a worker that has been removed
        trie.apply_event(create_store_event(worker_0, 0, vec![6], None))
            .unwrap();
        trie.apply_event(create_store_event(worker_1, 0, vec![6], None))
            .unwrap();
        trie.remove_worker(worker_0);
        trie.clear_all_blocks(worker_0);
        assert!(!trie.lookup.contains_key(&worker_0));
        let result = trie.find_matches(vec![LocalBlockHash(6)], false).scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&worker_1], 1);

        // Test clearing a worker that doesn't exist
        let worker_fake = 2;
        assert!(!trie.lookup.contains_key(&worker_fake));
        trie.clear_all_blocks(worker_fake);
        assert!(!trie.lookup.contains_key(&worker_fake));
        assert!(trie.lookup.contains_key(&worker_1));
        let result = trie.find_matches(vec![LocalBlockHash(6)], false).scores;
        assert_eq!(result.len(), 1);
        assert_eq!(result[&worker_1], 1);
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

        assert!(result.len() == 2 && result[&worker_0] == 2 && result[&worker_1] == 1);

        let result = trie
            .find_matches(vec![LocalBlockHash(0), LocalBlockHash(1)], true)
            .scores;
        assert!(result.len() == 2 && result[&worker_0] == 2 && result[&worker_1] == 1);
    }

    #[rstest]
    #[case(11)]
    #[case(32)]
    #[case(64)]
    fn test_compute_block_hash_for_seq(#[case] kv_block_size: u32) {
        setup();
        // create a sequence of 64 elements
        let sequence = (0..kv_block_size).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 65 elements
        let sequence = (0..(kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 129 elements
        let sequence = (0..(2 * kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
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
            ));
        } else {
            kv_indexer = Box::new(KvIndexerSharded::new_with_frequency(
                token,
                num_shards,
                Some(expiration),
                kv_block_size,
                metrics,
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
                    tokens_hash: LocalBlockHash(13226331709069118873),
                }],
            }),
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
        assert_eq!(trie.lookup.get(&worker_0).unwrap().len(), 3);

        // Verify that blocks have the correct workers
        let block_1 = trie
            .lookup
            .get(&worker_0)
            .unwrap()
            .get(&ExternalSequenceBlockHash(100))
            .unwrap();
        assert_eq!(block_1.borrow().workers.len(), 3); // worker_0, worker_1, and worker_2 (all have hash 1)
        assert!(block_1.borrow().workers.contains_key(&worker_0));
        assert!(block_1.borrow().workers.contains_key(&worker_1));
        assert!(block_1.borrow().workers.contains_key(&worker_2));

        // Remove worker_0
        trie.remove_worker(worker_0);

        // Verify worker_0 is completely removed from lookup table
        assert!(!trie.lookup.contains_key(&worker_0));
        assert_eq!(trie.lookup.len(), 2);

        // Verify that worker_0's hash is removed from the workers set
        let block_1 = trie
            .lookup
            .get(&worker_1)
            .unwrap()
            .get(&ExternalSequenceBlockHash(100))
            .unwrap();
        assert_eq!(block_1.borrow().workers.len(), 2); // worker_1 and worker_2 remain
        assert!(!block_1.borrow().workers.contains_key(&worker_0));
        assert!(block_1.borrow().workers.contains_key(&worker_1));
        assert!(block_1.borrow().workers.contains_key(&worker_2));

        // Verify that blocks with no remaining workers have their children cleared
        // This tests the optimization where empty blocks clear their children
        let block_2 = trie
            .lookup
            .get(&worker_1)
            .unwrap()
            .get(&ExternalSequenceBlockHash(200))
            .unwrap();
        assert_eq!(block_2.borrow().workers.len(), 1); // only worker_1
        assert!(block_2.borrow().workers.contains_key(&worker_1));

        // Verify match results no longer include worker_0
        let result = trie
            .find_matches(
                vec![LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)],
                false,
            )
            .scores;
        assert_eq!(result.len(), 2);
        assert!(!result.contains_key(&worker_0));
        assert!(result.contains_key(&worker_1));
        assert!(result.contains_key(&worker_2));
    }
}
