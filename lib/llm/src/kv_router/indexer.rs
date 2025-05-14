// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

use bytes::Bytes;
// use prometheus::{IntCounter, IntGauge};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    iter,
    rc::Rc,
    sync::OnceLock,
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use xxhash_rust::xxh3;

pub const XXH3_SEED: u64 = 1337;

use crate::kv_router::protocols::*;

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
pub fn compute_block_hash_for_seq(tokens: &[u32], kv_block_size: usize) -> Vec<LocalBlockHash> {
    tokens
        .chunks_exact(kv_block_size) // Split into chunks of kv_block_size elements
        .map(|chunk| {
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|&num| num.to_le_bytes()) // Convert each i32 to its little-endian bytes
                .collect();

            compute_block_hash(&Bytes::from(bytes)) // Convert the byte Vec to Bytes
        })
        .collect()
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
    /// A set of worker IDs associated with this block.
    workers: HashSet<WorkerId>,
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
            workers: HashSet::new(),
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
        for block_hash in sequence {
            let next_block = {
                let current_borrow = current.borrow();
                current_borrow.children.get(&block_hash).cloned()
            };
            if let Some(block) = next_block {
                scores.update_scores(&block.borrow().workers);

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
                break;
            }
        }

        scores
    }

    /// Apply a [`RouterEvent`] to the radix tree.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    pub fn apply_event(&mut self, event: RouterEvent) {
        let (worker_id, event) = (event.worker_id, event.event);
        let (id, op) = (event.event_id, event.data);
        tracing::trace!(id, "Store operation: {:?}", op);

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
                        return;
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

                    // add our worker_id to the block
                    block.borrow_mut().workers.insert(worker_id);

                    // add the block to the worker_id lookup table
                    worker_lookup.insert(block_id.block_hash, block.clone());

                    // drop inner so we can shift current to this block
                    drop(inner);

                    current = block;
                }
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
                            continue;
                        }
                    };

                    let mut guard = entry.borrow_mut();
                    guard.workers.remove(&worker_id);
                    if guard.workers.is_empty() {
                        // if no worker are using this block, that is true for all children
                        guard.children.clear();
                    }
                    // remove the block from the lookup table
                    worker_lookup.remove(&block);
                }
            }
        }
    }

    pub fn remove_worker(&mut self, worker: WorkerId) {
        if let Some((_, blocks)) = self.lookup.remove_entry(&worker) {
            blocks.iter().for_each(|(_, block)| {
                block.borrow_mut().workers.remove(&worker);
            });
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
    /// * `workers` - A reference to a `HashSet` of `WorkerId`s.
    pub fn update_scores(&mut self, workers: &HashSet<WorkerId>) {
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
    /// A handle to the background task managing the KV store.
    task: OnceLock<std::thread::JoinHandle<()>>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: usize,
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
        kv_block_size: usize,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(2048);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
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
                    let mut trie = RadixTree::new_with_frequency(expiration_duration);
                    loop {
                        tokio::select! {
                            biased;

                            Some(worker) = remove_worker_rx.recv() => {
                                trie.remove_worker(worker);
                            }

                            Some(req) = match_rx.recv() => {
                                let matches = trie.find_matches(req.sequence, req.early_exit);
                                let _ = req.resp.send(matches);
                            }

                            _ = cancel.cancelled() => {
                                tracing::debug!("KvCacheIndexer progress loop shutting down");
                                return;
                            }

                            Some(event) = event_rx.recv() => {
                                trie.apply_event(event);
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
            task: once,
            kv_block_size,
        }
    }

    pub fn block_size(&self) -> usize {
        self.kv_block_size
    }

    pub fn new(token: CancellationToken, kv_block_size: usize) -> Self {
        Self::new_with_frequency(token, None, kv_block_size)
    }

    /// Get a sender for `RouterEvent`s.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.event_tx.clone()
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
}

#[derive(Debug, Clone)]
pub struct ShardedMatchRequest {
    sequence: Vec<LocalBlockHash>,
    early_exit: bool,
    resp: mpsc::Sender<OverlapScores>,
}

/// The KV Indexer, managing the KV store and handling events and match requests.
pub struct KvIndexerSharded {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// The size of the KV block this indexer can handle.
    kv_block_size: usize,
    worker_assignments: HashMap<WorkerId, usize>,
    worker_counts: Vec<usize>,

    event_tx: Vec<mpsc::Sender<RouterEvent>>,
    request_broadcast_tx: broadcast::Sender<ShardedMatchRequest>,
    remove_worker_tx: Vec<mpsc::Sender<WorkerId>>,
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
        kv_block_size: usize,
    ) -> Self {
        let worker_assignments: HashMap<WorkerId, usize> = HashMap::new();
        let worker_counts: Vec<usize> = vec![0; num_shards];

        let mut event_tx = Vec::new();
        let mut remove_worker_tx = Vec::new();
        let mut tasks = Vec::new();

        let (request_broadcast_tx, _) = broadcast::channel::<ShardedMatchRequest>(1048576);

        for _ in 0..num_shards {
            let (shard_event_tx, mut shard_event_rx) = mpsc::channel::<RouterEvent>(2048);
            let (shard_remove_worker_tx, mut shard_remove_worker_rx) =
                mpsc::channel::<WorkerId>(16);
            let mut shard_broadcast_rx = request_broadcast_tx.subscribe();
            let cancel = token.clone();

            event_tx.push(shard_event_tx);
            remove_worker_tx.push(shard_remove_worker_tx);

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

                                Some(worker) = shard_remove_worker_rx.recv() => {
                                    trie.remove_worker(worker);
                                }

                                Ok(req) = shard_broadcast_rx.recv() => {
                                    let matches = trie.find_matches(req.sequence, req.early_exit);
                                    if let Err(e) = req.resp.send(matches).await {
                                        tracing::trace!("Failed to send match response: {:?}", e);
                                    }
                                }

                                _ = cancel.cancelled() => {
                                    tracing::debug!("KvCacheIndexer progress loop shutting down");
                                    return;
                                }

                                Some(event) = shard_event_rx.recv() => {
                                    trie.apply_event(event);
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
            tasks,
        }
    }

    pub fn block_size(&self) -> usize {
        self.kv_block_size
    }

    pub fn new(token: CancellationToken, num_shards: usize, kv_block_size: usize) -> Self {
        Self::new_with_frequency(token, num_shards, None, kv_block_size)
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
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use tokio::time;
    use tokio_util::sync::CancellationToken;

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
        let mut trie = RadixTree::new();

        let worker_1 = 0;
        let worker_2 = 1;

        trie.apply_event(create_store_event(worker_1, 1, vec![1, 2, 3], None));

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

        trie.apply_event(create_store_event(worker_2, 1, vec![1, 4, 5], None));

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

        trie.apply_event(create_remove_event(worker_2, 2, vec![5]));
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

        trie.apply_event(create_remove_event(worker_2, 3, vec![4]));

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
        ));

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
    fn test_remove_worker() {
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        assert!(trie
            .find_matches(vec![LocalBlockHash(0)], false)
            .scores
            .is_empty());

        trie.apply_event(create_store_event(worker_0, 0, vec![0], None));
        trie.apply_event(create_store_event(worker_1, 0, vec![0], None));

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(result.len() == 2 && result[&worker_0] == 1 && result[&worker_1] == 1);

        trie.remove_worker(worker_0);

        let result = trie.find_matches(vec![LocalBlockHash(0)], false).scores;
        assert!(result.len() == 1 && result[&worker_1] == 1);
    }

    #[test]
    fn test_early_stopping() {
        let mut trie = RadixTree::new();

        let worker_0 = 0;
        let worker_1 = 1;

        trie.apply_event(create_store_event(worker_0, 0, vec![0, 1, 2], None));
        trie.apply_event(create_store_event(worker_1, 0, vec![0], None));

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
    fn test_compute_block_hash_for_seq(#[case] kv_block_size: usize) {
        // create a sequence of 64 elements
        let sequence = (0..kv_block_size).map(|i| i as u32).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 65 elements
        let sequence = (0..(kv_block_size + 1))
            .map(|i| i as u32)
            .collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 129 elements
        let sequence = (0..(2 * kv_block_size + 1))
            .map(|i| i as u32)
            .collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size);
        assert_eq!(hashes.len(), 2);
    }

    fn make_indexer(
        token: &CancellationToken,
        num_shards: usize,
        kv_block_size: usize,
    ) -> Box<dyn KvIndexerInterface> {
        if num_shards == 1 {
            Box::new(KvIndexer::new(token.clone(), kv_block_size))
        } else {
            Box::new(KvIndexerSharded::new(
                token.clone(),
                num_shards,
                kv_block_size,
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
    async fn test_kv_indexer_new(num_shards: usize, kv_block_size: usize) {
        let token: CancellationToken = CancellationToken::new();
        let _ = make_indexer(&token, num_shards, kv_block_size);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_find_matches(num_shards: usize, kv_block_size: usize) {
        let token = CancellationToken::new();
        let kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let sequence = vec![compute_block_hash(b"test data")];
        let scores = kv_indexer.find_matches(sequence).await;

        assert!(scores.unwrap().scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_find_matches_for_request(num_shards: usize, kv_block_size: usize) {
        let token = CancellationToken::new();
        let kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let tokens = vec![1, 2, 3, 4];
        let scores = kv_indexer.find_matches_for_request(&tokens).await;

        assert!(scores.unwrap().scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_apply_event(num_shards: usize, kv_block_size: usize) {
        let worker_id = 0;

        let token = CancellationToken::new();
        let mut kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        let event = create_store_event(worker_id, 1, vec![1, 2, 3], None);
        kv_indexer.apply_event(event).await;

        // No assertion here, just ensuring it runs without panic
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_shutdown(num_shards: usize, kv_block_size: usize) {
        let token = CancellationToken::new();
        let mut kv_indexer = make_indexer(&token, num_shards, kv_block_size);

        kv_indexer.shutdown();
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_frequency(num_shards: usize, kv_block_size: usize) {
        let mut kv_indexer: Box<dyn KvIndexerInterface>;
        let token = CancellationToken::new();
        let duration = Some(Duration::from_millis(50));

        if num_shards == 1 {
            kv_indexer = Box::new(KvIndexer::new_with_frequency(
                token,
                duration,
                kv_block_size,
            ));
        } else {
            kv_indexer = Box::new(KvIndexerSharded::new_with_frequency(
                token,
                num_shards,
                duration,
                kv_block_size,
            ));
        }

        let worker_id = 0;

        let event = create_store_event(worker_id, 0, vec![1, 2, 3, 4], None);
        kv_indexer.apply_event(event).await;

        time::sleep(Duration::from_millis(5)).await;

        let block_hashes = vec![
            LocalBlockHash(1),
            LocalBlockHash(2),
            LocalBlockHash(3),
            LocalBlockHash(4),
        ];
        let scores = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();

        assert_eq!(scores.frequencies.len(), 0);

        let scores = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(scores.frequencies, vec![1, 1, 1, 1]);

        time::sleep(Duration::from_millis(100)).await;

        let scores = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(scores.frequencies.len(), 0);

        let scores = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(scores.frequencies, vec![1, 1, 1, 1]);

        let scores = kv_indexer
            .find_matches(block_hashes[0..3].to_vec())
            .await
            .unwrap();
        assert_eq!(scores.frequencies, vec![2, 2, 2]);

        let scores = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(scores.frequencies, vec![3, 3, 3, 2]);
    }

    #[test]
    fn test_router_event_new() {
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
        let radix_tree: RadixTree = Default::default();
        assert!(radix_tree.root.borrow().children.is_empty());
        assert!(radix_tree.root.borrow().workers.is_empty());
        assert!(radix_tree.lookup.is_empty());
    }

    #[test]
    fn test_overlap_scores_default() {
        let overlap_scores: OverlapScores = Default::default();
        assert!(overlap_scores.scores.is_empty());
    }
}
