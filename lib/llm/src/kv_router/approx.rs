// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Approximate KV Indexer
//!
//! - This module implements an approximate KV indexer that can be used to find matches for a given sequence of tokens.
//! - It is designed to be used in conjunction with the KV router to find matches for a given sequence of tokens.
//!
//! # Overview
//!
//! - The Approximate KV Indexer, unlike the regular KV Indexer, does not depend on KV events.
//! - The approximate indexer depends only on the input tokens. We can use input tokens + our routing decision to approximate the radix trees across workers.
//!
//! - The thinking behind this is that if we send a request to a worker, and shortly after get a request with a similar prefix, odds
//!   are that routing to the same worker will result in a large cache hit.
//! - Another benefit is the ability to bound the size of the radix tree, which is not possible if we were trying to accurately represent
//!   the state of each worker.

use async_trait::async_trait;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use std::sync::OnceLock;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::tokens::{SequenceHash, TokenBlockSequence};

use crate::kv_router::indexer::{
    DumpRequest, KvIndexerInterface, KvRouterError, OverlapScores, RadixTree, RouterEvent,
    compute_block_hash_for_seq,
};
use crate::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, WorkerId, WorkerWithDpRank,
};

#[derive(Debug)]
struct MatchRequest {
    /// Sequence of tokens.
    sequence: Vec<LocalBlockHash>,
    /// A channel to send the `OverlapScores` response.
    resp: oneshot::Sender<OverlapScores>,
}

#[derive(Debug)]
struct RouterResult {
    /// The worker (with dp_rank) that was selected.
    worker: WorkerWithDpRank,

    /// The local hashes of the tokens sent to the worker.
    local_hashes: Vec<LocalBlockHash>,

    /// The sequence hashes of the tokens sent to the worker.
    sequence_hashes: Vec<u64>,
}

/// Block entry to be inserted in the [`PruneManager::expirations`] heap.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct BlockEntry {
    /// The key of the block entry.
    key: ExternalSequenceBlockHash,
    /// The worker (with dp_rank) that stored this block.
    worker: WorkerWithDpRank,
    /// The position of this block in the sequence (0-indexed).
    seq_position: usize,
}

impl PartialOrd for BlockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Break ties by sequence position (important for pruning), then by key, then by worker.
        self.seq_position
            .cmp(&other.seq_position)
            .then_with(|| self.key.cmp(&other.key))
            .then_with(|| self.worker.cmp(&other.worker))
    }
}

#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// The maximum tree size before pruning is considered.
    pub max_tree_size: usize,
    /// The target size ratio to prune down to when max_tree_size is exceeded.
    /// For example, if max_tree_size is 100 and target_size_ratio is 0.5,
    /// we will prune down to 50 nodes when max_tree_size is exceeded.
    pub prune_target_ratio: f64,
}

/// A data structure to manage a collection of timers, addressable by a key.
/// This is structured as a sort of "priority queue" of keys, where the priority is the expiration time.
/// It supports insertion as well as updating the expiration time of a key.
/// The [`PruneManager::expirations`] heap is lazily updated to reflect the true expiration times in [`PruneManager::timers`]
/// For now, we have a fixed expiration time for all keys.
#[derive(Debug)]
struct PruneManager<K: Clone + Hash + Eq + Ord> {
    /// The source of truth. Maps a key to its current expiration instant.
    timers: HashMap<K, Instant>,

    /// A max-heap of (Reverse<expiration_instant>, key) used to efficiently find the
    /// next expiring timer. Reverse<Instant> makes earlier times pop first.
    /// An entry in this heap is "stale" if the instant does not match the one in the `timers` map.
    expirations: BinaryHeap<(Reverse<Instant>, K)>,

    /// Threshold for rebuilding the heap.
    /// The heap will be rebuilt from scratch to remove stale entries.
    threshold: usize,

    /// The expiration duration of the timers.
    ttl: Duration,

    /// The configuration for tree-size pruning.
    prune_config: Option<PruneConfig>,
}

impl<K: Clone + Hash + Eq + Ord> PruneManager<K> {
    /// Creates a new, empty PruneManager.
    pub fn new(ttl: Duration, threshold: usize, prune_config: Option<PruneConfig>) -> Self {
        PruneManager {
            timers: HashMap::new(),
            expirations: BinaryHeap::new(),
            ttl,
            threshold,
            prune_config,
        }
    }

    /// Rebuilds the expirations heap from the timers map, removing all stale entries.
    fn rebuild_heap(&mut self) {
        self.expirations = self
            .timers
            .iter()
            .map(|(key, &expiry)| (Reverse(expiry), key.clone()))
            .collect();
    }

    /// Inserts a new timer or updates an existing one for the given key.
    ///
    /// # Arguments
    /// * `key` - The unique key for the timer.
    /// * `duration` - The duration from now when the timer should expire.
    pub fn insert(&mut self, keys: Vec<K>) {
        let expiry_time = Instant::now() + self.ttl;

        for key in keys {
            // Insert or update the authoritative time in the map.
            self.timers.insert(key.clone(), expiry_time);

            // Push the new expiration onto the heap. If the key was updated,
            // this leaves a "stale" entry on the heap for the old time,
            // which will be ignored when it's popped.
            self.expirations.push((Reverse(expiry_time), key));
        }

        // Check if we should rebuild the heap to remove stale entries
        if self.expirations.len() > self.timers.len() * self.threshold {
            self.rebuild_heap();
        }
    }

    /// Polls for expired timers and returns a list of keys for all timers
    /// that have expired up to the current moment.
    pub fn pop_expired(&mut self) -> Vec<K> {
        let mut expired_keys = Vec::new();
        let now = Instant::now();

        while let Some((Reverse(expiry_time), _)) = self.expirations.peek() {
            // If the next timer in the heap is not yet expired, we can stop.
            if *expiry_time > now {
                break;
            }

            // The timer might be expired, so pop it from the heap.
            let (Reverse(expiry_time), key) = self.expirations.pop().unwrap();

            if self.timers.get(&key) == Some(&expiry_time) {
                // This is a valid, non-stale, expired timer.
                self.timers.remove(&key);
                expired_keys.push(key);
            }
        }

        expired_keys
    }

    /// Returns the next expiry time, if it exists.
    pub fn peek_next_expiry(&self) -> Option<Instant> {
        self.expirations
            .peek()
            .map(|(Reverse(expiry_time), _)| *expiry_time)
    }

    /// Prunes the tree if the current size is greater than the max tree size.
    pub fn prune(&mut self, current_size: usize) -> Result<Vec<K>, KvRouterError> {
        let max_tree_size: usize;
        let prune_target_ratio: f64;

        if let Some(prune_config) = &self.prune_config {
            max_tree_size = prune_config.max_tree_size;
            prune_target_ratio = prune_config.prune_target_ratio;
        } else {
            tracing::error!("Prune was called but prune config is None. This should never happen");
            return Err(KvRouterError::PruneFailed(
                "prune config is missing".to_string(),
            ));
        }

        if current_size <= max_tree_size {
            // Tree size within bounds, no pruning needed.
            return Ok(Vec::new());
        }

        tracing::info!(
            "Pruning: tree size ({}) exceeded max tree size ({}), starting pruning",
            current_size,
            max_tree_size
        );

        // Number of blocks that will be kept after pruning.
        let target_size = (max_tree_size as f64 * prune_target_ratio) as usize;

        let mut pruned_keys = Vec::new();
        let mut num_pruned = 0;

        while num_pruned < current_size.saturating_sub(target_size) {
            if let Some((Reverse(expiry_time), key)) = self.expirations.pop() {
                if self.timers.get(&key) == Some(&expiry_time) {
                    // This is a valid, non-stale timer.
                    self.timers.remove(&key);
                    pruned_keys.push(key);
                    num_pruned += 1;
                }
            } else {
                break;
            }
        }

        tracing::info!("Pruning: pruned ({}) blocks from tree", num_pruned);

        Ok(pruned_keys)
    }
}

pub struct ApproxKvIndexer {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// A sender for `MatchRequest`s.
    match_tx: mpsc::Sender<MatchRequest>,
    /// A sender for `RouterResult`s.
    route_tx: mpsc::Sender<RouterResult>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A handle to the background task managing the KV store.
    task: OnceLock<std::thread::JoinHandle<()>>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
}

impl ApproxKvIndexer {
    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        ttl: Duration,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let (match_tx, mut match_rx) = mpsc::channel::<MatchRequest>(2048);
        let (route_tx, mut route_rx) = mpsc::channel::<RouterResult>(2048);
        let (remove_worker_tx, mut remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (_get_workers_tx, mut get_workers_rx) =
            mpsc::channel::<super::indexer::GetWorkersRequest>(16);
        let (dump_tx, mut dump_rx) = mpsc::channel::<DumpRequest>(16);
        let (prune_tx, mut prune_rx) = watch::channel(false);
        let cancel_clone = token.clone();
        let task = std::thread::spawn(move || {
            // create a new tokio runtime which will only perform work on a single thread
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let mut trie = RadixTree::new();
                // Use a reasonable threshold for ttl - can be made configurable if needed
                let mut prune_manager: PruneManager<BlockEntry> = PruneManager::new(ttl, 50, prune_config.clone());
                let mut event_id = 0;

                loop {
                    // Create a future that sleeps until the next expiration time.
                    let expiry_fut = if let Some(next_expiry) = prune_manager.peek_next_expiry() {
                        tokio::time::sleep_until(next_expiry)
                    } else {
                        // If there are no timers, sleep forever.
                        tokio::time::sleep(Duration::MAX)
                    };

                    tokio::select! {
                        _ = cancel_clone.cancelled() => {
                            tracing::debug!("Approximate Indexer progress loop shutting down");
                            return;
                        }

                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                        }

                        Some(get_workers_req) = get_workers_rx.recv() => {
                            let workers = trie.get_workers();
                            let _ = get_workers_req.resp.send(workers);
                        }

                        Some(result) = route_rx.recv() => {
                            let hashes = result.local_hashes.iter().zip(result.sequence_hashes.iter());

                            let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash: None,
                                blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                    tokens_hash: *local_hash,
                                    block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                }).collect(),
                            });
                            event_id += 1;

                            let event = RouterEvent::new(
                                result.worker.worker_id,
                                KvCacheEvent {
                                    event_id,
                                    data: stored_event,
                                    dp_rank: result.worker.dp_rank,
                                }
                            );

                            if trie.apply_event(event).is_ok() {
                                prune_manager.insert(result.sequence_hashes.iter().enumerate().map(|(idx, h)| BlockEntry {
                                    key: ExternalSequenceBlockHash(*h),
                                    worker: result.worker,
                                    seq_position: idx,
                                }).collect());

                                // Check if we need to prune due to tree size exceeding max threshold.
                                if let Some(prune_config) = &prune_manager.prune_config {
                                    let current_size = trie.current_size();
                                    if current_size > prune_config.max_tree_size {
                                        tracing::info!(
                                            "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                            current_size,
                                            prune_config.max_tree_size
                                        );
                                        // Send a signal to the pruning watcher to schedule pruning.
                                        if let Err(e) = prune_tx.send(true) {
                                            tracing::error!("Failed to send prune schedule signal: {:?}", e);
                                        }
                                    }
                                }
                            }
                        }

                        Some(dump_req) = dump_rx.recv() => {
                            let events = trie.dump_tree_as_events();
                            let _ = dump_req.resp.send(events);
                        }

                        Some(request) = match_rx.recv() => {
                            let scores = trie.find_matches(request.sequence, false);
                            request.resp.send(scores).unwrap();
                        }

                        Ok(_) = prune_rx.changed() => {
                            // The tree has exceeded the max tree size, so proceed with pruning.
                            if let Ok(pruned) = prune_manager.prune(trie.current_size()) {
                                pruned.iter().for_each(|p| {
                                    event_id += 1;

                                    let event = RouterEvent::new(
                                        p.worker.worker_id,
                                        KvCacheEvent {
                                            event_id,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![p.key],
                                            }),
                                            dp_rank: p.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                });
                                // Reset the pruning watcher to false to indicate that pruning is complete.
                                if let Err(e) = prune_tx.send(true) {
                                    tracing::error!("Failed to send prune completion signal: {:?}", e);
                                }
                            }
                        }

                        _ = expiry_fut => {
                            let expired = prune_manager.pop_expired();

                            expired.iter().for_each(|e| {
                                event_id += 1;

                                let event = RouterEvent::new(
                                    e.worker.worker_id,
                                    KvCacheEvent {
                                        event_id,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![e.key],
                                        }),
                                        dp_rank: e.worker.dp_rank,
                                    }
                                );

                                let _ = trie.apply_event(event);
                            });
                        }
                    }
                }
            });
        });

        let once = OnceLock::new();
        once.set(task).unwrap();

        Self {
            cancel: token,
            match_tx,
            route_tx,
            remove_worker_tx,
            dump_tx,
            task: once,
            kv_block_size,
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    /// Core function to process a routing decision with pre-computed hashes
    pub async fn process_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.route_tx
            .send(RouterResult {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;

        Ok(())
    }

    /// Wrapper function that computes hashes from tokens and calls the core function
    pub async fn process_routing_decision_for_request(
        &self,
        tokens: &[u32],
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = compute_block_hash_for_seq(tokens, self.kv_block_size);

        let sequence = TokenBlockSequence::new(tokens.into(), self.kv_block_size, None);
        let sequence_hashes = sequence
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        self.process_routing_decision(worker, local_hashes, sequence_hashes)
            .await
    }
}

#[async_trait]
impl KvIndexerInterface for ApproxKvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let request = MatchRequest {
            sequence,
            resp: resp_tx,
        };

        if let Err(e) = self.match_tx.send(request).await {
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
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size);
        self.find_matches(sequence).await
    }

    async fn apply_event(&mut self, _event: RouterEvent) {
        panic!("Approximate Indexer does not support apply_event");
    }

    async fn remove_worker(&mut self, worker: WorkerId) {
        self.remove_worker_tx.send(worker).await.unwrap();
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

    fn shutdown(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.join()
                .expect("Failed to join approximate indexer task");
        }
    }
}

impl Drop for ApproxKvIndexer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::time::{self, Duration, Instant};
    use tokio_util::sync::CancellationToken;

    const KV_BLOCK_SIZE: u32 = 4;

    impl<T: Clone + Hash + Eq + Ord> PruneManager<T> {
        pub fn get_expiry(&self, key: &T) -> Option<&Instant> {
            self.timers.get(key)
        }
    }

    /// Helper to spin until a future evaluates to `true`, or a timeout is reached.
    async fn spin_until<F, Fut>(timeout: Duration, mut predicate: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let start = Instant::now();
        const POLL: Duration = Duration::from_millis(1);
        loop {
            if predicate().await {
                return;
            }
            if Instant::now().duration_since(start) >= timeout {
                panic!("timeout waiting for condition");
            }
            time::sleep(POLL).await;
        }
    }

    /// Validate basic insert / expiry behaviour of [`PruneManager`].
    #[tokio::test]
    async fn test_prune_manager_expiry() {
        const TTL: Duration = Duration::from_millis(50);
        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, None);

        pm.insert(vec![1, 2, 3]);
        assert!(pm.get_expiry(&1).is_some());
        assert!(pm.get_expiry(&2).is_some());
        assert!(pm.get_expiry(&3).is_some());

        // Wait until after the TTL
        time::sleep(TTL + Duration::from_millis(20)).await;
        let expired = pm.pop_expired();
        assert_eq!(expired.len(), 3);
        assert!(pm.get_expiry(&1).is_none());
        assert!(pm.get_expiry(&2).is_none());
        assert!(pm.get_expiry(&3).is_none());
    }

    /// Validate that reinserting an existing key extends its TTL and prevents premature expiry.
    #[tokio::test]
    async fn test_prune_manager_update_resets_ttl() {
        // Validate that reinserting an existing key extends its TTL and prevents premature expiry.
        const TTL: Duration = Duration::from_millis(50);
        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, None);

        // Initial insert and capture the original expiry.
        pm.insert(vec![42]);
        let first_expiry = *pm
            .get_expiry(&42)
            .expect("expiry missing after first insert");

        // Wait for half of the original TTL before reinserting.
        time::sleep(Duration::from_millis(25)).await;
        pm.insert(vec![42]);
        let second_expiry = *pm
            .get_expiry(&42)
            .expect("expiry missing after reinsertion");

        // The expiry after reinsertion must be strictly later than the first one.
        assert!(second_expiry > first_expiry);

        // Wait until *after* the first expiry would have fired, but *before* the new expiry.
        time::sleep(Duration::from_millis(30)).await; // 25ms already elapsed, +30ms = 55ms > first TTL
        let expired = pm.pop_expired();
        assert!(
            expired.is_empty(),
            "key expired prematurely despite TTL refresh"
        );

        // Now wait until after the second expiry should have occurred.
        time::sleep(Duration::from_millis(30)).await; // Ensure we pass the refreshed TTL
        let expired_after = pm.pop_expired();
        assert_eq!(expired_after, vec![42]);
    }

    /// End-to-end test for [`ApproxKvIndexer`]:
    ///   1. No matches before routing decision
    ///   2. Matches appear after `process_routing_decision`
    ///   3. Matches disappear after TTL expiry
    #[tokio::test]
    async fn test_approx_kv_indexer_basic_flow() {
        const TTL: Duration = Duration::from_millis(200);
        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, None);

        let tokens: Vec<u32> = vec![1, 2, 3, 4]; // Exactly one KV block
        let worker_id: WorkerId = 0;

        // 1. Before routing decision there should be no matches
        let pre_scores = indexer
            .find_matches_for_request(&tokens)
            .await
            .expect("indexer offline");
        assert!(pre_scores.scores.is_empty());

        // 2. Inform indexer about routing decision
        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_id),
            )
            .await
            .unwrap();

        // Poll until we observe the match being registered
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores
                .get(&WorkerWithDpRank::from_worker_id(worker_id))
                .copied()
                == Some(1)
        })
        .await;

        // 3. After the TTL has passed the entry should expire automatically
        time::sleep(TTL + Duration::from_millis(50)).await;
        let post_scores = indexer.find_matches_for_request(&tokens).await.unwrap();
        assert!(post_scores.scores.is_empty());
    }

    /// Verify that `remove_worker` clears all entries for the specified worker.
    #[tokio::test]
    async fn test_remove_worker() {
        const TTL: Duration = Duration::from_secs(5); // Large enough to avoid expiry during test
        let cancel = CancellationToken::new();
        let mut indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, None);

        let tokens: Vec<u32> = vec![10, 11, 12, 13];
        let worker_id: WorkerId = 7;

        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_id),
            )
            .await
            .unwrap();

        // Wait until the worker is registered
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_id))
        })
        .await;

        // Remove the worker
        indexer.remove_worker(worker_id).await;

        // Ensure the worker's entries are gone
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            !s.scores
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_id))
        })
        .await;
    }

    /// After removing one of multiple workers that share the same block, the remaining worker's entries should persist.
    #[tokio::test]
    async fn test_remove_worker_preserves_other_workers() {
        const TTL: Duration = Duration::from_secs(5); // Large enough to avoid expiry during test

        let cancel = CancellationToken::new();
        let mut indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, None);

        let tokens: Vec<u32> = vec![100, 101, 102, 103];
        let worker_0: WorkerId = 30;
        let worker_1: WorkerId = 31;

        // Register on both workers
        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_0),
            )
            .await
            .unwrap();
        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_1),
            )
            .await
            .unwrap();

        // Ensure both workers are registered
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .copied()
                == Some(1)
                && s.scores
                    .get(&WorkerWithDpRank::from_worker_id(worker_1))
                    .copied()
                    == Some(1)
        })
        .await;

        // Remove one worker
        indexer.remove_worker(worker_0).await;

        // Confirm the removed worker is gone, and the other remains.
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            !s.scores
                .contains_key(&WorkerWithDpRank::from_worker_id(worker_0))
                && s.scores
                    .get(&WorkerWithDpRank::from_worker_id(worker_1))
                    .copied()
                    == Some(1)
        })
        .await;
    }

    /// Two sequences with a shared prefix should yield overlap scores reflecting the common blocks.
    #[tokio::test]
    async fn test_common_prefix_overlap() {
        const TTL: Duration = Duration::from_secs(5);

        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, None);

        // Sequence A : single block
        let seq_a: Vec<u32> = vec![1, 2, 3, 4];
        let worker_a: WorkerId = 11;

        // Register Sequence A on worker A
        indexer
            .process_routing_decision_for_request(
                &seq_a,
                WorkerWithDpRank::from_worker_id(worker_a),
            )
            .await
            .unwrap();

        // Ensure the indexer has registered the block
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&seq_a).await.unwrap();
            s.scores
                .get(&WorkerWithDpRank::from_worker_id(worker_a))
                .copied()
                == Some(1)
        })
        .await;

        // Sequence B : shares the first block with Sequence A, plus an extra block
        let seq_b: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Query the indexer for overlaps of Sequence B (before it has been routed anywhere)
        let overlap = indexer.find_matches_for_request(&seq_b).await.unwrap();

        // Expect worker A to have an overlap score of 1 (shared first block)
        assert_eq!(
            overlap
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_a)),
            Some(&1)
        );
    }

    /// When the same block resides on multiple workers, all should appear in the overlap scores.
    #[tokio::test]
    async fn test_multiple_workers_same_block() {
        const TTL: Duration = Duration::from_secs(5);

        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, None);

        let tokens: Vec<u32> = vec![9, 8, 7, 6];
        let worker_0: WorkerId = 21;
        let worker_1: WorkerId = 22;

        // Register the same sequence on two different workers
        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_0),
            )
            .await
            .unwrap();
        indexer
            .process_routing_decision_for_request(
                &tokens,
                WorkerWithDpRank::from_worker_id(worker_1),
            )
            .await
            .unwrap();

        // Wait until both workers are reflected in overlap scores
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores
                .get(&WorkerWithDpRank::from_worker_id(worker_0))
                .copied()
                == Some(1)
                && s.scores
                    .get(&WorkerWithDpRank::from_worker_id(worker_1))
                    .copied()
                    == Some(1)
        })
        .await;

        let scores = indexer.find_matches_for_request(&tokens).await.unwrap();

        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_0)),
            Some(&1)
        );
        assert_eq!(
            scores
                .scores
                .get(&WorkerWithDpRank::from_worker_id(worker_1)),
            Some(&1)
        );
    }

    /// Test that pruning returns empty when tree size is within the max tree size.
    #[tokio::test]
    async fn test_prune_manager_no_prune_when_within_bounds() {
        const TTL: Duration = Duration::from_secs(10);
        let prune_config = PruneConfig {
            max_tree_size: 100,
            prune_target_ratio: 0.5,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, Some(prune_config));

        // Insert 50 keys (well below max_tree_size of 100)
        pm.insert((0..50).collect());

        // Pruning should return empty vec when size is within bounds
        let pruned = pm.prune(50).unwrap();
        assert!(pruned.is_empty());

        // All keys should still be present
        for i in 0..50 {
            assert!(pm.get_expiry(&i).is_some());
        }
    }

    /// Test that pruning removes the oldest entries first.
    #[tokio::test]
    async fn test_prune_manager_prune_removes_oldest_first() {
        const TTL: Duration = Duration::from_secs(10);
        let prune_config = PruneConfig {
            max_tree_size: 10,
            prune_target_ratio: 0.5,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, Some(prune_config));

        // Insert keys one at a time with delays to ensure different timestamps
        for i in 1..=15 {
            pm.insert(vec![i]);
            time::sleep(Duration::from_millis(1)).await;
        }

        // Total: 15 keys. Trigger pruning with current_size = 15
        let pruned = pm.prune(15).unwrap();

        // Should prune down to 5 (10 * 0.5), so 10 keys should be pruned (15 - 5)
        assert_eq!(pruned.len(), 10);

        // The oldest keys should be pruned first
        for i in 1..=10 {
            assert!(pruned.contains(&i));
        }

        // The newer keys should still be present
        for i in 11..=15 {
            assert!(pm.get_expiry(&i).is_some());
        }
    }

    /// Test that pruning fails gracefully when config is None.
    #[tokio::test]
    async fn test_prune_manager_prune_fails_without_config() {
        const TTL: Duration = Duration::from_secs(10);
        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, None);

        pm.insert(vec![1, 2, 3]);

        // Pruning should fail when prune_config is None
        let result = pm.prune(150);
        assert!(result.is_err());
        assert!(matches!(result, Err(KvRouterError::PruneFailed(_))));
    }

    /// Test that BlockEntry ordering prioritizes sequence position.
    #[test]
    fn test_block_entry_ordering() {
        let worker = WorkerWithDpRank::from_worker_id(0);

        let entry1 = BlockEntry {
            key: ExternalSequenceBlockHash(100),
            worker,
            seq_position: 0,
        };
        let entry2 = BlockEntry {
            key: ExternalSequenceBlockHash(50),
            worker,
            seq_position: 1,
        };

        // entry1 < entry2 because seq_position 0 < 1
        assert!(entry1 < entry2);
    }

    /// End-to-end test for [`ApproxKvIndexer`] with pruning
    ///   0. Max tree size is 5, target size is 2 (prune_target_ratio = 0.4)
    ///   1. Insert 5 blocks (at max_tree_size but not exceeding)
    ///   2. Verify all 5 blocks are present
    ///   3. Insert 6th block (exceeds threshold, triggers reactive pruning)
    ///   4. Verify pruning occurred: 4 oldest blocks removed
    ///   5. Verify 2 newest blocks remain
    #[tokio::test]
    async fn test_approx_indexer_e2e_pruning() {
        const TTL: Duration = Duration::from_secs(60); // Long TTL to avoid expiry
        let prune_config = PruneConfig {
            max_tree_size: 5,        // Very small to trigger pruning quickly
            prune_target_ratio: 0.4, // target size is 5 * 0.4 = 2
        };

        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL, Some(prune_config));

        let worker = WorkerWithDpRank::from_worker_id(42);

        // Insert 5 sequences (5 blocks total, at max_tree_size but not exceeding)
        for i in 0..5 {
            let tokens: Vec<u32> = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            indexer
                .process_routing_decision_for_request(&tokens, worker)
                .await
                .unwrap();
            time::sleep(Duration::from_millis(1)).await; // Ensure different timestamps
        }

        // Verify all 5 blocks are present (no pruning yet)
        for i in 0..5 {
            let tokens: Vec<u32> = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            let scores = indexer.find_matches_for_request(&tokens).await.unwrap();
            assert_eq!(
                scores.scores.get(&worker).copied(),
                Some(1),
                "Block {} should be present before threshold is exceeded",
                i
            );
        }

        // Insert 6th block - this exceeds max_tree_size and should trigger reactive pruning
        let tokens: Vec<u32> = vec![50, 51, 52, 53];
        indexer
            .process_routing_decision_for_request(&tokens, worker)
            .await
            .unwrap();

        // Wait for pruning to complete
        time::sleep(Duration::from_millis(100)).await;

        // After pruning, we will have exactly 2 blocks (5 * 0.4 = 2)
        // The 2 newest blocks (i=4, i=5) will remain, oldest 4 blocks (i=0,1,2,3) will be pruned

        // Verify that the 4 oldest blocks are pruned
        for i in 0..4 {
            let tokens: Vec<u32> = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            let scores = indexer.find_matches_for_request(&tokens).await.unwrap();
            assert!(
                scores.scores.get(&worker).copied().unwrap_or(0) == 0,
                "Block {} should have been pruned but is still present",
                i
            );
        }

        // Verify the 2 newest blocks are present
        for i in 4..6 {
            let tokens: Vec<u32> = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            let scores = indexer.find_matches_for_request(&tokens).await.unwrap();
            assert_eq!(
                scores.scores.get(&worker).copied(),
                Some(1),
                "Block {} should have been present but was pruned",
                i
            );
        }
    }

    /// Test that re-inserting a key updates its position in the pruning queue.
    #[tokio::test]
    async fn test_prune_manager_prune_reinsertion_updates_position() {
        const TTL: Duration = Duration::from_secs(10);
        let prune_config = PruneConfig {
            max_tree_size: 5,
            prune_target_ratio: 0.8,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(TTL, 50, Some(prune_config));

        // Insert keys
        for i in 1..=10 {
            pm.insert(vec![i]);
            time::sleep(Duration::from_millis(1)).await;
        }

        // Re-insert key 1 (should move it to the back of the queue)
        pm.insert(vec![1]);

        // Total: 10 unique keys. Trigger pruning: current_size = 10, target = 4, so prune 6 keys
        // Order by expiry (oldest first): 2, 3, 4, 5, 6, 7, 8, 9, 10, 1 (re-inserted)
        let pruned = pm.prune(10).unwrap();
        assert_eq!(pruned.len(), 6);

        // The oldest keys (2-7) should be pruned
        for i in 2..=7 {
            assert!(pruned.contains(&i));
        }

        // The newest keys (8-10) should still be present
        for i in 8..=10 {
            assert!(pm.get_expiry(&i).is_some());
        }

        // Key 1 should still be present (it was refreshed and is now near the end)
        assert!(pm.get_expiry(&1).is_some());
    }
}
