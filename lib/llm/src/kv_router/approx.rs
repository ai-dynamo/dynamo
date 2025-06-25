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

use async_trait::async_trait;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::tokens::TokenBlockSequence;

use super::indexer::{
    compute_block_hash_for_seq, KvIndexerInterface, KvRouterError, OverlapScores, RadixTree,
    WorkerId,
};
use super::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use super::RouterEvent;

#[derive(Debug)]
struct MatchRequest {
    /// Sequence of tokens.
    sequence: Vec<LocalBlockHash>,
    /// A channel to send the `OverlapScores` response.
    resp: oneshot::Sender<OverlapScores>,
}

#[derive(Debug)]
struct RouterResult {
    /// The id of the selected worker.
    worker_id: WorkerId,

    /// The local hashes of the sequence.
    local_hashes: Vec<LocalBlockHash>,

    /// The tokens that were sent to the worker.
    sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct TimerEntry {
    /// The key of the timer.
    key: ExternalSequenceBlockHash,
    /// The duration from now when the timer should expire.
    worker: WorkerId,
}

/// A data structure to manage a collection of timers, addressable by a key.
#[derive(Debug)]
pub struct TimerManager<K: Clone + Hash + Eq + Ord> {
    /// The source of truth. Maps a key to its current expiration instant.
    timers: HashMap<K, Instant>,
    /// A min-heap of (expiration_instant, key) used to efficiently find the
    /// next expiring timer. An entry in this heap is "stale" if the instant
    /// does not match the one in the `timers` map.
    expirations: BinaryHeap<Reverse<(Instant, K)>>,

    /// The expiration duration of the timers.
    ttl: Duration,
}

impl<K: Clone + Hash + Eq + Ord> TimerManager<K> {
    /// Creates a new, empty TimerManager.
    pub fn new(ttl: Duration) -> Self {
        TimerManager {
            timers: HashMap::new(),
            expirations: BinaryHeap::new(),
            ttl,
        }
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
            self.expirations.push(Reverse((expiry_time, key)));
        }
    }

    /// Removes a timer, preventing it from expiring.
    ///
    /// Returns true if the timer was present, false otherwise.
    pub fn remove(&mut self, key: &K) -> bool {
        // Just remove it from the map. The heap entry becomes stale.
        self.timers.remove(key).is_some()
    }

    /// Returns the expiration Instant for a given key, if it exists.
    pub fn get_expiry(&self, key: &K) -> Option<&Instant> {
        self.timers.get(key)
    }

    /// Polls for expired timers and returns a list of keys for all timers
    /// that have expired up to the current moment.
    pub fn poll_expired(&mut self) -> Vec<K> {
        let mut expired_keys = Vec::new();
        let now = Instant::now();

        while let Some(Reverse((expiry_time, _))) = self.expirations.peek() {
            // If the next timer in the heap is not yet expired, we can stop.
            if *expiry_time > now {
                break;
            }

            // The timer might be expired, so pop it from the heap.
            // We can safely unwrap because we just peeked.
            let Reverse((expiry_time, key)) = self.expirations.pop().unwrap();

            // CRUCIAL STEP: Check if the popped timer is stale.
            // A timer is stale if its key is no longer in our authoritative map,
            // or if the expiration time in the map is different (i.e., it was updated).
            match self.timers.get(&key) {
                Some(authoritative_expiry) if *authoritative_expiry == expiry_time => {
                    // This is a valid, non-stale, expired timer.
                    // Remove it from the map and add its key to our results.
                    self.timers.remove(&key);
                    expired_keys.push(key);
                }
                _ => {
                    // This entry in the heap was stale. It was either removed
                    // or updated with a new time. We just ignore it and continue.
                }
            }
        }

        expired_keys
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
    /// A handle to the background task managing the KV store.
    task: OnceLock<std::thread::JoinHandle<()>>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: usize,
}

impl ApproxKvIndexer {
    pub fn new(token: CancellationToken, kv_block_size: usize, ttl: Duration) -> Self {
        let (match_tx, mut match_rx) = mpsc::channel::<MatchRequest>(2048);
        let (route_tx, mut route_rx) = mpsc::channel::<RouterResult>(2048);
        let (remove_worker_tx, mut remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let cancel_clone = token.clone();
        let task = std::thread::spawn(move || {
            // create a new tokio runtime which will only perform work on a single thread
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1) // Single-threaded environment
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let mut trie = RadixTree::new();
                let mut timer_manager: TimerManager<TimerEntry> = TimerManager::new(ttl);
                let mut event_id = 0;
                loop {
                    tokio::select! {
                        Some(request) = match_rx.recv() => {
                            let expired = timer_manager.poll_expired();

                            expired.iter().for_each(|e| {
                                event_id += 1;

                                let event = RouterEvent::new(
                                    e.worker,
                                    KvCacheEvent {
                                        event_id,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![e.key],
                                        }),
                                    }
                                );

                                trie.apply_event(event);
                            });

                            let scores = trie.find_matches(request.sequence, false);
                            request.resp.send(scores).unwrap();
                        }
                        Some(result) = route_rx.recv() => {
                            println!("GOT ROUTING DECISION {:?}", result);

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
                                result.worker_id,
                                KvCacheEvent {
                                    event_id,
                                    data: stored_event,
                                }
                            );

                            trie.apply_event(event);

                            timer_manager.insert(result.sequence_hashes.iter().map(|h| TimerEntry {
                                key: ExternalSequenceBlockHash(*h),
                                worker: result.worker_id,
                            }).collect());
                        }
                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                        }
                        _ = cancel_clone.cancelled() => {
                            tracing::debug!("Approximate Indexer progress loop shutting down");
                            return;
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
            task: once,
            kv_block_size,
        }
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

    fn shutdown(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.join()
                .expect("Failed to join approximate indexer task");
        }
    }
}

impl ApproxKvIndexer {
    pub async fn process_routing_decision_for_request(
        &self,
        tokens: &[u32],
        lora_id: u64,
        worker_id: WorkerId,
    ) {
        let local_hashes = compute_block_hash_for_seq(tokens, self.kv_block_size);

        let sequence = TokenBlockSequence::new(tokens.into(), self.kv_block_size, Some(lora_id));
        let sequence_hashes = sequence
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        self.route_tx
            .send(RouterResult {
                worker_id,
                local_hashes,
                sequence_hashes,
            })
            .await
            .unwrap();
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
    use std::time::{Duration, Instant};
    use tokio::time;
    use tokio_util::sync::CancellationToken;

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

    /// Validate basic insert / expiry behaviour of [`TimerManager`].
    #[tokio::test]
    async fn test_timer_manager_expiry() {
        const TTL: Duration = Duration::from_millis(50);
        let mut tm: TimerManager<u32> = TimerManager::new(TTL);

        tm.insert(vec![1, 2, 3]);
        assert!(tm.get_expiry(&1).is_some());
        assert!(tm.get_expiry(&2).is_some());
        assert!(tm.get_expiry(&3).is_some());

        // Wait until after the TTL
        time::sleep(TTL + Duration::from_millis(20)).await;
        let expired = tm.poll_expired();
        assert_eq!(expired.len(), 3);
        assert!(tm.get_expiry(&1).is_none());
        assert!(tm.get_expiry(&2).is_none());
        assert!(tm.get_expiry(&3).is_none());
    }

    /// End-to-end test for [`ApproxKvIndexer`]:
    ///   1. No matches before routing decision
    ///   2. Matches appear after `process_routing_decision`
    ///   3. Matches disappear after TTL expiry
    #[tokio::test]
    async fn test_approx_kv_indexer_basic_flow() {
        const KV_BLOCK_SIZE: usize = 4;
        const TTL: Duration = Duration::from_millis(200);
        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL);

        let tokens: Vec<u32> = vec![1, 2, 3, 4]; // Exactly one KV block
        let worker_id: WorkerId = 0;
        let lora_id: u64 = 42;

        // 1. Before routing decision there should be no matches
        let pre_scores = indexer
            .find_matches_for_request(&tokens)
            .await
            .expect("indexer offline");
        assert!(pre_scores.scores.is_empty());

        // 2. Inform indexer about routing decision
        indexer
            .process_routing_decision_for_request(&tokens, lora_id, worker_id)
            .await;

        // Poll until we observe the match being registered
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores.get(&worker_id).copied() == Some(1)
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
        const KV_BLOCK_SIZE: usize = 4;
        const TTL: Duration = Duration::from_secs(5); // Large enough to avoid expiry during test
        let cancel = CancellationToken::new();
        let mut indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL);

        let tokens: Vec<u32> = vec![10, 11, 12, 13];
        let worker_id: WorkerId = 7;

        indexer
            .process_routing_decision_for_request(&tokens, 0, worker_id)
            .await;

        // Wait until the worker is registered
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores.contains_key(&worker_id)
        })
        .await;

        // Remove the worker
        indexer.remove_worker(worker_id).await;

        // Ensure the worker's entries are gone
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            !s.scores.contains_key(&worker_id)
        })
        .await;
    }

    /// Two sequences with a shared prefix should yield overlap scores reflecting the common blocks.
    #[tokio::test]
    async fn test_common_prefix_overlap() {
        const KV_BLOCK_SIZE: usize = 4;
        const TTL: Duration = Duration::from_secs(5);

        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL);

        // Sequence A : single block
        let seq_a: Vec<u32> = vec![1, 2, 3, 4];
        let worker_a: WorkerId = 11;

        // Register Sequence A on worker A
        indexer
            .process_routing_decision_for_request(&seq_a, 0, worker_a)
            .await;

        // Ensure the indexer has registered the block
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&seq_a).await.unwrap();
            s.scores.get(&worker_a).copied() == Some(1)
        })
        .await;

        // Sequence B : shares the first block with Sequence A, plus an extra block
        let seq_b: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Query the indexer for overlaps of Sequence B (before it has been routed anywhere)
        let overlap = indexer.find_matches_for_request(&seq_b).await.unwrap();

        // Expect worker A to have an overlap score of 1 (shared first block)
        assert_eq!(overlap.scores.get(&worker_a), Some(&1));
    }

    /// When the same block resides on multiple workers, all should appear in the overlap scores.
    #[tokio::test]
    async fn test_multiple_workers_same_block() {
        const KV_BLOCK_SIZE: usize = 4;
        const TTL: Duration = Duration::from_secs(5);

        let cancel = CancellationToken::new();
        let indexer = ApproxKvIndexer::new(cancel.clone(), KV_BLOCK_SIZE, TTL);

        let tokens: Vec<u32> = vec![9, 8, 7, 6];
        let worker_0: WorkerId = 21;
        let worker_1: WorkerId = 22;

        // Register the same sequence on two different workers
        indexer
            .process_routing_decision_for_request(&tokens, 0, worker_0)
            .await;
        indexer
            .process_routing_decision_for_request(&tokens, 0, worker_1)
            .await;

        // Wait until both workers are reflected in overlap scores
        spin_until(Duration::from_millis(100), || async {
            let s = indexer.find_matches_for_request(&tokens).await.unwrap();
            s.scores.get(&worker_0).copied() == Some(1)
                && s.scores.get(&worker_1).copied() == Some(1)
        })
        .await;

        let scores = indexer.find_matches_for_request(&tokens).await.unwrap();

        assert_eq!(scores.scores.get(&worker_0), Some(&1));
        assert_eq!(scores.scores.get(&worker_1), Some(&1));
    }
}
