// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pruning and TTL utilities for KV Indexers
//!
//! This module provides utilities for managing TTL-based expiration and size-based pruning
//! of blocks in the radix tree. These utilities are used by the KvIndexer to manage
//! memory usage and keep the cache fresh.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use tokio::time::{Duration, Instant};

use crate::indexer::KvRouterError;
use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// Block entry to be inserted in the [`PruneManager::expirations`] heap.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BlockEntry {
    /// The key of the block entry.
    pub key: ExternalSequenceBlockHash,
    /// The worker (with dp_rank) that stored this block.
    pub worker: WorkerWithDpRank,
    /// The position of this block in the sequence (0-indexed).
    pub seq_position: usize,
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
    /// Time-to-live duration for blocks before they expire.
    pub ttl: Duration,
    /// The maximum tree size before pruning is considered.
    pub max_tree_size: usize,
    /// The target size ratio to prune down to when max_tree_size is exceeded.
    /// For example, if max_tree_size is 100 and target_size_ratio is 0.5,
    /// we will prune down to 50 nodes when max_tree_size is exceeded.
    pub prune_target_ratio: f64,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(120), // 120 seconds
            max_tree_size: 2usize.pow(20), // 2^20 = 1048576
            prune_target_ratio: 0.8,       // Prune down to 80% of max
        }
    }
}

/// A data structure to manage a collection of timers, addressable by a key.
/// This is structured as a sort of "priority queue" of keys, where the priority is the expiration time.
/// It supports insertion as well as updating the expiration time of a key.
/// The [`PruneManager::expirations`] heap is lazily updated to reflect the true expiration times in [`PruneManager::timers`]
/// For now, we have a fixed expiration time for all keys.
#[derive(Debug)]
pub struct PruneManager<K: Clone + Hash + Eq + Ord> {
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
    pub prune_config: Option<PruneConfig>,
}

impl<K: Clone + Hash + Eq + Ord> PruneManager<K> {
    /// Creates a new, empty PruneManager.
    pub fn new(threshold: usize, prune_config: PruneConfig) -> Self {
        let ttl = prune_config.ttl;
        PruneManager {
            timers: HashMap::new(),
            expirations: BinaryHeap::new(),
            ttl,
            threshold,
            prune_config: Some(prune_config),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
    use crate::protocols::{TokensWithHashes, WorkerId, WorkerWithDpRank};
    use std::sync::Arc;
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
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX, // Effectively disable size-based pruning
            prune_target_ratio: 0.5,
        };
        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);

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
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);

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

    /// End-to-end test for [`KvIndexer`] with TTL:
    ///   1. No matches before routing decision
    ///   2. Matches appear after `process_routing_decision`
    ///   3. Matches disappear after TTL expiry
    #[tokio::test]
    async fn test_approx_kv_indexer_basic_flow() {
        const TTL: Duration = Duration::from_millis(200);
        let cancel = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        let tokens: Vec<u32> = vec![1, 2, 3, 4]; // Exactly one KV block
        let worker_id: WorkerId = 0;

        // 1. Before routing decision there should be no matches
        let pre_scores = indexer
            .find_matches_for_request(&tokens)
            .await
            .expect("indexer offline");
        assert!(pre_scores.scores.is_empty());

        // 2. Inform indexer about routing decision
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
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
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let mut indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        let tokens: Vec<u32> = vec![10, 11, 12, 13];
        let worker_id: WorkerId = 7;

        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
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
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let mut indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        let tokens: Vec<u32> = vec![100, 101, 102, 103];
        let worker_0: WorkerId = 30;
        let worker_1: WorkerId = 31;

        // Register on both workers
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
                WorkerWithDpRank::from_worker_id(worker_0),
            )
            .await
            .unwrap();
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
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
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        // Sequence A : single block
        let seq_a: Vec<u32> = vec![1, 2, 3, 4];
        let worker_a: WorkerId = 11;

        // Register Sequence A on worker A
        let mut tokens_with_hashes = TokensWithHashes::new(seq_a.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
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
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        let tokens: Vec<u32> = vec![9, 8, 7, 6];
        let worker_0: WorkerId = 21;
        let worker_1: WorkerId = 22;

        // Register the same sequence on two different workers
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
                WorkerWithDpRank::from_worker_id(worker_0),
            )
            .await
            .unwrap();
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
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
            ttl: TTL,
            max_tree_size: 100,
            prune_target_ratio: 0.5,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);

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
            ttl: TTL,
            max_tree_size: 10,
            prune_target_ratio: 0.5,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);

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
        let prune_config = PruneConfig {
            ttl: TTL,
            max_tree_size: usize::MAX,
            prune_target_ratio: 0.5,
        };
        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);
        // Temporarily set prune_config to None to test the error case
        pm.prune_config = None;

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

    /// End-to-end test for [`KvIndexer`] with TTL and pruning
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
            ttl: TTL,
            max_tree_size: 5,        // Very small to trigger pruning quickly
            prune_target_ratio: 0.4, // target size is 5 * 0.4 = 2
        };

        let cancel = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let indexer = KvIndexer::new_with_frequency(
            cancel.clone(),
            None,
            KV_BLOCK_SIZE,
            metrics,
            Some(prune_config),
        );

        let worker = WorkerWithDpRank::from_worker_id(42);

        // Insert 5 sequences (5 blocks total, at max_tree_size but not exceeding)
        for i in 0..5 {
            let tokens: Vec<u32> = vec![i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3];
            let mut tokens_with_hashes = TokensWithHashes::new(tokens, KV_BLOCK_SIZE);
            indexer
                .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
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
        let mut tokens_with_hashes = TokensWithHashes::new(tokens, KV_BLOCK_SIZE);
        indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
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
            ttl: TTL,
            max_tree_size: 5,
            prune_target_ratio: 0.8,
        };

        let mut pm: PruneManager<u32> = PruneManager::new(50, prune_config);

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
