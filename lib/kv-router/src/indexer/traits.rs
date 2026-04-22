// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use std::sync::Arc;

use super::{KvIndexerMetrics, KvRouterError, WorkerTask};
use crate::protocols::*;

/// Trait for querying an external shared KV cache pool.
///
/// Implementations check which blocks/pages from a request's token sequence
/// exist in the shared cache. The returned `SharedCacheHits` describes which
/// block positions are available externally (and thus cheaper to prefill).
#[async_trait]
pub trait SharedKvCache: Send + Sync {
    /// Query which blocks exist in the shared cache for the given token sequence.
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
    ) -> Result<SharedCacheHits, KvRouterError>;
}

/// Per-shard size snapshot returned by [`KvIndexerInterface::shard_sizes`].
///
/// `worker_count` and `block_count` are always populated.
/// `node_count` is populated only when the `shard-metrics` feature is enabled
/// on the `dynamo-kv-router` crate; otherwise it is `0`.
#[derive(Debug, Clone)]
pub struct ShardSizeSnapshot {
    /// Zero-based shard index.
    pub shard_idx: usize,
    /// Distinct `(worker_id, dp_rank)` pairs stored in this shard.
    pub worker_count: usize,
    /// Total cached blocks across all workers in this shard.
    pub block_count: usize,
    /// Radix-tree node count (only non-zero with `shard-metrics` feature).
    pub node_count: usize,
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
    /// * `lora_name` - Optional LoRA adapter name to include in block hash computation.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError>;

    /// Apply a `RouterEvent` to the KV store.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    async fn apply_event(&self, event: RouterEvent);

    /// Remove a worker's entries from the trie.
    ///
    /// ### Arguments
    ///
    /// * `worker` - The worker to remove from the trie.
    async fn remove_worker(&self, worker: WorkerId);

    /// Remove a single dp_rank for a worker from the trie.
    ///
    /// Default implementation falls back to removing the entire worker.
    /// Indexers that track dp_rank-level granularity should override this.
    async fn remove_worker_dp_rank(&self, worker: WorkerId, _dp_rank: DpRank) {
        self.remove_worker(worker).await;
    }

    /// Shutdown the KV Indexer.
    fn shutdown(&self);

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

    /// Async task that returns when all pending events have been processed.
    /// For now, we assume that no requests or events are being sent in the meantime.
    /// Returns the amount of events still in the queue at the time of the flush.
    /// Used primarily for debugging.
    async fn flush(&self) -> usize;

    /// Return a human-readable timing breakdown of `find_matches` overhead.
    ///
    /// Implementations that track per-phase timing (e.g. scatter/gather overhead
    /// vs. actual shard work) override this to return a multi-line report string.
    /// The default returns an empty string so callers can skip printing it.
    fn timing_report(&self) -> String {
        String::new()
    }

    /// Return a size snapshot for each shard.
    ///
    /// Single-shard indexers return one entry (shard 0).  Multi-shard indexers
    /// return one entry per shard.  Non-sharded indexers (and implementations
    /// that don't override this) return an empty `Vec`.
    ///
    /// See [`ShardSizeSnapshot`] for the fields exposed per shard.
    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        vec![]
    }

    /// Edge lengths (hashes per node) for every non-root node.
    /// Returns an empty vec for backends that don't support this.
    fn node_edge_lengths(&self) -> Vec<usize> {
        vec![]
    }
}

// ============================================================================
// SyncIndexer trait
// ============================================================================

/// Trait for thread-safe data structures that support KV cache indexing operations.
///
/// All methods take `&self` and are synchronous. Implementations must be safe for
/// concurrent access (via internal locking, DashMap, etc).
///
/// This trait is used with [`ThreadPoolIndexer`](super::ThreadPoolIndexer), which wraps a `SyncIndexer` to
/// provide the async [`KvIndexerInterface`] with:
/// - Sticky event routing to N worker threads
/// - Inline reads on the caller's thread (no channel dispatch for find_matches)
pub trait SyncIndexer: Send + Sync + 'static {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()>;

    /// Find matches for a sequence of block hashes.
    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores;

    /// Returns true when a maintenance task should be enqueued.
    fn try_schedule_cleanup(&self) -> bool {
        false
    }

    /// Rolls back a scheduled cleanup when enqueueing the task fails.
    fn cancel_scheduled_cleanup(&self) {}

    /// Executes a maintenance task on a worker thread.
    fn run_cleanup_task(&self) {}

    /// Dump events directly from the shared structure, bypassing worker channels.
    /// Returns `Some(events)` for backends whose tree state is fully shared (e.g.
    /// ConcurrentRadixTree). Returns `None` for backends that keep per-thread
    /// state and must dump via the worker channel.
    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        None
    }

    /// Number of distinct workers registered in this backend.
    fn worker_count(&self) -> usize {
        0
    }

    /// Total cached blocks across all workers.
    fn block_count(&self) -> usize {
        0
    }

    /// Number of radix-tree nodes created since construction.
    /// Only meaningful when the `shard-metrics` feature is enabled; returns 0 otherwise.
    fn node_count(&self) -> usize {
        0
    }

    /// Edge lengths (hashes per node) for every non-root node in the tree.
    /// Returns an empty vec for backends that don't support this.
    fn node_edge_lengths(&self) -> Vec<usize> {
        vec![]
    }
}
