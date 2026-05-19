// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `AsyncShardHandle` — an async abstraction over a single routing shard.
//!
//! [`BranchShardedIndexer`] is parameterised over `S: AsyncShardHandle` so that
//! the same routing-trie logic can drive either:
//!
//! - **In-process shards**: `ThreadPoolIndexer<T>` with `T: AnchorCapableSyncIndexer`.
//! - **Remote shards**: a velo-backed client that sends requests over UDS/TCP
//!   (see the `shard_router` module, feature-gated behind `velo-runtime`).
//!
//! ## Write vs. read semantics
//!
//! * **Write operations** (`apply_event`, `enqueue_anchor`, `remove_worker`) are
//!   fire-and-forget: the caller does not wait for the shard to apply the event.
//!   For in-process shards this is a channel send; for remote shards it is an
//!   active-messaging (AM) send over velo.
//!
//! * **Read operations** (`find_matches_from_anchor`, `dump_events`) are
//!   request-response: the caller awaits the result.

use async_trait::async_trait;
use std::sync::Arc;

use super::{AnchorCapableSyncIndexer, AnchorRef, AnchorTask, KvRouterError, ShardSizeSnapshot};
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
use crate::protocols::*;

/// Async abstraction over one routing shard.
///
/// Implementations must be cheap to clone (e.g. wrap the real handle in `Arc`).
#[async_trait]
pub trait AsyncShardHandle: Send + Sync + 'static {
    /// Apply a KV cache event to this shard (fire-and-forget).
    async fn apply_event(&self, event: RouterEvent);

    /// Enqueue a structural anchor before a dependent suffix event.
    ///
    /// Ordering guarantee: the anchor is guaranteed to be applied before any
    /// subsequent `apply_event` call for the same `WorkerWithDpRank` on this
    /// shard, regardless of the transport (channel order for in-process; AM
    /// ordering for remote).
    ///
    /// Returns `Err` only if the shard is offline / the channel is closed.
    fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvRouterError>;

    /// Read: find block matches starting from a previously installed anchor.
    async fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError>;

    /// Remove all state associated with a worker (fire-and-forget).
    async fn remove_worker(&self, worker_id: WorkerId);

    /// Remove state for a specific (worker_id, dp_rank) pair (fire-and-forget).
    ///
    /// Semantically narrower than [`remove_worker`]: only the state for the
    /// given `dp_rank` is removed; other dp_ranks of the same `worker_id` are
    /// left intact on the shard.
    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank);

    /// Dump all stored events (used for recovery and state transfer).
    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError>;

    /// Return the current size snapshot for this shard.
    async fn shard_sizes(&self) -> ShardSizeSnapshot;

    /// Flush all pending write events; returns the queue depth at call time.
    async fn flush(&self) -> usize;

    /// Shut down the shard, terminating any background threads or tasks.
    fn shutdown(&self);

    /// Return per-node edge-length histogram for bench/diagnostic tooling.
    ///
    /// In-process shard handles delegate to the underlying trie.  Remote
    /// shard handles should return an empty `Vec` — the data lives on the
    /// remote host and is not available locally.
    fn node_edge_lengths(&self) -> Vec<usize>;
}

// ---------------------------------------------------------------------------
// In-process implementation
// ---------------------------------------------------------------------------

/// `AsyncShardHandle` implementation that dispatches to a `ThreadPoolIndexer<T>`
/// running in the same process.
///
/// * Writes are channel sends (sync, wrapped in async).
/// * `find_matches_from_anchor` runs on a `spawn_blocking` thread since the
///   underlying CRTC `find_matches_from_anchor` is synchronous.
#[async_trait]
impl<T: AnchorCapableSyncIndexer> AsyncShardHandle for ThreadPoolIndexer<T> {
    async fn apply_event(&self, event: RouterEvent) {
        KvIndexerInterface::apply_event(self, event).await;
    }

    fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvRouterError> {
        ThreadPoolIndexer::enqueue_anchor(self, worker, anchor)
    }

    async fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let backend = self.backend_arc();
        tokio::task::spawn_blocking(move || backend.find_matches_from_anchor(anchor, &suffix))
            .await
            .map_err(|_| KvRouterError::IndexerOffline)?
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        KvIndexerInterface::remove_worker(self, worker_id).await;
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        KvIndexerInterface::remove_worker_dp_rank(self, worker_id, dp_rank).await;
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        KvIndexerInterface::dump_events(self).await
    }

    async fn shard_sizes(&self) -> ShardSizeSnapshot {
        KvIndexerInterface::shard_sizes(self)
            .await
            .into_iter()
            .next()
            .unwrap_or(ShardSizeSnapshot {
                shard_idx: 0,
                worker_count: 0,
                block_count: 0,
                node_count: 0,
            })
    }

    async fn flush(&self) -> usize {
        KvIndexerInterface::flush(self).await
    }

    fn shutdown(&self) {
        KvIndexerInterface::shutdown(self);
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        KvIndexerInterface::node_edge_lengths(self)
    }
}

// `Arc<S>` forwards to the inner `S`.
#[async_trait]
impl<S: AsyncShardHandle> AsyncShardHandle for Arc<S> {
    async fn apply_event(&self, event: RouterEvent) {
        (**self).apply_event(event).await;
    }

    fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvRouterError> {
        (**self).enqueue_anchor(worker, anchor)
    }

    async fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        (**self).find_matches_from_anchor(anchor, suffix).await
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        (**self).remove_worker(worker_id).await;
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        (**self).remove_worker_dp_rank(worker_id, dp_rank).await;
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        (**self).dump_events().await
    }

    async fn shard_sizes(&self) -> ShardSizeSnapshot {
        (**self).shard_sizes().await
    }

    async fn flush(&self) -> usize {
        (**self).flush().await
    }

    fn shutdown(&self) {
        (**self).shutdown();
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        (**self).node_edge_lengths()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
    use crate::test_utils::router_event;

    type TestTPI = ThreadPoolIndexer<ConcurrentRadixTreeCompressed>;

    fn make_tpi() -> TestTPI {
        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32)
    }

    fn store_event_for(worker_id: u64, dp_rank: u32, values: &[u64]) -> RouterEvent {
        let locals: Vec<LocalBlockHash> = values.iter().copied().map(LocalBlockHash).collect();
        let seq_hashes = crate::protocols::compute_seq_hash_for_block(&locals);
        router_event(
            worker_id,
            0,
            dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: locals
                    .iter()
                    .zip(seq_hashes.iter())
                    .map(|(&th, &sh)| KvCacheStoredBlockData {
                        tokens_hash: th,
                        block_hash: ExternalSequenceBlockHash(sh),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        )
    }

    /// `node_edge_lengths` must delegate to the backend trie rather than
    /// returning a stub empty `Vec`.  The CRTC produces at least one
    /// histogram entry once events are stored.
    #[tokio::test]
    async fn node_edge_lengths_delegates_to_backend_trie() {
        let tpi = make_tpi();
        AsyncShardHandle::apply_event(&tpi, store_event_for(1, 0, &[10, 20, 30])).await;
        AsyncShardHandle::flush(&tpi).await;
        let lengths = AsyncShardHandle::node_edge_lengths(&tpi);
        assert!(
            !lengths.is_empty() || lengths.iter().sum::<usize>() > 0,
            "expected non-empty node_edge_lengths after storing blocks, got {lengths:?}"
        );
    }

    /// `remove_worker_dp_rank` must only remove the named dp_rank, leaving
    /// sibling dp_ranks intact.  This exercises the fire-and-forget write
    /// path via the `AsyncShardHandle` trait delegation chain.
    #[tokio::test]
    async fn remove_worker_dp_rank_preserves_sibling_dp_ranks() {
        let tpi = make_tpi();
        AsyncShardHandle::apply_event(&tpi, store_event_for(7, 0, &[1, 2, 3])).await;
        AsyncShardHandle::apply_event(&tpi, store_event_for(7, 1, &[4, 5, 6])).await;
        AsyncShardHandle::flush(&tpi).await;

        let before = AsyncShardHandle::dump_events(&tpi).await.unwrap();
        assert!(
            before.iter().any(|e| e.event.dp_rank == 0),
            "dp_rank=0 should have events before removal"
        );
        assert!(
            before.iter().any(|e| e.event.dp_rank == 1),
            "dp_rank=1 should have events before removal"
        );

        AsyncShardHandle::remove_worker_dp_rank(&tpi, 7, 0).await;
        AsyncShardHandle::flush(&tpi).await;

        let after = AsyncShardHandle::dump_events(&tpi).await.unwrap();
        assert!(
            !after.iter().any(|e| e.event.dp_rank == 0),
            "dp_rank=0 should be gone after remove_worker_dp_rank"
        );
        assert!(
            after.iter().any(|e| e.event.dp_rank == 1),
            "dp_rank=1 should still be present after removing dp_rank=0 only"
        );
    }
}
