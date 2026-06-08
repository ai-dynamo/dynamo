// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `AsyncShardHandle` — an async abstraction over a single routing shard.
//!
//! [`BranchShardedIndexer`] is parameterised over `S: AsyncShardHandle` so that
//! the same routing-trie logic can drive either:
//!
//! - **In-process shards**: `ThreadPoolIndexer<T>` with `T: AnchorCapableSyncIndexer`.
//! - **Remote shards**: a velo-backed client that sends requests over UDS/TCP
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

use std::future::Future;

use super::{AnchorCapableSyncIndexer, AnchorRef, AnchorTask, KvRouterError, ShardSizeSnapshot};
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
use crate::protocols::*;

/// Async abstraction over one routing shard.
///
/// `BranchShardedIndexer<S>` is generic over `S: AsyncShardHandle`; all
/// dispatch is statically resolved, so this trait does not require
/// `#[async_trait]` or `dyn`-compatibility.  Implementations declare
/// `async fn` bodies; the compiler infers the concrete future types.
pub trait AsyncShardHandle: Send + Sync + 'static {
    /// Apply a KV cache event to this shard (fire-and-forget).
    fn apply_event(&self, event: RouterEvent) -> impl Future<Output = ()> + Send;

    /// Enqueue a structural anchor before a dependent suffix event.
    ///
    /// Ordering requirement: implementations must guarantee that the anchor
    /// is visible on the shard before any subsequent suffix event for the
    /// same `WorkerWithDpRank`.  The in-process implementation achieves this
    /// by placing both in the same FIFO worker queue.  Remote implementations
    /// must provide equivalent ordering — for example by awaiting an ack
    /// before sending the suffix event, or via a single compound operation.
    ///
    /// Returns `Err` only if the shard is offline / the channel is closed.
    fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    ) -> Result<(), KvRouterError>;

    /// Read: find block matches starting from a previously installed anchor.
    fn find_matches_from_anchor(
        &self,
        anchor: AnchorRef,
        suffix: Vec<LocalBlockHash>,
    ) -> impl Future<Output = Result<OverlapScores, KvRouterError>> + Send;

    /// Remove all state associated with a worker (fire-and-forget).
    fn remove_worker(&self, worker_id: WorkerId) -> impl Future<Output = ()> + Send;

    /// Remove state for a specific (worker_id, dp_rank) pair (fire-and-forget).
    ///
    /// Semantically narrower than [`remove_worker`]: only the state for the
    /// given `dp_rank` is removed; other dp_ranks of the same `worker_id` are
    /// left intact on the shard.
    fn remove_worker_dp_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> impl Future<Output = ()> + Send;

    /// Dump all stored events (used for recovery and state transfer).
    fn dump_events(&self) -> impl Future<Output = Result<Vec<RouterEvent>, KvRouterError>> + Send;

    /// Return the current size snapshot for this shard.
    fn shard_sizes(&self) -> impl Future<Output = ShardSizeSnapshot> + Send;

    /// Flush all pending write events; returns the queue depth at call time.
    fn flush(&self) -> impl Future<Output = usize> + Send;

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
/// * `find_matches_from_anchor` runs inline on the caller's task, matching
///   `ThreadPoolIndexer::find_matches` and avoiding scheduler overhead in the
///   in-process branch-sharded hot path.
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
        self.backend().find_matches_from_anchor(anchor, &suffix)
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
