// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use rstest::rstest;
use rstest_reuse::{self, *};
use tokio::time;
use tokio_util::sync::CancellationToken;

use super::concurrent_radix_tree::ConcurrentRadixTree;
use super::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
use super::positional::{PositionalIndexer, SearchMode};
use super::*;
use crate::indexer::pruning::PruneConfig;
use crate::protocols::*;
use crate::test_utils::{
    assert_overlap_scores_eq, assert_score, flush_and_settle, make_clear_event,
    make_clear_event_with_dp_rank, make_remove_event, make_remove_event_with_parent,
    make_store_event, make_store_event_with_dp_rank, make_store_event_with_parent,
    make_store_event_with_start_position, query_scores, remove_event, router_event,
    snapshot_events, snapshot_tree, stored_blocks_with_sequence_hashes,
};

// ============================================================================
// KvIndexerInterface tests - parametrized over all implementations
// ============================================================================

#[template]
#[rstest]
// CKF is added selectively through `matching_indexer_template`; this template also drives
// dump/restore, parent-structure, and implementation-specific tests that CKF does not support.
fn indexer_template(
    #[values("single", "flat", "flat_binary", "concurrent", "concurrent_compressed")] variant: &str,
) {
}

#[template]
#[rstest]
fn matching_indexer_template(
    #[values("single", "flat", "flat_binary", "concurrent", "concurrent_compressed")] variant: &str,
) {
}

#[template]
#[rstest]
// CKF exposes logical resident counts through Stats, not tree node shape.
fn tree_size_indexer_template(
    #[values("single", "concurrent", "concurrent_compressed")] variant: &str,
) {
}

#[template]
#[rstest]
// CKF has no compressed-tree node representation.
fn compressed_tree_size_indexer_template(
    #[values("single", "concurrent_compressed")] variant: &str,
) {
}

#[template]
#[rstest]
// CKF intentionally rejects approximate routing and pruning construction.
fn approx_indexer_template(
    #[values("single", "flat", "flat_binary", "concurrent", "concurrent_compressed")] variant: &str,
) {
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchSemantics {
    Exact,
    ProbabilisticNoUnderreport,
}

fn make_matching_indexer(
    variant: &str,
    workers: &[WorkerWithDpRank],
) -> Box<dyn KvIndexerInterface + Sync> {
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    make_matching_indexer_with_metrics(variant, workers, metrics).0
}

fn make_matching_indexer_with_metrics(
    variant: &str,
    workers: &[WorkerWithDpRank],
    metrics: Arc<KvIndexerMetrics>,
) -> (Box<dyn KvIndexerInterface + Sync>, Arc<KvIndexerMetrics>) {
    let _ = workers;
    make_indexer_with_metrics(variant, metrics)
}

fn assert_scores_with_semantics(
    variant: &str,
    actual: &OverlapScores,
    query_len: usize,
    configured_workers: &[WorkerWithDpRank],
    expected_scores: &[(WorkerWithDpRank, u32)],
    _ckf_semantics: MatchSemantics,
) {
    let expected: std::collections::HashMap<_, _> = expected_scores.iter().copied().collect();
    let _ = (variant, query_len, configured_workers);
    assert_eq!(actual.scores.len(), expected.len());
    for (&worker, &expected_depth) in &expected {
        assert_eq!(actual.scores.get(&worker), Some(&expected_depth));
    }
}

async fn assert_query_scores_with_semantics(
    variant: &str,
    index: &dyn KvIndexerInterface,
    query: &[u64],
    configured_workers: &[WorkerWithDpRank],
    expected_scores: &[(WorkerWithDpRank, u32)],
    ckf_semantics: MatchSemantics,
) {
    let scores = query_scores(index, query).await;
    assert_scores_with_semantics(
        variant,
        &scores,
        query.len(),
        configured_workers,
        expected_scores,
        ckf_semantics,
    );
}

fn make_indexer(variant: &str) -> Box<dyn KvIndexerInterface + Sync> {
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    make_indexer_with_metrics(variant, metrics).0
}

fn make_indexer_with_metrics(
    variant: &str,
    metrics: Arc<KvIndexerMetrics>,
) -> (Box<dyn KvIndexerInterface + Sync>, Arc<KvIndexerMetrics>) {
    let token = CancellationToken::new();
    let kv_block_size = 32;

    let indexer: Box<dyn KvIndexerInterface + Sync> = match variant {
        "single" => Box::new(KvIndexer::new(token, kv_block_size, metrics.clone())),
        // Pin the mode explicitly (not via `new`, which reads DYN_ROUTER_POSITIONAL_SEARCH_MODE)
        // so the matrix always exercises both strided and binary regardless of the ambient env.
        "flat" => Box::new(ThreadPoolIndexer::new_with_metrics(
            PositionalIndexer::new_with_mode(32, SearchMode::Strided),
            4,
            kv_block_size,
            Some(metrics.clone()),
        )),
        "flat_binary" => Box::new(ThreadPoolIndexer::new_with_metrics(
            PositionalIndexer::new_with_mode(32, SearchMode::Binary),
            4,
            kv_block_size,
            Some(metrics.clone()),
        )),
        "concurrent" => Box::new(ThreadPoolIndexer::new_with_metrics(
            ConcurrentRadixTree::new(),
            4,
            kv_block_size,
            Some(metrics.clone()),
        )),
        "concurrent_compressed" => Box::new(ThreadPoolIndexer::new_with_metrics(
            ConcurrentRadixTreeCompressed::new(),
            4,
            kv_block_size,
            Some(metrics.clone()),
        )),
        _ => panic!("Unknown variant: {}", variant),
    };

    (indexer, metrics)
}

fn make_approx_indexer(variant: &str, ttl: Duration) -> Box<dyn KvIndexerInterface + Sync> {
    let token = CancellationToken::new();
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let kv_block_size = 4;
    let prune_config = PruneConfig { ttl };

    match variant {
        "single" => Box::new(KvIndexer::new_with_pruning(
            token,
            kv_block_size,
            metrics,
            Some(prune_config),
        )),
        // Pin the mode explicitly (see make_indexer_with_metrics) so coverage stays deterministic.
        "flat" => Box::new(ThreadPoolIndexer::new_with_pruning(
            PositionalIndexer::new_with_mode(32, SearchMode::Strided),
            4,
            kv_block_size,
            prune_config,
        )),
        "flat_binary" => Box::new(ThreadPoolIndexer::new_with_pruning(
            PositionalIndexer::new_with_mode(32, SearchMode::Binary),
            4,
            kv_block_size,
            prune_config,
        )),
        "concurrent" => Box::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTree::new(),
            4,
            kv_block_size,
            prune_config,
        )),
        "concurrent_compressed" => Box::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            4,
            kv_block_size,
            prune_config,
        )),
        _ => panic!("Unknown variant: {}", variant),
    }
}

enum TreeSizeTestIndexer {
    Single(RadixTree),
    Concurrent(ThreadPoolIndexer<ConcurrentRadixTree>),
    ConcurrentCompressed(ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
}

impl TreeSizeTestIndexer {
    fn new(variant: &str) -> Self {
        match variant {
            "single" => Self::Single(RadixTree::new()),
            "concurrent" => {
                Self::Concurrent(ThreadPoolIndexer::new(ConcurrentRadixTree::new(), 4, 4))
            }
            "concurrent_compressed" => Self::ConcurrentCompressed(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                4,
                4,
            )),
            _ => panic!("Unknown tree-size test variant: {}", variant),
        }
    }

    async fn apply_event(&mut self, event: RouterEvent) {
        match self {
            Self::Single(index) => {
                let _ = index.apply_event(event);
            }
            Self::Concurrent(index) => {
                KvIndexerInterface::apply_event(index, event).await;
            }
            Self::ConcurrentCompressed(index) => {
                KvIndexerInterface::apply_event(index, event).await;
            }
        }
    }

    async fn flush(&self) {
        match self {
            Self::Single(_) => {}
            Self::Concurrent(index) => {
                index.flush().await;
            }
            Self::ConcurrentCompressed(index) => {
                index.flush().await;
            }
        }
    }

    async fn tree_size_for_worker(&self, worker: WorkerWithDpRank) -> Option<usize> {
        match self {
            Self::Single(index) => index.tree_size_for_worker(worker),
            Self::Concurrent(index) => Self::thread_pool_size_for_worker(index, worker).await,
            Self::ConcurrentCompressed(index) => {
                Self::thread_pool_size_for_worker(index, worker).await
            }
        }
    }

    async fn thread_pool_size_for_worker<T: SyncIndexer>(
        index: &ThreadPoolIndexer<T>,
        worker: WorkerWithDpRank,
    ) -> Option<usize> {
        Some(
            index
                .worker_lookup_stats()
                .await
                .block_count_for_worker(worker)
                .unwrap_or(0),
        )
    }

    async fn live_thread_pool_worker_count(&self) -> Option<usize> {
        match self {
            Self::Single(_) => None,
            Self::Concurrent(index) => Some(Self::thread_pool_worker_count(index).await),
            Self::ConcurrentCompressed(index) => Some(Self::thread_pool_worker_count(index).await),
        }
    }

    async fn thread_pool_worker_count<T: SyncIndexer>(index: &ThreadPoolIndexer<T>) -> usize {
        index
            .shard_sizes()
            .await
            .iter()
            .map(|snapshot| snapshot.worker_count)
            .sum()
    }

    fn scores(&self, query: &[u64]) -> OverlapScores {
        let query = query.iter().copied().map(LocalBlockHash).collect();
        match self {
            Self::Single(index) => index.find_matches(query, false),
            Self::Concurrent(index) => index.backend().find_matches_impl(&query, false),
            Self::ConcurrentCompressed(index) => index.backend().find_matches_impl(&query, false),
        }
    }

    async fn assert_score_and_tree_size(
        &self,
        query: &[u64],
        worker: WorkerWithDpRank,
        expected_score: u32,
        expected_tree_size: usize,
    ) {
        let scores = self.scores(query);
        assert_eq!(scores.scores.get(&worker), Some(&expected_score));
        assert_eq!(
            self.tree_size_for_worker(worker).await,
            Some(expected_tree_size),
            "internal tree-size accounting mismatch"
        );
    }

    async fn snapshot_tree(&self) -> Vec<RouterEvent> {
        match self {
            Self::Single(index) => snapshot_events(index.dump_tree_as_events()),
            Self::Concurrent(index) => {
                snapshot_events(KvIndexerInterface::dump_events(index).await.unwrap())
            }
            Self::ConcurrentCompressed(index) => {
                snapshot_events(KvIndexerInterface::dump_events(index).await.unwrap())
            }
        }
    }
}

async fn route_approx_tokens(
    index: &dyn KvIndexerInterface,
    tokens: &[u32],
    worker: WorkerWithDpRank,
) {
    let mut tokens_with_hashes = TokensWithHashes::new(tokens.to_vec(), 4);
    index
        .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
        .await
        .unwrap();
    flush_and_settle(index).await;
}

async fn request_scores(index: &dyn KvIndexerInterface, tokens: &[u32]) -> OverlapScores {
    index
        .find_matches_for_request(tokens, None, None, None)
        .await
        .unwrap()
}

async fn assert_request_score(
    index: &dyn KvIndexerInterface,
    tokens: &[u32],
    worker: WorkerWithDpRank,
    expected_score: u32,
) {
    let scores = request_scores(index, tokens).await;
    assert_eq!(scores.scores.get(&worker), Some(&expected_score));
}

#[cfg(feature = "metrics")]
fn event_metric_value(
    metrics: &KvIndexerMetrics,
    event_type: &'static str,
    status: &'static str,
) -> u64 {
    metrics
        .kv_cache_events_applied
        .get_metric_with_label_values(&[event_type, status])
        .unwrap()
        .get()
}

#[cfg(feature = "metrics")]
fn warning_metric_value(metrics: &KvIndexerMetrics, warning_kind: &'static str) -> u64 {
    metrics
        .kv_cache_event_warnings
        .get_metric_with_label_values(&[warning_kind])
        .unwrap()
        .get()
}

#[cfg(feature = "metrics")]
fn assert_no_event_errors(metrics: &KvIndexerMetrics) {
    let invalid_count = [
        (METRIC_EVENT_STORED, METRIC_STATUS_PARENT_NOT_FOUND),
        (METRIC_EVENT_STORED, METRIC_STATUS_BLOCK_NOT_FOUND),
        (METRIC_EVENT_STORED, METRIC_STATUS_INVALID_BLOCK),
        (METRIC_EVENT_REMOVED, METRIC_STATUS_PARENT_NOT_FOUND),
        (METRIC_EVENT_REMOVED, METRIC_STATUS_BLOCK_NOT_FOUND),
        (METRIC_EVENT_REMOVED, METRIC_STATUS_INVALID_BLOCK),
    ]
    .into_iter()
    .map(|(event_type, status)| event_metric_value(metrics, event_type, status))
    .sum::<u64>();
    assert_eq!(
        invalid_count, 0,
        "router indexer reported invalid KV events"
    );
}

#[cfg(feature = "metrics")]
fn assert_no_event_warnings(metrics: &KvIndexerMetrics) {
    assert_eq!(
        warning_metric_value(metrics, METRIC_WARNING_DUPLICATE_STORE),
        0,
        "router indexer reported suspicious KV events",
    );
}

mod interface_tests {
    use super::*;
    use rstest_reuse::apply;

    #[cfg(feature = "metrics")]
    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_duplicate_store_replay_warns_without_error(variant: &str) {
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let (index, metrics) = make_indexer_with_metrics(variant, metrics);
        let worker = WorkerWithDpRank::new(0, 0);
        let event = make_store_event(0, &[1, 2, 3]);

        index.apply_event(event.clone()).await;
        flush_and_settle(index.as_ref()).await;
        let first_snapshot = snapshot_tree(index.as_ref()).await;

        index.apply_event(event).await;
        flush_and_settle(index.as_ref()).await;

        assert_eq!(
            first_snapshot,
            snapshot_tree(index.as_ref()).await,
            "replaying the same store event should not change the tree structure"
        );
        assert_score(index.as_ref(), &[1, 2, 3], worker, 3).await;
        assert_no_event_errors(metrics.as_ref());
        assert_eq!(
            warning_metric_value(metrics.as_ref(), METRIC_WARNING_DUPLICATE_STORE),
            1
        );
    }

    #[cfg(feature = "metrics")]
    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_continuation_store_does_not_warn(variant: &str) {
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let (index, metrics) = make_matching_indexer_with_metrics(variant, &workers, metrics);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        flush_and_settle(index.as_ref()).await;

        index
            .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]))
            .await;
        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3, 4, 5],
            &workers,
            &[(worker, 5)],
            MatchSemantics::Exact,
        )
        .await;
        assert_no_event_errors(metrics.as_ref());
        assert_no_event_warnings(metrics.as_ref());
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_store_and_find(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store a sequence for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(tree_size_indexer_template)]
    async fn test_tree_size_accounting_stays_stable(variant: &str) {
        let mut index = TreeSizeTestIndexer::new(variant);
        let worker = WorkerWithDpRank::new(0, 0);
        let prefix_event = make_store_event(0, &[1, 2, 3]);
        let continuation_event = make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]);
        let continuation_remove = make_remove_event_with_parent(0, &[1, 2, 3], &[4, 5]);
        let prefix_remove = make_remove_event(0, &[1, 2, 3]);

        // The uncompressed concurrent implementation still cleans descendant
        // lookup entries lazily after mid-chain removes, so the shared matrix
        // keeps this duplicate store/remove baseline. Compressed variants cover
        // eager suffix cleanup in a focused test below.

        index.apply_event(prefix_event.clone()).await;
        index.flush().await;

        index
            .assert_score_and_tree_size(&[1, 2, 3], worker, 3, 3)
            .await;
        let prefix_snapshot = index.snapshot_tree().await;

        index.apply_event(prefix_event).await;
        index.flush().await;

        assert_eq!(
            prefix_snapshot,
            index.snapshot_tree().await,
            "replaying the same store event should not change the tree structure"
        );
        index
            .assert_score_and_tree_size(&[1, 2, 3], worker, 3, 3)
            .await;

        index.apply_event(continuation_event.clone()).await;
        index.flush().await;

        index
            .assert_score_and_tree_size(&[1, 2, 3, 4, 5], worker, 5, 5)
            .await;
        let full_snapshot = index.snapshot_tree().await;

        index.apply_event(continuation_event).await;
        index.flush().await;

        assert_eq!(
            full_snapshot,
            index.snapshot_tree().await,
            "replaying the same continuation store should not change the tree structure"
        );
        index
            .assert_score_and_tree_size(&[1, 2, 3, 4, 5], worker, 5, 5)
            .await;

        index.apply_event(continuation_remove.clone()).await;
        index.flush().await;

        index
            .assert_score_and_tree_size(&[1, 2, 3, 4, 5], worker, 3, 3)
            .await;
        let trimmed_snapshot = index.snapshot_tree().await;

        index.apply_event(continuation_remove).await;
        index.flush().await;

        assert_eq!(
            trimmed_snapshot,
            index.snapshot_tree().await,
            "replaying the same remove event should not change the tree structure"
        );
        index
            .assert_score_and_tree_size(&[1, 2, 3, 4, 5], worker, 3, 3)
            .await;

        index.apply_event(prefix_remove.clone()).await;
        index.flush().await;

        let empty_scores = index.scores(&[1, 2, 3, 4, 5]);
        assert!(empty_scores.scores.is_empty());
        assert_eq!(index.tree_size_for_worker(worker).await, Some(0));
        if let Some(worker_count) = index.live_thread_pool_worker_count().await {
            assert_eq!(worker_count, 0);
        }
        assert!(index.snapshot_tree().await.is_empty());

        index.apply_event(prefix_remove).await;
        index.flush().await;

        let duplicate_empty_scores = index.scores(&[1, 2, 3, 4, 5]);
        assert!(duplicate_empty_scores.scores.is_empty());
        assert_eq!(index.tree_size_for_worker(worker).await, Some(0));
        assert!(index.snapshot_tree().await.is_empty());
    }

    #[tokio::test]
    #[apply(compressed_tree_size_indexer_template)]
    async fn test_mid_edge_remove_repairs_lookup_and_restores_explicitly(variant: &str) {
        let mut index = TreeSizeTestIndexer::new(variant);
        let worker = WorkerWithDpRank::new(0, 0);
        let local_hashes = [1, 2, 3, 4, 5];
        let sequence_hashes = compute_seq_hash_for_block(
            &local_hashes
                .iter()
                .copied()
                .map(LocalBlockHash)
                .collect::<Vec<_>>(),
        );

        index
            .apply_event(make_store_event(worker.worker_id, &local_hashes))
            .await;
        index.flush().await;
        index
            .assert_score_and_tree_size(&local_hashes, worker, 5, 5)
            .await;

        index
            .apply_event(remove_event(
                worker.worker_id,
                1,
                worker.dp_rank,
                vec![ExternalSequenceBlockHash(sequence_hashes[2])],
            ))
            .await;
        index.flush().await;
        index
            .assert_score_and_tree_size(&local_hashes, worker, 2, 2)
            .await;

        index
            .apply_event(make_store_event_with_parent(
                worker.worker_id,
                &[1, 2],
                &[3],
            ))
            .await;
        index.flush().await;
        index
            .assert_score_and_tree_size(&local_hashes, worker, 3, 3)
            .await;

        index
            .apply_event(make_store_event_with_parent(
                worker.worker_id,
                &[1, 2, 3],
                &[4, 5],
            ))
            .await;
        index.flush().await;
        index
            .assert_score_and_tree_size(&local_hashes, worker, 5, 5)
            .await;
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_shared_prefix_branch_rejects_invalid_suffix_parent(variant: &str) {
        let index = make_indexer(variant);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let worker2 = WorkerWithDpRank::new(2, 0);

        index.apply_event(make_store_event(1, &[1, 2, 3])).await;
        index.apply_event(make_store_event(2, &[1, 2, 3])).await;
        flush_and_settle(index.as_ref()).await;

        index
            .apply_event(make_store_event_with_parent(1, &[1, 2, 3], &[4, 5, 6]))
            .await;
        flush_and_settle(index.as_ref()).await;

        let scores = query_scores(index.as_ref(), &[1, 2, 3, 4, 5, 6]).await;
        assert_eq!(scores.scores.get(&worker1), Some(&6));
        assert_eq!(scores.scores.get(&worker2), Some(&3));

        index
            .apply_event(make_store_event_with_parent(2, &[1, 2, 3], &[7, 8]))
            .await;
        flush_and_settle(index.as_ref()).await;

        let branch_scores = query_scores(index.as_ref(), &[1, 2, 3, 7, 8]).await;
        assert_eq!(branch_scores.scores.get(&worker1), Some(&3));
        assert_eq!(branch_scores.scores.get(&worker2), Some(&5));

        let original_tail_scores = query_scores(index.as_ref(), &[1, 2, 3, 4, 5, 6]).await;
        assert_eq!(original_tail_scores.scores.get(&worker1), Some(&6));
        assert_eq!(original_tail_scores.scores.get(&worker2), Some(&3));

        index
            .apply_event(make_store_event_with_parent(
                2,
                &[1, 2, 3, 4, 5, 6],
                &[9, 10],
            ))
            .await;
        flush_and_settle(index.as_ref()).await;

        let rejected_parent_scores = query_scores(index.as_ref(), &[1, 2, 3, 4, 5, 6, 9, 10]).await;
        assert_eq!(rejected_parent_scores.scores.get(&worker1), Some(&6));
        assert_eq!(rejected_parent_scores.scores.get(&worker2), Some(&3));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_prefix_suffix_reuse_then_tail_append(variant: &str) {
        let index = make_indexer(variant);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let worker2 = WorkerWithDpRank::new(2, 0);

        index
            .apply_event(make_store_event(1, &[1, 2, 3, 4, 5, 6]))
            .await;
        index.apply_event(make_store_event(2, &[1, 2, 3])).await;
        flush_and_settle(index.as_ref()).await;

        assert_score(index.as_ref(), &[1, 2, 3, 4, 5, 6], worker1, 6).await;
        assert_score(index.as_ref(), &[1, 2, 3, 4, 5, 6], worker2, 3).await;

        index
            .apply_event(make_store_event_with_parent(2, &[1, 2, 3], &[4, 5, 6]))
            .await;
        index
            .apply_event(make_store_event_with_parent(
                2,
                &[1, 2, 3, 4, 5, 6],
                &[7, 8, 9],
            ))
            .await;
        flush_and_settle(index.as_ref()).await;

        assert_score(index.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8, 9], worker1, 6).await;
        assert_score(index.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8, 9], worker2, 9).await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_partial_match(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store [1, 2, 3] for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 999],
            &workers,
            &[(worker, 2)],
            MatchSemantics::ProbabilisticNoUnderreport,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_remove(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store sequence for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Remove all blocks
        index.apply_event(make_remove_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_multiple_workers_shared_prefix(variant: &str) {
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let workers = [worker0, worker1];
        let index = make_matching_indexer(variant, &workers);

        // Worker 0 has [1, 2], Worker 1 has [1, 3]
        // Since sequence hashes are cumulative, [1] has same hash for both,
        // but [1, 2] and [1, 3] have different hashes.
        index.apply_event(make_store_event(0, &[1, 2])).await;
        index.apply_event(make_store_event(1, &[1, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1],
            &workers,
            &[(worker0, 1), (worker1, 1)],
            MatchSemantics::Exact,
        )
        .await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2],
            &workers,
            &[(worker0, 2), (worker1, 1)],
            MatchSemantics::ProbabilisticNoUnderreport,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_remove_worker(variant: &str) {
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let workers = [worker0, worker1];
        let index = make_matching_indexer(variant, &workers);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 3])).await;

        // Allow time for async event processing
        flush_and_settle(index.as_ref()).await;

        index.remove_worker(0).await;

        // Allow time for async remove_worker processing
        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker1, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_dump_and_restore(variant: &str) {
        let index = make_indexer(variant);

        // Store some data
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 4])).await;

        // Allow background worker threads to process events.
        flush_and_settle(index.as_ref()).await;

        // Dump the tree as events and replay into a new index
        let events = index.dump_events().await.unwrap();
        assert!(!events.is_empty());

        let restored = make_indexer(variant);
        for event in events {
            restored.apply_event(event).await;
        }

        flush_and_settle(restored.as_ref()).await;

        assert_eq!(
            snapshot_tree(index.as_ref()).await,
            snapshot_tree(restored.as_ref()).await
        );
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_clear_all_blocks(variant: &str) {
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let workers = [worker0, worker1];
        let index = make_matching_indexer(variant, &workers);

        // Store some data for two workers
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 3])).await;

        // Clear worker 0's blocks using the Cleared event
        index.apply_event(make_clear_event(0)).await;

        flush_and_settle(index.as_ref()).await;

        // Worker 0's blocks should be gone, worker 1's remain
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker1, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_empty_query(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[],
            &workers,
            &[],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_miss_query(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[999, 998],
            &workers,
            &[],
            MatchSemantics::ProbabilisticNoUnderreport,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_shutdown_idempotent(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let index = make_matching_indexer(variant, &[worker]);
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        flush_and_settle(index.as_ref()).await;
        index.shutdown();
        index.shutdown();
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_find_matches_for_request(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        let tokens: Vec<u32> = (1..=96).collect();
        let scores = index
            .find_matches_for_request(&tokens, None, None, None)
            .await
            .unwrap();
        assert_scores_with_semantics(variant, &scores, 3, &workers, &[], MatchSemantics::Exact);

        let block_hashes = compute_block_hash_for_seq(&tokens, 32, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
        index
            .apply_event(router_event(
                0,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&block_hashes, &sequence_hashes),
                }),
            ))
            .await;
        flush_and_settle(index.as_ref()).await;

        let scores = index
            .find_matches_for_request(&tokens, None, None, None)
            .await
            .unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        );
    }

    #[tokio::test]
    #[apply(approx_indexer_template)]
    async fn test_approx_routing_decision_creates_match(variant: &str) {
        let index = make_approx_indexer(variant, Duration::from_secs(60));
        let tokens = vec![1, 2, 3, 4];
        let worker = WorkerWithDpRank::new(7, 0);

        route_approx_tokens(index.as_ref(), &tokens, worker).await;

        assert_request_score(index.as_ref(), &tokens, worker, 1).await;
    }

    #[tokio::test]
    async fn test_concurrent_compressed_approx_records_precomputed_hashes() {
        let index = ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            4,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        );
        let tokens = vec![1, 2, 3, 4];
        let worker = WorkerWithDpRank::new(7, 0);
        let block_hashes = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);

        index
            .process_routing_decision_with_hashes(worker, block_hashes, sequence_hashes)
            .await
            .unwrap();
        flush_and_settle(&index).await;

        assert_request_score(&index, &tokens, worker, 1).await;
    }

    #[tokio::test]
    async fn test_concurrent_compressed_approx_rejects_mismatched_hash_lengths() {
        let index = ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            4,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        );
        let worker = WorkerWithDpRank::new(7, 0);
        let local_hashes = [LocalBlockHash(1), LocalBlockHash(2)];
        let sequence_hashes = [1];

        let result = index
            .process_routing_decision_hash_slices(worker, &local_hashes, &sequence_hashes)
            .await;

        assert!(matches!(result, Err(KvRouterError::IndexerDroppedRequest)));
    }

    #[tokio::test]
    #[apply(approx_indexer_template)]
    async fn test_approx_ttl_expiry_removes_match(variant: &str) {
        let ttl = Duration::from_millis(25);
        let index = make_approx_indexer(variant, ttl);
        let tokens = vec![1, 2, 3, 4];
        let worker = WorkerWithDpRank::new(7, 0);

        route_approx_tokens(index.as_ref(), &tokens, worker).await;
        assert_request_score(index.as_ref(), &tokens, worker, 1).await;

        time::sleep(ttl + Duration::from_millis(125)).await;
        flush_and_settle(index.as_ref()).await;

        let scores = request_scores(index.as_ref(), &tokens).await;
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(approx_indexer_template)]
    async fn test_approx_remove_worker_dp_rank_cleans_prune_metadata(variant: &str) {
        let index = make_approx_indexer(variant, Duration::from_secs(60));
        let tokens = vec![1, 2, 3, 4];
        let removed_worker = WorkerWithDpRank::new(7, 0);
        let retained_worker = WorkerWithDpRank::new(7, 1);

        route_approx_tokens(index.as_ref(), &tokens, removed_worker).await;
        route_approx_tokens(index.as_ref(), &tokens, retained_worker).await;

        index
            .remove_worker_dp_rank(removed_worker.worker_id, removed_worker.dp_rank)
            .await;
        flush_and_settle(index.as_ref()).await;

        let scores = request_scores(index.as_ref(), &tokens).await;
        assert!(!scores.scores.contains_key(&removed_worker));
        assert_eq!(scores.scores.get(&retained_worker), Some(&1));
    }

    #[tokio::test]
    #[apply(approx_indexer_template)]
    async fn test_approx_remove_worker_cleans_prune_metadata(variant: &str) {
        let index = make_approx_indexer(variant, Duration::from_secs(60));
        let tokens = vec![1, 2, 3, 4];
        let removed_worker = WorkerWithDpRank::new(7, 0);
        let retained_worker = WorkerWithDpRank::new(8, 0);

        route_approx_tokens(index.as_ref(), &tokens, removed_worker).await;
        route_approx_tokens(index.as_ref(), &tokens, retained_worker).await;

        index.remove_worker(removed_worker.worker_id).await;
        flush_and_settle(index.as_ref()).await;

        let scores = request_scores(index.as_ref(), &tokens).await;
        assert!(!scores.scores.contains_key(&removed_worker));
        assert_eq!(scores.scores.get(&retained_worker), Some(&1));
    }

    #[tokio::test]
    #[apply(approx_indexer_template)]
    async fn test_kv_store_events_do_not_expire_through_approx_ttl(variant: &str) {
        let ttl = Duration::from_millis(25);
        let index = make_approx_indexer(variant, ttl);
        let worker = WorkerWithDpRank::new(0, 0);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        flush_and_settle(index.as_ref()).await;
        assert_score(index.as_ref(), &[1, 2, 3], worker, 3).await;

        time::sleep(ttl + Duration::from_millis(125)).await;
        flush_and_settle(index.as_ref()).await;

        assert_score(index.as_ref(), &[1, 2, 3], worker, 3).await;
    }

    #[tokio::test]
    async fn test_failed_threaded_approx_event_ack_does_not_register() {
        let index = ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTree::new(),
            1,
            32,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        );
        index.shutdown();

        let tokens = vec![1, 2, 3, 4];
        let mut tokens_with_hashes = TokensWithHashes::new(tokens.clone(), 32);
        let result = index
            .process_routing_decision_for_request(
                &mut tokens_with_hashes,
                WorkerWithDpRank::new(7, 0),
            )
            .await;

        assert!(matches!(result, Err(KvRouterError::IndexerOffline)));
        let scores = request_scores(&index, &tokens).await;
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_parent_hash_chains(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store initial sequence [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Store continuation [4, 5] with parent pointing to block 3
        index
            .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query for full sequence [1, 2, 3, 4, 5] should match all 5 blocks
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3, 4, 5],
            &workers,
            &[(worker, 5)],
            MatchSemantics::Exact,
        )
        .await;

        // Query for just [1, 2, 3] should match 3 blocks
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    async fn test_flat_dump_replay_preserves_start_positions() {
        let index = make_indexer("flat");
        index
            .apply_event(make_store_event_with_start_position(0, &[11, 12], 10))
            .await;

        flush_and_settle(index.as_ref()).await;

        let dumped = index.dump_events().await.unwrap();
        let stored = dumped
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => Some(data),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stored.len(), 2);
        assert_eq!(
            stored
                .iter()
                .map(|data| data.start_position)
                .collect::<Vec<_>>(),
            vec![Some(10), Some(11)]
        );
        assert!(stored.iter().all(|data| data.parent_hash.is_none()));

        let replay = make_indexer("flat");
        for event in dumped {
            replay.apply_event(event).await;
        }

        flush_and_settle(replay.as_ref()).await;

        assert_eq!(
            snapshot_tree(index.as_ref()).await,
            snapshot_tree(replay.as_ref()).await
        );
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_multiple_dp_ranks(variant: &str) {
        let workers = [
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(0, 2),
        ];
        let index = make_matching_indexer(variant, &workers);

        // Same worker_id but different dp_ranks should be tracked separately
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 1))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 2))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query should return all 3 dp_ranks as separate entries
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(workers[0], 3), (workers[1], 3), (workers[2], 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_remove_worker_dp_rank_only_clears_target_rank(variant: &str) {
        let workers = [WorkerWithDpRank::new(7, 0), WorkerWithDpRank::new(7, 1)];
        let index = make_matching_indexer(variant, &workers);
        index
            .apply_event(make_store_event_with_dp_rank(7, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(7, &[1, 2, 3], 1))
            .await;
        flush_and_settle(index.as_ref()).await;

        index.remove_worker_dp_rank(7, 0).await;
        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(workers[1], 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_acknowledged_rank_reset_orders_after_accepted_events(variant: &str) {
        let workers = [WorkerWithDpRank::new(7, 0), WorkerWithDpRank::new(7, 1)];
        let index = make_indexer(variant);
        index
            .apply_event(make_store_event_with_dp_rank(7, &[1, 2, 3], 0))
            .await;

        index.reset_worker_dp_rank_and_wait(7, 0).await.unwrap();

        // A rank reset does not fence sibling ranks, so order the sibling store explicitly.
        index
            .apply_event(make_store_event_with_dp_rank(7, &[1, 2, 3], 1))
            .await;
        flush_and_settle(index.as_ref()).await;

        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(workers[1], 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_partial_block_removal(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        // Verify all 3 blocks match
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            seq.len(),
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        );

        // Remove only the last block (block 3)
        // To do this correctly, we need to compute the seq_hash for block 3 specifically,
        // which requires the full sequence context [1,2,3].
        let full_hashes: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let seq_hashes = compute_seq_hash_for_block(&full_hashes);
        let block_3_seq_hash = ExternalSequenceBlockHash(seq_hashes[2]); // Last block's hash

        let remove_event = remove_event(0, 0, 0, vec![block_3_seq_hash]);
        index.apply_event(remove_event).await;

        flush_and_settle(index.as_ref()).await;

        // Query [1, 2, 3] - should only match 2 blocks now (block 3 is removed)
        let scores = index.find_matches(seq).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(worker, 2)],
            MatchSemantics::ProbabilisticNoUnderreport,
        );

        // Query [1, 2] - should still match 2 blocks
        let partial_seq: Vec<LocalBlockHash> = (1..=2).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_seq).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            2,
            &workers,
            &[(worker, 2)],
            MatchSemantics::Exact,
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_remove_mid_chain_block(variant: &str) {
        // TODO: positional indexer has no parent-child structure, so mid-chain removal
        // doesn't invalidate later positions — search skips over the gap and over-counts.
        // This limitation is independent of the search algorithm, so both flat variants skip.
        if variant == "flat" || variant == "flat_binary" {
            return;
        }

        let index = make_indexer(variant);

        // Store [1, 2, 3, 4, 5]
        index
            .apply_event(make_store_event(0, &[1, 2, 3, 4, 5]))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Verify all 5 blocks match
        let seq: Vec<LocalBlockHash> = (1..=5).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 5);

        // Remove only block 3 (index 2) — the middle of the chain
        let full_hashes: Vec<LocalBlockHash> = (1..=5).map(LocalBlockHash).collect();
        let seq_hashes = compute_seq_hash_for_block(&full_hashes);
        let block_3_seq_hash = ExternalSequenceBlockHash(seq_hashes[2]);

        let remove_event = remove_event(0, 0, 0, vec![block_3_seq_hash]);
        index.apply_event(remove_event).await;

        flush_and_settle(index.as_ref()).await;

        // Query [1, 2, 3, 4, 5] — only first 2 positions reachable (block 3 removed, orphaning 4 & 5)
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);

        // Query [1, 2] — prefix before the gap is still intact
        let prefix_seq: Vec<LocalBlockHash> = (1..=2).map(LocalBlockHash).collect();
        let scores = index.find_matches(prefix_seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);

        // Re-store block 3 as a continuation of [1, 2]
        index
            .apply_event(make_store_event_with_parent(0, &[1, 2], &[3]))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query [1, 2, 3, 4, 5] — block 3 is back but 4 & 5 were orphaned, so score = 3
        let scores = index.find_matches(seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_remove_nonexistent_worker(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store data for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        // Remove non-existent worker 999 - should not error or affect worker 0
        index.remove_worker(999).await;

        // Allow time for async processing
        flush_and_settle(index.as_ref()).await;

        // Worker 0's data should still be there
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_remove_nonexistent_blocks(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Try to remove blocks [999, 998] that don't exist - should not error
        index.apply_event(make_remove_event(0, &[999, 998])).await;

        flush_and_settle(index.as_ref()).await;

        // Original data should still be there
        assert_query_scores_with_semantics(
            variant,
            index.as_ref(),
            &[1, 2, 3],
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        )
        .await;
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_clear_then_reuse(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store initial data
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Clear the worker
        index.apply_event(make_clear_event(0)).await;

        flush_and_settle(index.as_ref()).await;

        // Verify data is gone
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            seq.len(),
            &workers,
            &[],
            MatchSemantics::Exact,
        );

        // Store new data for the same worker
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        flush_and_settle(index.as_ref()).await;

        // Verify new data is accessible
        let scores = index.find_matches(seq).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        );
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_multiple_sequences_per_worker(variant: &str) {
        let worker = WorkerWithDpRank::new(0, 0);
        let workers = [worker];
        let index = make_matching_indexer(variant, &workers);

        // Store two disjoint sequences for the same worker
        // Sequence 1: [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        // Sequence 2: [100, 101, 102] (completely different, no parent)
        index
            .apply_event(make_store_event(0, &[100, 101, 102]))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query first sequence
        let seq1: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq1).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        );

        // Query second sequence
        let seq2: Vec<LocalBlockHash> = (100..=102).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq2).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(worker, 3)],
            MatchSemantics::Exact,
        );

        // Query a mix that doesn't exist as a sequence - should only match first block
        let mixed: Vec<LocalBlockHash> = vec![LocalBlockHash(1), LocalBlockHash(100)];
        let scores = index.find_matches(mixed).await.unwrap();
        // Only block 1 is an exact match; CKF may over-report on the absent suffix.
        assert_scores_with_semantics(
            variant,
            &scores,
            2,
            &workers,
            &[(worker, 1)],
            MatchSemantics::ProbabilisticNoUnderreport,
        );
    }

    #[tokio::test]
    #[apply(matching_indexer_template)]
    async fn test_clear_only_removes_target_dp_rank(variant: &str) {
        let workers = [WorkerWithDpRank::new(0, 0), WorkerWithDpRank::new(0, 1)];
        let index = make_matching_indexer(variant, &workers);

        // Store same sequence for different dp_ranks
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 1))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Verify both dp_ranks are present
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(workers[0], 3), (workers[1], 3)],
            MatchSemantics::Exact,
        );

        // A clear is ordered within and applies only to its emitting rank.
        index.apply_event(make_clear_event_with_dp_rank(0, 0)).await;

        flush_and_settle(index.as_ref()).await;

        // Rank 0 is cleared while rank 1 retains the same sequence.
        let scores = index.find_matches(seq).await.unwrap();
        assert_scores_with_semantics(
            variant,
            &scores,
            3,
            &workers,
            &[(workers[1], 3)],
            MatchSemantics::Exact,
        );
    }
}

// ============================================================================
// LoRA isolation tests
// ============================================================================

mod lora_tests {
    use super::*;
    use rstest_reuse::apply;

    /// Reproduces the "block_hash mismatch: sequence hashes should be uniform
    /// across workers" warning seen when the same prompt is sent to both a base
    /// model worker and a LoRA worker.
    ///
    /// On main (without LoRA-aware hashing), both workers compute the same
    /// LocalBlockHash for identical tokens.  But vLLM's engine includes the
    /// adapter in its rolling ExternalSequenceBlockHash, so the radix tree
    /// sees conflicting sequence hashes at the same tree node.
    ///
    /// With LoRA-aware hashing, compute_block_hash_for_seq produces distinct
    /// LocalBlockHash values for different adapters, so the blocks land on
    /// separate tree paths and no mismatch occurs.
    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_lora_base_same_tokens_no_seq_hash_mismatch(variant: &str) {
        let index = make_indexer(variant);
        let kv_block_size: u32 = 32;

        let tokens: Vec<u32> = (0..kv_block_size * 3).collect();

        // With LoRA-aware hashing, base and adapter produce different LocalBlockHash
        let base_local =
            compute_block_hash_for_seq(&tokens, kv_block_size, BlockHashOptions::default());
        let lora_local = compute_block_hash_for_seq(
            &tokens,
            kv_block_size,
            BlockHashOptions {
                lora_name: Some("my-adapter"),
                ..Default::default()
            },
        );

        assert_ne!(
            base_local, lora_local,
            "LoRA-aware hashing must produce different LocalBlockHash values"
        );

        // Simulate what vLLM does: same tokens, different rolling seq hashes
        // because the engine accounts for the adapter internally.
        let base_seq = compute_seq_hash_for_block(&base_local);
        let lora_seq = compute_seq_hash_for_block(&lora_local);

        // Worker 0: base model
        index
            .apply_event(router_event(
                0,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&base_local, &base_seq),
                }),
            ))
            .await;

        // Worker 1: LoRA adapter — different LocalBlockHash, so this goes to
        // a separate tree path instead of colliding with worker 0's node.
        index
            .apply_event(router_event(
                1,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&lora_local, &lora_seq),
                }),
            ))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Base query finds only worker 0
        let base_scores = index.find_matches(base_local.clone()).await.unwrap();
        assert_eq!(base_scores.scores.len(), 1);
        assert_eq!(
            *base_scores
                .scores
                .get(&WorkerWithDpRank::new(0, 0))
                .unwrap(),
            3
        );
        assert!(
            !base_scores
                .scores
                .contains_key(&WorkerWithDpRank::new(1, 0))
        );

        // LoRA query finds only worker 1
        let lora_scores = index.find_matches(lora_local.clone()).await.unwrap();
        assert_eq!(lora_scores.scores.len(), 1);
        assert_eq!(
            *lora_scores
                .scores
                .get(&WorkerWithDpRank::new(1, 0))
                .unwrap(),
            3
        );
        assert!(
            !lora_scores
                .scores
                .contains_key(&WorkerWithDpRank::new(0, 0))
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_different_lora_adapters_do_not_conflict(variant: &str) {
        let index = make_indexer(variant);
        let kv_block_size: u32 = 32;

        let tokens: Vec<u32> = (0..kv_block_size * 2).collect();

        let hashes_a = compute_block_hash_for_seq(
            &tokens,
            kv_block_size,
            BlockHashOptions {
                lora_name: Some("adapter-a"),
                ..Default::default()
            },
        );
        let hashes_b = compute_block_hash_for_seq(
            &tokens,
            kv_block_size,
            BlockHashOptions {
                lora_name: Some("adapter-b"),
                ..Default::default()
            },
        );

        assert_ne!(
            hashes_a, hashes_b,
            "Different adapters must produce different hashes"
        );

        let seq_a = compute_seq_hash_for_block(&hashes_a);
        let seq_b = compute_seq_hash_for_block(&hashes_b);

        // Store adapter-a blocks on worker 0
        index
            .apply_event(router_event(
                0,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&hashes_a, &seq_a),
                }),
            ))
            .await;

        // Store adapter-b blocks on worker 1
        index
            .apply_event(router_event(
                1,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&hashes_b, &seq_b),
                }),
            ))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query adapter-a → only worker 0
        let scores_a = index.find_matches(hashes_a.clone()).await.unwrap();
        assert_eq!(scores_a.scores.len(), 1);
        assert!(scores_a.scores.contains_key(&WorkerWithDpRank::new(0, 0)));
        assert!(!scores_a.scores.contains_key(&WorkerWithDpRank::new(1, 0)));

        // Query adapter-b → only worker 1
        let scores_b = index.find_matches(hashes_b.clone()).await.unwrap();
        assert_eq!(scores_b.scores.len(), 1);
        assert!(scores_b.scores.contains_key(&WorkerWithDpRank::new(1, 0)));
        assert!(!scores_b.scores.contains_key(&WorkerWithDpRank::new(0, 0)));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_different_cache_namespaces_do_not_conflict(variant: &str) {
        let index = make_indexer(variant);
        let kv_block_size: u32 = 32;

        let tokens: Vec<u32> = (0..kv_block_size * 2).collect();

        let hashes_a = compute_block_hash_for_seq(
            &tokens,
            kv_block_size,
            BlockHashOptions {
                cache_namespace: Some("tenant-a"),
                ..Default::default()
            },
        );
        let hashes_b = compute_block_hash_for_seq(
            &tokens,
            kv_block_size,
            BlockHashOptions {
                cache_namespace: Some("tenant-b"),
                ..Default::default()
            },
        );

        assert_ne!(
            hashes_a, hashes_b,
            "Different cache namespaces must produce different hashes"
        );

        let seq_a = compute_seq_hash_for_block(&hashes_a);
        let seq_b = compute_seq_hash_for_block(&hashes_b);

        index
            .apply_event(router_event(
                0,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&hashes_a, &seq_a),
                }),
            ))
            .await;

        index
            .apply_event(router_event(
                1,
                0,
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: stored_blocks_with_sequence_hashes(&hashes_b, &seq_b),
                }),
            ))
            .await;

        flush_and_settle(index.as_ref()).await;

        let scores_a = index.find_matches(hashes_a.clone()).await.unwrap();
        assert_eq!(scores_a.scores.len(), 1);
        assert!(scores_a.scores.contains_key(&WorkerWithDpRank::new(0, 0)));
        assert!(!scores_a.scores.contains_key(&WorkerWithDpRank::new(1, 0)));

        let scores_b = index.find_matches(hashes_b.clone()).await.unwrap();
        assert_eq!(scores_b.scores.len(), 1);
        assert!(scores_b.scores.contains_key(&WorkerWithDpRank::new(1, 0)));
        assert!(!scores_b.scores.contains_key(&WorkerWithDpRank::new(0, 0)));

        let request_scores_a = index
            .find_matches_for_request(&tokens, None, Some("tenant-a"), None)
            .await
            .unwrap();
        assert_eq!(request_scores_a.scores.len(), 1);
        assert!(
            request_scores_a
                .scores
                .contains_key(&WorkerWithDpRank::new(0, 0))
        );
        assert!(
            !request_scores_a
                .scores
                .contains_key(&WorkerWithDpRank::new(1, 0))
        );

        let request_scores_b = index
            .find_matches_for_request(&tokens, None, Some("tenant-b"), None)
            .await
            .unwrap();
        assert_eq!(request_scores_b.scores.len(), 1);
        assert!(
            request_scores_b
                .scores
                .contains_key(&WorkerWithDpRank::new(1, 0))
        );
        assert!(
            !request_scores_b
                .scores
                .contains_key(&WorkerWithDpRank::new(0, 0))
        );
    }
}

// ============================================================================
// Long sequence tests - especially important for NestedMap/PositionalIndexer
// ============================================================================

mod long_sequence_tests {
    use super::*;
    use rstest_reuse::apply;

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_branching_continuations(variant: &str) {
        let index = make_indexer(variant);

        // Common prefix: blocks 1-30
        let common_prefix: Vec<u64> = (1..=30).collect();
        index.apply_event(make_store_event(0, &common_prefix)).await;

        // Branch A: blocks 31-60 on worker 0
        let branch_a: Vec<u64> = (31..=60).collect();
        index
            .apply_event(make_store_event_with_parent(0, &common_prefix, &branch_a))
            .await;

        // Branch B: blocks 131-160 (different content) on worker 1
        // First store the common prefix for worker 1
        index.apply_event(make_store_event(1, &common_prefix)).await;
        let branch_b: Vec<u64> = (131..=160).collect();
        index
            .apply_event(make_store_event_with_parent(1, &common_prefix, &branch_b))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query common prefix - both workers should match
        let prefix_query: Vec<LocalBlockHash> = (1..=30).map(LocalBlockHash).collect();
        let scores = index.find_matches(prefix_query).await.unwrap();
        assert_eq!(scores.scores.len(), 2);
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            30
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            30
        );

        // Query branch A path - only worker 0 should match fully
        let branch_a_query: Vec<LocalBlockHash> = (1..=60).map(LocalBlockHash).collect();
        let scores = index.find_matches(branch_a_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            60
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            30
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_partial_removal(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=100).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        flush_and_settle(index.as_ref()).await;

        // Verify full match
        let full_query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(full_query.clone()).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Remove blocks 80-100 (the tail)
        let tail_hashes: Vec<LocalBlockHash> = (1..=100).map(LocalBlockHash).collect();
        let seq_hashes = compute_seq_hash_for_block(&tail_hashes);
        let remove_hashes: Vec<ExternalSequenceBlockHash> = seq_hashes[79..100]
            .iter()
            .map(|&h| ExternalSequenceBlockHash(h))
            .collect();

        let remove_event = remove_event(0, 0, 0, remove_hashes);
        index.apply_event(remove_event).await;

        flush_and_settle(index.as_ref()).await;

        // Query should now only match first 79 blocks
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            79
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_jump_boundaries(variant: &str) {
        let index = make_indexer(variant);
        let cases = [
            (0, 1, 31),
            (1, 101, 32),
            (2, 201, 33),
            (3, 301, 63),
            (4, 401, 64),
            (5, 501, 65),
            (6, 601, 96),
        ];

        for (worker_id, start, len) in cases {
            let sequence: Vec<u64> = (start..start + len).collect();
            index
                .apply_event(make_store_event(worker_id, &sequence))
                .await;
        }

        flush_and_settle(index.as_ref()).await;

        for (worker_id, start, len) in cases {
            let query: Vec<LocalBlockHash> = (start..start + len).map(LocalBlockHash).collect();
            let scores = index.find_matches(query).await.unwrap();
            assert_eq!(
                scores.scores.get(&WorkerWithDpRank::new(worker_id, 0)),
                Some(&(len as u32))
            );
        }
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_divergence_at_jump_boundaries(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=128).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        flush_and_settle(index.as_ref()).await;

        // Test divergence exactly at jump boundaries (position 31, 32, 33, 63, 64, 65)
        for diverge_pos in [31usize, 32, 33, 63, 64, 65, 95, 96, 97] {
            let mut query: Vec<LocalBlockHash> = (1..=128).map(LocalBlockHash).collect();
            query[diverge_pos] = LocalBlockHash(99999);

            let scores = index.find_matches(query).await.unwrap();
            assert_eq!(
                *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
                diverge_pos as u32,
                "Divergence at position {} should match {} blocks",
                diverge_pos,
                diverge_pos
            );
        }
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_deep_continuation_chain(variant: &str) {
        let index = make_indexer(variant);

        // Build a very long sequence through many small continuations
        // This tests the parent_hash chain handling
        let chunk_size = 10;
        let num_chunks = 20; // Total 200 blocks

        let mut full_prefix: Vec<u64> = Vec::new();

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size + 1;
            let chunk: Vec<u64> = (chunk_start..chunk_start + chunk_size)
                .map(|x| x as u64)
                .collect();

            if chunk_idx == 0 {
                index.apply_event(make_store_event(0, &chunk)).await;
            } else {
                index
                    .apply_event(make_store_event_with_parent(0, &full_prefix, &chunk))
                    .await;
            }

            full_prefix.extend(&chunk);
        }

        flush_and_settle(index.as_ref()).await;

        // Query full sequence
        let full_query: Vec<LocalBlockHash> = (1..=200).map(LocalBlockHash).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            200
        );

        // Query partial prefix crossing multiple chunk boundaries
        let partial_query: Vec<LocalBlockHash> = (1..=75).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            75
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_clear_and_rebuild(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=100).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        flush_and_settle(index.as_ref()).await;

        // Verify it's stored
        let query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query.clone()).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Clear the worker
        index.apply_event(make_clear_event(0)).await;

        flush_and_settle(index.as_ref()).await;

        // Verify it's cleared
        let scores = index.find_matches(query.clone()).await.unwrap();
        assert!(scores.scores.is_empty());

        // Rebuild with a different sequence
        let new_sequence: Vec<u64> = (1001..=1100).collect();
        index.apply_event(make_store_event(0, &new_sequence)).await;

        flush_and_settle(index.as_ref()).await;

        // Verify new sequence works
        let new_query: Vec<LocalBlockHash> =
            new_sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(new_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Verify old sequence no longer matches
        let scores = index.find_matches(query).await.unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_multiple_workers_diverging(variant: &str) {
        let index = make_indexer(variant);

        // Multiple workers with long sequences that share a prefix then diverge
        // This tests precise drain point tracking across workers

        // All workers share prefix 1-40
        let shared_prefix: Vec<u64> = (1..=40).collect();

        // Worker 0: prefix + 41-100 (stores full sequence 1-100)
        let worker_0_full: Vec<u64> = (1..=100).collect();

        // Worker 1: prefix + 141-180 (diverges at block 41)
        let worker_1_suffix: Vec<u64> = (141..=180).collect();

        // Worker 2: prefix + 241-300 (diverges at block 41)
        let worker_2_suffix: Vec<u64> = (241..=300).collect();

        // Store for all workers
        index.apply_event(make_store_event(0, &worker_0_full)).await;

        index.apply_event(make_store_event(1, &shared_prefix)).await;
        index
            .apply_event(make_store_event_with_parent(
                1,
                &shared_prefix,
                &worker_1_suffix,
            ))
            .await;

        index.apply_event(make_store_event(2, &shared_prefix)).await;
        index
            .apply_event(make_store_event_with_parent(
                2,
                &shared_prefix,
                &worker_2_suffix,
            ))
            .await;

        flush_and_settle(index.as_ref()).await;

        // Query 1-100 - worker 0 matches 100, workers 1&2 match 40
        let query: Vec<LocalBlockHash> = worker_0_full.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query).await.unwrap();

        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            40
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            40
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_staggered_lengths(variant: &str) {
        let index = make_indexer(variant);

        // Workers with sequences of staggered lengths to test drain tracking
        // Worker 0: 10 blocks
        // Worker 1: 20 blocks
        // Worker 2: 35 blocks (just past first jump)
        // Worker 3: 64 blocks (exactly 2 jumps)
        // Worker 4: 100 blocks

        for (worker_id, len) in [(0, 10), (1, 20), (2, 35), (3, 64), (4, 100)] {
            let sequence: Vec<u64> = (1..=len).collect();
            index
                .apply_event(make_store_event(worker_id, &sequence))
                .await;
        }

        flush_and_settle(index.as_ref()).await;

        // Query for 100 blocks - each worker should match their stored length
        let query: Vec<LocalBlockHash> = (1..=100).map(LocalBlockHash).collect();
        let scores = index.find_matches(query).await.unwrap();

        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            10
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            20
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            35
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(3, 0)).unwrap(),
            64
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(4, 0)).unwrap(),
            100
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_very_long_sequence(variant: &str) {
        let index = make_indexer(variant);

        // Test with a very long sequence (1000 blocks)
        let seq_len = 1000u64;
        let sequence: Vec<u64> = (1..=seq_len).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        flush_and_settle(index.as_ref()).await;

        // Full match
        let full_query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            seq_len as u32
        );

        // Partial match (first 500)
        let partial_query: Vec<LocalBlockHash> = (1..=500).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            500
        );

        // Divergence in the middle
        let mut mid_diverge: Vec<LocalBlockHash> = (1..=1000).map(LocalBlockHash).collect();
        mid_diverge[499] = LocalBlockHash(99999);
        let scores = index.find_matches(mid_diverge).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            499
        );
    }
}

// ============================================================================
// Tests specific to pruning behavior.
// ============================================================================

#[tokio::test]
async fn test_routing_decision_assigns_first_seen_worker() {
    let token = CancellationToken::new();
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let index = KvIndexer::new_with_pruning(token, 32, metrics, Some(PruneConfig::default()));
    let worker = WorkerWithDpRank::new(42, 0);
    let local_hashes = vec![LocalBlockHash(11), LocalBlockHash(22)];
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);

    index
        .process_routing_decision_with_hashes(worker, local_hashes.clone(), sequence_hashes)
        .await
        .unwrap();
    flush_and_settle(&index).await;

    assert_score(&index, &[11, 22], worker, 2).await;

    index.remove_worker(worker.worker_id).await;
    flush_and_settle(&index).await;

    let scores = query_scores(&index, &[11, 22]).await;
    assert!(!scores.scores.contains_key(&worker));
}

// ============================================================================
// KvIndexerMetrics tests
// ============================================================================

mod metrics_tests {
    #[cfg(feature = "metrics")]
    use super::*;

    #[cfg(feature = "metrics")]
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

        metrics.increment_event_warning(METRIC_WARNING_DUPLICATE_STORE);
        assert_eq!(
            metrics
                .kv_cache_event_warnings
                .get_metric_with_label_values(&[METRIC_WARNING_DUPLICATE_STORE])
                .unwrap()
                .get(),
            1
        );
    }
}

// ============================================================================
// LocalKvIndexer tests
// ============================================================================

mod local_indexer_tests {
    use super::*;
    use rstest_reuse::apply;

    fn make_local_store_event(event_id: u64, block_hash: u64) -> RouterEvent {
        RouterEvent::new(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block_hash),
                        tokens_hash: LocalBlockHash(block_hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    fn make_local_remove_event(event_id: u64, block_hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: block_hashes
                        .iter()
                        .copied()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn make_local_clear_event(event_id: u64) -> RouterEvent {
        RouterEvent::new(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        )
    }

    #[tokio::test]
    async fn test_local_indexer_get_events_in_id_range_all_cases() {
        // Create indexer with small buffer (5 events max)
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        // Helper to create a test event
        let make_event = |id: u64| {
            RouterEvent::new(
                0,
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
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

        // Add 10 events (IDs 5-14), buffer keeps last 5: events 10-14
        for id in 5..15 {
            indexer
                .apply_event_with_buffer(make_event(id))
                .await
                .unwrap();
        }

        // Wait for events to be processed
        indexer.flush().await;

        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events { events: e, .. } => e,
                WorkerKvQueryResponse::TreeDump { events: e, .. } => e,
                _ => panic!("Unexpected response type: {:?}", resp),
            }
        };

        let extract_last_event_id = |resp: &WorkerKvQueryResponse| -> Option<u64> {
            match resp {
                WorkerKvQueryResponse::Events { last_event_id, .. } => Some(*last_event_id),
                WorkerKvQueryResponse::TreeDump { last_event_id, .. } => Some(*last_event_id),
                _ => None,
            }
        };

        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Verify buffer state
        let buffer_events = indexer.get_all_events_in_buffer();
        assert_eq!(get_ids(buffer_events), vec![10, 11, 12, 13, 14]);

        // Buffer hits return the contiguous suffix through the buffered tail.
        let result = indexer.get_events_in_id_range(Some(11), None).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(10), Some(14)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![10, 11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(11), Some(12)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(11), Some(20)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(12), Some(12)).await;
        assert_eq!(get_ids(extract_events(result.clone())), vec![12, 13, 14]);
        assert_eq!(extract_last_event_id(&result), Some(14));

        // Tree dump path tests
        let result = indexer.get_events_in_id_range(None, None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            } => {
                assert_eq!(events.len(), 10);
                assert_eq!(last_event_id, 14);
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }

        let result = indexer.get_events_in_id_range(Some(7), None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            } => {
                assert_eq!(events.len(), 10);
                assert_eq!(last_event_id, 14);
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }

        // Edge cases
        let result = indexer.get_events_in_id_range(Some(15), Some(10)).await;
        assert!(matches!(result, WorkerKvQueryResponse::InvalidRange { .. }));

        let result = indexer.get_events_in_id_range(Some(100), Some(200)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TooNew { .. }));

        let empty_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        let result = empty_indexer.get_events_in_id_range(None, None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                last_event_id,
                events,
            } => {
                assert_eq!(
                    last_event_id, 0,
                    "empty buffer should return last_event_id=0"
                );
                assert!(events.is_empty(), "empty indexer should have no events");
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_response_starts_at_last_clear() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        );

        for event in [
            make_local_store_event(10, 10),
            make_local_clear_event(11),
            make_local_store_event(12, 12),
            make_local_store_event(13, 13),
            make_local_clear_event(14),
            make_local_store_event(15, 15),
        ] {
            indexer.apply_event_with_buffer(event).await.unwrap();
        }
        indexer.flush().await;

        let event_ids = |response: WorkerKvQueryResponse| -> (Vec<u64>, u64) {
            match response {
                WorkerKvQueryResponse::Events {
                    events,
                    last_event_id,
                } => (
                    events.iter().map(|event| event.event.event_id).collect(),
                    last_event_id,
                ),
                other => panic!("Expected Events, got: {other:?}"),
            }
        };

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(10), None).await);
        assert_eq!(ids, vec![14, 15]);
        assert_eq!(last_event_id, 15);

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(12), None).await);
        assert_eq!(ids, vec![14, 15]);
        assert_eq!(last_event_id, 15);

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(15), None).await);
        assert_eq!(ids, vec![15]);
        assert_eq!(last_event_id, 15);
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_and_serialization() {
        let worker_id = 42u64;
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token, 4, metrics, 100));

        let test_event = RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        local_indexer
            .apply_event_with_buffer(test_event)
            .await
            .unwrap();

        local_indexer.flush().await;

        let buffered_events = local_indexer.get_all_events_in_buffer();
        assert_eq!(buffered_events.len(), 1);
        assert_eq!(buffered_events[0].worker_id, worker_id);

        // Test serialization round-trip
        let response = WorkerKvQueryResponse::Events {
            events: buffered_events,
            last_event_id: 1,
        };
        let serialized = serde_json::to_vec(&response).unwrap();
        let deserialized: WorkerKvQueryResponse = serde_json::from_slice(&serialized).unwrap();

        let (events, last_event_id) = match deserialized {
            WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            } => (events, last_event_id),
            _ => panic!("Expected Events variant"),
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].worker_id, worker_id);
        assert_eq!(last_event_id, 1);
    }

    #[tokio::test]
    async fn test_local_indexer_does_not_buffer_failed_send() {
        let local_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        let test_event = RouterEvent::new(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        let event_tx = local_indexer.event_sender();
        local_indexer.shutdown();
        event_tx.closed().await;

        let result = local_indexer.apply_event_with_buffer(test_event).await;
        assert!(matches!(result, Err(KvRouterError::IndexerOffline)));
        assert_eq!(local_indexer.buffer_len(), 0);

        match local_indexer.get_events_in_id_range(None, None).await {
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id,
                message,
            } => {
                assert_eq!(last_event_id, 0);
                assert_eq!(message, "Indexer is offline");
            }
            other => panic!("Expected TreeDumpFailed, got: {other:?}"),
        }
    }

    #[test]
    fn legacy_named_messagepack_query_defaults_explicit_dump_failure_capability_off() {
        #[derive(serde::Serialize)]
        struct LegacyWorkerKvQueryRequest {
            worker_id: WorkerId,
            dp_rank: DpRank,
            start_event_id: Option<u64>,
            end_event_id: Option<u64>,
        }

        let encoded = rmp_serde::to_vec_named(&LegacyWorkerKvQueryRequest {
            worker_id: 7,
            dp_rank: 3,
            start_event_id: None,
            end_event_id: Some(9),
        })
        .unwrap();
        let decoded: WorkerKvQueryRequest = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.worker_id, 7);
        assert_eq!(decoded.dp_rank, 3);
        assert_eq!(decoded.end_event_id, Some(9));
        assert!(!decoded.supports_tree_dump_failed);
    }

    #[tokio::test]
    async fn test_local_indexer_remove_worker_dp_rank_only_clears_target_rank() {
        let local_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        local_indexer
            .apply_event_with_buffer(make_store_event_with_dp_rank(7, &[101], 0))
            .await
            .unwrap();
        local_indexer
            .apply_event_with_buffer(make_store_event_with_dp_rank(7, &[202], 1))
            .await
            .unwrap();
        local_indexer.flush().await;

        local_indexer.remove_worker_dp_rank(7, 0).await;
        local_indexer.flush().await;

        let events = local_indexer.dump_events().await.unwrap();
        let mut rank0 = events
            .iter()
            .filter(|event| event.worker_id == 7 && event.event.dp_rank == 0)
            .collect::<Vec<_>>();
        let mut rank1 = events
            .iter()
            .filter(|event| event.worker_id == 7 && event.event.dp_rank == 1)
            .collect::<Vec<_>>();
        rank0.sort_by_key(|event| event.event.event_id);
        rank1.sort_by_key(|event| event.event.event_id);

        assert!(rank0.is_empty());
        assert_eq!(rank1.len(), 1);
        assert!(matches!(
            &rank1[0].event.data,
            KvCacheEventData::Stored(data)
                if data.blocks.first().map(|block| block.block_hash.0) == Some(202)
        ));
    }

    #[tokio::test]
    async fn test_local_indexer_coalesces_concurrent_tree_dumps() {
        let indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        ));
        indexer.set_dump_build_delay(Some(Duration::from_millis(50)));

        let first = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };
        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };

        let first = first.await.unwrap();
        let second = second.await.unwrap();

        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert!(matches!(second, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn test_local_indexer_reuses_cached_tree_dump_without_time_expiry() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        time::advance(Duration::from_secs(60)).await;
        let second = indexer.get_events_in_id_range(None, None).await;

        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert!(matches!(second, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);
    }

    #[tokio::test]
    async fn test_local_indexer_rebuilds_when_cumulative_append_budget_exceeded() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(2, 202))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(4, 404))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_appends_safe_tail_to_cached_dump() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_remove_event(2, &[101]))
            .await
            .unwrap();

        match indexer.get_events_in_id_range(None, None).await {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            } => {
                assert_eq!(last_event_id, 2);
                assert!(events.iter().any(|event| event.event.event_id == 2));
                assert!(
                    events
                        .iter()
                        .any(|event| matches!(event.event.data, KvCacheEventData::Removed(_)))
                );
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }
        assert_eq!(indexer.dump_build_count(), 1);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_clear() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_clear_event(2))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_event_gap() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_missing_tail_coverage() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            1,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(2, 202))
            .await
            .unwrap();
        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_failed_dump_is_not_cached() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        let dump_tx = indexer.snapshot_event_sender();
        indexer.shutdown();
        dump_tx.closed().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        let second = indexer.get_events_in_id_range(None, None).await;

        assert_eq!(indexer.dump_build_count(), 2);
        assert!(matches!(
            first,
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id: 0,
                ..
            }
        ));
        assert!(matches!(
            second,
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id: 0,
                ..
            }
        ));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_apply_events_idempotent(variant: &str) {
        let index = make_indexer(variant);

        // Setup: build initial tree
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[4, 5, 6])).await;
        index
            .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[7, 8]))
            .await;
        flush_and_settle(index.as_ref()).await;
        let s0 = snapshot_tree(index.as_ref()).await;

        // Mutation events: each add paired with its remove
        let adds = [
            make_store_event(2, &[1, 2, 9]),
            make_store_event_with_parent(1, &[4, 5, 6], &[10, 11, 12]),
        ];
        let removes = [
            make_remove_event(2, &[1, 2, 9]),
            make_remove_event_with_parent(1, &[4, 5, 6], &[10, 11, 12]),
        ];

        // Phase 1: interleaved add/remove
        index.apply_event(adds[0].clone()).await;
        index.apply_event(removes[0].clone()).await;
        index.apply_event(adds[1].clone()).await;
        index.apply_event(removes[1].clone()).await;
        flush_and_settle(index.as_ref()).await;
        let s1 = snapshot_tree(index.as_ref()).await;
        assert_eq!(
            s0, s1,
            "Phase 1: interleaved add/remove should restore tree"
        );

        // Phase 2: same interleaved again (idempotence of the full cycle)
        index.apply_event(adds[0].clone()).await;
        index.apply_event(removes[0].clone()).await;
        index.apply_event(adds[1].clone()).await;
        index.apply_event(removes[1].clone()).await;
        flush_and_settle(index.as_ref()).await;
        let s2 = snapshot_tree(index.as_ref()).await;
        assert_eq!(s1, s2, "Phase 2: repeated cycle should be idempotent");

        // Phase 3: non-interleaved (all adds then all removes)
        index.apply_event(adds[0].clone()).await;
        index.apply_event(adds[1].clone()).await;
        index.apply_event(removes[0].clone()).await;
        index.apply_event(removes[1].clone()).await;
        flush_and_settle(index.as_ref()).await;
        let s3 = snapshot_tree(index.as_ref()).await;
        assert_eq!(
            s2, s3,
            "Phase 3: non-interleaved ordering should restore tree"
        );
    }
}

/// Differential test: the binary search mode must produce byte-identical [`OverlapScores`] to the
/// strided mode for every query. Both indexers are fed an identical, seeded-random event stream
/// (multiple workers sharing prefixes that diverge at and around the jump_size=32 boundaries,
/// pure-prefix workers, and a 1300-block sequence), then queried with a curated edge-case set plus
/// hundreds of random queries. The whole comparison is repeated after a tail removal.
///
/// Only contiguous-from-zero stores and tail removals are used, which preserve the monotonic
/// subset property both searches rely on. Mid-chain removal is deliberately avoided (it is the
/// known positional-indexer limitation skipped by `test_remove_mid_chain_block`).
#[tokio::test]
async fn positional_binary_matches_strided_differential() {
    // Pin both modes explicitly (not via make_indexer, whose "flat" arm reads the env var) so the
    // comparison is genuinely strided-vs-binary regardless of any ambient DYN_ROUTER_* setting.
    let strided: Box<dyn KvIndexerInterface> = Box::new(ThreadPoolIndexer::new_with_metrics(
        PositionalIndexer::new_with_mode(32, SearchMode::Strided),
        4,
        32,
        None,
    ));
    let binary: Box<dyn KvIndexerInterface> = Box::new(ThreadPoolIndexer::new_with_metrics(
        PositionalIndexer::new_with_mode(32, SearchMode::Binary),
        4,
        32,
        None,
    ));

    let mut rng = fastrand::Rng::with_seed(0xC0FF_EED1_FF5E_ED42);

    const BASE_LEN: usize = 1300;
    let base: Vec<u64> = (0..BASE_LEN).map(|_| rng.u64(..)).collect();

    // Build the shared event stream once, then apply identical clones to both indexers.
    let mut events: Vec<RouterEvent> = Vec::new();

    // Worker 0 stores the entire base.
    events.push(make_store_event(0, &base));

    // Workers that share a base prefix then diverge with a unique random tail. Divergence at a
    // position d makes such a worker drain at d for any query that follows the base past d.
    let divergence_points = [
        1usize, 5, 31, 32, 33, 63, 64, 65, 100, 128, 500, 999, 1024, 1299,
    ];
    for (i, &d) in divergence_points.iter().enumerate() {
        let worker_id = (i + 1) as u64;
        let tail_len = 1 + rng.usize(0..40);
        let mut seq = base[..d].to_vec();
        seq.extend((0..tail_len).map(|_| rng.u64(..)));
        events.push(make_store_event(worker_id, &seq));
    }

    // Pure-prefix workers (store only base[..d]) drain queries exactly at d.
    let prefix_only = [10usize, 32, 64, 200, 1000];
    for (j, &d) in prefix_only.iter().enumerate() {
        let worker_id = (100 + j) as u64;
        events.push(make_store_event(worker_id, &base[..d]));
    }

    for ev in &events {
        strided.apply_event(ev.clone()).await;
        binary.apply_event(ev.clone()).await;
    }
    strided.flush().await;
    binary.flush().await;

    // Build the query set, starting with edge cases.
    let mut queries: Vec<Vec<u64>> = vec![
        vec![],                                 // empty
        vec![base[0]],                          // single element (hit)
        vec![rng.u64(..)],                      // single element (miss)
        (0..50).map(|_| rng.u64(..)).collect(), // pure miss
    ];

    // Base prefixes of many lengths, including jump_size boundaries.
    for len in [
        1usize, 2, 30, 31, 32, 33, 34, 62, 63, 64, 65, 66, 96, 127, 128, 129, 256, 500, 999, 1000,
        1024, 1100, 1300,
    ] {
        queries.push(base[..len.min(BASE_LEN)].to_vec());
    }

    // Queries that pass each divergence point (so the relevant worker drains there).
    for &d in divergence_points.iter() {
        queries.push(base[..(d + 10).min(BASE_LEN)].to_vec());
    }

    // Random divergent queries: a base prefix of random length plus a random tail.
    for _ in 0..600 {
        let p = rng.usize(0..BASE_LEN);
        let tail_len = rng.usize(0..30);
        let mut q = base[..p].to_vec();
        q.extend((0..tail_len).map(|_| rng.u64(..)));
        queries.push(q);
    }

    // Fully random queries.
    for _ in 0..600 {
        let len = rng.usize(0..(BASE_LEN + 10));
        queries.push((0..len).map(|_| rng.u64(..)).collect());
    }

    for q in &queries {
        let s = query_scores(strided.as_ref(), q).await;
        let b = query_scores(binary.as_ref(), q).await;
        assert_overlap_scores_eq(&s, &b, &format!("len={}", q.len()));
    }

    // Tail removal: drop base[1000..1300] from worker 0, leaving it matching only base[..1000].
    // This is a leaf/tail removal, so the subset property is preserved.
    let remove = make_remove_event_with_parent(0, &base[..1000], &base[1000..BASE_LEN]);
    strided.apply_event(remove.clone()).await;
    binary.apply_event(remove).await;
    strided.flush().await;
    binary.flush().await;

    for q in &queries {
        let s = query_scores(strided.as_ref(), q).await;
        let b = query_scores(binary.as_ref(), q).await;
        assert_overlap_scores_eq(&s, &b, &format!("after-remove len={}", q.len()));
    }
}
