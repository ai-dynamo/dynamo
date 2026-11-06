// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::{ExternalSequenceBlockHash, ResidencyOwnerKey, WorkerWithDpRank};

/// Optional query-local evidence collection during an attention-tree walk.
///
/// Candidate workers are announced before the first block. Block positions are
/// one-based from the request root; final worker scores bound each worker's
/// actual coverage. Callbacks must not mutate or reenter the indexer and should
/// only use query-local state or immutable snapshots: concurrent trees invoke
/// them under their existing node read guard.
pub trait MatchObserver: Send {
    /// Optional first-node candidate count, delivered before worker callbacks.
    /// This is a capacity hint; final match scores still determine coverage.
    fn reserve_workers(&mut self, _count: usize) {}
    fn observe_worker(&mut self, worker: WorkerWithDpRank);
    fn observe_cache_owner(&mut self, _owner: ResidencyOwnerKey) {}
    fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::indexer::{
        KvIndexer, KvIndexerMetrics, MatchDetails, RadixTree, ThreadPoolIndexer,
        concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed,
    };
    use crate::protocols::{LocalBlockHash, compute_seq_hash_for_block};
    use crate::test_utils::{make_remove_event_with_parent, make_store_event};

    #[derive(Default)]
    struct RecordingObserver {
        workers: Vec<WorkerWithDpRank>,
        blocks: Vec<(usize, ExternalSequenceBlockHash)>,
        reserved_workers: Option<usize>,
        worker_capacity: usize,
    }

    impl MatchObserver for RecordingObserver {
        fn reserve_workers(&mut self, count: usize) {
            assert!(self.workers.is_empty() && self.blocks.is_empty());
            assert!(self.reserved_workers.replace(count).is_none());
            self.workers.reserve(count);
            self.worker_capacity = self.workers.capacity();
        }

        fn observe_worker(&mut self, worker: WorkerWithDpRank) {
            assert!(self.reserved_workers.is_some());
            assert!(
                self.blocks.is_empty(),
                "workers must precede matched blocks"
            );
            self.workers.push(worker);
            assert_eq!(self.workers.capacity(), self.worker_capacity);
        }

        fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
            self.blocks.push((position, hash));
        }
    }

    fn query() -> Vec<LocalBlockHash> {
        vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(99)]
    }

    fn assert_observed_prefix(observer: &RecordingObserver, details: &MatchDetails) {
        assert_eq!(observer.reserved_workers, Some(2));
        let mut workers = observer.workers.clone();
        workers.sort_unstable();
        assert_eq!(
            workers,
            [WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)]
        );
        let hashes = compute_seq_hash_for_block(&query()[..2]);
        assert_eq!(
            observer.blocks,
            vec![
                (1, ExternalSequenceBlockHash(hashes[0])),
                (2, ExternalSequenceBlockHash(hashes[1])),
            ]
        );
        assert_eq!(
            details.overlap_scores.scores[&WorkerWithDpRank::new(1, 0)],
            2
        );
        assert_eq!(
            details.overlap_scores.scores[&WorkerWithDpRank::new(2, 0)],
            1
        );
    }

    #[test]
    fn single_tree_observes_positions_and_preserves_partial_worker_coverage() {
        let mut tree = RadixTree::new();
        for worker in [1, 2] {
            tree.apply_event(make_store_event(worker, &[11, 12, 13]))
                .unwrap();
        }
        tree.apply_event(make_remove_event_with_parent(2, &[11], &[12, 13]))
            .unwrap();
        let expected = tree.find_match_details_with_options(query(), false, true);
        let mut observer = RecordingObserver::default();
        let details = tree.find_match_details_with_observer(query(), false, true, &mut observer);
        assert_observed_prefix(&observer, &details);
        assert_eq!(
            details.overlap_scores.scores,
            expected.overlap_scores.scores
        );
        assert_eq!(details.last_matched_hashes, expected.last_matched_hashes);
        assert_eq!(
            details.kv_transfer_candidates.unwrap().block_hashes,
            expected.kv_transfer_candidates.unwrap().block_hashes,
        );
        for missed in [vec![], vec![LocalBlockHash(98)]] {
            let mut observer = RecordingObserver::default();
            let details =
                tree.find_match_details_with_observer(missed, false, false, &mut observer);
            assert!(details.overlap_scores.scores.is_empty());
            assert!(observer.workers.is_empty());
            assert!(observer.blocks.is_empty());
            assert!(observer.reserved_workers.is_none());
        }
    }

    #[tokio::test]
    async fn concurrent_tree_observes_positions_without_retaining_a_hint_chain() {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 16);
        for worker in [1, 2] {
            indexer
                .apply_event_and_wait(make_store_event(worker, &[11, 12, 13]))
                .await
                .unwrap();
        }
        indexer
            .apply_event_and_wait(make_remove_event_with_parent(2, &[11], &[12, 13]))
            .await
            .unwrap();
        let expected =
            indexer
                .backend()
                .find_match_details_impl_with_options(&query(), false, false);
        let mut observer = RecordingObserver::default();
        let details = indexer.backend().find_match_details_impl_with_observer(
            &query(),
            false,
            false,
            &mut observer,
        );
        assert_observed_prefix(&observer, &details);
        assert_eq!(
            details.overlap_scores.scores,
            expected.overlap_scores.scores
        );
        assert_eq!(details.last_matched_hashes, expected.last_matched_hashes);
        assert!(details.kv_transfer_candidates.is_none());
        for missed in [vec![], vec![LocalBlockHash(98)]] {
            let mut observer = RecordingObserver::default();
            let details = indexer.backend().find_match_details_impl_with_observer(
                &missed,
                false,
                false,
                &mut observer,
            );
            assert!(details.overlap_scores.scores.is_empty());
            assert!(observer.workers.is_empty());
            assert!(observer.blocks.is_empty());
            assert!(observer.reserved_workers.is_none());
        }
    }

    #[tokio::test]
    async fn concurrent_observer_keeps_absolute_positions_across_split_edges() {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 16);
        indexer
            .apply_event_and_wait(make_store_event(1, &[11, 12, 13]))
            .await
            .unwrap();
        indexer
            .apply_event_and_wait(make_store_event(2, &[11, 12, 14]))
            .await
            .unwrap();
        let query = [LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let mut observer = RecordingObserver::default();
        let details = indexer.backend().find_match_details_impl_with_observer(
            &query,
            false,
            false,
            &mut observer,
        );
        assert_eq!(
            observer.blocks,
            compute_seq_hash_for_block(&query)
                .into_iter()
                .enumerate()
                .map(|(i, hash)| (i + 1, ExternalSequenceBlockHash(hash)))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            details.overlap_scores.scores[&WorkerWithDpRank::new(1, 0)],
            3
        );
        assert_eq!(
            details.overlap_scores.scores[&WorkerWithDpRank::new(2, 0)],
            2
        );
    }

    #[tokio::test]
    async fn actor_returns_the_owned_collector_with_matches() {
        let stop = CancellationToken::new();
        let indexer = KvIndexer::new(
            stop.clone(),
            16,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        );
        for worker in [1, 2] {
            indexer
                .apply_event_and_wait(make_store_event(worker, &[11, 12, 13]))
                .await
                .unwrap();
        }
        indexer
            .apply_event_and_wait(make_remove_event_with_parent(2, &[11], &[12, 13]))
            .await
            .unwrap();
        let (details, observer) = indexer
            .find_match_details_with_observer(query(), false, RecordingObserver::default())
            .await
            .unwrap();
        assert_observed_prefix(&observer, &details);
        assert!(details.kv_transfer_candidates.is_none());
        stop.cancel();
    }
}
