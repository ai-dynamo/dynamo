// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::MatchDetails;
use crate::prelude::{
    ExternalSequenceBlockHash, LocalBlockHash, ResidencyOwnerKey, WorkerWithDpRank,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Coverage at the start of a compressed edge, before its blocks are compared.
#[derive(Clone, Copy)]
pub struct MatchEdge<'a> {
    pub full_workers: &'a FxHashSet<WorkerWithDpRank>,
    pub worker_cutoffs: &'a FxHashMap<WorkerWithDpRank, usize>,
    pub incoming_workers: &'a FxHashSet<WorkerWithDpRank>,
    pub prefix_depth: usize,
    pub first_node: bool,
    /// The attention walk's existing equal-set check needs no coverage update.
    pub all_workers_continue: bool,
}

/// The first edge's actual compared prefix, borrowed under its existing read
/// guard. Candidate cutoffs bound credit independently of the physical edge.
pub struct MatchFirstEdge<'a> {
    pub full_workers: &'a FxHashSet<WorkerWithDpRank>,
    pub worker_cutoffs: &'a FxHashMap<WorkerWithDpRank, usize>,
    pub matched_blocks: &'a [(LocalBlockHash, ExternalSequenceBlockHash)],
    pub query_blocks: usize,
    pub requested_endpoint: Option<ExternalSequenceBlockHash>,
    pub terminal: bool,
}

/// Optional query-local evidence collection during an attention-tree walk.
///
/// Candidate workers are announced before the first block. Block positions are
/// one-based from the request root; final worker scores bound each worker's
/// actual coverage. Callbacks must not mutate or reenter the indexer and should
/// only update query-local state. External reads may take checkpoint-record
/// locks, but must not acquire tree mutation locks: concurrent trees invoke
/// callbacks under their existing node read guard.
pub trait MatchObserver: Send {
    /// Return true to observe each block immediately after it matches. This
    /// replaces the first-edge proof and deferred callbacks for this edge.
    /// Coverage ending inside the edge must be frozen before a later block.
    fn begin_edge(&mut self, _edge: MatchEdge<'_>) -> bool {
        false
    }

    /// A proof-only observer needs the first-node boundary and final endpoint,
    /// but no per-worker or per-block callbacks. Returning false never changes
    /// the attention walk itself.
    fn observes_path(&self) -> bool {
        true
    }
    /// Optional first-node candidate count, delivered before worker callbacks.
    /// This is a capacity hint; final match scores still determine coverage.
    fn reserve_workers(&mut self, _count: usize) {}
    /// A final hash is available without another tree lookup when the request
    /// ends inside the first edge. This is only an admission hint: hashes still
    /// undergo their ordinary comparisons before any result is emitted.
    fn begin_path(&mut self, count: usize, _endpoint_hint: Option<ExternalSequenceBlockHash>) {
        self.reserve_workers(count);
    }
    /// Opt into a common credit bound after comparing the first edge. Returning
    /// a positive bound stops this query only; the tree additionally verifies
    /// it does not exceed the matched prefix or any candidate's cutoff. The
    /// observer is responsible for proving the bound applies to every worker.
    fn begin_first_edge(&mut self, edge: MatchFirstEdge<'_>) -> Option<usize> {
        self.begin_path(
            edge.full_workers.len() + edge.worker_cutoffs.len(),
            edge.requested_endpoint,
        );
        None
    }
    fn observe_worker(&mut self, worker: WorkerWithDpRank);
    fn observe_cache_owner(&mut self, _owner: ResidencyOwnerKey) {}
    fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash);

    /// Fused observers can borrow the walk's candidate coverage without copying it.
    fn observe_block_with_coverage(
        &mut self,
        position: usize,
        hash: ExternalSequenceBlockHash,
        _edge: &MatchEdge<'_>,
    ) {
        self.observe_block(position, hash);
    }

    /// Exact eligible workers at a common final endpoint. The tree verifies
    /// every returned worker while emitting its ordinary result. The observer
    /// must revalidate its evidence after the walk before using the caps.
    fn uniform_resume_workers(
        &mut self,
        _position: u32,
        _hash: ExternalSequenceBlockHash,
    ) -> Option<&FxHashSet<WorkerWithDpRank>> {
        None
    }

    /// Consume the query-owned surviving set while emitting final credit. Prior
    /// cutoff results are already in `details`; implementations must handle
    /// those too. Returning the set delegates ordinary emission to the tree.
    fn emit_match_details(
        &mut self,
        active: FxHashSet<WorkerWithDpRank>,
        _raw_end: u32,
        _raw_hash: Option<ExternalSequenceBlockHash>,
        _details: &mut MatchDetails,
    ) -> Result<(), FxHashSet<WorkerWithDpRank>> {
        Err(active)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::prelude::{LocalBlockHash, compute_seq_hash_for_block};
    use crate::testing::{make_remove_event_with_parent, make_store_event};
    use crate::{
        KvIndexer, KvIndexerMetrics, MatchDetails, RadixTree, ThreadPoolIndexer,
        concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed,
    };

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

    struct BoundedObserver {
        bound: usize,
        recording: RecordingObserver,
    }

    impl MatchObserver for BoundedObserver {
        fn begin_first_edge(&mut self, edge: MatchFirstEdge<'_>) -> Option<usize> {
            assert_eq!(edge.matched_blocks.len(), 2);
            assert!(edge.terminal);
            self.recording.begin_path(
                edge.full_workers.len() + edge.worker_cutoffs.len(),
                edge.requested_endpoint,
            );
            Some(self.bound)
        }

        fn observe_worker(&mut self, worker: WorkerWithDpRank) {
            self.recording.observe_worker(worker);
        }

        fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
            self.recording.observe_block(position, hash);
        }
    }

    #[tokio::test]
    async fn observer_credit_bound_is_limited_by_actual_matches_and_worker_cutoffs() {
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
        for bound in [0, 1, 2, 3, usize::MAX] {
            let mut observer = BoundedObserver {
                bound,
                recording: RecordingObserver::default(),
            };
            let details = indexer.backend().find_match_details_impl_with_observer(
                &query(),
                false,
                true,
                &mut observer,
            );
            let accepted = bound == 1;
            assert_eq!(
                details.overlap_scores.scores[&WorkerWithDpRank::new(1, 0)],
                if accepted { 1 } else { 2 },
            );
            assert_eq!(
                details.overlap_scores.scores[&WorkerWithDpRank::new(2, 0)],
                1
            );
            let chain = &details
                .kv_transfer_candidates
                .as_ref()
                .unwrap()
                .block_hashes;
            assert_eq!(chain.len(), if accepted { 1 } else { 2 });
            if accepted {
                assert!(observer.recording.blocks.is_empty());
            } else {
                assert_observed_prefix(&observer.recording, &details);
            }
        }
    }

    struct UniformProofObserver {
        eligible: FxHashSet<WorkerWithDpRank>,
        started: bool,
        endpoint: Option<(u32, ExternalSequenceBlockHash)>,
    }

    impl MatchObserver for UniformProofObserver {
        fn observes_path(&self) -> bool {
            false
        }

        fn reserve_workers(&mut self, _count: usize) {
            assert!(!self.started);
            self.started = true;
        }

        fn observe_worker(&mut self, _worker: WorkerWithDpRank) {
            panic!("proof-only observers do not capture individual workers");
        }

        fn observe_block(&mut self, _position: usize, _hash: ExternalSequenceBlockHash) {
            panic!("proof-only observers do not capture individual blocks");
        }

        fn uniform_resume_workers(
            &mut self,
            position: u32,
            hash: ExternalSequenceBlockHash,
        ) -> Option<&FxHashSet<WorkerWithDpRank>> {
            assert!(self.started);
            self.endpoint = Some((position, hash));
            Some(&self.eligible)
        }
    }

    #[tokio::test]
    async fn uniform_proof_requires_exact_workers_and_preserves_attention_results() {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 16);
        for worker in [1, 2] {
            indexer
                .apply_event_and_wait(make_store_event(worker, &[11, 12, 13]))
                .await
                .unwrap();
        }
        let query = query();
        let expected = indexer.backend().find_match_details_impl(&query, false);
        for eligible_ids in [[1, 2], [1, 3]] {
            let mut observer = UniformProofObserver {
                eligible: eligible_ids
                    .into_iter()
                    .map(|worker| WorkerWithDpRank::new(worker, 0))
                    .collect(),
                started: false,
                endpoint: None,
            };
            let details = indexer.backend().find_match_details_impl_with_observer(
                &query,
                false,
                false,
                &mut observer,
            );
            assert_eq!(
                details.overlap_scores.scores,
                expected.overlap_scores.scores
            );
            assert_eq!(details.last_matched_hashes, expected.last_matched_hashes);
            assert_eq!(observer.endpoint.unwrap().0, 2);
            if eligible_ids == [1, 2] {
                assert_eq!(details.hybrid_resume_caps.len(), 2);
                assert_eq!(
                    details.hybrid_resume_caps.get(&WorkerWithDpRank::new(2, 0)),
                    Some(&2)
                );
            } else {
                // Equal cardinality is not evidence of equal worker identities.
                assert!(details.hybrid_resume_caps.is_empty());
            }
        }
        indexer
            .apply_event_and_wait(make_remove_event_with_parent(2, &[11], &[12, 13]))
            .await
            .unwrap();
        let mut observer = UniformProofObserver {
            eligible: [WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)]
                .into_iter()
                .collect(),
            started: false,
            endpoint: None,
        };
        let details = indexer.backend().find_match_details_impl_with_observer(
            &query,
            false,
            false,
            &mut observer,
        );
        assert!(
            observer.endpoint.is_none(),
            "mixed cutoffs require the ordinary evaluator"
        );
        assert!(details.hybrid_resume_caps.is_empty());
        assert_eq!(
            details.overlap_scores.scores[&WorkerWithDpRank::new(2, 0)],
            1
        );
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
