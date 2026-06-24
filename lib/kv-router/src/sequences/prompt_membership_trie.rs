// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

#[cfg(any(test, feature = "bench"))]
pub(super) use crate::indexer::concurrent_radix_tree_compressed::lookup_live_hashes;
use crate::indexer::concurrent_radix_tree_compressed::{PromptMembershipIndex, PromptWorkerLookup};
use crate::protocols::WorkerWithDpRank;

pub(super) type WorkerLookup = PromptWorkerLookup;

#[derive(Default)]
pub(super) struct PromptMembershipTrie {
    index: PromptMembershipIndex,
}

impl PromptMembershipTrie {
    pub(super) fn new() -> Self {
        Self {
            index: PromptMembershipIndex::new(),
        }
    }

    pub(super) fn maybe_cleanup(&self) {
        self.index.maybe_cleanup();
    }

    pub(super) fn store_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
        parent: Option<SequenceHash>,
        hashes: &[SequenceHash],
    ) {
        self.index.store_chain(worker, lookup, parent, hashes);
    }

    pub(super) fn remove_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
        hashes: &[SequenceHash],
    ) {
        self.index.remove_chain(worker, lookup, hashes);
    }

    pub(super) fn remove_worker(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<WorkerLookup>>,
    ) {
        self.index.remove_worker(worker, lookup);
    }

    pub(super) fn compute_overlap_depths(
        &self,
        query: Option<&[SequenceHash]>,
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        self.index.compute_overlap_depths(query)
    }

    #[cfg(test)]
    pub(super) fn worker_hashes(
        &self,
    ) -> rustc_hash::FxHashMap<WorkerWithDpRank, rustc_hash::FxHashSet<SequenceHash>> {
        self.index.worker_hashes()
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    fn lookup() -> Arc<RwLock<WorkerLookup>> {
        Arc::new(RwLock::new(WorkerLookup::default()))
    }

    #[test]
    fn parent_continuation_chains_extend_and_trim() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3]);
        trie.store_chain(worker, &lookup, Some(3), &[4, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 5)]),
        );

        trie.remove_chain(worker, &lookup, &[4, 5]);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 3)]),
        );
    }

    #[test]
    fn branching_continuations_across_workers_match_expected_depths() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3, 4]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_a, 4), (worker_b, 2)]),
        );
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 5])),
            FxHashMap::from_iter([(worker_a, 2), (worker_b, 3)]),
        );
    }

    #[test]
    fn partial_suffix_removal_keeps_prefix() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3, 4, 5]);
        trie.remove_chain(worker, &lookup, &[3, 4, 5]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4, 5])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn remove_worker_preserves_other_workers() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 3]);

        trie.remove_worker(worker_a, &lookup_a);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_b, 3)]),
        );
    }

    #[test]
    fn multiple_dp_ranks_with_same_worker_id_remain_isolated() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(1, 1);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker_a, 3), (worker_b, 2)]),
        );
    }

    #[test]
    fn clear_worker_state_then_reuse_starts_empty() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3]);
        trie.remove_worker(worker, &lookup);
        assert!(trie.compute_overlap_depths(Some(&[1, 2, 3])).is_empty());

        trie.store_chain(worker, &lookup, None, &[1, 2]);
        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3])),
            FxHashMap::from_iter([(worker, 2)]),
        );
    }

    #[test]
    fn redundant_batched_remove_is_idempotent() {
        let trie = PromptMembershipTrie::new();
        let worker = worker(1, 0);
        let lookup = lookup();

        trie.store_chain(worker, &lookup, None, &[1, 2, 3, 4]);
        trie.remove_chain(worker, &lookup, &[2, 3, 4]);
        trie.remove_chain(worker, &lookup, &[2, 3, 4]);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker, 1)]),
        );
    }

    #[test]
    fn stale_lookup_repair_removes_worker_from_split_suffix() {
        let trie = PromptMembershipTrie::new();
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let lookup_a = lookup();
        let lookup_b = lookup();

        trie.store_chain(worker_a, &lookup_a, None, &[1, 2, 3, 4]);
        trie.store_chain(worker_b, &lookup_b, None, &[1, 2, 5]);
        trie.remove_worker(worker_a, &lookup_a);

        assert_eq!(
            trie.compute_overlap_depths(Some(&[1, 2, 3, 4])),
            FxHashMap::from_iter([(worker_b, 2)]),
        );
    }
}
