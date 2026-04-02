// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact lower-tier KV continuation index.
//!
//! This structure stores worker ownership over shared continuation edges in the
//! event hash space: `(parent_sequence_hash, local_hash) -> child_sequence_hash`.
//!
//! Unlike the primary KV indexers, this index does not attempt prefix-overlap
//! scoring. Queries continue from a caller-provided per-worker continuation
//! point and count how many consecutive lower-tier blocks are present.
//!
//! The index treats lower-tier state as a set of unique continuation edges. If a
//! duplicate or conflicting store arrives, the existing mapping wins and the new
//! event is ignored.

use std::{collections::BTreeMap, hash::BuildHasher};

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::{SyncIndexer, WorkerTask};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheStoreData,
    LocalBlockHash, OverlapScores, RouterEvent, WorkerWithDpRank,
};

type WorkerSet = FxHashSet<WorkerWithDpRank>;
type FrontierBuckets = FxHashMap<Option<ExternalSequenceBlockHash>, WorkerSet>;
type Frontier = BTreeMap<usize, FrontierBuckets>;
type FinalStates = FxHashMap<WorkerWithDpRank, (usize, Option<ExternalSequenceBlockHash>)>;
type WorkerBlockIndex =
    FxHashMap<WorkerWithDpRank, FxHashMap<ExternalSequenceBlockHash, TransitionKey>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    parent_hash: Option<ExternalSequenceBlockHash>,
    local_hash: LocalBlockHash,
}

#[derive(Debug, Clone)]
enum EdgeOwnersEntry {
    Single {
        child_hash: ExternalSequenceBlockHash,
        owner: WorkerWithDpRank,
    },
    Multi {
        child_hash: ExternalSequenceBlockHash,
        owners: WorkerSet,
    },
}

impl EdgeOwnersEntry {
    fn new(child_hash: ExternalSequenceBlockHash, owner: WorkerWithDpRank) -> Self {
        Self::Single { child_hash, owner }
    }

    fn child_hash(&self) -> ExternalSequenceBlockHash {
        match self {
            Self::Single { child_hash, .. } | Self::Multi { child_hash, .. } => *child_hash,
        }
    }

    fn insert(&mut self, child_hash: ExternalSequenceBlockHash, owner: WorkerWithDpRank) -> bool {
        match self {
            Self::Single {
                child_hash: existing_hash,
                owner: existing_owner,
            } => {
                if *existing_hash != child_hash {
                    return false;
                }

                if *existing_owner == owner {
                    return true;
                }

                let mut owners = WorkerSet::default();
                owners.insert(*existing_owner);
                owners.insert(owner);
                *self = Self::Multi { child_hash, owners };
                true
            }
            Self::Multi {
                child_hash: existing_hash,
                owners,
            } => {
                if *existing_hash != child_hash {
                    return false;
                }
                owners.insert(owner);
                true
            }
        }
    }

    fn remove(&mut self, owner: WorkerWithDpRank) -> bool {
        match self {
            Self::Single {
                owner: existing_owner,
                ..
            } => *existing_owner == owner,
            Self::Multi { child_hash, owners } => {
                if !owners.remove(&owner) {
                    return false;
                }

                if owners.is_empty() {
                    return true;
                }

                if owners.len() == 1 {
                    let remaining_owner = owners.iter().next().copied().unwrap();
                    *self = Self::Single {
                        child_hash: *child_hash,
                        owner: remaining_owner,
                    };
                }

                false
            }
        }
    }

    fn contains(&self, owner: &WorkerWithDpRank) -> bool {
        match self {
            Self::Single {
                owner: existing_owner,
                ..
            } => existing_owner == owner,
            Self::Multi { owners, .. } => owners.contains(owner),
        }
    }

    fn collect_workers(&self) -> Vec<WorkerWithDpRank> {
        match self {
            Self::Single { owner, .. } => vec![*owner],
            Self::Multi { owners, .. } => owners.iter().copied().collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowerTierContinuation {
    pub start_pos: usize,
    pub last_matched_hash: Option<ExternalSequenceBlockHash>,
}

impl LowerTierContinuation {
    pub fn new(start_pos: usize, last_matched_hash: ExternalSequenceBlockHash) -> Self {
        Self {
            start_pos,
            last_matched_hash: Some(last_matched_hash),
        }
    }

    pub fn from_root(start_pos: usize) -> Self {
        Self {
            start_pos,
            last_matched_hash: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LowerTierMatchDetails {
    pub hits: FxHashMap<WorkerWithDpRank, usize>,
    pub next_continuations: FxHashMap<WorkerWithDpRank, LowerTierContinuation>,
}

/// Standalone lower-tier continuation index.
pub struct LowerTierIndexer {
    edges: DashMap<TransitionKey, EdgeOwnersEntry, FxBuildHasher>,
}

impl LowerTierIndexer {
    pub fn new() -> Self {
        Self {
            edges: DashMap::with_hasher(FxBuildHasher),
        }
    }

    fn apply_event(
        &self,
        worker_blocks: &mut WorkerBlockIndex,
        event: RouterEvent,
    ) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                self.store_blocks_impl(worker_blocks, worker, store_data);
                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                self.remove_blocks_impl(worker_blocks, worker, &remove_data.block_hashes)
            }
            KvCacheEventData::Cleared => {
                self.clear_worker_impl(worker_blocks, event.worker_id);
                Ok(())
            }
        }
    }

    fn store_blocks_impl(
        &self,
        worker_blocks: &mut WorkerBlockIndex,
        worker: WorkerWithDpRank,
        store_data: KvCacheStoreData,
    ) {
        let mut parent_hash = store_data.parent_hash;
        let worker_map = worker_blocks.entry(worker).or_default();

        for block in store_data.blocks {
            let key = TransitionKey {
                parent_hash,
                local_hash: block.tokens_hash,
            };

            if worker_map
                .get(&block.block_hash)
                .is_some_and(|existing_key| *existing_key != key)
            {
                parent_hash = Some(block.block_hash);
                continue;
            }

            let inserted = match self.edges.entry(key) {
                dashmap::mapref::entry::Entry::Occupied(mut edge) => {
                    edge.get_mut().insert(block.block_hash, worker)
                }
                dashmap::mapref::entry::Entry::Vacant(edge) => {
                    edge.insert(EdgeOwnersEntry::new(block.block_hash, worker));
                    true
                }
            };

            if inserted {
                worker_map.insert(block.block_hash, key);
            }
            parent_hash = Some(block.block_hash);
        }
    }

    fn remove_blocks_impl(
        &self,
        worker_blocks: &mut WorkerBlockIndex,
        worker: WorkerWithDpRank,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), KvCacheEventError> {
        let remove_worker_entry = {
            let Some(worker_map) = worker_blocks.get_mut(&worker) else {
                return Err(KvCacheEventError::BlockNotFound);
            };

            for block_hash in block_hashes {
                let Some(key) = worker_map.remove(block_hash) else {
                    return Err(KvCacheEventError::BlockNotFound);
                };

                let remove_edge = match self.edges.get_mut(&key) {
                    Some(mut edge) => edge.remove(worker),
                    None => false,
                };
                if remove_edge {
                    self.edges.remove(&key);
                }
            }

            worker_map.is_empty()
        };

        if remove_worker_entry {
            worker_blocks.remove(&worker);
        }

        Ok(())
    }

    fn clear_worker_impl(&self, worker_blocks: &mut WorkerBlockIndex, worker_id: u64) {
        let workers: Vec<_> = worker_blocks
            .keys()
            .copied()
            .filter(|worker| worker.worker_id == worker_id)
            .collect();

        for worker in workers {
            self.remove_worker_dp_rank_impl(worker_blocks, worker);
        }
    }

    fn remove_worker_dp_rank_impl(
        &self,
        worker_blocks: &mut WorkerBlockIndex,
        worker: WorkerWithDpRank,
    ) {
        let Some(worker_map) = worker_blocks.remove(&worker) else {
            return;
        };

        for (_, key) in worker_map {
            let remove_edge = match self.edges.get_mut(&key) {
                Some(mut edge) => edge.remove(worker),
                None => false,
            };
            if remove_edge {
                self.edges.remove(&key);
            }
        }
    }

    fn remove_worker(&self, worker_blocks: &mut WorkerBlockIndex, worker_id: u64) {
        self.clear_worker_impl(worker_blocks, worker_id);
    }

    fn remove_worker_dp_rank(
        &self,
        worker_blocks: &mut WorkerBlockIndex,
        worker_id: u64,
        dp_rank: u32,
    ) {
        self.remove_worker_dp_rank_impl(worker_blocks, WorkerWithDpRank::new(worker_id, dp_rank));
    }

    pub fn root_workers(&self, local_hash: LocalBlockHash) -> Vec<WorkerWithDpRank> {
        self.edges
            .get(&TransitionKey {
                parent_hash: None,
                local_hash,
            })
            .map(|edge| edge.collect_workers())
            .unwrap_or_default()
    }

    pub fn query_contiguous_hits<S>(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
    ) -> FxHashMap<WorkerWithDpRank, usize>
    where
        S: BuildHasher,
    {
        self.query_match_details(local_hashes, continuations).hits
    }

    pub fn query_match_details<S>(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
    ) -> LowerTierMatchDetails
    where
        S: BuildHasher,
    {
        let mut frontier = Frontier::new();
        for (worker, continuation) in continuations {
            frontier
                .entry(continuation.start_pos)
                .or_default()
                .entry(continuation.last_matched_hash)
                .or_default()
                .insert(*worker);
        }

        let mut final_states = FinalStates::default();

        while let Some((&pos, _)) = frontier.iter().next() {
            let states = frontier.remove(&pos).unwrap();
            let next_breakpoint = frontier
                .keys()
                .next()
                .copied()
                .unwrap_or(local_hashes.len());

            for (parent_hash, workers) in states {
                advance_state_to_breakpoint(
                    self,
                    local_hashes,
                    pos,
                    parent_hash,
                    workers,
                    next_breakpoint,
                    &mut frontier,
                    &mut final_states,
                );
            }
        }

        let mut results = LowerTierMatchDetails::default();
        for (worker, continuation) in continuations {
            let (final_pos, final_hash) = final_states
                .get(worker)
                .copied()
                .unwrap_or((continuation.start_pos, continuation.last_matched_hash));

            let hits = final_pos.saturating_sub(continuation.start_pos);
            results.hits.insert(*worker, hits);

            let next_continuation = if hits == 0 {
                *continuation
            } else {
                LowerTierContinuation {
                    start_pos: final_pos,
                    last_matched_hash: final_hash.or(continuation.last_matched_hash),
                }
            };
            results
                .next_continuations
                .insert(*worker, next_continuation);
        }

        results
    }
}

impl Default for LowerTierIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncIndexer for LowerTierIndexer {
    fn worker(&self, event_receiver: flume::Receiver<WorkerTask>) -> anyhow::Result<()> {
        let mut worker_blocks = WorkerBlockIndex::default();

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    if let Err(error) = self.apply_event(&mut worker_blocks, event) {
                        tracing::warn!(%error, "Failed to apply lower-tier event");
                    }
                }
                WorkerTask::RemoveWorker(worker_id) => {
                    self.remove_worker(&mut worker_blocks, worker_id);
                }
                WorkerTask::RemoveWorkerDpRank(worker_id, dp_rank) => {
                    self.remove_worker_dp_rank(&mut worker_blocks, worker_id, dp_rank);
                }
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Ok(Vec::new()));
                }
                WorkerTask::Terminate => {
                    break;
                }
            }
        }

        tracing::debug!("LowerTierIndexer worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        let Some(&first_hash) = sequence.first() else {
            return OverlapScores::default();
        };

        let mut continuations = FxHashMap::default();
        for worker in self.root_workers(first_hash) {
            continuations.insert(worker, LowerTierContinuation::from_root(0));
        }

        let hits = self.query_contiguous_hits(sequence, &continuations);
        let mut scores = OverlapScores::default();
        for (worker, hits) in hits {
            if hits > 0 {
                scores
                    .scores
                    .insert(worker, hits.min(u32::MAX as usize) as u32);
            }
        }

        scores
    }
}

fn advance_state_to_breakpoint(
    index: &LowerTierIndexer,
    local_hashes: &[LocalBlockHash],
    start_pos: usize,
    start_hash: Option<ExternalSequenceBlockHash>,
    workers: WorkerSet,
    next_breakpoint: usize,
    frontier: &mut Frontier,
    final_states: &mut FinalStates,
) {
    let mut cur_pos = start_pos;
    let mut cur_hash = start_hash;
    let mut active = workers;

    while cur_pos < next_breakpoint && !active.is_empty() {
        let Some(edge) = index.edges.get(&TransitionKey {
            parent_hash: cur_hash,
            local_hash: local_hashes[cur_pos],
        }) else {
            finalize_workers(final_states, active.drain(), cur_pos, cur_hash);
            break;
        };

        let mut matched = WorkerSet::default();
        let mut unmatched = WorkerSet::default();
        for worker in active.drain() {
            if edge.contains(&worker) {
                matched.insert(worker);
            } else {
                unmatched.insert(worker);
            }
        }

        finalize_workers(final_states, unmatched, cur_pos, cur_hash);
        if matched.is_empty() {
            break;
        }

        active = matched;
        cur_hash = Some(edge.child_hash());
        cur_pos += 1;
    }

    if active.is_empty() {
        return;
    }

    if cur_pos >= local_hashes.len() {
        finalize_workers(final_states, active, cur_pos, cur_hash);
    } else {
        frontier
            .entry(cur_pos)
            .or_default()
            .entry(cur_hash)
            .or_default()
            .extend(active);
    }
}

fn finalize_workers(
    final_states: &mut FinalStates,
    workers: impl IntoIterator<Item = WorkerWithDpRank>,
    pos: usize,
    parent_hash: Option<ExternalSequenceBlockHash>,
) {
    for worker in workers {
        final_states.insert(worker, (pos, parent_hash));
    }
}

#[cfg(test)]
mod tests {
    use super::{LowerTierContinuation, LowerTierIndexer, WorkerBlockIndex};
    use rustc_hash::FxHashMap;

    use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, LocalBlockHash,
        WorkerWithDpRank,
    };
    use crate::test_utils::{remove_event, router_event, stored_blocks_with_sequence_hashes};

    fn local_hashes(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        parent_hash: Option<u64>,
        local_values: &[u64],
        external_hashes: &[u64],
    ) -> crate::protocols::RouterEvent {
        router_event(
            worker_id,
            event_id,
            dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: stored_blocks_with_sequence_hashes(
                    &local_hashes(local_values),
                    external_hashes,
                ),
            }),
        )
    }

    struct TestLowerTierIndex {
        index: LowerTierIndexer,
        worker_blocks: WorkerBlockIndex,
    }

    impl TestLowerTierIndex {
        fn new() -> Self {
            Self {
                index: LowerTierIndexer::new(),
                worker_blocks: WorkerBlockIndex::default(),
            }
        }

        fn apply_event(
            &mut self,
            event: crate::protocols::RouterEvent,
        ) -> Result<(), crate::protocols::KvCacheEventError> {
            self.index.apply_event(&mut self.worker_blocks, event)
        }

        fn remove_worker(&mut self, worker_id: u64) {
            self.index.remove_worker(&mut self.worker_blocks, worker_id);
        }

        fn remove_worker_dp_rank(&mut self, worker_id: u64, dp_rank: u32) {
            self.index
                .remove_worker_dp_rank(&mut self.worker_blocks, worker_id, dp_rank);
        }

        fn root_workers(&self, local_hash: LocalBlockHash) -> Vec<WorkerWithDpRank> {
            self.index.root_workers(local_hash)
        }

        fn query_contiguous_hits<S>(
            &self,
            local_hashes: &[LocalBlockHash],
            continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
        ) -> FxHashMap<WorkerWithDpRank, usize>
        where
            S: std::hash::BuildHasher,
        {
            self.index
                .query_contiguous_hits(local_hashes, continuations)
        }

        fn query_match_details<S>(
            &self,
            local_hashes: &[LocalBlockHash],
            continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
        ) -> super::LowerTierMatchDetails
        where
            S: std::hash::BuildHasher,
        {
            self.index.query_match_details(local_hashes, continuations)
        }
    }

    #[test]
    fn root_query_uses_none_parent_transition() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(7, 0, 0, None, &[11, 12, 13], &[101, 102, 103]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(7, 0),
            LowerTierContinuation::from_root(0),
        );

        let hits = index.query_contiguous_hits(&local_hashes(&[11, 12, 13]), &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(7, 0)), Some(&3));
    }

    #[test]
    fn root_workers_only_include_matching_root_edges() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(7, 0, 0, None, &[11, 12], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(8, 0, 1, Some(500), &[11], &[201]))
            .unwrap();

        let workers = index.root_workers(LocalBlockHash(11));
        assert_eq!(workers.len(), 1);
        assert!(workers.contains(&WorkerWithDpRank::new(7, 0)));
    }

    #[tokio::test]
    async fn thread_pool_backend_applies_lower_tier_events() {
        let index = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, 1);
        let worker = WorkerWithDpRank::new(7, 0);

        index
            .apply_event(store_event(7, 0, 0, None, &[11, 12], &[101, 102]))
            .await;
        let _ = index.dump_events().await.unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker, LowerTierContinuation::from_root(0));

        let hits = index
            .backend()
            .query_contiguous_hits(&local_hashes(&[11, 12]), &continuations);
        assert_eq!(hits.get(&worker), Some(&2));
    }

    #[tokio::test]
    async fn thread_pool_backend_remove_worker_dp_rank_keeps_other_rank() {
        let index = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, 1);
        let worker_dp0 = WorkerWithDpRank::new(43, 0);
        let worker_dp1 = WorkerWithDpRank::new(43, 1);

        index
            .apply_event(store_event(43, 0, 0, None, &[11], &[101]))
            .await;
        index
            .apply_event(store_event(43, 1, 1, None, &[11], &[101]))
            .await;
        let _ = index.dump_events().await.unwrap();

        index.remove_worker_dp_rank(43, 0).await;
        let _ = index.dump_events().await.unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_dp0, LowerTierContinuation::from_root(0));
        continuations.insert(worker_dp1, LowerTierContinuation::from_root(0));

        let hits = index
            .backend()
            .query_contiguous_hits(&local_hashes(&[11]), &continuations);
        assert_eq!(hits.get(&worker_dp0), Some(&0));
        assert_eq!(hits.get(&worker_dp1), Some(&1));
    }

    #[tokio::test]
    async fn thread_pool_backend_cleared_event_preserves_other_workers() {
        let index = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, 1);
        let worker_a = WorkerWithDpRank::new(29, 0);
        let worker_b = WorkerWithDpRank::new(30, 0);

        index
            .apply_event(store_event(29, 0, 0, None, &[101, 102], &[1001, 1002]))
            .await;
        index
            .apply_event(store_event(30, 0, 1, None, &[101, 102], &[1001, 1002]))
            .await;
        index
            .apply_event(router_event(29, 2, 0, KvCacheEventData::Cleared))
            .await;
        let _ = index.dump_events().await.unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_a, LowerTierContinuation::from_root(0));
        continuations.insert(worker_b, LowerTierContinuation::from_root(0));

        let hits = index
            .backend()
            .query_contiguous_hits(&local_hashes(&[101, 102]), &continuations);
        assert_eq!(hits.get(&worker_a), Some(&0));
        assert_eq!(hits.get(&worker_b), Some(&2));
    }

    #[test]
    fn missing_parent_tail_queries_exactly_from_last_matched_hash() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                3,
                0,
                0,
                Some(999),
                &[21, 22, 23],
                &[201, 202, 203],
            ))
            .unwrap();

        let query = local_hashes(&[1, 2, 21, 22, 23]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(3, 0),
            LowerTierContinuation::new(2, ExternalSequenceBlockHash(999)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(3, 0)), Some(&3));
    }

    #[test]
    fn mid_segment_continuation_works_without_materialization() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                5,
                0,
                0,
                Some(700),
                &[31, 32, 33],
                &[301, 302, 303],
            ))
            .unwrap();

        let query = local_hashes(&[10, 31, 32, 33]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(5, 0),
            LowerTierContinuation::new(2, ExternalSequenceBlockHash(301)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(5, 0)), Some(&2));
    }

    #[test]
    fn branch_matching_is_exact_by_parent_hash() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(9, 0, 0, Some(500), &[91, 92], &[901, 902]))
            .unwrap();
        index
            .apply_event(store_event(9, 0, 1, Some(700), &[91, 93], &[903, 904]))
            .unwrap();

        let query = local_hashes(&[91, 92]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(9, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(500)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(9, 0)), Some(&2));

        continuations.insert(
            WorkerWithDpRank::new(9, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(700)),
        );
        let branch_b_hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(branch_b_hits.get(&WorkerWithDpRank::new(9, 0)), Some(&1));
    }

    #[test]
    fn shared_worker_traversal_fuses_at_descendant_breakpoint() {
        let mut index = TestLowerTierIndex::new();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);

        index
            .apply_event(store_event(
                1,
                0,
                0,
                None,
                &[11, 12, 13, 14],
                &[101, 102, 103, 104],
            ))
            .unwrap();
        index
            .apply_event(store_event(2, 0, 1, Some(102), &[13, 14], &[103, 104]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_a, LowerTierContinuation::from_root(0));
        continuations.insert(
            worker_b,
            LowerTierContinuation::new(2, ExternalSequenceBlockHash(102)),
        );

        let details = index.query_match_details(&local_hashes(&[11, 12, 13, 14]), &continuations);
        assert_eq!(details.hits.get(&worker_a), Some(&4));
        assert_eq!(details.hits.get(&worker_b), Some(&2));
        assert_eq!(
            details.next_continuations.get(&worker_a),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104)
            ))
        );
        assert_eq!(
            details.next_continuations.get(&worker_b),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104)
            ))
        );
    }

    #[test]
    fn shared_worker_traversal_fuses_across_multiple_breakpoints() {
        let mut index = TestLowerTierIndex::new();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let worker_c = WorkerWithDpRank::new(3, 0);

        index
            .apply_event(store_event(
                1,
                0,
                0,
                None,
                &[11, 12, 13, 14],
                &[101, 102, 103, 104],
            ))
            .unwrap();
        index
            .apply_event(store_event(
                2,
                0,
                1,
                Some(101),
                &[12, 13, 14],
                &[102, 103, 104],
            ))
            .unwrap();
        index
            .apply_event(store_event(3, 0, 2, Some(103), &[14], &[104]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_a, LowerTierContinuation::from_root(0));
        continuations.insert(
            worker_b,
            LowerTierContinuation::new(1, ExternalSequenceBlockHash(101)),
        );
        continuations.insert(
            worker_c,
            LowerTierContinuation::new(3, ExternalSequenceBlockHash(103)),
        );

        let details = index.query_match_details(&local_hashes(&[11, 12, 13, 14]), &continuations);
        assert_eq!(details.hits.get(&worker_a), Some(&4));
        assert_eq!(details.hits.get(&worker_b), Some(&3));
        assert_eq!(details.hits.get(&worker_c), Some(&1));
        assert_eq!(
            details.next_continuations.get(&worker_a),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104)
            ))
        );
        assert_eq!(
            details.next_continuations.get(&worker_b),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104)
            ))
        );
        assert_eq!(
            details.next_continuations.get(&worker_c),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104)
            ))
        );
    }

    #[test]
    fn duplicate_store_is_idempotent_for_remove() {
        let mut index = TestLowerTierIndex::new();
        let event = store_event(13, 0, 0, Some(800), &[61], &[601]);
        index.apply_event(event.clone()).unwrap();
        index.apply_event(event).unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(13, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(800)),
        );
        let query = local_hashes(&[61]);
        let initial = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(initial.get(&WorkerWithDpRank::new(13, 0)), Some(&1));

        index
            .apply_event(remove_event(13, 1, 0, vec![ExternalSequenceBlockHash(601)]))
            .unwrap();
        let after_one_remove = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(
            after_one_remove.get(&WorkerWithDpRank::new(13, 0)),
            Some(&0)
        );
    }

    #[test]
    fn removing_one_owner_preserves_shared_edge_for_other_workers() {
        let mut index = TestLowerTierIndex::new();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);

        index
            .apply_event(store_event(1, 0, 0, None, &[11, 12], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(2, 0, 1, None, &[11, 12], &[101, 102]))
            .unwrap();
        index
            .apply_event(remove_event(
                1,
                2,
                0,
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                ],
            ))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_a, LowerTierContinuation::from_root(0));
        continuations.insert(worker_b, LowerTierContinuation::from_root(0));
        let hits = index.query_contiguous_hits(&local_hashes(&[11, 12]), &continuations);

        assert_eq!(hits.get(&worker_a), Some(&0));
        assert_eq!(hits.get(&worker_b), Some(&2));
    }

    #[test]
    fn remove_stops_contiguous_walk_at_missing_edge() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                17,
                0,
                0,
                Some(900),
                &[71, 72, 73],
                &[701, 702, 703],
            ))
            .unwrap();

        index
            .apply_event(remove_event(17, 1, 0, vec![ExternalSequenceBlockHash(702)]))
            .unwrap();

        let query = local_hashes(&[71, 72, 73]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(17, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(900)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(17, 0)), Some(&1));
    }

    #[test]
    fn unknown_last_matched_hash_returns_zero() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(19, 0, 0, Some(1000), &[81, 82], &[801, 802]))
            .unwrap();

        let query = local_hashes(&[81, 82]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(19, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(9999)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(19, 0)), Some(&0));
    }

    #[test]
    fn start_pos_past_end_returns_zero() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(23, 0, 0, Some(1100), &[91], &[901]))
            .unwrap();

        let query = local_hashes(&[91]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(23, 0),
            LowerTierContinuation::new(1, ExternalSequenceBlockHash(1100)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(23, 0)), Some(&0));
    }

    #[test]
    fn cleared_event_removes_all_lower_tier_state() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                29,
                0,
                0,
                Some(1200),
                &[101, 102],
                &[1001, 1002],
            ))
            .unwrap();
        index
            .apply_event(router_event(29, 1, 0, KvCacheEventData::Cleared))
            .unwrap();

        let query = local_hashes(&[101, 102]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(29, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(1200)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(29, 0)), Some(&0));
    }

    #[test]
    fn cleared_event_is_worker_wide_across_dp_ranks() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(29, 0, 0, Some(1200), &[101], &[1001]))
            .unwrap();
        index
            .apply_event(store_event(29, 1, 1, Some(2200), &[201], &[2001]))
            .unwrap();
        index
            .apply_event(router_event(29, 2, 0, KvCacheEventData::Cleared))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(29, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(1200)),
        );
        continuations.insert(
            WorkerWithDpRank::new(29, 1),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(2200)),
        );

        let hits = index.query_contiguous_hits(&local_hashes(&[101]), &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(29, 0)), Some(&0));
        assert_eq!(hits.get(&WorkerWithDpRank::new(29, 1)), Some(&0));
    }

    #[test]
    fn cleared_event_preserves_shared_edges_for_other_workers() {
        let mut index = TestLowerTierIndex::new();
        let worker_a = WorkerWithDpRank::new(29, 0);
        let worker_b = WorkerWithDpRank::new(30, 0);

        index
            .apply_event(store_event(29, 0, 0, None, &[101, 102], &[1001, 1002]))
            .unwrap();
        index
            .apply_event(store_event(30, 0, 1, None, &[101, 102], &[1001, 1002]))
            .unwrap();
        index
            .apply_event(router_event(29, 2, 0, KvCacheEventData::Cleared))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker_a, LowerTierContinuation::from_root(0));
        continuations.insert(worker_b, LowerTierContinuation::from_root(0));

        let hits = index.query_contiguous_hits(&local_hashes(&[101, 102]), &continuations);
        assert_eq!(hits.get(&worker_a), Some(&0));
        assert_eq!(hits.get(&worker_b), Some(&2));
    }

    #[test]
    fn remove_worker_drops_all_ranks() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(41, 0, 0, Some(3000), &[1], &[301]))
            .unwrap();
        index
            .apply_event(store_event(41, 1, 1, Some(4000), &[2], &[401]))
            .unwrap();
        index.remove_worker(41);

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(41, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(3000)),
        );
        continuations.insert(
            WorkerWithDpRank::new(41, 1),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(4000)),
        );

        let hits = index.query_contiguous_hits(&local_hashes(&[1]), &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(41, 0)), Some(&0));
        assert_eq!(hits.get(&WorkerWithDpRank::new(41, 1)), Some(&0));
    }

    #[test]
    fn remove_worker_dp_rank_keeps_other_ranks() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(43, 0, 0, Some(5000), &[1], &[501]))
            .unwrap();
        index
            .apply_event(store_event(43, 1, 1, Some(6000), &[2], &[601]))
            .unwrap();
        index.remove_worker_dp_rank(43, 0);

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(43, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(5000)),
        );
        continuations.insert(
            WorkerWithDpRank::new(43, 1),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(6000)),
        );

        let hits = index.query_contiguous_hits(&local_hashes(&[2]), &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(43, 0)), Some(&0));
        assert_eq!(hits.get(&WorkerWithDpRank::new(43, 1)), Some(&1));
    }

    #[test]
    fn removing_parent_block_keeps_child_continuation_edge() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                31,
                0,
                0,
                Some(1300),
                &[111, 112],
                &[1101, 1102],
            ))
            .unwrap();

        index
            .apply_event(remove_event(
                31,
                1,
                0,
                vec![ExternalSequenceBlockHash(1101)],
            ))
            .unwrap();

        let root_query = local_hashes(&[111, 112]);
        let mut root_continuations = FxHashMap::default();
        root_continuations.insert(
            WorkerWithDpRank::new(31, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(1300)),
        );
        let root_hits = index.query_contiguous_hits(&root_query, &root_continuations);
        assert_eq!(root_hits.get(&WorkerWithDpRank::new(31, 0)), Some(&0));

        let child_query = local_hashes(&[111, 112]);
        let mut child_continuations = FxHashMap::default();
        child_continuations.insert(
            WorkerWithDpRank::new(31, 0),
            LowerTierContinuation::new(1, ExternalSequenceBlockHash(1101)),
        );
        let child_hits = index.query_contiguous_hits(&child_query, &child_continuations);
        assert_eq!(child_hits.get(&WorkerWithDpRank::new(31, 0)), Some(&1));
    }

    #[test]
    fn conflicting_transition_insert_is_ignored() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(37, 0, 0, Some(1400), &[121], &[1201]))
            .unwrap();
        index
            .apply_event(store_event(37, 0, 1, Some(1400), &[121], &[1202]))
            .unwrap();

        let query = local_hashes(&[121]);
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(37, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(1400)),
        );

        let hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(37, 0)), Some(&1));
    }

    #[test]
    fn conflicting_child_hash_mapping_is_ignored() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(47, 0, 0, Some(1500), &[131], &[1301]))
            .unwrap();
        index
            .apply_event(store_event(47, 0, 1, Some(2500), &[231], &[1301]))
            .unwrap();

        let mut original_continuations = FxHashMap::default();
        original_continuations.insert(
            WorkerWithDpRank::new(47, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(1500)),
        );
        let original_hits =
            index.query_contiguous_hits(&local_hashes(&[131]), &original_continuations);
        assert_eq!(original_hits.get(&WorkerWithDpRank::new(47, 0)), Some(&1));

        let mut conflicting_continuations = FxHashMap::default();
        conflicting_continuations.insert(
            WorkerWithDpRank::new(47, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(2500)),
        );
        let conflicting_hits =
            index.query_contiguous_hits(&local_hashes(&[231]), &conflicting_continuations);
        assert_eq!(
            conflicting_hits.get(&WorkerWithDpRank::new(47, 0)),
            Some(&0)
        );
    }
}
