// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact lower-tier KV continuation index.
//!
//! This structure stores worker-local transition edges in the event hash space:
//! `(parent_sequence_hash, local_hash) -> child_sequence_hash`.
//!
//! Unlike the primary KV indexers, this index does not attempt prefix-overlap
//! scoring. Queries continue from a caller-provided per-worker continuation
//! point and count how many consecutive lower-tier blocks are present.

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheStoreData,
    LocalBlockHash, RouterEvent, WorkerWithDpRank,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    parent_hash: Option<ExternalSequenceBlockHash>,
    local_hash: LocalBlockHash,
}

#[derive(Debug, Clone, Copy)]
struct StoredBlock {
    key: TransitionKey,
    refcount: usize,
}

#[derive(Debug, Clone)]
enum TransitionEntry {
    Single(ExternalSequenceBlockHash, usize),
    Multi(FxHashMap<ExternalSequenceBlockHash, usize>),
}

impl TransitionEntry {
    fn new(child_hash: ExternalSequenceBlockHash) -> Self {
        Self::Single(child_hash, 1)
    }

    fn insert(&mut self, child_hash: ExternalSequenceBlockHash) {
        match self {
            Self::Single(existing_hash, refcount) if *existing_hash == child_hash => {
                *refcount += 1;
            }
            Self::Single(existing_hash, existing_refcount) => {
                let mut map = FxHashMap::default();
                map.insert(*existing_hash, *existing_refcount);
                *map.entry(child_hash).or_insert(0) += 1;
                *self = Self::Multi(map);
            }
            Self::Multi(map) => {
                *map.entry(child_hash).or_insert(0) += 1;
            }
        }
    }

    fn remove(&mut self, child_hash: ExternalSequenceBlockHash) -> bool {
        match self {
            Self::Single(existing_hash, refcount) if *existing_hash == child_hash => {
                *refcount = refcount.saturating_sub(1);
                *refcount == 0
            }
            Self::Single(_, _) => false,
            Self::Multi(map) => {
                if let Some(refcount) = map.get_mut(&child_hash) {
                    *refcount = refcount.saturating_sub(1);
                    if *refcount == 0 {
                        map.remove(&child_hash);
                    }
                }

                if map.len() == 1 {
                    let (&only_hash, &only_refcount) = map.iter().next().unwrap();
                    *self = Self::Single(only_hash, only_refcount);
                    false
                } else {
                    map.is_empty()
                }
            }
        }
    }

    fn next_child(&self) -> Option<ExternalSequenceBlockHash> {
        match self {
            Self::Single(child_hash, refcount) if *refcount > 0 => Some(*child_hash),
            Self::Single(_, _) => None,
            Self::Multi(map) if map.len() == 1 => map.keys().next().copied(),
            Self::Multi(_) => None,
        }
    }
}

#[derive(Debug, Default)]
struct WorkerState {
    transitions: FxHashMap<TransitionKey, TransitionEntry>,
    blocks: FxHashMap<ExternalSequenceBlockHash, StoredBlock>,
    total_blocks: usize,
}

impl WorkerState {
    fn apply_store(&mut self, store_data: KvCacheStoreData) {
        let mut parent_hash = store_data.parent_hash;

        for block in store_data.blocks {
            self.insert_block(parent_hash, block.tokens_hash, block.block_hash);
            parent_hash = Some(block.block_hash);
        }
    }

    fn insert_block(
        &mut self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        local_hash: LocalBlockHash,
        child_hash: ExternalSequenceBlockHash,
    ) {
        let key = TransitionKey {
            parent_hash,
            local_hash,
        };

        self.transitions
            .entry(key)
            .and_modify(|entry| entry.insert(child_hash))
            .or_insert_with(|| TransitionEntry::new(child_hash));

        match self.blocks.get_mut(&child_hash) {
            Some(stored) => {
                if stored.key != key {
                    tracing::warn!(
                        child_hash = ?child_hash,
                        existing_parent = ?stored.key.parent_hash,
                        new_parent = ?key.parent_hash,
                        existing_local = ?stored.key.local_hash,
                        new_local = ?key.local_hash,
                        "Conflicting lower-tier child hash mapping for worker"
                    );
                } else {
                    stored.refcount += 1;
                }
            }
            None => {
                self.blocks
                    .insert(child_hash, StoredBlock { key, refcount: 1 });
            }
        }

        self.total_blocks += 1;
    }

    fn apply_remove(
        &mut self,
        remove_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), KvCacheEventError> {
        for child_hash in remove_hashes {
            let Some(stored_block) = self.blocks.get_mut(child_hash) else {
                return Err(KvCacheEventError::BlockNotFound);
            };

            let key = stored_block.key;
            stored_block.refcount = stored_block.refcount.saturating_sub(1);
            let remove_block_entry = stored_block.refcount == 0;

            if remove_block_entry {
                self.blocks.remove(child_hash);
            }

            let remove_transition = match self.transitions.get_mut(&key) {
                Some(entry) => entry.remove(*child_hash),
                None => false,
            };

            if remove_transition {
                self.transitions.remove(&key);
            }

            self.total_blocks = self.total_blocks.saturating_sub(1);
        }

        Ok(())
    }

    fn next_child(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        local_hash: LocalBlockHash,
    ) -> Option<ExternalSequenceBlockHash> {
        self.transitions
            .get(&TransitionKey {
                parent_hash,
                local_hash,
            })
            .and_then(TransitionEntry::next_child)
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

/// Standalone lower-tier continuation index.
pub struct LowerTierIndexer {
    workers: DashMap<WorkerWithDpRank, WorkerState, FxBuildHasher>,
}

impl LowerTierIndexer {
    pub fn new() -> Self {
        Self {
            workers: DashMap::with_hasher(FxBuildHasher),
        }
    }

    fn clear_worker(&self, worker_id: u64) {
        let worker_keys: Vec<_> = self
            .workers
            .iter()
            .filter_map(|entry| (entry.key().worker_id == worker_id).then_some(*entry.key()))
            .collect();

        for worker in worker_keys {
            if let Some(mut worker_state) = self.workers.get_mut(&worker) {
                worker_state.transitions.clear();
                worker_state.blocks.clear();
                worker_state.total_blocks = 0;
            }
        }
    }

    pub fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut worker_state = self.workers.entry(worker).or_default();

        match event.event.data {
            KvCacheEventData::Stored(store_data) => {
                worker_state.apply_store(store_data);
                Ok(())
            }
            KvCacheEventData::Removed(remove_data) => {
                worker_state.apply_remove(&remove_data.block_hashes)
            }
            KvCacheEventData::Cleared => {
                drop(worker_state);
                self.clear_worker(event.worker_id);
                Ok(())
            }
        }
    }

    pub fn remove_worker(&self, worker_id: u64) {
        let worker_keys: Vec<_> = self
            .workers
            .iter()
            .filter_map(|entry| (entry.key().worker_id == worker_id).then_some(*entry.key()))
            .collect();

        for worker in worker_keys {
            self.workers.remove(&worker);
        }
    }

    pub fn remove_worker_dp_rank(&self, worker_id: u64, dp_rank: u32) {
        self.workers
            .remove(&WorkerWithDpRank::new(worker_id, dp_rank));
    }

    pub fn query_contiguous_hits(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &FxHashMap<WorkerWithDpRank, LowerTierContinuation>,
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        let mut results = FxHashMap::default();

        for (worker, continuation) in continuations {
            let hits = match self.workers.get(worker) {
                Some(worker_state) => {
                    contiguous_hits_for_worker(&worker_state, local_hashes, *continuation)
                }
                None => 0,
            };
            results.insert(*worker, hits);
        }

        results
    }
}

impl Default for LowerTierIndexer {
    fn default() -> Self {
        Self::new()
    }
}

fn contiguous_hits_for_worker(
    worker_state: &WorkerState,
    local_hashes: &[LocalBlockHash],
    continuation: LowerTierContinuation,
) -> usize {
    if continuation.start_pos >= local_hashes.len() {
        return 0;
    }

    let mut count = 0usize;
    let mut position = continuation.start_pos;
    let mut parent_hash = continuation.last_matched_hash;

    while position < local_hashes.len() {
        let Some(child_hash) = worker_state.next_child(parent_hash, local_hashes[position]) else {
            break;
        };
        count += 1;
        parent_hash = Some(child_hash);
        position += 1;
    }

    count
}

#[cfg(test)]
mod tests {
    use super::{LowerTierContinuation, LowerTierIndexer};
    use rustc_hash::FxHashMap;

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

    #[test]
    fn root_query_uses_none_parent_transition() {
        let index = LowerTierIndexer::new();
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
    fn missing_parent_tail_queries_exactly_from_last_matched_hash() {
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
    fn duplicate_store_is_refcounted_for_exact_remove() {
        let index = LowerTierIndexer::new();
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
            Some(&1)
        );

        index
            .apply_event(remove_event(13, 2, 0, vec![ExternalSequenceBlockHash(601)]))
            .unwrap();
        let after_two_removes = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(
            after_two_removes.get(&WorkerWithDpRank::new(13, 0)),
            Some(&0)
        );
    }

    #[test]
    fn remove_stops_contiguous_walk_at_missing_edge() {
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
    fn remove_worker_drops_all_ranks() {
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
        let index = LowerTierIndexer::new();
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
    fn ambiguous_transition_returns_zero_until_conflict_removed() {
        let index = LowerTierIndexer::new();
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

        let ambiguous_hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(ambiguous_hits.get(&WorkerWithDpRank::new(37, 0)), Some(&0));

        index
            .apply_event(remove_event(
                37,
                2,
                0,
                vec![ExternalSequenceBlockHash(1202)],
            ))
            .unwrap();

        let resolved_hits = index.query_contiguous_hits(&query, &continuations);
        assert_eq!(resolved_hits.get(&WorkerWithDpRank::new(37, 0)), Some(&1));
    }
}
