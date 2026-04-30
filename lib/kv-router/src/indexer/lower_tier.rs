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

use std::hash::BuildHasher;
use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::{KvIndexerMetrics, SyncIndexer, WorkerTask};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheEventError, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, OverlapScores, RouterEvent, WorkerWithDpRank,
};

type WorkerSet = FxHashSet<WorkerWithDpRank>;
type FrontierBuckets = FxHashMap<Option<ExternalSequenceBlockHash>, WorkerSet>;
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

            // If this worker already has a different parent/local for the same
            // block_hash, or if the shared edge is owned by a conflicting
            // child_hash, stop the walk: any further blocks in this chain would
            // hang off an edge this index never accepted for the worker.
            if worker_map
                .get(&block.block_hash)
                .is_some_and(|existing_key| *existing_key != key)
            {
                break;
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

            if !inserted {
                break;
            }

            worker_map.insert(block.block_hash, key);
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

    /// Reconstruct store events from the per-worker block index. Each block
    /// becomes a single-block `Stored` event with the correct parent hash,
    /// suitable for replaying into a fresh indexer to recreate the same state.
    fn dump_events(worker_blocks: &WorkerBlockIndex) -> Vec<RouterEvent> {
        let mut events = Vec::new();
        let mut event_id = 0u64;

        for (worker, block_map) in worker_blocks {
            for (block_hash, key) in block_map {
                events.push(RouterEvent::new(
                    worker.worker_id,
                    KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: key.parent_hash,
                            start_position: None,
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash: *block_hash,
                                tokens_hash: key.local_hash,
                                mm_extra_info: None,
                            }],
                        }),
                        dp_rank: worker.dp_rank,
                    },
                ));
                event_id += 1;
            }
        }

        events
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

    /// For each worker, counts how many contiguous lower-tier blocks match
    /// starting from the worker's continuation point, and returns the updated
    /// continuation state.
    ///
    /// Workers may start at different positions in `local_hashes` (each has its
    /// own `LowerTierContinuation`). The algorithm groups workers that share a
    /// start position into "breakpoints", sorts them, and advances each group
    /// forward through the hash sequence one position at a time. When a group
    /// reaches the next breakpoint it pauses so the two groups can be merged
    /// (workers that converge onto the same edge path are walked together).
    pub fn query_match_details<S>(
        &self,
        local_hashes: &[LocalBlockHash],
        continuations: &std::collections::HashMap<WorkerWithDpRank, LowerTierContinuation, S>,
    ) -> LowerTierMatchDetails
    where
        S: BuildHasher,
    {
        // Build the sorted breakpoint list. Each entry is a position in the
        // hash sequence and a set of (parent_hash -> workers) groups that start
        // walking from that position. The set of positions is fixed — the walk
        // never creates new breakpoints, it only merges overflow workers into
        // the next existing one.
        let mut breakpoints: Vec<(usize, FrontierBuckets)> = Vec::new();
        {
            let mut pos_index: FxHashMap<usize, usize> = FxHashMap::default();
            for (worker, continuation) in continuations {
                let idx = match pos_index.get(&continuation.start_pos) {
                    Some(&idx) => idx,
                    None => {
                        let idx = breakpoints.len();
                        pos_index.insert(continuation.start_pos, idx);
                        breakpoints.push((continuation.start_pos, FrontierBuckets::default()));
                        idx
                    }
                };
                breakpoints[idx]
                    .1
                    .entry(continuation.last_matched_hash)
                    .or_default()
                    .insert(*worker);
            }
            breakpoints.sort_unstable_by_key(|(pos, _)| *pos);
        }

        let mut final_states = FinalStates::default();

        // Process breakpoints front-to-back. Each group walks forward until it
        // hits the next breakpoint or runs out of matching edges. Workers that
        // survive to the next breakpoint are collected as "overflow" and merged
        // into that breakpoint's buckets before it gets processed.
        for idx in 0..breakpoints.len() {
            let pos = breakpoints[idx].0;
            let states = std::mem::take(&mut breakpoints[idx].1);
            let next_breakpoint = breakpoints
                .get(idx + 1)
                .map(|(p, _)| *p)
                .unwrap_or(local_hashes.len())
                .min(local_hashes.len());

            let mut overflow = FrontierBuckets::default();

            for (parent_hash, workers) in states {
                advance_state_to_breakpoint(
                    self,
                    local_hashes,
                    pos,
                    parent_hash,
                    workers,
                    next_breakpoint,
                    &mut overflow,
                    &mut final_states,
                );
            }

            if !overflow.is_empty()
                && let Some((_, next_buckets)) = breakpoints.get_mut(idx + 1)
            {
                for (hash, workers) in overflow {
                    next_buckets.entry(hash).or_default().extend(workers);
                }
            }
        }

        // Convert final_states into the result. Workers that never appeared in
        // final_states (e.g. empty sequence) keep their original continuation.
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
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        _metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
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
                    let _ = sender.send(Ok(Self::dump_events(&worker_blocks)));
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::CleanupStaleChildren => {}
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

/// Walks a group of workers sharing the same `(start_pos, parent_hash)` forward
/// through `local_hashes`, one position at a time, until `next_breakpoint`.
///
/// At each position the function looks up the edge `(cur_hash, local_hash) ->
/// child_hash` and partitions workers into those that own the edge (they
/// continue) and those that don't (they are finalized at this position).
///
/// Workers that survive all the way to `next_breakpoint` are placed into
/// `overflow` so the caller can merge them into the next breakpoint's groups.
/// Workers that reach the end of `local_hashes` are finalized instead.
#[allow(clippy::too_many_arguments)]
fn advance_state_to_breakpoint(
    index: &LowerTierIndexer,
    local_hashes: &[LocalBlockHash],
    start_pos: usize,
    start_hash: Option<ExternalSequenceBlockHash>,
    workers: WorkerSet,
    next_breakpoint: usize,
    overflow: &mut FrontierBuckets,
    final_states: &mut FinalStates,
) {
    let mut cur_pos = start_pos;
    let mut cur_hash = start_hash;
    let mut active = workers;

    // When only one worker is active we can skip all set bookkeeping and just
    // do a straight edge-lookup loop.
    if active.len() == 1 {
        let worker = active.into_iter().next().unwrap();
        advance_single_worker(
            index,
            local_hashes,
            worker,
            &mut cur_pos,
            &mut cur_hash,
            next_breakpoint,
            overflow,
            final_states,
        );
        return;
    }

    // Reusable scratch buffer for partitioning workers each iteration, avoids
    // allocating new HashSets on every step.
    let mut scratch = WorkerSet::default();

    while cur_pos < next_breakpoint && !active.is_empty() {
        // Look up the edge for the current (parent_hash, local_hash) pair.
        // If no edge exists, no worker can continue — finalize everyone.
        let Some(edge) = index.edges.get(&TransitionKey {
            parent_hash: cur_hash,
            local_hash: local_hashes[cur_pos],
        }) else {
            finalize_workers(final_states, active.drain(), cur_pos, cur_hash);
            break;
        };

        // Partition active workers into matched (own the edge) and unmatched.
        // For single-owner edges we can check membership in O(1) instead of
        // iterating all active workers. For multi-owner edges we iterate
        // whichever side is smaller.
        match edge.value() {
            EdgeOwnersEntry::Single { owner, .. } => {
                if active.remove(owner) {
                    finalize_workers(final_states, active.drain(), cur_pos, cur_hash);
                    active.insert(*owner);
                } else {
                    finalize_workers(final_states, active.drain(), cur_pos, cur_hash);
                    break;
                }
            }
            EdgeOwnersEntry::Multi { owners, .. } => {
                if owners.len() <= active.len() {
                    scratch.clear();
                    for owner in owners {
                        if active.remove(owner) {
                            scratch.insert(*owner);
                        }
                    }
                    finalize_workers(final_states, active.drain(), cur_pos, cur_hash);
                    std::mem::swap(&mut active, &mut scratch);
                } else {
                    scratch.clear();
                    for worker in active.drain() {
                        if owners.contains(&worker) {
                            scratch.insert(worker);
                        } else {
                            final_states.insert(worker, (cur_pos, cur_hash));
                        }
                    }
                    std::mem::swap(&mut active, &mut scratch);
                }

                if active.is_empty() {
                    break;
                }
            }
        }

        cur_hash = Some(edge.child_hash());
        cur_pos += 1;

        // If we're down to one worker, switch to the scalar loop for the
        // remaining positions to avoid set overhead.
        if active.len() == 1 {
            let worker = active.into_iter().next().unwrap();
            advance_single_worker(
                index,
                local_hashes,
                worker,
                &mut cur_pos,
                &mut cur_hash,
                next_breakpoint,
                overflow,
                final_states,
            );
            return;
        }
    }

    if active.is_empty() {
        return;
    }

    // Workers that reached the breakpoint without dropping off. If we're past
    // the end of the sequence they're finalized; otherwise they overflow into
    // the next breakpoint for continued walking.
    if cur_pos >= local_hashes.len() {
        finalize_workers(final_states, active, cur_pos, cur_hash);
    } else {
        overflow.entry(cur_hash).or_default().extend(active);
    }
}

/// Simplified walk for exactly one worker. Just does sequential edge lookups
/// without any set operations — either the worker owns each edge and continues,
/// or it stops.
#[allow(clippy::too_many_arguments)]
fn advance_single_worker(
    index: &LowerTierIndexer,
    local_hashes: &[LocalBlockHash],
    worker: WorkerWithDpRank,
    cur_pos: &mut usize,
    cur_hash: &mut Option<ExternalSequenceBlockHash>,
    next_breakpoint: usize,
    overflow: &mut FrontierBuckets,
    final_states: &mut FinalStates,
) {
    while *cur_pos < next_breakpoint {
        let Some(edge) = index.edges.get(&TransitionKey {
            parent_hash: *cur_hash,
            local_hash: local_hashes[*cur_pos],
        }) else {
            final_states.insert(worker, (*cur_pos, *cur_hash));
            return;
        };

        if !edge.contains(&worker) {
            final_states.insert(worker, (*cur_pos, *cur_hash));
            return;
        }

        *cur_hash = Some(edge.child_hash());
        *cur_pos += 1;
    }

    if *cur_pos >= local_hashes.len() {
        final_states.insert(worker, (*cur_pos, *cur_hash));
    } else {
        overflow.entry(*cur_hash).or_default().insert(worker);
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
                start_position: None,
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

        fn dump_events(&self) -> Vec<crate::protocols::RouterEvent> {
            LowerTierIndexer::dump_events(&self.worker_blocks)
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

    // --- Tests targeting optimization edge cases ---

    /// Single-worker fast path: exercises the scalar loop that skips set
    /// operations when only one worker is in the continuation map.
    #[test]
    fn single_worker_fast_path_full_match() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                50,
                0,
                0,
                None,
                &[1, 2, 3, 4, 5],
                &[101, 102, 103, 104, 105],
            ))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(50, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2, 3, 4, 5]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(50, 0)), Some(&5));
        assert_eq!(
            details
                .next_continuations
                .get(&WorkerWithDpRank::new(50, 0)),
            Some(&LowerTierContinuation::new(
                5,
                ExternalSequenceBlockHash(105),
            )),
        );
    }

    /// Single-worker fast path where the worker doesn't own the edge.
    #[test]
    fn single_worker_fast_path_no_match() {
        let mut index = TestLowerTierIndex::new();
        // Worker 50 owns the chain, but we query with worker 51.
        index
            .apply_event(store_event(50, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(51, 0),
            LowerTierContinuation::from_root(0),
        );

        let hits = index.query_contiguous_hits(&local_hashes(&[1, 2]), &continuations);
        assert_eq!(hits.get(&WorkerWithDpRank::new(51, 0)), Some(&0));
    }

    /// Single-worker partial match: worker owns the first two edges but the
    /// third edge doesn't exist, testing early termination in the scalar loop.
    #[test]
    fn single_worker_fast_path_partial_match() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(52, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(52, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2, 3]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(52, 0)), Some(&2));
        assert_eq!(
            details
                .next_continuations
                .get(&WorkerWithDpRank::new(52, 0)),
            Some(&LowerTierContinuation::new(
                2,
                ExternalSequenceBlockHash(102),
            )),
        );
    }

    /// Exercises the Single-edge flip: two workers query, but the edge is
    /// owned by only one of them (Single variant). The non-owner should be
    /// finalized immediately.
    #[test]
    fn single_edge_owner_splits_active_set() {
        let mut index = TestLowerTierIndex::new();
        // Only worker 60 owns this chain.
        index
            .apply_event(store_event(60, 0, 0, None, &[1, 2, 3], &[101, 102, 103]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(60, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(61, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2, 3]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(60, 0)), Some(&3));
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(61, 0)), Some(&0));
    }

    /// Multiple workers share an edge (Multi variant), but only a subset are
    /// active. Tests the min-side iteration path.
    #[test]
    fn multi_edge_subset_of_owners_active() {
        let mut index = TestLowerTierIndex::new();
        // Workers 70, 71, 72 all own the same chain.
        index
            .apply_event(store_event(70, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(71, 0, 1, None, &[1, 2], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(72, 0, 2, None, &[1, 2], &[101, 102]))
            .unwrap();

        // Query with only workers 70 and 71 (active < owners wouldn't apply
        // here since counts are close, but the Multi branch is exercised).
        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(70, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(71, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(70, 0)), Some(&2));
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(71, 0)), Some(&2));
    }

    /// Multi-worker walk where one worker drops off mid-sequence, causing the
    /// set to shrink to 1 and triggering the mid-loop scalar fast path.
    #[test]
    fn multi_to_single_worker_transition_mid_walk() {
        let mut index = TestLowerTierIndex::new();
        // Worker 80 owns [1,2,3,4], worker 81 owns only [1,2].
        index
            .apply_event(store_event(
                80,
                0,
                0,
                None,
                &[1, 2, 3, 4],
                &[101, 102, 103, 104],
            ))
            .unwrap();
        index
            .apply_event(store_event(81, 0, 1, None, &[1, 2], &[101, 102]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(80, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(81, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2, 3, 4]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(80, 0)), Some(&4));
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(81, 0)), Some(&2));
        assert_eq!(
            details
                .next_continuations
                .get(&WorkerWithDpRank::new(80, 0)),
            Some(&LowerTierContinuation::new(
                4,
                ExternalSequenceBlockHash(104),
            )),
        );
        assert_eq!(
            details
                .next_continuations
                .get(&WorkerWithDpRank::new(81, 0)),
            Some(&LowerTierContinuation::new(
                2,
                ExternalSequenceBlockHash(102),
            )),
        );
    }

    /// All active workers drop off at the same position because none of them
    /// own the edge (Single variant, owner not in active set).
    #[test]
    fn single_edge_no_active_worker_owns_it() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(90, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(91, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(92, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(91, 0)), Some(&0));
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(92, 0)), Some(&0));
    }

    /// Single-worker fast path hitting the breakpoint boundary — worker starts
    /// at pos 0 but a second worker's start_pos creates a breakpoint at pos 2.
    /// The first worker should stop at the breakpoint, then be re-merged in the
    /// frontier and continue.
    #[test]
    fn single_worker_stops_at_breakpoint_then_continues() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                95,
                0,
                0,
                None,
                &[1, 2, 3, 4],
                &[101, 102, 103, 104],
            ))
            .unwrap();
        index
            .apply_event(store_event(96, 0, 1, Some(102), &[3, 4], &[103, 104]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(95, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(96, 0),
            LowerTierContinuation::new(2, ExternalSequenceBlockHash(102)),
        );

        let details = index.query_match_details(&local_hashes(&[1, 2, 3, 4]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(95, 0)), Some(&4));
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(96, 0)), Some(&2));
    }

    /// Exercises the Multi-edge path where the active set is larger than the
    /// owner set (iterate owners side).
    #[test]
    fn multi_edge_fewer_owners_than_active_workers() {
        let mut index = TestLowerTierIndex::new();
        // Edge owned by workers 100 and 101 (Multi with 2 owners).
        index
            .apply_event(store_event(100, 0, 0, None, &[1], &[101]))
            .unwrap();
        index
            .apply_event(store_event(101, 0, 1, None, &[1], &[101]))
            .unwrap();

        // Query with 4 workers — only 2 own the edge.
        let mut continuations = FxHashMap::default();
        for id in 100..104 {
            continuations.insert(
                WorkerWithDpRank::new(id, 0),
                LowerTierContinuation::from_root(0),
            );
        }

        let details = index.query_match_details(&local_hashes(&[1]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(100, 0)), Some(&1),);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(101, 0)), Some(&1),);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(102, 0)), Some(&0),);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(103, 0)), Some(&0),);
    }

    /// Empty continuations map — should return empty results without panicking.
    #[test]
    fn empty_continuations_returns_empty_results() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(110, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();

        let continuations: FxHashMap<WorkerWithDpRank, LowerTierContinuation> =
            FxHashMap::default();

        let details = index.query_match_details(&local_hashes(&[1, 2]), &continuations);
        assert!(details.hits.is_empty());
        assert!(details.next_continuations.is_empty());
    }

    /// Empty sequence — every worker should get 0 hits.
    #[test]
    fn empty_sequence_returns_zero_hits() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(111, 0, 0, None, &[1], &[101]))
            .unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(111, 0),
            LowerTierContinuation::from_root(0),
        );

        let details = index.query_match_details(&local_hashes(&[]), &continuations);
        assert_eq!(details.hits.get(&WorkerWithDpRank::new(111, 0)), Some(&0));
    }

    // --- dump_events tests ---

    /// Helper: replay dumped events into a fresh indexer and return it.
    fn replay_dump(events: Vec<crate::protocols::RouterEvent>) -> TestLowerTierIndex {
        let mut fresh = TestLowerTierIndex::new();
        for event in events {
            fresh.apply_event(event).unwrap();
        }
        fresh
    }

    #[test]
    fn dump_empty_indexer_returns_no_events() {
        let index = TestLowerTierIndex::new();
        assert!(index.dump_events().is_empty());
    }

    #[test]
    fn dump_round_trip_single_chain() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(7, 0, 0, None, &[11, 12, 13], &[101, 102, 103]))
            .unwrap();

        let events = index.dump_events();
        assert_eq!(events.len(), 3);

        let restored = replay_dump(events);

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(7, 0),
            LowerTierContinuation::from_root(0),
        );

        let original = index.query_contiguous_hits(&local_hashes(&[11, 12, 13]), &continuations);
        let replayed = restored.query_contiguous_hits(&local_hashes(&[11, 12, 13]), &continuations);
        assert_eq!(original, replayed);
        assert_eq!(replayed.get(&WorkerWithDpRank::new(7, 0)), Some(&3));
    }

    #[test]
    fn dump_round_trip_multiple_workers() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(1, 0, 0, None, &[11, 12], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(2, 0, 1, Some(500), &[21, 22], &[201, 202]))
            .unwrap();

        let events = index.dump_events();
        assert_eq!(events.len(), 4);

        let restored = replay_dump(events);

        // Worker 1: root chain
        let mut c1 = FxHashMap::default();
        c1.insert(
            WorkerWithDpRank::new(1, 0),
            LowerTierContinuation::from_root(0),
        );
        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[11, 12]), &c1),
            restored.query_contiguous_hits(&local_hashes(&[11, 12]), &c1),
        );

        // Worker 2: non-root chain
        let mut c2 = FxHashMap::default();
        c2.insert(
            WorkerWithDpRank::new(2, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(500)),
        );
        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[21, 22]), &c2),
            restored.query_contiguous_hits(&local_hashes(&[21, 22]), &c2),
        );
    }

    #[test]
    fn dump_round_trip_shared_edges() {
        let mut index = TestLowerTierIndex::new();
        // Two workers own the same chain.
        index
            .apply_event(store_event(1, 0, 0, None, &[11, 12], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(2, 0, 1, None, &[11, 12], &[101, 102]))
            .unwrap();

        let events = index.dump_events();
        // 2 blocks * 2 workers = 4 events (each worker's blocks are dumped
        // independently even if they share the same underlying edges).
        assert_eq!(events.len(), 4);

        let restored = replay_dump(events);

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(1, 0),
            LowerTierContinuation::from_root(0),
        );
        continuations.insert(
            WorkerWithDpRank::new(2, 0),
            LowerTierContinuation::from_root(0),
        );

        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[11, 12]), &continuations),
            restored.query_contiguous_hits(&local_hashes(&[11, 12]), &continuations),
        );
    }

    #[test]
    fn dump_after_removal_excludes_removed_blocks() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(
                5,
                0,
                0,
                Some(800),
                &[31, 32, 33],
                &[301, 302, 303],
            ))
            .unwrap();

        // Remove the middle block.
        index
            .apply_event(remove_event(5, 1, 0, vec![ExternalSequenceBlockHash(302)]))
            .unwrap();

        let events = index.dump_events();
        // Only 2 blocks remain (301 and 303).
        assert_eq!(events.len(), 2);

        let restored = replay_dump(events);

        let mut continuations = FxHashMap::default();
        continuations.insert(
            WorkerWithDpRank::new(5, 0),
            LowerTierContinuation::new(0, ExternalSequenceBlockHash(800)),
        );

        // Original and restored should give the same result: only 1 hit
        // (block 301 matches, 302 is gone so the chain breaks).
        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[31, 32, 33]), &continuations),
            restored.query_contiguous_hits(&local_hashes(&[31, 32, 33]), &continuations),
        );
    }

    #[test]
    fn dump_round_trip_multiple_dp_ranks() {
        let mut index = TestLowerTierIndex::new();
        index
            .apply_event(store_event(10, 0, 0, None, &[1, 2], &[101, 102]))
            .unwrap();
        index
            .apply_event(store_event(10, 1, 1, None, &[3, 4], &[301, 302]))
            .unwrap();

        let events = index.dump_events();
        assert_eq!(events.len(), 4);

        let restored = replay_dump(events);

        // Verify dp_rank=0 chain
        let mut c0 = FxHashMap::default();
        c0.insert(
            WorkerWithDpRank::new(10, 0),
            LowerTierContinuation::from_root(0),
        );
        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[1, 2]), &c0),
            restored.query_contiguous_hits(&local_hashes(&[1, 2]), &c0),
        );

        // Verify dp_rank=1 chain
        let mut c1 = FxHashMap::default();
        c1.insert(
            WorkerWithDpRank::new(10, 1),
            LowerTierContinuation::from_root(0),
        );
        assert_eq!(
            index.query_contiguous_hits(&local_hashes(&[3, 4]), &c1),
            restored.query_contiguous_hits(&local_hashes(&[3, 4]), &c1),
        );
    }

    #[tokio::test]
    async fn thread_pool_dump_events_round_trip() {
        let index = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, 1);
        let worker = WorkerWithDpRank::new(7, 0);

        index
            .apply_event(store_event(7, 0, 0, None, &[11, 12, 13], &[101, 102, 103]))
            .await;

        let events = index.dump_events().await.unwrap();
        assert_eq!(events.len(), 3);

        // Replay into a fresh ThreadPoolIndexer.
        let restored = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, 1);
        for event in events {
            restored.apply_event(event).await;
        }
        let _ = restored.dump_events().await.unwrap();

        let mut continuations = FxHashMap::default();
        continuations.insert(worker, LowerTierContinuation::from_root(0));

        let original = index
            .backend()
            .query_contiguous_hits(&local_hashes(&[11, 12, 13]), &continuations);
        let replayed = restored
            .backend()
            .query_contiguous_hits(&local_hashes(&[11, 12, 13]), &continuations);
        assert_eq!(original, replayed);
        assert_eq!(replayed.get(&worker), Some(&3));
    }
}
