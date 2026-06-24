// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
#[cfg(test)]
use rustc_hash::FxHashSet;

use super::*;

struct PromptHashSequence<'a>(&'a [SequenceHash]);

impl HashSequence for PromptHashSequence<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, index: usize) -> LocalBlockHash {
        LocalBlockHash(self.0[index])
    }
}

/// Opaque per-worker reverse lookup for prompt membership.
#[derive(Default)]
pub(crate) struct PromptWorkerLookup {
    inner: WorkerLookup,
}

/// Prompt-membership projection backed by [`ConcurrentRadixTreeCompressed`].
///
/// Prompt membership uses a single `SequenceHash` identity. The adapter maps it
/// to both CRTC hash dimensions: `LocalBlockHash` for traversal and
/// `ExternalSequenceBlockHash` for parent/removal lookup.
pub(crate) struct PromptMembershipIndex {
    index: ConcurrentRadixTreeCompressed,
    next_event_id: AtomicU64,
}

impl Default for PromptMembershipIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptMembershipIndex {
    pub(crate) fn new() -> Self {
        Self {
            index: ConcurrentRadixTreeCompressed::new(),
            next_event_id: AtomicU64::new(0),
        }
    }

    pub(crate) fn store_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<PromptWorkerLookup>>,
        parent: Option<SequenceHash>,
        hashes: &[SequenceHash],
    ) {
        if hashes.is_empty() {
            return;
        }

        let blocks = hashes
            .iter()
            .map(|&hash| KvCacheStoredBlockData {
                tokens_hash: LocalBlockHash(hash),
                block_hash: ExternalSequenceBlockHash(hash),
                mm_extra_info: None,
            })
            .collect();

        let event = self.router_event(
            worker,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks,
            }),
        );
        self.apply_event_with_worker_lookup(worker, lookup, event);
    }

    pub(crate) fn remove_chain(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<PromptWorkerLookup>>,
        hashes: &[SequenceHash],
    ) {
        if hashes.is_empty() {
            return;
        }

        let event = self.router_event(
            worker,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .map(|&hash| ExternalSequenceBlockHash(hash))
                    .collect(),
            }),
        );
        self.apply_event_with_worker_lookup(worker, lookup, event);
    }

    pub(crate) fn remove_worker(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<PromptWorkerLookup>>,
    ) {
        let mut prompt_lookup = lookup.write();
        let mut direct_lookup = self.take_worker_lookup(worker, &mut prompt_lookup);

        self.repair_worker_lookup(&mut direct_lookup, worker);
        self.index
            .remove_worker_dp_rank(&mut direct_lookup, worker.worker_id, worker.dp_rank);

        prompt_lookup.inner = direct_lookup.remove(&worker).unwrap_or_default();
    }

    pub(crate) fn compute_overlap_depths(
        &self,
        query: Option<&[SequenceHash]>,
    ) -> FxHashMap<WorkerWithDpRank, usize> {
        let Some(query) = query else {
            return FxHashMap::default();
        };
        if query.is_empty() {
            return FxHashMap::default();
        }

        let next_child = self.index.root.child_snapshot(LocalBlockHash(query[0]));
        let details =
            self.index
                .find_details_from_seq(next_child, PromptHashSequence(query), false);
        details
            .overlap_scores
            .scores
            .into_iter()
            .map(|(worker, depth)| (worker, depth as usize))
            .collect()
    }

    pub(crate) fn maybe_cleanup(&self) {
        if !self.index.try_schedule_cleanup() {
            return;
        }
        self.index.run_cleanup_task();
    }

    #[cfg(test)]
    pub(crate) fn worker_hashes(&self) -> FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>> {
        let mut worker_hashes = FxHashMap::default();
        for event in self.index.dump_tree_as_events() {
            let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
            if let KvCacheEventData::Stored(store) = event.event.data {
                worker_hashes
                    .entry(worker)
                    .or_insert_with(FxHashSet::default)
                    .extend(store.blocks.into_iter().map(|block| block.block_hash.0));
            }
        }
        worker_hashes
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn is_empty(&self) -> bool {
        self.index.dump_tree_as_events().is_empty()
    }

    fn router_event(&self, worker: WorkerWithDpRank, data: KvCacheEventData) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: self.next_event_id.fetch_add(1, Ordering::Relaxed),
                data,
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn apply_event_with_worker_lookup(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<RwLock<PromptWorkerLookup>>,
        event: RouterEvent,
    ) {
        let mut prompt_lookup = lookup.write();
        let mut direct_lookup = self.take_worker_lookup(worker, &mut prompt_lookup);
        let result = self.index.apply_event(&mut direct_lookup, event, None);
        if let Err(error) = result {
            tracing::warn!(?worker, ?error, "failed to apply prompt membership event");
        }
        prompt_lookup.inner = direct_lookup.remove(&worker).unwrap_or_default();
    }

    fn take_worker_lookup(
        &self,
        worker: WorkerWithDpRank,
        prompt_lookup: &mut PromptWorkerLookup,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLookup> {
        let mut direct_lookup = FxHashMap::default();
        direct_lookup.insert(worker, std::mem::take(&mut prompt_lookup.inner));
        direct_lookup
    }

    fn repair_worker_lookup(
        &self,
        direct_lookup: &mut FxHashMap<WorkerWithDpRank, WorkerLookup>,
        worker: WorkerWithDpRank,
    ) {
        let hashes: Vec<_> = direct_lookup
            .get(&worker)
            .map(|worker_lookup| worker_lookup.keys().copied().collect())
            .unwrap_or_default();

        for hash in hashes {
            let _ = self.index.resolve_lookup(
                direct_lookup,
                worker,
                hash,
                LookupRepairDirection::TowardTail,
            );
        }
    }
}

#[cfg(any(test, feature = "bench"))]
pub(crate) fn lookup_live_hashes(lookup: &Arc<RwLock<PromptWorkerLookup>>) -> Vec<SequenceHash> {
    let lookup = lookup.read();
    lookup
        .inner
        .iter()
        .filter_map(|(&hash, node)| node.dump_snapshot().has_any_workers.then_some(hash.0))
        .collect()
}
