// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use indexmap::IndexMap;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap};
use seqlock::SeqLock;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::time::Instant;

use super::PrefillTokenDeltas;
use super::prefill_tracker::{PrefillLoadSnapshot, PrefillTimeLoadError};
use super::topology::WorkerTopologyChange;
use super::unified_prompt_tracker::UnifiedPromptTracker;
use crate::protocols::WorkerWithDpRank;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorkerLoadProjection {
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    pub additional_active_blocks: usize,
}

impl WorkerLoadProjection {
    pub fn potential_decode_blocks(self) -> usize {
        self.active_decode_blocks + self.additional_active_blocks
    }
}

pub type PotentialLoadMaps = (
    FxHashMap<WorkerWithDpRank, usize>,
    FxHashMap<WorkerWithDpRank, usize>,
    Option<FxHashMap<WorkerWithDpRank, usize>>,
);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorkerLoadSnapshot {
    pub(super) active_blocks: usize,
    pub(super) active_requests: usize,
    pub(super) prefill: PrefillLoadSnapshot,
}

impl WorkerLoadSnapshot {
    pub(super) fn active_tokens(&self, decay_now: Instant) -> usize {
        self.prefill.active_tokens_at(decay_now)
    }

    pub(super) fn modeled_remaining_prefill_time_ms(
        &self,
        now: Instant,
    ) -> Result<u64, PrefillTimeLoadError> {
        self.prefill.modeled_remaining_prefill_time_ms_at(now)
    }
}

#[derive(Debug)]
struct WorkerLoadSlot {
    load: SeqLock<WorkerLoadSnapshot>,
}

impl WorkerLoadSlot {
    fn new(load: WorkerLoadSnapshot) -> Self {
        Self {
            load: SeqLock::new(load),
        }
    }

    fn snapshot(&self) -> WorkerLoadSnapshot {
        self.load.read()
    }

    fn replace(&self, load: WorkerLoadSnapshot) {
        *self.load.lock_write() = load;
    }
}

#[derive(Debug)]
struct WorkerLoadTable {
    entries: IndexMap<WorkerWithDpRank, WorkerLoadSlot, FxBuildHasher>,
}

impl Default for WorkerLoadTable {
    fn default() -> Self {
        Self {
            entries: IndexMap::with_hasher(FxBuildHasher),
        }
    }
}

impl WorkerLoadTable {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn iter(&self) -> impl Iterator<Item = (WorkerWithDpRank, WorkerLoadSnapshot)> + '_ {
        self.entries
            .iter()
            .map(|(&worker, slot)| (worker, slot.snapshot()))
    }

    fn ensure_worker(&mut self, worker: WorkerWithDpRank) {
        self.entries
            .entry(worker)
            .or_insert_with(|| WorkerLoadSlot::new(WorkerLoadSnapshot::default()));
    }

    fn update(&self, worker: WorkerWithDpRank, load: WorkerLoadSnapshot) -> bool {
        let Some(slot) = self.entries.get(&worker) else {
            return false;
        };
        slot.replace(load);
        true
    }

    fn upsert(&mut self, worker: WorkerWithDpRank, load: WorkerLoadSnapshot) {
        if let Some(slot) = self.entries.get(&worker) {
            slot.replace(load);
        } else {
            self.entries.insert(worker, WorkerLoadSlot::new(load));
        }
    }

    fn remove(&mut self, worker: WorkerWithDpRank) {
        self.entries.swap_remove(&worker);
    }
}

/// Load projection paired with the authoritative global prompt trie.
pub(super) struct PromptRegistry {
    prompts: Arc<UnifiedPromptTracker>,
    loads: RwLock<WorkerLoadTable>,
    #[cfg(test)]
    cleanup_attempts: AtomicUsize,
}

impl PromptRegistry {
    pub(super) fn new(
        workers: impl IntoIterator<Item = WorkerWithDpRank>,
        prompts: Arc<UnifiedPromptTracker>,
    ) -> Self {
        let registry = Self {
            prompts,
            loads: RwLock::new(WorkerLoadTable::default()),
            #[cfg(test)]
            cleanup_attempts: AtomicUsize::new(0),
        };
        let mut loads = registry.loads.write();
        for worker in workers {
            registry.prompts.ensure_worker(worker);
            loads.ensure_worker(worker);
        }
        drop(loads);
        registry
    }

    pub(super) fn replace_worker_load_state(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
    ) {
        if self.loads.read().update(worker, load) {
            return;
        }
        self.loads.write().upsert(worker, load);
    }

    pub(super) fn maybe_cleanup(&self) {
        #[cfg(test)]
        self.cleanup_attempts.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn cleanup_attempts(&self) -> usize {
        self.cleanup_attempts.load(Ordering::Relaxed)
    }

    pub(super) fn apply_topology_change_without_cleanup(&self, change: &WorkerTopologyChange) {
        for removed in &change.removed {
            self.prompts.remove_worker(removed.worker);
            self.loads.write().remove(removed.worker);
        }
        for &worker in &change.added {
            self.prompts.ensure_worker(worker);
            self.loads.write().ensure_worker(worker);
        }
    }

    fn project_loads<const INCLUDE_ACTIVE_REQUESTS: bool>(
        &self,
        query_len: usize,
        matched_depth: &FxHashMap<WorkerWithDpRank, usize>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> PotentialLoadMaps {
        let loads = self.loads.read();
        let mut potential_blocks = FxHashMap::with_capacity_and_hasher(loads.len(), FxBuildHasher);
        let mut potential_tokens = FxHashMap::with_capacity_and_hasher(loads.len(), FxBuildHasher);
        let mut active_requests = INCLUDE_ACTIVE_REQUESTS
            .then(|| FxHashMap::with_capacity_and_hasher(loads.len(), FxBuildHasher));
        for (worker, load) in loads.iter() {
            let overlap_depth = matched_depth.get(&worker).copied().unwrap_or(0);
            potential_blocks.insert(
                worker,
                load.active_blocks + query_len.saturating_sub(overlap_depth),
            );
            potential_tokens.insert(
                worker,
                load.active_tokens(decay_now) + prefill_token_deltas.tokens_for(worker),
            );
            if let Some(active_requests) = active_requests.as_mut() {
                active_requests.insert(worker, load.active_requests);
            }
        }
        (potential_blocks, potential_tokens, active_requests)
    }

    pub(super) fn potential_blocks_and_tokens<const INCLUDE_ACTIVE_REQUESTS: bool>(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> PotentialLoadMaps {
        let query_len = token_sequence.map_or(0, |query| query.len());
        let matched = self.prompts.compute_overlap_depths(token_sequence);
        self.project_loads::<INCLUDE_ACTIVE_REQUESTS>(
            query_len,
            &matched,
            prefill_token_deltas,
            decay_now,
        )
    }

    pub(super) fn project_worker_loads(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        decay_now: Instant,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLoadProjection> {
        let query_len = token_sequence.map_or(0, |query| query.len());
        let matched = self.prompts.compute_overlap_depths(token_sequence);
        let loads = self.loads.read();
        let mut projections = FxHashMap::with_capacity_and_hasher(loads.len(), FxBuildHasher);
        for (worker, load) in loads.iter() {
            let overlap = matched.get(&worker).copied().unwrap_or(0);
            projections.insert(
                worker,
                WorkerLoadProjection {
                    active_prefill_tokens: load.active_tokens(decay_now),
                    active_decode_blocks: load.active_blocks,
                    additional_active_blocks: query_len.saturating_sub(overlap),
                },
            );
        }
        projections
    }

    pub(super) fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .read()
            .iter()
            .map(|(worker, load)| (worker, load.active_blocks))
            .collect()
    }

    pub(super) fn active_request_counts(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .read()
            .iter()
            .map(|(worker, load)| (worker, load.active_requests))
            .collect()
    }

    pub(super) fn active_tokens(&self, decay_now: Instant) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .read()
            .iter()
            .map(|(worker, load)| (worker, load.active_tokens(decay_now)))
            .collect()
    }

    pub(super) fn modeled_remaining_prefill_times_ms(
        &self,
        now: Instant,
    ) -> HashMap<WorkerWithDpRank, Result<u64, PrefillTimeLoadError>> {
        self.loads
            .read()
            .iter()
            .map(|(worker, load)| (worker, load.modeled_remaining_prefill_time_ms(now)))
            .collect()
    }

    pub(super) fn any_worker_matches_active_tokens(
        &self,
        decay_now: Instant,
        mut predicate: impl FnMut(WorkerWithDpRank, usize) -> bool,
    ) -> bool {
        self.loads
            .read()
            .iter()
            .any(|(worker, load)| predicate(worker, load.active_tokens(decay_now)))
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_block_index_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_use_authoritative_unified_overlap() {
        let worker = WorkerWithDpRank::new(1, 0);
        let prompts = Arc::new(UnifiedPromptTracker::default());
        let registry = PromptRegistry::new([worker], prompts.clone());
        let handle = prompts.acquire(worker, &[1, 2, 3]);
        registry.replace_worker_load_state(
            worker,
            WorkerLoadSnapshot {
                active_blocks: 3,
                active_requests: 1,
                prefill: PrefillLoadSnapshot::default(),
            },
        );
        let projected = registry.project_worker_loads(Some(&[1, 2, 4, 5]), Instant::now());
        assert_eq!(projected[&worker].additional_active_blocks, 2);
        prompts.release(worker, handle);
    }
}
