// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use dynamo_tokens::SequenceHash;
use indexmap::IndexMap;
use parking_lot::RwLock;
#[cfg(test)]
use rustc_hash::FxHashSet;
use rustc_hash::{FxBuildHasher, FxHashMap};
use seqlock::SeqLock;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::time::Instant;

use super::PrefillTokenDeltas;
use super::prefill_tracker::{PrefillLoadSnapshot, PrefillTimeLoadError};
use super::prompt_membership_trie::PromptMembershipTrie;
use super::single::PromptMembershipDelta;
use super::topology::WorkerTopologyChange;
use crate::active_sequence::{ActiveSequenceStride, TrackedSequenceHashes};
use crate::protocols::WorkerWithDpRank;

/// Ephemeral, request-specific view of worker load.
///
/// Values are materialized for one incoming request at a specific instant and
/// should be passed to worker selection immediately. Do not persist, reuse, or
/// publish this projection as durable worker state.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorkerLoadProjection {
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    /// Request blocks not already shared with active sequences on this worker.
    ///
    /// These blocks may still exist in an inactive cache; this field describes
    /// additional active block footprint, not cache misses.
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

/// Reusable snapshot of a worker's currently tracked execution state.
///
/// Prompt load is retained in sampled units while output load remains in
/// physical blocks. The outer registry applies the configured prompt stride
/// before exposing decode-block estimates. `prefill` retains the decay state
/// needed to evaluate active prefill load in tokens at a caller-provided
/// instant. Unlike [`WorkerLoadProjection`], this snapshot contains no
/// incoming-request-specific state.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(super) struct WorkerLoadSnapshot {
    pub(super) active_prompt_units: f64,
    pub(super) active_output_blocks: f64,
    pub(super) active_requests: usize,
    pub(super) prefill: PrefillLoadSnapshot,
}

impl WorkerLoadSnapshot {
    pub(super) fn active_decode_blocks(&self, prompt_stride: ActiveSequenceStride) -> usize {
        super::estimate_physical_active_blocks(
            self.active_prompt_units,
            self.active_output_blocks,
            prompt_stride,
        )
    }

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
    // SeqLock gives the hot projection path a lock-free read of the latest
    // whole-worker snapshot. Writers mark the sequence odd, replace the
    // snapshot, then mark the sequence even. Readers only return after seeing
    // the same even sequence before and after copying the `Copy` payload, so a
    // copy that overlaps a write is retried instead of exposing a mixed
    // snapshot. Writers still serialize with each other, but they do not wait
    // for readers.
    //
    // The upstream crate implements the copy with `read_volatile`, which is
    // technically UB under Rust/LLVM if a writer mutates the same memory at the
    // same time: volatile is not atomic and does not by itself legalize a data
    // race. See https://github.com/Amanieu/seqlock/issues/2#issuecomment-473606523.
    //
    // `WorkerLoadSnapshot` is a small `Copy` value with no references or drop
    // state. The sequence check is what makes the read logically race-free for
    // this slot: a reader either observes a stable whole snapshot or retries.
    // In practice this is the same seqlock/READ_ONCE pattern the crate is
    // designed for, while avoiding the per-worker RwLock read on every request.
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
    // IndexMap gives us the dense full-worker scan plus point lookup shape that was previously
    // hand-rolled as Vec<WorkerLoadSlot> + FxHashMap<WorkerWithDpRank, usize>.
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

pub(super) struct PromptRegistry {
    // WARNING: prompt membership and worker load are only eventually consistent.
    // Each mutation still starts from one worker-local source of truth: we mutate the chosen
    // `ActiveSequences`, derive an exact `PromptMembershipDelta` plus `WorkerLoadSnapshot`, then
    // publish them separately here. The trie and load map converge to the correct final state
    // after the write finishes, but reads can still observe a mixed membership/load state that
    // never existed atomically and make a suboptimal routing choice.
    membership: PromptMembershipTrie,
    loads: RwLock<WorkerLoadTable>,
    active_sequence_stride: ActiveSequenceStride,
    #[cfg(test)]
    cleanup_attempts: AtomicUsize,
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self {
            membership: PromptMembershipTrie::new(),
            loads: RwLock::new(WorkerLoadTable::default()),
            active_sequence_stride: ActiveSequenceStride::ONE,
            #[cfg(test)]
            cleanup_attempts: AtomicUsize::new(0),
        }
    }
}

impl PromptRegistry {
    pub(super) fn new(
        workers: impl IntoIterator<Item = WorkerWithDpRank>,
        active_sequence_stride: ActiveSequenceStride,
    ) -> Self {
        let registry = Self {
            active_sequence_stride,
            ..Self::default()
        };
        let mut loads = registry.loads.write();
        for worker in workers {
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
        self.upsert_worker_load(worker, load);
    }

    pub(super) fn apply_membership_delta_and_load(
        &self,
        worker: WorkerWithDpRank,
        delta: PromptMembershipDelta,
        load: WorkerLoadSnapshot,
    ) {
        self.apply_membership_delta_and_load_without_cleanup(worker, delta, load);
        self.maybe_cleanup();
    }

    /// Apply one per-worker sequence mutation in lifecycle order.
    ///
    /// `delta` generation and application must remain serialized by the worker
    /// slot's `sequences.write()` lock. Removals must be applied before stores
    /// because expiry is evaluated before the new request is acquired, and both
    /// path boundaries describe that exact intermediate state.
    pub(super) fn apply_membership_delta_and_load_without_cleanup(
        &self,
        worker: WorkerWithDpRank,
        delta: PromptMembershipDelta,
        load: WorkerLoadSnapshot,
    ) {
        for remove in delta.removes {
            self.membership
                .remove_path(worker, &remove.path, remove.remove_from);
        }
        for store in delta.stores {
            self.membership
                .store_path(worker, &store.path, store.new_suffix_start);
        }
        self.upsert_worker_load(worker, load);
    }

    fn upsert_worker_load(&self, worker: WorkerWithDpRank, load: WorkerLoadSnapshot) {
        if self.loads.read().update(worker, load) {
            return;
        }
        self.loads.write().upsert(worker, load);
    }

    pub(super) fn maybe_cleanup(&self) {
        #[cfg(test)]
        self.cleanup_attempts.fetch_add(1, Ordering::Relaxed);
        self.membership.maybe_cleanup();
    }

    #[cfg(test)]
    pub(super) fn cleanup_attempts(&self) -> usize {
        self.cleanup_attempts.load(Ordering::Relaxed)
    }

    pub(super) fn apply_topology_change_without_cleanup(&self, change: &WorkerTopologyChange) {
        for removed in &change.removed {
            self.membership.remove_worker(removed.worker);
            self.loads.write().remove(removed.worker);
        }

        for &worker in &change.added {
            self.loads.write().ensure_worker(worker);
        }
    }

    fn project_loads_from_membership<const INCLUDE_ACTIVE_REQUESTS: bool>(
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
            let new_blocks = query_len.saturating_sub(overlap_depth);
            let active_tokens = load.active_tokens(decay_now);
            let added_tokens = prefill_token_deltas.tokens_for(worker);

            potential_blocks.insert(
                worker,
                load.active_decode_blocks(self.active_sequence_stride)
                    .saturating_add(new_blocks.saturating_mul(self.active_sequence_stride.get())),
            );
            potential_tokens.insert(worker, active_tokens + added_tokens);
            if let Some(active_requests) = active_requests.as_mut() {
                active_requests.insert(worker, load.active_requests);
            }
        }

        (potential_blocks, potential_tokens, active_requests)
    }

    pub(super) fn potential_blocks_and_tokens<const INCLUDE_ACTIVE_REQUESTS: bool>(
        &self,
        token_sequence: Option<&TrackedSequenceHashes>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> PotentialLoadMaps {
        let token_sequence = token_sequence.map(TrackedSequenceHashes::as_slice);
        let query_len = token_sequence.map_or(0, |query| query.len());
        let matched_depth = self.membership.compute_overlap_depths(token_sequence);
        self.project_loads_from_membership::<INCLUDE_ACTIVE_REQUESTS>(
            query_len,
            &matched_depth,
            prefill_token_deltas,
            decay_now,
        )
    }

    pub(super) fn project_worker_loads(
        &self,
        token_sequence: Option<&TrackedSequenceHashes>,
        decay_now: Instant,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLoadProjection> {
        let token_sequence = token_sequence.map(TrackedSequenceHashes::as_slice);
        let query_len = token_sequence.map_or(0, |query| query.len());
        let matched_depth = self.membership.compute_overlap_depths(token_sequence);
        let loads = self.loads.read();
        let mut projections = FxHashMap::with_capacity_and_hasher(loads.len(), FxBuildHasher);

        for (worker, load) in loads.iter() {
            let overlap_depth = matched_depth.get(&worker).copied().unwrap_or(0);
            projections.insert(
                worker,
                WorkerLoadProjection {
                    active_prefill_tokens: load.active_tokens(decay_now),
                    active_decode_blocks: load.active_decode_blocks(self.active_sequence_stride),
                    additional_active_blocks: query_len
                        .saturating_sub(overlap_depth)
                        .saturating_mul(self.active_sequence_stride.get()),
                },
            );
        }

        projections
    }

    pub(super) fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .read()
            .iter()
            .map(|(worker, load)| {
                (
                    worker,
                    load.active_decode_blocks(self.active_sequence_stride),
                )
            })
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

    #[cfg(test)]
    pub(super) fn assert_consistent_with_workers(
        &self,
        expected_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
        expected_blocks: &FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
    ) {
        let actual_loads: FxHashMap<_, _> = self.loads.read().iter().collect();
        let actual_blocks = self.membership.worker_hashes();
        assert_eq!(
            actual_loads, *expected_loads,
            "prompt registry worker loads drifted from per-worker state",
        );
        assert_eq!(
            actual_blocks, *expected_blocks,
            "prompt registry prompt membership drifted from per-worker state",
        );
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_block_index_empty(&self) -> bool {
        self.membership.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rustc_hash::{FxHashMap, FxHashSet};

    use super::*;
    use crate::protocols::WorkerWithDpRank;
    use crate::sequences::prefill_tracker::AnchoredPrefillSnapshot;
    use crate::sequences::single::{PromptMembershipRemove, PromptMembershipStore};
    use crate::sequences::topology::RemovedWorkerState;

    fn tracked(sequence: Vec<SequenceHash>) -> TrackedSequenceHashes {
        crate::active_sequence::tracked_sequence_hashes(sequence)
    }

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    fn store(path: &[SequenceHash], new_suffix_start: usize) -> PromptMembershipDelta {
        PromptMembershipDelta {
            stores: vec![PromptMembershipStore {
                path: path.to_vec(),
                new_suffix_start,
            }],
            removes: Vec::new(),
        }
    }

    fn worker_load_snapshot(active_blocks: usize) -> WorkerLoadSnapshot {
        WorkerLoadSnapshot {
            active_prompt_units: active_blocks as f64,
            active_output_blocks: 0.0,
            active_requests: 0,
            prefill: PrefillLoadSnapshot::default(),
        }
    }

    fn anchored_load_snapshot(
        active_blocks: usize,
        prefill_full_tokens_sum: usize,
        anchored_tokens: usize,
        expected_prefill_duration: Option<Duration>,
        anchored_since: Instant,
    ) -> WorkerLoadSnapshot {
        WorkerLoadSnapshot {
            active_prompt_units: active_blocks as f64,
            active_output_blocks: 0.0,
            active_requests: 0,
            prefill: PrefillLoadSnapshot {
                prefill_full_tokens_sum,
                anchored_prefill: Some(AnchoredPrefillSnapshot {
                    initial_effective_prefill_tokens: anchored_tokens,
                    expected_prefill_duration,
                    anchored_since,
                }),
                total_modeled_prefill_time_ms: expected_prefill_duration
                    .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64),
            },
        }
    }

    fn hash_set(hashes: &[SequenceHash]) -> FxHashSet<SequenceHash> {
        hashes.iter().copied().collect()
    }

    fn naive_potential_loads(
        expected_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
        expected_blocks: &FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
        token_sequence: Option<&[SequenceHash]>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        let mut potential_blocks =
            FxHashMap::with_capacity_and_hasher(expected_loads.len(), FxBuildHasher);
        let mut potential_tokens =
            FxHashMap::with_capacity_and_hasher(expected_loads.len(), FxBuildHasher);

        for (&worker, load) in expected_loads {
            let overlap_depth = token_sequence.map_or(0, |query| {
                let worker_blocks = expected_blocks
                    .get(&worker)
                    .expect("worker must have a prompt membership entry");
                query
                    .iter()
                    .position(|hash| !worker_blocks.contains(hash))
                    .unwrap_or(query.len())
            });
            let new_blocks =
                token_sequence.map_or(0, |query| query.len().saturating_sub(overlap_depth));
            let added_tokens = prefill_token_deltas.tokens_for(worker);

            potential_blocks.insert(
                worker,
                load.active_decode_blocks(ActiveSequenceStride::ONE) + new_blocks,
            );
            potential_tokens.insert(worker, load.active_tokens(decay_now) + added_tokens);
        }

        (potential_blocks, potential_tokens)
    }

    #[test]
    fn removed_hash_can_be_restored_by_later_store() {
        let worker = worker(1, 0);
        let registry = PromptRegistry::new([worker], ActiveSequenceStride::ONE);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        registry.apply_membership_delta_and_load(worker, store(&[42], 0), worker_load_snapshot(1));
        let load = worker_load_snapshot(1);
        registry.apply_membership_delta_and_load(
            worker,
            PromptMembershipDelta {
                removes: vec![PromptMembershipRemove {
                    path: vec![42],
                    remove_from: 0,
                }],
                ..Default::default()
            },
            load,
        );
        registry.apply_membership_delta_and_load(worker, store(&[42], 0), load);
        expected_loads.insert(worker, load);
        expected_blocks.insert(worker, hash_set(&[42]));

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
    }

    #[test]
    fn staggered_prefix_overlap_matches_naive_projection() {
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let worker_c = worker(3, 0);
        let registry =
            PromptRegistry::new([worker_a, worker_b, worker_c], ActiveSequenceStride::ONE);
        let decay_now = Instant::now();
        let full_prompt: Vec<SequenceHash> = (1_u64..=96).collect();
        let tracked_full_prompt = tracked(full_prompt.clone());
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        for (worker, prompt_len) in [(worker_a, 96usize), (worker_b, 64), (worker_c, 33)] {
            let blocks = full_prompt[..prompt_len].to_vec();
            let load = worker_load_snapshot(prompt_len);
            registry.apply_membership_delta_and_load(worker, store(&blocks, 0), load);
            expected_loads.insert(worker, load);
            expected_blocks.insert(worker, blocks.into_iter().collect());
        }

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);

        let expected = naive_potential_loads(
            &expected_loads,
            &expected_blocks,
            Some(&full_prompt),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        let actual = registry.potential_blocks_and_tokens::<false>(
            Some(&tracked_full_prompt),
            &PrefillTokenDeltas::none(),
            decay_now,
        );

        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
    }

    #[test]
    fn load_only_update_preserves_prompt_membership_and_active_token_projection() {
        let worker = worker(1, 0);
        let registry = PromptRegistry::new([worker], ActiveSequenceStride::ONE);
        let now = Instant::now();
        let anchored_since = now.checked_sub(Duration::from_secs(3)).unwrap_or(now);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        registry.apply_membership_delta_and_load(
            worker,
            store(&[1, 2, 3], 0),
            worker_load_snapshot(3),
        );
        expected_blocks.insert(worker, hash_set(&[1, 2, 3]));

        let updated_load =
            anchored_load_snapshot(5, 12, 10, Some(Duration::from_secs(10)), anchored_since);
        registry.replace_worker_load_state(worker, updated_load);
        expected_loads.insert(worker, updated_load);

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
        assert_eq!(registry.active_tokens(now).get(&worker).copied(), Some(9));

        let actual = registry.potential_blocks_and_tokens::<false>(
            Some(&tracked(vec![1, 2, 3])),
            &PrefillTokenDeltas::none(),
            now,
        );
        assert_eq!(actual.0.get(&worker).copied(), Some(5));
        assert_eq!(actual.1.get(&worker).copied(), Some(9));
    }

    #[test]
    fn removing_worker_clears_prompt_membership_and_load_state() {
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let registry = PromptRegistry::new([worker_a, worker_b], ActiveSequenceStride::ONE);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(2);
        registry.apply_membership_delta_and_load(worker_a, store(&[1, 2, 3], 0), load_a);
        registry.apply_membership_delta_and_load(worker_b, store(&[1, 2], 0), load_b);
        expected_loads.insert(worker_a, load_a);
        expected_loads.insert(worker_b, load_b);
        expected_blocks.insert(worker_a, hash_set(&[1, 2, 3]));
        expected_blocks.insert(worker_b, hash_set(&[1, 2]));

        registry.apply_topology_change_without_cleanup(&WorkerTopologyChange {
            added: Vec::new(),
            removed: vec![RemovedWorkerState { worker: worker_a }],
        });
        expected_loads.remove(&worker_a);
        expected_blocks.remove(&worker_a);

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
        assert!(!registry.active_blocks().contains_key(&worker_a));

        let actual = registry.potential_blocks_and_tokens::<false>(
            Some(&tracked(vec![1, 2, 3])),
            &PrefillTokenDeltas::none(),
            Instant::now(),
        );
        assert_eq!(actual.0.get(&worker_b).copied(), Some(3));
    }

    #[test]
    fn dp_ranks_with_same_worker_id_remain_isolated() {
        let worker_a = worker(1, 0);
        let worker_b = worker(1, 1);
        let registry = PromptRegistry::new([worker_a, worker_b], ActiveSequenceStride::ONE);
        let decay_now = Instant::now();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(1);
        registry.apply_membership_delta_and_load(worker_a, store(&[1, 2, 3], 0), load_a);
        registry.apply_membership_delta_and_load(worker_b, store(&[1], 0), load_b);
        expected_loads.insert(worker_a, load_a);
        expected_loads.insert(worker_b, load_b);
        expected_blocks.insert(worker_a, hash_set(&[1, 2, 3]));
        expected_blocks.insert(worker_b, hash_set(&[1]));

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);

        let expected = naive_potential_loads(
            &expected_loads,
            &expected_blocks,
            Some(&[1, 2, 3]),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        let actual = registry.potential_blocks_and_tokens::<false>(
            Some(&tracked(vec![1, 2, 3])),
            &PrefillTokenDeltas::none(),
            decay_now,
        );

        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
    }

    #[test]
    fn sparse_projection_scales_prompt_units_and_preserves_output_blocks() {
        let worker = worker(1, 0);
        let registry = PromptRegistry::new([worker], ActiveSequenceStride::new(4).unwrap());
        let load = WorkerLoadSnapshot {
            active_prompt_units: 2.5,
            active_output_blocks: 1.5,
            active_requests: 1,
            prefill: PrefillLoadSnapshot::default(),
        };
        registry.apply_membership_delta_and_load(worker, store(&[10, 20], 0), load);

        assert_eq!(registry.active_blocks().get(&worker), Some(&12));

        let identical = registry.project_worker_loads(Some(&tracked(vec![10, 20])), Instant::now());
        assert_eq!(
            identical.get(&worker),
            Some(&WorkerLoadProjection {
                active_prefill_tokens: 0,
                active_decode_blocks: 12,
                additional_active_blocks: 0,
            })
        );

        let diverged = registry.project_worker_loads(Some(&tracked(vec![10, 99])), Instant::now());
        assert_eq!(diverged[&worker].active_decode_blocks, 12);
        assert_eq!(diverged[&worker].additional_active_blocks, 4);

        let (potential_blocks, _, _) = registry.potential_blocks_and_tokens::<false>(
            Some(&tracked(vec![10, 99])),
            &PrefillTokenDeltas::none(),
            Instant::now(),
        );
        assert_eq!(potential_blocks.get(&worker), Some(&16));
    }
}
