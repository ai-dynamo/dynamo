// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::DashMap;
use dynamo_tokens::SequenceHash;
#[cfg(test)]
use rustc_hash::FxHashSet;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::Instant;

use super::prefill_tracker::{PrefillLoadSnapshot, added_prefill_tokens};
use super::prompt_membership_trie::{PromptMembershipTrie, WorkerLookup};
use super::single::PromptMembershipDelta;
use super::topology::WorkerTopologyChange;
use crate::protocols::{OverlapScores, WorkerWithDpRank};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorkerLoadSnapshot {
    pub(super) active_blocks: usize,
    pub(super) prefill: PrefillLoadSnapshot,
}

impl WorkerLoadSnapshot {
    pub(super) fn active_tokens(&self, decay_now: Instant) -> usize {
        self.prefill.active_tokens_at(decay_now)
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
    loads: DashMap<WorkerWithDpRank, WorkerLoadSnapshot, FxBuildHasher>,
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self {
            membership: PromptMembershipTrie::new(),
            loads: DashMap::with_hasher(FxBuildHasher),
        }
    }
}

impl PromptRegistry {
    pub(super) fn new(workers: impl IntoIterator<Item = WorkerWithDpRank>) -> Self {
        let registry = Self::default();
        for worker in workers {
            registry.loads.entry(worker).or_default();
        }
        registry
    }

    pub(super) fn replace_worker_load_state(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
    ) {
        self.loads.insert(worker, load);
    }

    pub(super) fn apply_membership_delta_and_load(
        &self,
        worker: WorkerWithDpRank,
        lookup: &Arc<parking_lot::RwLock<WorkerLookup>>,
        delta: PromptMembershipDelta,
        load: WorkerLoadSnapshot,
    ) {
        for remove in delta.removes {
            self.membership.remove_chain(worker, lookup, &remove.hashes);
        }
        for store in delta.stores {
            self.membership
                .store_chain(worker, lookup, store.parent, &store.hashes);
        }
        self.loads.insert(worker, load);
        self.membership.maybe_cleanup();
    }

    pub(super) fn apply_topology_change(&self, change: WorkerTopologyChange) {
        for removed in change.removed {
            self.membership
                .remove_worker(removed.worker, &removed.trie_lookup);
            self.loads.remove(&removed.worker);
        }

        for worker in change.added {
            self.loads.entry(worker).or_default();
        }
        self.membership.maybe_cleanup();
    }

    #[expect(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    fn project_loads_from_overlap(
        &self,
        query_len: usize,
        matched_depth: &FxHashMap<WorkerWithDpRank, usize>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
        block_size: usize,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        let mut potential_blocks =
            FxHashMap::with_capacity_and_hasher(self.loads.len(), FxBuildHasher);
        let mut potential_tokens =
            FxHashMap::with_capacity_and_hasher(self.loads.len(), FxBuildHasher);

        for entry in &self.loads {
            let worker = *entry.key();
            let load = *entry.value();
            let overlap_depth = matched_depth.get(&worker).copied().unwrap_or(0);
            let new_blocks = query_len.saturating_sub(overlap_depth);
            let active_tokens = load.active_tokens(decay_now);
            let overlap = *overlaps.scores.get(&worker).unwrap_or(&0);
            let added_tokens = if track_prefill_tokens {
                added_prefill_tokens(block_size, isl, overlap)
            } else {
                0
            };

            potential_blocks.insert(worker, load.active_blocks + new_blocks);
            potential_tokens.insert(worker, active_tokens + added_tokens);
        }

        (potential_blocks, potential_tokens)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn potential_blocks_and_tokens_with_prefill_tracking(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
        block_size: usize,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        let query_len = token_sequence.map_or(0, |query| query.len());
        let matched_depth = self.membership.compute_overlap_depths(token_sequence);
        self.project_loads_from_overlap(
            query_len,
            &matched_depth,
            isl,
            overlaps,
            track_prefill_tokens,
            block_size,
            decay_now,
        )
    }

    pub(super) fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .iter()
            .map(|entry| (*entry.key(), entry.value().active_blocks))
            .collect()
    }

    pub(super) fn active_tokens(&self, decay_now: Instant) -> HashMap<WorkerWithDpRank, usize> {
        self.loads
            .iter()
            .map(|entry| (*entry.key(), entry.value().active_tokens(decay_now)))
            .collect()
    }

    pub(super) fn any_worker_matches_active_tokens(
        &self,
        decay_now: Instant,
        mut predicate: impl FnMut(WorkerWithDpRank, usize) -> bool,
    ) -> bool {
        self.loads
            .iter()
            .any(|entry| predicate(*entry.key(), entry.value().active_tokens(decay_now)))
    }

    #[cfg(test)]
    pub(super) fn assert_consistent_with_workers(
        &self,
        expected_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
        expected_blocks: &FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
    ) {
        let actual_loads: FxHashMap<_, _> = self
            .loads
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
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

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    fn lookup() -> Arc<parking_lot::RwLock<WorkerLookup>> {
        Arc::new(parking_lot::RwLock::new(WorkerLookup::default()))
    }

    fn store(parent: Option<SequenceHash>, hashes: &[SequenceHash]) -> PromptMembershipDelta {
        PromptMembershipDelta {
            stores: vec![PromptMembershipStore {
                parent,
                hashes: hashes.to_vec(),
            }],
            removes: Vec::new(),
        }
    }

    fn worker_load_snapshot(active_blocks: usize) -> WorkerLoadSnapshot {
        WorkerLoadSnapshot {
            active_blocks,
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
            active_blocks,
            prefill: PrefillLoadSnapshot {
                prefill_full_tokens_sum,
                anchored_prefill: Some(AnchoredPrefillSnapshot {
                    initial_effective_prefill_tokens: anchored_tokens,
                    expected_prefill_duration,
                    anchored_since,
                }),
            },
        }
    }

    fn hash_set(hashes: &[SequenceHash]) -> FxHashSet<SequenceHash> {
        hashes.iter().copied().collect()
    }

    #[expect(clippy::too_many_arguments)]
    fn naive_potential_loads(
        expected_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
        expected_blocks: &FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
        block_size: usize,
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
            let overlap = *overlaps.scores.get(&worker).unwrap_or(&0);
            let added_tokens = if track_prefill_tokens {
                added_prefill_tokens(block_size, isl, overlap)
            } else {
                0
            };

            potential_blocks.insert(worker, load.active_blocks + new_blocks);
            potential_tokens.insert(worker, load.active_tokens(decay_now) + added_tokens);
        }

        (potential_blocks, potential_tokens)
    }

    #[test]
    fn removed_hash_can_be_restored_by_later_store() {
        let worker = worker(1, 0);
        let registry = PromptRegistry::new([worker]);
        let lookup = lookup();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        registry.apply_membership_delta_and_load(
            worker,
            &lookup,
            store(None, &[42]),
            worker_load_snapshot(1),
        );
        let load = worker_load_snapshot(1);
        registry.apply_membership_delta_and_load(
            worker,
            &lookup,
            PromptMembershipDelta {
                removes: vec![PromptMembershipRemove { hashes: vec![42] }],
                ..Default::default()
            },
            load,
        );
        registry.apply_membership_delta_and_load(worker, &lookup, store(None, &[42]), load);
        expected_loads.insert(worker, load);
        expected_blocks.insert(worker, hash_set(&[42]));

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
    }

    #[test]
    fn staggered_prefix_overlap_matches_naive_projection() {
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let worker_c = worker(3, 0);
        let registry = PromptRegistry::new([worker_a, worker_b, worker_c]);
        let lookup_a = lookup();
        let lookup_b = lookup();
        let lookup_c = lookup();
        let decay_now = Instant::now();
        let full_prompt: Vec<SequenceHash> = (1_u64..=96).collect();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        for (worker, lookup, prompt_len) in [
            (worker_a, &lookup_a, 96usize),
            (worker_b, &lookup_b, 64),
            (worker_c, &lookup_c, 33),
        ] {
            let blocks = full_prompt[..prompt_len].to_vec();
            let load = worker_load_snapshot(prompt_len);
            registry.apply_membership_delta_and_load(worker, lookup, store(None, &blocks), load);
            expected_loads.insert(worker, load);
            expected_blocks.insert(worker, blocks.into_iter().collect());
        }

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);

        let expected = naive_potential_loads(
            &expected_loads,
            &expected_blocks,
            Some(&full_prompt),
            384,
            &OverlapScores::default(),
            false,
            4,
            decay_now,
        );
        let actual = registry.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&full_prompt),
            384,
            &OverlapScores::default(),
            false,
            4,
            decay_now,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn load_only_update_preserves_prompt_membership_and_active_token_projection() {
        let worker = worker(1, 0);
        let registry = PromptRegistry::new([worker]);
        let lookup = lookup();
        let now = Instant::now();
        let anchored_since = now.checked_sub(Duration::from_secs(3)).unwrap_or(now);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        registry.apply_membership_delta_and_load(
            worker,
            &lookup,
            store(None, &[1, 2, 3]),
            worker_load_snapshot(3),
        );
        expected_blocks.insert(worker, hash_set(&[1, 2, 3]));

        let updated_load =
            anchored_load_snapshot(5, 12, 10, Some(Duration::from_secs(10)), anchored_since);
        registry.replace_worker_load_state(worker, updated_load);
        expected_loads.insert(worker, updated_load);

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
        assert_eq!(registry.active_tokens(now).get(&worker).copied(), Some(9));

        let actual = registry.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&[1, 2, 3]),
            12,
            &OverlapScores::default(),
            false,
            4,
            now,
        );
        assert_eq!(actual.0.get(&worker).copied(), Some(5));
        assert_eq!(actual.1.get(&worker).copied(), Some(9));
    }

    #[test]
    fn removing_worker_clears_prompt_membership_and_load_state() {
        let worker_a = worker(1, 0);
        let worker_b = worker(2, 0);
        let registry = PromptRegistry::new([worker_a, worker_b]);
        let lookup_a = lookup();
        let lookup_b = lookup();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(2);
        registry.apply_membership_delta_and_load(
            worker_a,
            &lookup_a,
            store(None, &[1, 2, 3]),
            load_a,
        );
        registry.apply_membership_delta_and_load(worker_b, &lookup_b, store(None, &[1, 2]), load_b);
        expected_loads.insert(worker_a, load_a);
        expected_loads.insert(worker_b, load_b);
        expected_blocks.insert(worker_a, hash_set(&[1, 2, 3]));
        expected_blocks.insert(worker_b, hash_set(&[1, 2]));

        registry.apply_topology_change(WorkerTopologyChange {
            added: Vec::new(),
            removed: vec![RemovedWorkerState {
                worker: worker_a,
                trie_lookup: Arc::clone(&lookup_a),
            }],
        });
        expected_loads.remove(&worker_a);
        expected_blocks.remove(&worker_a);

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
        assert!(!registry.active_blocks().contains_key(&worker_a));

        let actual = registry.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&[1, 2, 3]),
            12,
            &OverlapScores::default(),
            false,
            4,
            Instant::now(),
        );
        assert_eq!(actual.0.get(&worker_b).copied(), Some(3));
    }

    #[test]
    fn dp_ranks_with_same_worker_id_remain_isolated() {
        let worker_a = worker(1, 0);
        let worker_b = worker(1, 1);
        let registry = PromptRegistry::new([worker_a, worker_b]);
        let lookup_a = lookup();
        let lookup_b = lookup();
        let decay_now = Instant::now();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(1);
        registry.apply_membership_delta_and_load(
            worker_a,
            &lookup_a,
            store(None, &[1, 2, 3]),
            load_a,
        );
        registry.apply_membership_delta_and_load(worker_b, &lookup_b, store(None, &[1]), load_b);
        expected_loads.insert(worker_a, load_a);
        expected_loads.insert(worker_b, load_b);
        expected_blocks.insert(worker_a, hash_set(&[1, 2, 3]));
        expected_blocks.insert(worker_b, hash_set(&[1]));

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);

        let expected = naive_potential_loads(
            &expected_loads,
            &expected_blocks,
            Some(&[1, 2, 3]),
            12,
            &OverlapScores::default(),
            false,
            4,
            decay_now,
        );
        let actual = registry.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&[1, 2, 3]),
            12,
            &OverlapScores::default(),
            false,
            4,
            decay_now,
        );

        assert_eq!(actual, expected);
    }
}
