// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use tokio::time::Instant;

use super::prefill_tracker::{PrefillLoadSnapshot, added_prefill_tokens};
use super::single::BlockPresenceDelta;
use super::topology::WorkerTopologyChange;
use crate::active_set::reconcile_active_workers;
use crate::protocols::{OverlapScores, WorkerWithDpRank};

const OVERLAP_JUMP_SIZE: usize = 32;

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

#[derive(Debug, Default)]
struct PromptRegistryInner {
    block_workers: FxHashMap<SequenceHash, FxHashSet<WorkerWithDpRank>>,
    worker_blocks: FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
    workers: FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
}

#[derive(Debug, Default)]
pub(super) struct PromptRegistry {
    // TODO: This global RwLock is still coarse. Revisit lock granularity separately.
    inner: RwLock<PromptRegistryInner>,
}

impl PromptRegistry {
    fn apply_block_delta_inner(
        inner: &mut PromptRegistryInner,
        worker: WorkerWithDpRank,
        block_delta: BlockPresenceDelta,
    ) {
        inner.worker_blocks.entry(worker).or_default();

        for hash in block_delta.blocks_became_absent {
            if let Some(worker_blocks) = inner.worker_blocks.get_mut(&worker) {
                worker_blocks.remove(&hash);
            }
            let should_remove = inner.block_workers.get_mut(&hash).is_some_and(|workers| {
                workers.remove(&worker);
                workers.is_empty()
            });
            if should_remove {
                inner.block_workers.remove(&hash);
            }
        }

        for hash in block_delta.blocks_became_present {
            inner.worker_blocks.entry(worker).or_default().insert(hash);
            inner.block_workers.entry(hash).or_default().insert(worker);
        }
    }

    pub(super) fn new(workers: impl IntoIterator<Item = WorkerWithDpRank>) -> Self {
        let registry = Self::default();
        {
            let mut inner = registry.inner.write();
            for worker in workers {
                inner.workers.entry(worker).or_default();
                inner.worker_blocks.entry(worker).or_default();
            }
        }
        registry
    }

    pub(super) fn replace_worker_load_state(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
    ) {
        let mut inner = self.inner.write();
        inner.workers.entry(worker).or_default();
        inner.worker_blocks.entry(worker).or_default();
        inner.workers.insert(worker, load);
    }

    pub(super) fn apply_block_delta_and_load(
        &self,
        worker: WorkerWithDpRank,
        block_delta: BlockPresenceDelta,
        load: WorkerLoadSnapshot,
    ) {
        let mut inner = self.inner.write();
        Self::apply_block_delta_inner(&mut inner, worker, block_delta);
        inner.workers.entry(worker).or_default();
        inner.worker_blocks.entry(worker).or_default();
        inner.workers.insert(worker, load);
    }

    pub(super) fn apply_topology_change(&self, change: WorkerTopologyChange) {
        let mut inner = self.inner.write();

        for worker in change.removed {
            inner.workers.remove(&worker);
            let Some(blocks) = inner.worker_blocks.remove(&worker) else {
                continue;
            };

            for hash in blocks {
                let should_remove = inner.block_workers.get_mut(&hash).is_some_and(|workers| {
                    workers.remove(&worker);
                    workers.is_empty()
                });
                if should_remove {
                    inner.block_workers.remove(&hash);
                }
            }
        }

        for worker in change.added {
            inner.workers.entry(worker).or_default();
            inner.worker_blocks.entry(worker).or_default();
        }
    }

    fn linear_scan_drain(
        inner: &PromptRegistryInner,
        query: &[SequenceHash],
        active: &mut FxHashSet<WorkerWithDpRank>,
        matched_depth: &mut FxHashMap<WorkerWithDpRank, usize>,
        lo: usize,
        hi: usize,
    ) {
        if active.is_empty() {
            return;
        }

        for (pos, hash) in query.iter().enumerate().take(hi).skip(lo) {
            if active.is_empty() {
                break;
            }

            let Some(workers) = inner.block_workers.get(hash) else {
                for worker in active.drain() {
                    matched_depth.insert(worker, pos);
                }
                break;
            };

            if workers.len() == active.len() {
                continue;
            }

            reconcile_active_workers(active, workers, |worker| {
                matched_depth.insert(worker, pos);
            });
        }
    }

    fn compute_overlap_depths(
        inner: &PromptRegistryInner,
        token_sequence: Option<&[SequenceHash]>,
    ) -> (usize, FxHashMap<WorkerWithDpRank, usize>) {
        let mut matched_depth = FxHashMap::default();
        let query_len = token_sequence.map_or(0, |query| query.len());

        if let Some(query) = token_sequence
            && !query.is_empty()
        {
            // TODO: This is generic cached-block overlap with prefix-drain semantics. It is
            // weaker than the positional/indexer path and output blocks still use random hashes.
            if let Some(first_workers) = inner.block_workers.get(&query[0]) {
                let mut active = first_workers.clone();
                let mut current_pos = 0;

                while current_pos < query.len().saturating_sub(1) && !active.is_empty() {
                    let next_pos = (current_pos + OVERLAP_JUMP_SIZE).min(query.len() - 1);

                    match inner.block_workers.get(&query[next_pos]) {
                        None => {
                            Self::linear_scan_drain(
                                inner,
                                query,
                                &mut active,
                                &mut matched_depth,
                                current_pos + 1,
                                next_pos + 1,
                            );
                            break;
                        }
                        Some(workers) if workers.len() == active.len() => {
                            current_pos = next_pos;
                        }
                        Some(_) => {
                            Self::linear_scan_drain(
                                inner,
                                query,
                                &mut active,
                                &mut matched_depth,
                                current_pos + 1,
                                next_pos + 1,
                            );
                            current_pos = next_pos;
                        }
                    }
                }

                for worker in active {
                    matched_depth.insert(worker, query.len());
                }
            }
        }

        (query_len, matched_depth)
    }

    #[expect(clippy::too_many_arguments)]
    fn project_loads_from_overlap(
        inner: &PromptRegistryInner,
        query_len: usize,
        matched_depth: &FxHashMap<WorkerWithDpRank, usize>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
        block_size: usize,
        decay_now: Instant,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        let mut potential_blocks = HashMap::with_capacity(inner.workers.len());
        let mut potential_tokens = HashMap::with_capacity(inner.workers.len());
        for (&worker, load) in &inner.workers {
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

    pub(super) fn potential_blocks_and_tokens_with_prefill_tracking(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
        block_size: usize,
        decay_now: Instant,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        let inner = self.inner.read();
        let (query_len, matched_depth) = Self::compute_overlap_depths(&inner, token_sequence);
        Self::project_loads_from_overlap(
            &inner,
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
        let inner = self.inner.read();
        inner
            .workers
            .iter()
            .map(|(&worker, load)| (worker, load.active_blocks))
            .collect()
    }

    pub(super) fn active_tokens(&self, decay_now: Instant) -> HashMap<WorkerWithDpRank, usize> {
        let inner = self.inner.read();
        inner
            .workers
            .iter()
            .map(|(&worker, load)| (worker, load.active_tokens(decay_now)))
            .collect()
    }

    pub(super) fn any_worker_matches_active_tokens(
        &self,
        decay_now: Instant,
        mut predicate: impl FnMut(WorkerWithDpRank, usize) -> bool,
    ) -> bool {
        let inner = self.inner.read();
        inner
            .workers
            .iter()
            .any(|(&worker, load)| predicate(worker, load.active_tokens(decay_now)))
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent_with_workers(
        &self,
        expected_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadSnapshot>,
        expected_blocks: &FxHashMap<WorkerWithDpRank, FxHashSet<SequenceHash>>,
    ) {
        let mut expected_block_workers =
            FxHashMap::<SequenceHash, FxHashSet<WorkerWithDpRank>>::default();
        for (&worker, hashes) in expected_blocks {
            for hash in hashes {
                expected_block_workers
                    .entry(*hash)
                    .or_default()
                    .insert(worker);
            }
        }

        let inner = self.inner.read();
        assert_eq!(
            inner.workers, *expected_loads,
            "prompt registry worker loads drifted from per-worker state",
        );
        assert_eq!(
            inner.worker_blocks, *expected_blocks,
            "prompt registry worker block membership drifted from per-worker state",
        );
        assert_eq!(
            inner.block_workers, expected_block_workers,
            "prompt registry reverse index drifted from per-worker state",
        );
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_block_index_empty(&self) -> bool {
        self.inner.read().block_workers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rustc_hash::{FxHashMap, FxHashSet};

    use super::*;
    use crate::protocols::WorkerWithDpRank;
    use crate::sequences::prefill_tracker::AnchoredPrefillSnapshot;

    fn block_delta(present: &[SequenceHash], absent: &[SequenceHash]) -> BlockPresenceDelta {
        BlockPresenceDelta {
            blocks_became_present: present.to_vec(),
            blocks_became_absent: absent.to_vec(),
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
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        let mut potential_blocks = HashMap::with_capacity(expected_loads.len());
        let mut potential_tokens = HashMap::with_capacity(expected_loads.len());

        for (&worker, load) in expected_loads {
            let overlap_depth = token_sequence.map_or(0, |query| {
                let worker_blocks = expected_blocks
                    .get(&worker)
                    .expect("worker must have a block membership entry");
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
    fn same_hash_absent_and_present_in_one_delta_remains_present() {
        let worker = WorkerWithDpRank::new(1, 0);
        let registry = PromptRegistry::new([worker]);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load = worker_load_snapshot(1);
        registry.apply_block_delta_and_load(worker, block_delta(&[42], &[42]), load);
        expected_loads.insert(worker, load);
        expected_blocks.insert(worker, hash_set(&[42]));

        registry.assert_consistent_with_workers(&expected_loads, &expected_blocks);
    }

    #[test]
    fn staggered_prefix_overlap_matches_naive_projection() {
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let worker_c = WorkerWithDpRank::new(3, 0);
        let registry = PromptRegistry::new([worker_a, worker_b, worker_c]);
        let decay_now = Instant::now();
        let full_prompt: Vec<SequenceHash> = (1_u64..=96).collect();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        for (worker, prompt_len) in [(worker_a, 96usize), (worker_b, 64), (worker_c, 33)] {
            let blocks = full_prompt[..prompt_len].to_vec();
            let load = worker_load_snapshot(prompt_len);
            registry.apply_block_delta_and_load(worker, block_delta(&blocks, &[]), load);
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
    fn load_only_update_preserves_block_membership_and_active_token_projection() {
        let worker = WorkerWithDpRank::new(1, 0);
        let registry = PromptRegistry::new([worker]);
        let now = Instant::now();
        let anchored_since = now.checked_sub(Duration::from_secs(3)).unwrap_or(now);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        registry.apply_block_delta_and_load(
            worker,
            block_delta(&[1, 2, 3], &[]),
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
    fn removing_worker_clears_block_membership_and_load_state() {
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let registry = PromptRegistry::new([worker_a, worker_b]);
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(2);
        registry.apply_block_delta_and_load(worker_a, block_delta(&[1, 2, 3], &[]), load_a);
        registry.apply_block_delta_and_load(worker_b, block_delta(&[1, 2], &[]), load_b);
        expected_loads.insert(worker_a, load_a);
        expected_loads.insert(worker_b, load_b);
        expected_blocks.insert(worker_a, hash_set(&[1, 2, 3]));
        expected_blocks.insert(worker_b, hash_set(&[1, 2]));

        registry.apply_topology_change(WorkerTopologyChange {
            added: Vec::new(),
            removed: vec![worker_a],
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
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(1, 1);
        let registry = PromptRegistry::new([worker_a, worker_b]);
        let decay_now = Instant::now();
        let mut expected_loads = FxHashMap::default();
        let mut expected_blocks = FxHashMap::default();

        let load_a = worker_load_snapshot(3);
        let load_b = worker_load_snapshot(1);
        registry.apply_block_delta_and_load(worker_a, block_delta(&[1, 2, 3], &[]), load_a);
        registry.apply_block_delta_and_load(worker_b, block_delta(&[1], &[]), load_b);
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
