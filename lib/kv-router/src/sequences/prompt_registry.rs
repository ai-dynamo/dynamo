// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use tokio::time::Instant;

use super::prefill_tracker::{PrefillLoadSnapshot, added_prefill_tokens};
use super::single::BlockPresenceDelta;
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

    pub(super) fn insert_empty_worker(&self, worker: WorkerWithDpRank) {
        let mut inner = self.inner.write();
        inner.workers.entry(worker).or_default();
        inner.worker_blocks.entry(worker).or_default();
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

    pub(super) fn remove_worker(&self, worker: WorkerWithDpRank) {
        let mut inner = self.inner.write();
        inner.workers.remove(&worker);
        let Some(blocks) = inner.worker_blocks.remove(&worker) else {
            return;
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
        let mut matched_depth = FxHashMap::default();

        if let Some(query) = token_sequence
            && !query.is_empty()
        {
            // TODO: This is generic cached-block overlap with prefix-drain semantics. It is
            // weaker than the positional/indexer path and output blocks still use random hashes.
            if let Some(first_workers) = inner.block_workers.get(&query[0]) {
                let mut active = first_workers.clone();

                for (idx, hash) in query.iter().enumerate().skip(1) {
                    let Some(workers) = inner.block_workers.get(hash) else {
                        for worker in active.drain() {
                            matched_depth.insert(worker, idx);
                        }
                        break;
                    };

                    active.retain(|worker| {
                        if workers.contains(worker) {
                            true
                        } else {
                            matched_depth.insert(*worker, idx);
                            false
                        }
                    });

                    if active.is_empty() {
                        break;
                    }
                }

                for worker in active {
                    matched_depth.insert(worker, query.len());
                }
            }
        }

        let query_len = token_sequence.map_or(0, |query| query.len());
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

    #[cfg(test)]
    pub(super) fn is_block_index_empty(&self) -> bool {
        self.inner.read().block_workers.is_empty()
    }
}
