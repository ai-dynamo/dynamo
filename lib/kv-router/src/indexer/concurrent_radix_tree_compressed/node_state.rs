// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::RemoveOutcome;
use crate::protocols::*;

#[derive(Debug)]
pub(super) struct NodeState {
    /// Compressed edge: sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
    /// Empty for the root node; non-empty for all other nodes.
    pub(super) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` -> position in `edge`.
    pub(super) edge_index: FxHashMap<ExternalSequenceBlockHash, usize>,
    /// Workers with partial edge coverage. `worker_cutoffs[w] = k` means worker `w`
    /// has cached `edge[0..k]`, where `0 < k < edge.len()`.
    pub(super) worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    /// Workers with full edge coverage (match index == edge.len()).
    pub(super) full_edge_workers: FxHashSet<WorkerWithDpRank>,
}

impl NodeState {
    pub(super) fn edge_index_for(
        edge: &[(LocalBlockHash, ExternalSequenceBlockHash)],
    ) -> FxHashMap<ExternalSequenceBlockHash, usize> {
        let mut edge_index = FxHashMap::with_capacity_and_hasher(edge.len(), FxBuildHasher);
        for (i, &(_, hash)) in edge.iter().enumerate() {
            edge_index.insert(hash, i);
        }
        edge_index
    }

    #[inline]
    pub(super) fn current_cutoff(&self, worker: WorkerWithDpRank) -> usize {
        if self.full_edge_workers.contains(&worker) {
            self.edge.len()
        } else {
            self.worker_cutoffs.get(&worker).copied().unwrap_or(0)
        }
    }

    #[inline]
    pub(super) fn covers_pos(&self, worker: WorkerWithDpRank, pos: usize) -> bool {
        self.full_edge_workers.contains(&worker)
            || matches!(self.worker_cutoffs.get(&worker), Some(&cutoff) if pos < cutoff)
    }

    fn uncovered_suffix_hashes(&self, cutoff: usize) -> Vec<ExternalSequenceBlockHash> {
        debug_assert!(cutoff <= self.edge.len());
        self.edge[cutoff..].iter().map(|&(_, hash)| hash).collect()
    }

    #[inline]
    pub(super) fn drop_worker(&mut self, worker: WorkerWithDpRank) {
        self.full_edge_workers.remove(&worker);
        self.worker_cutoffs.remove(&worker);
    }

    #[inline]
    pub(super) fn promote_to_full(&mut self, worker: WorkerWithDpRank) -> bool {
        if !self.full_edge_workers.contains(&worker) {
            self.worker_cutoffs.remove(&worker);
            self.full_edge_workers.insert(worker);
            true
        } else {
            false
        }
    }

    pub(super) fn cover_prefix_for_worker(
        &mut self,
        worker: WorkerWithDpRank,
        cutoff: usize,
    ) -> bool {
        debug_assert!(cutoff <= self.edge.len());
        if cutoff == 0 {
            return false;
        }
        if cutoff >= self.edge.len() {
            return self.promote_to_full(worker);
        }
        if self.full_edge_workers.contains(&worker) {
            return false;
        }

        match self.worker_cutoffs.get_mut(&worker) {
            Some(existing) if *existing >= cutoff => false,
            Some(existing) => {
                *existing = cutoff;
                true
            }
            None => {
                self.worker_cutoffs.insert(worker, cutoff);
                true
            }
        }
    }

    pub(super) fn tail_hash_is(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.edge
            .last()
            .is_some_and(|&(_, edge_hash)| edge_hash == hash)
    }

    pub(super) fn suffix_matches_store(
        &self,
        parent_pos: usize,
        blocks: &[KvCacheStoredBlockData],
    ) -> bool {
        let Some(suffix) = self.edge.get(parent_pos + 1..) else {
            return false;
        };
        if blocks.len() > suffix.len() {
            return false;
        }
        for (&(local_hash, block_hash), block) in suffix.iter().zip(blocks) {
            if local_hash != block.tokens_hash || block_hash != block.block_hash {
                return false;
            }
        }

        true
    }

    pub(super) fn store_starts_with_suffix(
        &self,
        parent_pos: usize,
        blocks: &[KvCacheStoredBlockData],
    ) -> Option<usize> {
        let suffix = self.edge.get(parent_pos + 1..)?;
        if blocks.len() <= suffix.len() {
            return None;
        }
        for (&(local_hash, block_hash), block) in suffix.iter().zip(blocks) {
            if local_hash != block.tokens_hash || block_hash != block.block_hash {
                return None;
            }
        }

        Some(suffix.len())
    }

    pub(super) fn append_blocks_to_leaf(
        &mut self,
        worker: WorkerWithDpRank,
        blocks: &[KvCacheStoredBlockData],
    ) {
        debug_assert!(!blocks.is_empty());

        let old_len = self.edge.len();
        let downgraded_workers: Vec<WorkerWithDpRank> = self
            .full_edge_workers
            .iter()
            .copied()
            .filter(|&full_worker| full_worker != worker)
            .collect();
        for downgraded_worker in downgraded_workers {
            self.full_edge_workers.remove(&downgraded_worker);
            self.worker_cutoffs.insert(downgraded_worker, old_len);
        }
        self.promote_to_full(worker);

        self.edge.reserve(blocks.len());
        self.edge_index.reserve(blocks.len());
        for (offset, block) in blocks.iter().enumerate() {
            self.edge.push((block.tokens_hash, block.block_hash));
            self.edge_index.insert(block.block_hash, old_len + offset);
        }
    }

    pub(super) fn remove_worker_at_pos(
        &mut self,
        worker: WorkerWithDpRank,
        pos: usize,
        removed_hash: ExternalSequenceBlockHash,
    ) -> RemoveOutcome {
        let current_cutoff = self.current_cutoff(worker);
        if pos >= current_cutoff {
            return RemoveOutcome {
                stale_hashes: vec![removed_hash],
            };
        }

        let new_cutoff = pos;
        let stale_hashes = self.uncovered_suffix_hashes(new_cutoff);

        if new_cutoff == 0 {
            self.drop_worker(worker);
        } else {
            self.full_edge_workers.remove(&worker);
            self.worker_cutoffs.insert(worker, new_cutoff);
        }

        RemoveOutcome { stale_hashes }
    }

    pub(super) fn has_any_workers(&self) -> bool {
        !self.full_edge_workers.is_empty() || !self.worker_cutoffs.is_empty()
    }
}
