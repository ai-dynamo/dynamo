// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::protocols::*;

pub(crate) fn append_dump_events(
    events: &mut Vec<RouterEvent>,
    event_id: &mut u64,
    parent_hash: Option<ExternalSequenceBlockHash>,
    edge: &[(LocalBlockHash, ExternalSequenceBlockHash)],
    full_workers: &[WorkerWithDpRank],
    worker_cutoffs: &[(WorkerWithDpRank, usize)],
) {
    let blocks = edge
        .iter()
        .map(|&(tokens_hash, block_hash)| KvCacheStoredBlockData {
            block_hash,
            tokens_hash,
            mm_extra_info: None,
        })
        .collect::<Vec<_>>();

    for &worker in full_workers {
        events.push(dump_event(worker, *event_id, parent_hash, blocks.clone()));
        *event_id += 1;
    }
    for &(worker, cutoff) in worker_cutoffs {
        events.push(dump_event(
            worker,
            *event_id,
            parent_hash,
            blocks[..cutoff].to_vec(),
        ));
        *event_id += 1;
    }
}

fn dump_event(
    worker: WorkerWithDpRank,
    event_id: u64,
    parent_hash: Option<ExternalSequenceBlockHash>,
    blocks: Vec<KvCacheStoredBlockData>,
) -> RouterEvent {
    RouterEvent::new(
        worker.worker_id,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                start_position: None,
                blocks,
            }),
            dp_rank: worker.dp_rank,
        },
    )
}

pub(crate) struct RemoveOutcome {
    pub(crate) stale_hashes: Vec<ExternalSequenceBlockHash>,
}

#[derive(Debug)]
pub(crate) struct NodeState {
    /// Compressed edge: sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
    /// Empty for the root node; non-empty for all other nodes.
    pub(crate) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` -> position in `edge`.
    pub(crate) edge_index: FxHashMap<ExternalSequenceBlockHash, usize>,
    /// Workers with partial edge coverage. `worker_cutoffs[w] = k` means worker `w`
    /// has cached `edge[0..k]`, where `0 < k < edge.len()`.
    pub(crate) worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    /// Workers with full edge coverage.
    pub(crate) full_edge_workers: FxHashSet<WorkerWithDpRank>,
}

impl NodeState {
    pub(crate) fn empty() -> Self {
        Self {
            edge: Vec::new(),
            edge_index: FxHashMap::default(),
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers: FxHashSet::default(),
        }
    }

    pub(crate) fn for_blocks(blocks: &[KvCacheStoredBlockData], worker: WorkerWithDpRank) -> Self {
        let edge = blocks
            .iter()
            .map(|block| (block.tokens_hash, block.block_hash))
            .collect::<Vec<_>>();
        let mut full_edge_workers = FxHashSet::with_capacity_and_hasher(1, FxBuildHasher);
        full_edge_workers.insert(worker);

        Self {
            edge_index: Self::edge_index_for(&edge),
            edge,
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers,
        }
    }

    pub(crate) fn edge_index_for(
        edge: &[(LocalBlockHash, ExternalSequenceBlockHash)],
    ) -> FxHashMap<ExternalSequenceBlockHash, usize> {
        let mut edge_index = FxHashMap::with_capacity_and_hasher(edge.len(), FxBuildHasher);
        for (i, &(_, hash)) in edge.iter().enumerate() {
            edge_index.insert(hash, i);
        }
        edge_index
    }

    #[inline]
    pub(crate) fn current_cutoff(&self, worker: WorkerWithDpRank) -> usize {
        if self.full_edge_workers.contains(&worker) {
            self.edge.len()
        } else {
            self.worker_cutoffs.get(&worker).copied().unwrap_or(0)
        }
    }

    #[inline]
    pub(crate) fn covers_pos(&self, worker: WorkerWithDpRank, pos: usize) -> bool {
        self.full_edge_workers.contains(&worker)
            || matches!(self.worker_cutoffs.get(&worker), Some(&cutoff) if pos < cutoff)
    }

    fn uncovered_suffix_hashes(&self, cutoff: usize) -> Vec<ExternalSequenceBlockHash> {
        debug_assert!(cutoff <= self.edge.len());
        self.edge[cutoff..].iter().map(|&(_, hash)| hash).collect()
    }

    #[inline]
    pub(crate) fn drop_worker(&mut self, worker: WorkerWithDpRank) {
        self.full_edge_workers.remove(&worker);
        self.worker_cutoffs.remove(&worker);
    }

    #[inline]
    pub(crate) fn promote_to_full(&mut self, worker: WorkerWithDpRank) -> bool {
        if self.full_edge_workers.contains(&worker) {
            return false;
        }

        self.worker_cutoffs.remove(&worker);
        self.full_edge_workers.insert(worker);
        true
    }

    pub(crate) fn cover_prefix_for_worker(
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

    pub(crate) fn tail_hash_is(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.edge
            .last()
            .is_some_and(|&(_, edge_hash)| edge_hash == hash)
    }

    pub(crate) fn suffix_matches_store(
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

        suffix
            .iter()
            .zip(blocks)
            .all(|(&(local_hash, block_hash), block)| {
                local_hash == block.tokens_hash && block_hash == block.block_hash
            })
    }

    pub(crate) fn store_starts_with_suffix(
        &self,
        parent_pos: usize,
        blocks: &[KvCacheStoredBlockData],
    ) -> Option<usize> {
        let suffix = self.edge.get(parent_pos + 1..)?;
        if blocks.len() <= suffix.len() {
            return None;
        }
        if !suffix
            .iter()
            .zip(blocks)
            .all(|(&(local_hash, block_hash), block)| {
                local_hash == block.tokens_hash && block_hash == block.block_hash
            })
        {
            return None;
        }

        Some(suffix.len())
    }

    pub(crate) fn append_blocks_to_leaf(
        &mut self,
        worker: WorkerWithDpRank,
        blocks: &[KvCacheStoredBlockData],
    ) {
        debug_assert!(!blocks.is_empty());

        let old_len = self.edge.len();
        let downgraded_workers = self
            .full_edge_workers
            .iter()
            .copied()
            .filter(|&full_worker| full_worker != worker)
            .collect::<Vec<_>>();
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

    pub(crate) fn remove_worker_at_pos(
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

    pub(crate) fn has_any_workers(&self) -> bool {
        !self.full_edge_workers.is_empty() || !self.worker_cutoffs.is_empty()
    }
}
