// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Edge and coverage state of one node, kept behind the node's state lock.

use rustc_hash::{FxBuildHasher, FxHashMap};

use super::edge_index::EdgeIndex;
use super::worker_set::WorkerSet;
use crate::protocols::*;

pub(super) struct RemoveOutcome {
    pub(super) stale_hashes: Vec<ExternalSequenceBlockHash>,
}

/// A node's compressed edge, its position index, and which workers cover how much of it.
#[derive(Debug)]
pub(super) struct CrtcNodeState {
    /// Compressed edge: sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
    /// Empty for the root node; non-empty for all other nodes. Changed only by methods
    /// that keep `edge_index` in step.
    pub(super) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` -> position in `edge`.
    edge_index: EdgeIndex,
    /// Workers with partial edge coverage. `worker_cutoffs[w] = k` means worker `w`
    /// has cached `edge[0..k]`, where `0 < k < edge.len()`.
    pub(super) worker_cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    /// Workers with full edge coverage.
    pub(super) full_edge_workers: WorkerSet,
    /// Hash of the block just before `edge[0]`, or 0 for root children. The
    /// head never moves, so however often the tree splits, the parent is the node whose
    /// edge ends with this block.
    pub(super) parent_tail: ExternalSequenceBlockHash,
}

impl CrtcNodeState {
    /// An edge no worker covers yet.
    pub(super) fn new(edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>) -> Self {
        Self {
            edge_index: EdgeIndex::build(&edge),
            edge,
            worker_cutoffs: FxHashMap::default(),
            full_edge_workers: WorkerSet::default(),
            parent_tail: ExternalSequenceBlockHash(0),
        }
    }

    /// An edge of `blocks` that `worker` fully covers.
    pub(super) fn for_blocks(blocks: &[KvCacheStoredBlockData], worker: WorkerWithDpRank) -> Self {
        let mut state = Self::new(
            blocks
                .iter()
                .map(|block| (block.tokens_hash, block.block_hash))
                .collect(),
        );
        state.full_edge_workers.insert(worker);
        state
    }

    /// The position of `hash` in the edge; its last one if the edge repeats it.
    #[inline]
    pub(super) fn position(&self, hash: ExternalSequenceBlockHash) -> Option<usize> {
        self.edge_index.position(&self.edge, hash)
    }

    #[inline]
    pub(super) fn contains_hash(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.position(hash).is_some()
    }

    /// Position of `hash` in the edge, checking the neighbors of `near` before the index:
    /// a removal run usually lists adjacent blocks, so most of its lookups skip the index.
    /// The shortcut runs only on a table-backed edge whose table saw no repeated hash, where a
    /// neighbor match is the position; otherwise the index finds the last copy.
    pub(super) fn position_near(
        &self,
        hash: ExternalSequenceBlockHash,
        near: Option<usize>,
    ) -> Option<usize> {
        if let Some(near) = near
            && self.edge_index.neighbors_are_unique()
        {
            for pos in [near + 1, near.wrapping_sub(1)] {
                if self
                    .edge
                    .get(pos)
                    .is_some_and(|&(_, edge_hash)| edge_hash == hash)
                {
                    debug_assert_eq!(self.position(hash), Some(pos));
                    return Some(pos);
                }
            }
        }
        self.position(hash)
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

    pub(super) fn has_any_workers(&self) -> bool {
        !self.full_edge_workers.is_empty() || !self.worker_cutoffs.is_empty()
    }

    #[inline]
    pub(super) fn promote_to_full(&mut self, worker: WorkerWithDpRank) -> bool {
        if self.full_edge_workers.contains(&worker) {
            return false;
        }

        self.worker_cutoffs.remove(&worker);
        self.full_edge_workers.insert(worker);
        true
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

        suffix
            .iter()
            .zip(blocks)
            .all(|(&(local_hash, block_hash), block)| {
                local_hash == block.tokens_hash && block_hash == block.block_hash
            })
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

    /// Appends `blocks` for `worker`. Every other full worker keeps only the old edge.
    pub(super) fn append_blocks_to_leaf(
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

        self.edge.extend(
            blocks
                .iter()
                .map(|block| (block.tokens_hash, block.block_hash)),
        );
        self.edge_index.extend(&self.edge, old_len);
    }

    /// Splits off `edge[pos..]` for a suffix node and moves partial coverage that reaches
    /// the split point: those workers become full on this prefix, and keep any remainder
    /// as a suffix cutoff. The suffix's full coverage is this node's full coverage before
    /// the split.
    pub(super) fn split_off_suffix(&mut self, pos: usize) -> Self {
        debug_assert!(
            pos > 0 && pos < self.edge.len(),
            "split position {pos} out of range for edge length {}",
            self.edge.len()
        );

        let suffix_edge = self.edge.split_off(pos);
        self.edge_index.truncate(&self.edge);

        let suffix_full = self.full_edge_workers.clone();
        let mut suffix_cutoffs =
            FxHashMap::with_capacity_and_hasher(self.worker_cutoffs.len(), FxBuildHasher);
        let mut to_promote: Vec<WorkerWithDpRank> = Vec::new();
        for (&worker, &cutoff) in &self.worker_cutoffs {
            if cutoff >= pos {
                to_promote.push(worker);
                let suffix_cutoff = cutoff - pos;
                if suffix_cutoff > 0 {
                    suffix_cutoffs.insert(worker, suffix_cutoff);
                }
            }
        }
        for worker in to_promote {
            self.worker_cutoffs.remove(&worker);
            self.full_edge_workers.insert(worker);
        }

        Self {
            edge_index: EdgeIndex::build(&suffix_edge),
            edge: suffix_edge,
            worker_cutoffs: suffix_cutoffs,
            full_edge_workers: suffix_full,
            // The prefix now ends with the block before the suffix's head.
            parent_tail: self.edge[pos - 1].1,
        }
    }

    fn newly_uncovered_hashes(
        &self,
        new_cutoff: usize,
        old_cutoff: usize,
    ) -> Vec<ExternalSequenceBlockHash> {
        debug_assert!(new_cutoff <= old_cutoff);
        debug_assert!(old_cutoff <= self.edge.len());
        self.edge[new_cutoff..old_cutoff]
            .iter()
            .map(|&(_, hash)| hash)
            .collect()
    }

    /// Cuts `worker`'s coverage to `edge[..pos]`.
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

        let stale_hashes = self.newly_uncovered_hashes(pos, current_cutoff);
        self.full_edge_workers.remove(&worker);
        if pos == 0 {
            self.worker_cutoffs.remove(&worker);
        } else {
            self.worker_cutoffs.insert(worker, pos);
        }

        RemoveOutcome { stale_hashes }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rustc_hash::FxHashSet;

    use super::super::edge_index::SCAN_MAX_LEN;
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn block(hash: u64) -> KvCacheStoredBlockData {
        KvCacheStoredBlockData {
            tokens_hash: LocalBlockHash(hash),
            block_hash: ExternalSequenceBlockHash(hash),
            mm_extra_info: None,
        }
    }

    fn blocks(hashes: &[u64]) -> Vec<KvCacheStoredBlockData> {
        hashes.iter().map(|&hash| block(hash)).collect()
    }

    #[test]
    fn leaf_to_root_removes_return_only_newly_uncovered_hashes() {
        let worker = worker(3);
        let mut state = CrtcNodeState::for_blocks(&blocks(&[1, 2, 3, 4]), worker);

        for hash in (1..=4).rev() {
            let hash = ExternalSequenceBlockHash(hash);
            let pos = state.position(hash).unwrap();
            let outcome = state.remove_worker_at_pos(worker, pos, hash);
            assert_eq!(outcome.stale_hashes, vec![hash]);
            assert_eq!(state.current_cutoff(worker), pos);
        }
        assert!(!state.has_any_workers());
    }

    fn assert_position_near_matches_index(state: &CrtcNodeState) {
        let hints = std::iter::once(None).chain((0..state.edge.len() + 1).map(Some));
        for near in hints {
            for &(_, hash) in &state.edge {
                assert_eq!(
                    state.position_near(hash, near),
                    state.position(hash),
                    "{hash:?} near {near:?}"
                );
            }
            assert_eq!(
                state.position_near(ExternalSequenceBlockHash(42), near),
                None
            );
        }
    }

    /// A hint next to an earlier copy of a repeated hash must still find the last copy,
    /// whether the edge scans or keeps a table, and whether it was built or appended to.
    #[test]
    fn position_near_finds_the_last_copy_of_a_repeated_hash() {
        let worker = worker(1);
        let long: Vec<u64> = std::iter::once(1)
            .chain(100..100 + SCAN_MAX_LEN as u64)
            .chain([1])
            .collect();
        for edge in [vec![1, 2, 3, 1], long] {
            let (head, tail) = edge.split_at(edge.len() - 1);
            for state in [
                CrtcNodeState::for_blocks(&blocks(&edge), worker),
                replay(&[head.to_vec(), tail.to_vec()], worker),
            ] {
                assert_eq!(state.edge_index.capacity() > 0, edge.len() > SCAN_MAX_LEN);
                assert_eq!(
                    state.position(ExternalSequenceBlockHash(1)),
                    Some(edge.len() - 1)
                );
                assert_position_near_matches_index(&state);
            }
        }

        // A split that keeps the prefix's table must keep remembering its repeat.
        let mut edge: Vec<u64> = std::iter::once(1)
            .chain(100..100 + SCAN_MAX_LEN as u64)
            .chain([1])
            .collect();
        let split = edge.len();
        edge.extend([200, 201]);
        let mut state = CrtcNodeState::for_blocks(&blocks(&edge), worker);
        let _suffix = state.split_off_suffix(split);
        assert!(state.edge_index.capacity() > 0);
        assert_position_near_matches_index(&state);
    }

    #[test]
    fn extension_demotes_other_full_workers_to_the_old_tail() {
        let extender = worker(1);
        let other = worker(300);
        let partial = worker(2);
        let mut state = CrtcNodeState::for_blocks(&blocks(&[1, 2, 3]), extender);
        state.full_edge_workers.insert(other);
        state.worker_cutoffs.insert(partial, 1);

        state.append_blocks_to_leaf(extender, &blocks(&[4, 5]));

        assert_eq!(state.edge.len(), 5);
        assert_eq!(state.position(ExternalSequenceBlockHash(5)), Some(4));
        assert_eq!(state.current_cutoff(extender), 5);
        assert_eq!(state.current_cutoff(other), 3);
        assert_eq!(state.current_cutoff(partial), 1);
        assert_eq!(state.full_edge_workers.len(), 1);
    }

    #[test]
    fn extension_promotes_a_partial_extender() {
        let extender = worker(1);
        let mut state = CrtcNodeState::new(blocks(&[1, 2, 3]).iter().map(edge_entry).collect());
        state.worker_cutoffs.insert(extender, 2);

        state.append_blocks_to_leaf(extender, &blocks(&[4]));

        assert!(state.full_edge_workers.contains(&extender));
        assert!(state.worker_cutoffs.is_empty());
        assert_eq!(state.current_cutoff(extender), 4);
    }

    /// A split promotes on the prefix every worker whose cutoff reaches the split point,
    /// and the suffix keeps the full workers as they were before that promotion, plus a
    /// cutoff for each promoted worker that reached past the split point.
    #[test]
    fn split_moves_cutoffs_at_and_past_the_boundary() {
        let full = worker(0);
        let short = worker(1);
        let at_split = worker(2);
        let past_split = worker(3);
        let mut state = CrtcNodeState::for_blocks(&blocks(&[1, 2, 3, 4, 5, 6]), full);
        state.worker_cutoffs.insert(short, 1);
        state.worker_cutoffs.insert(at_split, 3);
        state.worker_cutoffs.insert(past_split, 5);

        let suffix = state.split_off_suffix(3);

        assert_eq!(state.edge.len(), 3);
        assert_eq!(
            *state.full_edge_workers,
            FxHashSet::from_iter([full, at_split, past_split])
        );
        assert_eq!(state.worker_cutoffs, FxHashMap::from_iter([(short, 1)]));
        assert_eq!(state.position(ExternalSequenceBlockHash(3)), Some(2));
        assert_eq!(state.position(ExternalSequenceBlockHash(4)), None);

        assert_eq!(suffix.edge.len(), 3);
        assert_eq!(*suffix.full_edge_workers, FxHashSet::from_iter([full]));
        assert_eq!(
            suffix.worker_cutoffs,
            FxHashMap::from_iter([(past_split, 2)])
        );
        assert_eq!(suffix.position(ExternalSequenceBlockHash(4)), Some(0));
        assert_eq!(suffix.position(ExternalSequenceBlockHash(3)), None);
    }

    fn edge_entry(block: &KvCacheStoredBlockData) -> (LocalBlockHash, ExternalSequenceBlockHash) {
        (block.tokens_hash, block.block_hash)
    }

    fn replay(chunks: &[Vec<u64>], worker: WorkerWithDpRank) -> CrtcNodeState {
        let mut state = CrtcNodeState::for_blocks(&blocks(&chunks[0]), worker);
        for chunk in &chunks[1..] {
            state.append_blocks_to_leaf(worker, &blocks(chunk));
        }
        state
    }

    fn last_positions(hashes: &[u64]) -> HashMap<u64, usize> {
        hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos))
            .collect()
    }

    /// Checks every lookup against `model`, and that the table is at most 3/4 full and
    /// at most `slack` times the smallest such table: 1 for built and appended edges, 2
    /// for split prefixes.
    fn assert_index_matches(
        state: &CrtcNodeState,
        model: &HashMap<u64, usize>,
        probes: &[u64],
        slack: usize,
    ) {
        let len = state.edge.len();
        let capacity = state.edge_index.capacity();
        if len <= SCAN_MAX_LEN {
            assert_eq!(capacity, 0, "edge of {len} blocks should scan");
        } else {
            assert!(
                capacity.is_power_of_two()
                    && len * 4 <= capacity * 3
                    && len * 8 * slack > capacity * 3,
                "edge of {len} blocks has {capacity} slots"
            );
        }
        for &hash in model.keys().chain(probes) {
            let expected = model.get(&hash).copied();
            let hash = ExternalSequenceBlockHash(hash);
            assert_eq!(state.position(hash), expected, "{hash:?} in edge of {len}");
            assert_eq!(state.contains_hash(hash), expected.is_some());
        }
    }

    /// Splits `prefix`, which holds `edge[..prefix.edge.len()]`, at the decreasing points
    /// `pick` chooses until one block remains, checking both halves after every split.
    fn split_down(
        mut prefix: CrtcNodeState,
        edge: &[u64],
        probes: &[u64],
        mut pick: impl FnMut(usize) -> usize,
    ) {
        let mut len = prefix.edge.len();
        while len > 1 {
            let at = pick(len);
            let inner = prefix.split_off_suffix(at);
            assert_index_matches(&prefix, &last_positions(&edge[..at]), probes, 2);
            assert_index_matches(&inner, &last_positions(&edge[at..len]), probes, 1);
            len = at;
        }
    }

    /// Differential test of the edge position index against a `HashMap` maintained the way
    /// the old per-node map was: build and append insert in edge order, so a repeated hash
    /// keeps its last position, and a split removes every suffix hash from the prefix. Hashes
    /// sometimes come from a tiny pool to force repeats and shared probe runs, edges straddle
    /// the scan threshold and several table sizes, and each sequence is split at every
    /// position, after which the suffix grows and the prefix keeps splitting at decreasing
    /// points, piling up tombstones until its table shrinks or it falls back to a scan.
    #[test]
    fn edge_positions_match_a_hash_map_model() {
        let mut rng = fastrand::Rng::with_seed(0x9E37_79B9_7F4A_7C15);
        fn draw(rng: &mut fastrand::Rng) -> u64 {
            if rng.u32(..8) == 0 {
                rng.u64(1..=6)
            } else {
                rng.u64(..)
            }
        }
        let worker = worker(7);

        for _ in 0..300 {
            let max_build = if rng.u32(..4) == 0 {
                12 * SCAN_MAX_LEN
            } else {
                2 * SCAN_MAX_LEN + 8
            };
            let mut chunks = vec![
                (0..rng.usize(1..=max_build))
                    .map(|_| draw(&mut rng))
                    .collect::<Vec<_>>(),
            ];
            for _ in 0..rng.u32(..7) {
                let len = if rng.bool() { 1 } else { rng.usize(1..=12) };
                chunks.push((0..len).map(|_| draw(&mut rng)).collect());
            }
            let edge: Vec<u64> = chunks.concat();
            let misses: Vec<u64> = (0..8).map(|_| rng.u64(..)).collect();
            let probes: Vec<u64> = edge.iter().copied().chain(misses).collect();

            let mut state = CrtcNodeState::for_blocks(&blocks(&chunks[0]), worker);
            let mut model = HashMap::new();
            let mut len = 0;
            for (i, chunk) in chunks.iter().enumerate() {
                if i > 0 {
                    state.append_blocks_to_leaf(worker, &blocks(chunk));
                }
                for &hash in chunk {
                    model.insert(hash, len);
                    len += 1;
                }
                assert_index_matches(&state, &model, &probes, 1);
            }

            for split in 1..edge.len() {
                let mut prefix = replay(&chunks, worker);
                let mut suffix = prefix.split_off_suffix(split);

                let mut prefix_model = model.clone();
                for hash in &edge[split..] {
                    prefix_model.remove(hash);
                }
                // The one deliberate difference from the old map: a hash the prefix repeats
                // from the suffix stays indexed at its last prefix position, matching a scan.
                for (pos, &hash) in edge[..split].iter().enumerate().rev() {
                    prefix_model.entry(hash).or_insert(pos);
                }
                assert_index_matches(&prefix, &prefix_model, &probes, 2);

                let mut suffix_model = last_positions(&edge[split..]);
                assert_index_matches(&suffix, &suffix_model, &probes, 1);

                let tail: Vec<u64> = (0..rng.u64(1..=20)).map(|_| draw(&mut rng)).collect();
                let suffix_len = suffix.edge.len();
                suffix.append_blocks_to_leaf(worker, &blocks(&tail));
                for (offset, &hash) in tail.iter().enumerate() {
                    suffix_model.insert(hash, suffix_len + offset);
                }
                let suffix_probes: Vec<u64> = probes.iter().chain(&tail).copied().collect();
                assert_index_matches(&suffix, &suffix_model, &suffix_probes, 1);

                split_down(prefix, &edge, &probes, |len| rng.usize(1..len));
            }

            // Peel a few blocks off the tail at a time, as stores that diverge late do.
            split_down(replay(&chunks, worker), &edge, &probes, |len| {
                len - 1 - rng.usize(..(len - 1).min(3))
            });
        }
    }
}
