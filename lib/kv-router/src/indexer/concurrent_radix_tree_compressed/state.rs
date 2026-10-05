// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Edge and partial-coverage state of one node, kept behind the node's state lock.

use super::coverage::{FullCoverage, Slot};
use super::edge_index::EdgeIndex;
use crate::protocols::*;

pub(super) struct RemoveOutcome {
    pub(super) stale_hashes: Vec<ExternalSequenceBlockHash>,
}

/// Ranks that cover only a prefix of the edge, sorted by slot. `(slot, k)` means the rank
/// cached `edge[0..k]`, with `0 < k < edge.len()`.
#[derive(Debug, Default)]
pub(super) struct SlotCutoffs(Vec<(Slot, u32)>);

impl SlotCutoffs {
    fn position(&self, slot: Slot) -> Result<usize, usize> {
        self.0.binary_search_by_key(&slot, |&(entry, _)| entry)
    }

    #[inline]
    pub(super) fn get(&self, slot: Slot) -> Option<usize> {
        self.position(slot)
            .ok()
            .map(|index| self.0[index].1 as usize)
    }

    pub(super) fn insert(&mut self, slot: Slot, cutoff: usize) {
        debug_assert!(cutoff > 0);
        let cutoff = u32::try_from(cutoff).expect("edge positions fit in u32");
        match self.position(slot) {
            Ok(index) => self.0[index].1 = cutoff,
            Err(index) => self.0.insert(index, (slot, cutoff)),
        }
    }

    pub(super) fn remove(&mut self, slot: Slot) -> Option<usize> {
        let index = self.position(slot).ok()?;
        Some(self.0.remove(index).1 as usize)
    }

    /// Appends a slot larger than every slot already present.
    fn push(&mut self, slot: Slot, cutoff: usize) {
        debug_assert!(self.0.last().is_none_or(|&(last, _)| last < slot));
        self.0.push((slot, cutoff as u32));
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(Slot) -> bool) {
        self.0.retain(|&(slot, _)| keep(slot));
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (Slot, usize)> + '_ {
        self.0.iter().map(|&(slot, cutoff)| (slot, cutoff as usize))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

/// A node's compressed edge and partial coverage. Full-edge coverage is the node's
/// [`FullCoverage`], passed in by the caller; a slot is in at most one of them, except
/// while a promote that already set its bit deletes its cutoff.
#[derive(Debug)]
pub(super) struct CrtcNodeState {
    /// Compressed edge: sequence of `(LocalBlockHash, ExternalSequenceBlockHash)` pairs.
    /// Empty for the root node; non-empty for all other nodes. Changed only by methods
    /// that keep `edge_index` in step.
    pub(super) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` -> position in `edge`.
    edge_index: EdgeIndex,
    pub(super) cutoffs: SlotCutoffs,
}

impl CrtcNodeState {
    pub(super) fn new(edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>) -> Self {
        Self {
            edge_index: EdgeIndex::build(&edge),
            edge,
            cutoffs: SlotCutoffs::default(),
        }
    }

    pub(super) fn for_blocks(blocks: &[KvCacheStoredBlockData]) -> Self {
        Self::new(
            blocks
                .iter()
                .map(|block| (block.tokens_hash, block.block_hash))
                .collect(),
        )
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

    #[inline]
    pub(super) fn current_cutoff(&self, full: &FullCoverage, slot: Slot) -> usize {
        if full.contains(slot) {
            self.edge.len()
        } else {
            self.cutoffs.get(slot).unwrap_or(0)
        }
    }

    #[inline]
    pub(super) fn covers_pos(&self, full: &FullCoverage, slot: Slot, pos: usize) -> bool {
        full.contains(slot) || self.cutoffs.get(slot).is_some_and(|cutoff| pos < cutoff)
    }

    pub(super) fn has_any_workers(&self, full: &FullCoverage) -> bool {
        !self.cutoffs.is_empty() || !full.is_empty()
    }

    /// Sets the bit before dropping the cutoff, so the slot is never uncovered between.
    pub(super) fn promote_to_full(&mut self, full: &FullCoverage, slot: Slot) -> bool {
        if !full.insert(slot) {
            return false;
        }
        self.cutoffs.remove(slot);
        true
    }

    pub(super) fn cover_prefix(&mut self, full: &FullCoverage, slot: Slot, cutoff: usize) -> bool {
        debug_assert!(cutoff <= self.edge.len());
        if cutoff == 0 {
            return false;
        }
        if cutoff >= self.edge.len() {
            return self.promote_to_full(full, slot);
        }
        if full.contains(slot)
            || self
                .cutoffs
                .get(slot)
                .is_some_and(|existing| existing >= cutoff)
        {
            return false;
        }
        self.cutoffs.insert(slot, cutoff);
        true
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

    /// Appends `blocks` for `slot`. Every other full rank keeps only the old edge: its
    /// cutoff is published before its bit is cleared.
    pub(super) fn append_blocks_to_leaf(
        &mut self,
        full: &FullCoverage,
        slot: Slot,
        blocks: &[KvCacheStoredBlockData],
    ) {
        debug_assert!(!blocks.is_empty());

        let old_len = self.edge.len();
        full.for_each(|other| {
            if other != slot {
                self.cutoffs.insert(other, old_len);
                full.remove(other);
            }
        });
        self.promote_to_full(full, slot);

        self.edge.extend(
            blocks
                .iter()
                .map(|block| (block.tokens_hash, block.block_hash)),
        );
        self.edge_index.extend(&self.edge, old_len);
    }

    /// Splits off `edge[pos..]` for a suffix node and moves partial coverage that reaches
    /// the split point: those ranks become full on this prefix, and keep any remainder as
    /// a suffix cutoff. Returns the suffix state; its full coverage is this node's full
    /// coverage before the split, which the caller snapshots first.
    pub(super) fn split_off_suffix(&mut self, full: &FullCoverage, pos: usize) -> Self {
        debug_assert!(
            pos > 0 && pos < self.edge.len(),
            "split position {pos} out of range for edge length {}",
            self.edge.len()
        );

        let suffix_edge = self.edge.split_off(pos);
        self.edge_index.truncate(&self.edge);

        let mut suffix_cutoffs = SlotCutoffs::default();
        let mut promoted = Vec::new();
        for (slot, cutoff) in self.cutoffs.iter() {
            if cutoff < pos {
                continue;
            }
            promoted.push(slot);
            if cutoff > pos {
                suffix_cutoffs.push(slot, cutoff - pos);
            }
        }
        for slot in promoted {
            self.promote_to_full(full, slot);
        }

        Self {
            edge_index: EdgeIndex::build(&suffix_edge),
            edge: suffix_edge,
            cutoffs: suffix_cutoffs,
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

    /// Drops a full `slot` from the whole edge. Only its bit changes, so callers that
    /// hold the exclusive shape gate need no state write lock.
    pub(super) fn drop_full_slot(&self, full: &FullCoverage, slot: Slot) -> RemoveOutcome {
        debug_assert!(self.cutoffs.get(slot).is_none());
        let removed = full.remove(slot);
        debug_assert!(removed);
        RemoveOutcome {
            stale_hashes: self.newly_uncovered_hashes(0, self.edge.len()),
        }
    }

    /// Cuts `slot`'s coverage to `edge[..pos]`. A shortened full rank publishes its cutoff
    /// before its bit is cleared.
    pub(super) fn remove_worker_at_pos(
        &mut self,
        full: &FullCoverage,
        slot: Slot,
        pos: usize,
        removed_hash: ExternalSequenceBlockHash,
    ) -> RemoveOutcome {
        let current_cutoff = self.current_cutoff(full, slot);
        if pos >= current_cutoff {
            return RemoveOutcome {
                stale_hashes: vec![removed_hash],
            };
        }

        let stale_hashes = self.newly_uncovered_hashes(pos, current_cutoff);
        if pos == 0 {
            self.cutoffs.remove(slot);
        } else {
            self.cutoffs.insert(slot, pos);
        }
        full.remove(slot);

        RemoveOutcome { stale_hashes }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::edge_index::SCAN_MAX_LEN;
    use super::*;

    fn block(hash: u64) -> KvCacheStoredBlockData {
        KvCacheStoredBlockData {
            tokens_hash: LocalBlockHash(hash),
            block_hash: ExternalSequenceBlockHash(hash),
            mm_extra_info: None,
        }
    }

    #[test]
    fn leaf_to_root_removes_return_only_newly_uncovered_hashes() {
        let slot = Slot::new(3);
        let blocks = [block(1), block(2), block(3), block(4)];
        let full = FullCoverage::single(slot);
        let mut state = CrtcNodeState::for_blocks(&blocks);

        for hash in (1..=4).rev() {
            let hash = ExternalSequenceBlockHash(hash);
            let pos = state.position(hash).unwrap();
            let outcome = state.remove_worker_at_pos(&full, slot, pos, hash);
            assert_eq!(outcome.stale_hashes, vec![hash]);
            assert_eq!(state.current_cutoff(&full, slot), pos);
        }
        assert!(!state.has_any_workers(&full));
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
        let slot = Slot::new(1);
        let full = FullCoverage::single(slot);
        let long: Vec<u64> = std::iter::once(1)
            .chain(100..100 + SCAN_MAX_LEN as u64)
            .chain([1])
            .collect();
        for edge in [vec![1, 2, 3, 1], long] {
            let (head, tail) = edge.split_at(edge.len() - 1);
            for state in [
                CrtcNodeState::for_blocks(&blocks(&edge)),
                replay(&[head.to_vec(), tail.to_vec()], &full, slot),
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
        let mut state = CrtcNodeState::for_blocks(&blocks(&edge));
        let _suffix = state.split_off_suffix(&full, split);
        assert!(state.edge_index.capacity() > 0);
        assert_position_near_matches_index(&state);
    }

    #[test]
    fn extension_demotes_other_full_ranks_to_the_old_tail() {
        let extender = Slot::new(1);
        let other = Slot::new(300);
        let partial = Slot::new(2);
        let full = FullCoverage::single(extender);
        full.insert(other);
        let mut state = CrtcNodeState::for_blocks(&[block(1), block(2), block(3)]);
        state.cutoffs.insert(partial, 1);

        state.append_blocks_to_leaf(&full, extender, &[block(4), block(5)]);

        assert_eq!(state.edge.len(), 5);
        assert_eq!(state.position(ExternalSequenceBlockHash(5)), Some(4));
        assert_eq!(state.current_cutoff(&full, extender), 5);
        assert_eq!(state.current_cutoff(&full, other), 3);
        assert_eq!(state.current_cutoff(&full, partial), 1);
        assert_eq!(full.count(), 1);
    }

    #[test]
    fn extension_promotes_a_partial_extender() {
        let extender = Slot::new(1);
        let full = FullCoverage::default();
        let mut state = CrtcNodeState::for_blocks(&[block(1), block(2), block(3)]);
        state.cutoffs.insert(extender, 2);

        state.append_blocks_to_leaf(&full, extender, &[block(4)]);

        assert!(full.contains(extender));
        assert!(state.cutoffs.is_empty());
        assert_eq!(state.current_cutoff(&full, extender), 4);
    }

    fn blocks(hashes: &[u64]) -> Vec<KvCacheStoredBlockData> {
        hashes.iter().map(|&hash| block(hash)).collect()
    }

    /// Builds a node state from `chunks[0]` and appends each later chunk as a leaf extension.
    fn replay(chunks: &[Vec<u64>], full: &FullCoverage, slot: Slot) -> CrtcNodeState {
        let mut state = CrtcNodeState::for_blocks(&blocks(&chunks[0]));
        for chunk in &chunks[1..] {
            state.append_blocks_to_leaf(full, slot, &blocks(chunk));
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
        full: &FullCoverage,
        mut pick: impl FnMut(usize) -> usize,
    ) {
        let mut len = prefix.edge.len();
        while len > 1 {
            let at = pick(len);
            let inner = prefix.split_off_suffix(full, at);
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
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        fn draw(next: &mut impl FnMut() -> u64) -> u64 {
            if next().is_multiple_of(8) {
                1 + next() % 6
            } else {
                next()
            }
        }
        let slot = Slot::new(7);
        let full = FullCoverage::single(slot);

        for _ in 0..300 {
            let max_build = if next().is_multiple_of(4) {
                12 * SCAN_MAX_LEN
            } else {
                2 * SCAN_MAX_LEN + 8
            };
            let mut chunks = vec![
                (0..1 + next() as usize % max_build)
                    .map(|_| draw(&mut next))
                    .collect::<Vec<_>>(),
            ];
            for _ in 0..next() % 7 {
                let len = if next().is_multiple_of(2) {
                    1
                } else {
                    1 + next() as usize % 12
                };
                chunks.push((0..len).map(|_| draw(&mut next)).collect());
            }
            let edge: Vec<u64> = chunks.concat();
            let misses: Vec<u64> = (0..8).map(|_| next()).collect();
            let probes: Vec<u64> = edge.iter().copied().chain(misses).collect();

            let mut state = CrtcNodeState::for_blocks(&blocks(&chunks[0]));
            let mut model = HashMap::new();
            let mut len = 0;
            for (i, chunk) in chunks.iter().enumerate() {
                if i > 0 {
                    state.append_blocks_to_leaf(&full, slot, &blocks(chunk));
                }
                for &hash in chunk {
                    model.insert(hash, len);
                    len += 1;
                }
                assert_index_matches(&state, &model, &probes, 1);
            }

            for split in 1..edge.len() {
                let mut prefix = replay(&chunks, &full, slot);
                let mut suffix = prefix.split_off_suffix(&full, split);

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

                let tail: Vec<u64> = (0..1 + next() % 20).map(|_| draw(&mut next)).collect();
                let suffix_len = suffix.edge.len();
                suffix.append_blocks_to_leaf(&full, slot, &blocks(&tail));
                for (offset, &hash) in tail.iter().enumerate() {
                    suffix_model.insert(hash, suffix_len + offset);
                }
                let suffix_probes: Vec<u64> = probes.iter().chain(&tail).copied().collect();
                assert_index_matches(&suffix, &suffix_model, &suffix_probes, 1);

                split_down(prefix, &edge, &probes, &full, |len| {
                    1 + next() as usize % (len - 1)
                });
            }

            // Peel a few blocks off the tail at a time, as stores that diverge late do.
            split_down(replay(&chunks, &full, slot), &edge, &probes, &full, |len| {
                len - 1 - next() as usize % (len - 1).min(3)
            });
        }
    }
}
