// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Edge and partial-coverage state of one node, kept behind the node's state lock.

use rustc_hash::{FxBuildHasher, FxHashMap};

use super::coverage::{FullCoverage, Slot};
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
    /// Empty for the root node; non-empty for all other nodes.
    pub(super) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    /// Reverse index: `ExternalSequenceBlockHash` -> position in `edge`.
    pub(super) edge_index: FxHashMap<ExternalSequenceBlockHash, usize>,
    pub(super) cutoffs: SlotCutoffs,
}

impl CrtcNodeState {
    pub(super) fn new(edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>) -> Self {
        Self {
            edge_index: Self::edge_index_for(&edge),
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

        self.edge.reserve(blocks.len());
        self.edge_index.reserve(blocks.len());
        for (offset, block) in blocks.iter().enumerate() {
            self.edge.push((block.tokens_hash, block.block_hash));
            self.edge_index.insert(block.block_hash, old_len + offset);
        }
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
        for &(_, hash) in &suffix_edge {
            self.edge_index.remove(&hash);
        }

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
            edge_index: Self::edge_index_for(&suffix_edge),
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
            let pos = state.edge_index[&hash];
            let outcome = state.remove_worker_at_pos(&full, slot, pos, hash);
            assert_eq!(outcome.stale_hashes, vec![hash]);
            assert_eq!(state.current_cutoff(&full, slot), pos);
        }
        assert!(!state.has_any_workers(&full));
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
        assert_eq!(state.edge_index[&ExternalSequenceBlockHash(5)], 4);
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
}
