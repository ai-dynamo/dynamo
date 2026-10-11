// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Position index over one node's compressed edge.
//!
//! A hash never changes position inside an edge: edges grow only at the tail, and a split
//! truncates them. So the index stores only `u32` positions and reads keys back through
//! the edge, and short edges keep no index at all.

use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash};

type EdgeEntry = (LocalBlockHash, ExternalSequenceBlockHash);

/// Longest edge that lookups scan instead of indexing. Its 16 entries span four cache
/// lines read in order, while a table probe pays two dependent random reads (slot, then
/// edge entry) and every new node pays an allocation and a hash insert per block. The
/// scan starts at the tail, where store parents and evictions usually land.
pub(super) const SCAN_MAX_LEN: usize = 16;

/// Free table slot. Every stored position is smaller.
const EMPTY: u32 = u32::MAX;

/// Longest edge whose positions fit below [`EMPTY`]. Longer edges scan.
const MAX_INDEXED_LEN: usize = EMPTY as usize;

/// Maps each hash in an edge to its last position in that edge. Every method takes the
/// edge the index describes.
#[derive(Debug, Default)]
pub(super) struct EdgeIndex {
    /// Empty while lookups scan the edge. Otherwise a power-of-two table of edge positions,
    /// probed linearly from a Fibonacci hash of the key and at most 3/4 full. Every
    /// position the table was built or extended with keeps its slot, so a repeated hash
    /// lies along its probe run in descending position order. A split may leave positions
    /// past the edge's end behind as tombstones, which lookups skip.
    slots: Box<[u32]>,
}

impl EdgeIndex {
    pub(super) fn build(edge: &[EdgeEntry]) -> Self {
        let mut index = Self::default();
        index.rebuild(edge);
        index
    }

    /// The smallest table that holds `len` positions at most three quarters full, or zero
    /// if an edge of `len` entries scans.
    fn capacity_for(len: usize) -> usize {
        if len <= SCAN_MAX_LEN || len > MAX_INDEXED_LEN {
            return 0;
        }
        (len * 4).div_ceil(3).next_power_of_two()
    }

    /// Re-indexes all of `edge`, reusing the table when its size still fits.
    fn rebuild(&mut self, edge: &[EdgeEntry]) {
        let capacity = Self::capacity_for(edge.len());
        if capacity == 0 {
            self.slots = Box::default();
            return;
        }
        if self.slots.len() == capacity {
            self.slots.fill(EMPTY);
        } else {
            self.slots = vec![EMPTY; capacity].into_boxed_slice();
        }
        for pos in 0..edge.len() {
            self.insert(edge, pos);
        }
    }

    #[inline]
    fn home(&self, hash: ExternalSequenceBlockHash) -> usize {
        // Sequence hashes are already uniform; Fibonacci hashing takes the high bits.
        let shift = u64::BITS - self.slots.len().trailing_zeros();
        (hash.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize
    }

    /// Records `edge[pos]`, which must be past every indexed position. An earlier
    /// position of the same hash moves one step further along the probe run, so a
    /// lookup still meets the last position first, and meets the one before it if a
    /// split later drops the last.
    fn insert(&mut self, edge: &[EdgeEntry], pos: usize) {
        debug_assert!(pos < MAX_INDEXED_LEN);
        let hash = edge[pos].1;
        let mask = self.slots.len() - 1;
        let mut carried = pos as u32;
        let mut i = self.home(hash);
        loop {
            let slot = self.slots[i];
            if slot == EMPTY {
                self.slots[i] = carried;
                return;
            }
            if edge[slot as usize].1 == hash {
                self.slots[i] = carried;
                carried = slot;
            }
            i = (i + 1) & mask;
        }
    }

    /// The last position of `hash` in `edge`.
    #[inline]
    pub(super) fn position(
        &self,
        edge: &[EdgeEntry],
        hash: ExternalSequenceBlockHash,
    ) -> Option<usize> {
        if self.slots.is_empty() {
            return edge.iter().rposition(|&(_, edge_hash)| edge_hash == hash);
        }
        let mask = self.slots.len() - 1;
        let mut i = self.home(hash);
        loop {
            let slot = self.slots[i];
            if slot == EMPTY {
                return None;
            }
            // A position past the edge's end is a tombstone left by a split.
            if edge
                .get(slot as usize)
                .is_some_and(|&(_, edge_hash)| edge_hash == hash)
            {
                return Some(slot as usize);
            }
            i = (i + 1) & mask;
        }
    }

    /// Indexes `edge[old_len..]` after a tail append.
    pub(super) fn extend(&mut self, edge: &[EdgeEntry], old_len: usize) {
        // Only leaves append, and a split leaves its prefix internal for good, so no
        // tombstone can come back to life as a second slot for a live position.
        debug_assert!(
            self.slots
                .iter()
                .all(|&slot| slot == EMPTY || (slot as usize) < old_len),
            "edge index extended after a split truncated it"
        );
        // A scanning index has no slots, so crossing the scan threshold also rebuilds.
        if edge.len() * 4 > self.slots.len() * 3 || edge.len() > MAX_INDEXED_LEN {
            self.rebuild(edge);
            return;
        }
        for pos in old_len..edge.len() {
            self.insert(edge, pos);
        }
    }

    /// Drops the positions a split moved out of `edge`, the prefix that remains.
    pub(super) fn truncate(&mut self, edge: &[EdgeEntry]) {
        // A split runs under the node's exclusive shape gate and state write lock, so keep
        // the table in O(1) and leave the moved positions as tombstones: the prefix never
        // grows again, and its probe runs stay as short as before the split. Rebuild only
        // when a table half the size would still fit, which bounds the slack at 2x. By
        // then the edge holds under half the positions the table was sized for, so the
        // O(prefix) rebuild is paid for by the larger number of blocks split off since.
        if Self::capacity_for(edge.len()) * 2 >= self.slots.len() {
            return;
        }
        self.rebuild(edge);
    }

    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.slots.len()
    }
}
