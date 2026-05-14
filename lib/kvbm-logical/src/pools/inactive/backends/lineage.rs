// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lineage-aware inactive index — a slab-backed parent/child graph that
//! evicts only from leaves, oldest-first.
//!
//! # Structure
//!
//! Every node (real block or out-of-order ghost placeholder) lives in a
//! single pre-sized `Vec<LineageSlot>` arena addressed by `u32` index, so
//! insert / remove / find do **no heap allocation in steady state** — they
//! pop and recycle slots through a free list. The graph edges are slab
//! indices, not hash keys:
//!
//! - `parent` / `first_child` / `next_sibling` — an intrusive parent→child
//!   tree. A single-child chain (the common KV-prefix shape) is just
//!   `first_child`; branches extend the `next_sibling` chain.
//! - `lru_prev` / `lru_next` — an intrusive FIFO list threaded through the
//!   *Real leaf* nodes only. Its head is the oldest evictable block. This
//!   replaces the previous `BTreeMap` leaf queue **and** the per-node
//!   `last_used` tick counter — list position *is* insertion-recency.
//!
//! A single `index: HashMap<(position, fragment), u32>` resolves the
//! `(position, current_hash_fragment)` pair to a slot — needed because
//! lineage navigation is by *fragment* (a child's hash only carries its
//! parent's fragment, never the parent's full hash). It is identity-mixed
//! (see `PairHasher`) and pre-sized, so it does not rehash on the hot path.
//!
//! # Eviction-order note
//!
//! When a node *re-becomes a leaf* (its last child is removed) it is
//! appended at the **tail** of the leaf FIFO — treated as freshly
//! evictable — rather than re-entering at its original insertion position.
//! This differs from the previous `BTreeMap`-by-original-tick behavior; it
//! is the trade that makes the leaf queue O(1) and allocation-free, and is
//! arguably more correct (the node *was* just touched by losing a child).

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use dynamo_tokens::PositionalLineageHash;

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::store::InactiveIndex;

// ---------------------------------------------------------------------------
// `(position, fragment)` index hasher
// ---------------------------------------------------------------------------

/// Hand-rolled mixer for the `(u64, u64)` index key. `current_hash_fragment`
/// is already a well-mixed hash fragment and `position` is a small int;
/// SipHash over the pair would be wasted work on the lookup hot path. A
/// `(u64, u64)` derives `Hash` as two `write_u64` calls, so an FxHash-style
/// rotate-xor-multiply accumulator over those two words is sufficient and
/// cheap. `write` (the byte-slice path) is never exercised by `(u64, u64)`.
#[derive(Default)]
struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write_u64(&mut self, v: u64) {
        // FxHash-style: rotate to spread bits across words, xor in the new
        // word, multiply by an odd constant to avalanche.
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(K);
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("(position, fragment) keys hash via write_u64, not the byte-slice path");
    }
}

#[derive(Default, Clone)]
struct PairBuildHasher;

impl BuildHasher for PairBuildHasher {
    type Hasher = PairHasher;
    fn build_hasher(&self) -> PairHasher {
        PairHasher::default()
    }
}

type IndexMap = HashMap<(u64, u64), u32, PairBuildHasher>;

// ---------------------------------------------------------------------------
// Slab
// ---------------------------------------------------------------------------

/// Payload of a slab slot.
enum SlotData {
    /// A real inactive block.
    Real {
        block_id: BlockId,
        seq_hash: SequenceHash,
    },
    /// Out-of-order placeholder: a parent referenced by a child that was
    /// inserted before it. Always has at least one child while it exists.
    Ghost,
    /// Slot is on the free list; `next_sibling` is the free-list link.
    Free,
}

/// One arena slot. Graph edges and the leaf-FIFO links are `u32` slab
/// indices. `position` / `fragment` are stored on every slot (real *and*
/// ghost) because a ghost has no `seq_hash` yet still needs its index key
/// to remove itself during pruning.
struct LineageSlot {
    data: SlotData,
    position: u64,
    fragment: u64,
    /// Parent node's slot index (real or ghost). `None` for a root
    /// (`position == 0`) or a not-yet-linked node.
    parent: Option<u32>,
    /// Head of this node's intrusive child list.
    first_child: Option<u32>,
    /// Next sibling in the parent's child list. Reused as the free-list
    /// link while the slot is `Free`.
    next_sibling: Option<u32>,
    /// Intrusive leaf-FIFO links — meaningful only while this slot is a
    /// `Real` leaf (no children).
    lru_prev: Option<u32>,
    lru_next: Option<u32>,
}

impl LineageSlot {
    fn is_leaf(&self) -> bool {
        self.first_child.is_none()
    }
}

pub(crate) struct LineageBackend {
    slots: Vec<LineageSlot>,
    /// Free-list head; links through `LineageSlot::next_sibling`.
    free_head: Option<u32>,
    /// `(position, fragment)` → slot index, for parent resolution on
    /// insert and target resolution on remove.
    index: IndexMap,
    /// Intrusive FIFO over `Real` leaf nodes — `leaf_head` is the oldest
    /// (next to evict), `leaf_tail` the newest.
    leaf_head: Option<u32>,
    leaf_tail: Option<u32>,
    /// Number of `Real` nodes (ghosts excluded).
    count: usize,
}

impl Default for LineageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LineageBackend {
    /// Create with no pre-sized capacity (slab and index grow on demand).
    /// Production builds go through [`with_capacity`](Self::with_capacity).
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create pre-sized for `capacity` real blocks. The inactive pool is
    /// bounded by the store's `total_blocks`, so sizing the slab and index
    /// to that bound means the steady-state hot path never reallocates.
    /// (Out-of-order ghosts can briefly push past `capacity`; that grows
    /// the slab once, amortized — not a steady-state cost.)
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            index: HashMap::with_capacity_and_hasher(capacity, PairBuildHasher),
            leaf_head: None,
            leaf_tail: None,
            count: 0,
        }
    }

    // ---- slab alloc / free ----

    /// Place `slot` into a recycled or freshly-pushed arena cell.
    fn alloc_slot(&mut self, slot: LineageSlot) -> u32 {
        match self.free_head {
            Some(idx) => {
                self.free_head = self.slots[idx as usize].next_sibling;
                self.slots[idx as usize] = slot;
                idx
            }
            None => {
                let idx = self.slots.len();
                debug_assert!(
                    idx <= u32::MAX as usize,
                    "lineage slab exceeded u32 index space"
                );
                self.slots.push(slot);
                idx as u32
            }
        }
    }

    /// Return a slot to the free list. Other fields are left stale — the
    /// next `alloc_slot` overwrites the cell wholesale.
    fn free_slot(&mut self, idx: u32) {
        self.slots[idx as usize].data = SlotData::Free;
        self.slots[idx as usize].next_sibling = self.free_head;
        self.free_head = Some(idx);
    }

    // ---- intrusive leaf FIFO ----

    /// Append a `Real` leaf to the tail (newest) of the leaf FIFO.
    fn leaf_push_back(&mut self, idx: u32) {
        self.slots[idx as usize].lru_prev = self.leaf_tail;
        self.slots[idx as usize].lru_next = None;
        match self.leaf_tail {
            Some(t) => self.slots[t as usize].lru_next = Some(idx),
            None => self.leaf_head = Some(idx),
        }
        self.leaf_tail = Some(idx);
    }

    /// Unlink a node from the leaf FIFO. No-op semantics rely on the caller
    /// only invoking this for a slot actually in the list.
    fn leaf_unlink(&mut self, idx: u32) {
        let prev = self.slots[idx as usize].lru_prev;
        let next = self.slots[idx as usize].lru_next;
        match prev {
            Some(p) => self.slots[p as usize].lru_next = next,
            None => self.leaf_head = next,
        }
        match next {
            Some(n) => self.slots[n as usize].lru_prev = prev,
            None => self.leaf_tail = prev,
        }
        self.slots[idx as usize].lru_prev = None;
        self.slots[idx as usize].lru_next = None;
    }

    // ---- child list ----

    /// Remove `child` from `parent`'s intrusive child list. O(siblings) —
    /// KV-prefix branch factors are small.
    fn detach_child(&mut self, parent: u32, child: u32) {
        let head = self.slots[parent as usize].first_child;
        if head == Some(child) {
            self.slots[parent as usize].first_child = self.slots[child as usize].next_sibling;
            return;
        }
        let mut cur = head;
        while let Some(c) = cur {
            let next = self.slots[c as usize].next_sibling;
            if next == Some(child) {
                self.slots[c as usize].next_sibling = self.slots[child as usize].next_sibling;
                return;
            }
            cur = next;
        }
        debug_assert!(
            false,
            "detach_child: {child} not found under parent {parent}"
        );
    }

    // ---- core mutation ----

    fn insert_inner(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        let position = seq_hash.position();
        let fragment = seq_hash.current_hash_fragment();
        let parent_fragment = if position > 0 {
            Some(seq_hash.parent_hash_fragment())
        } else {
            None
        };

        // 1. Find-or-create this node. An existing entry must be a Ghost
        //    (a real-vs-real hit is a duplicate or a collision bug).
        let node_idx = match self.index.get(&(position, fragment)) {
            Some(&idx) => {
                match self.slots[idx as usize].data {
                    SlotData::Ghost => {
                        self.slots[idx as usize].data = SlotData::Real { block_id, seq_hash };
                        self.count += 1;
                    }
                    SlotData::Real {
                        seq_hash: existing, ..
                    } => {
                        if existing.as_u128() == seq_hash.as_u128() {
                            panic!(
                                "Duplicate insertion detected! position={}, fragment={:#x}, \
                                 hash={:#032x}.",
                                position,
                                fragment,
                                seq_hash.as_u128()
                            );
                        } else {
                            panic!(
                                "Hash collision detected! position={}, fragment={:#x}, \
                                 existing_hash={:#032x}, new_hash={:#032x}.",
                                position,
                                fragment,
                                existing.as_u128(),
                                seq_hash.as_u128()
                            );
                        }
                    }
                    SlotData::Free => unreachable!("index points at a freed slot"),
                }
                idx
            }
            None => {
                let idx = self.alloc_slot(LineageSlot {
                    data: SlotData::Real { block_id, seq_hash },
                    position,
                    fragment,
                    parent: None,
                    first_child: None,
                    next_sibling: None,
                    lru_prev: None,
                    lru_next: None,
                });
                self.index.insert((position, fragment), idx);
                self.count += 1;
                idx
            }
        };

        // 2. Link to parent (creating a ghost parent if it does not exist
        //    yet). A fresh node and a just-promoted ghost both have
        //    `parent == None` here, so this links exactly once.
        if let Some(p_frag) = parent_fragment
            && self.slots[node_idx as usize].parent.is_none()
        {
            let p_pos = position - 1;
            let parent_idx = match self.index.get(&(p_pos, p_frag)) {
                Some(&pidx) => pidx,
                None => {
                    let pidx = self.alloc_slot(LineageSlot {
                        data: SlotData::Ghost,
                        position: p_pos,
                        fragment: p_frag,
                        parent: None,
                        first_child: None,
                        next_sibling: None,
                        lru_prev: None,
                        lru_next: None,
                    });
                    self.index.insert((p_pos, p_frag), pidx);
                    pidx
                }
            };

            let parent_was_leaf = self.slots[parent_idx as usize].is_leaf();
            // Prepend node_idx into parent's child list.
            self.slots[node_idx as usize].parent = Some(parent_idx);
            self.slots[node_idx as usize].next_sibling =
                self.slots[parent_idx as usize].first_child;
            self.slots[parent_idx as usize].first_child = Some(node_idx);

            // A Real parent that was a leaf is now an interior node.
            if parent_was_leaf
                && matches!(self.slots[parent_idx as usize].data, SlotData::Real { .. })
            {
                self.leaf_unlink(parent_idx);
            }
        }

        // 3. If this node is a Real leaf, it is the newest evictable block.
        //    (A promoted ghost already has children — not a leaf.)
        if self.slots[node_idx as usize].is_leaf() {
            self.leaf_push_back(node_idx);
        }
    }

    /// Look up a node by its full `SequenceHash` and, if it is the real
    /// block stored under that `(position, fragment)` key, remove it.
    ///
    /// The full-hash verification matters: `(position, current_hash_fragment)`
    /// is not unique — distinct `PositionalLineageHash`es can share that pair
    /// (same current hash + position, different parent), so a key-only match
    /// would let a lookup for one PLH delete another's block.
    fn remove_by_hash(
        &mut self,
        lineage_hash: &PositionalLineageHash,
    ) -> Option<(SequenceHash, BlockId)> {
        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();
        let idx = *self.index.get(&(position, fragment))?;
        match self.slots[idx as usize].data {
            SlotData::Real { seq_hash, .. } if seq_hash == *lineage_hash => {
                Some(self.remove_node_at(idx))
            }
            _ => None,
        }
    }

    /// Turn the `Real` node at `idx` into a `Ghost`, then iteratively prune
    /// any now-childless ghost up the parent chain. Returns the evicted
    /// `(seq_hash, block_id)`.
    fn remove_node_at(&mut self, idx: u32) -> (SequenceHash, BlockId) {
        let was_leaf = self.slots[idx as usize].is_leaf();
        let payload = match std::mem::replace(&mut self.slots[idx as usize].data, SlotData::Ghost) {
            SlotData::Real { seq_hash, block_id } => (seq_hash, block_id),
            _ => unreachable!("remove_node_at called on a non-Real slot"),
        };
        self.count -= 1;
        if was_leaf {
            // It was a Real leaf, hence in the leaf FIFO.
            self.leaf_unlink(idx);
        }

        // Prune: a childless Ghost is removed from the graph entirely; if
        // that orphans its parent, recurse. A node that still has children
        // simply stays as a Ghost.
        let mut cur = idx;
        loop {
            if self.slots[cur as usize].first_child.is_some() {
                break;
            }
            let parent = self.slots[cur as usize].parent;
            let key = (
                self.slots[cur as usize].position,
                self.slots[cur as usize].fragment,
            );
            if let Some(p) = parent {
                self.detach_child(p, cur);
            }
            self.index.remove(&key);
            self.free_slot(cur);

            match parent {
                None => break,
                Some(p) => {
                    if self.slots[p as usize].first_child.is_some() {
                        break; // parent still has other children
                    }
                    match self.slots[p as usize].data {
                        SlotData::Real { .. } => {
                            // Parent is a Real leaf again — newest evictable.
                            self.leaf_push_back(p);
                            break;
                        }
                        SlotData::Ghost => {
                            cur = p; // childless ghost — prune it too
                        }
                        SlotData::Free => unreachable!("parent slot is free"),
                    }
                }
            }
        }
        payload
    }
}

impl InactiveIndex for LineageBackend {
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(pair) = self.remove_by_hash(hash) {
                matches.push(pair);
            } else {
                break;
            }
        }
        matches
    }

    fn find_match(&mut self, hash: SequenceHash, _touch: bool) -> Option<(SequenceHash, BlockId)> {
        self.remove_by_hash(&hash)
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::new();
        for hash in hashes {
            if let Some(pair) = self.remove_by_hash(hash) {
                matches.push(pair);
            }
        }
        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        while allocated.len() < count {
            // Oldest evictable leaf. `remove_node_at` unlinks it from the
            // FIFO and may expose its parent as the next leaf.
            match self.leaf_head {
                Some(idx) => allocated.push(self.remove_node_at(idx)),
                None => break,
            }
        }
        allocated
    }

    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        self.insert_inner(seq_hash, block_id);
    }

    fn len(&self) -> usize {
        self.count
    }

    fn has(&self, seq_hash: SequenceHash) -> bool {
        let position = seq_hash.position();
        let fragment = seq_hash.current_hash_fragment();
        self.index.get(&(position, fragment)).is_some_and(|&idx| {
            match self.slots[idx as usize].data {
                SlotData::Real {
                    seq_hash: stored, ..
                } => stored == seq_hash,
                _ => false,
            }
        })
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        // Match on the full `SequenceHash` AND the block id — `(position,
        // current_hash_fragment)` alone can collide across distinct PLHs.
        let position = seq_hash.position();
        let fragment = seq_hash.current_hash_fragment();
        let hit = self.index.get(&(position, fragment)).is_some_and(|&idx| {
            match self.slots[idx as usize].data {
                SlotData::Real {
                    seq_hash: stored,
                    block_id: stored_id,
                } => stored == seq_hash && stored_id == block_id,
                _ => false,
            }
        });
        if hit {
            self.remove_by_hash(&seq_hash).is_some()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::BlockSequenceBuilder;

    impl LineageBackend {
        /// Test-only: length of the intrusive leaf FIFO.
        fn get_queue_len(&self) -> usize {
            let mut n = 0;
            let mut cur = self.leaf_head;
            while let Some(idx) = cur {
                n += 1;
                cur = self.slots[idx as usize].lru_next;
            }
            n
        }

        /// Test-only: no live slots remain (all real + ghost nodes gone).
        fn is_graph_empty(&self) -> bool {
            self.index.is_empty()
        }
    }

    /// Build a chain of lineage hashes and return `(block_id, seq_hash)` pairs.
    fn create_chain(count: usize, offset: u32) -> Vec<(BlockId, SequenceHash)> {
        let tokens: Vec<u32> = (offset..offset + count as u32).collect();
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build()
    }

    fn create_blocks(count: usize) -> Vec<(BlockId, SequenceHash)> {
        create_chain(count, 0)
    }

    fn create_block(id: u32) -> (BlockId, SequenceHash) {
        BlockSequenceBuilder::from_tokens(vec![id])
            .with_block_size(1)
            .build()
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn test_leaf_insertion() {
        let mut backend = LineageBackend::new();
        let (id1, h1) = create_block(1);

        backend.insert(h1, id1);

        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].1, 0);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_parent_child_insertion() {
        let mut backend = LineageBackend::new();

        let mut blocks = create_blocks(2);
        let (id1, h1) = blocks.remove(0);
        let (id2, h2) = blocks.remove(0);

        backend.insert(h1, id1);
        assert_eq!(backend.get_queue_len(), 1);

        backend.insert(h2, id2);
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].1, 1);

        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(allocated2[0].1, 0);
    }

    #[test]
    fn test_out_of_order_insertion() {
        let mut backend = LineageBackend::new();

        let mut chain = create_blocks(2);
        let (id2, h2) = chain.remove(1);
        backend.insert(h2, id2);
        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1);

        let mut chain2 = create_blocks(2);
        let (id1, h1) = chain2.remove(0);
        backend.insert(h1, id1);

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated[0].1, 1);

        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2[0].1, 0);
    }

    #[test]
    fn test_branching() {
        let mut backend = LineageBackend::new();

        let seq1 = create_chain(3, 0);
        let seq2 = create_chain(3, 5000);

        for (id, h) in seq1 {
            backend.insert(h, id);
        }
        for (id, h) in seq2 {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), 6);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 5);

        assert_eq!(backend.get_queue_len(), 2);
    }

    /// Two interleaved 2-chains. Eviction order reflects the intrusive-FIFO
    /// semantics: when a node re-becomes a leaf it appends at the *tail*, so
    /// the two chains' roots are not evicted back-to-back with their own
    /// leaves — they round-robin. (The previous BTreeMap-by-original-tick
    /// backend evicted `B, A, Y, X`; see the module doc for the rationale.)
    #[test]
    fn test_interleaved_chains() {
        let mut backend = LineageBackend::new();

        let mut chain1 = create_chain(2, 0);
        let (a_id, a_h) = chain1.remove(0);
        let (b_id, b_h) = chain1.remove(0);

        let mut chain2 = create_chain(2, 1000);
        let (x_id, x_h) = chain2.remove(0);
        let (y_id, y_h) = chain2.remove(0);

        backend.insert(a_h, a_id); // chain1 root  (block_id 0)
        backend.insert(b_h, b_id); // chain1 leaf  (block_id 1)
        backend.insert(x_h, x_id); // chain2 root  (block_id 0)
        backend.insert(y_h, y_id); // chain2 leaf  (block_id 1)

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_queue_len(), 2);
        // Leaf FIFO: [B, Y].

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1[0].1, b_id); // B; A re-leafs → tail. FIFO: [Y, A]

        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].1, y_id); // Y; X re-leafs → tail. FIFO: [A, X]

        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].1, a_id); // A. FIFO: [X]

        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].1, x_id); // X
    }

    #[test]
    fn test_remove_by_hash() {
        let mut backend = LineageBackend::new();

        let (id1, h1) = create_block(1);
        backend.insert(h1, id1);
        assert_eq!(backend.len(), 1);

        let removed = backend.remove_by_hash(&h1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().1, 0);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_deep_chain_cleanup_iterative() {
        let depth = 1000;
        let mut backend = LineageBackend::new();

        let blocks = create_blocks(depth);
        let last_hash = blocks[depth - 1].1;
        for (id, h) in blocks {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), depth);
        assert_eq!(backend.get_queue_len(), 1);

        backend.remove_by_hash(&last_hash);

        assert_eq!(backend.len(), depth - 1);
        assert_eq!(backend.get_queue_len(), 1);

        backend = LineageBackend::new();

        let mut chain = create_blocks(101);
        let (leaf_id, leaf_h) = chain.remove(100);

        backend.insert(leaf_h, leaf_id);

        assert_eq!(backend.len(), 1);

        backend.remove_by_hash(&leaf_h);

        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_split_sequence_eviction() {
        let mut backend = LineageBackend::new();

        let branch1 = create_chain(5, 0);
        let branch2 = create_chain(5, 3000);

        for (id, h) in branch1 {
            backend.insert(h, id);
        }
        for (id, h) in branch2 {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), 10);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 9);

        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2.len(), 1);
        assert_eq!(backend.len(), 8);

        assert_eq!(backend.get_queue_len(), 2);

        backend.allocate(2);
        assert_eq!(backend.len(), 6);

        assert_eq!(backend.get_queue_len(), 2);
    }

    /// Regression: lookups must compare the full `SequenceHash`, not just
    /// `(position, current_hash_fragment)`. Two `PositionalLineageHash`es that
    /// share that key pair but have different parents must not collide —
    /// otherwise `find_matches` / `scan_matches` / `take` / `has` would
    /// return or remove the wrong block.
    #[test]
    fn lookup_rejects_same_position_fragment_but_different_full_hash() {
        let stored: SequenceHash = SequenceHash::new(0xAA, Some(0x11), 5);
        let impostor: SequenceHash = SequenceHash::new(0xAA, Some(0x22), 5);
        assert_eq!(stored.position(), impostor.position());
        assert_eq!(
            stored.current_hash_fragment(),
            impostor.current_hash_fragment()
        );
        assert_ne!(stored.as_u128(), impostor.as_u128());

        let mut backend = LineageBackend::new();
        backend.insert(stored, 42);

        assert!(backend.has(stored));
        assert!(
            !backend.has(impostor),
            "has() returned a false-positive for impostor PLH"
        );

        assert!(
            backend.remove_by_hash(&impostor).is_none(),
            "remove_by_hash matched impostor PLH and deleted stored block"
        );
        assert_eq!(backend.len(), 1);

        let scan_hits = backend.scan_matches(&[impostor], false);
        assert!(scan_hits.is_empty());
        assert_eq!(backend.len(), 1);

        let find_hits = backend.find_matches(&[impostor], false);
        assert!(find_hits.is_empty());
        assert_eq!(backend.len(), 1);

        assert!(!backend.take(impostor, 42));
        assert_eq!(backend.len(), 1);

        let removed = backend.remove_by_hash(&stored);
        assert_eq!(removed, Some((stored, 42)));
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    /// Slab cells are recycled through the free list rather than reallocated.
    #[test]
    fn slab_recycles_freed_slots() {
        let mut backend = LineageBackend::with_capacity(8);

        // Fill, drain, refill — the slab length must not grow past the
        // high-water mark because freed slots are reused.
        for (id, h) in create_chain(8, 0) {
            backend.insert(h, id);
        }
        let high_water = backend.slots.len();
        assert_eq!(high_water, 8, "no ghosts for an in-order chain");

        backend.allocate(8);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());

        for (id, h) in create_chain(8, 9000) {
            backend.insert(h, id);
        }
        assert_eq!(
            backend.slots.len(),
            high_water,
            "freed slots must be recycled, not reallocated"
        );
    }

    /// Re-becoming a leaf appends at the FIFO tail (documented eviction-order
    /// change vs. the previous BTreeMap-by-original-tick behavior).
    #[test]
    fn re_leafed_node_goes_to_fifo_tail() {
        // Two independent 2-chains: A0->A1 and B0->B1.
        let mut backend = LineageBackend::new();
        let chain_a = create_chain(2, 0);
        let chain_b = create_chain(2, 7000);
        let (a0_id, a0_h) = chain_a[0];
        let (a1_id, a1_h) = chain_a[1];
        let (b0_id, b0_h) = chain_b[0];
        let (b1_id, b1_h) = chain_b[1];

        backend.insert(a0_h, a0_id);
        backend.insert(a1_h, a1_id);
        backend.insert(b0_h, b0_id);
        backend.insert(b1_h, b1_id);
        // Leaf FIFO: [A1, B1].

        // Remove A1 by hash → A0 re-becomes a leaf and goes to the TAIL.
        // FIFO is now [B1, A0], not [A0, B1].
        assert!(backend.remove_by_hash(&a1_h).is_some());

        let order: Vec<BlockId> = backend.allocate(2).into_iter().map(|(_, id)| id).collect();
        assert_eq!(order, vec![b1_id, a0_id], "re-leafed A0 evicts after B1");
    }

    #[test]
    fn pair_hasher_distinguishes_keys() {
        use std::collections::HashSet;
        let keys = [
            (0u64, 0u64),
            (0, 1),
            (1, 0),
            (1, 1),
            (5, 0xdead_beef),
            (5, 0xbeef_dead),
        ];
        let digests: HashSet<u64> = keys
            .iter()
            .map(|&k| {
                let mut h = PairBuildHasher.build_hasher();
                std::hash::Hash::hash(&k, &mut h);
                h.finish()
            })
            .collect();
        assert_eq!(
            digests.len(),
            keys.len(),
            "distinct pairs → distinct digests"
        );
    }
}
