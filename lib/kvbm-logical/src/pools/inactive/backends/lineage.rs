// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lineage-aware inactive index — stores blocks in a parent/child graph
//! keyed by `(position, fragment)` and evicts only from leaves.

use std::collections::{BTreeMap, HashMap, HashSet};

use dynamo_tokens::PositionalLineageHash;

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::store::InactiveIndex;

/// The data stored in a lineage node — either a real block or a ghost
/// placeholder created for out-of-order insertions.
enum LineageNodeData {
    Real {
        block_id: BlockId,
        seq_hash: SequenceHash,
        last_used: u64,
    },
    Ghost,
}

struct LineageNode {
    data: LineageNodeData,
    parent_fragment: Option<u64>,
    children: HashSet<u64>,
}

impl LineageNode {
    fn new(block_id: BlockId, lineage_hash: PositionalLineageHash, tick: u64) -> Self {
        let parent_fragment = if lineage_hash.position() > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };
        Self {
            data: LineageNodeData::Real {
                block_id,
                seq_hash: lineage_hash,
                last_used: tick,
            },
            parent_fragment,
            children: HashSet::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

pub(crate) struct LineageBackend {
    nodes: HashMap<u64, HashMap<u64, LineageNode>>,
    leaf_queue: BTreeMap<(u64, u64, u64), ()>,
    count: usize,
    current_tick: u64,
}

impl Default for LineageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LineageBackend {
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            leaf_queue: BTreeMap::new(),
            count: 0,
            current_tick: 0,
        }
    }

    fn insert_inner(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        let lineage_hash = seq_hash;
        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();
        let full_hash = lineage_hash.as_u128();
        let parent_fragment = if position > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        let increment_count: bool;
        let tick = self.current_tick;
        self.current_tick += 1;

        let level = self.nodes.entry(position).or_default();
        match level.entry(fragment) {
            std::collections::hash_map::Entry::Vacant(e) => {
                increment_count = true;
                let node = LineageNode::new(block_id, lineage_hash, tick);
                e.insert(node);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let node = e.get_mut();
                match &node.data {
                    LineageNodeData::Ghost => {
                        increment_count = true;
                        node.data = LineageNodeData::Real {
                            block_id,
                            seq_hash: lineage_hash,
                            last_used: tick,
                        };
                        node.parent_fragment = parent_fragment;
                    }
                    LineageNodeData::Real {
                        seq_hash: existing_hash,
                        ..
                    } => {
                        let existing_full = existing_hash.as_u128();
                        if existing_full == full_hash {
                            panic!(
                                "Duplicate insertion detected! position={}, fragment={:#x}, hash={:#032x}.",
                                position, fragment, full_hash
                            );
                        } else {
                            panic!(
                                "Hash collision detected! position={}, fragment={:#x}, \
                                 existing_hash={:#032x}, new_hash={:#032x}.",
                                position, fragment, existing_full, full_hash
                            );
                        }
                    }
                }
            }
        }

        if increment_count {
            self.count += 1;
        }

        if let Some(p_frag) = parent_fragment {
            let p_pos = position - 1;
            let parent_level = self.nodes.entry(p_pos).or_default();
            let parent_node = parent_level.entry(p_frag).or_insert_with(|| LineageNode {
                data: LineageNodeData::Ghost,
                parent_fragment: None,
                children: HashSet::new(),
            });

            let was_parent_leaf = parent_node.is_leaf();
            parent_node.children.insert(fragment);

            if was_parent_leaf
                && let LineageNodeData::Real { last_used, .. } = parent_node.data
            {
                self.leaf_queue.remove(&(last_used, p_pos, p_frag));
            }
        }

        let node = self.nodes.get(&position).unwrap().get(&fragment).unwrap();
        if node.is_leaf()
            && let LineageNodeData::Real { last_used, .. } = node.data
        {
            self.leaf_queue.insert((last_used, position, fragment), ());
        }
    }

    fn allocate_inner(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        while allocated.len() < count {
            if let Some((&(tick, pos, frag), _)) = self.leaf_queue.iter().next() {
                let key = (tick, pos, frag);
                self.leaf_queue.remove(&key);
                if let Some(b) = self.remove_block(pos, frag) {
                    allocated.push(b);
                }
            } else {
                break;
            }
        }
        allocated
    }

    /// Remove a specific block by its lineage hash (cache-hit path).
    fn remove_by_hash(&mut self, lineage_hash: &PositionalLineageHash) -> Option<(SequenceHash, BlockId)> {
        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();

        let node_data = self
            .nodes
            .get(&position)
            .and_then(|level| level.get(&fragment))
            .and_then(|node| match &node.data {
                LineageNodeData::Real { last_used, .. } => Some(*last_used),
                LineageNodeData::Ghost => None,
            });

        if let Some(tick) = node_data {
            self.leaf_queue.remove(&(tick, position, fragment));
            self.remove_block(position, fragment)
        } else {
            None
        }
    }

    /// Remove the real block at `(position, fragment)`, leaving a ghost in
    /// place if needed. Iteratively prunes orphan ghosts upward.
    fn remove_block(&mut self, position: u64, fragment: u64) -> Option<(SequenceHash, BlockId)> {
        let payload = {
            let level = self.nodes.get_mut(&position)?;
            let node = level.get_mut(&fragment)?;
            match &mut node.data {
                LineageNodeData::Real { .. } => {
                    let prior = std::mem::replace(&mut node.data, LineageNodeData::Ghost);
                    if let LineageNodeData::Real {
                        block_id, seq_hash, ..
                    } = prior
                    {
                        Some((seq_hash, block_id))
                    } else {
                        unreachable!()
                    }
                }
                LineageNodeData::Ghost => None,
            }
        };

        if payload.is_some() {
            self.count -= 1;
        }

        let mut current_pos = position;
        let mut current_frag = fragment;

        loop {
            let mut should_remove_node = false;
            let mut parent_info = None;

            if let Some(level) = self.nodes.get(&current_pos)
                && let Some(node) = level.get(&current_frag)
            {
                let is_ghost = matches!(node.data, LineageNodeData::Ghost);
                if node.children.is_empty() && is_ghost {
                    should_remove_node = true;
                    parent_info = node
                        .parent_fragment
                        .map(|pf| (current_pos.saturating_sub(1), pf));
                }
            }

            if should_remove_node {
                if let Some(level) = self.nodes.get_mut(&current_pos) {
                    level.remove(&current_frag);
                    if level.is_empty() {
                        self.nodes.remove(&current_pos);
                    }
                }

                if let Some((p_pos, p_frag)) = parent_info {
                    let mut parent_became_leaf = false;
                    let mut parent_has_block = false;
                    let mut parent_tick = 0;

                    if let Some(level) = self.nodes.get_mut(&p_pos)
                        && let Some(parent) = level.get_mut(&p_frag)
                    {
                        parent.children.remove(&current_frag);
                        if parent.children.is_empty() {
                            parent_became_leaf = true;
                            match &parent.data {
                                LineageNodeData::Real { last_used, .. } => {
                                    parent_has_block = true;
                                    parent_tick = *last_used;
                                }
                                LineageNodeData::Ghost => {
                                    parent_has_block = false;
                                }
                            }
                        }
                    }

                    if parent_became_leaf {
                        if parent_has_block {
                            self.leaf_queue.insert((parent_tick, p_pos, p_frag), ());
                            break;
                        } else {
                            current_pos = p_pos;
                            current_frag = p_frag;
                            continue;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
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
        self.allocate_inner(count)
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
        self.nodes
            .get(&position)
            .and_then(|level| level.get(&fragment))
            .is_some_and(|node| matches!(node.data, LineageNodeData::Real { .. }))
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        match self.remove_by_hash(&seq_hash) {
            Some((_, id)) if id == block_id => true,
            Some((removed_hash, removed_id)) => {
                // Wrong block under that hash: re-insert and report miss.
                self.insert_inner(removed_hash, removed_id);
                false
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pools::tests::fixtures::BlockSequenceBuilder;

    /// Build a chain of lineage hashes via the testing helpers and return
    /// `(block_id, seq_hash)` pairs.
    fn create_chain(count: usize, offset: u32) -> Vec<(BlockId, SequenceHash)> {
        let tokens: Vec<u32> = (offset..offset + count as u32).collect();
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build()
            .into_iter()
            .map(|(block, hash)| (block.block_id(), hash))
            .collect()
    }

    fn create_blocks(count: usize) -> Vec<(BlockId, SequenceHash)> {
        create_chain(count, 0)
    }

    fn create_block(id: u32) -> (BlockId, SequenceHash) {
        let tokens = vec![id];
        let blocks = BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build();
        let (block, hash) = blocks.into_iter().next().unwrap();
        (block.block_id(), hash)
    }

    impl LineageBackend {
        pub fn get_queue_len(&self) -> usize {
            self.leaf_queue.len()
        }
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

    #[test]
    fn test_interleaved_chains() {
        let mut backend = LineageBackend::new();

        let mut chain1 = create_chain(2, 0);
        let (a_id, a_h) = chain1.remove(0);
        let (b_id, b_h) = chain1.remove(0);

        let mut chain2 = create_chain(2, 1000);
        let (x_id, x_h) = chain2.remove(0);
        let (y_id, y_h) = chain2.remove(0);

        backend.insert(a_h, a_id);
        backend.insert(b_h, b_id);
        backend.insert(x_h, x_id);
        backend.insert(y_h, y_id);

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1[0].1, 1); // B from chain1

        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].1, 0); // A from chain1

        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].1, 1); // Y from chain2

        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].1, 0); // X from chain2
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
        assert!(backend.nodes.is_empty());
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
}
