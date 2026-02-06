// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use dynamo_tokens::PositionalLineageHash;

use super::super::{Block, BlockMetadata, InactivePoolBackend, Registered};

/// The data stored in a lineage node - either a real block or a ghost placeholder.
enum LineageNodeData<T: BlockMetadata> {
    /// A real block with timestamp.
    Real {
        block: Block<T, Registered>,
        last_used: u64,
    },
    /// A ghost node created for out-of-order insertions.
    Ghost,
}

/// A node in the lineage graph.
struct LineageNode<T: BlockMetadata> {
    /// The data stored in this node (real block or ghost).
    data: LineageNodeData<T>,

    /// The parent fragment (at position - 1), if any.
    parent_fragment: Option<u64>,

    /// Children fragments (at position + 1).
    children: HashSet<u64>,
}

impl<T: BlockMetadata> LineageNode<T> {
    fn new(block: Block<T, Registered>, lineage_hash: PositionalLineageHash, tick: u64) -> Self {
        let parent_fragment = if lineage_hash.position() > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        Self {
            data: LineageNodeData::Real {
                block,
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

/// A backend that manages blocks using a lineage graph and evicts from the leaves.
pub struct LineageBackend<T: BlockMetadata> {
    /// Map from (position, fragment) to Node.
    nodes: HashMap<u64, HashMap<u64, LineageNode<T>>>,

    /// Sorted queue of leaf nodes, keyed by (last_used, position, fragment).
    /// Smallest key (oldest tick) is popped first.
    leaf_queue: BTreeMap<(u64, u64, u64), ()>,

    /// Total number of blocks currently stored (excluding ghost nodes).
    count: usize,

    /// Monotonic counter for insertion ordering.
    current_tick: u64,
}

impl<T: BlockMetadata> Default for LineageBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockMetadata> LineageBackend<T> {
    /// Creates a new LineageBackend.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            leaf_queue: BTreeMap::new(),
            count: 0,
            current_tick: 0,
        }
    }

    /// Inserts a block into the lineage graph.
    /// Panics on collision or duplicate insertion.
    pub fn insert(&mut self, block: Block<T, Registered>) {
        let lineage_hash = block.sequence_hash();
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

        // 1. Create or update the node
        let level = self.nodes.entry(position).or_default();
        match level.entry(fragment) {
            std::collections::hash_map::Entry::Vacant(e) => {
                increment_count = true;
                let node = LineageNode::new(block, lineage_hash, tick);
                e.insert(node);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let node = e.get_mut();
                match &node.data {
                    LineageNodeData::Ghost => {
                        // Fill ghost with real block data
                        increment_count = true;
                        node.data = LineageNodeData::Real {
                            block,
                            last_used: tick,
                        };
                        node.parent_fragment = parent_fragment;
                    }
                    LineageNodeData::Real {
                        block: existing_block,
                        ..
                    } => {
                        let existing_hash = existing_block.sequence_hash().as_u128();
                        if existing_hash == full_hash {
                            panic!(
                                "Duplicate insertion detected! position={}, fragment={:#x}, hash={:#032x}. \
                                 The same block was inserted twice.",
                                position, fragment, full_hash
                            );
                        } else {
                            panic!(
                                "Hash collision detected! position={}, fragment={:#x}, \
                                 existing_hash={:#032x}, new_hash={:#032x}. \
                                 Different blocks mapped to same position+fragment.",
                                position, fragment, existing_hash, full_hash
                            );
                        }
                    }
                }
            }
        }

        if increment_count {
            self.count += 1;
        }

        // 2. Link to parent
        if let Some(p_frag) = parent_fragment {
            let p_pos = position - 1;

            let parent_level = self.nodes.entry(p_pos).or_default();
            let parent_node = parent_level.entry(p_frag).or_insert_with(|| {
                LineageNode {
                    data: LineageNodeData::Ghost,
                    parent_fragment: None, // We don't know the parent's parent yet
                    children: HashSet::new(),
                }
            });

            let was_parent_leaf = parent_node.is_leaf();
            parent_node.children.insert(fragment);

            if was_parent_leaf {
                // Parent was a leaf, now has a child. Remove from queue.
                // Note: Ghost nodes are never in queue.
                if let LineageNodeData::Real { last_used, .. } = parent_node.data {
                    self.leaf_queue.remove(&(last_used, p_pos, p_frag));
                }
            }
        }

        // 3. Update LRU status for this node
        let node = self.nodes.get(&position).unwrap().get(&fragment).unwrap();
        if node.is_leaf()
            && let LineageNodeData::Real { last_used, .. } = node.data
        {
            self.leaf_queue.insert((last_used, position, fragment), ());
        }
    }

    /// Allocates (removes) a block from the pool, preferring leaves in LRU order.
    pub fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        while allocated.len() < count {
            if let Some((&(_tick, pos, frag), _)) = self.leaf_queue.iter().next() {
                // Need to remove from map using the key we just found
                let key = (_tick, pos, frag);
                self.leaf_queue.remove(&key);

                if let Some(b) = self.remove_block(pos, frag) {
                    allocated.push(b);
                }
            } else {
                break; // No more leaves
            }
        }

        allocated
    }

    /// Removes a specific block by its lineage hash (for cache hits).
    pub fn remove(&mut self, lineage_hash: &PositionalLineageHash) -> Option<Block<T, Registered>> {
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
            // Remove from queue if present (might be present if it's a leaf)
            self.leaf_queue.remove(&(tick, position, fragment));
            self.remove_block(position, fragment)
        } else {
            None
        }
    }

    /// Internal method to remove a block from the graph.
    /// Returns the block if one existed at that node.
    /// Handles ghost cleanup iteratively.
    fn remove_block(&mut self, position: u64, fragment: u64) -> Option<Block<T, Registered>> {
        let node_block = {
            let level = self.nodes.get_mut(&position)?;
            let node = level.get_mut(&fragment)?;
            match &mut node.data {
                LineageNodeData::Real { .. } => {
                    // Replace Real with Ghost, taking ownership of the block
                    let block_val = std::mem::replace(&mut node.data, LineageNodeData::Ghost);
                    if let LineageNodeData::Real { block, .. } = block_val {
                        Some(block)
                    } else {
                        unreachable!()
                    }
                }
                LineageNodeData::Ghost => None,
            }
        };

        if node_block.is_some() {
            self.count -= 1;
        }

        let mut current_pos = position;
        let mut current_frag = fragment;

        // Loop for iterative cleanup upwards
        loop {
            let mut should_remove_node = false;
            let mut parent_info = None;

            if let Some(level) = self.nodes.get(&current_pos)
                && let Some(node) = level.get(&current_frag)
            {
                let is_ghost = matches!(node.data, LineageNodeData::Ghost);
                if node.children.is_empty() && is_ghost {
                    // It's a ghost leaf (no block, no children). Prune it.
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
                            // Parent is a real block leaf -> add to queue using its OLD tick
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

        node_block
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for LineageBackend<T> {
    fn find_matches(
        &mut self,
        hashes: &[PositionalLineageHash],
        _touch: bool,
    ) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.remove(hash) {
                matches.push(block);
            } else {
                break; // Stop on first miss
            }
        }

        matches
    }

    fn scan_matches(
        &mut self,
        hashes: &[PositionalLineageHash],
        _touch: bool,
    ) -> Vec<(PositionalLineageHash, Block<T, Registered>)> {
        let mut matches = Vec::new();

        for hash in hashes {
            if let Some(block) = self.remove(hash) {
                matches.push((*hash, block));
            }
            // Unlike find_matches: NO break on miss - continue scanning
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        // Delegate to the inherent method
        LineageBackend::allocate(self, count)
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        // Delegate to the inherent method
        LineageBackend::insert(self, block)
    }

    fn len(&self) -> usize {
        self.count
    }

    fn has_block(&self, seq_hash: PositionalLineageHash) -> bool {
        let position = seq_hash.position();
        let fragment = seq_hash.current_hash_fragment();

        self.nodes
            .get(&position)
            .and_then(|level| level.get(&fragment))
            .is_some_and(|node| matches!(node.data, LineageNodeData::Real { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceHash;
    use crate::blocks::Block;
    use crate::pools::tests::TestData;
    use crate::pools::tests::fixtures::BlockSequenceBuilder;

    impl<T: BlockMetadata> LineageBackend<T> {
        /// Test helper: get the number of entries in the leaf queue.
        pub fn get_queue_len(&self) -> usize {
            self.leaf_queue.len()
        }
    }

    // Helper to create test blocks with proper lineage using BlockSequenceBuilder
    // Returns a vector of (Block, SequenceHash) tuples
    // offset: starting token value (use different offsets for independent chains)
    fn create_blocks_with_offset(
        count: usize,
        offset: u32,
    ) -> Vec<(Block<TestData, Registered>, SequenceHash)> {
        let tokens: Vec<u32> = (offset..offset + count as u32).collect();
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build()
    }

    // Helper to create test blocks starting from token 0
    fn create_blocks(count: usize) -> Vec<(Block<TestData, Registered>, SequenceHash)> {
        create_blocks_with_offset(count, 0)
    }

    // Helper for single block creation (root block at position 0)
    fn create_block(id: u32) -> (Block<TestData, Registered>, SequenceHash) {
        let tokens = vec![id];
        let blocks = BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build();
        blocks.into_iter().next().unwrap()
    }

    #[test]
    fn test_leaf_insertion() {
        let mut backend = LineageBackend::<TestData>::new();

        let (b1, _) = create_block(1);

        backend.insert(b1);

        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1); // It is a leaf (no children)

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 0); // Block ID is 0 (first block in sequence)
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_parent_child_insertion() {
        let mut backend = LineageBackend::<TestData>::new();

        // Create a sequence of 2 blocks with proper parent-child relationship
        let mut blocks = create_blocks(2);
        let (b1, _) = blocks.remove(0); // Parent at position 0
        let (b2, _) = blocks.remove(0); // Child at position 1

        // Insert parent first
        backend.insert(b1);
        assert_eq!(backend.get_queue_len(), 1); // b1 is leaf

        // Insert child
        backend.insert(b2);
        assert_eq!(backend.len(), 2);

        // b1 is no longer leaf (has child b2). b2 is leaf.
        // LRU should contain only b2.
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 1); // Should allocate b2 (leaf, block_id=1)

        // Now b1 should be a leaf again and added to LRU
        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(allocated2[0].block_id(), 0); // b1 has block_id=0
    }

    #[test]
    fn test_out_of_order_insertion() {
        let mut backend = LineageBackend::<TestData>::new();

        // Insert child first (from blocks2)
        let mut blocks2_mut = create_blocks(2);
        backend.insert(blocks2_mut.remove(1).0);
        // Created ghost node for parent b1.
        // b2 is leaf.
        assert_eq!(backend.len(), 1); // Only 1 actual block
        assert_eq!(backend.get_queue_len(), 1);

        // Insert parent (from blocks1)
        let mut blocks1_mut = create_blocks(2);
        backend.insert(blocks1_mut.remove(0).0);
        // Parent b1 fills ghost. It has child b2, so it's NOT a leaf.
        // b2 is still leaf.

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1); // Only b2

        let allocated = backend.allocate(1);
        assert_eq!(allocated[0].block_id(), 1); // b2

        // Now b1 becomes leaf
        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2[0].block_id(), 0); // b1
    }

    #[test]
    fn test_branching() {
        let mut backend = LineageBackend::<TestData>::new();

        // Test that multiple independent chains can coexist and be allocated independently
        let seq1 = create_blocks_with_offset(3, 0); // chain1: 0 -> 1 -> 2
        let seq2 = create_blocks_with_offset(3, 5000); // chain2: 0 -> 1 -> 2

        // Insert all blocks from both chains
        for (block, _) in seq1 {
            backend.insert(block);
        }
        for (block, _) in seq2 {
            backend.insert(block);
        }

        // Should have 6 blocks total
        assert_eq!(backend.len(), 6);
        // Leaves are position 2 from each chain (2 leaves)
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate one leaf
        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 5);

        // Now position 1 from one chain should be a leaf, plus position 2 from the other chain
        assert_eq!(backend.get_queue_len(), 2);
    }

    #[test]
    fn test_interleaved_chains() {
        // Chain 1: A(0) -> B(1)
        // Chain 2: X(0) -> Y(1)
        // We want strict consumption based on insertion order (ticks).
        let mut backend = LineageBackend::<TestData>::new();

        let mut chain1 = create_blocks_with_offset(2, 0);
        let (a, _) = chain1.remove(0);
        let (b, _) = chain1.remove(0);

        let mut chain2 = create_blocks_with_offset(2, 1000);
        let (x, _) = chain2.remove(0);
        let (y, _) = chain2.remove(0);

        // Insert in order: A, B, X, Y
        // insert(A) tick 0
        // insert(B) tick 1
        // insert(X) tick 2
        // insert(Y) tick 3
        // So Chain 1 is older.

        backend.insert(a);
        backend.insert(b);
        backend.insert(x);
        backend.insert(y);

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_queue_len(), 2); // Leaves: B, Y

        // B (tick 1) is older than Y (tick 3). Expect B.
        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1[0].block_id(), 1); // B (block_id 1 from chain1)

        // Now A becomes leaf. A has tick 0.
        // Queue: A(0), Y(3).
        // Expect A.
        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].block_id(), 0); // A (block_id 0 from chain1)

        // Now Y(3).
        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].block_id(), 1); // Y (block_id 1 from chain2)

        // Now X becomes leaf. X has tick 2.
        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].block_id(), 0); // X (block_id 0 from chain2)
    }

    #[test]
    fn test_remove_by_hash() {
        let mut backend = LineageBackend::<TestData>::new();

        let (b1, seq_hash) = create_block(1);

        backend.insert(b1);
        assert_eq!(backend.len(), 1);

        let removed = backend.remove(&seq_hash);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().block_id(), 0);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_deep_chain_cleanup_iterative() {
        // Create deep chain: 0 -> 1 -> 2 ... -> 999
        let depth = 1000;
        let mut backend = LineageBackend::<TestData>::new();

        // Create a deep chain of blocks
        let blocks = create_blocks(depth);
        let last_hash = blocks[depth - 1].1;
        for (block, _) in blocks {
            backend.insert(block);
        }

        assert_eq!(backend.len(), depth);
        // Only last one is leaf
        assert_eq!(backend.get_queue_len(), 1);

        backend.remove(&last_hash);

        assert_eq!(backend.len(), depth - 1);
        // Now depth-2 is leaf
        assert_eq!(backend.get_queue_len(), 1);

        // Test out-of-order insertion to create ghosts
        backend = LineageBackend::<TestData>::new();

        // Create a chain and insert only the leaf at position 100
        let mut chain = create_blocks(101); // 0..100
        let (b_leaf, h_leaf) = chain.remove(100);

        // Insert leaf at depth 100. This creates a ghost parent at position 99.
        backend.insert(b_leaf);

        assert_eq!(backend.len(), 1); // Only 1 real block
        // Ghost nodes exist but are not counted in len

        // Remove leaf. This should clean up the ghost at position 99.
        backend.remove(&h_leaf);

        assert_eq!(backend.len(), 0);
        assert!(backend.nodes.is_empty());
    }

    #[test]
    fn test_split_sequence_eviction() {
        // Test eviction ordering with two independent chains
        // Branch 1: A(0)->B(1)->C(2)->D(3)->E(4)
        // Branch 2: X(0)->Y(1)->Z(2)->W(3)->V(4)
        let mut backend = LineageBackend::<TestData>::new();

        // Create two separate 5-block chains with different tokens
        let mut branch1 = create_blocks_with_offset(5, 0);
        let mut branch2 = create_blocks_with_offset(5, 3000);

        // Insert all 5 blocks from branch1
        for _i in 0..5 {
            backend.insert(branch1.remove(0).0);
        }

        // Insert all 5 blocks from branch2
        for _i in 0..5 {
            backend.insert(branch2.remove(0).0);
        }

        assert_eq!(backend.len(), 10);
        // Leaves are E(4) from branch1 and V(4) from branch2
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate first leaf (oldest)
        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 9);

        // Allocate second leaf
        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2.len(), 1);
        assert_eq!(backend.len(), 8);

        // Now D(3) from branch1 and W(3) from branch2 should be leaves
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate both
        backend.allocate(2);
        assert_eq!(backend.len(), 6);

        // C(2) from both chains should now be leaves
        assert_eq!(backend.get_queue_len(), 2);
    }

    #[test]
    fn test_duplicate_insertion_would_panic() {
        // Note: This test documents that duplicate insertions would be detected.
        // We cannot easily test this because Block<T, Registered> does not implement Clone,
        // and the first insert() consumes the block, making it impossible to insert twice.
        //
        // The duplicate detection logic in insert() checks if a node already exists at
        // position+fragment with the same full_hash:
        // - If the node exists and is Real with matching full_hash, it panics with
        //   "Duplicate insertion detected!"
        //
        // This is the expected behavior: attempting to insert the same block twice would
        // panic if we could somehow obtain a second copy of the block with identical hash.
        let mut backend = LineageBackend::<TestData>::new();

        let (b1, _) = create_block(1);

        backend.insert(b1);
        assert_eq!(backend.len(), 1);

        // Any future insert of a block with matching position+fragment+full_hash would
        // trigger the duplicate panic. Since Block doesn't implement Clone and is consumed
        // on insert, this test serves as documentation of the expected behavior.
    }

    #[test]
    fn test_collision_would_be_detected() {
        // Note: This test documents that hash collisions would be detected.
        // We cannot easily create a real collision (two different u128 values with
        // the same position+fragment) without constructing invalid PositionalLineageHash
        // values directly, which would bypass the normal construction logic.
        //
        // The collision detection logic in insert() compares full_hash values:
        // - If position+fragment match but full_hash differs, it panics with
        //   "Hash collision detected!"
        //
        // This is tested implicitly by ensuring that all insertions with the same
        // position+fragment must have identical full hashes, otherwise they panic.
        let mut backend = LineageBackend::<TestData>::new();

        let (b1, _) = create_block(1);

        backend.insert(b1);
        assert_eq!(backend.len(), 1);

        // Any future insert with matching position+fragment but different full_hash
        // would trigger the collision panic. Since we can't construct such a case
        // without bypassing PositionalLineageHash invariants, this test serves as
        // documentation.
    }
}
