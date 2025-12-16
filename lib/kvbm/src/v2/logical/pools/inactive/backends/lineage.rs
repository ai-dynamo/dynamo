// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use dynamo_tokens::PositionalLineageHash;

use super::super::{Block, BlockMetadata, Registered};

/// A node in the lineage graph.
struct LineageNode<T: BlockMetadata> {
    /// The block stored at this node, if any.
    block: Option<Block<T, Registered>>,

    /// The sequence hash fragment of this node (redundant but useful).
    #[allow(dead_code)]
    fragment: u64,

    /// The position of this node.
    #[allow(dead_code)]
    position: u64,

    /// The parent fragment (at position - 1), if any.
    parent_fragment: Option<u64>,

    /// Children fragments (at position + 1).
    children: HashSet<u64>,

    /// The tick when this block was inserted into the pool.
    /// Used for LRU ordering.
    last_used: u64,
}

impl<T: BlockMetadata> LineageNode<T> {
    fn new(block: Block<T, Registered>, lineage_hash: PositionalLineageHash, tick: u64) -> Self {
        let parent_fragment = if lineage_hash.position() > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        Self {
            block: Some(block),
            fragment: lineage_hash.current_hash_fragment(),
            position: lineage_hash.position(),
            parent_fragment,
            children: HashSet::new(),
            last_used: tick,
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

    /// Maximum capacity (total blocks).
    capacity: usize,

    /// Monotonic counter for insertion ordering.
    current_tick: u64,
}

impl<T: BlockMetadata> LineageBackend<T> {
    /// Creates a new LineageBackend.
    pub fn new(capacity: std::num::NonZeroUsize) -> Self {
        Self {
            nodes: HashMap::new(),
            leaf_queue: BTreeMap::new(),
            count: 0,
            capacity: capacity.get(),
            current_tick: 0,
        }
    }

    /// Inserts a block into the lineage graph.
    /// Panics if capacity is exceeded.
    pub fn insert(&mut self, block: Block<T, Registered>, lineage_hash: PositionalLineageHash) {
        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();
        let parent_fragment = if position > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        let mut increment_count = false;
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
                if node.block.is_none() {
                    increment_count = true;
                } else {
                    // If block existed, we are updating it. Remove from leaf_queue if it was there
                    // because we will update its timestamp and potentially re-add it.
                    if node.is_leaf() {
                        self.leaf_queue.remove(&(node.last_used, position, fragment));
                    }
                }
                node.block = Some(block);
                node.parent_fragment = parent_fragment;
                node.last_used = tick;
            }
        }

        if increment_count {
            if self.count >= self.capacity {
                panic!(
                    "Lineage backend insert would cause overflow! len={}, cap={}. \
                     This indicates insufficient capacity for all blocks.",
                    self.count, self.capacity
                );
            }
            self.count += 1;
        }

        // 2. Link to parent
        if let Some(p_frag) = parent_fragment {
            let p_pos = position - 1;

            let parent_level = self.nodes.entry(p_pos).or_default();
            let parent_node = parent_level.entry(p_frag).or_insert_with(|| {
                LineageNode {
                    block: None, // Ghost node
                    fragment: p_frag,
                    position: p_pos,
                    parent_fragment: None, // We don't know the parent's parent yet
                    children: HashSet::new(),
                    last_used: 0, // Irrelevant for ghost
                }
            });

            let was_parent_leaf = parent_node.is_leaf();
            parent_node.children.insert(fragment);

            if was_parent_leaf {
                // Parent was a leaf, now has a child. Remove from queue.
                // Note: Ghost nodes (block=None) are never in queue, but check is cheap.
                if parent_node.block.is_some() {
                    self.leaf_queue.remove(&(parent_node.last_used, p_pos, p_frag));
                }
            }
        }

        // 3. Update LRU status for this node
        let node = self.nodes.get(&position).unwrap().get(&fragment).unwrap();
        if node.is_leaf() {
             self.leaf_queue.insert((node.last_used, position, fragment), ());
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

        let node_data = self.nodes.get(&position)
            .and_then(|level| level.get(&fragment))
            .map(|node| (node.block.is_some(), node.last_used));

        if let Some((has_block, tick)) = node_data {
            if !has_block {
                return None;
            }
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
            node.block.take()
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

            if let Some(level) = self.nodes.get(&current_pos) {
                if let Some(node) = level.get(&current_frag) {
                    if node.children.is_empty() && node.block.is_none() {
                        // It's a ghost leaf (no block, no children). Prune it.
                        should_remove_node = true;
                        parent_info = node.parent_fragment.map(|pf| (current_pos.saturating_sub(1), pf));
                    }
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

                    if let Some(level) = self.nodes.get_mut(&p_pos) {
                        if let Some(parent) = level.get_mut(&p_frag) {
                            parent.children.remove(&current_frag);
                            if parent.children.is_empty() {
                                parent_became_leaf = true;
                                parent_has_block = parent.block.is_some();
                                parent_tick = parent.last_used;
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

    pub fn len(&self) -> usize {
        self.count
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[allow(dead_code)]
    pub fn get_queue_len(&self) -> usize {
        self.leaf_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::logical::blocks::{Block, BlockRegistry};
    use dynamo_tokens::{PositionalLineageHash, TokenBlockSequence};
    use crate::v2::SequenceHash;
    use std::num::NonZeroUsize;

    #[derive(Clone, Debug, PartialEq)]
    struct TestData;

    // Helper to create a dummy registered block
    fn create_block(id: usize) -> Block<TestData, Registered> {
        let registry = BlockRegistry::new();
        let seq_hash = SequenceHash::default();
        let handle = registry.register_sequence_hash(seq_hash);

        let block = Block::new(id, 1);
        let tokens = vec![1u32];
        let binding = TokenBlockSequence::from_slice(&tokens, 1, None);
        let token_block = binding.blocks().into_iter().next().unwrap();

        let completed = block.complete(token_block.clone()).unwrap();
        completed.register(handle)
    }

    fn make_hash(pos: u64, current: u64, parent: u64) -> PositionalLineageHash {
         PositionalLineageHash::new(current, if pos > 0 { Some(parent) } else { None }, pos)
    }

    #[test]
    fn test_leaf_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0); // Root

        backend.insert(b1, h1);

        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1); // It is a leaf (no children)

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 1);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_parent_child_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100); // Child of h1

        // Insert parent first
        backend.insert(b1, h1);
        assert_eq!(backend.get_queue_len(), 1); // h1 is leaf

        // Insert child
        backend.insert(b2, h2);
        assert_eq!(backend.len(), 2);

        // h1 is no longer leaf (has child h2). h2 is leaf.
        // LRU should contain only h2.
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 2); // Should allocate h2 (leaf)

        // Now h1 should be a leaf again and added to LRU
        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(allocated2[0].block_id(), 1);
    }

    #[test]
    fn test_out_of_order_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100);

        // Insert child first
        backend.insert(b2, h2);
        // Created ghost node for parent h1.
        // h2 is leaf.
        assert_eq!(backend.len(), 1); // Only 1 actual block
        assert_eq!(backend.get_queue_len(), 1);

        // Insert parent
        backend.insert(b1, h1);
        // Parent h1 fills ghost. It has child h2, so it's NOT a leaf.
        // h2 is still leaf.

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1); // Only h2

        let allocated = backend.allocate(1);
        assert_eq!(allocated[0].block_id(), 2);

        // Now h1 becomes leaf
        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2[0].block_id(), 1);
    }

    #[test]
    fn test_branching() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let root = create_block(1);
        let root_hash = make_hash(0, 100, 0);

        let child1 = create_block(2);
        let child1_hash = make_hash(1, 201, 100);

        let child2 = create_block(3);
        let child2_hash = make_hash(1, 202, 100);

        backend.insert(root, root_hash);
        backend.insert(child1, child1_hash);
        backend.insert(child2, child2_hash);

        // Root has 2 children.
        // LRU should have child1 and child2. Root is not leaf.
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate 2 blocks (both children)
        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);

        // Now root should be leaf
        assert_eq!(backend.get_queue_len(), 1);

        let allocated_root = backend.allocate(1);
        assert_eq!(allocated_root[0].block_id(), 1);
    }

    #[test]
    fn test_interleaved_chains() {
        // Chain 1: A(0) -> B(1)
        // Chain 2: X(10) -> Y(11)
        // We want strict consumption of older chain.
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let a = create_block(1);
        let b = create_block(2);
        let x = create_block(3);
        let y = create_block(4);

        // Manually manipulate ticks? No, just insert in order.
        // insert(A) tick 0
        // insert(B) tick 1
        // insert(X) tick 2
        // insert(Y) tick 3
        // So Chain 1 is older.

        backend.insert(a, make_hash(0, 100, 0));
        backend.insert(b, make_hash(1, 101, 100));

        backend.insert(x, make_hash(0, 200, 0));
        backend.insert(y, make_hash(1, 201, 200));

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_queue_len(), 2); // Leaves: B, Y

        // B (tick 1) is older than Y (tick 3). Expect B.
        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1[0].block_id(), 2); // B

        // Now A becomes leaf. A has tick 0.
        // Queue: A(0), Y(3).
        // Expect A.
        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].block_id(), 1); // A

        // Now Y(3).
        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].block_id(), 4); // Y

        // Now X becomes leaf. X has tick 2.
        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].block_id(), 3); // X
    }

    #[test]
    #[should_panic(expected = "Lineage backend insert would cause overflow")]
    fn test_capacity_enforcement() {
        // Capacity 2
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100);

        backend.insert(b1, h1);
        backend.insert(b2, h2);

        assert_eq!(backend.len(), 2);

        // Insert 3rd block - should panic
        let b3 = create_block(3);
        let h3 = make_hash(0, 300, 0);

        backend.insert(b3, h3);
    }

    #[test]
    fn test_remove_by_hash() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        backend.insert(b1, h1);
        assert_eq!(backend.len(), 1);

        let removed = backend.remove(&h1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().block_id(), 1);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_deep_chain_cleanup_iterative() {
        // Create deep chain: 0 -> 1 -> 2 ... -> 1000
        let depth = 1000;
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2000).unwrap());

        for i in 0..depth {
            let b = create_block(i);
            let h = make_hash(i as u64, 100 + i as u64, if i > 0 { 100 + i as u64 - 1 } else { 0 });
            backend.insert(b, h);
        }

        assert_eq!(backend.len(), depth);
        // Only last one is leaf
        assert_eq!(backend.get_queue_len(), 1);

        let last_h = make_hash((depth-1) as u64, 100 + (depth-1) as u64, 100 + (depth-2) as u64);
        backend.remove(&last_h);

        assert_eq!(backend.len(), depth - 1);
        // Now 998 is leaf
        assert_eq!(backend.get_queue_len(), 1);

        // Now insert a chain out of order to create ghosts, then delete leaf to trigger cleanup
        backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2000).unwrap());

        let leaf_idx = 100;
        let b_leaf = create_block(leaf_idx);
        let h_leaf = make_hash(leaf_idx as u64, 200 + leaf_idx as u64, 200 + leaf_idx as u64 - 1);

        // Insert leaf at depth 100. This creates 100 ghost parents.
        backend.insert(b_leaf, h_leaf);

        assert_eq!(backend.len(), 1); // Only 1 real block
        // Ghost nodes exist but are not counted in len

        // Remove leaf. This should iteratively clean up all 100 ghosts.
        backend.remove(&h_leaf);

        assert_eq!(backend.len(), 0);
        assert!(backend.nodes.is_empty());
    }

    #[test]
    fn test_split_sequence_eviction() {
        // Scenario: A(0)->B(1)->C(2)-> {D(3)->E(4), F(3)->G(4)}
        // Leaves: E and G.
        // We want to ensure C is not exposed until both branches are gone.
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(20).unwrap());

        // Create blocks
        let a = create_block(0); let ha = make_hash(0, 100, 0);
        let b = create_block(1); let hb = make_hash(1, 101, 100);
        let c = create_block(2); let hc = make_hash(2, 102, 101);

        let d = create_block(3); let hd = make_hash(3, 103, 102);
        let e = create_block(4); let he = make_hash(4, 104, 103);

        let f = create_block(5); let hf = make_hash(3, 105, 102);
        let g = create_block(6); let hg = make_hash(4, 106, 105);

        // Insert
        backend.insert(a, ha);
        backend.insert(b, hb);
        backend.insert(c, hc);
        backend.insert(d, hd);
        backend.insert(e, he);
        backend.insert(f, hf);
        backend.insert(g, hg);

        assert_eq!(backend.len(), 7);
        // Leaves are E and G.
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate E (D->E branch tip)
        let alloc1 = backend.allocate(1);
        assert!(alloc1[0].block_id() == 4 || alloc1[0].block_id() == 6); // E or G

        // Assume E was allocated (based on insertion order E before G?
        // E tick=4, G tick=6. E is older. So E should be allocated first.
        assert_eq!(alloc1[0].block_id(), 4); // E

        // Now D is a leaf. D tick=3. G tick=6.
        // Queue: D(3), G(6).
        assert_eq!(backend.get_queue_len(), 2);

        // Allocate D.
        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].block_id(), 3); // D

        // Now D is gone. D's parent is C.
        // C has another child F. So C is NOT a leaf.
        // Queue should only have G(6).
        assert_eq!(backend.get_queue_len(), 1);

        // Allocate G.
        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].block_id(), 6); // G

        // Now F is a leaf. F tick=5.
        // Queue: F(5).
        assert_eq!(backend.get_queue_len(), 1);

        // Allocate F.
        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].block_id(), 5); // F

        // Now F is gone. F's parent is C.
        // C has no more children. C is now a leaf.
        // C should be in Queue.
        assert_eq!(backend.get_queue_len(), 1);

        let alloc5 = backend.allocate(1);
        assert_eq!(alloc5[0].block_id(), 2); // C
    }
}
