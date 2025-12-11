// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sparse radix tree for efficient distributed block lookup.
//!
//! The sparse radix tree uses power-of-2 positions as indices to create
//! a hierarchical structure for narrowing down block search space across
//! the fleet of workers.

use std::collections::HashSet;

use dashmap::DashMap;
use dynamo_tokens::PositionalRadixTree as TokensPositionalRadixTree;

use super::events::InstanceId;
use crate::v2::PositionalSequenceHash;

/// Sparse radix tree for tracking block locations across workers.
pub struct PositionalRadixTree {
    /// Map of positions to sequence hashes to instance IDs
    levels: TokensPositionalRadixTree<HashSet<InstanceId>>,
    workers: DashMap<InstanceId, HashSet<PositionalSequenceHash>>,
}

impl PositionalRadixTree {
    /// Creates a new sparse radix tree.
    pub fn new() -> Self {
        Self {
            levels: TokensPositionalRadixTree::new(),
            workers: DashMap::new(),
        }
    }

    /// Inserts a block location into the tree.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash of the block
    /// * `instance_id` - The worker instance that has the block
    pub fn insert(&self, seq_hash: PositionalSequenceHash, instance_id: InstanceId) {
        self.levels
            .prefix(&seq_hash)
            .entry(seq_hash)
            .or_default()
            .insert(instance_id);
        self.workers
            .entry(instance_id)
            .or_default()
            .insert(seq_hash);
    }

    /// Removes a block location from the tree.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash of the block
    /// * `instance_id` - The worker instance to remove
    pub fn remove(&self, seq_hash: PositionalSequenceHash, instance_id: InstanceId) {
        self.levels
            .prefix(&seq_hash)
            .get_mut(&seq_hash)
            .map(|mut map| map.remove(&instance_id));
        self.workers
            .get_mut(&instance_id)
            .map(|mut set| set.remove(&seq_hash));
    }

    /// Looks up which workers have a specific block.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash to look up
    ///
    /// # Returns
    /// Some(HashSet) of instance IDs if found, None otherwise
    pub fn lookup(&self, seq_hash: PositionalSequenceHash) -> Option<HashSet<InstanceId>> {
        self.levels
            .prefix(&seq_hash)
            .get(&seq_hash)
            .map(|entry| entry.value().clone())
    }

    /// Evaluate a function for a specific sequence hash.
    ///
    /// # Arguments
    /// * `seq_hash` - The positional sequence hash to evaluate the function for
    /// * `f` - The function to evaluate. The function will be passed a reference to the set of instance IDs that have the sequence hash.
    pub fn apply(&self, seq_hash: PositionalSequenceHash, mut f: impl FnMut(&HashSet<InstanceId>)) {
        if let Some(entry) = self.levels.prefix(&seq_hash).get(&seq_hash) {
            f(&entry)
        }
    }

    /// Returns the number of unique blocks tracked across all levels.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Returns the number of unique blocks tracked by a specific instance.
    pub fn instance_count(&self) -> usize {
        self.workers.len()
    }

    pub fn remove_instance(&self, instance_id: InstanceId) {
        // for each sequence hash in the instance, remove it from the tree
        if let Some((_, sequence_hashes)) = self.workers.remove(&instance_id) {
            // todo: parallelize the removal with rayon?
            for seq_hash in sequence_hashes {
                self.remove(seq_hash, instance_id);
            }
        }
    }
}

impl Default for PositionalRadixTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::TokenBlockSequence;

    fn create_seq_hash_at_position(position: usize) -> PositionalSequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].positional_sequence_hash()
    }

    #[test]
    fn test_insert_and_lookup() {
        let tree = PositionalRadixTree::new();
        let seq_hash = create_seq_hash_at_position(16);
        let instance_id = 12345;

        tree.insert(seq_hash, instance_id);

        let result = tree.lookup(seq_hash).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&instance_id));
    }

    #[test]
    fn test_insert_multiple_instances() {
        let tree = PositionalRadixTree::new();
        let seq_hash = create_seq_hash_at_position(32);

        tree.insert(seq_hash, 111);
        tree.insert(seq_hash, 222);
        tree.insert(seq_hash, 333);

        let result = tree.lookup(seq_hash).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&111));
        assert!(result.contains(&222));
        assert!(result.contains(&333));
    }

    #[test]
    fn test_remove() {
        let tree = PositionalRadixTree::new();
        let seq_hash = create_seq_hash_at_position(64);

        tree.insert(seq_hash, 111);
        tree.insert(seq_hash, 222);

        tree.remove(seq_hash, 111);

        let result = tree.lookup(seq_hash).unwrap();
        assert_eq!(result.len(), 1);
        assert!(!result.contains(&111));
        assert!(result.contains(&222));
    }

    #[test]
    fn test_lookup_nonexistent() {
        let tree = PositionalRadixTree::new();
        let seq_hash = create_seq_hash_at_position(128);

        assert!(tree.lookup(seq_hash).is_none());
    }

    #[test]
    fn test_len_and_is_empty() {
        let tree = PositionalRadixTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        let hash_16 = create_seq_hash_at_position(16);
        let hash_32 = create_seq_hash_at_position(32);

        tree.insert(hash_16, 111);
        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());

        tree.insert(hash_32, 222);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_apply_with_mutable_closure() {
        let tree = PositionalRadixTree::new();
        let seq_hash = create_seq_hash_at_position(16);

        // Insert some instances
        tree.insert(seq_hash, 111);
        tree.insert(seq_hash, 222);
        tree.insert(seq_hash, 333);

        // Test with a struct that needs mutable access
        struct InstanceCollector {
            collected: Vec<InstanceId>,
        }

        impl InstanceCollector {
            fn new() -> Self {
                Self {
                    collected: Vec::new(),
                }
            }

            fn update(&mut self, instances: &HashSet<InstanceId>) {
                self.collected.extend(instances);
                self.collected.sort();
            }
        }

        let mut collector = InstanceCollector::new();

        // This will work with FnMut but not with Fn
        tree.apply(seq_hash, |instances| {
            collector.update(instances);
        });

        assert_eq!(collector.collected.len(), 3);
        assert!(collector.collected.contains(&111));
        assert!(collector.collected.contains(&222));
        assert!(collector.collected.contains(&333));

        // Test that apply doesn't call the function if seq_hash doesn't exist
        let nonexistent = create_seq_hash_at_position(999);
        let mut call_count = 0;
        tree.apply(nonexistent, |_| {
            call_count += 1;
        });
        assert_eq!(call_count, 0);
    }
}
