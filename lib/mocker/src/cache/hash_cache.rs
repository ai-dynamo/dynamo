// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::evictor::LRUEvictor;
use dynamo_tokens::blocks::UniqueBlock;
use std::collections::HashMap;

/// Hash-based KV cache with O(1) block lookups, maintaining active (ref-counted) and
/// inactive (LRU-evictable) pools.
pub struct HashCache {
    active_blocks: HashMap<UniqueBlock, usize>,
    inactive_blocks: LRUEvictor<UniqueBlock>,
    max_capacity: usize,
}

impl HashCache {
    /// Create a new HashCache with the given maximum block capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            active_blocks: HashMap::new(),
            inactive_blocks: LRUEvictor::default(),
            max_capacity,
        }
    }

    /// Get the reference count of an active block, if it exists.
    pub fn get_active_ref_count(&self, block: &UniqueBlock) -> Option<usize> {
        self.active_blocks.get(block).copied()
    }

    /// Increment the reference count of an active block. Returns the new count.
    pub fn increment_ref(&mut self, block: &UniqueBlock) -> usize {
        let ref_count = self
            .active_blocks
            .get_mut(block)
            .expect("block must be active to increment ref");
        *ref_count += 1;
        *ref_count
    }

    /// Decrement the reference count of an active block. Returns the new count.
    pub fn decrement_ref(&mut self, block: &UniqueBlock) -> usize {
        let ref_count = self
            .active_blocks
            .get_mut(block)
            .expect("block must be active to decrement ref");
        *ref_count -= 1;
        *ref_count
    }

    /// Insert a block into the active pool with the given reference count.
    pub fn insert_active(&mut self, block: UniqueBlock, ref_count: usize) {
        self.active_blocks.insert(block, ref_count);
    }

    /// Remove a block from the active pool. Returns the reference count, or None if not found.
    pub fn remove_active(&mut self, block: &UniqueBlock) -> Option<usize> {
        self.active_blocks.remove(block)
    }

    /// Check if a block is in the active pool.
    pub fn contains_active(&self, block: &UniqueBlock) -> bool {
        self.active_blocks.contains_key(block)
    }

    /// Insert a block into the inactive pool (LRU order).
    pub fn insert_inactive(&mut self, block: UniqueBlock) {
        self.inactive_blocks.insert(block);
    }

    /// Remove a block from the inactive pool. Returns true if it was found.
    pub fn remove_inactive(&mut self, block: &UniqueBlock) -> bool {
        self.inactive_blocks.remove(block)
    }

    /// Evict the least-recently-used block from the inactive pool.
    pub fn evict_inactive(&mut self) -> Option<UniqueBlock> {
        self.inactive_blocks.evict()
    }

    /// Check if a block is in the inactive pool.
    pub fn contains_inactive(&self, block: &UniqueBlock) -> bool {
        self.inactive_blocks.contains(block)
    }

    /// Check if a block exists in either active or inactive pool.
    pub fn contains(&self, block: &UniqueBlock) -> bool {
        self.active_blocks.contains_key(block) || self.inactive_blocks.contains(block)
    }

    /// Move block from active to inactive (ref_count reached 0).
    pub fn deactivate(&mut self, block: &UniqueBlock) {
        debug_assert!(
            self.active_blocks.contains_key(block),
            "deactivate called on non-active block"
        );
        debug_assert!(
            !self.inactive_blocks.contains(block),
            "deactivate called on already-inactive block"
        );
        self.active_blocks.remove(block);
        self.inactive_blocks.insert(block.clone());
    }

    /// Move block from inactive to active with ref_count=1. Returns true if found.
    pub fn reactivate(&mut self, block: &UniqueBlock) -> bool {
        if self.inactive_blocks.remove(block) {
            self.active_blocks.insert(block.clone(), 1);
            true
        } else {
            false
        }
    }

    /// Check if total blocks (active + inactive) has reached max_capacity.
    pub fn is_at_capacity(&self) -> bool {
        self.active_blocks.len() + self.inactive_blocks.len() >= self.max_capacity
    }

    /// Get the number of active blocks.
    pub fn num_active(&self) -> usize {
        self.active_blocks.len()
    }

    /// Get the number of inactive blocks.
    pub fn num_inactive(&self) -> usize {
        self.inactive_blocks.len()
    }

    /// Get the maximum block capacity.
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Get the current capacity (active + inactive blocks).
    pub fn current_capacity(&self) -> usize {
        self.active_blocks.len() + self.inactive_blocks.len()
    }

    /// Iterate over active block keys.
    pub fn active_keys(&self) -> impl Iterator<Item = &UniqueBlock> {
        self.active_blocks.keys()
    }

    /// Iterate over inactive block keys.
    pub fn inactive_keys(&self) -> impl Iterator<Item = &UniqueBlock> {
        self.inactive_blocks.keys()
    }

    /// Direct access to active blocks map (for tests that check ref counts).
    pub fn active_blocks(&self) -> &HashMap<UniqueBlock, usize> {
        &self.active_blocks
    }
}
