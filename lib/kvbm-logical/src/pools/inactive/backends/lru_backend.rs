// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use lru::LruCache;

use super::{Block, BlockMetadata, Registered, SequenceHash};

use super::super::InactivePoolBackend;

pub struct LruBackend<T: BlockMetadata> {
    cache: LruCache<SequenceHash, Block<T, Registered>>,
}

impl<T: BlockMetadata> LruBackend<T> {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(capacity),
        }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for LruBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash], _touch: bool) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.cache.pop(hash) {
                matches.push(block);
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
    ) -> Vec<(SequenceHash, Block<T, Registered>)> {
        let mut matches = Vec::new();

        for hash in hashes {
            if let Some(block) = self.cache.pop(hash) {
                matches.push((*hash, block));
            }
            // Unlike find_matches: NO break on miss - continue scanning
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some((_seq_hash, block)) = self.cache.pop_lru() {
                allocated.push(block);
            } else {
                break;
            }
        }

        allocated
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        let seq_hash = block.sequence_hash();

        // Assert we're not causing an eviction
        assert!(
            self.cache.len() < self.cache.cap().get(),
            "LRU backend insert would cause eviction! len={}, cap={}. \
             This indicates insufficient capacity for all blocks.",
            self.cache.len(),
            self.cache.cap().get()
        );

        self.cache.put(seq_hash, block);
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.cache.peek(&seq_hash).is_some()
    }

    fn allocate_all(&mut self) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(self.cache.len());
        while let Some((_seq_hash, block)) = self.cache.pop_lru() {
            allocated.push(block);
        }
        allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pools::tests::fixtures::*;

    #[test]
    fn test_lru_eviction_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(3).unwrap());

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));
        let (block3, hash3) = create_registered_block(3, &tokens_for_id(3));

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

        assert_eq!(backend.len(), 3);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 1);

        assert!(!backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
    }

    #[test]
    fn test_lru_peek_doesnt_affect_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(3).unwrap());

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));

        backend.insert(block1);
        backend.insert(block2);
        assert_eq!(backend.len(), 2);

        // Peek at block1 (should not affect LRU order)
        assert!(backend.has_block(hash1));
        assert!(backend.has_block(hash2));

        // Allocate blocks - should still follow insertion order (block1 first, then block2)
        // despite the peek at block1
        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);
        assert_eq!(allocated[0].block_id(), 1); // block1 allocated first (oldest)
        assert_eq!(allocated[1].block_id(), 2); // block2 allocated second
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_lru_allocate_more_than_available() {
        let mut backend = LruBackend::new(NonZeroUsize::new(10).unwrap());

        let (block1, _) = create_registered_block(1, &tokens_for_id(1));
        let (block2, _) = create_registered_block(2, &tokens_for_id(2));
        backend.insert(block1);
        backend.insert(block2);

        let allocated = backend.allocate(5);
        assert_eq!(allocated.len(), 2);
        assert_eq!(backend.len(), 0);
    }
}
