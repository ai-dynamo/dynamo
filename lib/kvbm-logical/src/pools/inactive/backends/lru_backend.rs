// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LRU-backed inactive index — least-recently-inserted block evicted first.

use std::num::NonZeroUsize;

use lru::LruCache;

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::store::InactiveIndex;

pub(crate) struct LruBackend {
    cache: LruCache<SequenceHash, BlockId>,
}

impl LruBackend {
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(capacity),
        }
    }
}

impl InactiveIndex for LruBackend {
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(block_id) = self.cache.pop(hash) {
                matches.push((*hash, block_id));
            } else {
                break;
            }
        }
        matches
    }

    fn find_match(&mut self, hash: SequenceHash, _touch: bool) -> Option<(SequenceHash, BlockId)> {
        let block_id = self.cache.pop(&hash)?;
        Some((hash, block_id))
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::new();
        for hash in hashes {
            if let Some(block_id) = self.cache.pop(hash) {
                matches.push((*hash, block_id));
            }
        }
        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some((seq_hash, block_id)) = self.cache.pop_lru() {
                allocated.push((seq_hash, block_id));
            } else {
                break;
            }
        }
        allocated
    }

    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        assert!(
            self.cache.len() < self.cache.cap().get(),
            "LRU backend insert would cause eviction! len={}, cap={}. \
             This indicates insufficient capacity for all blocks.",
            self.cache.len(),
            self.cache.cap().get()
        );
        self.cache.put(seq_hash, block_id);
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn has(&self, seq_hash: SequenceHash) -> bool {
        self.cache.peek(&seq_hash).is_some()
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        if self.cache.peek(&seq_hash) == Some(&block_id) {
            self.cache.pop(&seq_hash);
            true
        } else {
            false
        }
    }

    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(self.cache.len());
        while let Some((seq_hash, block_id)) = self.cache.pop_lru() {
            allocated.push((seq_hash, block_id));
        }
        allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{block_id_and_hash, tokens_for_id};

    #[test]
    fn test_lru_eviction_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(3).unwrap());

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        backend.insert(hash3, id3);

        assert_eq!(backend.len(), 3);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].1, 1);

        assert!(!backend.has(hash1));
        assert!(backend.has(hash2));
        assert!(backend.has(hash3));
    }

    #[test]
    fn test_lru_peek_doesnt_affect_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(3).unwrap());

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        assert_eq!(backend.len(), 2);

        assert!(backend.has(hash1));
        assert!(backend.has(hash2));

        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);
        assert_eq!(allocated[0].1, 1);
        assert_eq!(allocated[1].1, 2);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_lru_allocate_more_than_available() {
        let mut backend = LruBackend::new(NonZeroUsize::new(10).unwrap());

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        backend.insert(hash1, id1);
        backend.insert(hash2, id2);

        let allocated = backend.allocate(5);
        assert_eq!(allocated.len(), 2);
        assert_eq!(backend.len(), 0);
    }
}
