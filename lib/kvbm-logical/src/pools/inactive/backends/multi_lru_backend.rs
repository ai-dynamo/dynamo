// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! 4-tier frequency-aware LRU inactive index — buckets blocks by their
//! TinyLFU access frequency and evicts cold tiers first.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use anyhow::{Result, bail};

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::IdBuildHasher;
use crate::pools::store::InactiveIndex;
use crate::tinylfu::FrequencyTracker;

pub(crate) struct MultiLruBackend {
    /// Identity-hashed: `SequenceHash` is already a content hash.
    priority_pools: [LruCache<SequenceHash, BlockId, IdBuildHasher>; 4],
    frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
    frequency_thresholds: [u8; 3],
}

impl MultiLruBackend {
    /// Create with custom frequency thresholds. The 4 levels are fixed; only
    /// the boundaries between them are configurable.
    pub(crate) fn new_with_thresholds(
        block_count: NonZeroUsize,
        thresholds: &[u8; 3],
        frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
    ) -> Result<Self> {
        if !(thresholds[0] < thresholds[1] && thresholds[1] < thresholds[2]) {
            bail!("Thresholds must be in ascending order: {:?}", thresholds);
        }
        if thresholds[2] > 15 {
            bail!(
                "Maximum threshold cannot exceed 15 (4-bit counter limit), got: {:?}",
                thresholds
            );
        }
        if thresholds[0] < 1 {
            bail!(
                "Cold threshold must be >= 1 to distinguish from never-accessed blocks, got: {:?}",
                thresholds
            );
        }
        Ok(Self {
            priority_pools: [
                LruCache::with_hasher(block_count, IdBuildHasher),
                LruCache::with_hasher(block_count, IdBuildHasher),
                LruCache::with_hasher(block_count, IdBuildHasher),
                LruCache::with_hasher(block_count, IdBuildHasher),
            ],
            frequency_tracker,
            frequency_thresholds: *thresholds,
        })
    }

    fn calculate_priority_level(&self, seq_hash: SequenceHash) -> usize {
        let frequency = self.frequency_tracker.count(seq_hash.as_u128());
        let [t1, t2, t3] = self.frequency_thresholds;

        if frequency < t1 as u32 {
            0
        } else if frequency < t2 as u32 {
            1
        } else if frequency < t3 as u32 {
            2
        } else {
            3
        }
    }
}

impl InactiveIndex for MultiLruBackend {
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        // The registry calls `frequency_tracker.touch()` at the API
        // boundary (`BlockRegistry::match_sequence_hash`), so backends
        // must NOT touch again here — doing so would double-count and
        // skew tier decisions. The `touch` flag is reserved for any
        // future backend-internal recency promotion that is *not* a
        // frequency-tracker increment.
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            let mut found = false;
            for pool in &mut self.priority_pools {
                if let Some(block_id) = pool.pop(hash) {
                    matches.push((*hash, block_id));
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        matches
    }

    fn find_match(
        &mut self,
        hash: SequenceHash,
        // See find_matches: registry calls frequency_tracker.touch()
        // at the API boundary; backends must not touch again.
        _touch: bool,
    ) -> Option<(SequenceHash, BlockId)> {
        for pool in &mut self.priority_pools {
            if let Some(block_id) = pool.pop(&hash) {
                return Some((hash, block_id));
            }
        }
        None
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::new();
        for hash in hashes {
            for pool in &mut self.priority_pools {
                if let Some(block_id) = pool.pop(hash) {
                    matches.push((*hash, block_id));
                    break;
                }
            }
        }
        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            let mut found = false;
            for pool in &mut self.priority_pools {
                if let Some((seq_hash, block_id)) = pool.pop_lru() {
                    allocated.push((seq_hash, block_id));
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        allocated
    }

    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        let level = self.calculate_priority_level(seq_hash);
        debug_assert!(
            self.priority_pools[level].len() < self.priority_pools[level].cap().get(),
            "MultiLRU level {} insert would cause eviction! len={}, cap={}.",
            level,
            self.priority_pools[level].len(),
            self.priority_pools[level].cap().get()
        );
        self.priority_pools[level].put(seq_hash, block_id);
    }

    fn len(&self) -> usize {
        self.priority_pools.iter().map(|pool| pool.len()).sum()
    }

    fn has(&self, seq_hash: SequenceHash) -> bool {
        self.priority_pools
            .iter()
            .any(|pool| pool.peek(&seq_hash).is_some())
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        for pool in &mut self.priority_pools {
            if pool.peek(&seq_hash) == Some(&block_id) {
                pool.pop(&seq_hash);
                return true;
            }
        }
        false
    }

    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let total_len: usize = self.priority_pools.iter().map(|p| p.len()).sum();
        let mut allocated = Vec::with_capacity(total_len);
        for pool in &mut self.priority_pools {
            while let Some((seq_hash, block_id)) = pool.pop_lru() {
                allocated.push((seq_hash, block_id));
            }
        }
        allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{block_id_and_hash, tokens_for_id};
    use crate::tinylfu::TinyLFUTracker;

    impl MultiLruBackend {
        pub fn new(
            capacity: NonZeroUsize,
            frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
        ) -> Self {
            Self::new_with_thresholds(capacity, &[2, 6, 15], frequency_tracker).unwrap()
        }
    }

    #[test]
    fn test_multi_lru_priority_levels() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend =
            MultiLruBackend::new(NonZeroUsize::new(12).unwrap(), frequency_tracker.clone());

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));
        let (id4, hash4) = block_id_and_hash(4, &tokens_for_id(4));

        frequency_tracker.touch(hash2.as_u128());
        frequency_tracker.touch(hash2.as_u128());

        for _ in 0..6 {
            frequency_tracker.touch(hash3.as_u128());
        }

        for _ in 0..16 {
            frequency_tracker.touch(hash4.as_u128());
        }

        assert_eq!(backend.calculate_priority_level(hash1), 0); // Cold
        assert_eq!(backend.calculate_priority_level(hash2), 1); // Warm
        assert_eq!(backend.calculate_priority_level(hash3), 2); // Hot
        assert_eq!(backend.calculate_priority_level(hash4), 3); // Very hot

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        backend.insert(hash3, id3);
        backend.insert(hash4, id4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has(hash1));
        assert!(backend.has(hash2));
        assert!(backend.has(hash3));
        assert!(backend.has(hash4));
    }

    #[test]
    fn test_multi_lru_eviction_order() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend =
            MultiLruBackend::new(NonZeroUsize::new(8).unwrap(), frequency_tracker.clone());

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));

        for _ in 0..6 {
            frequency_tracker.touch(hash3.as_u128());
        }

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        backend.insert(hash3, id3);

        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);
        assert_eq!(allocated[0].1, 1);
        assert_eq!(allocated[1].1, 2);

        assert!(!backend.has(hash1));
        assert!(!backend.has(hash2));
        assert!(backend.has(hash3));
    }

    #[test]
    fn test_multi_lru_find_matches() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new_with_thresholds(
            NonZeroUsize::new(8).unwrap(),
            &[2, 4, 8],
            frequency_tracker.clone(),
        )
        .unwrap();

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));

        for _ in 0..3 {
            frequency_tracker.touch(hash2.as_u128());
        }

        for _ in 0..10 {
            frequency_tracker.touch(hash3.as_u128());
        }

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        backend.insert(hash3, id3);

        let matches = backend.find_matches(&[hash1, hash2, hash3], true);
        assert_eq!(matches.len(), 3);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_multi_lru_capacity_distribution() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new_with_thresholds(
            NonZeroUsize::new(16).unwrap(),
            &[2, 6, 15],
            frequency_tracker.clone(),
        )
        .unwrap();

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));
        let (id4, hash4) = block_id_and_hash(4, &tokens_for_id(4));

        for _ in 0..3 {
            frequency_tracker.touch(hash2.as_u128());
        }

        for _ in 0..7 {
            frequency_tracker.touch(hash3.as_u128());
        }

        for _ in 0..15 {
            frequency_tracker.touch(hash4.as_u128());
        }

        assert_eq!(backend.calculate_priority_level(hash1), 0);
        assert_eq!(backend.calculate_priority_level(hash2), 1);
        assert_eq!(backend.calculate_priority_level(hash3), 2);
        assert_eq!(backend.calculate_priority_level(hash4), 3);

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);
        backend.insert(hash3, id3);
        backend.insert(hash4, id4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has(hash1));
        assert!(backend.has(hash2));
        assert!(backend.has(hash3));
        assert!(backend.has(hash4));

        let allocated = backend.allocate(4);
        assert_eq!(allocated.len(), 4);
        assert_eq!(backend.len(), 0);
    }
}
