// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use super::super::InactivePoolBackend;
use super::{Block, BlockMetadata, Registered, SequenceHash};
use crate::tinylfu::FrequencyTracker;

use anyhow::{Result, bail};

pub struct MultiLruBackend<T: BlockMetadata> {
    priority_pools: [LruCache<SequenceHash, Block<T, Registered>>; 4],
    frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
    frequency_thresholds: [u8; 3],
}

impl<T: BlockMetadata> MultiLruBackend<T> {
    /// Create with custom frequency thresholds
    /// The 4 levels are fixed, but thresholds can be customized
    ///
    /// # Arguments
    /// * `block_count` - Number of blocks in the pool
    /// * `thresholds` - Array of 3 thresholds: [cold->warm, warm->hot, hot->very_hot]
    /// * `frequency_tracker` - Shared frequency tracker
    pub fn new_with_thresholds(
        block_count: NonZeroUsize,
        thresholds: &[u8; 3],
        frequency_tracker: Arc<dyn FrequencyTracker<u128>>,
    ) -> Result<Self> {
        // Validate thresholds
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
                LruCache::new(block_count),
                LruCache::new(block_count),
                LruCache::new(block_count),
                LruCache::new(block_count),
            ],
            frequency_tracker,
            frequency_thresholds: *thresholds,
        })
    }

    fn calculate_priority_level(&self, seq_hash: SequenceHash) -> usize {
        let frequency = self.frequency_tracker.count(seq_hash.as_u128());
        let [t1, t2, t3] = self.frequency_thresholds;

        if frequency < t1 as u32 {
            0 // Cold: 0 to (t1 - 1)
        } else if frequency < t2 as u32 {
            1 // Warm: t1 to (t2 - 1)
        } else if frequency < t3 as u32 {
            2 // Hot: t2 to (t3 - 1)
        } else {
            3 // Very Hot: t3 to 15
        }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for MultiLruBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash], touch: bool) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            let mut found = false;

            for pool in &mut self.priority_pools {
                if let Some(block) = pool.pop(hash) {
                    matches.push(block);
                    if touch {
                        self.frequency_tracker.touch(hash.as_u128());
                    }
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

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Block<T, Registered>)> {
        let mut matches = Vec::new();

        for hash in hashes {
            for pool in &mut self.priority_pools {
                if let Some(block) = pool.pop(hash) {
                    if touch {
                        self.frequency_tracker.touch(hash.as_u128());
                    }
                    matches.push((*hash, block));
                    break; // Found in this pool, move to next hash
                }
            }
            // Unlike find_matches: NO break on miss - continue scanning
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        for _ in 0..count {
            let mut found = false;

            for pool in &mut self.priority_pools {
                if let Some((_seq_hash, block)) = pool.pop_lru() {
                    allocated.push(block);
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

    fn insert(&mut self, block: Block<T, Registered>) {
        let seq_hash = block.sequence_hash();
        let level = self.calculate_priority_level(seq_hash);

        // Assert the target pool isn't full (would cause eviction)
        debug_assert!(
            self.priority_pools[level].len() < self.priority_pools[level].cap().get(),
            "MultiLRU level {} insert would cause eviction! len={}, cap={}. \
             This indicates insufficient capacity for all blocks.",
            level,
            self.priority_pools[level].len(),
            self.priority_pools[level].cap().get()
        );

        self.priority_pools[level].put(seq_hash, block);
    }

    fn len(&self) -> usize {
        self.priority_pools.iter().map(|pool| pool.len()).sum()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.priority_pools
            .iter()
            .any(|pool| pool.peek(&seq_hash).is_some())
    }

    fn allocate_all(&mut self) -> Vec<Block<T, Registered>> {
        let total_len: usize = self.priority_pools.iter().map(|p| p.len()).sum();
        let mut allocated = Vec::with_capacity(total_len);
        for pool in &mut self.priority_pools {
            while let Some((_seq_hash, block)) = pool.pop_lru() {
                allocated.push(block);
            }
        }
        allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pools::tests::fixtures::*;
    use crate::tinylfu::TinyLFUTracker;

    impl<T: BlockMetadata> MultiLruBackend<T> {
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

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));
        let (block3, hash3) = create_registered_block(3, &tokens_for_id(3));
        let (block4, hash4) = create_registered_block(4, &tokens_for_id(4));

        frequency_tracker.touch(hash2.as_u128());
        frequency_tracker.touch(hash2.as_u128());

        for _ in 0..6 {
            frequency_tracker.touch(hash3.as_u128());
        }

        for _ in 0..16 {
            frequency_tracker.touch(hash4.as_u128());
        }

        let _freq1 = frequency_tracker.count(hash1.as_u128());
        let _freq2 = frequency_tracker.count(hash2.as_u128());
        let _freq3 = frequency_tracker.count(hash3.as_u128());
        let _freq4 = frequency_tracker.count(hash4.as_u128());

        assert_eq!(backend.calculate_priority_level(hash1), 0); // Cold
        assert_eq!(backend.calculate_priority_level(hash2), 1); // Warm
        assert_eq!(backend.calculate_priority_level(hash3), 2); // Hot
        assert_eq!(backend.calculate_priority_level(hash4), 3); // Very hot (15)

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);
        backend.insert(block4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
        assert!(backend.has_block(hash4));
    }

    #[test]
    fn test_multi_lru_eviction_order() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend =
            MultiLruBackend::new(NonZeroUsize::new(8).unwrap(), frequency_tracker.clone());

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));
        let (block3, hash3) = create_registered_block(3, &tokens_for_id(3));

        for _ in 0..6 {
            frequency_tracker.touch(hash3.as_u128());
        }

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);
        assert_eq!(allocated[0].block_id(), 1);
        assert_eq!(allocated[1].block_id(), 2);

        assert!(!backend.has_block(hash1));
        assert!(!backend.has_block(hash2));
        assert!(backend.has_block(hash3));
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

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));
        let (block3, hash3) = create_registered_block(3, &tokens_for_id(3));

        for _ in 0..3 {
            frequency_tracker.touch(hash2.as_u128());
        }

        for _ in 0..10 {
            frequency_tracker.touch(hash3.as_u128());
        }

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

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

        // Create blocks with different frequencies to test distribution across levels
        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1)); // Level 0 (cold, freq=0)
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2)); // Level 1 (warm, freq=3)
        let (block3, hash3) = create_registered_block(3, &tokens_for_id(3)); // Level 2 (hot, freq=7)
        let (block4, hash4) = create_registered_block(4, &tokens_for_id(4)); // Level 3 (very hot, freq=15)

        // Set up frequency tracking
        for _ in 0..3 {
            frequency_tracker.touch(hash2.as_u128()); // Warm: frequency 3
        }

        for _ in 0..7 {
            frequency_tracker.touch(hash3.as_u128()); // Hot: frequency 7
        }

        for _ in 0..15 {
            frequency_tracker.touch(hash4.as_u128()); // Very hot: frequency 15
        }

        // Verify priority level calculation
        assert_eq!(backend.calculate_priority_level(hash1), 0); // Cold (freq=0)
        assert_eq!(backend.calculate_priority_level(hash2), 1); // Warm (freq=3)
        assert_eq!(backend.calculate_priority_level(hash3), 2); // Hot (freq=7)
        assert_eq!(backend.calculate_priority_level(hash4), 3); // Very hot (freq=15)

        // Insert blocks - should be distributed across different levels
        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);
        backend.insert(block4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
        assert!(backend.has_block(hash4));

        // Test that we can allocate from all levels
        let allocated = backend.allocate(4);
        assert_eq!(allocated.len(), 4);
        assert_eq!(backend.len(), 0);
    }
}
