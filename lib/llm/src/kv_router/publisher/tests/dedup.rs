// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(test)]
mod test_event_dedup_filter {
    use super::*;

    fn store_data(hashes: &[u64]) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|&h| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(h),
                    tokens_hash: LocalBlockHash(h * 10),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn remove_data(hashes: &[u64]) -> KvCacheRemoveData {
        KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .map(|&h| ExternalSequenceBlockHash(h))
                .collect(),
        }
    }

    #[test]
    fn stores_track_refcounts_for_removes() {
        let mut filter = EventDedupFilter::new();
        let data = store_data(&[1, 2, 3]);

        // Store same hashes twice — refcount should be 2
        filter.track_store(0, StorageTier::Device, &data);
        filter.track_store(0, StorageTier::Device, &data);

        // First remove — refcounts 2→1, all filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_none());

        // Second remove — refcounts 1→0, all pass through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_some());
        assert_eq!(result.unwrap().block_hashes.len(), 3);
    }

    #[test]
    fn duplicate_removes_are_filtered() {
        let mut filter = EventDedupFilter::new();

        // Store same hash twice
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // First remove — refcount 2→1, filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_none());

        // Second remove — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
        assert_eq!(result.unwrap().block_hashes.len(), 1);
    }

    #[test]
    fn store_remove_store_cycle() {
        let mut filter = EventDedupFilter::new();

        // Store hash 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Remove hash 1 — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        // Store hash 1 again — refcount starts fresh at 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Remove again — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }

    #[test]
    fn clear_resets_all_ranks() {
        let mut filter = EventDedupFilter::new();

        // Store on rank 0 and rank 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1, 2]));

        // Clear wipes all ranks (matches indexer semantics where Cleared
        // from any rank removes all blocks for the entire worker).
        filter.clear();

        // Both ranks pass through defensively after clear
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        let result = filter.filter_remove(1, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }

    #[test]
    fn mixed_blocks_in_single_remove() {
        let mut filter = EventDedupFilter::new();

        // Hash 1: stored twice (refcount 2)
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Hash 2: stored once (refcount 1)
        filter.track_store(0, StorageTier::Device, &store_data(&[2]));

        // Hash 3: stored twice (refcount 2)
        filter.track_store(0, StorageTier::Device, &store_data(&[3]));
        filter.track_store(0, StorageTier::Device, &store_data(&[3]));

        // Remove all three — only hash 2 (refcount 1→0) passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.block_hashes.len(), 1);
        assert_eq!(result.block_hashes[0], ExternalSequenceBlockHash(2));
    }

    #[test]
    fn same_hash_on_different_ranks_are_independent() {
        let mut filter = EventDedupFilter::new();

        // Store hash 1 on rank 0 (twice) and rank 1 (once)
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1]));

        // Remove hash 1 on rank 1 — refcount 1→0, passes through
        let result = filter.filter_remove(1, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        // Remove hash 1 on rank 0 — refcount 2→1, filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_none());

        // Remove hash 1 on rank 0 again — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }
}
