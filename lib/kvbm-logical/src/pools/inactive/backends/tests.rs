// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod backend_tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use rstest::rstest;

    use crate::BlockId;
    use crate::pools::store::InactiveIndex;
    use crate::testing::{block_id_and_hash, hash_for_tokens, tokens_for_id};
    use crate::tinylfu::TinyLFUTracker;

    use super::super::*;

    #[derive(Clone, Copy, Debug)]
    enum BackendType {
        HashMap,
        Lru,
        MultiLru,
        Lineage,
    }

    fn create_backend(backend_type: BackendType) -> Box<dyn InactiveIndex> {
        match backend_type {
            BackendType::HashMap => Box::new(HashMapBackend::new(Box::new(FifoReusePolicy::new()))),
            BackendType::Lru => Box::new(LruBackend::new(NonZeroUsize::new(10).unwrap())),
            BackendType::MultiLru => Box::new(MultiLruBackend::new(
                NonZeroUsize::new(10).unwrap(),
                Arc::new(TinyLFUTracker::new(100)),
            )),
            BackendType::Lineage => Box::new(LineageBackend::new()),
        }
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_insert_and_len(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());

        let (id, hash) = block_id_and_hash(1, &tokens_for_id(1));
        backend.insert(hash, id);

        assert_eq!(backend.len(), 1);
        assert!(!backend.is_empty());
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_has_block(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id, hash) = block_id_and_hash(1, &tokens_for_id(1));
        backend.insert(hash, id);

        assert!(backend.has(hash));
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_find_matches(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));

        backend.insert(hash1, id1);
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        backend.insert(hash2, id2);

        let matches = backend.find_matches(&[hash1, hash2], true);
        assert_eq!(matches.len(), 2);
        assert_eq!(backend.len(), 0);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_find_matches_stops_on_miss(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let missing_hash = hash_for_tokens(&tokens_for_id(3));

        backend.insert(hash1, id1);
        backend.insert(hash2, id2);

        let matches = backend.find_matches(&[hash1, missing_hash, hash2], true);
        assert_eq!(matches.len(), 1);
        assert_eq!(backend.len(), 1);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_allocate(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        backend.insert(hash1, id1);

        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        backend.insert(hash2, id2);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(backend.len(), 1);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_allocate_more_than_available(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        backend.insert(hash1, id1);

        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        backend.insert(hash2, id2);

        let allocated = backend.allocate(5);
        assert_eq!(allocated.len(), 2);
        assert_eq!(backend.len(), 0);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_allocate_all(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let block_ids: Vec<u64> = vec![1, 2, 3, 4, 5];
        for &i in &block_ids {
            let (id, hash) = block_id_and_hash(i as BlockId, &tokens_for_id(i));
            backend.insert(hash, id);
            if matches!(backend_type, BackendType::HashMap) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        assert_eq!(backend.len(), 5);

        let allocated = backend.allocate_all();
        assert_eq!(allocated.len(), 5);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_allocate_all_empty_pool(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        assert_eq!(backend.len(), 0);

        let allocated = backend.allocate_all();
        assert_eq!(allocated.len(), 0);
        assert!(backend.is_empty());
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    #[case::lineage(BackendType::Lineage)]
    fn test_scan_matches(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (id1, hash1) = block_id_and_hash(1, &tokens_for_id(1));
        let (id2, hash2) = block_id_and_hash(2, &tokens_for_id(2));
        let (id3, hash3) = block_id_and_hash(3, &tokens_for_id(3));
        let missing_hash = hash_for_tokens(&tokens_for_id(4));

        backend.insert(hash1, id1);
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        backend.insert(hash2, id2);
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        backend.insert(hash3, id3);

        assert_eq!(backend.len(), 3);

        let matches = backend.scan_matches(&[hash1, missing_hash, hash3], true);
        assert_eq!(
            matches.len(),
            2,
            "scan_matches should find 2 blocks, skipping the miss"
        );

        let found_hashes: Vec<_> = matches.iter().map(|(h, _)| *h).collect();
        assert!(found_hashes.contains(&hash1));
        assert!(found_hashes.contains(&hash3));

        assert_eq!(backend.len(), 1, "Only block2 should remain");
    }
}
