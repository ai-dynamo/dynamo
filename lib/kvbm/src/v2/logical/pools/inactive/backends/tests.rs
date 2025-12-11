// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod backend_tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;

    use crate::{
        BlockId,
        logical::pools::tests::{TestData, fixtures::*},
        utils::tinylfu::TinyLFUTracker,
    };

    use super::super::*;

    #[derive(Clone, Copy, Debug)]
    enum BackendType {
        HashMap,
        Lru,
        MultiLru,
    }

    fn create_backend(backend_type: BackendType) -> Box<dyn InactivePoolBackend<TestData>> {
        match backend_type {
            BackendType::HashMap => Box::new(HashMapBackend::new(Box::new(FifoReusePolicy::new()))),
            BackendType::Lru => Box::new(LruBackend::new(NonZeroUsize::new(10).unwrap())),
            BackendType::MultiLru => Box::new(MultiLruBackend::new(
                NonZeroUsize::new(10).unwrap(),
                Arc::new(TinyLFUTracker::new(100)),
            )),
        }
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    fn test_insert_and_len(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());

        let (block, _) = create_registered_block(1, &tokens_for_id(1));
        backend.insert(block);

        assert_eq!(backend.len(), 1);
        assert!(!backend.is_empty());
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    fn test_has_block(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (block, hash) = create_registered_block(1, &tokens_for_id(1));
        backend.insert(block);

        assert!(backend.has_block(hash));
        // assert!(!backend.has_block(999));
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    fn test_find_matches(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));

        backend.insert(block1);
        // For HashMap backend with FIFO, we need a sleep to ensure different timestamps
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        backend.insert(block2);

        let matches = backend.find_matches(&[hash1, hash2], true);
        assert_eq!(matches.len(), 2);
        assert_eq!(backend.len(), 0);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    fn test_find_matches_stops_on_miss(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (block1, hash1) = create_registered_block(1, &tokens_for_id(1));
        let (block2, hash2) = create_registered_block(2, &tokens_for_id(2));
        let block_na = create_complete_block(3, &tokens_for_id(3));

        backend.insert(block1);
        backend.insert(block2);

        let matches = backend.find_matches(&[hash1, block_na.sequence_hash(), hash2], true);
        assert_eq!(matches.len(), 1);
        assert_eq!(backend.len(), 1);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    fn test_allocate(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (block1, _) = create_registered_block(1, &tokens_for_id(1));
        backend.insert(block1);

        // For HashMap backend with FIFO, we need a sleep to ensure different timestamps
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let (block2, _) = create_registered_block(2, &tokens_for_id(2));
        backend.insert(block2);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(backend.len(), 1);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    fn test_allocate_more_than_available(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        let (block1, _) = create_registered_block(1, &tokens_for_id(1));
        backend.insert(block1);

        // For HashMap backend with FIFO, we need a sleep to ensure different timestamps
        if matches!(backend_type, BackendType::HashMap) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let (block2, _) = create_registered_block(2, &tokens_for_id(2));
        backend.insert(block2);

        let allocated = backend.allocate(5);
        assert_eq!(allocated.len(), 2);
        assert_eq!(backend.len(), 0);
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    fn test_allocate_all(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        // Insert several blocks
        let block_ids: Vec<u64> = vec![1, 2, 3, 4, 5];
        for &i in &block_ids {
            let (block, _) = create_registered_block(i as BlockId, &tokens_for_id(i));
            backend.insert(block);
            // For HashMap backend with FIFO, we need a sleep to ensure different timestamps
            if matches!(backend_type, BackendType::HashMap) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        assert_eq!(backend.len(), 5);

        // Allocate all blocks
        let allocated = backend.allocate_all();
        assert_eq!(allocated.len(), 5);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[rstest]
    #[case::hashmap(BackendType::HashMap)]
    #[case::lru(BackendType::Lru)]
    #[case::multi_lru(BackendType::MultiLru)]
    fn test_allocate_all_empty_pool(#[case] backend_type: BackendType) {
        let mut backend = create_backend(backend_type);

        assert_eq!(backend.len(), 0);

        // Allocate all from empty pool
        let allocated = backend.allocate_all();
        assert_eq!(allocated.len(), 0);
        assert!(backend.is_empty());
    }
}
