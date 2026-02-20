// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Manager
//! A synchronous implementation of a block manager that handles MoveBlock signals for caching KV blocks.
//!
//! ## Backends
//! Two backends are available:
//! - **vLLM**: Original HashCache-based reference-counting implementation
//! - **KvbmLogical**: Production kvbm-logical BlockManager with RAII block lifecycle
//!
//! ## Block Operations
//! The KV manager processes four types of MoveBlock signals:
//!
//! ### Use
//! - Checks if block exists in active pool → increment reference count
//! - If in inactive pool → move to active pool
//! - If neither → try evicting from inactive pool to make room
//! - If inactive pool is empty → pre-empt the oldest running request
//!
//! ### Destroy
//! - Removes the block from the active pool
//!
//! ### Deref
//! - Decrements reference count of a block in active pool
//! - If count reaches zero → move block to inactive pool
//!
//! ### Promote
//! - Converts a partial block (uuid) into a full block (global block hash)
//!
//! ## Preemption
//! If a Use operation fails (typically due to insufficient space), a false boolean signal
//! is returned to the scheduler for preemption. Initial KV block allocations for new requests
//! should not fail due to the watermark checking.

mod backend;
mod kvbm_backend;
pub mod vllm_backend;

pub use backend::KvBackend;

use crate::common::protocols::{KvCacheEventSink, KvManagerBackend, MockerEvictionBackend, MoveBlock};
use dynamo_tokens::PositionalLineageHash;
use dynamo_tokens::blocks::UniqueBlock;
use std::sync::Arc;

use self::kvbm_backend::KvbmLogicalKvManager;
use self::vllm_backend::ManualKvManager;

/// Enum-based KV manager that dispatches to either the vLLM or kvbm-logical backend.
pub enum KvManager {
    Manual(ManualKvManager),
    KvbmLogical(KvbmLogicalKvManager),
}

impl KvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        Self::new_with_event_sink(
            max_capacity,
            block_size,
            None,
            0,
            KvManagerBackend::Manual,
            MockerEvictionBackend::default(),
        )
    }

    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        dp_rank: u32,
        backend: KvManagerBackend,
        eviction_backend: MockerEvictionBackend,
    ) -> Self {
        match backend {
            KvManagerBackend::Manual => Self::Manual(ManualKvManager::new_with_event_sink(
                max_capacity,
                block_size,
                kv_event_sink,
                dp_rank,
            )),
            KvManagerBackend::KvbmLogical => Self::KvbmLogical(KvbmLogicalKvManager::new(
                max_capacity,
                block_size,
                dp_rank,
                kv_event_sink,
                eviction_backend,
            )),
        }
    }
}

impl KvBackend for KvManager {
    fn process(&mut self, event: &MoveBlock) -> bool {
        match self {
            Self::Manual(m) => m.process(event),
            Self::KvbmLogical(m) => m.process(event),
        }
    }

    fn max_capacity(&self) -> usize {
        match self {
            Self::Manual(m) => m.max_capacity(),
            Self::KvbmLogical(m) => m.max_capacity(),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Self::Manual(m) => m.block_size(),
            Self::KvbmLogical(m) => m.block_size(),
        }
    }

    fn num_active_blocks(&self) -> usize {
        match self {
            Self::Manual(m) => m.num_active_blocks(),
            Self::KvbmLogical(m) => m.num_active_blocks(),
        }
    }

    fn num_inactive_blocks(&self) -> usize {
        match self {
            Self::Manual(m) => m.num_inactive_blocks(),
            Self::KvbmLogical(m) => m.num_inactive_blocks(),
        }
    }

    fn current_capacity(&self) -> usize {
        match self {
            Self::Manual(m) => m.current_capacity(),
            Self::KvbmLogical(m) => m.current_capacity(),
        }
    }

    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        match self {
            Self::Manual(m) => m.probe_new_blocks(blocks),
            Self::KvbmLogical(m) => m.probe_new_blocks(blocks),
        }
    }

    fn is_block_cached(&self, seq_hash: u64, plh: Option<PositionalLineageHash>) -> bool {
        match self {
            Self::Manual(m) => m.is_block_cached(seq_hash, plh),
            Self::KvbmLogical(m) => m.is_block_cached(seq_hash, plh),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_on_max_capacity() {
        let mut manager = KvManager::new(10, 16);

        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) -> bool {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![]))
        }

        let response = use_blocks(&mut manager, (0..10).collect());
        assert!(response, "Expected success response");
        assert_eq!(manager.current_capacity(), 10);

        let response = use_blocks(&mut manager, vec![10]);
        assert!(
            !response,
            "Expected failure response when exceeding max capacity"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        let mut manager = KvManager::new(10, 16);

        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![]));
        }

        fn destroy_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Destroy(blocks));
        }

        fn deref_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Deref(blocks));
        }

        fn assert_active_blocks(manager: &KvManager, expected_blocks: &[(u64, usize)]) {
            let KvManager::Manual(m) = manager else {
                panic!("Expected Manual backend for this test");
            };
            assert_eq!(
                m.active_blocks().len(),
                expected_blocks.len(),
                "Active blocks count doesn't match expected"
            );

            for &(id, ref_count) in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    m.active_blocks().contains_key(&block),
                    "Block {id} not found in active blocks",
                );
                assert_eq!(
                    m.active_blocks().get(&block),
                    Some(&ref_count),
                    "Block {id} has wrong reference count",
                );
            }
        }

        fn assert_inactive_blocks(
            manager: &KvManager,
            expected_size: usize,
            expected_blocks: &[u64],
        ) {
            let KvManager::Manual(m) = manager else {
                panic!("Expected Manual backend for this test");
            };
            let inactive_blocks = m.get_inactive_blocks();
            let inactive_blocks_count = m.num_inactive_blocks();

            assert_eq!(
                inactive_blocks_count, expected_size,
                "Inactive blocks count doesn't match expected"
            );

            for &id in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    inactive_blocks.iter().any(|&b| *b == block),
                    "Block {id} not found in inactive blocks",
                );
            }
        }

        use_blocks(&mut manager, (0..5).collect());
        use_blocks(&mut manager, vec![0, 1, 5, 6]);
        assert_active_blocks(
            &manager,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        destroy_blocks(&mut manager, vec![4]);
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);
        assert_inactive_blocks(&manager, 2, &[3, 2]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (5, 1), (6, 1)]);

        destroy_blocks(&mut manager, vec![6]);
        deref_blocks(&mut manager, vec![0, 1, 5]);
        assert_inactive_blocks(&manager, 5, &[0, 1, 2, 3, 5]);
        assert_active_blocks(&manager, &[]);

        use_blocks(&mut manager, vec![0, 1, 2, 7, 8, 9]);
        assert_inactive_blocks(&manager, 2, &[3, 5]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);

        let blocks_to_check: Vec<UniqueBlock> = vec![0, 1, 2, 3, 4]
            .into_iter()
            .map(UniqueBlock::FullBlock)
            .collect();
        assert_eq!(manager.probe_new_blocks(&blocks_to_check), 1);

        use_blocks(&mut manager, vec![10, 11, 12]);
        assert_inactive_blocks(&manager, 1, &[5]);

        use_blocks(&mut manager, vec![13]);
    }
}
