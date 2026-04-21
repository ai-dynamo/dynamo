// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test manager creation helpers.

use crate::blocks::BlockMetadata;
use crate::manager::{BlockManager, FrequencyTrackingCapacity};
use crate::registry::BlockRegistry;

/// Create a basic test manager with LRU backend.
pub fn create_test_manager<T: BlockMetadata>(block_count: usize) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(4) // Most tests use 4-token blocks
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}

/// Create a test manager with custom block size.
pub fn create_test_manager_with_block_size<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(block_size)
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}
