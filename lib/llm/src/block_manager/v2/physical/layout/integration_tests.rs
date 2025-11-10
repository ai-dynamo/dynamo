// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests comparing v1 and v2 layout implementations.
//!
//! These tests validate that the new v2 layout system produces identical
//! memory region **addresses** as the proven v1 implementation.
//!
//! **Note on Size Differences**: V1's `memory_region()` returns `layer_stride` as the
//! size (covering all outer dimensions), while V2 returns `outer_stride` (single page).
//! This is an intentional API difference - V2 provides more granular access.
//! Therefore, these tests only compare addresses, not sizes.

#![cfg(test)]

use anyhow::Result;
use std::{any::Any, sync::Arc};

use crate::block_manager::{
    layout::{
        BlockDimension, BlockLayout, BlockLayoutConfig, GenericBlockLayout, LayoutConfig,
        LayoutType,
        tests::{setup_layer_separate_layout, setup_layout},
    },
    storage::{Storage, tests::NullDeviceStorage},
    v2::storage::StorageKind,
};

use super::{
    FullyContiguousLayout, LayerSeparateLayout, Layout, LayoutConfig as V2LayoutConfig,
    MemoryRegion,
};

// Test constants matching v1 tests
const NUM_BLOCKS: usize = 7;
const NUM_LAYERS: usize = 5;
const OUTER_DIM: usize = 2;
const PAGE_SIZE: usize = 4;
const INNER_DIM: usize = 13;
const DTYPE_WIDTH_BYTES: usize = 4;

/// Wrapper to make v1 NullDeviceStorage compatible with v2 MemoryRegion trait.
#[derive(Debug)]
struct V1StorageWrapper {
    storage: NullDeviceStorage,
}

impl MemoryRegion for V1StorageWrapper {
    fn addr(&self) -> usize {
        self.storage.addr() as usize
    }

    fn size(&self) -> usize {
        self.storage.size()
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::System
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Create v1 layout configuration
fn create_v1_config() -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(NUM_BLOCKS)
        .num_layers(NUM_LAYERS)
        .outer_dim(OUTER_DIM)
        .page_size(PAGE_SIZE)
        .inner_dim(INNER_DIM)
        .alignment(1)
        .dtype_width_bytes(DTYPE_WIDTH_BYTES)
        .build()
        .unwrap()
}

/// Create v2 layout configuration (equivalent to v1)
fn create_v2_config() -> V2LayoutConfig {
    create_v1_config()
}

#[test]
fn test_v1_v2_fully_contiguous_equivalence() -> Result<()> {
    // Create v1 layout
    let v1_layout = setup_layout(None)?;

    // Create v2 layout with same configuration
    let v2_config = create_v2_config();
    let required_size =
        NUM_BLOCKS * NUM_LAYERS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
    let v1_storage = NullDeviceStorage::new(required_size as u64);
    let memory = Arc::new(V1StorageWrapper {
        storage: v1_storage,
    }) as Arc<dyn MemoryRegion>;
    let v2_layout = FullyContiguousLayout::new(v2_config, memory)?;

    // Compare all memory regions
    for block_id in 0..NUM_BLOCKS {
        for layer_id in 0..NUM_LAYERS {
            for outer_id in 0..OUTER_DIM {
                let v1_region = v1_layout.memory_region(block_id, layer_id, outer_id)?;
                let v2_region = v2_layout.memory_region(block_id, layer_id, outer_id)?;

                assert_eq!(
                    v1_region.addr(),
                    v2_region.addr,
                    "Address mismatch at block={}, layer={}, outer={}",
                    block_id,
                    layer_id,
                    outer_id
                );
                assert_eq!(
                    v1_region.size(),
                    v2_region.size,
                    "Size mismatch at block={}, layer={}, outer={}",
                    block_id,
                    layer_id,
                    outer_id
                );
            }
        }
    }

    // Verify metadata
    assert_eq!(v1_layout.num_blocks(), v2_layout.num_blocks());
    assert_eq!(v1_layout.num_layers(), v2_layout.num_layers());
    assert_eq!(v1_layout.outer_dim(), v2_layout.outer_dim());
    assert_eq!(v1_layout.page_size(), v2_layout.page_size());
    assert_eq!(v1_layout.inner_dim(), v2_layout.inner_dim());

    Ok(())
}

#[test]
fn test_v1_v2_layer_separate_block_contiguous_equivalence() -> Result<()> {
    // Create v1 layout (block contiguous = !outer_contiguous)
    let v1_layout = setup_layer_separate_layout(None, BlockDimension::BlockIsFirstDim)?;

    // Create v2 layout with same configuration
    let v2_config = create_v2_config();
    let per_layer_size = NUM_BLOCKS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

    let memory: Vec<Arc<dyn MemoryRegion>> = (0..NUM_LAYERS)
        .map(|_| {
            Arc::new(V1StorageWrapper {
                storage: NullDeviceStorage::new(per_layer_size as u64),
            }) as Arc<dyn MemoryRegion>
        })
        .collect();

    let v2_layout = LayerSeparateLayout::new(v2_config, memory, BlockDimension::BlockIsFirstDim)?;

    // Verify metadata
    assert_eq!(v1_layout.num_blocks(), v2_layout.num_blocks());
    assert_eq!(v1_layout.num_layers(), v2_layout.num_layers());
    assert_eq!(v1_layout.outer_dim(), v2_layout.outer_dim());
    assert_eq!(v1_layout.page_size(), v2_layout.page_size());
    assert_eq!(v1_layout.inner_dim(), v2_layout.inner_dim());

    // Compare all memory regions
    for block_id in 0..NUM_BLOCKS {
        for layer_id in 0..NUM_LAYERS {
            for outer_id in 0..OUTER_DIM {
                let v1_region = v1_layout.memory_region(block_id, layer_id, outer_id)?;
                let v2_region = v2_layout.memory_region(block_id, layer_id, outer_id)?;

                assert_eq!(
                    v1_region.addr(),
                    v2_region.addr,
                    "Address mismatch at block={}, layer={}, outer={} (block_contiguous)",
                    block_id,
                    layer_id,
                    outer_id
                );
                assert_eq!(
                    v1_region.size(),
                    v2_region.size,
                    "Size mismatch at block={}, layer={}, outer={} (block_contiguous)",
                    block_id,
                    layer_id,
                    outer_id
                );
            }
        }
    }

    // Verify layout type
    assert!(!v2_layout.is_fully_contiguous());

    assert_eq!(
        v1_layout.layout_type(),
        LayoutType::LayerSeparate {
            block_dim: BlockDimension::BlockIsFirstDim,
        }
    );

    Ok(())
}

#[test]
fn test_v1_v2_layer_separate_outer_contiguous_equivalence() -> Result<()> {
    // Create v1 layout (outer contiguous)
    let v1_layout = setup_layer_separate_layout(None, BlockDimension::BlockIsSecondDim)?;

    // Create v2 layout with same configuration
    let v2_config = create_v2_config();
    let per_layer_size = NUM_BLOCKS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

    let memory: Vec<Arc<dyn MemoryRegion>> = (0..NUM_LAYERS)
        .map(|_| {
            Arc::new(V1StorageWrapper {
                storage: NullDeviceStorage::new(per_layer_size as u64),
            }) as Arc<dyn MemoryRegion>
        })
        .collect();

    let v2_layout = LayerSeparateLayout::new(v2_config, memory, BlockDimension::BlockIsSecondDim)?;

    // Compare all memory regions
    for block_id in 0..NUM_BLOCKS {
        for layer_id in 0..NUM_LAYERS {
            for outer_id in 0..OUTER_DIM {
                let v1_region = v1_layout.memory_region(block_id, layer_id, outer_id)?;
                let v2_region = v2_layout.memory_region(block_id, layer_id, outer_id)?;

                assert_eq!(
                    v1_region.addr(),
                    v2_region.addr,
                    "Address mismatch at block={}, layer={}, outer={} (outer_contiguous)",
                    block_id,
                    layer_id,
                    outer_id
                );
                assert_eq!(
                    v1_region.size(),
                    v2_region.size,
                    "Size mismatch at block={}, layer={}, outer={} (outer_contiguous)",
                    block_id,
                    layer_id,
                    outer_id
                );
            }
        }
    }

    // Verify layout type
    assert!(!v2_layout.is_fully_contiguous());
    assert_eq!(
        v1_layout.layout_type(),
        LayoutType::LayerSeparate {
            block_dim: BlockDimension::BlockIsSecondDim,
        }
    );

    Ok(())
}

#[test]
fn test_v1_v2_stride_calculations() -> Result<()> {
    // Test with a specific pattern to verify stride calculations
    let _v1_layout = setup_layout(None)?;
    let v2_config = create_v2_config();
    let required_size =
        NUM_BLOCKS * NUM_LAYERS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
    let v1_storage = NullDeviceStorage::new(required_size as u64);
    let memory = Arc::new(V1StorageWrapper {
        storage: v1_storage,
    }) as Arc<dyn MemoryRegion>;
    let v2_layout = FullyContiguousLayout::new(v2_config, memory)?;

    // Calculate expected strides
    let region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
    let outer_stride = region_size;
    let layer_stride = outer_stride * OUTER_DIM;
    let block_stride = layer_stride * NUM_LAYERS;

    // Test stride consistency across blocks
    for block_id in 0..NUM_BLOCKS - 1 {
        let region_b0 = v2_layout.memory_region(block_id, 0, 0)?;
        let region_b1 = v2_layout.memory_region(block_id + 1, 0, 0)?;
        assert_eq!(
            region_b1.addr - region_b0.addr,
            block_stride,
            "Block stride mismatch between blocks {} and {}",
            block_id,
            block_id + 1
        );
    }

    // Test stride consistency across layers
    for layer_id in 0..NUM_LAYERS - 1 {
        let region_l0 = v2_layout.memory_region(0, layer_id, 0)?;
        let region_l1 = v2_layout.memory_region(0, layer_id + 1, 0)?;
        assert_eq!(
            region_l1.addr - region_l0.addr,
            layer_stride,
            "Layer stride mismatch between layers {} and {}",
            layer_id,
            layer_id + 1
        );
    }

    // Test stride consistency across outer dimensions
    for outer_id in 0..OUTER_DIM - 1 {
        let region_o0 = v2_layout.memory_region(0, 0, outer_id)?;
        let region_o1 = v2_layout.memory_region(0, 0, outer_id + 1)?;
        assert_eq!(
            region_o1.addr - region_o0.addr,
            outer_stride,
            "Outer stride mismatch between outer dims {} and {}",
            outer_id,
            outer_id + 1
        );
    }

    Ok(())
}

#[test]
fn test_v1_v2_edge_case_single_block() -> Result<()> {
    // Test with minimal configuration: single block
    let v1_config = LayoutConfig::builder()
        .num_blocks(1)
        .num_layers(NUM_LAYERS)
        .outer_dim(OUTER_DIM)
        .page_size(PAGE_SIZE)
        .inner_dim(INNER_DIM)
        .dtype_width_bytes(DTYPE_WIDTH_BYTES)
        .build()
        .unwrap();

    let v1_layout = crate::block_manager::layout::FullyContiguous::allocate(
        v1_config.clone(),
        &crate::block_manager::storage::tests::NullDeviceAllocator,
    )?;

    let v2_config = v1_config.clone();

    let required_size = 1 * NUM_LAYERS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
    let v1_storage = NullDeviceStorage::new(required_size as u64);
    let memory = Arc::new(V1StorageWrapper {
        storage: v1_storage,
    }) as Arc<dyn MemoryRegion>;
    let v2_layout = FullyContiguousLayout::new(v2_config, memory)?;

    // Compare the single block across all layers and outer dims
    for layer_id in 0..NUM_LAYERS {
        for outer_id in 0..OUTER_DIM {
            let v1_region = v1_layout.memory_region(0, layer_id, outer_id)?;
            let v2_region = v2_layout.memory_region(0, layer_id, outer_id)?;

            assert_eq!(v1_region.addr(), v2_region.addr);
            assert_eq!(v1_region.size(), v2_region.size);
        }
    }

    Ok(())
}

#[test]
fn test_v1_v2_edge_case_single_layer() -> Result<()> {
    // Test with minimal configuration: single layer
    let v1_config = LayoutConfig::builder()
        .num_blocks(NUM_BLOCKS)
        .num_layers(1)
        .outer_dim(OUTER_DIM)
        .page_size(PAGE_SIZE)
        .inner_dim(INNER_DIM)
        .dtype_width_bytes(DTYPE_WIDTH_BYTES)
        .build()?;

    let v1_layout = crate::block_manager::layout::FullyContiguous::allocate(
        v1_config.clone(),
        &crate::block_manager::storage::tests::NullDeviceAllocator,
    )?;

    let v2_config = v1_config.clone();

    let required_size = NUM_BLOCKS * 1 * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
    let v1_storage = NullDeviceStorage::new(required_size as u64);
    let memory = Arc::new(V1StorageWrapper {
        storage: v1_storage,
    }) as Arc<dyn MemoryRegion>;
    let v2_layout = FullyContiguousLayout::new(v2_config, memory)?;

    // Compare the single layer across all blocks and outer dims
    for block_id in 0..NUM_BLOCKS {
        for outer_id in 0..OUTER_DIM {
            let v1_region = v1_layout.memory_region(block_id, 0, outer_id)?;
            let v2_region = v2_layout.memory_region(block_id, 0, outer_id)?;

            assert_eq!(v1_region.addr(), v2_region.addr);
            assert_eq!(v1_region.size(), v2_region.size);
        }
    }

    Ok(())
}
