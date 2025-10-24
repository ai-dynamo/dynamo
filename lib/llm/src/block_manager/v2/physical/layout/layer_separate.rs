// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layer-separate layout implementation.
//!
//! This layout stores each layer in its own allocation, which is the typical
//! vLLM layout. Each layer can be either block-contiguous or outer-contiguous:
//! - Block-contiguous: [num_blocks, outer_dim, page_size, inner_dim]
//! - Outer-contiguous: [outer_dim, num_blocks, page_size, inner_dim]

use anyhow::{Result, anyhow};
use std::sync::Arc;
use validator::Validate;

use super::serialize::{LayerSeparateDetails, LayoutTypeDetails};
use super::{
    BlockDimension, Layout, LayoutConfig, MemoryDescriptor, MemoryRegion, OwnedMemoryRegion,
};

/// Layer-separate layout where each layer has its own allocation.
#[derive(Debug)]
pub struct LayerSeparateLayout {
    config: LayoutConfig,
    /// Base addresses for each layer
    layer_base_addrs: Vec<usize>,
    /// Whether the outer dimension is contiguous (vs block dimensionl
    block_dim: BlockDimension,
    /// Stride between blocks in bytes
    block_stride: usize,
    /// Stride between outer dimensions in bytes
    outer_stride: usize,
    /// Size of each memory region (page) in bytes
    region_size: usize,
    /// Owned memory regions backing this layout (one per layer)
    memory_regions: Vec<Arc<dyn MemoryRegion>>,
}

impl LayerSeparateLayout {
    /// Create a new layer-separate layout.
    ///
    /// # Arguments
    /// - `config` - Layout configuration
    /// - `memory` - Vector of owned memory regions (one per layer)
    /// - `outer_contiguous` - If true, outer dimension is contiguous with the inner dimension, i.e. (num_blocks, outer_dim, ...);
    ///   if false, block dimension is contiguous with the inner dimension, i.e. (outer_dim, num_blocks, ...).
    ///
    /// # Returns
    /// A new LayerSeparateLayout instance
    pub fn new(
        config: LayoutConfig,
        memory: Vec<Arc<dyn MemoryRegion>>,
        block_dim: BlockDimension,
    ) -> Result<Self> {
        config.validate()?;

        if memory.len() != config.num_layers {
            return Err(anyhow!(
                "Memory region count ({}) must match num_layers ({})",
                memory.len(),
                config.num_layers
            ));
        }

        // Calculate strides
        let region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

        let (block_stride, outer_stride) = if block_dim == BlockDimension::BlockIsSecondDim {
            // Layout: [outer_dim, num_blocks, page_size, inner_dim]
            let block_stride = region_size;
            let outer_stride = block_stride * config.num_blocks;
            (block_stride, outer_stride)
        } else {
            // Layout: [num_blocks, outer_dim, page_size, inner_dim]
            let outer_stride = region_size;
            let block_stride = outer_stride * config.outer_dim;
            (block_stride, outer_stride)
        };

        // Extract base addresses and validate sizes
        let mut layer_base_addrs = Vec::with_capacity(config.num_layers);
        let required_size = config.num_blocks * config.outer_dim * region_size;

        for (i, mem) in memory.iter().enumerate() {
            if mem.size() < required_size {
                return Err(anyhow!(
                    "Memory region {} too small for layout. Required: {} bytes, got: {} bytes",
                    i,
                    required_size,
                    mem.size()
                ));
            }
            layer_base_addrs.push(mem.addr());
        }

        Ok(Self {
            config,
            layer_base_addrs,
            block_dim,
            block_stride,
            outer_stride,
            region_size,
            memory_regions: memory,
        })
    }

    /// Calculate the address of a specific memory region.
    fn calculate_address(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<usize> {
        if block_id >= self.config.num_blocks {
            return Err(anyhow!(
                "Block ID {} out of range (max: {})",
                block_id,
                self.config.num_blocks
            ));
        }
        if layer_id >= self.config.num_layers {
            return Err(anyhow!(
                "Layer ID {} out of range (max: {})",
                layer_id,
                self.config.num_layers
            ));
        }
        if outer_id >= self.config.outer_dim {
            return Err(anyhow!(
                "Outer ID {} out of range (max: {})",
                outer_id,
                self.config.outer_dim
            ));
        }

        let base_addr = self.layer_base_addrs[layer_id];
        let offset = block_id * self.block_stride + outer_id * self.outer_stride;

        Ok(base_addr + offset)
    }

    pub fn block_dim(&self) -> BlockDimension {
        self.block_dim
    }

    /// Get mutable reference to the memory regions for NIXL registration.
    pub fn memory_regions_mut(&mut self) -> &mut [Arc<dyn MemoryRegion>] {
        &mut self.memory_regions
    }
}

impl Layout for LayerSeparateLayout {
    fn config(&self) -> &LayoutConfig {
        &self.config
    }

    fn memory_regions(&self) -> &[OwnedMemoryRegion] {
        &self.memory_regions
    }

    fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryDescriptor> {
        let addr = self.calculate_address(block_id, layer_id, outer_id)?;
        Ok(MemoryDescriptor::new(addr, self.region_size))
    }

    fn required_allocations(&self) -> Vec<usize> {
        // One allocation per layer
        let per_layer_size = self.config.num_blocks * self.config.outer_dim * self.region_size;
        vec![per_layer_size; self.config.num_layers]
    }

    fn is_fully_contiguous(&self) -> bool {
        false
    }

    fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn outer_dim(&self) -> usize {
        self.config.outer_dim
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn inner_dim(&self) -> usize {
        self.config.inner_dim
    }

    fn dtype_width_bytes(&self) -> usize {
        self.config.dtype_width_bytes
    }

    fn serialization_details(&self) -> LayoutTypeDetails {
        LayoutTypeDetails::LayerSeparate(LayerSeparateDetails {
            block_dim: self.block_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn test_layer_separate_block_contiguous() {
        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let per_layer_size = 10 * 2 * 16 * 128 * 2;
        let memory: Vec<Arc<dyn MemoryRegion>> = (0..4)
            .map(|i| {
                MockMemory::new(0x1000 + i * per_layer_size, per_layer_size)
                    as Arc<dyn MemoryRegion>
            })
            .collect();

        let layout =
            LayerSeparateLayout::new(config, memory, BlockDimension::BlockIsFirstDim).unwrap();

        assert_eq!(layout.num_blocks(), 10);
        assert!(!layout.is_fully_contiguous());
        assert_eq!(layout.required_allocations().len(), 4);
    }

    #[test]
    fn test_layer_separate_outer_contiguous() {
        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let per_layer_size = 10 * 2 * 16 * 128 * 2;
        let memory: Vec<Arc<dyn MemoryRegion>> = (0..4)
            .map(|i| {
                MockMemory::new(0x1000 + i * per_layer_size, per_layer_size)
                    as Arc<dyn MemoryRegion>
            })
            .collect();

        let layout =
            LayerSeparateLayout::new(config, memory, BlockDimension::BlockIsSecondDim).unwrap();
        assert_eq!(layout.num_blocks(), 10);
        assert!(!layout.is_fully_contiguous());
    }

    #[test]
    fn test_memory_region() {
        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let per_layer_size = 2 * 2 * 16 * 128 * 2;
        let memory: Vec<Arc<dyn MemoryRegion>> = (0..2)
            .map(|i| {
                MockMemory::new(0x1000 + i * per_layer_size, per_layer_size)
                    as Arc<dyn MemoryRegion>
            })
            .collect();

        let layout =
            LayerSeparateLayout::new(config, memory, BlockDimension::BlockIsFirstDim).unwrap();

        // Test accessing specific memory regions
        let region_size = 16 * 128 * 2;

        // Block 0, Layer 0, Outer 0 - should be at layer 0's base address
        let region = layout.memory_region(0, 0, 0).unwrap();
        assert_eq!(region.addr, 0x1000);
        assert_eq!(region.size, region_size);

        // Block 0, Layer 1, Outer 0 - should be at layer 1's base address
        let region = layout.memory_region(0, 1, 0).unwrap();
        assert_eq!(region.addr, 0x1000 + per_layer_size);
        assert_eq!(region.size, region_size);

        // Block 0, Layer 0, Outer 1 - should be offset within layer 0
        let region = layout.memory_region(0, 0, 1).unwrap();
        assert_eq!(region.addr, 0x1000 + region_size);
        assert_eq!(region.size, region_size);
    }
}
