// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fully contiguous layout implementation.
//!
//! This layout stores all blocks in a single contiguous memory allocation
//! with the shape: [num_blocks, num_layers, outer_dim, page_size, inner_dim].

use anyhow::{Result, anyhow};
use std::sync::Arc;
use validator::Validate;

use super::serialize::{BlockFormat, FullyContiguousDetails, LayoutTypeDetails};
use super::{Layout, LayoutConfig, MemoryDescriptor, MemoryRegion, OwnedMemoryRegion};

/// Fully contiguous layout where all blocks are in a single allocation.
#[derive(Debug)]
pub struct FullyContiguousLayout {
    config: LayoutConfig,
    /// Base address of the allocation
    base_addr: usize,
    /// Stride between blocks in bytes
    block_stride: usize,
    /// Stride between layers in bytes
    layer_stride: usize,
    /// Stride between outer dimensions in bytes
    outer_stride: usize,
    /// Size of each memory region (page) in bytes
    region_size: usize,
    /// Owned memory region backing this layout
    memory: Arc<dyn MemoryRegion>,
    /// Format of blocks in memory
    block_format: BlockFormat,
}

impl FullyContiguousLayout {
    /// Create a new fully contiguous layout.
    ///
    /// # Arguments
    /// * `config` - Layout configuration
    /// * `memory` - Owned memory region that backs this layout
    ///
    /// # Returns
    /// A new FullyContiguousLayout instance
    pub fn new(config: LayoutConfig, memory: Arc<dyn MemoryRegion>) -> Result<Self> {
        config.validate()?;

        let base_addr = memory.addr();

        // Calculate strides
        let region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;
        let outer_stride = region_size;
        let layer_stride = outer_stride * config.outer_dim;
        let block_stride = layer_stride * config.num_layers;

        // Validate that the memory region is large enough
        let required_size = block_stride * config.num_blocks;
        if memory.size() < required_size {
            return Err(anyhow!(
                "Memory region too small for layout. Required: {} bytes, got: {} bytes",
                required_size,
                memory.size()
            ));
        }

        Ok(Self {
            config,
            base_addr,
            block_stride,
            layer_stride,
            outer_stride,
            region_size,
            memory,
            block_format: BlockFormat::default(),
        })
    }

    /// Create a new fully contiguous layout with a specific block format.
    ///
    /// # Arguments
    /// * `config` - Layout configuration
    /// * `memory` - Owned memory region that backs this layout
    /// * `block_format` - Format of blocks in memory
    ///
    /// # Returns
    /// A new FullyContiguousLayout instance
    pub(crate) fn new_with_format(
        config: LayoutConfig,
        memory: Arc<dyn MemoryRegion>,
        block_format: BlockFormat,
    ) -> Result<Self> {
        let mut layout = Self::new(config, memory)?;
        layout.block_format = block_format;
        Ok(layout)
    }

    /// Get the block format.
    pub fn block_format(&self) -> BlockFormat {
        self.block_format
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

        Ok(self.base_addr
            + block_id * self.block_stride
            + layer_id * self.layer_stride
            + outer_id * self.outer_stride)
    }

    /// Get mutable reference to the memory Arc for NIXL registration.
    pub fn memory_arc_mut(&mut self) -> &mut Arc<dyn MemoryRegion> {
        &mut self.memory
    }
}

impl Layout for FullyContiguousLayout {
    fn config(&self) -> &LayoutConfig {
        &self.config
    }

    fn memory_regions(&self) -> &[OwnedMemoryRegion] {
        std::slice::from_ref(&self.memory)
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
        // Single contiguous allocation
        vec![self.block_stride * self.config.num_blocks]
    }

    fn is_fully_contiguous(&self) -> bool {
        true
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
        LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
            block_format: self.block_format,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn test_fully_contiguous_layout_creation() {
        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let required_bytes = config.required_bytes();
        assert_eq!(required_bytes, 10 * 4 * 2 * 16 * 128 * 2);

        let memory = MockMemory::new(0x1000, required_bytes);

        let layout = FullyContiguousLayout::new(config, memory).unwrap();
        assert_eq!(layout.num_blocks(), 10);
        assert!(layout.is_fully_contiguous());
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

        let required_size = config.required_bytes();
        let memory = MockMemory::new(0x1000, required_size);
        let layout = FullyContiguousLayout::new(config.clone(), memory).unwrap();

        // Test accessing specific memory regions
        let region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

        // Block 0, Layer 0, Outer 0
        let region = layout.memory_region(0, 0, 0).unwrap();
        assert_eq!(region.addr, 0x1000);
        assert_eq!(region.size, region_size);

        // Block 0, Layer 0, Outer 1
        let region = layout.memory_region(0, 0, 1).unwrap();
        assert_eq!(region.addr, 0x1000 + region_size);
        assert_eq!(region.size, region_size);

        // Block 0, Layer 1, Outer 0
        let region = layout.memory_region(0, 1, 0).unwrap();
        assert_eq!(region.addr, 0x1000 + 2 * region_size);
        assert_eq!(region.size, region_size);

        // Block 1, Layer 0, Outer 0
        let region = layout.memory_region(1, 0, 0).unwrap();
        assert_eq!(
            region.addr,
            0x1000 + (config.outer_dim * config.num_layers * region_size)
        );
        assert_eq!(region.size, region_size);
    }
}
