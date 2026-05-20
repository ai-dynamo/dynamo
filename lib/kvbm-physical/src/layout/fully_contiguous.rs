// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fully contiguous layout implementation.
//!
//! This layout stores all blocks in a single contiguous memory allocation
//! with the shape: [num_blocks, num_layers, outer_dim, page_size, inner_dim].

use anyhow::{Result, anyhow, bail};
use kvbm_common::{KvDim, KvDimLayout, KvDimStrides};
use validator::Validate;

use super::serialize::{BlockFormat, FullyContiguousDetails, LayoutTypeDetails};
use super::view::LayoutView;
use super::{
    Buffer, KvBlockLayout, Layout, LayoutConfig, MemoryDescriptor, MemoryRegion, resolve_head_dims,
};

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
    memory: Buffer,
    /// Format of blocks in memory
    block_format: BlockFormat,
    /// KV block layout describing dimension ordering within blocks
    kv_block_layout: KvBlockLayout,
}

/// Builder for creating [`FullyContiguousLayout`] instances.
///
/// # Example
///
/// ```ignore
/// let layout = FullyContiguousLayout::builder()
///     .config(config)
///     .memory(buffer)
///     .kv_block_layout(KvBlockLayout::Universal)
///     .build()?;
/// ```
#[derive(Debug, Default)]
pub struct FullyContiguousLayoutBuilder {
    config: Option<LayoutConfig>,
    memory: Option<Buffer>,
    kv_block_layout: KvBlockLayout,
    block_format: BlockFormat,
}

impl FullyContiguousLayoutBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            config: None,
            memory: None,
            kv_block_layout: KvBlockLayout::Unknown,
            block_format: BlockFormat::default(),
        }
    }

    /// Set the layout configuration.
    #[allow(dead_code)]
    pub fn config(&mut self, config: LayoutConfig) -> &mut Self {
        self.config = Some(config);
        self
    }

    /// Set the memory buffer backing this layout.
    #[allow(dead_code)]
    pub fn memory(&mut self, memory: Buffer) -> &mut Self {
        self.memory = Some(memory);
        self
    }

    /// Set the KV block layout describing dimension ordering.
    ///
    /// Default: `KvBlockLayout::Unknown`
    #[allow(dead_code)]
    pub fn kv_block_layout(&mut self, layout: KvBlockLayout) -> &mut Self {
        self.kv_block_layout = layout;
        self
    }

    /// Set the block format.
    ///
    /// Default: `BlockFormat::default()` (Operational)
    #[allow(dead_code)]
    pub fn block_format(&mut self, format: BlockFormat) -> &mut Self {
        self.block_format = format;
        self
    }

    /// Build the [`FullyContiguousLayout`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `config` is not set
    /// - `memory` is not set
    /// - The memory region is too small for the layout
    /// - The config validation fails
    #[allow(dead_code)]
    pub fn build(&self) -> Result<FullyContiguousLayout> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| anyhow!("config is required"))?;
        let memory = self
            .memory
            .clone()
            .ok_or_else(|| anyhow!("memory is required"))?;

        FullyContiguousLayout::new_internal(config, memory, self.kv_block_layout, self.block_format)
    }
}

impl FullyContiguousLayout {
    /// Create a builder for `FullyContiguousLayout`.
    #[allow(dead_code)]
    pub fn builder() -> FullyContiguousLayoutBuilder {
        FullyContiguousLayoutBuilder::new()
    }

    /// Convenience constructor with `KvBlockLayout::Unknown` and the
    /// default `BlockFormat`. Used by in-module tests; production callers
    /// go through `PhysicalLayoutBuilder` (which threads `KvBlockLayout`
    /// from the labelled probe).
    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn new(config: LayoutConfig, memory: Buffer) -> Result<Self> {
        Self::new_internal(
            config,
            memory,
            KvBlockLayout::Unknown,
            BlockFormat::default(),
        )
    }

    /// Internal constructor with all parameters.
    fn new_internal(
        config: LayoutConfig,
        memory: Buffer,
        kv_block_layout: KvBlockLayout,
        block_format: BlockFormat,
    ) -> Result<Self> {
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
            block_format,
            kv_block_layout,
        })
    }

    /// Create a new fully contiguous layout with a specific block format and KV block layout.
    ///
    /// # Arguments
    /// * `config` - Layout configuration
    /// * `memory` - Owned memory region that backs this layout
    /// * `block_format` - Format of blocks in memory
    /// * `kv_block_layout` - KV block layout describing dimension ordering
    ///
    /// # Returns
    /// A new FullyContiguousLayout instance
    pub(crate) fn new_with_format(
        config: LayoutConfig,
        memory: Buffer,
        block_format: BlockFormat,
        kv_block_layout: KvBlockLayout,
    ) -> Result<Self> {
        Self::new_internal(config, memory, kv_block_layout, block_format)
    }

    /// Get the block format.
    #[allow(dead_code)]
    pub fn block_format(&self) -> BlockFormat {
        self.block_format
    }

    /// Get the KV block layout.
    pub fn kv_block_layout(&self) -> KvBlockLayout {
        self.kv_block_layout
    }

    /// Set the KV block layout.
    #[allow(dead_code)]
    pub fn set_kv_block_layout(&mut self, layout: KvBlockLayout) {
        self.kv_block_layout = layout;
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
                "Block ID {} out of range (count: {})",
                block_id,
                self.config.num_blocks
            ));
        }
        if layer_id >= self.config.num_layers {
            return Err(anyhow!(
                "Layer ID {} out of range (count: {})",
                layer_id,
                self.config.num_layers
            ));
        }
        if outer_id >= self.config.outer_dim {
            return Err(anyhow!(
                "Outer ID {} out of range (count: {})",
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
    #[allow(dead_code)]
    pub fn memory_arc_mut(&mut self) -> &mut Buffer {
        &mut self.memory
    }
}

impl Layout for FullyContiguousLayout {
    /// Projection: single allocation, no region axis.
    ///
    /// Axis order depends on [`KvBlockLayout`]:
    /// - `OperationalNHD`: `[Block, Layer, Outer, Page, HeadCount, HeadSize]`
    /// - `OperationalHND`: `[Block, Layer, Outer, HeadCount, Page, HeadSize]`
    /// - `Universal`:      `[Block, HeadCount, Layer, Outer, Page, HeadSize]`
    ///
    /// The three orderings are what makes the kernel catalog's
    /// signature-keyed dispatch work — NHD vs HND vs Universal differ
    /// only in where `HeadCount` sits relative to the inner axes.
    fn layout_view(&self) -> Result<LayoutView> {
        let cfg = &self.config;
        let block_layout = self.kv_block_layout();
        if matches!(block_layout, KvBlockLayout::Unknown) {
            bail!("FullyContiguousLayout::layout_view: Unknown block layout");
        }
        let (nh, hd) = resolve_head_dims(cfg, block_layout)?;
        let elem = cfg.dtype_width_bytes;
        let buffers = self.memory_regions();
        if buffers.len() != 1 {
            bail!(
                "FullyContiguousLayout::layout_view: expected 1 Buffer, got {}",
                buffers.len()
            );
        }
        let regions = vec![buffers[0].addr()];

        let (dims, sizes, byte_strides) = match block_layout {
            KvBlockLayout::OperationalNHD => {
                // [Block, Layer, Outer, Page, HeadCount, HeadSize]
                let s_hs = elem;
                let s_hc = s_hs * hd;
                let s_pg = s_hc * nh;
                let s_ot = s_pg * cfg.page_size;
                let s_la = s_ot * cfg.outer_dim;
                let s_bk = s_la * cfg.num_layers;
                (
                    vec![
                        KvDim::Block,
                        KvDim::Layer,
                        KvDim::Outer,
                        KvDim::Page,
                        KvDim::HeadCount,
                        KvDim::HeadSize,
                    ],
                    vec![
                        cfg.num_blocks,
                        cfg.num_layers,
                        cfg.outer_dim,
                        cfg.page_size,
                        nh,
                        hd,
                    ],
                    vec![s_bk, s_la, s_ot, s_pg, s_hc, s_hs],
                )
            }
            KvBlockLayout::OperationalHND => {
                // [Block, Layer, Outer, HeadCount, Page, HeadSize]
                let s_hs = elem;
                let s_pg = s_hs * hd;
                let s_hc = s_pg * cfg.page_size;
                let s_ot = s_hc * nh;
                let s_la = s_ot * cfg.outer_dim;
                let s_bk = s_la * cfg.num_layers;
                (
                    vec![
                        KvDim::Block,
                        KvDim::Layer,
                        KvDim::Outer,
                        KvDim::HeadCount,
                        KvDim::Page,
                        KvDim::HeadSize,
                    ],
                    vec![
                        cfg.num_blocks,
                        cfg.num_layers,
                        cfg.outer_dim,
                        nh,
                        cfg.page_size,
                        hd,
                    ],
                    vec![s_bk, s_la, s_ot, s_hc, s_pg, s_hs],
                )
            }
            KvBlockLayout::Universal => {
                // [Block, HeadCount, Layer, Outer, Page, HeadSize] — per-block
                // shape `[nh, nl, no, nt, hd]` (matches universal_from_block /
                // block_from_universal kernel layout in kvbm-kernels).
                let s_hs = elem;
                let s_pg = s_hs * hd;
                let s_ot = s_pg * cfg.page_size;
                let s_la = s_ot * cfg.outer_dim;
                let s_hc = s_la * cfg.num_layers;
                let s_bk = s_hc * nh;
                (
                    vec![
                        KvDim::Block,
                        KvDim::HeadCount,
                        KvDim::Layer,
                        KvDim::Outer,
                        KvDim::Page,
                        KvDim::HeadSize,
                    ],
                    vec![
                        cfg.num_blocks,
                        nh,
                        cfg.num_layers,
                        cfg.outer_dim,
                        cfg.page_size,
                        hd,
                    ],
                    vec![s_bk, s_hc, s_la, s_ot, s_pg, s_hs],
                )
            }
            KvBlockLayout::Custom(_) => bail!(
                "FullyContiguousLayout::layout_view: KvBlockLayout::Custom is not \
                 supported by the planner-driven path"
            ),
            KvBlockLayout::Unknown => unreachable!("Unknown rejected above"),
        };

        let layout = KvDimLayout::new(dims, sizes)?;
        let strides = KvDimStrides::from_byte_strides(byte_strides, elem)?;
        // Homogeneous per-axis storage: the FC layout has a single allocation
        // so every axis lives in the same StorageKind.
        let sk = buffers[0].storage_kind();
        let axis_storage_kinds = vec![sk; layout.dims().len()];
        LayoutView::full(layout, strides, regions, None, axis_storage_kinds)
    }

    fn config(&self) -> &LayoutConfig {
        &self.config
    }

    fn memory_regions(&self) -> &[Buffer] {
        std::slice::from_ref(&self.memory)
    }

    fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryRegion> {
        let addr = self.calculate_address(block_id, layer_id, outer_id)?;
        Ok(MemoryRegion::new(addr, self.region_size))
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
            kv_block_layout: self.kv_block_layout,
        })
    }

    fn block_layout(&self) -> KvBlockLayout {
        self.kv_block_layout
    }
}

impl super::ContiguousBlockLayout for FullyContiguousLayout {
    fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    fn bytes_per_block(&self) -> usize {
        self.block_stride
    }

    fn raw_block(&self, block_id: usize) -> Result<MemoryRegion> {
        if block_id >= self.config.num_blocks {
            return Err(anyhow!(
                "Block ID {} out of range (max: {})",
                block_id,
                self.config.num_blocks
            ));
        }
        let addr = self.base_addr + block_id * self.block_stride;
        Ok(MemoryRegion::new(addr, self.block_stride))
    }

    fn block_layout(&self) -> KvBlockLayout {
        self.kv_block_layout
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
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

        let memory = Buffer::from_arc(MockMemory::new(0x1000, required_bytes));

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
        let memory = Buffer::from_arc(MockMemory::new(0x1000, required_size));
        let layout = FullyContiguousLayout::new(config.clone(), memory).unwrap();

        // Test accessing specific memory regions
        let region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

        // Block 0, Layer 0, Outer 0
        let region = layout.memory_region(0, 0, 0).unwrap();
        assert_eq!(region.addr, 0x1000);
        assert_eq!(region.size(), region_size);

        // Block 0, Layer 0, Outer 1
        let region = layout.memory_region(0, 0, 1).unwrap();
        assert_eq!(region.addr, 0x1000 + region_size);
        assert_eq!(region.size(), region_size);

        // Block 0, Layer 1, Outer 0
        let region = layout.memory_region(0, 1, 0).unwrap();
        assert_eq!(region.addr, 0x1000 + 2 * region_size);
        assert_eq!(region.size(), region_size);

        // Block 1, Layer 0, Outer 0
        let region = layout.memory_region(1, 0, 0).unwrap();
        assert_eq!(
            region.addr,
            0x1000 + (config.outer_dim * config.num_layers * region_size)
        );
        assert_eq!(region.size(), region_size);
    }
}
