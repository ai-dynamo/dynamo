// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decoupled layout system for block management.
//!
//! This module provides a simplified layout abstraction that:
//! - Maps block IDs to physical memory regions (address + size)
//! - Decouples memory regions from storage type information
//! - Specifies allocation requirements without performing allocation
//! - Uses trait objects for memory ownership

pub(crate) mod builder;

mod config;
mod fully_contiguous;
mod kv_block_layout;
mod layer_separate;
mod physical;
mod serialize;
mod validation;
mod view;

#[cfg(all(test, feature = "testing-kvbm"))]
pub(super) mod tests;

// #[cfg(test)]
// mod integration_tests;

pub use builder::PhysicalLayoutBuilder;
pub use config::{BlockDimension, LayoutConfig};
// Re-export `TensorDataType` (consumed by `LayoutConfig.dtype`) so downstream
// crates can populate it without taking a direct `kvbm-kernels` dep.
pub(crate) use fully_contiguous::FullyContiguousLayout;
pub use kv_block_layout::{BlockDim, InnerShape, KvBlockLayout};
pub use kvbm_kernels::TensorDataType;
pub(crate) use layer_separate::LayerSeparateLayout;
pub use physical::NixlMetadata;
pub use physical::PhysicalLayout;
pub use serialize::{
    BlockFormat, FullyContiguousDetails, LayerSeparateDetails, LayoutDescriptor, LayoutTypeDetails,
};
// `intersect_views` is exposed for planner / executor wiring (PR-5+).
// Today only view.rs's tests call it directly.
#[allow(unused_imports)]
pub use view::{LayoutView, intersect_views};

// mod registration;
// pub use registration::{RegisteredLayout, RegisteredStorageMetadata, RegistrationManager};

use std::any::Any;

use anyhow::{Result, bail};

pub(crate) use dynamo_memory::MemoryDescriptor;
pub use dynamo_memory::{Buffer, MemoryRegion};

/// Resolve `(num_heads, head_dim)` from a config, requiring `num_heads`
/// to be set when the projection has been asked to honour a known
/// [`KvBlockLayout`] (the kernel catalog distinguishes NHD/HND/Universal
/// by axis order, which can only be expressed once `inner_dim` is split).
///
/// Validation lives at the projection site, not at `LayoutConfig::build()`,
/// so legacy callers that don't enable `use_planner = true` are not
/// disturbed. Shared by [`Layout::layout_view`] impls.
pub(crate) fn resolve_head_dims(
    cfg: &LayoutConfig,
    block_layout: KvBlockLayout,
) -> Result<(usize, usize)> {
    let nh = cfg.num_heads.ok_or_else(|| {
        anyhow::anyhow!(
            "Layout::layout_view: cfg.num_heads is required when KvBlockLayout = \
             {block_layout:?} (set num_heads on LayoutConfig)"
        )
    })?;
    if !cfg.inner_dim.is_multiple_of(nh) {
        bail!(
            "Layout::layout_view: inner_dim ({}) is not divisible by num_heads ({}) \
             — cannot split into HeadCount × HeadSize",
            cfg.inner_dim,
            nh,
        );
    }
    Ok((nh, cfg.inner_dim / nh))
}

/// Core layout trait for mapping block IDs to memory regions.
///
/// Layouts specify how KV cache blocks are organized in memory without
/// performing allocation themselves. They provide:
/// - Memory region lookup for specific blocks
/// - Allocation requirements for external allocators
/// - Metadata about block organization
pub trait Layout: Any + Send + Sync + std::fmt::Debug {
    /// Project this layout to a labelled [`LayoutView`] for the
    /// stride-aware planner.
    ///
    /// Each concrete impl owns the projection of its own per-axis stride
    /// math: [`FullyContiguousLayout`] emits a single-region view with
    /// the full `[Block, Layer, Outer, …]` axis order keyed off
    /// [`KvBlockLayout`]; [`LayerSeparateLayout`] emits a per-layer
    /// region view with `region_axis = Some(KvDim::Layer)`. Layouts
    /// that cannot project (test mocks, future variants the planner
    /// doesn't yet understand) bail with a precise error.
    ///
    /// `LayoutView` is `pub(crate)` — external callers consume layouts
    /// through [`PhysicalLayout`] and never see this method's return
    /// value. The `#[allow]` suppresses the `private_interfaces`
    /// warning that would otherwise fire because [`Layout`] is `pub`.
    /// A follow-up may demote [`Layout`] itself to `pub(crate)`
    /// (external crates only use [`PhysicalLayout`]) and remove this.
    #[allow(private_interfaces)]
    fn layout_view(&self) -> Result<LayoutView>;

    /// Get the configuration for this layout.
    fn config(&self) -> &LayoutConfig;

    /// Get the root memory regions backing this layout.
    ///
    /// These regions correspond to the concrete allocations that store the layout's data.
    /// Implementations that derive memory procedurally can return an empty slice.
    fn memory_regions(&self) -> &[Buffer];

    /// Get memory regions for a specific block_id, layer_id, outer_id.
    ///
    /// Returns a [MemoryRegion] for the continuous region specified by the given block_id,
    /// layer_id, outer_id.
    ///
    /// # Arguments
    /// * `block_id` - The ID of the block to query (0..num_blocks)
    /// * `layer_id` - The ID of the layer to query (0..num_layers)
    /// * `outer_id` - The ID of the outer dimension to query (0..outer_dim)
    fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryRegion>;

    /// Get the allocation requirements for this layout.
    ///
    /// Returns a vector of allocation sizes needed to back this layout.
    /// For fully contiguous layouts, this will be a single size.
    /// For layer-separate layouts, this will contain one size per layer.
    ///
    /// # Returns
    /// Vector of allocation sizes in bytes.
    fn required_allocations(&self) -> Vec<usize>;

    /// Check if this layout uses fully contiguous memory.
    ///
    /// Fully contiguous layouts have all blocks in a single allocation,
    /// which enables certain optimizations.
    fn is_fully_contiguous(&self) -> bool;

    /// Get the total number of blocks in this layout.
    fn num_blocks(&self) -> usize;

    /// Get the number of layers per block.
    fn num_layers(&self) -> usize;

    /// Get the outer dimension size.
    ///
    /// In typical KV cache layouts, this is often 2 (for K and V),
    /// but can be 1 for architectures like MLA.
    fn outer_dim(&self) -> usize;

    /// Get the page size (often corresponds to block size in tokens).
    fn page_size(&self) -> usize;

    /// Get the inner dimension size.
    ///
    /// This is typically the hidden size divided by tensor parallel size.
    fn inner_dim(&self) -> usize;

    /// Get the data type width in bytes.
    fn dtype_width_bytes(&self) -> usize;

    /// Get serialization details for this layout type.
    ///
    /// This provides the layout-type-specific information needed to serialize
    /// and reconstruct the layout on a remote node.
    fn serialization_details(&self) -> serialize::LayoutTypeDetails;

    /// Get the KV block layout describing how dimensions are permuted within blocks.
    ///
    /// Returns the internal tensor ordering for blocks in this layout.
    /// For layer-separate layouts, this describes the inner tensor format.
    /// For fully contiguous layouts, this describes the full block format.
    fn block_layout(&self) -> KvBlockLayout;
}

/// Trait for layouts that provide contiguous per-block memory regions.
///
/// This trait enables direct access to entire blocks as contiguous memory,
/// without requiring layer/outer indexing. It is implemented by
/// [`FullyContiguousLayout`] but NOT by [`LayerSeparateLayout`] (which
/// stores each layer separately).
///
/// Use this trait when you need to:
/// - Access raw block memory for transformation kernels
/// - Reinterpret block memory under different [`KvBlockLayout`] formats
/// - Perform whole-block operations without layer decomposition
pub trait ContiguousBlockLayout: Send + Sync + std::fmt::Debug {
    /// Get the total number of blocks in this layout.
    fn num_blocks(&self) -> usize;

    /// Get the size of each block in bytes.
    fn bytes_per_block(&self) -> usize;

    /// Get the contiguous memory region for a specific block.
    ///
    /// # Arguments
    /// * `block_id` - The ID of the block to query (0..num_blocks)
    ///
    /// # Returns
    /// A [`MemoryRegion`] covering the entire block's memory.
    ///
    /// # Errors
    /// Returns an error if `block_id` is out of range.
    fn raw_block(&self, block_id: usize) -> Result<MemoryRegion>;

    /// Get the KV block layout for this contiguous layout.
    fn block_layout(&self) -> KvBlockLayout;
}
