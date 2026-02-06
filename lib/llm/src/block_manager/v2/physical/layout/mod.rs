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
mod layer_separate;
mod physical;
mod serialize;
mod validation;

#[cfg(all(test, feature = "testing-nixl"))]
pub(super) mod tests;

// #[cfg(test)]
// mod integration_tests;

pub use builder::{LayoutKind, PhysicalLayoutBuilder};
pub use config::{BlockDimension, LayoutConfig};
pub use fully_contiguous::FullyContiguousLayout;
pub use layer_separate::LayerSeparateLayout;
pub use physical::{NixlMetadata, PhysicalLayout};
pub use serialize::{
    BlockFormat, FullyContiguousDetails, LayerSeparateDetails, LayoutDescriptor, LayoutTypeDetails,
};
pub use validation::{TensorFormat, validate_tensor_shapes, validate_tensor_strides};

// mod registration;
// pub use registration::{RegisteredLayout, RegisteredStorageMetadata, RegistrationManager};

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use crate::block_manager::v2::memory::{MemoryDescriptor, MemoryRegion, OwnedMemoryRegion};

/// Core layout trait for mapping block IDs to memory regions.
///
/// Layouts specify how KV cache blocks are organized in memory without
/// performing allocation themselves. They provide:
/// - Memory region lookup for specific blocks
/// - Allocation requirements for external allocators
/// - Metadata about block organization
pub trait Layout: Send + Sync + std::fmt::Debug {
    /// Get the configuration for this layout.
    fn config(&self) -> &LayoutConfig;

    /// Get the root memory regions backing this layout.
    ///
    /// These regions correspond to the concrete allocations that store the layout's data.
    /// Implementations that derive memory procedurally can return an empty slice.
    fn memory_regions(&self) -> &[OwnedMemoryRegion];

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
    ) -> Result<MemoryDescriptor>;

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
}

/// Inner shape format for tensor layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnerShape {
    /// Unknown shape - fallback when we can't determine the format
    Unknown,
    /// NHD format: [block_size, num_heads, head_dim]
    /// Common for attention layers where N=tokens, H=heads, D=dimension
    NHD,
    /// HND format: [num_heads, block_size, head_dim]
    /// Alternative layout with heads first
    HND,
}
