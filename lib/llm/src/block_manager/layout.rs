// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! # Block Layout Management 🧱
//!
//! This module is responsible for defining and managing the memory layout of data blocks.
//! It provides the foundational traits and concrete implementations for how blocks,
//! composed of multiple layers and pages, are arranged within a given [`Storage`].
//! The primary goal is to abstract the complexities of memory organization, including
//! contiguity, strides, and alignment, to ensure efficient data access and manipulation.
//!
//! ## Core Concepts
//!
//! ### 1. Layout Traits
//! The module defines a set of traits to ensure a consistent interface across different layout strategies:
//! - [`BlockLayout`]: The central trait that combines configuration and lookup capabilities. It specifies the
//!   associated [`StorageType`].
//! - [`BlockLayoutConfig`]: Provides metadata about the layout, such as the number of blocks, layers, page size,
//!   and data type.
//! - [`BlockLayoutLookup`]: Offers methods to retrieve the memory address and size of a specific memory region
//!   (page) within the layout.
//!
//! ### 2. Layout Configuration
//! The [`LayoutConfig`] struct is used to define the parameters of a block layout, including:
//! - `num_blocks`: Total number of blocks.
//! - `num_layers`: Number of layers per block.
//! - `page_size`: Size of each page (often corresponds to a dimension like sequence length or number of tokens).
//! - `inner_dim`: The inner dimension of the data (e.g., hidden size).
//! - `alignment`: Required memory alignment for certain operations or hardware. Must be a power of 2.
//! - `dtype`: The data type ([`DType`]) of the elements stored.
//!
//! This configuration is validated to ensure consistency and correctness (e.g., alignment must be a power of 2).
//!
//! ### 3. Concrete Layouts
//! Currently, the primary implemented layout is:
//! - [`FullyContiguous<S>`]: Represents a layout where all blocks and their constituent layers are stored sequentially
//!   in a single contiguous memory region provided by the generic storage `S`. It handles potential alignment
//!   requirements by calculating a `base_offset` within the provided storage and adjusting strides between blocks if
//!   necessary.
//!
//! ### 4. Strides and Alignment
//! The layout calculations meticulously handle strides between layers and blocks. For instance, in [`FullyContiguousConfig`]:
//! - `layer_stride_in_bytes`: The size of one memory region (page).
//! - `natural_block_stride`: The size of one block if there were no additional alignment padding between blocks.
//! - `block_stride_in_bytes`: The actual stride between the start of consecutive blocks, potentially larger than
//!   `natural_block_stride` to meet `alignment` requirements.
//! - `base_offset`: An offset applied from the start of the allocated [`Storage`] to ensure the first block's
//!   data begins at an aligned address.
//!
//! The function `align_up` is a utility to ensure values are aligned to the nearest multiple of a power-of-2 alignment.
//!
//! ### 5. Storage Interaction
//! Layouts are tightly coupled with the [`Storage`] trait from the `super::storage` module.
//! The [`BlockLayout::allocate`] method uses a [`StorageAllocator`] to obtain the necessary memory,
//! calculating the required size including any padding for alignment.
//!
//! ### 6. Error Handling
//! Operations within this module can result in [`LayoutError`], which covers issues like invalid configuration, validation errors, or out-of-bounds indexing.
//!
//! ## Usage Example
//!
//! ```rust
//! use dynamo_llm::block_manager::layout::{
//!     LayoutConfig, FullyContiguous, BlockLayout, BlockLayoutLookup, BlockLayoutConfig,
//! };
//! use dynamo_llm::block_manager::storage::{SystemAllocator, StorageType};
//! use dynamo_llm::common::dtype::DType;
//!
//! // Define the layout configuration
//! let config = LayoutConfig::builder()
//!     .num_blocks(10)
//!     .num_layers(4)
//!     .outer_dim(1)
//!     .page_size(16)
//!     .inner_dim(128)
//!     .dtype(DType::FP16)
//!     .build()
//!     .unwrap();
//!
//!
//! // Allocate a FullyContiguous layout using a SystemAllocator
//! let allocator = SystemAllocator;
//! let layout = FullyContiguous::allocate(config, &allocator).unwrap();
//!
//! // Access layout properties
//! assert_eq!(layout.num_blocks(), 10);
//! assert_eq!(layout.storage_type(), StorageType::System);
//!
//! // Get the address of a specific page
//! let addr = layout.memory_region_addr(0, 0).unwrap();
//! println!("Address of block 0, layer 0: {}", addr);
//! ```
//!
//! ## NIXL Integration
//! This module also includes a submodule `nixl` ([`crate::block_manager::layout::nixl`])
//! which extends these layout concepts for NIXL (NVIDIA Interface eXchange Layer), enabling
//! layouts to be registered and serialized for use in distributed environments.

// todo: coming soon...
// pub mod distributed;

pub mod nixl;
/// Utility functions for layout validation and verification
pub mod utils;

use utils::*;

use derive_getters::Getters;
use thiserror::Error;

use crate::block_manager::storage::{Storage, StorageAllocator};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use validator::Validate;

use super::storage::StorageType;

/// Errors that can occur during layout operations
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum LayoutError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Validation failed: {0}")]
    ValidationError(#[from] validator::ValidationErrors),

    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(usize),

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Invalid outer index: {0}")]
    InvalidOuterIndex(usize),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

/// Storage pattern for layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutType {
    /// All layers are contiguous in memory [n_blocks, n_layers, outer_dim, ...]
    FullyContiguous,

    /// All layers are stored separately.
    /// The outer_contiguous field is auto-detected from tensor shapes when not explicitly set.
    /// If outer_contiguous: for each layer: [outer_dim, n_blocks, ...]
    /// If !outer_contiguous: for each layer: [n_blocks, outer_dim, ...]
    /// When outer_dim is 1, these two modes are equivalent.
    LayerSeparate {
        /// If true, the outer dimension is contiguous. Auto-detected from tensor shapes when possible.
        outer_contiguous: bool,
    },
}

impl LayoutType {
    /// Create a LayerSeparate layout type with auto-detection based on tensor shapes
    pub fn layer_separate_auto(shape: &[usize], num_device_blocks: usize) -> anyhow::Result<Self> {
        let outer_contiguous = if shape[0] >= num_device_blocks {
            false // Block contiguous: [n_blocks, outer_dim, ...]
        } else if shape[1] >= num_device_blocks {
            true // Outer contiguous: [outer_dim, n_blocks, ...]
        } else {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        };

        Ok(LayoutType::LayerSeparate { outer_contiguous })
    }

    /// Create a LayerSeparate layout type with default auto-detection (defaults to outer_contiguous=true)
    /// Use this when tensor shapes are not available
    pub fn layer_separate_auto_default() -> Self {
        LayoutType::LayerSeparate {
            outer_contiguous: true,
        }
    }

    /// Create a LayerSeparate layout type with explicit outer_contiguous setting
    pub fn layer_separate(outer_contiguous: bool) -> Self {
        LayoutType::LayerSeparate { outer_contiguous }
    }
}

/// Local Memory Region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Getters)]
pub struct LocalMemoryRegion {
    #[getter(copy)]
    addr: usize,

    #[getter(copy)]
    size: usize,

    #[getter(copy)]
    storage_type: StorageType,
}

/// Core trait for block layouts
pub trait BlockLayout: GenericBlockLayout {
    /// The type of storage this layout uses
    type StorageType: Storage;

    /// Returns the layout type
    fn layout_type(&self) -> LayoutType;

    /// Get the memory regions for all blocks and layers
    fn storage(&self) -> Vec<&Self::StorageType>;

    /// Get the mutable memory regions for all blocks and layers
    fn storage_mut(&mut self) -> Vec<&mut Self::StorageType>;
}

/// Generic trait for block layouts - type-erased on the [Storage] object.
pub trait GenericBlockLayout: BlockLayoutConfig + Send + Sync {
    /// Storage type for the layout
    fn storage_type(&self) -> &StorageType;

    /// Full configuration for the layout
    fn config(&self) -> &LayoutConfig;

    /// Get the memory region for a specific page [page_size, inner_dim]
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block
    /// * `layer_idx` - The index of the layer
    /// * `outer_idx` - The index of the outer dimension, e.g. if
    ///
    fn memory_region(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<LocalMemoryRegion, LayoutError>;
}

/// Configuration for block layouts
pub trait BlockLayoutConfig: std::fmt::Debug {
    /// Returns the layout config
    fn layout_config(&self) -> LayoutConfig;

    /// Returns the total number of blocks this layout manages
    fn num_blocks(&self) -> usize {
        self.layout_config().num_blocks
    }

    /// Returns the number of layers per block
    fn num_layers(&self) -> usize {
        self.layout_config().num_layers
    }

    /// Returns the number of outer dimensions per block
    /// In some cases, K and V might be indexed separately, so in that example one might have 2 outer dimensions
    /// For MLA, this is 1.
    /// The location of the outer dimension in the shape of the tensor layout is defined by the layout type.
    fn outer_dim(&self) -> usize {
        self.layout_config().outer_dim
    }

    /// Returns the size of each block in bytes
    fn page_size(&self) -> usize {
        self.layout_config().page_size
    }

    /// Returns the inner dimension size
    fn inner_dim(&self) -> usize {
        self.layout_config().inner_dim
    }

    /// The size of the data for a layout (pre base_offset)
    fn layout_data_bytes(&self) -> usize;
}

/// Configuration for block layouts
#[derive(Debug, Clone, Builder, Validate, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutConfig {
    /// Number of blocks
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    /// Number of layers
    #[validate(range(min = 1))]
    pub num_layers: usize,

    /// Number of outer dimensions
    #[validate(range(min = 1, max = 2))]
    pub outer_dim: usize,

    /// Page size
    #[validate(range(min = 1))]
    pub page_size: usize,

    /// Inner dimension
    #[validate(range(min = 1))]
    pub inner_dim: usize,

    /// Alignment
    #[validate(custom(function = "validate_power_of_2"))]
    #[builder(default = "1")]
    pub alignment: usize,

    /// Data type
    #[builder(default = "2")]
    pub dtype_width_bytes: usize,
}

impl LayoutConfig {
    /// Builder for LayoutConfig
    pub fn builder() -> LayoutConfigBuilder {
        LayoutConfigBuilder::default()
    }
}

/// Internal struct to hold calculated layout dimensions specific to FullyContiguous.
// Module-level, but only used internally by FullyContiguous
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FullyContiguousConfig {
    inner: LayoutConfig,

    /// Minimum contiguous memory region size
    /// Inner dimension * page size * dtype size
    memory_region_size: usize,

    /// Stride between outer dimensions
    outer_dim_stride_in_bytes: usize,

    /// Stride between layers
    layer_stride_in_bytes: usize,

    /// Natural block stride
    natural_block_stride: usize,

    /// Block stride in bytes
    block_stride_in_bytes: usize, // Aligned if necessary

    /// Size of the layout data itself (post base offset)
    layout_data_bytes: usize, // Size of the layout data itself (post base offset)
}

impl FullyContiguousConfig {
    /// Calculates the core dimensions based on the configuration.
    /// Returns an error if the configuration is invalid.
    fn new(config: LayoutConfig) -> Result<Self, LayoutError> {
        // Validate first, propagating errors via `?`
        config.validate()?;

        let alignment = config.alignment;
        let outer_dim_stride_in_bytes =
            config.page_size * config.inner_dim * config.dtype_width_bytes;
        let layer_stride_in_bytes = outer_dim_stride_in_bytes * config.outer_dim;
        let memory_region_size = outer_dim_stride_in_bytes;
        let natural_block_stride = config.num_layers * layer_stride_in_bytes;

        let block_stride_in_bytes = if alignment > 1 {
            align_up(natural_block_stride, alignment)
        } else {
            natural_block_stride
        };

        let layout_data_bytes =
            (config.num_blocks - 1) * block_stride_in_bytes + natural_block_stride;

        Ok(Self {
            inner: config,
            memory_region_size,
            outer_dim_stride_in_bytes,
            layer_stride_in_bytes,
            natural_block_stride,
            block_stride_in_bytes,
            layout_data_bytes,
        })
    }

    /// Calculate the total number of bytes required for allocation, including initial alignment padding.
    /// Panics if the provided configuration is invalid.
    pub fn required_allocation_size(&self) -> usize {
        let initial_padding = self.inner.alignment.saturating_sub(1);
        self.layout_data_bytes + initial_padding
    }
}

impl BlockLayoutConfig for FullyContiguousConfig {
    fn layout_config(&self) -> LayoutConfig {
        self.inner.clone()
    }

    fn layout_data_bytes(&self) -> usize {
        self.layout_data_bytes
    }
}

/// Contiguous memory layout where all blocks and layers are sequential
#[derive(Debug)]
pub struct FullyContiguous<S: Storage> {
    /// Configuration for the layout
    config: FullyContiguousConfig,

    /// Storage for the layoutk
    storage: S,

    /// Storage type for the layout
    storage_type: StorageType,

    // Offset from storage.addr() to the aligned start of block 0
    base_offset: usize,
}

impl<S: Storage> FullyContiguous<S> {
    /// Create a new contiguous layout using the provided configuration and pre-allocated storage.
    /// Performs validation and calculates strides/offsets.
    #[instrument(level = "debug", skip(storage), fields(config = ?config))]
    pub fn new(config: LayoutConfig, mut storage: Vec<S>) -> Result<Self, LayoutError> {
        // Calculate dimensions, which includes validation.
        let config = FullyContiguousConfig::new(config)?;

        if storage.len() != 1 {
            return Err(LayoutError::InvalidConfig(
                "FullyContiguous layout requires exactly one storage region".to_string(),
            ));
        }
        let storage = storage.remove(0);
        let storage_type = storage.storage_type();

        let base_offset = validate_storage(&storage, &config)?;

        tracing::debug!(
            config.memory_region_size,
            config.layer_stride_in_bytes,
            config.block_stride_in_bytes,
            config.natural_block_stride,
            alignment = config.inner.alignment,
            base_offset,
            "Calculated layout strides (aligned)"
        );

        Ok(Self {
            config,
            storage,
            storage_type,
            base_offset,
        })
    }

    /// Internal constructor used for reconstruction from serialized parts.
    /// Assumes the provided config, storage, and base_offset are consistent
    /// and skips size/alignment validation against the storage.
    pub(crate) fn new_internal(
        config: FullyContiguousConfig,
        storage: S,
        storage_type: StorageType,
        base_offset: usize,
    ) -> Result<Self, LayoutError> {
        // Basic check: Ensure the storage address matches expectations based on offset if possible?
        // Maybe not strictly necessary if we trust the serialized data.
        Ok(Self {
            config,
            storage,
            storage_type,
            base_offset,
        })
    }

    /// Allocate storage using the provided allocator and create a new FullyContiguous layout.
    ///
    /// Calculates the required size based on the configuration, allocates the storage
    /// (including potential padding for initial alignment), and then constructs the
    /// `FullyContiguous` layout instance.
    ///
    /// # Type Parameters
    ///
    /// * `A`: The type of the storage allocator, implementing `StorageAllocator<S>`.
    ///
    /// # Arguments
    ///
    /// * `config` - The layout configuration.
    /// * `allocator` - A reference to the storage allocator.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `FullyContiguous<S>` instance or an error if allocation
    /// or layout creation fails.
    #[instrument(level = "debug", skip(allocator), fields(config = ?config))]
    pub fn allocate(
        config: LayoutConfig,
        allocator: &dyn StorageAllocator<S>,
    ) -> Result<Self, LayoutError> {
        // Calculate total bytes needed. Propagate error if config is invalid.
        let config = FullyContiguousConfig::new(config)?;
        let bytes_to_allocate = config.required_allocation_size();

        tracing::debug!(
            bytes_to_allocate,
            alignment = config.inner.alignment,
            "Calculated storage size for allocation (with alignment padding)"
        );

        let storage = allocator.allocate(bytes_to_allocate).map_err(|e| {
            LayoutError::OperationFailed(format!("Storage allocation failed: {}", e))
        })?;
        tracing::debug!(
            allocated_size = storage.size(),
            allocated_addr = storage.addr(),
            "Storage allocated successfully"
        );

        // Pass the config by value as Self::new takes ownership
        Self::new(config.inner, vec![storage])
    }
}

impl<S: Storage> BlockLayout for FullyContiguous<S> {
    type StorageType = S;

    fn layout_type(&self) -> LayoutType {
        LayoutType::FullyContiguous
    }

    fn storage(&self) -> Vec<&Self::StorageType> {
        vec![&self.storage]
    }

    fn storage_mut(&mut self) -> Vec<&mut Self::StorageType> {
        vec![&mut self.storage]
    }
}

impl<S: Storage> GenericBlockLayout for FullyContiguous<S> {
    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }

    fn config(&self) -> &LayoutConfig {
        &self.config.inner
    }

    fn memory_region(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<LocalMemoryRegion, LayoutError> {
        validate_indices(&self.config, block_idx, layer_idx, outer_idx)?;

        // Start from the aligned base address
        let aligned_start_addr = self.storage.addr() as usize + self.base_offset;

        // Calculate offset relative to the aligned start using stored config
        let block_offset = block_idx * self.config.block_stride_in_bytes;
        let layer_offset = layer_idx * self.config.layer_stride_in_bytes;
        let outer_offset = outer_idx * self.config.outer_dim_stride_in_bytes;
        let final_addr = aligned_start_addr + block_offset + layer_offset + outer_offset;

        Ok(LocalMemoryRegion {
            addr: final_addr,
            size: self.config.memory_region_size,
            storage_type: self.storage_type,
        })
    }
}

impl<S: Storage> BlockLayoutConfig for FullyContiguous<S> {
    fn layout_config(&self) -> LayoutConfig {
        self.config.inner.clone()
    }

    fn layout_data_bytes(&self) -> usize {
        self.config.layout_data_bytes
    }
}

impl<S: Storage> FullyContiguous<S> {
    /// Verify memory region addressing is correct for this layout
    pub fn verify_memory_regions(&self) -> Result<(), LayoutError> {
        use crate::block_manager::layout::utils::WorkerLayoutVerifier;

        let mut verifier = WorkerLayoutVerifier::new();
        let results = verifier.verify_layout_consistency(self)?;

        if verifier.has_critical_mismatches() {
            tracing::error!(
                "FullyContiguous layout verification failed: {} regions checked, {} size mismatches",
                results.len(),
                results.iter().filter(|r| !r.size_matches).count()
            );
            return Err(LayoutError::InvalidConfig(
                "Memory region verification failed".to_string(),
            ));
        }

        tracing::debug!(
            "FullyContiguous layout verification passed: {} regions checked",
            results.len()
        );
        Ok(())
    }

    /// Get expected memory address for a region (for testing/verification)
    pub fn expected_memory_address(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<usize, LayoutError> {
        validate_indices(&self.config, block_idx, layer_idx, outer_idx)?;

        let aligned_start_addr = self.storage.addr() as usize + self.base_offset;
        let block_offset = block_idx * self.config.block_stride_in_bytes;
        let layer_offset = layer_idx * self.config.layer_stride_in_bytes;
        let outer_offset = outer_idx * self.config.outer_dim_stride_in_bytes;

        Ok(aligned_start_addr + block_offset + layer_offset + outer_offset)
    }

    /// Verify a specific memory region matches expected calculations
    pub fn verify_memory_region(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<bool, LayoutError> {
        let actual_region = self.memory_region(block_idx, layer_idx, outer_idx)?;
        let expected_addr = self.expected_memory_address(block_idx, layer_idx, outer_idx)?;
        let expected_size = self.config.memory_region_size;

        let addr_matches = actual_region.addr == expected_addr;
        let size_matches = actual_region.size == expected_size;

        if !addr_matches || !size_matches {
            tracing::warn!(
                "Memory region mismatch at ({}, {}, {}): addr {} vs {} (expected), size {} vs {} (expected)",
                block_idx,
                layer_idx,
                outer_idx,
                actual_region.addr,
                expected_addr,
                actual_region.size,
                expected_size
            );
        }

        Ok(addr_matches && size_matches)
    }
}

/// Configuration for layer-separated layouts.
/// This is used in vLLM, where every layer has its own allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LayerSeparateConfig {
    inner: LayoutConfig,

    /// Size of each contiguous memory region
    memory_region_size: usize,

    /// Stride between outer dimensions
    outer_dim_stride_in_bytes: usize,

    /// Block stride in bytes
    block_stride_in_bytes: usize,

    /// Size of the layout data itself (post base offset)
    layout_data_bytes: usize,

    /// Indicator for outer contiguous or block contiguous
    is_outer_contiguous: bool,
}

impl LayerSeparateConfig {
    fn new(config: LayoutConfig, is_outer_contiguous: bool) -> Result<Self, LayoutError> {
        config.validate()?;

        let alignment = config.alignment;
        let memory_region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

        let outer_dim_stride_in_bytes;
        let block_stride_in_bytes;
        let layout_data_bytes;

        if is_outer_contiguous {
            block_stride_in_bytes = if alignment > 1 {
                align_up(memory_region_size, alignment)
            } else {
                memory_region_size
            };
            outer_dim_stride_in_bytes = block_stride_in_bytes * config.num_blocks;
            layout_data_bytes = outer_dim_stride_in_bytes * config.outer_dim;
        } else {
            outer_dim_stride_in_bytes = memory_region_size;
            let natural_block_stride = outer_dim_stride_in_bytes * config.outer_dim;
            block_stride_in_bytes = if alignment > 1 {
                align_up(natural_block_stride, alignment)
            } else {
                natural_block_stride
            };
            layout_data_bytes = block_stride_in_bytes * config.num_blocks;
        }

        Ok(Self {
            inner: config,
            memory_region_size,
            outer_dim_stride_in_bytes,
            block_stride_in_bytes,
            layout_data_bytes,
            is_outer_contiguous,
        })
    }

    pub fn required_allocation_size(&self) -> usize {
        let initial_padding = self.inner.alignment.saturating_sub(1);
        self.layout_data_bytes + initial_padding
    }
}

impl BlockLayoutConfig for LayerSeparateConfig {
    fn layout_config(&self) -> LayoutConfig {
        self.inner.clone()
    }

    fn layout_data_bytes(&self) -> usize {
        self.layout_data_bytes
    }
}

/// Layer-separated layout where each layer has its own allocation.
#[derive(Debug)]
pub struct LayerSeparate<S: Storage> {
    /// Configuration for the layout
    config: LayerSeparateConfig,

    /// Storage for the layout
    storages: Vec<S>,

    /// Storage type for the layout
    storage_type: StorageType,

    /// Base offset from storage.addr() to the aligned start of block 0
    base_offsets: Vec<usize>,
}

impl<S: Storage> LayerSeparate<S> {
    /// Create a new LayerSeparate layout.
    #[instrument(level = "debug", skip(storages), fields(config = ?config))]
    pub fn new(
        config: LayoutConfig,
        storages: Vec<S>,
        is_outer_contiguous: bool,
    ) -> Result<Self, LayoutError> {
        if storages.len() != config.num_layers {
            return Err(LayoutError::InvalidConfig(
                "LayerSeparate layout requires exactly one storage region per layer".to_string(),
            ));
        }

        let config = LayerSeparateConfig::new(config, is_outer_contiguous)?;

        let storage_type = storages[0].storage_type();
        let mut base_offsets = Vec::new();
        for storage in &storages {
            let base_offset = validate_storage(storage, &config)?;

            tracing::debug!(
                config.memory_region_size,
                config.block_stride_in_bytes,
                config.outer_dim_stride_in_bytes,
                alignment = config.inner.alignment,
                base_offset,
                "Calculated layout strides (aligned)"
            );

            base_offsets.push(base_offset);
        }

        Ok(Self {
            config,
            storages,
            storage_type,
            base_offsets,
        })
    }

    pub(crate) fn new_internal(
        config: LayerSeparateConfig,
        storages: Vec<S>,
        storage_type: StorageType,
        base_offsets: Vec<usize>,
    ) -> Result<Self, LayoutError> {
        Ok(Self {
            config,
            storages,
            storage_type,
            base_offsets,
        })
    }

    /// Allocate a new LayerSeparate layout.
    /// `is_outer_contiguous` determines whether the outer dimension or the block dimension is contiguous.
    /// The amount of [`Storage`]s allocated is equal to the number of layers in the config.
    pub fn allocate(
        config: LayoutConfig,
        allocator: &dyn StorageAllocator<S>,
        is_outer_contiguous: bool,
    ) -> Result<Self, LayoutError> {
        // Calculate total bytes needed. Propagate error if config is invalid.
        let config = LayerSeparateConfig::new(config, is_outer_contiguous)?;
        let bytes_to_allocate = config.required_allocation_size();

        tracing::debug!(
            bytes_to_allocate,
            alignment = config.inner.alignment,
            "Calculated storage size for allocation (with alignment padding)"
        );

        let mut storages = Vec::new();

        for _ in 0..config.inner.num_layers {
            let storage = allocator.allocate(bytes_to_allocate).map_err(|e| {
                LayoutError::OperationFailed(format!("Storage allocation failed: {}", e))
            })?;
            storages.push(storage);
        }

        tracing::debug!(
            allocated_size = storages[0].size(),
            allocated_addr = storages[0].addr(),
            "Storage allocated successfully"
        );

        // Pass the config by value as Self::new takes ownership
        Self::new(config.inner, storages, is_outer_contiguous)
    }
}

impl<S: Storage> GenericBlockLayout for LayerSeparate<S> {
    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }

    fn config(&self) -> &LayoutConfig {
        &self.config.inner
    }

    fn memory_region(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<LocalMemoryRegion, LayoutError> {
        validate_indices(&self.config, block_idx, layer_idx, outer_idx)?;

        // Start from the aligned base address
        let aligned_start_addr =
            self.storages[layer_idx].addr() as usize + self.base_offsets[layer_idx];

        // Calculate offset relative to the aligned start using stored config
        let block_offset = block_idx * self.config.block_stride_in_bytes;
        let outer_offset = outer_idx * self.config.outer_dim_stride_in_bytes;
        let final_addr = aligned_start_addr + block_offset + outer_offset;

        Ok(LocalMemoryRegion {
            addr: final_addr,
            size: self.config.memory_region_size,
            storage_type: self.storages[layer_idx].storage_type(),
        })
    }
}

impl<S: Storage> BlockLayout for LayerSeparate<S> {
    type StorageType = S;

    fn layout_type(&self) -> LayoutType {
        LayoutType::LayerSeparate {
            outer_contiguous: self.config.is_outer_contiguous,
        }
    }

    fn storage(&self) -> Vec<&Self::StorageType> {
        self.storages.iter().collect()
    }

    fn storage_mut(&mut self) -> Vec<&mut Self::StorageType> {
        self.storages.iter_mut().collect()
    }
}

impl<S: Storage> BlockLayoutConfig for LayerSeparate<S> {
    fn layout_config(&self) -> LayoutConfig {
        self.config.inner.clone()
    }

    fn layout_data_bytes(&self) -> usize {
        self.config.layout_data_bytes
    }
}

impl<S: Storage> LayerSeparate<S> {
    /// Verify memory region addressing is correct for this layout
    pub fn verify_memory_regions(&self) -> Result<(), LayoutError> {
        use crate::block_manager::layout::utils::WorkerLayoutVerifier;

        let mut verifier = WorkerLayoutVerifier::new();
        let results = verifier.verify_layout_consistency(self)?;

        if verifier.has_critical_mismatches() {
            tracing::error!(
                "LayerSeparate layout verification failed: {} regions checked, {} size mismatches",
                results.len(),
                results.iter().filter(|r| !r.size_matches).count()
            );
            return Err(LayoutError::InvalidConfig(
                "Memory region verification failed".to_string(),
            ));
        }

        tracing::debug!(
            "LayerSeparate layout verification passed: {} regions checked",
            results.len()
        );
        Ok(())
    }

    /// Get expected memory address for a region (for testing/verification)
    pub fn expected_memory_address(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<usize, LayoutError> {
        validate_indices(&self.config, block_idx, layer_idx, outer_idx)?;

        let aligned_start_addr =
            self.storages[layer_idx].addr() as usize + self.base_offsets[layer_idx];
        let block_offset = block_idx * self.config.block_stride_in_bytes;
        let outer_offset = outer_idx * self.config.outer_dim_stride_in_bytes;

        Ok(aligned_start_addr + block_offset + outer_offset)
    }

    /// Verify a specific memory region matches expected calculations
    pub fn verify_memory_region(
        &self,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<bool, LayoutError> {
        let actual_region = self.memory_region(block_idx, layer_idx, outer_idx)?;
        let expected_addr = self.expected_memory_address(block_idx, layer_idx, outer_idx)?;
        let expected_size = self.config.memory_region_size;

        let addr_matches = actual_region.addr == expected_addr;
        let size_matches = actual_region.size == expected_size;

        if !addr_matches || !size_matches {
            tracing::warn!(
                "LayerSeparate memory region mismatch at ({}, {}, {}): addr {} vs {} (expected), size {} vs {} (expected)",
                block_idx,
                layer_idx,
                outer_idx,
                actual_region.addr,
                expected_addr,
                actual_region.size,
                expected_size
            );
        }

        Ok(addr_matches && size_matches)
    }

    /// Verify all storage regions are properly aligned and sized
    pub fn verify_storage_alignment(&self) -> Result<(), LayoutError> {
        let alignment = self.config.inner.alignment;

        for (layer_idx, storage) in self.storages.iter().enumerate() {
            let storage_addr = storage.addr() as usize;
            let base_offset = self.base_offsets[layer_idx];
            let aligned_addr = storage_addr + base_offset;

            // Check alignment
            if alignment > 1 && !aligned_addr.is_multiple_of(alignment) {
                return Err(LayoutError::InvalidConfig(format!(
                    "Layer {} storage not properly aligned: addr {} + offset {} = {} is not {} byte aligned",
                    layer_idx, storage_addr, base_offset, aligned_addr, alignment
                )));
            }

            // Check storage size
            let required_size = self.config.layout_data_bytes + base_offset;
            if storage.size() < required_size {
                return Err(LayoutError::InvalidConfig(format!(
                    "Layer {} storage too small: {} bytes available, {} bytes required",
                    layer_idx,
                    storage.size(),
                    required_size
                )));
            }
        }

        tracing::debug!(
            "LayerSeparate storage alignment verification passed for {} layers",
            self.storages.len()
        );
        Ok(())
    }
}

#[allow(missing_docs)]
#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};
    use crate::block_manager::storage::{StorageType, SystemAllocator};
    use dynamo_runtime::logging::init as init_logging;

    const NUM_BLOCKS: usize = 7;
    const NUM_LAYERS: usize = 5;
    const OUTER_DIM: usize = 2;
    const PAGE_SIZE: usize = 4;
    const INNER_DIM: usize = 13;
    const DTYPE_WIDTH_BYTES: usize = 4;

    /// Helper function to calculate expected memory offset
    fn calculate_expected_offset(
        base_addr: u64,
        block_idx: usize,
        layer_idx: usize,
        block_stride: usize,
        layer_stride: usize,
    ) -> u64 {
        base_addr + (block_idx * block_stride + layer_idx * layer_stride) as u64
    }

    // Updated setup_layout: Calculates size internally, uses default alignment for simplicity in non-alignment tests.
    pub fn setup_layout(
        alignment: Option<usize>, // Option to override default alignment
    ) -> Result<FullyContiguous<NullDeviceStorage>, LayoutError> {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: alignment.unwrap_or(1),
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        FullyContiguous::allocate(config, &NullDeviceAllocator)
    }

    #[test]
    fn test_fc_creation_invalid_alignment() {
        let config = LayoutConfig::builder()
            .num_blocks(NUM_BLOCKS)
            .num_layers(NUM_LAYERS)
            .outer_dim(OUTER_DIM)
            .page_size(PAGE_SIZE)
            .inner_dim(INNER_DIM)
            .alignment(3)
            .build()
            .unwrap();

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fc_creation_success() {
        // Setup with default (None) alignment
        let layout_result = setup_layout(None);
        assert!(
            layout_result.is_ok(),
            "Layout creation failed: {:?}",
            layout_result.err()
        );
    }

    #[test]
    fn test_fc_creation_insufficient_storage() {
        init_logging();
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };
        // Calculate correct size needed
        let fc_config = FullyContiguousConfig::new(config.clone()).unwrap();
        let required_size = fc_config.required_allocation_size();
        let storage = NullDeviceStorage::new((required_size - 1) as u64);
        let layout_result = FullyContiguous::new(config, vec![storage]);

        assert!(layout_result.is_err());
        match layout_result.err().unwrap() {
            LayoutError::InvalidConfig(_) => {} // Expected error
            e => panic!("Expected InvalidConfig error, got {:?}", e),
        }
    }

    #[test]
    fn test_fc_accessor_methods() {
        let layout = setup_layout(None).expect("Layout setup failed");

        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.outer_dim(), OUTER_DIM);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
    }

    #[test]
    fn test_fc_offset_calculation() {
        let layout = setup_layout(None).expect("Layout setup failed");

        let dims = layout.config.clone();
        let block_stride = dims.block_stride_in_bytes;
        let layer_stride = dims.layer_stride_in_bytes;
        let base_addr = layout.storage.addr() + layout.base_offset as u64;

        // Test first block, first layer
        let expected_offset_0_0 =
            calculate_expected_offset(base_addr, 0, 0, block_stride, layer_stride);
        assert_eq!(
            layout.memory_region(0, 0, 0).unwrap().addr as u64,
            expected_offset_0_0
        );

        // Test first block, last layer
        let last_layer_idx = NUM_LAYERS - 1;
        let expected_offset_0_last =
            calculate_expected_offset(base_addr, 0, last_layer_idx, block_stride, layer_stride);
        assert_eq!(
            layout.memory_region(0, last_layer_idx, 0).unwrap().addr as u64,
            expected_offset_0_last
        );

        // Test last block, first layer
        let last_block_idx = NUM_BLOCKS - 1;
        let expected_offset_last_0 =
            calculate_expected_offset(base_addr, last_block_idx, 0, block_stride, layer_stride);
        assert_eq!(
            layout.memory_region(last_block_idx, 0, 0).unwrap().addr as u64,
            expected_offset_last_0
        );

        // Test last block, last layer
        let expected_offset_last_last = calculate_expected_offset(
            base_addr,
            last_block_idx,
            last_layer_idx,
            block_stride,
            layer_stride,
        );
        assert_eq!(
            layout
                .memory_region(last_block_idx, last_layer_idx, 0)
                .unwrap()
                .addr as u64,
            expected_offset_last_last
        );

        // Test intermediate block/layer
        let mid_block_idx = NUM_BLOCKS / 2;
        let mid_layer_idx = NUM_LAYERS / 2;
        let expected_offset_mid_mid = calculate_expected_offset(
            base_addr,
            mid_block_idx,
            mid_layer_idx,
            block_stride,
            layer_stride,
        );
        assert_eq!(
            layout
                .memory_region(mid_block_idx, mid_layer_idx, 0)
                .unwrap()
                .addr as u64,
            expected_offset_mid_mid
        );
    }

    #[test]
    fn test_fc_invalid_block_index() {
        let layout = setup_layout(None).expect("Layout setup failed");
        let result = layout.memory_region(NUM_BLOCKS, 0, 0); // Index == num_blocks (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidBlockIndex(NUM_BLOCKS)
        ));
    }

    #[test]
    fn test_fc_invalid_layer_index() {
        let layout = setup_layout(None).expect("Layout setup failed");
        let result = layout.memory_region(0, NUM_LAYERS, 0); // Index == num_layers (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidLayerIndex(NUM_LAYERS)
        ));
    }

    #[test]
    fn test_fc_invalid_outer_index() {
        let layout = setup_layout(None).expect("Layout setup failed");
        let result = layout.memory_region(0, 0, OUTER_DIM); // Index == num_outer_dims (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidOuterIndex(OUTER_DIM)
        ));
    }

    #[test]
    fn test_fc_allocation_system() {
        init_logging();
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        let allocator = SystemAllocator;
        let layout_result = FullyContiguous::allocate(config, &allocator);

        assert!(layout_result.is_ok());
        let layout = layout_result.unwrap();

        // Basic checks on the allocated layout
        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
        assert_eq!(layout.storage.storage_type(), StorageType::System);
        assert_eq!(
            layout.storage.size(),
            layout.config.required_allocation_size()
        );

        assert_eq!(
            layout.storage.size(),
            NUM_BLOCKS * NUM_LAYERS * OUTER_DIM * PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES
        );
    }

    #[test]
    fn test_fc_alignment() {
        init_logging();
        const ALIGNMENT: usize = 256; // Must be power of 2

        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: ALIGNMENT,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        // Calculate expected size needed *for the data layout itself*
        let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;
        assert_eq!(memory_region_size, 208);

        let natural_block_stride = OUTER_DIM * NUM_LAYERS * memory_region_size;
        assert_eq!(natural_block_stride, 2080);

        let aligned_block_stride = align_up(natural_block_stride, ALIGNMENT);
        assert_eq!(aligned_block_stride, 2304);

        // Calculate the expected *allocated* size (data + initial padding)
        let fc_config = FullyContiguousConfig::new(config.clone()).unwrap();
        let expected_allocated_size = fc_config.required_allocation_size();

        // Use allocate method
        let allocator = SystemAllocator;
        let layout_result = FullyContiguous::allocate(config.clone(), &allocator);

        assert!(
            layout_result.is_ok(),
            "Allocation failed: {:?}",
            layout_result.err()
        );
        let layout = layout_result.unwrap();

        // Verify total *allocated* size matches expectation
        assert_eq!(
            layout.storage.size(),
            expected_allocated_size,
            "Allocated storage size mismatch"
        );
        assert_eq!(
            layout.config.block_stride_in_bytes, aligned_block_stride,
            "Stored block stride mismatch"
        );

        // Check alignment of block starts
        let addr_block_0 = layout
            .memory_region(0, 0, 0)
            .expect("Failed to get addr block 0");
        let addr_block_1 = layout
            .memory_region(1, 0, 0)
            .expect("Failed to get addr block 1");
        let addr_block_2 = layout
            .memory_region(2, 0, 0)
            .expect("Failed to get addr block 2");

        // All blocks should now be aligned due to base_offset adjustment
        assert_eq!(
            addr_block_0.addr as u64 % ALIGNMENT as u64,
            0,
            "Block 0 start address is not aligned"
        );
        assert_eq!(
            addr_block_1.addr as u64 % ALIGNMENT as u64,
            0,
            "Block 1 start address is not aligned"
        );
        assert_eq!(
            addr_block_2.addr as u64 % ALIGNMENT as u64,
            0,
            "Block 2 start address is not aligned"
        );

        // Verify the difference matches the aligned stride
        assert_eq!(
            addr_block_1.addr as u64 - addr_block_0.addr as u64,
            aligned_block_stride as u64,
            "Stride between block 0 and 1 mismatch"
        );
        assert_eq!(
            addr_block_2.addr as u64 - addr_block_1.addr as u64,
            aligned_block_stride as u64,
            "Stride between block 1 and 2 mismatch"
        );
    }

    // LayerSeparate Tests

    /// Helper function to setup LayerSeparate layout with specified configuration
    pub fn setup_layer_separate_layout(
        alignment: Option<usize>,
        is_outer_contiguous: bool,
    ) -> Result<LayerSeparate<NullDeviceStorage>, LayoutError> {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: alignment.unwrap_or(1),
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        // Create one storage per layer
        let ls_config = LayerSeparateConfig::new(config.clone(), is_outer_contiguous)?;
        let required_size = ls_config.required_allocation_size();
        let mut storages = Vec::new();
        for _ in 0..NUM_LAYERS {
            storages.push(NullDeviceStorage::new(required_size as u64));
        }

        LayerSeparate::new(config, storages, is_outer_contiguous)
    }

    #[test]
    fn test_ls_creation_success_outer_contiguous() {
        let layout_result = setup_layer_separate_layout(None, true);
        assert!(
            layout_result.is_ok(),
            "LayerSeparate creation failed: {:?}",
            layout_result.err()
        );

        let layout = layout_result.unwrap();
        assert_eq!(
            layout.layout_type(),
            LayoutType::LayerSeparate {
                outer_contiguous: true
            }
        );
    }

    #[test]
    fn test_ls_creation_success_block_contiguous() {
        let layout_result = setup_layer_separate_layout(None, false);
        assert!(
            layout_result.is_ok(),
            "LayerSeparate creation failed: {:?}",
            layout_result.err()
        );

        let layout = layout_result.unwrap();
        assert_eq!(
            layout.layout_type(),
            LayoutType::LayerSeparate {
                outer_contiguous: false
            }
        );
    }

    #[test]
    fn test_ls_creation_wrong_storage_count() {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        // Create wrong number of storages (should be NUM_LAYERS, but provide NUM_LAYERS - 1)
        let mut storages = Vec::new();
        for _ in 0..(NUM_LAYERS - 1) {
            storages.push(NullDeviceStorage::new(1000));
        }

        let layout_result = LayerSeparate::new(config, storages, true);
        assert!(layout_result.is_err());
        match layout_result.err().unwrap() {
            LayoutError::InvalidConfig(_) => {} // Expected error
            e => panic!("Expected InvalidConfig error, got {:?}", e),
        }
    }

    #[test]
    fn test_ls_accessor_methods() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.outer_dim(), OUTER_DIM);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
        assert_eq!(layout.storage().len(), NUM_LAYERS);
        assert_eq!(layout.storage_type(), &StorageType::Null);
    }

    #[test]
    fn test_ls_memory_region_outer_contiguous() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        // Test accessing different blocks within the same layer
        let region_0_0_0 = layout.memory_region(0, 0, 0).unwrap();
        let region_1_0_0 = layout.memory_region(1, 0, 0).unwrap();

        // In outer_contiguous mode, blocks are sequential within each layer
        let expected_block_stride = layout.config.block_stride_in_bytes;
        assert_eq!(
            region_1_0_0.addr - region_0_0_0.addr,
            expected_block_stride,
            "Block stride mismatch in outer_contiguous mode"
        );

        // Test accessing different outer dimensions
        let region_0_0_1 = layout.memory_region(0, 0, 1).unwrap();
        let expected_outer_stride = layout.config.outer_dim_stride_in_bytes;
        assert_eq!(
            region_0_0_1.addr - region_0_0_0.addr,
            expected_outer_stride,
            "Outer dimension stride mismatch"
        );

        // Test accessing different layers (should be in different storage)
        let region_0_1_0 = layout.memory_region(0, 1, 0).unwrap();
        let region_0_0_0_storage_addr = layout.storages[0].addr() as usize + layout.base_offsets[0];
        let region_0_1_0_storage_addr = layout.storages[1].addr() as usize + layout.base_offsets[1];

        assert_eq!(region_0_0_0.addr, region_0_0_0_storage_addr);
        assert_eq!(region_0_1_0.addr, region_0_1_0_storage_addr);
    }

    #[test]
    fn test_ls_memory_region_block_contiguous() {
        let layout = setup_layer_separate_layout(None, false).expect("Layout setup failed");

        // Test accessing different blocks within the same layer
        let region_0_0_0 = layout.memory_region(0, 0, 0).unwrap();
        let region_1_0_0 = layout.memory_region(1, 0, 0).unwrap();

        // In block_contiguous mode, blocks have different stride calculation
        let expected_block_stride = layout.config.block_stride_in_bytes;
        assert_eq!(
            region_1_0_0.addr - region_0_0_0.addr,
            expected_block_stride,
            "Block stride mismatch in block_contiguous mode"
        );

        // Test accessing different outer dimensions within same block
        let region_0_0_1 = layout.memory_region(0, 0, 1).unwrap();
        let expected_outer_stride = layout.config.outer_dim_stride_in_bytes;
        assert_eq!(
            region_0_0_1.addr - region_0_0_0.addr,
            expected_outer_stride,
            "Outer dimension stride mismatch in block_contiguous mode"
        );
    }

    #[test]
    fn test_ls_invalid_indices() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        // Test invalid block index
        let result = layout.memory_region(NUM_BLOCKS, 0, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidBlockIndex(NUM_BLOCKS)
        ));

        // Test invalid layer index
        let result = layout.memory_region(0, NUM_LAYERS, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidLayerIndex(NUM_LAYERS)
        ));

        // Test invalid outer index
        let result = layout.memory_region(0, 0, OUTER_DIM);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidOuterIndex(OUTER_DIM)
        ));
    }

    #[test]
    fn test_ls_memory_region_size() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        let region = layout.memory_region(0, 0, 0).unwrap();
        let expected_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

        assert_eq!(region.size, expected_size);
    }

    #[test]
    fn test_ls_all_blocks_layers_accessible() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        // Test that we can access all valid combinations of indices
        for block_idx in 0..NUM_BLOCKS {
            for layer_idx in 0..NUM_LAYERS {
                for outer_idx in 0..OUTER_DIM {
                    let result = layout.memory_region(block_idx, layer_idx, outer_idx);
                    assert!(
                        result.is_ok(),
                        "Failed to access block {}, layer {}, outer {}: {:?}",
                        block_idx,
                        layer_idx,
                        outer_idx,
                        result.err()
                    );
                }
            }
        }
    }

    #[test]
    fn test_ls_storage_mutability() {
        let mut layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        // Test that we can get mutable references to storage
        let mut_storages = layout.storage_mut();
        assert_eq!(mut_storages.len(), NUM_LAYERS);

        // Verify each storage is accessible
        for (i, storage) in mut_storages.iter().enumerate() {
            assert!(storage.size() > 0, "Storage {} has zero size", i);
        }
    }

    #[test]
    fn test_ls_alignment() {
        init_logging();
        const ALIGNMENT: usize = 128; // Must be power of 2

        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: ALIGNMENT,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        // Create storages with sufficient size
        let ls_config = LayerSeparateConfig::new(config.clone(), true).unwrap();
        let required_size = ls_config.required_allocation_size();
        let mut storages = Vec::new();
        for _ in 0..NUM_LAYERS {
            storages.push(NullDeviceStorage::new(required_size as u64));
        }

        let layout_result = LayerSeparate::new(config, storages, true);
        assert!(
            layout_result.is_ok(),
            "Layout creation with alignment failed"
        );

        let layout = layout_result.unwrap();

        // Check that block addresses are properly aligned within each layer
        for layer_idx in 0..NUM_LAYERS {
            let addr_block_0 = layout.memory_region(0, layer_idx, 0).unwrap();
            let addr_block_1 = layout.memory_region(1, layer_idx, 0).unwrap();

            // First block should be aligned
            assert_eq!(
                addr_block_0.addr % ALIGNMENT,
                0,
                "Block 0 in layer {} is not aligned",
                layer_idx
            );

            // Subsequent blocks should maintain alignment
            assert_eq!(
                addr_block_1.addr % ALIGNMENT,
                0,
                "Block 1 in layer {} is not aligned",
                layer_idx
            );
        }
    }

    #[test]
    fn test_ls_stride_calculations_outer_contiguous() {
        let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

        let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

        // In outer_contiguous mode:
        // outer_dim_stride = block_stride * num_blocks
        // block_stride = memory_region_size (aligned)
        assert_eq!(layout.config.memory_region_size, memory_region_size);
        assert_eq!(layout.config.block_stride_in_bytes, memory_region_size); // No alignment needed
        assert_eq!(
            layout.config.outer_dim_stride_in_bytes,
            layout.config.block_stride_in_bytes * NUM_BLOCKS
        );
    }

    #[test]
    fn test_ls_stride_calculations_block_contiguous() {
        let layout = setup_layer_separate_layout(None, false).expect("Layout setup failed");

        let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

        // In block_contiguous mode:
        // outer_dim_stride = memory_region_size
        // block_stride = outer_dim_stride * outer_dim (aligned)
        assert_eq!(layout.config.memory_region_size, memory_region_size);
        assert_eq!(layout.config.outer_dim_stride_in_bytes, memory_region_size);
        assert_eq!(
            layout.config.block_stride_in_bytes,
            memory_region_size * OUTER_DIM
        );
    }

    #[test]
    fn test_ls_layout_data_bytes() {
        let layout_outer = setup_layer_separate_layout(None, true).expect("Layout setup failed");
        let layout_block = setup_layer_separate_layout(None, false).expect("Layout setup failed");

        // For outer_contiguous: layout_data_bytes = outer_dim_stride * outer_dim
        let expected_outer = layout_outer.config.outer_dim_stride_in_bytes * OUTER_DIM;
        assert_eq!(layout_outer.layout_data_bytes(), expected_outer);

        // For block_contiguous: layout_data_bytes = block_stride * num_blocks
        let expected_block = layout_block.config.block_stride_in_bytes * NUM_BLOCKS;
        assert_eq!(layout_block.layout_data_bytes(), expected_block);
    }

    // ============================================================================
    // COMPREHENSIVE LAYOUT CORRECTNESS TESTS
    // ============================================================================

    /// Test suite for layout correctness across different configurations
    mod layout_correctness_tests {
        use super::*;
        use std::collections::HashSet;

        /// Verify that memory regions don't overlap within the same layout
        #[test]
        fn test_fc_memory_regions_no_overlap() {
            let layout = setup_layout(None).expect("Layout setup failed");
            let mut used_ranges = Vec::new();

            // Collect all memory regions
            for block_idx in 0..NUM_BLOCKS {
                for layer_idx in 0..NUM_LAYERS {
                    for outer_idx in 0..OUTER_DIM {
                        let region = layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        used_ranges.push((region.addr, region.addr + region.size));
                    }
                }
            }

            // Check for overlaps
            for i in 0..used_ranges.len() {
                for j in (i + 1)..used_ranges.len() {
                    let (start_i, end_i) = used_ranges[i];
                    let (start_j, end_j) = used_ranges[j];

                    let overlaps = !(end_i <= start_j || end_j <= start_i);
                    assert!(
                        !overlaps,
                        "Memory regions overlap: [{}, {}) and [{}, {})",
                        start_i, end_i, start_j, end_j
                    );
                }
            }
        }

        #[test]
        fn test_ls_memory_regions_no_overlap() {
            let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

            // For each layer, collect memory regions and check for overlaps
            for layer_idx in 0..NUM_LAYERS {
                let mut used_ranges = Vec::new();

                for block_idx in 0..NUM_BLOCKS {
                    for outer_idx in 0..OUTER_DIM {
                        let region = layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        used_ranges.push((region.addr, region.addr + region.size));
                    }
                }

                // Check for overlaps within this layer
                for i in 0..used_ranges.len() {
                    for j in (i + 1)..used_ranges.len() {
                        let (start_i, end_i) = used_ranges[i];
                        let (start_j, end_j) = used_ranges[j];

                        let overlaps = !(end_i <= start_j || end_j <= start_i);
                        assert!(
                            !overlaps,
                            "Memory regions overlap in layer {}: [{}, {}) and [{}, {})",
                            layer_idx, start_i, end_i, start_j, end_j
                        );
                    }
                }
            }
        }

        /// Test that memory regions are properly aligned
        #[test]
        fn test_fc_memory_alignment_correctness() {
            const ALIGNMENT: usize = 256;
            let config = LayoutConfig {
                num_blocks: NUM_BLOCKS,
                num_layers: NUM_LAYERS,
                outer_dim: OUTER_DIM,
                page_size: PAGE_SIZE,
                inner_dim: INNER_DIM,
                alignment: ALIGNMENT,
                dtype_width_bytes: DTYPE_WIDTH_BYTES,
            };

            let layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();

            // Test all block starting addresses are aligned
            for block_idx in 0..NUM_BLOCKS {
                let region = layout.memory_region(block_idx, 0, 0).unwrap();
                assert_eq!(
                    region.addr % ALIGNMENT,
                    0,
                    "Block {} is not aligned to {} bytes",
                    block_idx,
                    ALIGNMENT
                );
            }
        }

        /// Test data integrity patterns across layout types
        #[test]
        fn test_layout_data_integrity_patterns() {
            init_logging();

            // Test pattern: write unique values to each memory region and verify they don't interfere
            let fc_layout = setup_layout(None).expect("FC Layout setup failed");
            let ls_layout =
                setup_layer_separate_layout(None, true).expect("LS Layout setup failed");

            // For FullyContiguous layout
            test_data_integrity_for_layout(&fc_layout, "FullyContiguous");

            // For LayerSeparate layout
            test_data_integrity_for_layout(&ls_layout, "LayerSeparate");
        }

        fn test_data_integrity_for_layout<L: GenericBlockLayout>(layout: &L, layout_name: &str) {
            let mut written_patterns = HashSet::new();

            // Write unique patterns to each memory region
            for block_idx in 0..layout.num_blocks() {
                for layer_idx in 0..layout.num_layers() {
                    for outer_idx in 0..layout.outer_dim() {
                        let region = layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        // Create unique pattern for this location
                        let pattern = (block_idx << 16) | (layer_idx << 8) | outer_idx;

                        // Verify we haven't used this pattern before
                        assert!(
                            !written_patterns.contains(&pattern),
                            "Duplicate pattern {} in {} layout",
                            pattern,
                            layout_name
                        );
                        written_patterns.insert(pattern);

                        // Verify the region has expected size
                        let expected_size = layout.page_size()
                            * layout.inner_dim()
                            * layout.layout_config().dtype_width_bytes;
                        assert_eq!(
                            region.size, expected_size,
                            "Region size mismatch in {} layout at ({}, {}, {})",
                            layout_name, block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        /// Test stride calculations across different layout types
        #[test]
        fn test_layout_stride_correctness() {
            let fc_layout = setup_layout(None).expect("FC Layout setup failed");
            let ls_outer = setup_layer_separate_layout(None, true).expect("LS outer setup failed");
            let ls_block = setup_layer_separate_layout(None, false).expect("LS block setup failed");

            // Test FullyContiguous strides
            test_fc_stride_correctness(&fc_layout);

            // Test LayerSeparate strides
            test_ls_stride_correctness(&ls_outer, true);
            test_ls_stride_correctness(&ls_block, false);
        }

        fn test_fc_stride_correctness(layout: &FullyContiguous<NullDeviceStorage>) {
            let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

            // Test layer stride
            let region_0_0_0 = layout.memory_region(0, 0, 0).unwrap();
            let region_0_1_0 = layout.memory_region(0, 1, 0).unwrap();
            let layer_stride = region_0_1_0.addr - region_0_0_0.addr;
            assert_eq!(layer_stride, memory_region_size * OUTER_DIM);

            // Test outer dimension stride
            let region_0_0_1 = layout.memory_region(0, 0, 1).unwrap();
            let outer_stride = region_0_0_1.addr - region_0_0_0.addr;
            assert_eq!(outer_stride, memory_region_size);

            // Test block stride
            let region_1_0_0 = layout.memory_region(1, 0, 0).unwrap();
            let block_stride = region_1_0_0.addr - region_0_0_0.addr;
            assert_eq!(block_stride, memory_region_size * OUTER_DIM * NUM_LAYERS);
        }

        fn test_ls_stride_correctness(
            layout: &LayerSeparate<NullDeviceStorage>,
            is_outer_contiguous: bool,
        ) {
            let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE_WIDTH_BYTES;

            // Test strides within the same layer
            let region_0_0_0 = layout.memory_region(0, 0, 0).unwrap();
            let region_1_0_0 = layout.memory_region(1, 0, 0).unwrap();
            let region_0_0_1 = layout.memory_region(0, 0, 1).unwrap();

            let block_stride = region_1_0_0.addr - region_0_0_0.addr;
            let outer_stride = region_0_0_1.addr - region_0_0_0.addr;

            if is_outer_contiguous {
                // In outer_contiguous mode: [outer_dim, n_blocks, ...]
                assert_eq!(block_stride, memory_region_size);
                assert_eq!(outer_stride, memory_region_size * NUM_BLOCKS);
            } else {
                // In block_contiguous mode: [n_blocks, outer_dim, ...]
                assert_eq!(block_stride, memory_region_size * OUTER_DIM);
                assert_eq!(outer_stride, memory_region_size);
            }
        }

        /// Test layout compatibility for mixed scenarios
        #[test]
        fn test_layout_compatibility_scenarios() {
            init_logging();

            // Scenario: FullyContiguous host buffer with LayerSeparate device
            let host_fc = setup_layout(None).expect("Host FC setup failed");
            let device_ls =
                setup_layer_separate_layout(None, true).expect("Device LS setup failed");

            // Verify they have compatible dimensions
            assert_eq!(host_fc.num_blocks(), device_ls.num_blocks());
            assert_eq!(host_fc.num_layers(), device_ls.num_layers());
            assert_eq!(host_fc.outer_dim(), device_ls.outer_dim());
            assert_eq!(host_fc.page_size(), device_ls.page_size());
            assert_eq!(host_fc.inner_dim(), device_ls.inner_dim());

            // Verify memory region sizes are compatible
            for block_idx in 0..host_fc.num_blocks() {
                for layer_idx in 0..host_fc.num_layers() {
                    for outer_idx in 0..host_fc.outer_dim() {
                        let host_region = host_fc
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let device_region = device_ls
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        assert_eq!(
                            host_region.size, device_region.size,
                            "Memory region size mismatch at ({}, {}, {})",
                            block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        /// Test edge cases and boundary conditions
        #[test]
        fn test_layout_edge_cases() {
            // Test with minimal configuration
            let minimal_config = LayoutConfig {
                num_blocks: 1,
                num_layers: 1,
                outer_dim: 1,
                page_size: 1,
                inner_dim: 1,
                alignment: 1,
                dtype_width_bytes: 1,
            };

            let minimal_fc =
                FullyContiguous::allocate(minimal_config.clone(), &SystemAllocator).unwrap();
            let region = minimal_fc.memory_region(0, 0, 0).unwrap();
            assert_eq!(region.size, 1);

            // Test with maximum supported outer_dim
            let max_outer_config = LayoutConfig {
                num_blocks: 2,
                num_layers: 2,
                outer_dim: 2, // Maximum supported
                page_size: 4,
                inner_dim: 4,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let max_outer_fc =
                FullyContiguous::allocate(max_outer_config, &SystemAllocator).unwrap();

            // Verify all combinations are accessible
            for block_idx in 0..2 {
                for layer_idx in 0..2 {
                    for outer_idx in 0..2 {
                        let region = max_outer_fc.memory_region(block_idx, layer_idx, outer_idx);
                        assert!(
                            region.is_ok(),
                            "Failed to access region ({}, {}, {})",
                            block_idx,
                            layer_idx,
                            outer_idx
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ls_allocate() {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        };

        LayerSeparate::allocate(config, &NullDeviceAllocator, true)
            .expect("Layout allocation failed");
    }

    // ============================================================================
    // MEMORY REGION VERIFICATION TESTS
    // ============================================================================

    mod memory_region_verification_tests {
        use super::*;

        #[test]
        fn test_fc_memory_region_verification() {
            let layout = setup_layout(None).expect("Layout setup failed");

            // Test overall verification
            assert!(
                layout.verify_memory_regions().is_ok(),
                "Memory region verification should pass"
            );

            // Test individual region verification
            for block_idx in 0..NUM_BLOCKS {
                for layer_idx in 0..NUM_LAYERS {
                    for outer_idx in 0..OUTER_DIM {
                        let matches = layout
                            .verify_memory_region(block_idx, layer_idx, outer_idx)
                            .expect("Memory region verification failed");
                        assert!(
                            matches,
                            "Memory region ({}, {}, {}) should match expected calculations",
                            block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        #[test]
        fn test_fc_expected_address_calculation() {
            let layout = setup_layout(None).expect("Layout setup failed");

            // Test that expected addresses match actual addresses
            for block_idx in 0..NUM_BLOCKS {
                for layer_idx in 0..NUM_LAYERS {
                    for outer_idx in 0..OUTER_DIM {
                        let actual_region = layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let expected_addr = layout
                            .expected_memory_address(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        assert_eq!(
                            actual_region.addr, expected_addr,
                            "Address mismatch at ({}, {}, {})",
                            block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        #[test]
        fn test_ls_memory_region_verification() {
            let layout = setup_layer_separate_layout(None, true).expect("Layout setup failed");

            // Test overall verification
            assert!(
                layout.verify_memory_regions().is_ok(),
                "LayerSeparate memory region verification should pass"
            );

            // Test storage alignment verification
            assert!(
                layout.verify_storage_alignment().is_ok(),
                "LayerSeparate storage alignment verification should pass"
            );

            // Test individual region verification
            for block_idx in 0..NUM_BLOCKS {
                for layer_idx in 0..NUM_LAYERS {
                    for outer_idx in 0..OUTER_DIM {
                        let matches = layout
                            .verify_memory_region(block_idx, layer_idx, outer_idx)
                            .expect("Memory region verification failed");
                        assert!(
                            matches,
                            "LayerSeparate memory region ({}, {}, {}) should match expected calculations",
                            block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        #[test]
        fn test_ls_expected_address_calculation() {
            let layout = setup_layer_separate_layout(None, false).expect("Layout setup failed");

            // Test that expected addresses match actual addresses for block-contiguous layout
            for block_idx in 0..NUM_BLOCKS {
                for layer_idx in 0..NUM_LAYERS {
                    for outer_idx in 0..OUTER_DIM {
                        let actual_region = layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let expected_addr = layout
                            .expected_memory_address(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        assert_eq!(
                            actual_region.addr, expected_addr,
                            "LayerSeparate address mismatch at ({}, {}, {})",
                            block_idx, layer_idx, outer_idx
                        );
                    }
                }
            }
        }

        #[test]
        fn test_memory_region_verification_with_alignment() {
            const ALIGNMENT: usize = 512;
            let config = LayoutConfig {
                num_blocks: NUM_BLOCKS,
                num_layers: NUM_LAYERS,
                outer_dim: OUTER_DIM,
                page_size: PAGE_SIZE,
                inner_dim: INNER_DIM,
                alignment: ALIGNMENT,
                dtype_width_bytes: DTYPE_WIDTH_BYTES,
            };

            let fc_layout = FullyContiguous::allocate(config.clone(), &SystemAllocator).unwrap();
            let ls_layout = LayerSeparate::allocate(config, &NullDeviceAllocator, true).unwrap();

            // Both layouts should pass verification with alignment
            assert!(
                fc_layout.verify_memory_regions().is_ok(),
                "FullyContiguous with alignment should pass verification"
            );

            assert!(
                ls_layout.verify_memory_regions().is_ok(),
                "LayerSeparate with alignment should pass verification"
            );

            assert!(
                ls_layout.verify_storage_alignment().is_ok(),
                "LayerSeparate storage alignment should pass verification"
            );
        }

        #[test]
        fn test_cross_layout_address_compatibility() {
            let config = LayoutConfig {
                num_blocks: 2,
                num_layers: 2,
                outer_dim: 1,
                page_size: 8,
                inner_dim: 16,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let fc_layout = FullyContiguous::allocate(config.clone(), &SystemAllocator).unwrap();
            let ls_layout = LayerSeparate::allocate(config, &NullDeviceAllocator, true).unwrap();

            // Both layouts should have compatible memory region sizes
            for block_idx in 0..2 {
                for layer_idx in 0..2 {
                    let fc_region = fc_layout.memory_region(block_idx, layer_idx, 0).unwrap();
                    let ls_region = ls_layout.memory_region(block_idx, layer_idx, 0).unwrap();

                    assert_eq!(
                        fc_region.size, ls_region.size,
                        "Memory region sizes should be compatible between layouts at ({}, {})",
                        block_idx, layer_idx
                    );
                }
            }
        }

        #[test]
        fn test_memory_region_bounds_checking() {
            let layout = setup_layout(None).expect("Layout setup failed");

            // Test invalid indices
            assert!(
                layout.verify_memory_region(NUM_BLOCKS, 0, 0).is_err(),
                "Should fail for invalid block index"
            );

            assert!(
                layout.verify_memory_region(0, NUM_LAYERS, 0).is_err(),
                "Should fail for invalid layer index"
            );

            assert!(
                layout.verify_memory_region(0, 0, OUTER_DIM).is_err(),
                "Should fail for invalid outer index"
            );
        }
    }
}
