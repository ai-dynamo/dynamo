// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage layout implementation.
//!
//! This layout is designed for object storage where each "block" in the layout
//! corresponds to one complete Object. This enables efficient batch operations
//! where multiple Objects can be read/written in a single NIXL transfer call.
//!
//! ## Architecture
//!
//! ```text
//! Block 0 → object key_0 (complete KV cache with tp_world_size partitions)
//! Block 1 → object key_1 (complete KV cache with tp_world_size partitions)
//! Block 2 → object key_2 (complete KV cache with tp_world_size partitions)
//! ...
//! ```
//!
//! ## Use Cases
//!
//! 1. **Batch onboarding**: Read multiple objects in one NIXL call
//! 2. **Batch offloading**: Write multiple objects in one NIXL call
//!
//! ## Example
//!
//! ```ignore
//! // Batch: 10 Objects
//! let layout = PhysicalLayout::builder(agent)
//!     .with_config(config)  // num_blocks = 10
//!     .object_layout()
//!     .allocate_objects("bucket", vec![key1, key2, ..., key10])?
//!     .build()?;
//!
//! // Single: 1 Object
//! let layout = PhysicalLayout::builder(agent)
//!     .with_config(config)  // num_blocks = 1
//!     .object_layout()
//!     .allocate_object("bucket", key)
//!     .build()?;
//! ```

use anyhow::{Result, anyhow};
use std::sync::Arc;
use validator::Validate;

use super::serialize::{LayoutTypeDetails, ObjectLayoutDetails};
use super::{Layout, LayoutConfig};
use dynamo_memory::{Buffer, MemoryDescriptor, MemoryRegion, StorageKind};

/// Object storage layout where each block is a separate Object.
///
/// This layout is optimized for object storage operations where:
/// - Each block_id maps directly to one Object (memory region)
/// - All regions are independent (different object keys)
/// - Batch operations read/write multiple objects in parallel via NIXL
#[derive(Debug)]
pub struct ObjectLayout {
    config: LayoutConfig,
    /// Memory regions, one per block (one per Object)
    regions: Vec<Buffer>,
    /// Total size of each complete object in bytes
    object_size: usize,
}

impl ObjectLayout {
    /// Create a new object storage layout.
    ///
    /// # Arguments
    /// * `config` - Layout configuration (num_blocks = number of Objects)
    /// * `regions` - Memory regions, one per Object (must match num_blocks)
    ///
    /// # Returns
    /// A new ObjectLayout instance
    ///
    /// # Errors
    /// Returns error if:
    /// - Number of regions doesn't match num_blocks
    /// - Any region is too small for the configured object size
    pub fn new(config: LayoutConfig, regions: Vec<Buffer>) -> Result<Self> {
        config.validate()?;

        if regions.len() != config.num_blocks {
            return Err(anyhow!(
                "Number of memory regions ({}) must match num_blocks ({})",
                regions.len(),
                config.num_blocks
            ));
        }

        // Calculate required size for each complete Object
        // Size = num_layers × outer_dim × page_size × inner_dim × dtype_bytes
        // Note: In object layout, each "block" is a complete Object containing
        // all layers, so we don't multiply by num_blocks here
        let object_size = config.num_layers
            * config.outer_dim
            * config.page_size
            * config.inner_dim
            * config.dtype_width_bytes;

        // Validate each region is large enough
        for (i, region) in regions.iter().enumerate() {
            if region.size() < object_size {
                return Err(anyhow!(
                    "Memory region {} too small for object. Required: {} bytes, got: {} bytes",
                    i,
                    object_size,
                    region.size()
                ));
            }
        }

        Ok(Self {
            config,
            regions,
            object_size,
        })
    }

    /// Get the size of each complete object in bytes.
    pub fn object_size(&self) -> usize {
        self.object_size
    }

    /// Get the number of objects in this layout.
    pub fn num_objects(&self) -> usize {
        self.regions.len()
    }
}

impl Layout for ObjectLayout {
    fn config(&self) -> &LayoutConfig {
        &self.config
    }

    fn memory_regions(&self) -> &[Buffer] {
        &self.regions
    }

    fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryRegion> {
        // In ObjectLayout, block_id directly maps to the Object (memory region)
        // Within that object, we compute the offset for the specific layer/outer_id

        if block_id >= self.regions.len() {
            return Err(anyhow!(
                "Block ID {} out of range (num_objects: {})",
                block_id,
                self.regions.len()
            ));
        }

        if layer_id >= self.config.num_layers {
            return Err(anyhow!(
                "Layer ID {} out of range (num_layers: {})",
                layer_id,
                self.config.num_layers
            ));
        }

        if outer_id >= self.config.outer_dim {
            return Err(anyhow!(
                "Outer ID {} out of range (outer_dim: {})",
                outer_id,
                self.config.outer_dim
            ));
        }

        let current_region = &self.regions[block_id];

        // For object storage, NIXL uses the addr field as the offset within the Object
        // for byte-ranged GET requests (see nixl obj_backend.cpp line 245, 268)
        // The device_id (object key) identifies WHICH object, addr is the offset WITHIN it.

        // ObjectLayout supports two use cases:
        // 1. Multiple separate Objects: each block → different object key, offset = 0
        // 2. TP partitions in ONE Object: all blocks → same object key, sequential offsets
        //
        // To determine which case: check if this block's key matches earlier blocks.
        // If we find an earlier block with the same key, compute offset relative to the first.

        // Stride calculations
        let region_size = self.config.page_size * self.config.inner_dim * self.config.dtype_width_bytes;
        let outer_stride = region_size;
        let layer_stride = outer_stride * self.config.outer_dim;
        let block_stride = self.object_size;  // Full size of one complete block/partition

        // Find the "base block" - the first block that shares this object key
        let current_key = current_region.storage_kind();
        let base_block_offset = self.regions.iter()
            .position(|r| r.storage_kind() == current_key)
            .unwrap_or(block_id);  // Fallback to block_id if not found

        // Calculate offset within the Object:
        // - If this is the first block with this key: offset = 0 + layer/outer
        // - If this is a later partition: offset = (block_id - base) * stride + layer/outer
        let blocks_from_base = block_id - base_block_offset;
        let offset = blocks_from_base * block_stride + layer_id * layer_stride + outer_id * outer_stride;

        Ok(MemoryRegion {
            addr: offset,  // Offset within Object (for byte-ranged GET)
            size: region_size,
        })
    }

    fn required_allocations(&self) -> Vec<usize> {
        // Each object requires object_size bytes
        vec![self.object_size; self.config.num_blocks]
    }

    fn is_fully_contiguous(&self) -> bool {
        // ObjectLayout is NOT fully contiguous - each object is separate
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
        LayoutTypeDetails::ObjectLayout(ObjectLayoutDetails {
            num_objects: self.regions.len(),
            object_size: self.object_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_memory::{ObjectStorage, create_buffer};

    #[test]
    fn test_object_layout_single() {
        let config = LayoutConfig::builder()
            .num_blocks(1)
            .num_layers(32)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let object_size = 32 * 2 * 16 * 64 * 2;
        let region = create_buffer(ObjectStorage::new("test-bucket", 1234567890u64, object_size).unwrap());

        let layout = ObjectLayout::new(config, vec![region]).unwrap();

        assert_eq!(layout.num_objects(), 1);
        assert_eq!(layout.object_size(), object_size);
        assert!(!layout.is_fully_contiguous());
    }

    #[test]
    fn test_object_layout_batch() {
        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(32)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let object_size = 32 * 2 * 16 * 64 * 2;
        let regions: Vec<Buffer> = (0..10)
            .map(|i| {
                create_buffer(ObjectStorage::new("test-bucket", 1000000000u64 + i, object_size).unwrap())
            })
            .collect();

        let layout = ObjectLayout::new(config, regions).unwrap();

        assert_eq!(layout.num_objects(), 10);
        assert_eq!(layout.object_size(), object_size);
        assert_eq!(layout.num_blocks(), 10);
    }

    #[test]
    fn test_object_layout_memory_descriptor() {
        let config = LayoutConfig::builder()
            .num_blocks(3)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let object_size = 2 * 2 * 4 * 8 * 2; // 256 bytes per object
        let regions: Vec<Buffer> = (0..3)
            .map(|i| {
                create_buffer(ObjectStorage::new("test-bucket", 2000000000u64 + i, object_size).unwrap())
            })
            .collect();

        let layout = ObjectLayout::new(config, regions).unwrap();

        // Test memory_region for block 0, layer 0, outer 0
        let desc = layout.memory_region(0, 0, 0).unwrap();
        assert_eq!(desc.size, 4 * 8 * 2); // page_size * inner_dim * dtype_bytes

        // Test memory_region for block 1 (different Object)
        let desc1 = layout.memory_region(1, 0, 0).unwrap();
        let desc0 = layout.memory_region(0, 0, 0).unwrap();
        // For different objects in ObjectLayout, offsets should both be 0
        // (since each block maps to a different object key)
        assert_eq!(desc0.addr, 0);
        assert_eq!(desc1.addr, 0);
    }
}
