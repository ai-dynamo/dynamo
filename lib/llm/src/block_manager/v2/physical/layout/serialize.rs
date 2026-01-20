// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization types for physical layouts.
//!
//! This module provides types for serializing and deserializing physical layouts
//! so they can be transmitted to remote nodes and reconstructed there for RDMA operations.

use super::physical::NixlMetadata;
use super::{BlockDimension, LayoutConfig};
use crate::block_manager::v2::memory::{MemoryDescriptor, StorageKind};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Format of blocks in a fully contiguous layout.
///
/// This enum describes how the blocks are organized and formatted in memory.
/// Currently only `Operational` is supported, but future variants may include
/// different compression schemes or memory layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockFormat {
    /// Standard operational format - blocks are stored in their normal, uncompressed form.
    Operational,
}

impl Default for BlockFormat {
    fn default() -> Self {
        Self::Operational
    }
}

/// Details specific to fully contiguous layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullyContiguousDetails {
    /// Format of the blocks in memory
    pub block_format: BlockFormat,
}

/// Details specific to layer-separate layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSeparateDetails {
    /// Block dimension ordering (block-first or block-second)
    pub block_dim: BlockDimension,
}

/// Layout-type-specific details.
///
/// This enum captures the information that differs between layout types
/// and is needed to reconstruct the layout on a remote node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutTypeDetails {
    /// Fully contiguous layout details
    FullyContiguous(FullyContiguousDetails),
    /// Layer-separate layout details
    LayerSeparate(LayerSeparateDetails),
}

/// Serializable representation of a physical layout.
///
/// This structure contains all information needed to reconstruct a layout
/// on a remote node, including:
/// - Layout configuration (dimensions, sizes, etc.)
/// - Storage location and NIXL metadata
/// - Memory descriptors for all regions
/// - Layout-type-specific details
///
/// The serialized form can be transmitted over the network and used to
/// build NIXL transfer descriptors for remote memory access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDescriptor {
    /// Serialization format version (for future compatibility)
    pub version: u32,

    /// Layout configuration
    pub layout_config: LayoutConfig,

    /// Storage location
    pub location: StorageKind,

    /// NIXL metadata from the source node
    pub nixl_metadata: NixlMetadata,

    /// Memory descriptors for all regions backing this layout
    pub memory_descriptors: Vec<MemoryDescriptor>,

    /// Layout-type-specific details
    pub layout_type_details: LayoutTypeDetails,
}

impl LayoutDescriptor {
    /// Current serialization version
    pub const CURRENT_VERSION: u32 = 1;

    /// Serialize this layout to a JSON string.
    ///
    /// # Returns
    /// JSON string representation of the layout
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| anyhow::anyhow!("failed to serialize layout to JSON: {}", e))
    }

    /// Serialize this layout to JSON bytes.
    ///
    /// # Returns
    /// UTF-8 encoded JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| anyhow::anyhow!("failed to serialize layout to JSON bytes: {}", e))
    }

    /// Deserialize a layout from a JSON string.
    ///
    /// # Arguments
    /// * `json` - JSON string representation
    ///
    /// # Returns
    /// Deserialized layout
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("failed to deserialize layout from JSON: {}", e))
    }

    /// Deserialize a layout from JSON bytes.
    ///
    /// # Arguments
    /// * `bytes` - UTF-8 encoded JSON bytes
    ///
    /// # Returns
    /// Deserialized layout
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| anyhow::anyhow!("failed to deserialize layout from JSON bytes: {}", e))
    }

    /// Get the layout configuration.
    pub fn layout_config(&self) -> &LayoutConfig {
        &self.layout_config
    }

    /// Get the storage location.
    pub fn location(&self) -> StorageKind {
        self.location
    }

    /// Get the NIXL metadata from the source node.
    pub fn nixl_metadata(&self) -> &NixlMetadata {
        &self.nixl_metadata
    }

    /// Get the memory descriptors.
    pub fn memory_descriptors(&self) -> &[MemoryDescriptor] {
        &self.memory_descriptors
    }

    /// Get the layout type details.
    pub fn layout_type_details(&self) -> &LayoutTypeDetails {
        &self.layout_type_details
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    #[test]
    fn test_block_format_default() {
        assert_eq!(BlockFormat::default(), BlockFormat::Operational);
    }

    #[test]
    fn test_serialized_layout_json_roundtrip() {
        let layout = LayoutDescriptor {
            version: LayoutDescriptor::CURRENT_VERSION,
            layout_config: make_test_config(),
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test_agent".to_string(), nixl_sys::MemType::Dram, 0),
            memory_descriptors: vec![MemoryDescriptor::new(0x1000, 4096)],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
            }),
        };

        // Test to_json/from_json
        let json = layout.to_json().unwrap();
        let deserialized = LayoutDescriptor::from_json(&json).unwrap();

        assert_eq!(deserialized.version, layout.version);
        assert_eq!(deserialized.layout_config, layout.layout_config);
        assert_eq!(deserialized.location, layout.location);
        assert_eq!(
            deserialized.nixl_metadata.agent_name(),
            layout.nixl_metadata.agent_name()
        );
        assert_eq!(deserialized.memory_descriptors.len(), 1);
    }

    #[test]
    fn test_serialized_layout_json_bytes_roundtrip() {
        let layout = LayoutDescriptor {
            version: LayoutDescriptor::CURRENT_VERSION,
            layout_config: make_test_config(),
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test_agent".to_string(), nixl_sys::MemType::Vram, 5),
            memory_descriptors: vec![
                MemoryDescriptor::new(0x1000, 2048),
                MemoryDescriptor::new(0x2000, 2048),
            ],
            layout_type_details: LayoutTypeDetails::LayerSeparate(LayerSeparateDetails {
                block_dim: BlockDimension::BlockIsFirstDim,
            }),
        };

        // Test to_json_bytes/from_json_bytes
        let bytes = layout.to_json_bytes().unwrap();
        let deserialized = LayoutDescriptor::from_json_bytes(&bytes).unwrap();

        assert_eq!(deserialized.version, layout.version);
        assert_eq!(deserialized.nixl_metadata.device_id(), 5);
        assert_eq!(deserialized.memory_descriptors.len(), 2);
    }

    #[test]
    fn test_fully_contiguous_details_serialization() {
        let details = LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
            block_format: BlockFormat::Operational,
        });

        let json = serde_json::to_string(&details).unwrap();
        let deserialized: LayoutTypeDetails = serde_json::from_str(&json).unwrap();

        match deserialized {
            LayoutTypeDetails::FullyContiguous(d) => {
                assert_eq!(d.block_format, BlockFormat::Operational);
            }
            _ => panic!("Expected FullyContiguous variant"),
        }
    }

    #[test]
    fn test_layer_separate_details_serialization() {
        let details = LayoutTypeDetails::LayerSeparate(LayerSeparateDetails {
            block_dim: BlockDimension::BlockIsSecondDim,
        });

        let json = serde_json::to_string(&details).unwrap();
        let deserialized: LayoutTypeDetails = serde_json::from_str(&json).unwrap();

        match deserialized {
            LayoutTypeDetails::LayerSeparate(d) => {
                assert_eq!(d.block_dim, BlockDimension::BlockIsSecondDim);
            }
            _ => panic!("Expected LayerSeparate variant"),
        }
    }
}
