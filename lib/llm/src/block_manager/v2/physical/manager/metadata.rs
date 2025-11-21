// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization types for exporting/importing layout metadata with NIXL integration.

use super::handle::LayoutHandle;
use crate::block_manager::v2::physical::layout::LayoutDescriptor;
use anyhow::Result;
use bincode::{Decode, Encode};
use bytes::Bytes;

/// Worker identification combining worker_id and NIXL agent name.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub struct WorkerAddress {
    /// Unique identifier for this worker
    pub worker_id: u64,
    /// NIXL agent name on this worker
    pub nixl_agent_name: String,
}

impl WorkerAddress {
    /// Create a new worker address.
    pub fn new(worker_id: u64, nixl_agent_name: String) -> Self {
        Self {
            worker_id,
            nixl_agent_name,
        }
    }
}

/// Local layout descriptor with its assigned handle from the TransportManager.
#[derive(Debug, Clone, Encode, Decode)]
pub struct LocalLayoutDescriptor {
    /// Unique handle for this layout
    pub handle: LayoutHandle,
    /// Serialized layout data (uses Serde, bridged via bincode)
    #[bincode(with_serde)]
    pub layout: LayoutDescriptor,
}

impl LocalLayoutDescriptor {
    /// Create a new serialized layout with handle.
    pub fn new(handle: LayoutHandle, layout: LayoutDescriptor) -> Self {
        Self { handle, layout }
    }
}

/// The set of [`LocalLayoutDescriptor`] that are RDMA enabled. This object packages the detail
/// about the layouts and the NIXL RDMA metadata required to reconstruct the layouts and access
/// the memory via NIXL RDMA.
#[derive(Debug, Encode, Decode)]
pub struct RdmaLayoutDescriptors {
    /// Worker identification
    pub worker_address: WorkerAddress,
    /// Exported NIXL metadata from nixl_sys::Agent::get_local_md()
    pub nixl_metadata: Vec<u8>,
    /// Serialized layouts (handle + layout data)
    pub layouts: Vec<LocalLayoutDescriptor>,
}

/// Managed memory metadata package for export/import.
///
/// This is the wire format for transmitting layout metadata between workers.
/// It contains everything needed to reconstruct remote layouts and load their
/// NIXL registration data.
pub struct SerializedLayout(Bytes);

impl SerializedLayout {
    /// Pack metadata into a serialized form.
    ///
    /// # Arguments
    /// * `worker_address` - Worker identification
    /// * `nixl_metadata` - NIXL metadata blob from get_local_md()
    /// * `layouts` - Vector of layouts with handles to export
    ///
    /// # Returns
    /// Packed metadata ready for transmission
    pub fn pack(
        worker_address: WorkerAddress,
        nixl_metadata: Vec<u8>,
        layouts: Vec<LocalLayoutDescriptor>,
    ) -> Result<Self> {
        let inner = RdmaLayoutDescriptors {
            worker_address,
            nixl_metadata,
            layouts,
        };
        let bytes = bincode::encode_to_vec(&inner, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("failed to encode managed memory metadata: {}", e))?;
        Ok(Self(Bytes::from(bytes)))
    }

    /// Unpack metadata from serialized form.
    ///
    /// # Returns
    /// Unpacked metadata structure
    pub fn unpack(&self) -> Result<RdmaLayoutDescriptors> {
        let (inner, _) = bincode::decode_from_slice(&self.0, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("failed to decode managed memory metadata: {}", e))?;
        Ok(inner)
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &Bytes {
        &self.0
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: Bytes) -> Self {
        Self(bytes)
    }

    /// Get the size in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for SerializedLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SerializedLayout")
            .field("size_bytes", &self.len())
            .finish()
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::v2::memory::{MemoryDescriptor, StorageKind};
    use crate::block_manager::v2::physical::layout::{
        BlockFormat, FullyContiguousDetails, LayoutConfig, LayoutDescriptor, LayoutTypeDetails,
        NixlMetadata,
    };

    fn make_test_serialized_layout() -> LayoutDescriptor {
        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        LayoutDescriptor {
            version: 1,
            layout_config: config,
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("test".to_string(), nixl_sys::MemType::Dram, 0),
            memory_descriptors: vec![MemoryDescriptor::new(0x1000, 4096)],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
            }),
        }
    }

    #[test]
    fn test_worker_address() {
        let addr = WorkerAddress::new(42, "test_agent".to_string());
        assert_eq!(addr.worker_id, 42);
        assert_eq!(addr.nixl_agent_name, "test_agent");
    }

    #[test]
    fn test_serialized_layout_with_handle() {
        let handle = LayoutHandle::new(1, 2);
        let layout = make_test_serialized_layout();
        let with_handle = LocalLayoutDescriptor::new(handle, layout);

        assert_eq!(with_handle.handle, handle);
    }

    #[test]
    fn test_metadata_pack_unpack() {
        let worker_address = WorkerAddress::new(100, "worker_100".to_string());
        let nixl_metadata = vec![1, 2, 3, 4, 5];
        let layouts = vec![LocalLayoutDescriptor::new(
            LayoutHandle::new(100, 1),
            make_test_serialized_layout(),
        )];

        let packed =
            SerializedLayout::pack(worker_address.clone(), nixl_metadata.clone(), layouts).unwrap();

        assert!(!packed.is_empty());
        assert!(!packed.is_empty());

        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.worker_address, worker_address);
        assert_eq!(unpacked.nixl_metadata, nixl_metadata);
        assert_eq!(unpacked.layouts.len(), 1);
        assert_eq!(unpacked.layouts[0].handle.worker_id(), 100);
        assert_eq!(unpacked.layouts[0].handle.layout_id(), 1);
    }

    #[test]
    fn test_metadata_multiple_layouts() {
        let worker_address = WorkerAddress::new(200, "worker_200".to_string());
        let nixl_metadata = vec![10, 20, 30];
        let layouts = vec![
            LocalLayoutDescriptor::new(LayoutHandle::new(200, 1), make_test_serialized_layout()),
            LocalLayoutDescriptor::new(LayoutHandle::new(200, 2), make_test_serialized_layout()),
            LocalLayoutDescriptor::new(LayoutHandle::new(200, 3), make_test_serialized_layout()),
        ];

        let packed =
            SerializedLayout::pack(worker_address, nixl_metadata, layouts.clone()).unwrap();
        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.layouts.len(), 3);
        for (i, layout) in unpacked.layouts.iter().enumerate() {
            assert_eq!(layout.handle.worker_id(), 200);
            assert_eq!(layout.handle.layout_id(), (i + 1) as u16);
        }
    }

    #[test]
    fn test_metadata_from_bytes() {
        let worker_address = WorkerAddress::new(42, "test".to_string());
        let nixl_metadata = vec![1, 2, 3];
        let layouts = vec![];

        let packed = SerializedLayout::pack(worker_address, nixl_metadata, layouts).unwrap();
        let bytes = packed.as_bytes().clone();

        let restored = SerializedLayout::from_bytes(bytes);
        let unpacked = restored.unpack().unwrap();

        assert_eq!(unpacked.worker_address.worker_id, 42);
    }
}
