// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization types for exporting/importing layout metadata with NIXL integration.

use super::handle::LayoutHandle;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::physical::layout::LayoutDescriptor;
use anyhow::Result;
use bincode::{Decode, Encode};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

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

/// Layout descriptor with its assigned handle and logical type for RDMA metadata exchange.
///
/// This includes the logical layout type (G1, G2, G3, G4) so that remote instances
/// know which physical handle corresponds to which tier.
#[derive(Debug, Clone, Encode, Decode)]
pub struct LogicalLayoutDescriptor {
    /// Unique handle for this layout
    pub handle: LayoutHandle,
    /// The logical layout type (G1, G2, G3, G4)
    pub logical_type: LogicalLayoutHandle,
    /// Serialized layout data (uses Serde, bridged via bincode)
    #[bincode(with_serde)]
    pub layout: LayoutDescriptor,
}

impl LogicalLayoutDescriptor {
    /// Create a new layout descriptor with handle and logical type.
    pub fn new(
        handle: LayoutHandle,
        logical_type: LogicalLayoutHandle,
        layout: LayoutDescriptor,
    ) -> Self {
        Self {
            handle,
            logical_type,
            layout,
        }
    }

    /// Create a layout descriptor with G2 as the default logical type.
    ///
    /// This is provided for backwards compatibility with code that doesn't
    /// track logical types. G2 is used as the default since it's the most
    /// common tier for RDMA transfers (GPU memory for KV cache).
    ///
    /// For proper RDMA transfers between instances, use `new()` with the
    /// correct logical type from the Worker's registered handles.
    pub fn new_with_default_type(handle: LayoutHandle, layout: LayoutDescriptor) -> Self {
        Self {
            handle,
            logical_type: LogicalLayoutHandle::G2,
            layout,
        }
    }
}

/// Type alias for backwards compatibility.
pub type LocalLayoutDescriptor = LogicalLayoutDescriptor;

/// The set of [`LogicalLayoutDescriptor`] that are RDMA enabled. This object packages the detail
/// about the layouts and the NIXL RDMA metadata required to reconstruct the layouts and access
/// the memory via NIXL RDMA.
#[derive(Debug, Encode, Decode)]
pub struct RdmaLayoutDescriptors {
    /// Worker identification
    pub worker_address: WorkerAddress,
    /// Exported NIXL metadata from nixl_sys::Agent::get_local_md()
    pub nixl_metadata: Vec<u8>,
    /// Serialized layouts (handle + logical type + layout data)
    pub layouts: Vec<LogicalLayoutDescriptor>,
}

/// Managed memory metadata package for export/import.
///
/// This is the wire format for transmitting layout metadata between workers.
/// It contains everything needed to reconstruct remote layouts and load their
/// NIXL registration data.
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializedLayout(Bytes);

impl SerializedLayout {
    /// Pack metadata into a serialized form.
    ///
    /// # Arguments
    /// * `worker_address` - Worker identification
    /// * `nixl_metadata` - NIXL metadata blob from get_local_md()
    /// * `layouts` - Vector of layouts with handles and logical types to export
    ///
    /// # Returns
    /// Packed metadata ready for transmission
    pub fn pack(
        worker_address: WorkerAddress,
        nixl_metadata: Vec<u8>,
        layouts: Vec<LogicalLayoutDescriptor>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        physical::layout::KvBlockLayout,
        v2::physical::layout::{
            BlockFormat, FullyContiguousDetails, LayoutConfig, LayoutDescriptor, LayoutTypeDetails,
            NixlMetadata,
        },
    };
    use dynamo_memory::{MemoryRegion, StorageKind, nixl};

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
            nixl_metadata: NixlMetadata::new("test".to_string(), nixl::MemType::Dram, 0),
            memory_descriptors: vec![MemoryRegion {
                addr: 0x1000,
                size: 4096,
            }],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
                kv_block_layout: KvBlockLayout::OperationalNHD,
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
        use crate::v2::logical::LogicalLayoutHandle;

        let handle = LayoutHandle::new(1, 2);
        let layout = make_test_serialized_layout();
        let with_handle = LogicalLayoutDescriptor::new(handle, LogicalLayoutHandle::G2, layout);

        assert_eq!(with_handle.handle, handle);
        assert_eq!(with_handle.logical_type, LogicalLayoutHandle::G2);
    }

    #[test]
    fn test_metadata_pack_unpack() {
        use crate::v2::logical::LogicalLayoutHandle;

        let worker_address = WorkerAddress::new(100, "worker_100".to_string());
        let nixl_metadata = vec![1, 2, 3, 4, 5];
        let layouts = vec![LogicalLayoutDescriptor::new(
            LayoutHandle::new(100, 1),
            LogicalLayoutHandle::G2,
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
        assert_eq!(unpacked.layouts[0].logical_type, LogicalLayoutHandle::G2);
    }

    #[test]
    fn test_metadata_multiple_layouts() {
        use crate::v2::logical::LogicalLayoutHandle;

        let worker_address = WorkerAddress::new(200, "worker_200".to_string());
        let nixl_metadata = vec![10, 20, 30];
        let layouts = vec![
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 1),
                LogicalLayoutHandle::G1,
                make_test_serialized_layout(),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 2),
                LogicalLayoutHandle::G2,
                make_test_serialized_layout(),
            ),
            LogicalLayoutDescriptor::new(
                LayoutHandle::new(200, 3),
                LogicalLayoutHandle::G3,
                make_test_serialized_layout(),
            ),
        ];

        let packed =
            SerializedLayout::pack(worker_address, nixl_metadata, layouts.clone()).unwrap();
        let unpacked = packed.unpack().unwrap();

        assert_eq!(unpacked.layouts.len(), 3);
        let expected_logical_types = [
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G3,
        ];
        for (i, layout) in unpacked.layouts.iter().enumerate() {
            assert_eq!(layout.handle.worker_id(), 200);
            assert_eq!(layout.handle.layout_id(), (i + 1) as u16);
            assert_eq!(layout.logical_type, expected_logical_types[i]);
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
