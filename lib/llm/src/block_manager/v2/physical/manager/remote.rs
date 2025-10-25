// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote layout wrapper reconstructed from imported metadata.

use super::handle::LayoutHandle;
use crate::block_manager::v2::physical::layout::PhysicalLayout;

/// A remote physical layout reconstructed from imported metadata.
///
/// This wraps a `PhysicalLayout` that was deserialized from another worker's
/// exported metadata. The layout's memory regions point to addresses on the
/// remote worker and are used for building NIXL RDMA transfer descriptors.
///
/// This type is cheap to clone as `PhysicalLayout` contains `Arc` internally.
#[derive(Debug, Clone)]
pub struct RemoteLayout {
    handle: LayoutHandle,
    layout: PhysicalLayout,
}

#[allow(dead_code)]
impl RemoteLayout {
    /// Create a new remote layout.
    ///
    /// # Arguments
    /// * `handle` - Unique handle for this layout (from remote worker)
    /// * `layout` - The reconstructed physical layout
    pub fn new(handle: LayoutHandle, layout: PhysicalLayout) -> Self {
        Self { handle, layout }
    }

    /// Get the handle for this layout.
    pub fn handle(&self) -> LayoutHandle {
        self.handle
    }

    /// Get a reference to the physical layout.
    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Get the worker_id from the handle (identifies the remote worker).
    pub fn worker_id(&self) -> u64 {
        self.handle.worker_id()
    }

    /// Get the layout_id from the handle.
    pub fn layout_id(&self) -> u16 {
        self.handle.layout_id()
    }

    /// Consume this remote layout and return the physical layout.
    pub fn into_layout(self) -> PhysicalLayout {
        self.layout
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::v2::physical::layout::{
        LayoutConfig, LayoutDescriptor, PhysicalLayout,
    };

    fn make_serialized_layout() -> LayoutDescriptor {
        use crate::block_manager::v2::memory::{MemoryDescriptor, StorageKind};
        use crate::block_manager::v2::physical::layout::{
            BlockFormat, FullyContiguousDetails, LayoutTypeDetails, NixlMetadata,
        };

        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let required_size = config.num_blocks
            * config.num_layers
            * config.outer_dim
            * config.page_size
            * config.inner_dim
            * config.dtype_width_bytes;

        LayoutDescriptor {
            version: 1,
            layout_config: config,
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new(
                "remote_agent".to_string(),
                nixl_sys::MemType::Dram,
                0,
            ),
            memory_descriptors: vec![MemoryDescriptor::new(0x1000, required_size)],
            layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
                block_format: BlockFormat::Operational,
            }),
        }
    }

    #[test]
    fn test_remote_layout_creation() {
        let handle = LayoutHandle::new(999, 42);
        let serialized = make_serialized_layout();
        let layout = PhysicalLayout::from_descriptor(serialized).unwrap();
        let remote = RemoteLayout::new(handle, layout);

        assert_eq!(remote.handle(), handle);
        assert_eq!(remote.worker_id(), 999);
        assert_eq!(remote.layout_id(), 42);
    }

    #[test]
    fn test_remote_layout_into_layout() {
        let handle = LayoutHandle::new(100, 200);
        let serialized = make_serialized_layout();
        let layout = PhysicalLayout::from_descriptor(serialized).unwrap();
        let remote = RemoteLayout::new(handle, layout);

        let _recovered = remote.into_layout();
        // Successfully consumed and returned the layout
    }
}
