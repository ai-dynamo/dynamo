// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical layout types that combine abstract layouts with storage location metadata.

use super::{
    FullyContiguousLayout, LayerSeparateLayout, Layout, MemoryDescriptor,
    builder::{PhysicalLayoutBuilder, PhysicalLayoutBuilderDefault},
    serialize::{LayoutDescriptor, LayoutTypeDetails},
};

use crate::block_manager::v2::memory::{MemoryRegion, StorageKind};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::Arc;

use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;

/// Runtime representation of a layout with its physical storage location.
///
/// A `PhysicalLayout` wraps an abstract [`Layout`] with information about where
/// its memory physically resides (GPU, host, disk) and whether it's local or remote.
/// This enables the transfer system to select appropriate copy strategies and build
/// NIXL transfer descriptors.
#[derive(Debug, Clone)]
pub struct PhysicalLayout {
    /// The abstract layout defining memory organization
    layout: Arc<dyn Layout>,

    /// Physical storage location (System, Device, Pinned, Disk)
    location: StorageKind,

    /// NIXL registration metadata
    nixl_metadata: NixlMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlMetadata {
    agent_name: String,
    mem_type: nixl_sys::MemType,
    device_id: u64,
}

impl NixlMetadata {
    pub fn new(agent_name: String, mem_type: nixl_sys::MemType, device_id: u64) -> Self {
        Self {
            agent_name,
            mem_type,
            device_id,
        }
    }

    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    pub fn mem_type(&self) -> nixl_sys::MemType {
        self.mem_type
    }

    pub fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl PhysicalLayout {
    /// Create a typed builder that enforces NIXL registration.
    pub fn builder(agent: NixlAgent) -> PhysicalLayoutBuilderDefault {
        PhysicalLayoutBuilder::new(agent)
    }

    /// Create a new local physical layout.
    ///
    /// # Arguments
    /// * `layout` - The abstract layout to wrap
    /// * `location` - Where the layout's memory resides
    pub(crate) fn new_local(
        layout: Arc<dyn Layout>,
        location: StorageKind,
        nixl_metadata: NixlMetadata,
    ) -> Self {
        Self {
            layout,
            location,
            nixl_metadata,
        }
    }

    // /// Create a new remote physical layout from a descriptor.
    // ///
    // /// # Arguments
    // /// * `layout` - The abstract layout to wrap
    // /// * `location` - Where the layout's memory resides (on remote node)
    // /// * `remote_agent` - Name of the NIXL agent on the remote node
    // pub fn new_remote(
    //     layout: Arc<dyn Layout>,
    //     location: StorageKind,
    //     remote_agent: String,
    // ) -> Self {
    //     let metadata = NixlMetadata::new(
    //         remote_agent.clone(),
    //         location.to_nixl_mem_type(),
    //         location.device_id(),
    //     );
    //     let registrations = vec![RegisteredStorageMetadata::new(
    //         metadata.agent_name().to_string(),
    //         location,
    //     )];
    //     Self {
    //         layout,
    //         location,
    //         locality: Locality::Remote(remote_agent),
    //         nixl_metadata: Some(metadata),
    //         registered: registrations,
    //     }
    // }

    /// Get the underlying layout.
    pub fn layout(&self) -> &Arc<dyn Layout> {
        &self.layout
    }

    /// Get the storage location.
    pub fn location(&self) -> StorageKind {
        self.location
    }

    /// Get the NIXL metadata.
    pub fn nixl_metadata(&self) -> &NixlMetadata {
        &self.nixl_metadata
    }

    /// Get a memory region with location information.
    ///
    /// # Arguments
    /// * `block_id` - Block identifier
    /// * `layer_id` - Layer identifier
    /// * `outer_id` - Outer dimension identifier
    pub fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryDescriptor> {
        self.layout.memory_region(block_id, layer_id, outer_id)
    }

    /// Serialize this physical layout for transmission to remote nodes.
    ///
    /// This converts the runtime `PhysicalLayout` into a `LayoutDescriptor` that
    /// contains all information needed to reconstruct the layout on a remote node,
    /// including layout configuration, memory descriptors, NIXL metadata, and
    /// layout-type-specific details.
    ///
    /// # Returns
    /// A serializable representation of this layout
    pub fn to_descriptor(&self) -> Result<LayoutDescriptor> {
        // Extract memory descriptors
        let memory_descriptors = self
            .layout
            .memory_regions()
            .iter()
            .map(|region| MemoryDescriptor {
                addr: region.addr(),
                size: region.size(),
            })
            .collect();

        // Get layout type details from the layout itself
        let layout_type_details = self.layout.serialization_details();

        Ok(LayoutDescriptor {
            version: LayoutDescriptor::CURRENT_VERSION,
            layout_config: self.layout.config().clone(),
            location: self.location,
            nixl_metadata: self.nixl_metadata.clone(),
            memory_descriptors,
            layout_type_details,
        })
    }

    /// Reconstruct a physical layout from serialized data received from a remote node.
    ///
    /// This creates a new `PhysicalLayout` from a `LayoutDescriptor`. The reconstructed
    /// layout will have memory descriptors that point to the remote node's memory,
    /// allowing NIXL to build RDMA descriptors for remote access.
    ///
    /// # Arguments
    /// * `serialized` - Serialized layout data from a remote node
    ///
    /// # Returns
    /// A new `PhysicalLayout` representing the remote layout
    ///
    /// # Note
    /// The memory regions in the reconstructed layout are not valid for local access;
    /// they represent remote memory addresses and are used to build NIXL transfer descriptors.
    pub fn from_descriptor(serialized: LayoutDescriptor) -> Result<Self> {
        // Validate version
        if serialized.version > LayoutDescriptor::CURRENT_VERSION {
            return Err(anyhow!(
                "Unsupported serialization version: {}. Maximum supported: {}",
                serialized.version,
                LayoutDescriptor::CURRENT_VERSION
            ));
        }

        // Create remote memory regions from descriptors
        let remote_regions: Vec<Arc<dyn MemoryRegion>> = serialized
            .memory_descriptors
            .iter()
            .map(|desc| {
                Arc::new(RemoteMemoryDescriptor {
                    addr: desc.addr,
                    size: desc.size,
                    storage_kind: serialized.location,
                }) as Arc<dyn MemoryRegion>
            })
            .collect();

        // Reconstruct the layout based on type
        let layout: Arc<dyn Layout> = match serialized.layout_type_details {
            LayoutTypeDetails::FullyContiguous(details) => {
                if remote_regions.len() != 1 {
                    return Err(anyhow!(
                        "FullyContiguous layout requires exactly 1 memory region, got {}",
                        remote_regions.len()
                    ));
                }
                let layout = FullyContiguousLayout::new_with_format(
                    serialized.layout_config.clone(),
                    remote_regions[0].clone(),
                    details.block_format,
                )?;
                Arc::new(layout)
            }
            LayoutTypeDetails::LayerSeparate(details) => {
                if remote_regions.len() != serialized.layout_config.num_layers {
                    return Err(anyhow!(
                        "LayerSeparate layout requires {} memory regions (one per layer), got {}",
                        serialized.layout_config.num_layers,
                        remote_regions.len()
                    ));
                }
                let layout = LayerSeparateLayout::new(
                    serialized.layout_config.clone(),
                    remote_regions,
                    details.block_dim,
                )?;
                Arc::new(layout)
            }
        };

        Ok(Self {
            layout,
            location: serialized.location,
            nixl_metadata: serialized.nixl_metadata,
        })
    }
}

/// A memory region that represents remote memory addresses.
///
/// This type is used when reconstructing layouts from serialized data.
/// The addresses are not valid for local access but can be used to
/// build NIXL transfer descriptors for remote memory access.
#[derive(Debug)]
struct RemoteMemoryDescriptor {
    addr: usize,
    size: usize,
    storage_kind: StorageKind,
}

impl MemoryRegion for RemoteMemoryDescriptor {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        self.storage_kind
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
