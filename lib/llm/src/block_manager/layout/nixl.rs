// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # NIXL Integration for Block Layouts 🤝
//!
//! This module extends the core block layout functionalities defined in the parent `layout` module
//! with [NIXL](http://github.com/ai-dynamo/nixl) specific capabilities. It enables block layouts,
//! whose underlying storage is NIXL-registerable, to be registered with a NIXL agent and
//! serialized into a format suitable for sharing and reconstruction in distributed environments.
//!
//! ## Key Features & Components
//!
//! ### 1. NIXL-Specific Layout Traits
//! - [`NixlLayout`]: An umbrella trait that augments a [`BlockLayout`]. It requires the layout's
//!   associated `StorageType` to implement [`NixlRegisterableStorage`]. This trait provides the
//!   `nixl_register` method to register all underlying storage regions of the layout with a NIXL agent.
//! - [`BlockLayoutNixlStorage`]: A trait implemented by layouts to provide NIXL-specific memory
//!   information like `mem_type` and `device_id` directly from the layout structure, typically
//!   derived from its underlying storage.
//! - [`ToSerializedNixlBlockLayout`]: Implemented by layouts that can be converted into a
//!   [`SerializedNixlBlockLayout`]. This involves capturing the layout configuration and the NIXL
//!   descriptors of its storage.
//!
//! ### 2. Serializable NIXL Layout
//! - [`SerializedNixlBlockLayout`]: A struct that holds the serialized representation (as `Vec<u8>`)
//!   of a NIXL-compatible block layout. It can be deserialized to reconstruct the layout, typically
//!   on a remote node, assuming the described NIXL memory regions are accessible.
//! - `NixlBlockLayoutKinds`: An internal enum used during serialization to differentiate between
//!   different types of layouts (e.g., `FullyContiguous`).
//! - `SerializableNixlLayout<C>`: An internal generic struct that captures the configuration (`C`),
//!   base offset, NIXL storage descriptors, and storage type for a specific layout kind.
//!
//! ### 3. Integration with Core Layouts
//! The module provides implementations of these NIXL traits for concrete layout types from the
//! parent module, such as [`FullyContiguous`]. For example:
//! - `FullyContiguous<S>` (where `S:` [`NixlRegisterableStorage`]) implements [`NixlLayout`], allowing
//!   its storage to be registered.
//! - It also implements [`ToSerializedNixlBlockLayout`], enabling its configuration and NIXL storage
//!   descriptors to be serialized.
//!
//! ### 4. Layout Creation and Allocation Extensions
//! The [`LayoutConfig`] from the parent module is extended with methods like:
//! - `create_layout`: To create a NIXL-aware layout from existing NIXL-registerable storage.
//! - `allocate_layout`: To allocate storage using a NIXL-registerable storage allocator and then
//!   create the NIXL-aware layout.
//!
//! ## Usage Flow
//!
//! 1.  **Create/Allocate Layout**: A block layout (e.g., [`FullyContiguous`]) is created or allocated,
//!     ensuring its underlying storage is NIXL-compatible (e.g., using [`SystemStorage`] that implements
//!     [`NixlRegisterableStorage`]).
//! 2.  **Register with NIXL**: The [`nixl_register`] method from the [`NixlLayout`] trait is called on the
//!     layout instance with a [`NixlAgent`].
//! 3.  **Serialize**: The [`serialize`] method from [`ToSerializedNixlBlockLayout`] is used to get a
//!     [`SerializedNixlBlockLayout`].
//! 4.  **Transmit**: The [`SerializedNixlBlockLayout`] (or its byte representation) is sent to another
//!     process/node.
//! 5.  **Deserialize**: On the receiving end, [`SerializedNixlBlockLayout::deserialize`] is called to
//!     reconstruct an `Arc<dyn BlockLayout<StorageType = NixlStorage>>`. This reconstructed layout now
//!     refers to the remote NIXL memory regions.
//!
//! ```rust
//! use dynamo_llm::block_manager::layout::{LayoutConfig, LayoutType};
//! use dynamo_llm::block_manager::layout::nixl::{NixlLayout, ToSerializedNixlBlockLayout, SerializedNixlBlockLayout};
//! use dynamo_llm::block_manager::storage::nixl::NixlAgent;
//! use dynamo_llm::block_manager::storage::PinnedAllocator; // Assuming PinnedStorage is NixlRegisterable
//! use std::sync::Arc;
//!
//! // Configuration
//! let config = LayoutConfig::builder()
//!     .num_blocks(10)
//!     .num_layers(2)
//!     .page_size(4)
//!     .inner_dim(13)
//!     .build().unwrap();
//!
//! // 1. Allocate a NIXL-compatible layout
//! let allocator = Arc::new(PinnedAllocator::new().unwrap()); // PinnedAllocator provides NixlRegisterable PinnedStorage
//! let mut layout = config.allocate_layout(LayoutType::FullyContiguous, allocator).unwrap();
//!
//! // 2. Register with NIXL Agent
//! let agent = NixlAgent::new("my_agent").unwrap();
//! layout.nixl_register(&agent, None).unwrap();
//!
//! // 3. Serialize the layout
//! let serialized_layout = layout.serialize().unwrap();
//!
//! // 4. (Transmit serialized_layout to another process)
//!
//! // 5. Deserialize on the other end
//! let reconstructed_layout = SerializedNixlBlockLayout::deserialize(&serialized_layout).unwrap();
//! println!("Reconstructed layout refers to storage type: {:?}", reconstructed_layout.storage_type());
//! ```
//!
//! This module effectively bridges the local layout definitions with the requirements of distributed memory management via NIXL.

use crate::block_manager::storage::StorageType;

use super::{BlockLayout, BlockLayoutConfig, LayoutConfig, LayoutError, LayoutType};

use super::super::storage::{
    nixl::{MemType, NixlAgent, NixlRegisterableStorage, NixlStorage, OptArgs},
    Storage, StorageAllocator,
};
use super::{FullyContiguous, FullyContiguousConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Extends [BlockLayout] with NIXL-specific methods for registering with an NIXL agent.
pub trait NixlLayout: BlockLayout + BlockLayoutNixlStorage + ToSerializedNixlBlockLayout {
    /// Register the layout with an NIXL agent
    ///
    /// This will register all the individual memory regions associated with the [BlockLayout].
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> anyhow::Result<()>;
}

/// Trait for providing NIXL-specific memory information
pub trait BlockLayoutNixlStorage {
    /// Returns the memory type of the storage
    fn mem_type(&self) -> MemType;

    /// Returns the device ID of the storage
    fn device_id(&self) -> u64;
}

// Umbrella impl for all BlockLayout types that are NixlRegisterableStorage
impl<T> NixlLayout for T
where
    T: BlockLayout + BlockLayoutNixlStorage + ToSerializedNixlBlockLayout + ?Sized, // Implement for any T that is BlockLayout (potentially unsized)
    T::StorageType: NixlRegisterableStorage, // T's associated StorageType must be NixlStorage
{
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> anyhow::Result<()> {
        for storage in self.storage_mut() {
            storage.nixl_register(agent, opt_args)?;
        }
        Ok(())
    }
}

impl LayoutConfig {
    /// Create a new NIXL-aware layout from existing NIXL-registerable storage.
    pub fn create_layout<S: Storage + NixlRegisterableStorage>(
        &self,
        layout_type: LayoutType,
        storage: Vec<S>,
    ) -> Result<impl NixlLayout<StorageType = S>, LayoutError> {
        match layout_type {
            LayoutType::FullyContiguous => FullyContiguous::new(self.clone(), storage),
        }
    }

    /// Allocate a new NIXL-aware layout using a NIXL-registerable storage allocator.
    pub fn allocate_layout<S: Storage + NixlRegisterableStorage>(
        &self,
        layout_type: LayoutType,
        allocator: Arc<dyn StorageAllocator<S>>,
    ) -> Result<impl NixlLayout<StorageType = S>, LayoutError> {
        match layout_type {
            LayoutType::FullyContiguous => {
                FullyContiguous::allocate(self.clone(), allocator.as_ref())
            }
        }
    }
}

/// Trait to convert a BlockLayout instance into its NIXL-specific serializable representation.
pub trait ToSerializedNixlBlockLayout: BlockLayout<StorageType: NixlRegisterableStorage> {
    /// Converts the layout into a serializable format, ensuring it's backed by NIXL storage.
    /// Returns an error if the layout is not backed by storage providing NIXL descriptors.
    fn serialize(&self) -> Result<SerializedNixlBlockLayout, LayoutError>;
}

/// Serializable representation of a BlockLayout backed by NIXL storage.
#[derive(Serialize, Deserialize, Clone)]
pub struct SerializedNixlBlockLayout(Vec<u8>);

/// Enum representing the serializable state of different BlockLayout types
/// specifically when backed by NIXL-compatible storage.
#[derive(Serialize, Deserialize, Debug, Clone)]
enum NixlBlockLayoutKinds {
    FullyContiguous(SerializableNixlLayout<FullyContiguousConfig>),
    // Add variants for other layout types here
}

/// Serializable representation of FullyContiguous layout backed by NIXL storage.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializableNixlLayout<C: BlockLayoutConfig> {
    config: C,
    base_offset: usize,
    storage_descriptors: Vec<NixlStorage>,
    storage_type: StorageType,
}

impl<C> SerializableNixlLayout<C>
where
    C: BlockLayoutConfig + Serialize + for<'de> Deserialize<'de> + Clone + std::fmt::Debug,
{
    /// Create a new SerializableNixlLayout
    fn new(
        config: C,
        base_offset: usize,
        storage_descriptors: Vec<NixlStorage>,
        storage_type: StorageType,
    ) -> Self {
        Self {
            config,
            base_offset,
            storage_descriptors,
            storage_type,
        }
    }
}

impl<S: NixlRegisterableStorage> ToSerializedNixlBlockLayout for FullyContiguous<S> {
    fn serialize(&self) -> Result<SerializedNixlBlockLayout, LayoutError> {
        // Use accessors added previously
        let config = self.config.clone();
        let base_offset = self.base_offset;

        let storages = self.storage();

        if storages.len() != 1 {
            return Err(LayoutError::InvalidConfig(
                "FullyContiguous reconstruction expects exactly one NixlStorage descriptor"
                    .to_string(),
            ));
        }

        // FullyContiguous uses a Vec<Storage>, but should only contain one element.
        let storage_instance = storages.first().ok_or_else(|| {
            LayoutError::OperationFailed("FullyContiguous requires one storage element".to_string())
        })?;

        let storage_descriptors =
            unsafe { storage_instance.as_nixl_descriptor() }.ok_or_else(|| {
                LayoutError::OperationFailed(
                    "Storage does not provide NIXL descriptors for serialization".to_string(),
                )
            })?;

        let serializable_data = SerializableNixlLayout::new(
            config,
            base_offset,
            vec![storage_descriptors],
            self.storage_type(),
        );

        let nixl_block_layout = NixlBlockLayoutKinds::FullyContiguous(serializable_data);

        Ok(SerializedNixlBlockLayout(serde_json::to_vec(
            &nixl_block_layout,
        )?))
    }
}

impl SerializedNixlBlockLayout {
    /// Reconstructs a dynamic BlockLayout trait object backed by NixlStorage
    /// from the serialized layout information.
    /// Assumes the NixlStorage regions described within already exist and are valid.
    pub fn deserialize(
        &self,
    ) -> Result<Arc<dyn BlockLayout<StorageType = NixlStorage>>, LayoutError> {
        let nixl_block_layout: NixlBlockLayoutKinds = serde_json::from_slice(&self.0)?;
        match nixl_block_layout {
            NixlBlockLayoutKinds::FullyContiguous(config) => {
                if config.storage_descriptors.len() != 1 {
                    return Err(LayoutError::InvalidConfig(
                        "FullyContiguous reconstruction expects exactly one NixlStorage descriptor"
                            .to_string(),
                    ));
                }
                // Clone the single NixlStorage descriptor to become the storage instance
                let storage = config.storage_descriptors[0].clone();

                // Use the internal constructor which skips allocation checks
                let layout = FullyContiguous::new_internal(
                    config.config.clone(),
                    storage, // Pass the NixlStorage instance
                    config.base_offset,
                    config.storage_type,
                )?;
                Ok(Arc::new(layout))
            } // Handle other variants when added...
        }
    }
}

impl<S> BlockLayoutNixlStorage for FullyContiguous<S>
where
    S: Storage + NixlRegisterableStorage,
{
    fn mem_type(&self) -> MemType {
        self.storage.mem_type()
    }

    fn device_id(&self) -> u64 {
        self.storage.device_id()
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::block_manager::storage::SystemAllocator;
    use dynamo_runtime::logging::init as init_logging;

    #[test]
    fn test_nixl_layout() {
        init_logging();

        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(2)
            .page_size(4)
            .inner_dim(13)
            .build()
            .unwrap();

        config.validate().unwrap();

        let mut layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
        let agent = NixlAgent::new("test").unwrap();

        tracing::info!("Registering layout");
        layout.nixl_register(&agent, None).unwrap();
        tracing::info!("Layout registered");
        let local_storage_type = layout.storage_type();

        let serialized = layout.serialize().unwrap();

        let remote_layout = SerializedNixlBlockLayout::deserialize(&serialized).unwrap();
        println!("Nixl layout: {:?}", remote_layout);
        let remote_storage_type = remote_layout.storage_type();

        assert_eq!(local_storage_type, remote_storage_type);

        drop(layout);
        tracing::info!("Layout dropped");
    }
}
