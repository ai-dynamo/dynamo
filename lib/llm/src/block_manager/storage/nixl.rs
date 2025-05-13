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

//! # NIXL Storage Support
//!
//! This module provides NIXL-specific storage implementations and integration for the block manager.
//! It is conditionally compiled based on the `nixl` feature flag.
//!
//! ## Features
//!
//! The following functionality is available when the `nixl` feature is enabled:
//! - [`NixlStorage`] - Remote memory representation
//! - [`NixlRegisterableStorage`] - Trait for NIXL-compatible storage types
//! - Integration with the NIXL agent system for remote memory access
//!
//! ## Memory Registration
//!
//! The module extends the core storage types with NIXL registration capabilities:
//! - Automatic registration handle management
//! - Memory type mapping between storage and NIXL types
//! - Device ID tracking for GPU memory
//!
//! ## Usage
//!
//! ```rust
//! use dynamo_llm::block_manager::storage::{
//!     PinnedAllocator, StorageAllocator,
//!     nixl::NixlRegisterableStorage
//! };
//! use nixl_sys::Agent as NixlAgent;
//!
//! // Create a NIXL agent
//! let agent = NixlAgent::new("my_agent").unwrap();
//!
//! // Create storage using an allocator
//! let pinned_allocator = PinnedAllocator::default();
//! let mut storage = pinned_allocator.allocate(1024).unwrap();
//!
//! // Initially no NIXL descriptors are available
//! assert!(unsafe { storage.as_nixl_descriptor() }.is_none());
//!
//! // Register with NIXL
//! storage.nixl_register(&agent, None).unwrap();
//!
//! // Now we can get NIXL descriptors
//! // NIXL descriptors are not owned by the storage, so we need to access them
//! // through an unsafe method.
//! if let Some(nixl_desc) = unsafe { storage.as_nixl_descriptor() } {
//!     // Use NIXL memory region
//!     println!("NIXL memory at addr: {}", nixl_desc.addr());
//!     println!("Memory type: {:?}", nixl_desc.mem_type());
//!     println!("Device ID: {}", nixl_desc.device_id());
//! }
//! ```
//!
//! ## Safety
//!
//! The module ensures safe interaction with NIXL by:
//! - Managing registration lifetimes
//! - Validating memory types and device IDs
//! - Providing type-safe interfaces for remote memory access
//! - Automatic cleanup of NIXL resources

pub use nixl_sys::{
    Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, OptArgs,
    RegistrationHandle as NixlRegistrationHandle,
};

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use super::{
    CudaContextProivder, DeviceStorage, PinnedStorage, RegistationHandle, RegisterableStorage,
    Remote, Storage, StorageError, StorageType, SystemStorage,
};

/// Marker trait for storage types that can be accessed by NIXL.
///
/// This trait is different from [`NixlRegisterableStorage`] which has further restrictions
/// that the [`Storage`] must be [`RegisterableStorage`].
///
/// Remote memory described by [`NixlStorage`] is [`NixlAccessible`] but is not [`NixlRegisterableStorage`]
/// due to the fact it represents memory that is registered to another NIXL agent.
pub trait NixlAccessible {}

impl StorageType {
    /// Get the NIXL memory type for a given storage type.
    pub fn nixl_mem_type(&self) -> MemType {
        match self {
            StorageType::System => MemType::Dram,
            StorageType::Pinned => MemType::Dram,
            StorageType::Device(_) => MemType::Vram,
            StorageType::Nixl => MemType::Unknown,
            StorageType::Null => MemType::Unknown,
        }
    }

    /// Get the NIXL device ID for a given storage type.
    pub fn nixl_device_id(&self) -> u64 {
        match self {
            StorageType::System => 0,
            StorageType::Pinned => 0,
            StorageType::Device(id) => *id as u64,
            StorageType::Nixl => 0,
            StorageType::Null => 0,
        }
    }
}

impl RegistationHandle for NixlRegistrationHandle {
    fn release(&mut self) {
        if let Err(e) = self.deregister() {
            tracing::error!("Failed to deregister Nixl storage: {}", e);
        }
    }
}

/// Extension to the [`RegisterableStorage`] trait for NIXL-compatible storage.
pub trait NixlRegisterableStorage: RegisterableStorage + NixlDescriptor + Sized {
    /// Register the storage with the NIXL agent.
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), StorageError> {
        let handle = Box::new(agent.register_memory(self, opt_args)?);
        // Assuming PinnedStorage has `handles: RegistrationHandles`
        self.register("nixl", handle)
    }

    /// Check if the storage is registered with the NIXL agent.
    fn is_nixl_registered(&self) -> bool {
        self.is_registered("nixl")
    }

    /// Get the NIXL agent name for the storage.
    fn nixl_agent_name(&self) -> Option<String> {
        // Get the registration handle associated with "nixl".
        self.registration_handle("nixl")
            // If a handle exists, attempt to downcast it.
            .and_then(|handle_box| {
                // Cast the trait object &dyn RegistationHandle to &dyn Any
                // then attempt to downcast to the concrete NixlRegistrationHandle type.
                // Note: This requires RegistationHandle: Any + 'static
                (handle_box as &dyn std::any::Any)
                    .downcast_ref::<NixlRegistrationHandle>()
                    // If downcast succeeds, get the agent name.
                    .map(|nixl_handle| nixl_handle.agent_name())
            })?
    }

    /// If the underlying storage is NIXL-compatible, return descriptions of the NIXL memory regions.
    /// This is used for serialization/deserialization of NIXL-specific layouts.
    ///
    /// # Safety
    ///
    /// This function is unsafe because because ownership of the storage is not transferred.
    unsafe fn as_nixl_descriptor(&self) -> Option<NixlStorage> {
        if self.is_nixl_registered() {
            Some(NixlStorage {
                addr: self.addr(),
                size: MemoryRegion::size(self),
                mem_type: self.mem_type(),
                device_id: self.device_id(),
            })
        } else {
            None
        }
    }
}

/// NIXL-compatible storage
///
/// This object does not own any memory, it is meant to hold descriptions
/// of non-local/remote memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Getters)]
pub struct NixlStorage {
    addr: u64,
    size: usize,
    mem_type: MemType,
    device_id: u64,
}

impl Remote for NixlStorage {}
impl NixlAccessible for NixlStorage {}

impl Storage for NixlStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Nixl
    }

    fn addr(&self) -> u64 {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.addr as *mut u8
    }
}

impl MemoryRegion for NixlStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for NixlStorage {
    fn mem_type(&self) -> MemType {
        self.mem_type
    }

    fn device_id(&self) -> u64 {
        self.device_id
    }
}

// SystemStorage

impl NixlRegisterableStorage for SystemStorage {}

impl MemoryRegion for SystemStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for SystemStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

// PinnedStorage

impl NixlAccessible for PinnedStorage {}
impl NixlRegisterableStorage for PinnedStorage {}

impl MemoryRegion for PinnedStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        Storage::as_ptr(self)
    }

    fn size(&self) -> usize {
        Storage::size(self)
    }
}

impl NixlDescriptor for PinnedStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

// DeviceStorage

impl NixlAccessible for DeviceStorage {}
impl NixlRegisterableStorage for DeviceStorage {}

impl MemoryRegion for DeviceStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        Storage::as_ptr(self)
    }

    fn size(&self) -> usize {
        Storage::size(self)
    }
}

impl NixlDescriptor for DeviceStorage {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }

    fn device_id(&self) -> u64 {
        CudaContextProivder::cuda_context(self).cu_device() as u64
    }
}
