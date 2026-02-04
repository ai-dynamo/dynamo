// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
//! ```rust,ignore
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
    CudaContextProivder, DeviceStorage, DiskStorage, PinnedStorage, RegistationHandle,
    RegisterableStorage, Remote, Storage, StorageError, StorageType, SystemStorage,
};

/// NIXL remote descriptor
///
/// This struct is used to describe a remote memory region that is accessible by a NIXL agent.
///
/// This object is capable of being serialized and transfered to other nodes.  It carries with it
/// the necessary information to create [`nixl_sys::XferDescList`].
///
/// The [`NixlRemoteDescriptor`] can be used in traits that READ from remote memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlRemoteDescriptor {
    storage: NixlStorage,
    agent: String,
    notif: Option<String>,
}

impl NixlRemoteDescriptor {
    pub(crate) fn new(storage: NixlStorage, agent: String) -> Self {
        Self {
            storage,
            agent,
            notif: None,
        }
    }

    /// Size in bytes of the remote memory region
    pub fn size(&self) -> usize {
        *self.storage.size()
    }

    /// Notification to be delivered to remote agent after the memory is fetched by a
    /// NIXL READ operation.
    pub fn set_notif(&mut self, notif: String) {
        self.notif = Some(notif);
    }

    /// Clear the notification.
    pub fn clear_notif(&mut self) {
        self.notif = None;
    }

    /// Get the notification value to be delivered to remote agent after the memory is fetched by a
    /// NIXL READ operation.
    pub fn get_notif(&self) -> Option<String> {
        self.notif.clone()
    }

    /// Get the NIXL agent name for the storage.
    pub fn agent_name(&self) -> &str {
        self.agent.as_str()
    }
}

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
            StorageType::Disk(_) => MemType::File,
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

fn handle_nixl_register<S: NixlRegisterableStorage>(
    storage: &mut S,
    agent: &NixlAgent,
    opt_args: Option<&OptArgs>,
) -> Result<(), StorageError> {
    let handle = Box::new(agent.register_memory(storage, opt_args)?);
    storage.register("nixl", handle)
}

/// Extension to the [`RegisterableStorage`] trait for NIXL-compatible storage.
pub trait NixlRegisterableStorage: RegisterableStorage + NixlDescriptor + Sized {
    /// Register the storage with the NIXL agent.
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), StorageError> {
        handle_nixl_register(self, agent, opt_args)
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

impl NixlStorage {
    pub(crate) fn from_storage_with_offset<S: Storage + NixlDescriptor>(
        storage: &S,
        offset: usize,
        size: usize,
    ) -> Result<Self, StorageError> {
        if offset + size > Storage::size(storage) {
            return Err(StorageError::OutOfBounds(format!(
                "Offset: {}, Size: {}, Total Size: {}",
                offset,
                size,
                Storage::size(storage)
            )));
        }

        Ok(Self {
            addr: storage.addr() + offset as u64,
            size,
            mem_type: storage.mem_type(),
            device_id: storage.device_id(),
        })
    }
}

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
        unsafe { Storage::as_ptr(self) }
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
        unsafe { Storage::as_ptr(self) }
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

impl NixlAccessible for DiskStorage {}
impl NixlRegisterableStorage for DiskStorage {
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), StorageError> {
        if self.unlinked() {
            return Err(StorageError::AllocationFailed(
                "Disk storage has already been unlinked. GDS registration will fail.".to_string(),
            ));
        }

        handle_nixl_register(self, agent, opt_args)?;
        self.unlink()?;
        Ok(())
    }
}

impl MemoryRegion for DiskStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        unsafe { Storage::as_ptr(self) }
    }

    fn size(&self) -> usize {
        Storage::size(self)
    }
}

impl NixlDescriptor for DiskStorage {
    fn mem_type(&self) -> MemType {
        MemType::File
    }

    /// Nixl treats the file descriptor as the device ID.
    fn device_id(&self) -> u64 {
        self.fd()
    }
}
