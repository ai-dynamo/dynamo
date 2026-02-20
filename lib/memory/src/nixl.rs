// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL registration wrapper for storage types.

mod agent;
mod config;

use super::{MemoryDescriptor, StorageKind};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub use agent::NixlAgent;
pub use config::NixlBackendConfig;

pub use nixl_sys::{
    Agent, MemType, NotificationMap, OptArgs, RegistrationHandle, XferDescList, XferOp, XferRequest,
};
pub use serde::{Deserialize, Serialize};

/// Trait for storage types that can be registered with NIXL.
pub trait NixlCompatible {
    /// Get parameters needed for NIXL registration.
    ///
    /// Returns (ptr, size, mem_type, device_id)
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64);
}

/// Combined trait for memory that can be registered with NIXL.
///
/// This supertrait enables type erasure via `Arc<dyn NixlMemory>`.
/// Any type implementing both `MemoryDescriptor` and `NixlCompatible`
/// automatically implements this trait via the blanket implementation.
pub trait NixlMemory: MemoryDescriptor + NixlCompatible {}

// Blanket impl - any type with both traits automatically implements NixlMemory
impl<T: MemoryDescriptor + NixlCompatible + ?Sized> NixlMemory for T {}

/// NIXL descriptor containing registration information.
///
/// This struct holds the information needed to describe a memory region
/// to NIXL for transfer operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlDescriptor {
    /// Base address of the memory region.
    pub addr: u64,
    /// Size of the memory region in bytes.
    pub size: usize,
    /// Type of memory (host, device, etc.).
    pub mem_type: MemType,
    /// Device identifier (GPU index for device memory, 0 for host memory).
    pub device_id: u64,
}

impl nixl_sys::MemoryRegion for NixlDescriptor {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl nixl_sys::NixlDescriptor for NixlDescriptor {
    fn mem_type(&self) -> MemType {
        self.mem_type
    }

    fn device_id(&self) -> u64 {
        self.device_id
    }
}

/// View trait for accessing registration information without unwrapping.
pub trait RegisteredView {
    /// Get the name of the NIXL agent that registered this memory.
    fn agent_name(&self) -> &str;

    /// Get the NIXL descriptor for this registered memory.
    fn descriptor(&self) -> NixlDescriptor;
}

/// Wrapper for storage that has been registered with NIXL.
///
/// This wrapper ensures proper drop order: the registration handle is
/// dropped before the storage, ensuring deregistration happens before
/// the memory is freed.
pub struct NixlRegistered<S: NixlCompatible> {
    storage: S,
    handle: Option<RegistrationHandle>,
    agent_name: String,
}

impl<S: NixlCompatible> Drop for NixlRegistered<S> {
    fn drop(&mut self) {
        // Explicitly drop the registration handle first
        drop(self.handle.take());
        // Storage drops naturally after
    }
}

impl<S: NixlCompatible + fmt::Debug> fmt::Debug for NixlRegistered<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NixlRegistered")
            .field("storage", &self.storage)
            .field("agent_name", &self.agent_name)
            .field("handle", &self.handle.is_some())
            .finish()
    }
}

impl<S: MemoryDescriptor + NixlCompatible + 'static> MemoryDescriptor for NixlRegistered<S> {
    fn addr(&self) -> usize {
        self.storage.addr()
    }

    fn size(&self) -> usize {
        self.storage.size()
    }

    fn storage_kind(&self) -> StorageKind {
        self.storage.storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        Some(self.descriptor())
    }
}

impl<S: MemoryDescriptor + NixlCompatible> RegisteredView for NixlRegistered<S> {
    fn agent_name(&self) -> &str {
        &self.agent_name
    }

    fn descriptor(&self) -> NixlDescriptor {
        let (ptr, size, mem_type, device_id) = self.storage.nixl_params();
        NixlDescriptor {
            addr: ptr as u64,
            size,
            mem_type,
            device_id,
        }
    }
}

impl<S: MemoryDescriptor + NixlCompatible> NixlRegistered<S> {
    /// Get a reference to the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get a mutable reference to the underlying storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Check if the registration handle is still valid.
    pub fn is_registered(&self) -> bool {
        self.handle.is_some()
    }

    /// Consume this wrapper and return the underlying storage.
    ///
    /// This will deregister the storage from NIXL.
    pub fn into_storage(mut self) -> S {
        drop(self.handle.take());
        let mut this = std::mem::ManuallyDrop::new(self);
        unsafe {
            let storage = std::ptr::read(&this.storage);
            std::ptr::drop_in_place(&mut this.agent_name);
            storage
        }
    }
}

/// Register storage with a NIXL agent.
///
/// This consumes the storage and returns a `NixlRegistered` wrapper that
/// manages the registration lifetime. The registration handle will be
/// automatically dropped when the wrapper is dropped, ensuring proper
/// cleanup order.
///
/// # Arguments
/// * `storage` - The storage to register (consumed)
/// * `agent` - The NIXL agent to register with
/// * `opt` - Optional arguments for registration
///
/// # Returns
/// A `NixlRegistered` wrapper containing the storage and registration handle.
pub fn register_with_nixl<S>(
    storage: S,
    agent: &Agent,
    opt: Option<&OptArgs>,
) -> std::result::Result<NixlRegistered<S>, S>
where
    S: MemoryDescriptor + NixlCompatible,
{
    // let storage_kind = storage.storage_kind();

    // // Determine if registration is needed based on storage type and available backends
    // let should_register = match storage_kind {
    //     StorageKind::System | StorageKind::Pinned => {
    //         // System/Pinned memory needs UCX for remote transfers
    //         agent.has_backend("UCX") || agent.has_backend("POSIX")
    //     }
    //     StorageKind::Device(_) => {
    //         // Device memory needs UCX for remote transfers OR GDS for direct disk transfers
    //         agent.has_backend("UCX") || agent.has_backend("GDS_MT")
    //     }
    //     StorageKind::Disk(_) => {
    //         // Disk storage needs POSIX for regular I/O OR GDS for GPU direct I/O
    //         agent.has_backend("POSIX") || agent.has_backend("GDS_MT")
    //     } // StorageKind::Object(_) => {
    //       //     // Object storage is always registered via NIXL's OBJ plugin
    //       //     agent.has_backend("OBJ")
    //       // }
    // };

    // this is not true for our future object storage. so let's rethink this.
    // for object, if there is no device_id or device_id is 0, then we need to register
    // alternatively, the object storage holds it's own internal metadata but does not
    // expose as a nixl descriptor, thus ObjectStorag will by default like all other storage
    // types have a None for nixl_descriptor(), and we will use the internal
    if storage.nixl_descriptor().is_some() {
        return Ok(NixlRegistered {
            storage,
            handle: None,
            agent_name: agent.name().to_string(),
        });
    }

    // Get NIXL parameters
    let (ptr, size, mem_type, device_id) = storage.nixl_params();

    // Create a NIXL descriptor for registration
    let descriptor = NixlDescriptor {
        addr: ptr as u64,
        size,
        mem_type,
        device_id,
    };

    match agent.register_memory(&descriptor, opt) {
        Ok(handle) => Ok(NixlRegistered {
            storage,
            handle: Some(handle),
            agent_name: agent.name().to_string(),
        }),
        Err(_) => Err(storage),
    }
}

// =============================================================================
// Arc<dyn NixlMemory> support
// =============================================================================

impl NixlCompatible for Arc<dyn NixlMemory + Send + Sync> {
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
        (**self).nixl_params()
    }
}

impl MemoryDescriptor for Arc<dyn NixlMemory + Send + Sync> {
    fn addr(&self) -> usize {
        (**self).addr()
    }

    fn size(&self) -> usize {
        (**self).size()
    }

    fn storage_kind(&self) -> StorageKind {
        (**self).storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        (**self).nixl_descriptor()
    }
}

// =============================================================================
// Extension trait for ergonomic API
// =============================================================================

/// Extension trait providing ergonomic `.register()` method for NIXL registration.
///
/// This trait is automatically implemented for all types that implement both
/// `MemoryDescriptor` and `NixlCompatible`. Import this trait to use the
/// method syntax:
///
///
pub trait NixlRegisterExt: MemoryDescriptor + NixlCompatible + Sized {
    /// Get this memory as NIXL-registered.
    ///
    /// This operation is idempotent - it's a no-op if the memory is already registered.
    ///
    /// # Arguments
    /// * `agent` - The NIXL agent to register with
    /// * `opt` - Optional arguments for registration
    ///
    /// # Returns
    /// A `NixlRegistered` wrapper on success, or the original storage on failure.
    fn register(
        self,
        agent: &NixlAgent,
        opt: Option<&OptArgs>,
    ) -> std::result::Result<NixlRegistered<Self>, Self> {
        register_with_nixl(self, agent, opt)
    }
}

// Blanket impl for all compatible types
impl<T: MemoryDescriptor + NixlCompatible + Sized> NixlRegisterExt for T {}
