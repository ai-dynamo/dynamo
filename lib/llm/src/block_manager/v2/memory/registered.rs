// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL registration wrapper for storage types.

use super::{MemoryRegion, StorageKind};
use nixl_sys::{Agent as NixlAgent, MemType, OptArgs, RegistrationHandle};
use std::any::Any;
use std::fmt;

/// Trait for storage types that can be registered with NIXL.
pub trait NixlCompatible {
    /// Get parameters needed for NIXL registration.
    ///
    /// Returns (ptr, size, mem_type, device_id)
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64);
}

/// NIXL descriptor containing registration information.
#[derive(Debug, Clone)]
pub struct NixlDescriptor {
    pub addr: u64,
    pub size: usize,
    pub mem_type: MemType,
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

impl<S: MemoryRegion + NixlCompatible + 'static> MemoryRegion for NixlRegistered<S> {
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

impl<S: MemoryRegion + NixlCompatible> RegisteredView for NixlRegistered<S> {
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

impl<S: MemoryRegion + NixlCompatible> NixlRegistered<S> {
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
        // Manually drop the handle first
        self.handle = None;
        // Now we can move out the storage
        // We need to use mem::forget to prevent Drop from running
        let storage = std::mem::replace(&mut self.storage, unsafe { std::mem::zeroed() });
        std::mem::forget(self);
        storage
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
    agent: &NixlAgent,
    opt: Option<&OptArgs>,
) -> std::result::Result<NixlRegistered<S>, S>
where
    S: MemoryRegion + NixlCompatible,
{
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
