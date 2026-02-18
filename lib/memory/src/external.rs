// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External memory wrapper for memory allocated by external frameworks.
//!
//! This module provides `ExternalDeviceMemory` for wrapping pointers to GPU
//! memory allocated by external frameworks (e.g., vLLM's KV cache). This type
//! does NOT own the memory - ownership remains with the external framework.
//!
//! The primary use case is registering external GPU memory with NIXL for RDMA
//! transfers without copying.

use crate::nixl::{MemType, NixlCompatible, NixlDescriptor};
use crate::{MemoryDescriptor, StorageKind};
use std::any::Any;
use std::fmt;

/// Wrapper for externally-allocated device (GPU) memory.
///
/// This type wraps a raw pointer to GPU memory that is owned by an external
/// framework (like vLLM). It provides the necessary traits for NIXL registration
/// without taking ownership of the underlying memory.
///
/// # Safety
///
/// This type relies on the caller to guarantee that:
/// - The pointer points to valid GPU memory on the specified device
/// - The memory remains valid for the lifetime of this wrapper
/// - The memory size is exactly as specified
/// - The external framework doesn't free the memory while this wrapper exists
///
/// # Example
///
/// ```ignore
/// // vLLM allocates KV cache tensors
/// let tensor_ptr = tensor.data_ptr();
/// let tensor_size = tensor.size_bytes();
/// let device_id = tensor.device.index;
///
/// // Wrap without taking ownership
/// let external = unsafe {
///     ExternalDeviceMemory::new(tensor_ptr as *const u8, tensor_size, device_id as u64)
/// };
///
/// // Register with NIXL for RDMA
/// let registered = register_with_nixl(external, &agent, None)?;
/// ```
pub struct ExternalDeviceMemory {
    /// Raw pointer to externally-allocated GPU memory.
    ptr: *const u8,
    /// Size of the memory region in bytes.
    size: usize,
    /// CUDA device ID where this memory resides.
    device_id: u64,
}

// Safety: The external framework (e.g., vLLM) guarantees the memory remains valid
// for the lifetime of the KV cache. The pointer is only used for NIXL registration
// and transfer operations which are synchronized by the framework.
unsafe impl Send for ExternalDeviceMemory {}
unsafe impl Sync for ExternalDeviceMemory {}

impl ExternalDeviceMemory {
    /// Create a wrapper for external device memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `ptr` points to valid GPU memory on CUDA device `device_id`
    /// - The memory remains valid for the lifetime of this wrapper
    /// - The memory size is exactly `size` bytes
    /// - The external framework doesn't free the memory while this wrapper exists
    #[inline]
    pub unsafe fn new(ptr: *const u8, size: usize, device_id: u64) -> Self {
        Self {
            ptr,
            size,
            device_id,
        }
    }

    /// Get the raw pointer to the external memory.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get the CUDA device ID where this memory resides.
    #[inline]
    pub fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl fmt::Debug for ExternalDeviceMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalDeviceMemory")
            .field("ptr", &format_args!("{:p}", self.ptr))
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl MemoryDescriptor for ExternalDeviceMemory {
    #[inline]
    fn addr(&self) -> usize {
        self.ptr as usize
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id as u32)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        // External memory doesn't have a pre-existing NIXL descriptor
        // It will be registered and get one via NixlRegistered wrapper
        None
    }
}

impl NixlCompatible for ExternalDeviceMemory {
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
        (self.ptr, self.size, MemType::Vram, self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_device_memory_traits() {
        // Create with a dummy pointer (not actually valid GPU memory)
        let ptr = 0x1000 as *const u8;
        let size = 1024;
        let device_id = 0;

        let external = unsafe { ExternalDeviceMemory::new(ptr, size, device_id) };

        // Check MemoryDescriptor
        assert_eq!(external.addr(), 0x1000);
        assert_eq!(external.size(), 1024);
        assert_eq!(external.storage_kind(), StorageKind::Device(0));
        assert!(external.nixl_descriptor().is_none());

        // Check NixlCompatible
        let (p, s, mem_type, dev) = external.nixl_params();
        assert_eq!(p as usize, 0x1000);
        assert_eq!(s, 1024);
        assert_eq!(mem_type, MemType::Vram);
        assert_eq!(dev, 0);
    }
}
