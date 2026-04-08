// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic pinned (page-locked) host memory storage.
//!
//! [`PinnedStorage`] allocates pinned host memory via a [`super::DeviceAllocator`]
//! implementation, supporting any hardware backend (CUDA, Level-Zero, HPU, …).
//!
//! Backend-specific details (CUDA write-combined probing, NUMA-aware placement,
//! Level-Zero host-shared allocation) are handled entirely by the
//! [`super::DeviceAllocator`] implementation passed at construction time.

use super::{DeviceAllocator, MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use std::any::Any;
use std::sync::Arc;

/// Pinned host memory allocated via a [`DeviceAllocator`].
#[derive(Debug)]
pub struct PinnedStorage {
    /// Host pointer to the pinned memory.
    ptr: usize,
    /// Size of the allocation in bytes.
    len: usize,
    /// Device allocator used for allocation and deallocation.
    ctx: Arc<dyn DeviceAllocator>,
}

unsafe impl Send for PinnedStorage {}
unsafe impl Sync for PinnedStorage {}

impl PinnedStorage {
    /// Allocate new pinned memory of the given size.
    ///
    /// The allocation is delegated to the provided [`DeviceAllocator`],
    /// which handles backend-specific details (CUDA write-combined,
    /// Level-Zero host alloc, NUMA placement, etc.).
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `ctx` - Device allocator (CUDA, Level-Zero, etc.)
    ///
    /// # Errors
    /// Returns an error if:
    /// - `len` is 0
    /// - The underlying allocator fails
    pub fn new(len: usize, ctx: Arc<dyn DeviceAllocator>) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let ptr = ctx.allocate_pinned(len)? as usize;

        Ok(Self { ptr, len, ctx })
    }

    /// Get a pointer to the underlying memory.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this storage is dropped.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Get a mutable pointer to the underlying memory.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this storage is dropped
    /// and that there are no other references to this memory.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.free_pinned(self.ptr as u64) {
            tracing::debug!("failed to free pinned memory: {e}");
        }
    }
}

impl MemoryDescriptor for PinnedStorage {
    fn addr(&self) -> usize {
        self.ptr
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Pinned
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for PinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        let ptr = unsafe { self.as_ptr() };
        (ptr, self.len, nixl_sys::MemType::Dram, 0)
    }
}

impl actions::Memset for PinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::OperationFailed("memset: offset overflow".into()))?;
        if end > self.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = (self.ptr as *mut u8).add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}
