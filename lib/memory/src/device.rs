// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic device memory storage.
//!
//! [`DeviceStorage`] allocates device memory via an [`super::DeviceAllocator`]
//! implementation, supporting any hardware backend (CUDA, Level-Zero, HPU, …).

use super::{DeviceAllocator, MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use std::any::Any;
use std::sync::Arc;

/// Device memory allocated via a [`DeviceAllocator`].
#[derive(Debug)]
pub struct DeviceStorage {
    /// Device allocator used for allocation and deallocation.
    ctx: Arc<dyn DeviceAllocator>,
    /// Device pointer to the allocated memory.
    ptr: u64,
    /// Device ID where memory is allocated.
    device_id: u32,
    /// Size of the allocation in bytes.
    len: usize,
}

unsafe impl Send for DeviceStorage {}
unsafe impl Sync for DeviceStorage {}

impl DeviceStorage {
    /// Allocate new device memory of the given size.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `ctx` - Device allocator (CUDA, Level-Zero, etc.)
    pub fn new(len: usize, ctx: Arc<dyn DeviceAllocator>) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let device_id = ctx.device_id();
        let ptr = ctx.allocate_device(len)?;

        Ok(Self {
            ctx,
            ptr,
            device_id,
            len,
        })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the device ID this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.free_device(self.ptr) {
            tracing::debug!("failed to free device memory: {e}");
        }
    }
}

impl MemoryDescriptor for DeviceStorage {
    fn addr(&self) -> usize {
        self.device_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for DeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id as u64,
        )
    }
}
