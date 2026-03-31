// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-agnostic device memory storage for v2.
//!
//! Unlike `dynamo-memory::DeviceStorage` which is CUDA-only, this implementation
//! supports multiple backends (CUDA, HPU, XPU) via the v2 device abstraction layer.

use super::{MemoryRegion, Result, StorageError, StorageKind, NixlDescriptor, NixlCompatible};
use crate::block_manager::v2::device::{DeviceBackend, DeviceContext};
use std::any::Any;
use std::sync::Arc;

/// Multi-backend device memory allocated via the v2 device abstraction layer.
#[derive(Debug)]
pub struct DeviceStorage {
    /// Device context used for allocation and deallocation.
    ctx: Arc<DeviceContext>,
    /// Device pointer to the allocated memory.
    ptr: u64,
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
    /// * `ctx` - Device context (CUDA/HPU/XPU)
    pub fn new(len: usize, ctx: Arc<DeviceContext>) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let ptr = ctx.allocate_device(len).map_err(|e| {
            StorageError::AllocationFailed(format!("device allocation failed: {}", e))
        })?;

        Ok(Self { ctx, ptr, len })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the device ID this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.ctx.device_id()
    }

    /// Get the device backend.
    pub fn backend(&self) -> DeviceBackend {
        self.ctx.backend()
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.free_device(self.ptr) {
            tracing::debug!("failed to free device memory: {e}");
        }
    }
}

impl MemoryRegion for DeviceStorage {
    fn addr(&self) -> usize {
        self.device_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl NixlCompatible for DeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id() as u64,
        )
    }
}
