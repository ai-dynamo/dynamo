// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-agnostic pinned host memory storage for v2.
//!
//! Unlike `dynamo-memory::PinnedStorage` which is CUDA-only, this implementation
//! supports multiple backends (CUDA, HPU, XPU) via the v2 device abstraction layer.

use super::{MemoryRegion, NixlCompatible, NixlDescriptor, Result, StorageError, StorageKind};
use crate::block_manager::v2::device::DeviceContext;
use std::any::Any;
use std::sync::Arc;

/// Multi-backend pinned host memory allocated via the v2 device abstraction layer.
#[derive(Debug)]
pub struct PinnedStorage {
    /// Device context used for allocation and deallocation.
    ctx: Arc<DeviceContext>,
    /// Host pointer to the pinned memory.
    ptr: u64,
    /// Size of the allocation in bytes.
    len: usize,
}

unsafe impl Send for PinnedStorage {}
unsafe impl Sync for PinnedStorage {}

impl PinnedStorage {
    /// Allocate new pinned memory of the given size.
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

        let ptr = ctx.allocate_pinned(len).map_err(|e| {
            StorageError::AllocationFailed(format!("pinned allocation failed: {}", e))
        })?;

        Ok(Self { ctx, ptr, len })
    }

    /// Get the host pointer value.
    pub fn host_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the device ID this pinned memory is associated with.
    pub fn device_id(&self) -> u32 {
        self.ctx.device_id()
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.free_pinned(self.ptr) {
            tracing::debug!("failed to free pinned memory: {e}");
        }
    }
}

impl MemoryRegion for PinnedStorage {
    fn addr(&self) -> usize {
        self.host_ptr() as usize
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
impl NixlCompatible for PinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Dram,
            self.device_id() as u64,
        )
    }
}
