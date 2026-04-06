// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Level Zero host-pinned memory storage.
//!
//! Allocates DMA-accessible host memory via `zeMemAllocHost`, which is the
//! Level Zero equivalent of CUDA's `cuMemHostAlloc`. This memory is pinned
//! and accessible by the XPU device for efficient H2D/D2H transfers.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use syclrc::level_zero::ze::safe::ZeDevice;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Get or create a Level Zero device for the given ordinal.
fn ze_device(ordinal: u32) -> Result<Arc<ZeDevice>> {
    static DEVICES: OnceLock<Mutex<HashMap<u32, Arc<ZeDevice>>>> = OnceLock::new();
    let mut map = DEVICES.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&ordinal) {
        return Ok(existing.clone());
    }

    let dev = ZeDevice::new(ordinal as usize).map_err(|e| {
        StorageError::AllocationFailed(format!("Level Zero device {ordinal}: {e}"))
    })?;
    map.insert(ordinal, dev.clone());
    Ok(dev)
}

/// Level Zero host-pinned memory allocated via `zeMemAllocHost`.
///
/// This is the XPU equivalent of [`super::PinnedStorage`] (CUDA).
/// The memory is accessible by the host CPU and is DMA-mapped for
/// efficient transfers to/from XPU devices.
/// Memory is freed via `zeMemFree` on drop.
pub struct ZeHostStorage {
    /// Level Zero device used for context (allocation and deallocation).
    dev: Arc<ZeDevice>,
    /// Host pointer to the pinned memory.
    ptr: *mut std::ffi::c_void,
    /// Size of the allocation in bytes.
    len: usize,
}

impl std::fmt::Debug for ZeHostStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeHostStorage")
            .field("ptr", &(self.ptr as u64))
            .field("len", &self.len)
            .finish()
    }
}

// SAFETY: The host pointer is allocated via zeMemAllocHost and is accessible
// from any thread. Level Zero APIs are thread-safe for memory operations.
unsafe impl Send for ZeHostStorage {}
unsafe impl Sync for ZeHostStorage {}

impl ZeHostStorage {
    /// Allocate new host-pinned memory of the given size.
    ///
    /// Uses `zeMemAllocHost` to allocate DMA-accessible host memory
    /// associated with the given XPU device's context.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - Level Zero device ordinal (for context)
    pub fn new(len: usize, device_id: u32) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let dev = ze_device(device_id)?;
        let ptr = unsafe {
            syclrc::level_zero::ze::result::memory::alloc_host(
                dev.ze_context(),
                len,
                64, // alignment
            )
        }
        .map_err(|e| {
            StorageError::AllocationFailed(format!(
                "zeMemAllocHost {len} bytes on device {device_id} context: {e}"
            ))
        })?;

        Ok(Self { dev, ptr, len })
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

impl Drop for ZeHostStorage {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Err(e) = syclrc::level_zero::ze::result::memory::free(
                    self.dev.ze_context(),
                    self.ptr,
                ) {
                    eprintln!("zeMemFree failed for ZeHostStorage: {e}");
                }
            }
        }
    }
}

impl MemoryDescriptor for ZeHostStorage {
    fn addr(&self) -> usize {
        unsafe { self.as_ptr() as usize }
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        // Returns Pinned — from the transfer engine's perspective, this is
        // DMA-accessible host memory, functionally identical to CUDA pinned.
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
impl super::nixl::NixlCompatible for ZeHostStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        let ptr = unsafe { self.as_ptr() };
        (ptr, self.len, nixl_sys::MemType::Dram, 0)
    }
}

impl actions::Memset for ZeHostStorage {
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
