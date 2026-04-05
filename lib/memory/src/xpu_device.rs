// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intel XPU (Level Zero) device memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
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
    // ZeDevice::new() already returns Arc<ZeDevice>
    map.insert(ordinal, dev.clone());
    Ok(dev)
}

/// Intel XPU device memory allocated via Level Zero (`zeMemAllocDevice`).
///
/// This is the XPU equivalent of [`super::DeviceStorage`] (CUDA).
/// Memory is freed via `zeMemFree` on drop.
pub struct XpuDeviceStorage {
    /// Level Zero device used for allocation and deallocation.
    dev: Arc<ZeDevice>,
    /// Device pointer to the allocated memory.
    ptr: *mut std::ffi::c_void,
    /// Level Zero device ordinal where memory is allocated.
    device_id: u32,
    /// Size of the allocation in bytes.
    len: usize,
}

impl std::fmt::Debug for XpuDeviceStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XpuDeviceStorage")
            .field("ptr", &(self.ptr as u64))
            .field("device_id", &self.device_id)
            .field("len", &self.len)
            .finish()
    }
}

// SAFETY: The device pointer is only accessed through Level Zero APIs
// which are thread-safe with respect to memory operations.
unsafe impl Send for XpuDeviceStorage {}
unsafe impl Sync for XpuDeviceStorage {}

impl XpuDeviceStorage {
    /// Allocate new device memory of the given size.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - Level Zero device ordinal on which to allocate
    pub fn new(len: usize, device_id: u32) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let dev = ze_device(device_id)?;
        let ptr = unsafe {
            syclrc::level_zero::ze::result::memory::alloc_device(
                dev.ze_context(),
                len,
                64, // alignment
                dev.ze_device(),
            )
        }
        .map_err(|e| {
            StorageError::AllocationFailed(format!(
                "zeMemAllocDevice {len} bytes on device {device_id}: {e}"
            ))
        })?;

        Ok(Self {
            dev,
            ptr,
            device_id,
            len,
        })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// Get the Level Zero device ordinal this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

impl Drop for XpuDeviceStorage {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Err(e) = syclrc::level_zero::ze::result::memory::free(
                    self.dev.ze_context(),
                    self.ptr,
                ) {
                    eprintln!("zeMemFree failed for XpuDeviceStorage: {e}");
                }
            }
        }
    }
}

impl MemoryDescriptor for XpuDeviceStorage {
    fn addr(&self) -> usize {
        self.device_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::XpuDevice(self.device_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for XpuDeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id as u64,
        )
    }
}
