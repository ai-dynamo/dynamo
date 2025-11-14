// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory storage.

use crate::block_manager::DeviceStorage as V1DeviceStorage;
use crate::block_manager::Storage as V1Storage;
use crate::block_manager::storage::cuda::DeviceStorageType as V1DeviceStorageType;

use super::{MemoryRegion, Result, StorageError, StorageKind};
use cudarc::driver::CudaContext;
use nixl_sys::NixlDescriptor;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Get or create a CUDA context for the given device.
fn cuda_context(device_id: u32) -> Result<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<u32, Arc<CudaContext>>>> = OnceLock::new();
    let mut map = CONTEXTS.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&device_id) {
        return Ok(existing.clone());
    }

    let ctx = CudaContext::new(device_id as usize)?;
    map.insert(device_id, ctx.clone());
    Ok(ctx)
}

/// CUDA device memory allocated via cudaMalloc.
#[derive(Debug)]
pub struct DeviceStorage {
    ctx: Arc<CudaContext>,
    ptr: u64,
    device_id: u32,
    len: usize,
    // TODO: This is a bit ugly. We need to translate our v1 device layout to v2.
    device_storage_type: V1DeviceStorageType,
}

unsafe impl Send for DeviceStorage {}
unsafe impl Sync for DeviceStorage {}

impl DeviceStorage {
    /// Allocate new device memory of the given size.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - CUDA device on which to allocate
    pub fn new(len: usize, device_id: u32) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let ctx = cuda_context(device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(len).map_err(StorageError::Cuda)? };

        Ok(Self {
            ctx,
            ptr,
            device_id,
            len,
            device_storage_type: V1DeviceStorageType::Owned,
        })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the CUDA device ID this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn from_v1(v1_storage: &V1DeviceStorage) -> Result<Self> {
        let device_id = v1_storage.device_id() as u32;
        let ctx = cuda_context(device_id)?;
        let ptr;
        unsafe {
            ptr = v1_storage.as_ptr() as u64;
        }

        let len = v1_storage.size();

        if !matches!(
            v1_storage.device_storage_type(),
            V1DeviceStorageType::Torch { .. }
        ) {
            return Err(StorageError::Unsupported(
                "Unable to convert owned device tensors.".into(),
            ));
        }

        Ok(Self {
            ctx,
            ptr,
            device_id,
            len,
            device_storage_type: v1_storage.device_storage_type().clone(),
        })
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        match self.device_storage_type {
            V1DeviceStorageType::Owned => {
                if let Err(e) = self.ctx.bind_to_thread() {
                    tracing::debug!("failed to bind CUDA context for free: {e}");
                }
                unsafe {
                    if let Err(e) = cudarc::driver::result::free_sync(self.ptr) {
                        tracing::debug!("failed to free device memory: {e}");
                    }
                }
            }
            V1DeviceStorageType::Torch { .. } => {} // Do nothing.
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
        StorageKind::Device(self.device_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Support for NIXL registration
impl super::registered::NixlCompatible for DeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id as u64,
        )
    }
}
