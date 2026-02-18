// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA pinned host memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use cudarc::driver::CudaContext;
use cudarc::driver::sys;
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

/// CUDA pinned host memory allocated via cudaHostAlloc.
#[derive(Debug)]
pub struct PinnedStorage {
    /// Host pointer to the pinned memory.
    ptr: usize,
    /// Size of the allocation in bytes.
    len: usize,
    /// CUDA context used for allocation and deallocation.
    ctx: Arc<CudaContext>,
}

unsafe impl Send for PinnedStorage {}
unsafe impl Sync for PinnedStorage {}

impl PinnedStorage {
    /// Allocate new pinned memory of the given size.
    ///
    /// This is a convenience method that calls `new_for_device(len, None)`.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    pub fn new(len: usize) -> Result<Self> {
        Self::new_for_device(len, None)
    }

    /// Allocate pinned memory, optionally NUMA-aware for a specific GPU.
    ///
    /// When `device_id` is `Some`, the allocation is performed on a worker thread
    /// pinned to the GPU's NUMA node, ensuring optimal memory placement via
    /// first-touch policy, However, NUMA is only used if enabled via the
    /// `DYN_KVBM_ENABLE_NUMA=1` environment variable.
    ///
    /// When `device_id` is `None`, a direct allocation is performed on device 0.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - If Some, use NUMA-aware allocation on the GPU's NUMA node
    ///
    /// # Errors
    /// Returns an error if:
    /// - `len` is 0
    /// - CUDA context creation fails
    /// - Memory allocation fails
    pub fn new_for_device(len: usize, device_id: Option<u32>) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        let gpu_id = device_id.unwrap_or(0);
        let ctx = cuda_context(gpu_id)?;

        let ptr = match device_id {
            #[cfg(target_os = "linux")]
            Some(gpu_id) if super::numa::is_numa_enabled() => {
                tracing::debug!(
                    "Using NUMA-aware allocation for {} bytes on GPU {}",
                    len,
                    gpu_id
                );
                super::numa::worker_pool::NumaWorkerPool::global()
                    .allocate_pinned_for_gpu(len, gpu_id)
                    .map_err(StorageError::AllocationFailed)? as usize
            }
            _ => unsafe {
                ctx.bind_to_thread().map_err(StorageError::Cuda)?;

                let ptr = cudarc::driver::result::malloc_host(len, sys::CU_MEMHOSTALLOC_DEVICEMAP)
                    .map_err(StorageError::Cuda)?;

                let ptr = ptr as *mut u8;
                assert!(!ptr.is_null(), "Failed to allocate pinned memory");
                assert!(ptr.is_aligned(), "Pinned memory is not aligned");
                assert!(len < isize::MAX as usize);

                ptr as usize
            },
        };

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
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::debug!("failed to bind CUDA context for free: {e}");
        }
        unsafe {
            if let Err(e) = cudarc::driver::result::free_host(self.ptr as _) {
                tracing::debug!("failed to free pinned memory: {e}");
            }
        };
    }
}

impl MemoryDescriptor for PinnedStorage {
    fn addr(&self) -> usize {
        unsafe { self.as_ptr() as usize }
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
