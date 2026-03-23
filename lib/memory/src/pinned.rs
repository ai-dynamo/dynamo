// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA pinned host memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use cudarc::driver::CudaContext;
use std::any::Any;
use std::sync::Arc;

/// Whether to use write-combined pinned allocations.
///
/// Probed once at first use: returns `false` if `DYN_KVBM_DISABLE_WRITE_COMBINED`
/// is set, or if a test allocation reveals the hardware does not support it
/// (e.g. Grace Hopper / Blackwell with NVLink-C2C). Must be accessed only after
/// a CUDA context has been bound to the current thread.
static USE_WRITE_COMBINED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    if dynamo_config::env_is_truthy("DYN_KVBM_DISABLE_WRITE_COMBINED") {
        tracing::debug!("DYN_KVBM_DISABLE_WRITE_COMBINED set; write-combined disabled");
        return false;
    }
    // Probe hardware support with a 1-byte test allocation.
    // SAFETY: called from an allocation path that has already bound a CUDA context.
    unsafe {
        match cudarc::driver::result::malloc_host(
            1,
            cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        ) {
            Ok(ptr) => {
                let _ = cudarc::driver::result::free_host(ptr);
                true
            }
            Err(_) => {
                tracing::debug!(
                    "Write-combined memory not supported on this system; \
                     will use regular pinned memory"
                );
                false
            }
        }
    }
});

/// Allocates pinned host memory, using write-combined if [`USE_WRITE_COMBINED`]
/// allows it, otherwise falling back to `CU_MEMHOSTALLOC_DEVICEMAP`.
///
/// # Safety
/// Caller must ensure a valid CUDA context is bound to the current thread.
unsafe fn malloc_host_prefer_writecombined(size: usize) -> Result<*mut u8> {
    if *USE_WRITE_COMBINED {
        // SAFETY: caller guarantees a valid CUDA context is bound to the current thread
        unsafe {
            cudarc::driver::result::malloc_host(
                size,
                cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            )
        }
        .map(|ptr| ptr as *mut u8)
        .map_err(StorageError::Cuda)
    } else {
        // SAFETY: caller guarantees a valid CUDA context is bound to the current thread
        unsafe {
            cudarc::driver::result::malloc_host(
                size,
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        }
        .map(|ptr| ptr as *mut u8)
        .map_err(StorageError::Cuda)
    }
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
    /// When `device_id` is `Some`, NUMA-aware allocation is attempted by default:
    /// a worker thread pinned to the GPU's NUMA node performs the allocation,
    /// ensuring optimal memory placement via first-touch policy. If the GPU's
    /// NUMA node cannot be determined, allocation falls back to the direct path.
    /// Set `DYN_MEMORY_DISABLE_NUMA=1` to skip NUMA optimization entirely.
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
        let ctx = crate::device::cuda_context(gpu_id)?;

        // Try NUMA-aware allocation unless explicitly disabled
        #[cfg(target_os = "linux")]
        let numa_ptr = if let Some(gpu_id) = device_id {
            if !super::numa::is_numa_disabled() {
                match super::numa::worker_pool::NumaWorkerPool::global()
                    .allocate_pinned_for_gpu(len, gpu_id)
                {
                    Ok(Some(ptr)) => {
                        tracing::debug!(
                            "Using NUMA-aware allocation for {} bytes on GPU {}",
                            len,
                            gpu_id
                        );
                        Some(ptr as usize)
                    }
                    Ok(None) => None, // NUMA node unknown, fall through
                    Err(e) => return Err(StorageError::AllocationFailed(e)),
                }
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(target_os = "linux"))]
        let numa_ptr: Option<usize> = None;

        let ptr = if let Some(ptr) = numa_ptr {
            ptr
        } else {
            unsafe {
                ctx.bind_to_thread().map_err(StorageError::Cuda)?;

                let ptr = malloc_host_prefer_writecombined(len)?;

                assert!(!ptr.is_null(), "Failed to allocate pinned memory");
                assert!(ptr.is_aligned(), "Pinned memory is not aligned");
                assert!(len < isize::MAX as usize);

                ptr as usize
            }
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

    /// Get a reference to the CUDA context used for this allocation.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
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
