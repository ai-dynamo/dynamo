// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pinned host memory storage supporting CUDA and Ze backends.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};
use super::device::DeviceContext;
use cudarc::driver::CudaContext;
use std::any::Any;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Backend Type
// ---------------------------------------------------------------------------

/// Backend type identifier for storage operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CUDA backend (NVIDIA GPUs).
    Cuda,
    /// Level Zero backend (Intel GPUs).
    Ze,
}

// ---------------------------------------------------------------------------
// StorageBackendOps – single trait for backend-specific alloc/free
// ---------------------------------------------------------------------------

/// Backend operations for pinned and device memory allocation.
///
/// This is the single definition of this trait – the llm layer imports and
/// reuses it rather than defining its own copy.
pub trait StorageBackendOps: Send + Sync {
    /// Allocate pinned host memory.
    ///
    /// # Safety
    /// Caller must ensure proper context binding if required by the backend.
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8>;

    /// Free pinned host memory.
    ///
    /// # Safety
    /// Caller must ensure ptr was allocated by this backend and size matches.
    unsafe fn free_pinned(&self, ptr: u64, size: usize) -> Result<()>;

    /// Allocate device memory. Returns `(ptr, device_id, optional_metadata)`.
    ///
    /// The metadata (`Option<Box<dyn Any + Send + Sync>>`) allows backends to
    /// keep ownership handles alive (e.g. Ze `DeviceBuffer` whose `Drop` frees
    /// the memory). CUDA returns `None`.
    ///
    /// # Safety
    /// Caller must ensure proper context binding if required by the backend.
    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, Option<Box<dyn Any + Send + Sync>>)>;

    /// Free device memory.
    ///
    /// # Safety
    /// Caller must ensure ptr was allocated by this backend.
    unsafe fn free_device(&self, ptr: u64) -> Result<()>;

    /// Get the device ID for this backend context.
    fn device_id(&self) -> u32;

    /// Get the backend type (CUDA or Ze).
    fn backend_type(&self) -> BackendType;

    /// Downcast to concrete type for backend-specific operations.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Create device storage from a torch tensor descriptor.
    ///
    /// The caller extracts tensor metadata into a [`TorchTensorDescriptor`],
    /// and this method validates that the tensor matches the backend
    /// (device type and device ID).
    ///
    /// Returns `(ptr, size)` on success.
    ///
    /// # Errors
    /// Returns error if:
    /// - Tensor device type doesn't match backend type
    /// - Tensor device ID doesn't match context device ID (where applicable)
    fn new_from_torch(&self, desc: &TorchTensorDescriptor) -> Result<(u64, usize)>;
}

/// Descriptor carrying extracted torch tensor metadata across crate boundaries.
///
/// Higher-level crates (e.g. llm) populate this from their `TorchTensor` trait
/// objects and pass it to [`StorageBackendOps::new_from_torch`] for validation.
#[derive(Debug, Clone)]
pub struct TorchTensorDescriptor {
    /// Device pointer to tensor data.
    pub data_ptr: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// True if tensor is on a CUDA device.
    pub is_cuda: bool,
    /// True if tensor is on an XPU/Ze device.
    pub is_xpu: bool,
    /// Device ID (CUDA device ordinal, or 0 for XPU).
    pub device_id: u32,
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

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
pub(crate) unsafe fn malloc_host_prefer_writecombined(size: usize) -> Result<*mut u8> {
    let flags = if *USE_WRITE_COMBINED {
        cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED
    } else {
        cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP
    };
    unsafe { cudarc::driver::result::malloc_host(size, flags) }
        .map(|ptr| ptr as *mut u8)
        .map_err(StorageError::Cuda)
}

impl StorageBackendOps for Arc<CudaContext> {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8> {
        self.bind_to_thread().map_err(StorageError::Cuda)?;
        unsafe { malloc_host_prefer_writecombined(size) }
    }

    unsafe fn free_pinned(&self, ptr: u64, _size: usize) -> Result<()> {
        unsafe { cudarc::driver::result::free_host(ptr as _) }.map_err(StorageError::Cuda)
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, Option<Box<dyn Any + Send + Sync>>)> {
        self.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr =
            unsafe { cudarc::driver::result::malloc_sync(size) }.map_err(StorageError::Cuda)?;
        Ok((ptr, self.cu_device() as u32, None))
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<()> {
        unsafe { cudarc::driver::result::free_sync(ptr as _) }.map_err(StorageError::Cuda)
    }

    fn device_id(&self) -> u32 {
        self.cu_device() as u32
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_from_torch(&self, desc: &TorchTensorDescriptor) -> Result<(u64, usize)> {
        if !desc.is_cuda {
            return Err(StorageError::OperationFailed(
                "Tensor is not CUDA!".into(),
            ));
        }
        if desc.device_id != self.cu_device() as u32 {
            return Err(StorageError::OperationFailed(
                "Tensor is not on the same device as the context!".into(),
            ));
        }
        Ok((desc.data_ptr, desc.size_bytes))
    }
}

// ---------------------------------------------------------------------------
// PinnedStorage
// ---------------------------------------------------------------------------

/// Pinned host memory supporting CUDA and Ze backends.
///
/// For CUDA: allocated via `cudaHostAlloc` (page-locked, optionally write-combined).
/// For Ze: allocated via system allocator with 64-byte alignment.
#[derive(Debug)]
pub struct PinnedStorage {
    /// Host pointer to the pinned memory.
    ptr: usize,
    /// Size of the allocation in bytes.
    len: usize,
    /// Device context used for allocation and deallocation.
    ctx: DeviceContext,
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

        Ok(Self { ptr, len, ctx: DeviceContext::new(ctx) })
    }

    /// Allocate pinned host memory using an existing [`DeviceContext`].
    ///
    /// Dispatches to the appropriate backend based on the context type:
    /// - CUDA: delegates to [`new_for_device`](Self::new_for_device)
    /// - Ze: allocates via the Ze backend directly
    pub fn new_with_context(len: usize, ctx: &DeviceContext) -> Result<Self> {
        match ctx.backend.backend_type() {
            BackendType::Cuda => {
                Self::new_for_device(len, Some(ctx.backend.device_id()))
            }
            BackendType::Ze => {
                if len == 0 {
                    return Err(StorageError::AllocationFailed(
                        "zero-sized allocations are not supported".into(),
                    ));
                }
                let raw = unsafe { ctx.backend.alloc_pinned(len)? };
                assert!(!raw.is_null(), "Failed to allocate pinned memory");
                assert!(raw.is_aligned(), "Pinned memory is not aligned");
                assert!(len < isize::MAX as usize);
                Ok(Self {
                    ptr: raw as usize,
                    len,
                    ctx: ctx.clone(),
                })
            }
        }
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

    /// Get a reference to the device context.
    pub fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.ctx.backend.free_pinned(self.ptr as u64, self.len) {
                tracing::debug!("failed to free pinned memory: {e}");
            }
        }
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
