// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # CUDA Storage Support
//!
//! This module provides CUDA context management and allocation helpers used by
//! storage implementations in the parent [`storage`](super) module.
//!
//! The module provides a singleton [`Cuda`] type for managing CUDA contexts:
//! - Thread-safe context management
//! - Lazy initialization of device contexts
//! - Automatic cleanup of resources

use super::StorageError;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::{CudaContext, sys};

/// Allocates pinned host memory, preferring write-combined if supported.
///
/// Write-combined (WC) memory is optimal for PCIe DMA transfers but may not be
/// supported on systems with cache-coherent CPU-GPU interconnects (e.g., Grace
/// Hopper/Blackwell with NVLink-C2C). This function tries WC first and falls
/// back to regular pinned memory if not supported.
///
/// # Safety
///
/// Caller must ensure a valid CUDA context is bound to the current thread.
pub(crate) unsafe fn malloc_host_prefer_writecombined(
    size: usize,
) -> Result<*mut u8, StorageError> {
    // First, try write-combined allocation (optimal for PCIe systems)
    // SAFETY: Caller guarantees a valid CUDA context is bound to the current thread
    match unsafe { cudarc::driver::result::malloc_host(size, sys::CU_MEMHOSTALLOC_WRITECOMBINED) } {
        Ok(ptr) => Ok(ptr as *mut u8),
        Err(_) => {
            // Write-combined not supported (e.g., Grace Hopper/Blackwell),
            // fall back to regular pinned memory
            tracing::debug!("Write-combined memory not supported, using regular pinned memory");
            // SAFETY: Same as above - caller guarantees valid CUDA context
            unsafe { cudarc::driver::result::malloc_host(size, 0) }
                .map(|ptr| ptr as *mut u8)
                .map_err(StorageError::Cuda)
        }
    }
}

/// Singleton for managing CUDA contexts.
pub struct Cuda {
    contexts: HashMap<usize, Arc<CudaContext>>,
}

impl Cuda {
    // Private constructor
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Get a CUDA context for a specific device_id.
    /// If the context does not exist, it will return None.
    ///
    /// This will not lazily instantiate a context for a device. Use
    /// [Cuda::device_or_create]
    pub fn device(device_id: usize) -> Option<Arc<CudaContext>> {
        Cuda::instance()
            .lock()
            .unwrap()
            .get_existing_context(device_id)
    }

    /// Get or initialize a CUDA context for a specific device_id.
    /// If the context does not exist, it will be created or fail.
    ///
    /// This will lazily instantiate a context for a device. Use
    /// [CudaContextManager::device] to get an existing context.
    pub fn device_or_create(device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        Cuda::instance().lock().unwrap().get_context(device_id)
    }

    /// Check if a CUDA context exists for a specific device_id.
    pub fn is_initialized(device_id: usize) -> bool {
        Cuda::instance().lock().unwrap().has_context(device_id)
    }

    // Get the singleton instance
    fn instance() -> &'static Mutex<Cuda> {
        static INSTANCE: OnceLock<Mutex<Cuda>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(Cuda::new()))
    }

    // Get or create a CUDA context for a specific device
    fn get_context(&mut self, device_id: usize) -> Result<Arc<CudaContext>, StorageError> {
        // Check if we already have a context for this device
        if let Some(ctx) = self.contexts.get(&device_id) {
            return Ok(ctx.clone());
        }

        // Create a new context for this device
        let ctx = CudaContext::new(device_id)?;

        // Store the context
        self.contexts.insert(device_id, ctx.clone());

        Ok(ctx)
    }

    // Get a context if it exists, but don't create one
    pub fn get_existing_context(&self, device_id: usize) -> Option<Arc<CudaContext>> {
        self.contexts.get(&device_id).cloned()
    }

    // Check if a context exists for a device
    pub fn has_context(&self, device_id: usize) -> bool {
        self.contexts.contains_key(&device_id)
    }
}

impl super::StorageBackendOps for std::sync::Arc<CudaContext> {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, super::StorageError> {
        self.bind_to_thread()
            .map_err(super::StorageError::Cuda)?;

        // Try NUMA-aware allocation if enabled, otherwise use direct allocation.
        if crate::block_manager::numa_allocator::is_numa_enabled() {
            let device_id = self.cu_device() as u32;
            match crate::block_manager::numa_allocator::worker_pool::NumaWorkerPool::global()
                .allocate_pinned_for_gpu(size, device_id)
            {
                Ok(ptr) => return Ok(ptr),
                Err(e) => {
                    tracing::warn!("NUMA allocation failed: {}, using direct allocation", e);
                }
            }
        }

        unsafe { malloc_host_prefer_writecombined(size) }
    }

    unsafe fn free_pinned(&self, ptr: u64, _size: usize) -> Result<(), super::StorageError> {
        unsafe { cudarc::driver::result::free_host(ptr as _).map_err(super::StorageError::Cuda) }
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, super::DeviceStorageType), super::StorageError> {
        self.bind_to_thread()
            .map_err(super::StorageError::Cuda)?;

        let ptr = unsafe { cudarc::driver::result::malloc_sync(size).map_err(super::StorageError::Cuda)? };

        Ok((
            ptr,
            self.cu_device() as u32,
            super::DeviceStorageType::Owned {
                _ze_device_buffer: None,
            },
        ))
    }

    unsafe fn free_device(&self, ptr: u64) -> Result<(), super::StorageError> {
        unsafe { cudarc::driver::result::free_sync(ptr as _).map_err(super::StorageError::Cuda) }
    }

    fn device_id(&self) -> u32 {
        self.cu_device() as u32
    }
}

/// CUDA-specific DeviceStorage methods
impl super::DeviceStorage {
    /// Create a CUDA device storage from a torch tensor.
    pub fn new_from_torch_cuda(
        ctx: &Arc<CudaContext>,
        tensor: Arc<dyn super::torch::TorchTensor>,
    ) -> Result<Self, super::StorageError> {
        use super::torch::{TorchDevice, is_cuda};

        if !is_cuda(tensor.as_ref()) {
            return Err(super::StorageError::InvalidConfig("Tensor is not CUDA!".into()));
        }

        let TorchDevice::Cuda(device_id) = tensor.device() else {
            unreachable!("is_cuda() returned true but device is not CUDA");
        };

        if device_id != ctx.cu_device() as usize {
            return Err(super::StorageError::InvalidConfig(
                "Tensor is not on the same device as the context!".into(),
            ));
        }

        Ok(Self {
            ptr: tensor.data_ptr(),
            size: tensor.size_bytes(),
            ctx: super::DeviceContext::Cuda(ctx.clone()),
            handles: super::RegistrationHandles::new(),
            storage_type: super::DeviceStorageType::Torch { _tensor: tensor },
        })
    }
}
