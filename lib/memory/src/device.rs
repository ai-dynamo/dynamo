// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use super::pinned::StorageBackendOps;
use cudarc::driver::CudaContext;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Unified device context backed by a type-erased [`StorageBackendOps`].
///
/// Construct via [`DeviceContext::new`] with any backend that implements
/// `StorageBackendOps` (e.g. `Arc<CudaContext>`, `Arc<ZeContext>`).
#[derive(Clone)]
pub struct DeviceContext {
    /// The underlying backend for memory operations.
    pub backend: Arc<dyn StorageBackendOps>,
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceContext(device={})", self.backend.device_id())
    }
}

impl DeviceContext {
    /// Create a new device context from any backend implementing
    /// [`StorageBackendOps`].
    pub fn new<B: StorageBackendOps + 'static>(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Get the backend type (CUDA or Ze).
    pub fn backend_type(&self) -> super::pinned::BackendType {
        self.backend.backend_type()
    }

    /// Check if this context uses CUDA backend.
    pub fn is_cuda(&self) -> bool {
        self.backend.backend_type() == super::pinned::BackendType::Cuda
    }

    /// Check if this context uses Ze backend.
    pub fn is_ze(&self) -> bool {
        self.backend.backend_type() == super::pinned::BackendType::Ze
    }
}

/// Get or create a CUDA context for the given device.
pub(crate) fn cuda_context(device_id: u32) -> Result<Arc<CudaContext>> {
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
    /// CUDA context used for allocation and deallocation.
    ctx: Arc<CudaContext>,
    /// Device pointer to the allocated memory.
    ptr: u64,
    /// CUDA device ID where memory is allocated.
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
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::debug!("failed to bind CUDA context for free: {e}");
        }
        unsafe {
            if let Err(e) = cudarc::driver::result::free_sync(self.ptr) {
                tracing::debug!("failed to free device memory: {e}");
            }
        };
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
