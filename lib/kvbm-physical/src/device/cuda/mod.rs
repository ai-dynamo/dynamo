// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA backend implementation.
//!
//! Wraps cudarc types with the device abstraction traits.

use crate::device::traits::*;
use anyhow::{Result, Context as _};
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::CUresult;
use cudarc::driver::DriverError;
use std::sync::Arc;
use dynamo_memory::CudaMemPool;

/// Whether to use write-combined pinned allocations.
///
/// Probed once at first use: returns `false` if `DYN_KVBM_DISABLE_WRITE_COMBINED`
/// is set, or if a test allocation reveals the hardware does not support it
/// (e.g. Grace Hopper / Blackwell with NVLink-C2C). Must be accessed only after
/// a CUDA context has been bound to the current thread.
static USE_WRITE_COMBINED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    if dynamo_memory::env_is_truthy("DYN_KVBM_DISABLE_WRITE_COMBINED") {
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
        .map_err(|e| anyhow::anyhow!("CUDA pinned host allocation failed: {:?}", e))
    } else {
        // SAFETY: caller guarantees a valid CUDA context is bound to the current thread
        unsafe {
            cudarc::driver::result::malloc_host(
                size,
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        }
        .map(|ptr| ptr as *mut u8)
        .map_err(|e| anyhow::anyhow!("CUDA pinned host allocation failed: {:?}", e))
    }
}

/// CUDA device context wrapping cudarc::CudaContext.
#[derive(Debug)]
pub struct CudaContext {
    context: Arc<cudarc::driver::CudaContext>,
    device_id: u32,
}

impl CudaContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(device_id as usize)
            .with_context(|| format!("Failed to create CUDA context for device {}", device_id))?;
        Ok(Self { context, device_id })
    }

    /// Create from an existing CudaContext (for compatibility with existing code).
    pub fn from_context(context: Arc<cudarc::driver::CudaContext>, device_id: u32) -> Self {
        Self { context, device_id }
    }

    /// Get the underlying cudarc context.
    pub fn inner(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.context
    }
}

impl DeviceContextOps for CudaContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self, _hint: EngineHint) -> Result<Box<dyn DeviceStreamOps>> {
        let stream = self.context.new_stream()
            .context("Failed to create CUDA stream")?;
        Ok(Box::new(CudaStreamWrapper { stream }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        self.context.bind_to_thread()?;
        let ptr = unsafe {
            cuda_result::malloc_sync(size)
                .context("Failed to allocate device memory")?
        };
        Ok(ptr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        self.context.bind_to_thread()?;
        unsafe {
            cuda_result::free_sync(ptr)
                .context("Failed to free device memory")?;
        }
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        // Try NUMA-aware allocation on Linux unless explicitly disabled
        #[cfg(target_os = "linux")]
        {
            if dynamo_memory::numa::is_numa_enabled() {
                match dynamo_memory::numa::worker_pool::NumaWorkerPool::global()
                    .allocate_pinned_for_gpu(size, self.device_id)
                {
                    Ok(Some(ptr)) => {
                        tracing::debug!(
                            "Using NUMA-aware allocation for {} bytes on GPU {}",
                            size, self.device_id
                        );
                        return Ok(ptr as u64);
                    }
                    Ok(None) => {} // NUMA node unknown, fall through
                    Err(e) => return Err(anyhow::anyhow!(
                        "NUMA-aware pinned allocation failed: {}", e
                    )),
                }
            }
        }

        // Fall back to write-combined or device-mapped pinned memory.
        // Bind CUDA context only for the fallback path (NUMA workers
        // manage their own contexts on pinned NUMA threads).
        self.context.bind_to_thread()
            .context("Failed to bind CUDA context for pinned allocation")?;

        let ptr = unsafe { malloc_host_prefer_writecombined(size)? };

        assert!(!ptr.is_null(), "Failed to allocate pinned memory");
        assert!(ptr.is_aligned(), "Pinned memory is not aligned");
        assert!(size < isize::MAX as usize);

        Ok(ptr as u64)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        unsafe {
            cuda_result::free_host(ptr as *mut std::ffi::c_void)
                .context("Failed to free pinned host memory")?;
        }
        Ok(())
    }

    fn bind_to_thread(&self) -> Result<()> {
        self.context.bind_to_thread()
            .context("Failed to bind CUDA context to thread")
    }

    unsafe fn disable_event_tracking(&self) -> Result<()> {
        unsafe { self.context.disable_event_tracking(); }
        Ok(())
    }

    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>> {
        let mut builder = CudaMemPool::builder(self.context.clone(), reserve_size);
        if let Some(threshold) = release_threshold {
            builder = builder.release_threshold(threshold);
        }
        let pool = builder.build()?;
        Ok(Box::new(CudaMemPoolWrapper { pool }))
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.context.cu_device() as u64)
    }
}

/// Check if CUDA is available.
pub fn is_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

/// CUDA memory pool wrapper implementing DeviceMemPoolOps.
pub struct CudaMemPoolWrapper {
    pool: CudaMemPool,
}

impl std::fmt::Debug for CudaMemPoolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMemPoolWrapper").finish()
    }
}

impl CudaMemPoolWrapper {
    /// Get the underlying CudaMemPool.
    pub fn inner(&self) -> &CudaMemPool {
        &self.pool
    }
}

impl DeviceMemPoolOps for CudaMemPoolWrapper {
    fn alloc_async(&self, size: usize, stream: &dyn DeviceStreamOps) -> Result<u64> {
        let raw_handle = stream.raw_handle()
            .ok_or_else(|| anyhow::anyhow!("Stream has no raw handle for pool allocation"))?;
        // SAFETY: raw_handle returns a valid CUstream handle from CudaStreamWrapper
        unsafe { self.pool.alloc_async_raw(size, raw_handle as cudarc::driver::sys::CUstream) }
    }

    fn free_async(&self, ptr: u64, stream: &dyn DeviceStreamOps) -> Result<()> {
        let raw_handle = stream.raw_handle()
            .ok_or_else(|| anyhow::anyhow!("Stream has no raw handle for pool free"))?;
        // SAFETY: raw_handle returns a valid CUstream handle from CudaStreamWrapper
        unsafe { self.pool.free_async_raw(ptr, raw_handle as cudarc::driver::sys::CUstream) }
    }
}

/// CUDA stream wrapper.
#[derive(Debug)]
pub struct CudaStreamWrapper {
    stream: Arc<cudarc::driver::CudaStream>,
}

impl CudaStreamWrapper {
    /// Get the underlying CUDA stream.
    pub fn inner(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.stream
    }
}

impl DeviceStreamOps for CudaStreamWrapper {
    fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()> {
        assert_eq!(src_ptrs.len(), dst_ptrs.len(), "batch_copy: src/dst length mismatch");
        let num_copies = src_ptrs.len();
        if num_copies == 0 {
            return Ok(());
        }

        let cuda_stream = self.stream.cu_stream() as cudarc::runtime::sys::cudaStream_t;

        // Build c_void pointer arrays for memcpy_batch
        let src_cvoid: Vec<*const std::ffi::c_void> = src_ptrs
            .iter()
            .map(|&p| p as *const std::ffi::c_void)
            .collect();
        let dst_cvoid: Vec<*mut std::ffi::c_void> = dst_ptrs
            .iter()
            .map(|&p| p as *mut std::ffi::c_void)
            .collect();

        let status = unsafe {
            kvbm_kernels::memcpy_batch(
                src_cvoid.as_ptr(),
                dst_cvoid.as_ptr(),
                size,
                num_copies,
                kvbm_kernels::MemcpyBatchMode::BatchedWithFallback,
                cuda_stream,
            )
        };

        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("CUDA batch_copy (memcpy_batch) failed: {:?}", status));
        }

        tracing::debug!(
            num_copies,
            size,
            batch_available = kvbm_kernels::is_memcpy_batch_available(),
            "CUDA batch_copy completed"
        );
        Ok(())
    }

    fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()> {
        unsafe {
            cuda_result::memcpy_htod_async(
                dst_device,
                src_host,
                self.stream.cu_stream(),
            )
            .map_err(|e| anyhow::anyhow!("CUDA memcpy_htod_async failed: {:?}", e))?;
        }
        Ok(())
    }

    fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
        unsafe {
            cuda_result::memcpy_dtoh_async(
                dst_host,
                src_device,
                self.stream.cu_stream(),
            )
            .map_err(|e| anyhow::anyhow!("CUDA memcpy_dtoh_async failed: {:?}", e))?;
        }
        Ok(())
    }

    fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let cuda_stream = self.stream.cu_stream() as cudarc::runtime::sys::cudaStream_t;

        let status = unsafe {
            kvbm_kernels::vectorized_copy(
                src_ptrs_device as *mut *mut std::ffi::c_void,
                dst_ptrs_device as *mut *mut std::ffi::c_void,
                chunk_size,
                count as i32,
                cuda_stream,
            )
        };

        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("CUDA vectorized_copy kernel failed: {:?}", status));
        }

        tracing::debug!(count, chunk_size, "CUDA vectorized_copy kernel launched");
        Ok(())
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let event = self.stream.record_event(None)
            .context("Failed to record CUDA event")?;
        Ok(Box::new(CudaEventWrapper { event }))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
            .context("CUDA stream synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.stream.cu_stream() as u64)
    }
}

/// CUDA event wrapper.
#[derive(Debug)]
pub struct CudaEventWrapper {
    pub event: cudarc::driver::CudaEvent,
}

impl DeviceEventOps for CudaEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        unsafe {
            match cuda_result::event::query(self.event.cu_event()) {
                Ok(()) => Ok(true),
                Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
                Err(e) => Err(anyhow::anyhow!("CUDA event query failed: {:?}", e)),
            }
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
            .context("CUDA event synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.event.cu_event() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_context_creation() {
        if !is_available() {
            return;
        }
        let ctx = CudaContext::new(0);
        assert!(ctx.is_ok());
    }
}
