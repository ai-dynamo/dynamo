// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA backend implementation.
//!
//! Wraps cudarc types with the device abstraction traits.

use crate::device::traits::*;
use anyhow::{Result, Context as _};
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUresult, CU_MEMHOSTALLOC_PORTABLE};
use cudarc::driver::DriverError;
use std::sync::Arc;
use dynamo_memory::CudaMemPool;

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

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
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
        tracing::warn!("CUDA free_device called with raw pointer {} - memory managed by allocator", ptr);
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        unsafe {
            let ptr = cuda_result::malloc_host(size, CU_MEMHOSTALLOC_PORTABLE)
                .context("Failed to allocate pinned host memory")?;
            Ok(ptr as u64)
        }
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        unsafe {
            cuda_result::free_host(ptr as *mut std::ffi::c_void)
                .context("Failed to free pinned host memory")?;
        }
        Ok(())
    }

    fn bind_to_thread(&self) -> Result<()> {
        Ok(())
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

    fn vectorized_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], chunk_size: usize) -> Result<()> {
        assert_eq!(src_ptrs.len(), dst_ptrs.len(), "vectorized_copy: src/dst length mismatch");
        let total_chunks = src_ptrs.len();
        if total_chunks == 0 {
            return Ok(());
        }

        // Bind CUDA context to current thread before any CUDA operations.
        self.stream.context().bind_to_thread()?;

        let cuda_stream = self.stream.cu_stream() as cudarc::runtime::sys::cudaStream_t;
        let cu_stream = self.stream.cu_stream();

        let ptr_array_bytes = total_chunks * std::mem::size_of::<u64>();

        // Allocate device memory for pointer arrays
        let src_ptrs_device = unsafe {
            cuda_result::malloc_sync(ptr_array_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to allocate device memory for src pointers: {:?}", e))?
        };
        let dst_ptrs_device = unsafe {
            cuda_result::malloc_sync(ptr_array_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to allocate device memory for dst pointers: {:?}", e))?
        };

        // Upload pointer arrays to device
        unsafe {
            cuda_result::memcpy_htod_async(
                src_ptrs_device,
                std::slice::from_raw_parts(src_ptrs.as_ptr() as *const u8, ptr_array_bytes),
                cu_stream,
            ).map_err(|e| anyhow::anyhow!("Failed to upload src pointer array: {:?}", e))?;

            cuda_result::memcpy_htod_async(
                dst_ptrs_device,
                std::slice::from_raw_parts(dst_ptrs.as_ptr() as *const u8, ptr_array_bytes),
                cu_stream,
            ).map_err(|e| anyhow::anyhow!("Failed to upload dst pointer array: {:?}", e))?;
        }

        // Record event after pointer upload: ensures host-side src_ptrs/dst_ptrs
        // Vecs remain valid until the async uploads complete.
        let pointers_uploaded_event = self.stream.record_event(None)
            .map_err(|e| anyhow::anyhow!("Failed to record pointer upload event: {:?}", e))?;

        // Launch vectorized_copy kernel
        let status = unsafe {
            kvbm_kernels::vectorized_copy(
                src_ptrs_device as *mut *mut std::ffi::c_void,
                dst_ptrs_device as *mut *mut std::ffi::c_void,
                chunk_size,
                total_chunks as i32,
                cuda_stream,
            )
        };

        // Synchronize to ensure kernel completes before freeing temp allocations.
        // Note: with sync alloc/free we must wait for the kernel, not just uploads.
        // TODO: Switch to pool.alloc_async/free_async to only sync on upload event.
        unsafe {
            cuda_result::stream::synchronize(cu_stream)
                .map_err(|e| anyhow::anyhow!("Stream sync failed after vectorized_copy: {:?}", e))?;
        }

        // Free temporary device allocations
        unsafe {
            let _ = cuda_result::free_sync(src_ptrs_device);
            let _ = cuda_result::free_sync(dst_ptrs_device);
        }

        // Ensure pointer uploads completed (host Vecs safe to drop on return).
        pointers_uploaded_event.synchronize()?;

        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("CUDA vectorized_copy failed: {:?}", status));
        }

        tracing::debug!(total_chunks, chunk_size, "CUDA vectorized_copy completed");
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
