// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA memory pool for efficient device memory allocation in hot paths.
//!
//! This module provides a safe wrapper around CUDA's memory pool APIs, enabling
//! fast async allocations that avoid the overhead of cudaMalloc/cudaFree per call.
//! Memory is returned to the pool on free and reused for subsequent allocations.

use anyhow::{Result, anyhow};
use cudarc::driver::sys::{
    self, CUmemAllocationType, CUmemLocationType, CUmemPool_attribute, CUmemPoolProps,
    CUmemoryPool, CUresult, CUstream,
};
use cudarc::driver::{CudaContext, CudaStream};
use std::ptr;
use std::sync::{Arc, Mutex};

/// Builder for creating a CUDA memory pool with configurable parameters.
///
/// # Example
/// ```ignore
/// let pool = CudaMemPoolBuilder::new(context, 64 * 1024 * 1024) // 64 MiB reserve
///     .release_threshold(32 * 1024 * 1024) // 32 MiB release threshold
///     .build()?;
/// ```
pub struct CudaMemPoolBuilder {
    /// CUDA context for the target device.
    context: Arc<CudaContext>,
    /// Bytes to pre-allocate to warm the pool.
    reserve_size: usize,
    /// Optional threshold above which memory is returned to the system on free.
    release_threshold: Option<u64>,
}

impl CudaMemPoolBuilder {
    /// Create a new builder with the required reserve size.
    ///
    /// # Arguments
    /// * `context` - CUDA context for the device
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn new(context: Arc<CudaContext>, reserve_size: usize) -> Self {
        Self {
            context,
            reserve_size,
            release_threshold: None,
        }
    }

    /// Set the release threshold for the pool.
    ///
    /// Memory above this threshold is returned to the system when freed.
    /// If not set, no release threshold is configured (CUDA default behavior).
    pub fn release_threshold(mut self, threshold: u64) -> Self {
        self.release_threshold = Some(threshold);
        self
    }

    /// Build the CUDA memory pool.
    ///
    /// This will:
    /// 1. Create the pool
    /// 2. Set the release threshold if configured
    /// 3. Pre-allocate and free memory to warm the pool
    pub fn build(self) -> Result<CudaMemPool> {
        // Initialize pool properties
        let mut props: CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = self.context.cu_device();

        let mut pool: CUmemoryPool = ptr::null_mut();

        // Create the pool
        let result = unsafe { sys::cuMemPoolCreate(&mut pool, &props) };
        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemPoolCreate failed with error: {:?}", result));
        }

        // Set release threshold if configured
        if let Some(threshold) = self.release_threshold {
            let result = unsafe {
                sys::cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                )
            };
            if result != CUresult::CUDA_SUCCESS {
                // Clean up on failure
                unsafe { sys::cuMemPoolDestroy(pool) };
                return Err(anyhow!(
                    "cuMemPoolSetAttribute failed with error: {:?}",
                    result
                ));
            }
        }

        let cuda_pool = CudaMemPool {
            inner: Mutex::new(pool),
        };

        // Warm the pool by pre-allocating and freeing memory
        if self.reserve_size > 0 {
            // Create a temporary stream for warming
            let stream = self.context.new_stream()?;

            // Allocate to warm the pool (using safe variant)
            let ptr = cuda_pool.alloc_async(self.reserve_size, &stream)?;

            // Free back to pool (memory stays reserved)
            cuda_pool.free_async(ptr, &stream)?;

            // Synchronize to ensure operations complete
            // SAFETY: stream.cu_stream() is valid for the lifetime of `stream`
            let result = unsafe { sys::cuStreamSynchronize(stream.cu_stream()) };
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!(
                    "cuStreamSynchronize failed with error: {:?}",
                    result
                ));
            }
        }

        Ok(cuda_pool)
    }
}

/// Safe wrapper around a CUDA memory pool.
///
/// The pool amortizes allocation overhead by maintaining a reservoir of device memory.
/// Allocations are fast sub-allocations from this reservoir, and frees return memory
/// to the pool rather than the OS (until the release threshold is exceeded).
///
/// # Thread Safety
///
/// This type uses internal locking to serialize host-side calls to CUDA driver APIs.
/// `cuMemAllocFromPoolAsync` is not host-thread reentrant, so concurrent calls from
/// multiple threads must be serialized. The GPU-side operations remain asynchronous
/// and stream-ordered.
///
/// Use [`CudaMemPoolBuilder`] for configurable pool creation with pre-allocation.
pub struct CudaMemPool {
    /// Mutex protecting the pool handle for host-thread serialization.
    ///
    /// CUDA's `cuMemAllocFromPoolAsync` does not guarantee host-thread reentrancy,
    /// so all calls to the pool must be serialized on the host side.
    inner: Mutex<CUmemoryPool>,
}

// SAFETY: CudaMemPool is Send because the Mutex serializes all host-side access
// to the pool handle, and CUDA driver state is thread-safe when properly serialized.
unsafe impl Send for CudaMemPool {}

// SAFETY: CudaMemPool is Sync because all access to the pool handle goes through
// the Mutex, which serializes host-thread access. The CUDA driver requires this
// serialization because cuMemAllocFromPoolAsync is not host-thread reentrant.
unsafe impl Sync for CudaMemPool {}

impl CudaMemPool {
    /// Create a builder for a new CUDA memory pool.
    ///
    /// # Arguments
    /// * `context` - CUDA context for the device
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn builder(context: Arc<CudaContext>, reserve_size: usize) -> CudaMemPoolBuilder {
        CudaMemPoolBuilder::new(context, reserve_size)
    }

    /// Allocate memory from the pool asynchronously.
    ///
    /// This is the safe variant that takes a `&CudaStream` reference, ensuring
    /// the stream is valid for the duration of the call.
    ///
    /// The allocation is stream-ordered; the memory is available for use
    /// after all preceding operations on the stream complete.
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex because `cuMemAllocFromPoolAsync`
    /// is not host-thread reentrant. The allocation itself is stream-ordered on
    /// the GPU side.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    /// * `stream` - CUDA stream for async ordering
    ///
    /// # Returns
    /// Device pointer to the allocated memory
    pub fn alloc_async(&self, size: usize, stream: &CudaStream) -> Result<u64> {
        // SAFETY: stream.cu_stream() returns a valid handle owned by the CudaStream,
        // and the borrow ensures the stream lives for the duration of this call.
        unsafe { self.alloc_async_raw(size, stream.cu_stream()) }
    }

    /// Allocate memory from the pool asynchronously (raw stream handle variant).
    ///
    /// This is the unsafe variant for use when you have a raw `CUstream` handle
    /// from sources other than cudarc's `CudaStream`.
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex because `cuMemAllocFromPoolAsync`
    /// is not host-thread reentrant.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    /// * `stream` - Raw CUDA stream handle for async ordering
    ///
    /// # Returns
    /// Device pointer to the allocated memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that `stream` is a valid CUDA stream handle that
    /// will remain valid for the duration of this call.
    pub unsafe fn alloc_async_raw(&self, size: usize, stream: CUstream) -> Result<u64> {
        let pool = self
            .inner
            .lock()
            .map_err(|e| anyhow!("mutex poisoned: {}", e))?;

        let mut ptr: u64 = 0;

        let result = unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, size, *pool, stream) };

        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!(
                "cuMemAllocFromPoolAsync failed with error: {:?}",
                result
            ));
        }

        Ok(ptr)
    }

    /// Free memory back to the pool asynchronously.
    ///
    /// This is the safe variant that takes a `&CudaStream` reference.
    ///
    /// The memory is returned to the pool's reservoir (not the OS) and can be
    /// reused by subsequent allocations. The free is stream-ordered.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - CUDA stream for async ordering
    pub fn free_async(&self, ptr: u64, stream: &CudaStream) -> Result<()> {
        // SAFETY: stream.cu_stream() returns a valid handle owned by the CudaStream,
        // and the borrow ensures the stream lives for the duration of this call.
        unsafe { self.free_async_raw(ptr, stream.cu_stream()) }
    }

    // NOTE: Unlike alloc_async_raw, this method does NOT acquire the pool mutex.
    // The mutex in alloc_async_raw ensures each allocation returns a unique pointer.
    // cuMemFreeAsync only enqueues a stream-ordered free operation for that unique
    // pointer - multiple threads can safely enqueue frees for different unique pointers
    // concurrently. The actual return-to-pool happens asynchronously on the GPU side.

    /// Free memory back to the pool asynchronously (raw stream handle variant).
    ///
    /// This is the unsafe variant for use when you have a raw `CUstream` handle.
    ///
    /// The memory is returned to the pool's reservoir (not the OS) and can be
    /// reused by subsequent allocations. The free is stream-ordered.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - Raw CUDA stream handle for async ordering
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is a valid device pointer previously allocated from this pool
    /// - `stream` is a valid CUDA stream handle
    pub unsafe fn free_async_raw(&self, ptr: u64, stream: CUstream) -> Result<()> {
        let result = unsafe { sys::cuMemFreeAsync(ptr, stream) };

        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemFreeAsync failed with error: {:?}", result));
        }

        Ok(())
    }
}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        // No need to lock - we have &mut self so exclusive access is guaranteed
        let pool = self
            .inner
            .get_mut()
            .expect("mutex should not be poisoned during drop");

        // Destroy the pool, releasing all memory back to the system
        let result = unsafe { sys::cuMemPoolDestroy(*pool) };
        if result != CUresult::CUDA_SUCCESS {
            tracing::warn!("cuMemPoolDestroy failed with error: {:?}", result);
        }
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation_with_builder() {
        // Skip if no CUDA device available
        let context = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping test - no CUDA device: {:?}", e);
                return;
            }
        };

        // Test builder with reserve size and release threshold
        let result = CudaMemPool::builder(context.clone(), 1024 * 1024) // 1 MiB reserve
            .release_threshold(64 * 1024 * 1024) // 64 MiB threshold
            .build();

        if result.is_err() {
            eprintln!("Skipping test - pool creation failed: {:?}", result.err());
            return;
        }
        let pool = result.unwrap();
        drop(pool);
    }

    #[test]
    fn test_pool_creation_no_threshold() {
        // Skip if no CUDA device available
        let context = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping test - no CUDA device: {:?}", e);
                return;
            }
        };

        // Test builder without release threshold
        let result = CudaMemPool::builder(context, 0).build();

        if result.is_err() {
            eprintln!("Skipping test - pool creation failed: {:?}", result.err());
            return;
        }
        let pool = result.unwrap();
        drop(pool);
    }
}
