// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SYCL/oneAPI memory pool for efficient device memory allocation in hot paths.
//!
//! This module provides a free-list-based memory pool over SYCL's
//! `sycl::malloc_device` / `sycl::free` (via oneAPI-rs raw FFI), mirroring
//! the [`super::cuda::CudaMemPool`] API surface for Intel GPUs.
//!
//! SYCL does **not** have a native memory-pool API equivalent to CUDA's
//! `cuMemPoolCreate` / `cuMemAllocFromPoolAsync`. This pool implements the same
//! semantics in software:
//!
//! * **`alloc`** — scans an internal free-list for a block >= the requested size.
//!   On a miss it falls back to `sycl::malloc_device`.
//! * **`free`** — returns the block to the free-list.  If total cached memory
//!   exceeds the release threshold the oldest / largest blocks are actually freed
//!   via `sycl::free`.
//! * **`Drop`** — frees every cached block and releases the pool.
//!
//! # Thread Safety
//!
//! [`SyclMemPool`] uses an internal [`Mutex`] to serialize host-side access,
//! matching the CUDA pool's serialization model.
//!
//! # Differences from [`super::cuda::CudaMemPool`]
//!
//! | Aspect | CUDA pool | SYCL pool |
//! |---|---|---|
//! | Backing API | `cuMemAllocFromPoolAsync` (stream-ordered) | `sycl::malloc_device` (synchronous) |
//! | Stream parameter | Required (alloc/free are stream-ordered) | **Not needed** — alloc/free are CPU-synchronous |
//! | GPU ordering | Implicit via stream | Caller must use a SYCL queue + events |
//!
//! The `alloc` / `free` methods intentionally omit the stream parameter because
//! SYCL device allocation is a host-synchronous operation. GPU-side ordering
//! is achieved by the caller via `SyclQueue` operations.

use anyhow::{Result, anyhow};
use oneapi_rs::safe::SyclQueue;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

// ── Free-list internals ──────────────────────────────────────────────────────

/// A single cached device-memory block.
#[derive(Debug)]
struct FreeBlock {
    /// Device pointer returned by `sycl::malloc_device`.
    ptr: u64,
    /// Actual allocated size in bytes.
    size: usize,
}

/// Internal pool state protected by a [`Mutex`].
#[derive(Debug)]
struct PoolInner {
    /// Free blocks keyed by size for best-fit lookup.
    ///
    /// `BTreeMap<size, Vec<FreeBlock>>` enables O(log n) `range(size..)`
    /// to find the smallest block that satisfies the request.
    free_list: BTreeMap<usize, Vec<FreeBlock>>,
    /// Sum of all bytes currently sitting in the free-list.
    cached_bytes: usize,
}

impl PoolInner {
    fn new() -> Self {
        Self {
            free_list: BTreeMap::new(),
            cached_bytes: 0,
        }
    }

    /// Try to pop a block >= `size` from the free-list (best-fit).
    fn pop_best_fit(&mut self, size: usize) -> Option<FreeBlock> {
        // Find the smallest bucket whose key >= size.
        let bucket_size = *self.free_list.range(size..).next()?.0;
        let vec = self.free_list.get_mut(&bucket_size)?;
        let block = vec.pop()?;
        if vec.is_empty() {
            self.free_list.remove(&bucket_size);
        }
        self.cached_bytes -= block.size;
        Some(block)
    }

    /// Push a block back onto the free-list.
    fn push(&mut self, block: FreeBlock) {
        self.cached_bytes += block.size;
        self.free_list
            .entry(block.size)
            .or_default()
            .push(block);
    }

    /// Drain blocks until `cached_bytes` <= `target`.
    /// Returns the drained blocks (caller must free them).
    fn drain_to(&mut self, target: usize) -> Vec<FreeBlock> {
        let mut drained = Vec::new();
        while self.cached_bytes > target {
            // Pop from the *largest* bucket first to reduce cached_bytes fastest.
            let largest_key = match self.free_list.keys().next_back().copied() {
                Some(k) => k,
                None => break,
            };
            let vec = self.free_list.get_mut(&largest_key).unwrap();
            let block = vec.pop().unwrap();
            if vec.is_empty() {
                self.free_list.remove(&largest_key);
            }
            self.cached_bytes -= block.size;
            drained.push(block);
        }
        drained
    }
}

// ── Builder ──────────────────────────────────────────────────────────────────

/// Builder for creating a SYCL device memory pool with configurable parameters.
///
/// # Example
/// ```ignore
/// use oneapi_rs::safe::SyclQueue;
///
/// let queue = SyclQueue::new_for_device_ordinal(0).unwrap();
/// let pool = SyclMemPoolBuilder::new(queue, 64 * 1024 * 1024) // 64 MiB reserve
///     .release_threshold(32 * 1024 * 1024) // 32 MiB release threshold
///     .build()
///     .unwrap();
/// ```
pub struct SyclMemPoolBuilder {
    /// SYCL queue (owns device + context + queue handles).
    queue: Arc<SyclQueue>,
    /// Bytes to pre-allocate to warm the pool.
    reserve_size: usize,
    /// Optional threshold above which memory is returned to the system on free.
    release_threshold: Option<u64>,
}

impl SyclMemPoolBuilder {
    /// Create a new builder with the required reserve size.
    ///
    /// # Arguments
    /// * `queue` - SYCL queue (wraps device + context + queue)
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn new(queue: Arc<SyclQueue>, reserve_size: usize) -> Self {
        Self {
            queue,
            reserve_size,
            release_threshold: None,
        }
    }

    /// Set the release threshold for the pool.
    ///
    /// Memory above this threshold is returned to the system when freed.
    /// If not set, the pool caches all freed memory indefinitely.
    pub fn release_threshold(mut self, threshold: u64) -> Self {
        self.release_threshold = Some(threshold);
        self
    }

    /// Build the SYCL memory pool.
    ///
    /// This will:
    /// 1. Create the pool (initialize the free-list)
    /// 2. Pre-allocate and cache memory to warm the pool
    pub fn build(self) -> Result<SyclMemPool> {
        let pool = SyclMemPool {
            queue: self.queue,
            inner: Mutex::new(PoolInner::new()),
            release_threshold: self.release_threshold.unwrap_or(u64::MAX),
        };

        // Warm the pool: allocate -> immediately return to free-list.
        if self.reserve_size > 0 {
            let ptr = pool.raw_alloc(self.reserve_size)?;
            let mut inner = pool
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            inner.push(FreeBlock {
                ptr,
                size: self.reserve_size,
            });
        }

        Ok(pool)
    }
}

// ── SyclMemPool ────────────────────────────────────────────────────────────────

/// Free-list memory pool for Intel GPUs via SYCL/oneAPI.
///
/// The pool amortizes allocation overhead by caching freed device-memory blocks
/// and reusing them for subsequent allocations. Blocks above the release threshold
/// are returned to the SYCL runtime via `sycl::free`.
///
/// # Thread Safety
///
/// This type uses internal locking to serialize host-side access, matching the
/// serialization model of [`super::cuda::CudaMemPool`].
///
/// Use [`SyclMemPoolBuilder`] for configurable pool creation with pre-allocation.
pub struct SyclMemPool {
    /// SYCL queue this pool allocates through.
    queue: Arc<SyclQueue>,
    /// Mutex protecting the free-list.
    inner: Mutex<PoolInner>,
    /// Cached bytes above this threshold trigger actual `sycl::free` calls.
    release_threshold: u64,
}

// SAFETY: SyclMemPool is Send because the Mutex serializes all host-side access
// to the free-list, and SYCL malloc/free are thread-safe when properly serialized.
unsafe impl Send for SyclMemPool {}

// SAFETY: SyclMemPool is Sync because all access to the inner state goes through
// the Mutex.
unsafe impl Sync for SyclMemPool {}

impl SyclMemPool {
    /// Create a builder for a new SYCL memory pool.
    ///
    /// # Arguments
    /// * `queue` - SYCL queue (wraps device + context + queue)
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn builder(queue: Arc<SyclQueue>, reserve_size: usize) -> SyclMemPoolBuilder {
        SyclMemPoolBuilder::new(queue, reserve_size)
    }

    /// Allocate memory from the pool.
    ///
    /// Tries the free-list first (best-fit); falls back to `sycl::malloc_device`.
    ///
    /// Unlike [`super::cuda::CudaMemPool::alloc_async`] this does **not** take a
    /// stream parameter because SYCL device allocation is CPU-synchronous.
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex to serialize access to the free-list.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    ///
    /// # Returns
    /// Device pointer (as `u64`) to the allocated memory.
    pub fn alloc(&self, size: usize) -> Result<u64> {
        if size == 0 {
            return Ok(0);
        }

        // Try the free-list first.
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            if let Some(block) = inner.pop_best_fit(size) {
                return Ok(block.ptr);
            }
        }

        // Cache miss — allocate from the SYCL runtime.
        self.raw_alloc(size)
    }

    /// Free memory back to the pool.
    ///
    /// The block is returned to the pool's free-list for reuse. If the total
    /// cached memory exceeds the release threshold, the largest blocks are
    /// freed via `sycl::free` until the cache is within budget.
    ///
    /// Unlike [`super::cuda::CudaMemPool::free_async`] this does **not** take a
    /// stream parameter because SYCL memory free is CPU-synchronous.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously obtained from [`alloc`](Self::alloc)
    /// * `size` - Size in bytes of the allocation (must match the original alloc)
    pub fn free(&self, ptr: u64, size: usize) -> Result<()> {
        if ptr == 0 || size == 0 {
            return Ok(());
        }

        let drained = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            inner.push(FreeBlock { ptr, size });

            // Enforce release threshold.
            if inner.cached_bytes as u64 > self.release_threshold {
                inner.drain_to(self.release_threshold as usize)
            } else {
                Vec::new()
            }
        };

        // Actually free drained blocks (outside the lock).
        for block in drained {
            self.raw_free(block.ptr)?;
        }

        Ok(())
    }

    /// Get the SYCL queue this pool allocates through.
    pub fn queue(&self) -> &Arc<SyclQueue> {
        &self.queue
    }

    /// Current number of bytes cached in the free-list.
    pub fn cached_bytes(&self) -> usize {
        self.inner.lock().map(|g| g.cached_bytes).unwrap_or(0)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Raw allocation via `sycl::malloc_device` (oneAPI-rs raw FFI).
    fn raw_alloc(&self, size: usize) -> Result<u64> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            oneapi_rs::sys::sycl_rs_alloc(
                self.queue.handle(),
                oneapi_rs::sys::sycl_rs_alloc_kind_t::SYCL_RS_ALLOC_KIND_DEVICE,
                size,
                0, // alignment — 0 lets the runtime choose
                &mut ptr,
            )
            .result()
        }
        .map_err(|e| anyhow!("sycl_rs_alloc DEVICE failed: {}", e))?;

        Ok(ptr as u64)
    }

    /// Raw free via `sycl::free` (oneAPI-rs raw FFI).
    fn raw_free(&self, ptr: u64) -> Result<()> {
        unsafe {
            oneapi_rs::sys::sycl_rs_free(
                self.queue.handle(),
                ptr as *mut c_void,
            )
            .result()
        }
        .map_err(|e| anyhow!("sycl_rs_free failed: {}", e))?;

        Ok(())
    }
}

impl Drop for SyclMemPool {
    fn drop(&mut self) {
        // No need to lock — we have &mut self so exclusive access is guaranteed.
        let inner = self
            .inner
            .get_mut()
            .expect("mutex should not be poisoned during drop");

        // Drain every block from the free-list and return to the runtime.
        let all_blocks = inner.drain_to(0);
        for block in all_blocks {
            unsafe {
                if let Err(e) = oneapi_rs::sys::sycl_rs_free(
                    self.queue.handle(),
                    block.ptr as *mut c_void,
                )
                .result()
                {
                    tracing::warn!("sycl_rs_free failed during SyclMemPool drop: {e}");
                }
            }
        }
    }
}

#[cfg(all(test, feature = "testing-sycl"))]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation_with_builder() {
        let queue = match SyclQueue::new_for_device_ordinal(0) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("Skipping test - no SYCL device: {e}");
                return;
            }
        };

        let pool = SyclMemPool::builder(queue, 1024 * 1024) // 1 MiB reserve
            .release_threshold(64 * 1024 * 1024) // 64 MiB threshold
            .build()
            .expect("pool creation should succeed");

        assert!(pool.cached_bytes() >= 1024 * 1024, "pool should be warmed");
        drop(pool);
    }

    #[test]
    fn test_pool_alloc_free_reuse() {
        let queue = match SyclQueue::new_for_device_ordinal(0) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("Skipping test - no SYCL device: {e}");
                return;
            }
        };

        let pool = SyclMemPool::builder(queue, 0).build().expect("pool build");

        // First allocation — cache miss, goes to SYCL runtime.
        let ptr1 = pool.alloc(4096).expect("alloc 4096");
        assert_ne!(ptr1, 0);

        // Free — goes to free-list.
        pool.free(ptr1, 4096).expect("free");
        assert_eq!(pool.cached_bytes(), 4096);

        // Second allocation of same size — cache hit, should return ptr1.
        let ptr2 = pool.alloc(4096).expect("alloc 4096 again");
        assert_eq!(ptr1, ptr2, "should reuse the freed block");
        assert_eq!(pool.cached_bytes(), 0, "free-list should be empty after hit");

        // Clean up.
        pool.free(ptr2, 4096).expect("free");
        drop(pool);
    }

    #[test]
    fn test_pool_release_threshold() {
        let queue = match SyclQueue::new_for_device_ordinal(0) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("Skipping test - no SYCL device: {e}");
                return;
            }
        };

        // Threshold of 8 KiB.
        let pool = SyclMemPool::builder(queue, 0)
            .release_threshold(8192)
            .build()
            .expect("pool build");

        // Allocate and free 12 KiB — exceeds threshold, should drain.
        let ptr = pool.alloc(12288).expect("alloc 12k");
        pool.free(ptr, 12288).expect("free 12k");

        // Cached should be 0 because 12288 > threshold(8192) triggers drain.
        assert_eq!(
            pool.cached_bytes(),
            0,
            "should have been drained below threshold"
        );

        drop(pool);
    }

    #[test]
    fn test_pool_creation_no_threshold() {
        let queue = match SyclQueue::new_for_device_ordinal(0) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("Skipping test - no SYCL device: {e}");
                return;
            }
        };

        let pool = SyclMemPool::builder(queue, 0).build().expect("pool build");
        drop(pool);
    }
}
