// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Level Zero memory pool for efficient device memory allocation in hot paths.
//!
//! This module provides a free-list-based memory pool over Level Zero's
//! `zeMemAllocDevice` / `zeMemFree` (via `level-zero-rc`), mirroring the
//! [`super::cuda::CudaMemPool`] API surface for Intel GPUs.
//!
//! Level Zero does **not** have a native memory-pool API equivalent to CUDA's
//! `cuMemPoolCreate` / `cuMemAllocFromPoolAsync`. This pool implements the same
//! semantics in software:
//!
//! * **`alloc`** -- scans an internal free-list for a block >= the requested size.
//!   On a miss it falls back to `context.alloc_device()`.
//! * **`free`** -- returns the block to the free-list.  If total cached memory
//!   exceeds the release threshold the largest blocks are actually freed
//!   by dropping the `DeviceBuffer` (which calls `zeMemFree` via RAII).
//! * **`Drop`** -- drops every cached block, releasing all device memory.
//!
//! # Thread Safety
//!
//! [`ZeMemPool`] uses an internal [`Mutex`] to serialize host-side access,
//! matching the CUDA pool's serialization model.
//!
//! # Differences from [`super::cuda::CudaMemPool`]
//!
//! | Aspect | CUDA pool | L0 pool |
//! |---|---|---|
//! | Backing API | `cuMemAllocFromPoolAsync` (stream-ordered) | `zeMemAllocDevice` (synchronous) |
//! | Stream parameter | Required (alloc/free are stream-ordered) | **Not needed** -- alloc/free are CPU-synchronous |
//! | GPU ordering | Implicit via stream | Caller must use an immediate command list + events |
//! | RAII | Manual via `cuMemPoolDestroy` | `DeviceBuffer` drop calls `zeMemFree` |
//!
//! The `alloc` / `free` methods intentionally omit the stream parameter because
//! Level Zero memory allocation is a host-synchronous operation.  GPU-side ordering
//! is achieved by the caller via immediate command lists.

use anyhow::{Result, anyhow};
use level_zero::{Context, Device, DeviceBuffer};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tracing::warn;

// -- Free-list internals ---------------------------------------------------

/// A single cached device-memory block (owns the allocation via RAII).
struct FreeBlock {
    /// The level-zero-rc `DeviceBuffer` that owns the allocation.
    /// Dropping this calls `zeMemFree` automatically.
    buffer: DeviceBuffer,
    /// Device pointer (cached from `buffer.as_mut_ptr()`).
    ptr: u64,
    /// Allocated size in bytes.
    size: usize,
}

// SAFETY: DeviceBuffer is Send+Sync (declared in level-zero-rc).
unsafe impl Send for FreeBlock {}
unsafe impl Sync for FreeBlock {}

impl std::fmt::Debug for FreeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreeBlock")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .finish()
    }
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
    /// Returns the drained blocks (caller drops them to free via RAII).
    fn drain_to(&mut self, target: usize) -> Vec<FreeBlock> {
        let mut drained = Vec::new();
        while self.cached_bytes > target {
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

// -- Builder ---------------------------------------------------------------

/// Builder for creating a Level Zero memory pool with configurable parameters.
pub struct ZeMemPoolBuilder {
    context: Arc<Context>,
    device: Device,
    reserve_size: usize,
    release_threshold: Option<u64>,
}

impl ZeMemPoolBuilder {
    /// Create a new builder.
    ///
    /// # Arguments
    /// * `context` - Level Zero context (shared, Arc-wrapped)
    /// * `device` - Level Zero device to allocate on
    /// * `reserve_size` - Bytes to pre-allocate to warm the pool
    pub fn new(context: Arc<Context>, device: Device, reserve_size: usize) -> Self {
        Self {
            context,
            device,
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

    /// Build the Level Zero memory pool.
    pub fn build(self) -> Result<ZeMemPool> {
        let pool = ZeMemPool {
            context: self.context,
            device: self.device,
            inner: Mutex::new(PoolInner::new()),
            release_threshold: self.release_threshold.unwrap_or(u64::MAX),
        };

        // Warm the pool: allocate -> immediately return to free-list.
        if self.reserve_size > 0 {
            let (buffer, ptr) = pool.raw_alloc(self.reserve_size)?;
            let mut inner = pool
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            inner.push(FreeBlock {
                buffer,
                ptr,
                size: self.reserve_size,
            });
        }

        Ok(pool)
    }
}

// -- ZeMemPool -------------------------------------------------------------

/// Free-list memory pool for Intel GPUs via Level Zero (`level-zero-rc`).
///
/// The pool amortizes allocation overhead by caching freed device-memory blocks
/// and reusing them for subsequent allocations.  Blocks above the release
/// threshold are returned to the driver by dropping the `DeviceBuffer` RAII
/// wrapper (which calls `zeMemFree`).
///
/// Use [`ZeMemPoolBuilder`] for configurable pool creation with pre-allocation.
pub struct ZeMemPool {
    context: Arc<Context>,
    device: Device,
    inner: Mutex<PoolInner>,
    release_threshold: u64,
}

// SAFETY: ZeMemPool is Send because the Mutex serializes all host-side access.
unsafe impl Send for ZeMemPool {}
// SAFETY: ZeMemPool is Sync because all access goes through the Mutex.
unsafe impl Sync for ZeMemPool {}

impl std::fmt::Debug for ZeMemPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeMemPool")
            .field("release_threshold", &self.release_threshold)
            .field("cached_bytes", &self.cached_bytes())
            .finish()
    }
}

impl ZeMemPool {
    /// Create a builder for a new Level Zero memory pool.
    pub fn builder(context: Arc<Context>, device: Device, reserve_size: usize) -> ZeMemPoolBuilder {
        ZeMemPoolBuilder::new(context, device, reserve_size)
    }

    /// Allocate memory from the pool.
    ///
    /// Tries the free-list first (best-fit); falls back to `context.alloc_device()`.
    ///
    /// # Returns
    /// `(DeviceBuffer, u64)` - the RAII owner and the device pointer.
    pub fn alloc(&self, size: usize) -> Result<(DeviceBuffer, u64)> {
        if size == 0 {
            return Err(anyhow!("ZeMemPool: cannot allocate 0 bytes"));
        }

        // Try the free-list first.
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            if let Some(block) = inner.pop_best_fit(size) {
                return Ok((block.buffer, block.ptr));
            }
        }

        // Cache miss -- allocate from the driver.
        self.raw_alloc(size)
    }

    /// Free memory back to the pool.
    ///
    /// The block is returned to the pool's free-list for reuse.  If the total
    /// cached memory exceeds the release threshold, the largest blocks are
    /// dropped (triggering `zeMemFree` via RAII).
    pub fn free(&self, buffer: DeviceBuffer, ptr: u64, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }

        let drained = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| anyhow!("mutex poisoned: {}", e))?;
            inner.push(FreeBlock { buffer, ptr, size });

            if inner.cached_bytes as u64 > self.release_threshold {
                inner.drain_to(self.release_threshold as usize)
            } else {
                Vec::new()
            }
        };

        // Drained blocks dropped here -> zeMemFree via DeviceBuffer's Drop.
        drop(drained);
        Ok(())
    }

    /// Get the context this pool allocates in.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Get the device this pool allocates on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Current number of bytes cached in the free-list.
    pub fn cached_bytes(&self) -> usize {
        self.inner.lock().map(|g| g.cached_bytes).unwrap_or(0)
    }

    /// Raw allocation via `context.alloc_device()`.
    fn raw_alloc(&self, size: usize) -> Result<(DeviceBuffer, u64)> {
        let buffer = self
            .context
            .alloc_device(&self.device, size, 1)
            .map_err(|e| anyhow!("zeMemAllocDevice failed: {:?}", e))?;
        let ptr = buffer.as_mut_ptr() as u64;
        Ok((buffer, ptr))
    }
}

impl Drop for ZeMemPool {
    fn drop(&mut self) {
        // No need to lock — we have &mut self so exclusive access is guaranteed.
        let inner = self
            .inner
            .get_mut()
            .expect("mutex should not be poisoned during drop");

        // Drain every block from the free-list.
        let all_blocks = inner.drain_to(0);
        if !all_blocks.is_empty() {
            let total_bytes: usize = all_blocks.iter().map(|b| b.size).sum();
            let count = all_blocks.len();
            // DeviceBuffer::drop calls zeMemFree (errors silently ignored by level-zero-rc).
            drop(all_blocks);
            warn!(
                "ZeMemPool::drop: released {} cached blocks ({} bytes)",
                count, total_bytes,
            );
        }
    }
}

#[cfg(all(test, feature = "testing-ze"))]
mod tests {
    use super::*;

    fn get_test_context_and_device() -> Option<(Arc<Context>, Device)> {
        level_zero::init().ok()?;
        let drivers = level_zero::drivers().ok()?;
        let driver = drivers.into_iter().next()?;
        let devices = driver.devices().ok()?;
        let device = devices.into_iter().next()?;
        let context = Arc::new(Context::create(&driver).ok()?);
        Some((context, device))
    }

    #[test]
    fn test_pool_creation_with_builder() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 1024 * 1024)
            .release_threshold(64 * 1024 * 1024)
            .build()
            .expect("pool creation should succeed");

        assert!(pool.cached_bytes() >= 1024 * 1024, "pool should be warmed");
    }

    #[test]
    fn test_pool_alloc_free_reuse() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");

        let (buf1, ptr1) = pool.alloc(4096).expect("alloc 4096");
        assert_ne!(ptr1, 0);

        pool.free(buf1, ptr1, 4096).expect("free");
        assert_eq!(pool.cached_bytes(), 4096);

        let (_buf2, ptr2) = pool.alloc(4096).expect("alloc 4096 again");
        assert_eq!(ptr1, ptr2, "should reuse the freed block");
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn test_pool_release_threshold() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0)
            .release_threshold(8192)
            .build()
            .expect("pool build");

        let (buf, ptr) = pool.alloc(12288).expect("alloc 12k");
        pool.free(buf, ptr, 12288).expect("free 12k");

        assert_eq!(pool.cached_bytes(), 0, "should have been drained below threshold");
    }

    #[test]
    fn test_pool_creation_no_threshold() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");
        drop(pool);
    }

    #[test]
    fn test_pool_best_fit_selection() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");

        // Allocate 4K and 8K blocks.
        let (buf_4k, ptr_4k) = pool.alloc(4096).expect("alloc 4K");
        let (buf_8k, ptr_8k) = pool.alloc(8192).expect("alloc 8K");

        // Free both — free-list now has {4096: [block], 8192: [block]}.
        pool.free(buf_4k, ptr_4k, 4096).expect("free 4K");
        pool.free(buf_8k, ptr_8k, 8192).expect("free 8K");
        assert_eq!(pool.cached_bytes(), 4096 + 8192);

        // Alloc 5K — should get the 8K block (best-fit >= 5K), not the 4K block.
        let (_buf_5k, ptr_5k) = pool.alloc(5120).expect("alloc 5K");
        assert_eq!(ptr_5k, ptr_8k, "should pick 8K block (best-fit), not 4K");
        assert_eq!(pool.cached_bytes(), 4096, "4K block should remain cached");
    }

    #[test]
    fn test_pool_alloc_zero_size_error() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");
        let result = pool.alloc(0);
        assert!(result.is_err(), "alloc(0) should return an error");
    }

    #[test]
    fn test_pool_multiple_alloc_free_cycles() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");

        // Run 100 alloc/free cycles with same size — should always reuse.
        let mut last_ptr = 0u64;
        for i in 0..100 {
            let (buf, ptr) = pool.alloc(4096).expect("alloc");
            if i > 0 {
                assert_eq!(ptr, last_ptr, "cycle {i}: should reuse freed block");
            }
            last_ptr = ptr;
            pool.free(buf, ptr, 4096).expect("free");
        }
        assert_eq!(pool.cached_bytes(), 4096, "one block should remain cached");
    }

    #[test]
    fn test_pool_concurrent_access() {
        use std::sync::Arc;

        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let pool = Arc::new(
            ZeMemPool::builder(ctx, dev, 0).build().expect("pool build"),
        );

        let mut handles = Vec::new();
        for _ in 0..4 {
            let pool = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let (buf, ptr) = pool.alloc(4096).expect("alloc");
                    assert_ne!(ptr, 0);
                    pool.free(buf, ptr, 4096).expect("free");
                }
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    #[test]
    fn test_pool_free_zero_and_getters() {
        let (ctx, dev) = match get_test_context_and_device() {
            Some(v) => v,
            None => {
                eprintln!("Skipping test - no Level Zero device");
                return;
            }
        };

        let ctx_clone = Arc::clone(&ctx);
        let pool = ZeMemPool::builder(ctx, dev, 0).build().expect("pool build");

        // Verify getters return the correct handles.
        assert!(Arc::ptr_eq(pool.context(), &ctx_clone), "context() should return the same Arc<Context>");
        // device() should not panic.
        let _ = pool.device();

        // free(_, _, 0) should be a no-op, not an error.
        // We need a dummy buffer to pass ownership — allocate one and free with size=0.
        let (buf, ptr) = pool.alloc(4096).expect("alloc");
        let result = pool.free(buf, ptr, 0);
        assert!(result.is_ok(), "free with size=0 should succeed as no-op");
        // The buffer was consumed but not returned to pool (size=0 early return),
        // so cached_bytes should be 0 (the DeviceBuffer was dropped/leaked).
        assert_eq!(pool.cached_bytes(), 0, "nothing should be cached after free(_, _, 0)");
    }
}
