// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device abstraction traits for multi-backend support.
//!
//! Defines the core traits that all hardware backends
//! (CUDA, Level-Zero/XPU) must implement.

use anyhow::Result;
use std::fmt::Debug;

/// Hint for which hardware engine to bind a stream to.
///
/// - `Copy`: dedicated DMA engine (ZE: BCS queue group; CUDA: ignored).
/// - `Compute`: compute engine for kernel launch (ZE: CCS queue group; CUDA: ignored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineHint {
    Copy,
    Compute,
}

/// Device context operations — the main interface for device management.
pub trait DeviceContextOps: Send + Sync + Debug {
    /// Get the device ID this context is bound to.
    fn device_id(&self) -> u32;

    /// Create a new stream/queue for async operations.
    ///
    /// The `hint` selects the hardware engine class:
    /// - `EngineHint::Copy`: dedicated DMA engine for bulk transfers.
    /// - `EngineHint::Compute`: compute engine for kernel + small memcpy.
    fn create_stream(&self, hint: EngineHint) -> Result<Box<dyn DeviceStreamOps>>;

    /// Allocate device memory, returning a device pointer.
    fn allocate_device(&self, size: usize) -> Result<u64>;

    /// Free device memory.
    fn free_device(&self, ptr: u64) -> Result<()>;

    /// Allocate pinned (page-locked) host memory.
    fn allocate_pinned(&self, size: usize) -> Result<u64>;

    /// Free pinned host memory.
    fn free_pinned(&self, ptr: u64) -> Result<()>;

    /// Bind context to current thread (if needed by the backend).
    fn bind_to_thread(&self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Disable automatic event tracking (CUDA-specific optimization).
    ///
    /// # Safety
    /// Only safe when caller manually manages event synchronization.
    unsafe fn disable_event_tracking(&self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Create a memory pool for stream-ordered async allocations.
    ///
    /// # Arguments
    /// * `reserve_size` - Bytes to pre-allocate to warm the pool
    /// * `release_threshold` - Memory above this threshold is returned to system on free
    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>>;

    /// Get raw context handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}

/// Device stream/queue operations — async execution interface.
///
/// Operations are defined by copy pattern rather than direction:
/// - `batch_copy`: N independent DMA copies, each of the same size.
///   Direction is auto-detected from pointer addresses.
/// - `vectorized_copy`: N independent copies executed in parallel via a GPU kernel.
///   Pointer arrays are uploaded to device memory for kernel consumption.
pub trait DeviceStreamOps: Send + Sync + Debug {
    /// Batch copy: enqueue N independent memcpy operations, each of `size` bytes.
    ///
    /// Direction (H2D, D2H, D2D) is auto-detected from pointer addresses by
    /// the underlying runtime (`cudaMemcpyDefault` / `zeCommandListAppendMemoryCopy`).
    ///
    /// Used for whole-block FC→FC transfers (1 entry per block) and per-chunk
    /// transfers when GPU kernel launch is not available.
    fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()>;

    /// Async host-to-device memcpy on this stream.
    ///
    /// Enqueues a copy of `src_host` bytes to `dst_device` (a device pointer).
    /// The copy is stream-ordered: it executes after all preceding operations
    /// on this stream and before any subsequent ones.
    ///
    /// # Safety contract
    /// `dst_device` must point to at least `src_host.len()` bytes of device memory.
    /// `src_host` must remain valid until the copy completes (caller should
    /// record an event and synchronize before dropping the source buffer).
    fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()>;

    /// Async device-to-host memcpy on this stream.
    ///
    /// Enqueues a copy of `dst_host.len()` bytes from `src_device` into `dst_host`.
    /// The copy is stream-ordered. Caller must synchronize the stream
    /// before reading `dst_host`.
    fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()>;

    /// Vectorized copy: N independent copies executed in parallel via a GPU kernel.
    ///
    /// Both `src_ptrs_device` and `dst_ptrs_device` are device pointers to arrays
    /// of `count` device pointers (previously uploaded via [`memcpy_htod`]).
    /// The kernel reads these arrays and copies `chunk_size` bytes per pair.
    ///
    /// Used for FC↔LW per-chunk transfers where many small copies benefit from
    /// GPU-parallel execution rather than sequential DMA enqueues.
    ///
    /// # Arguments
    /// * `src_ptrs_device` - Device pointer to array of `count` source pointers
    /// * `dst_ptrs_device` - Device pointer to array of `count` destination pointers
    /// * `chunk_size` - Bytes to copy per pointer pair
    /// * `count` - Number of pointer pairs
    fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()>;

    /// Record an event on this stream.
    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>>;

    /// Synchronize stream (wait for all operations to complete).
    fn synchronize(&self) -> Result<()>;

    /// Get raw stream handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}

/// Device event operations — async completion tracking.
pub trait DeviceEventOps: Send + Sync + Debug {
    /// Check if event has completed (non-blocking).
    fn is_complete(&self) -> Result<bool>;

    /// Wait for event to complete (blocking).
    fn synchronize(&self) -> Result<()>;

    /// Get raw event handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}


/// Device memory pool operations — stream-ordered async allocation.
///
/// Wraps backend-specific memory pools (CUDA `cuMemAllocFromPoolAsync`,
/// Level-Zero USM pool, etc.) behind a unified interface.
///
/// Allocations and frees are stream-ordered: memory becomes available
/// after all preceding operations on the stream complete.
pub trait DeviceMemPoolOps: Send + Sync + Debug {
    /// Allocate memory from the pool, ordered on the given stream.
    ///
    /// # Arguments
    /// * `size` - Bytes to allocate
    /// * `stream` - Device stream ops for ordering (raw handle used internally)
    ///
    /// # Returns
    /// Device pointer to the allocated memory.
    fn alloc_async(&self, size: usize, stream: &dyn DeviceStreamOps) -> Result<u64>;

    /// Free memory back to the pool, ordered on the given stream.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - Device stream ops for ordering
    fn free_async(&self, ptr: u64, stream: &dyn DeviceStreamOps) -> Result<()>;
}
