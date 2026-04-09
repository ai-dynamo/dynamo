// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device abstraction traits for multi-backend support
//!
//! This module defines the core traits that all hardware backends
//! (CUDA, Level-Zero, Synapse, etc.) must implement.

use anyhow::Result;
use std::fmt::Debug;

/// Device context operations - the main interface for device management
pub trait DeviceContextOps: Send + Sync + Debug {
    /// Get the device ID this context is bound to
    fn device_id(&self) -> u32;

    /// Create a new stream/queue for async operations
    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>>;

    /// Allocate device memory
    fn allocate_device(&self, size: usize) -> Result<u64>;

    /// Free device memory
    fn free_device(&self, ptr: u64) -> Result<()>;

    /// Allocate pinned (page-locked) host memory
    fn allocate_pinned(&self, size: usize) -> Result<u64>;

    /// Free pinned host memory
    fn free_pinned(&self, ptr: u64) -> Result<()>;

    /// Bind context to current thread (if needed)
    fn bind_to_thread(&self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Disable automatic event tracking (CUDA-specific optimization)
    ///
    /// For backends like cudarc that add automatic event tracking for safety,
    /// this disables that overhead when managing events manually.
    /// Other backends (HPU, XPU) that don't have wrapper-level tracking can use the default no-op.
    ///
    /// # Safety
    /// Only safe when caller manually manages event synchronization.
    unsafe fn disable_event_tracking(&self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Get raw context handle for interop (optional)
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}

/// Device stream/queue operations - async execution interface
pub trait DeviceStreamOps: Send + Sync + Debug {
    /// Copy host to device (async)
    ///
    /// # Safety Requirements
    ///
    /// **IMPORTANT**: This operation is asynchronous. The DMA transfer is queued and may not
    /// complete until after this method returns. Callers MUST ensure that:
    ///
    /// 1. `src_host_data` remains valid and unmodified until the transfer completes
    /// 2. Either call `synchronize()` on this stream before dropping/reusing the buffer, OR
    /// 3. Use pinned memory allocated via `DeviceContextOps::allocate_pinned()` which has
    ///    guaranteed lifetime semantics
    ///
    /// Violating these requirements results in undefined behavior (use-after-free or data races).
    ///
    /// **Future API consideration**: This method should ideally be marked `unsafe` or accept
    /// owned/pinned buffer types to enforce safety at compile time.
    fn copy_h2d(&self, dst_device_ptr: u64, src_host_data: &[u8]) -> Result<()>;

    /// Copy device to host (async)
    ///
    /// # Safety Requirements
    ///
    /// **IMPORTANT**: This operation is asynchronous. The DMA transfer is queued and may not
    /// complete until after this method returns. Callers MUST ensure that:
    ///
    /// 1. `dst_host_data` remains valid and is not read until the transfer completes
    /// 2. Either call `synchronize()` on this stream before reading the buffer, OR
    /// 3. Use pinned memory allocated via `DeviceContextOps::allocate_pinned()` which has
    ///    guaranteed lifetime semantics
    ///
    /// Violating these requirements results in undefined behavior (reading uninitialized data
    /// or data races).
    ///
    /// **Future API consideration**: This method should ideally be marked `unsafe` or accept
    /// owned/pinned buffer types to enforce safety at compile time.
    fn copy_d2h(&self, dst_host_data: &mut [u8], src_device_ptr: u64) -> Result<()>;

    /// Copy device to device (async)
    ///
    /// Device-to-device copies are safe since both pointers are managed by the device runtime.
    fn copy_d2d(&self, dst_device_ptr: u64, src_device_ptr: u64, size: usize) -> Result<()>;

    /// Record an event on this stream
    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>>;

    /// Synchronize stream (wait for all operations to complete)
    fn synchronize(&self) -> Result<()>;

    /// Get raw stream handle for interop (optional)
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}

/// Device event operations - async completion tracking
pub trait DeviceEventOps: Send + Sync + Debug {
    /// Check if event has completed (non-blocking)
    fn is_complete(&self) -> Result<bool>;

    /// Wait for event to complete (blocking)
    fn synchronize(&self) -> Result<()>;

    /// Get raw event handle for interop (optional)
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}
