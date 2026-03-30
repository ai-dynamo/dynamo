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
        Ok(())  // Default: no-op
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
        Ok(())  // Default: no-op
    }

    /// Get raw context handle for interop (optional)
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}

/// Device stream/queue operations - async execution interface
pub trait DeviceStreamOps: Send + Sync + Debug {
    /// Copy host to device (async)
    fn copy_h2d(&self, dst_device_ptr: u64, src_host_data: &[u8]) -> Result<()>;

    /// Copy device to host (async)
    fn copy_d2h(&self, dst_host_data: &mut [u8], src_device_ptr: u64) -> Result<()>;

    /// Copy device to device (async)
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
