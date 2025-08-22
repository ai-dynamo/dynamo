// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module to integration with CUDA
//!
//! This module will be a standalong crates, likely called `dynamo-cuda`; however, for the time, it will
//! life as a submodule of `dynamo-llm`.
//!
//! This implementation will include a set of traits for extracting raw `cudarc::driver::sys` objects.
//!
//! Dynamo will generally not be the primary compute driver within an application, but a secondary source
//! of logic that may be used inconjunction with the primary compute driver, e.g. vLLM use of PyTorch is
//! the primary CUDA context.
//!
//! In order for Dynamo to avoid creating its own CUDA context, the following traits are provided so
//! that we may tap the lower level CUDA context, streams, events, etcs from external sources and leverage
//! them within Dynamo.

pub mod cudarc;
pub mod external;
pub mod safe;

pub mod v2;

pub mod sys {
    //! This module re-exports the raw CUDA types from [`cudarc`] for convenience.
    pub use ::cudarc::driver::sys::{CUcontext, CUevent, CUstream};
}

/// Re-export the DriverError type from [`cudarc`] for convenience.
pub use ::cudarc::driver::DriverError;

use ::cudarc::driver::sys::{
    cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cudaError_enum, CUctx_st, CUevent_flags,
};

use std::marker::PhantomData;
use std::{pin::Pin, ptr::NonNull};

/// Helper function to check CUDA results using cudarc's DriverError
/// This is much better than raw FFI error handling
#[inline]
pub fn check_cuda(result: cudaError_enum) -> Result<(), DriverError> {
    if result == cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        // cudarc's DriverError already handles error name/description lookup internally
        Err(DriverError(result))
    }
}

pub trait CudaContext {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA context.
    /// The caller must ensure that the context is valid and that the CUDA context is active.
    ///
    /// # Returns
    /// A NonNull wrapper around the CUDA context, guaranteeing it's not null.
    unsafe fn cu_context(&self) -> NonNull<CUctx_st>;

    fn bind_to_thread(&self) -> Pin<Box<DynamoCudaContextGuard>> {
        unsafe { DynamoCudaContextGuard::new(self.cu_context().as_ptr()) }
    }
}

pub trait CudaStream: Send + Sync {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA stream.
    /// The caller must ensure that the stream is valid and that the CUDA context is active.
    ///
    /// Similarly, any pointers/references to data for which the stream will be accessed must
    /// have proper lifetimes and scoping, which is not guaranteed by this trait.
    unsafe fn cu_stream(&self) -> sys::CUstream;

    fn context(&self) -> &dyn CudaContext;
}

pub trait CudaEvent {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA event.
    /// The caller must ensure that the event is valid and a valid CUDA context is active.
    unsafe fn cu_event(&self) -> sys::CUevent;

    /// Destroys the CUDA event and releases its resources.
    ///
    /// Maps to: `cudaEventDestroy(cudaEvent_t event)`
    ///
    /// Note: This may also be called automatically in Drop implementations.
    ///
    /// # Returns
    /// * `Ok(())` - Event successfully destroyed
    /// * `Err(anyhow::Error)` - Failed to destroy event (invalid event, etc.)
    fn destroy_event(&self) -> anyhow::Result<()>;

    /// Computes the elapsed time between this event (start) and the end event.
    ///
    /// Maps to: `cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)`
    ///
    /// # Arguments
    /// * `end_event` - The ending event for the time measurement
    ///
    /// # Returns
    /// * `Ok(f64)` - Time elapsed in milliseconds
    /// * `Err(anyhow::Error)` - Failed to compute elapsed time (events not recorded, invalid events, etc.)
    fn elapsed_time(&self, end_event: &dyn CudaEvent) -> anyhow::Result<f64>;

    /// Records the event on the specified stream (or default stream if None).
    ///
    /// Maps to: `cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)`
    ///
    /// # Arguments
    /// * `stream` - Stream to record on, or None for default stream (stream 0)
    ///
    /// # Returns
    /// * `Ok(())` - Event successfully recorded on stream
    /// * `Err(anyhow::Error)` - Failed to record event (invalid stream, invalid event, etc.)
    fn record_event(&self, stream: Option<&dyn CudaStream>) -> anyhow::Result<()>;

    /// Records the event on the specified stream with additional flags.
    ///
    /// Maps to: `cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned int flags = 0)`
    ///
    /// # Arguments
    /// * `stream` - Stream to record on, or None for default stream (stream 0)
    /// * `flags` - Additional flags for recording
    ///
    /// # Returns
    /// * `Ok(())` - Event successfully recorded on stream with flags
    /// * `Err(anyhow::Error)` - Failed to record event (invalid stream, invalid flags, etc.)
    fn record_event_with_flags(
        &self,
        stream: Option<&dyn CudaStream>,
        flags: CUevent_flags,
    ) -> anyhow::Result<()>;

    /// Queries the event's completion status without blocking.
    ///
    /// Maps to: `cudaEventQuery(cudaEvent_t event)`
    ///
    /// # Returns
    /// * `Ok(true)` - Event has completed
    /// * `Ok(false)` - Event has not completed yet
    /// * `Err(anyhow::Error)` - Query failed (invalid event, etc.)
    fn query_event(&self) -> anyhow::Result<bool>;

    /// Blocks until the event completes.
    ///
    /// Maps to: `cudaEventSynchronize(cudaEvent_t event)`
    ///
    /// # Returns
    /// * `Ok(())` - Event completed successfully
    /// * `Err(anyhow::Error)` - Synchronization failed (invalid event, etc.)
    fn synchronize_event(&self) -> anyhow::Result<()>;
}

/// A CUDA context guard that ensures safe access to CUDA contexts.
///
/// This guard:
/// - Cannot be moved (uses PhantomPinned)
/// - Cannot be cloned
/// - Cannot pass across async boundaries (!Send + !Sync)
/// - Provides safe access to the underlying CUDA context
/// - Automatically manages context lifecycle
pub struct DynamoCudaContextGuard {
    context: NonNull<CUctx_st>,
    // Prevent the guard from being moved
    _pin: std::marker::PhantomPinned,
    // Prevent Send + Sync to avoid crossing async boundaries
    _not_send_sync: PhantomData<*const ()>,
}

impl DynamoCudaContextGuard {
    /// Create a new context guard from a context provider.
    ///
    /// This is a safe constructor that pushes the context onto the CUDA context stack
    /// and ensures it will be properly popped when the guard is dropped.
    ///
    /// # Arguments
    /// * `provider` - A reference to something that can provide a CUDA context
    ///
    /// # Returns
    /// A pinned context guard that manages the CUDA context safely
    ///
    /// # Panics
    /// Panics if the CUDA context push operation fails
    /// # Safety
    ///
    /// This function dereferences a raw pointer and interacts with the CUDA driver API.
    /// The caller must ensure the context is valid.
    pub unsafe fn new(context: sys::CUcontext) -> Pin<Box<Self>> {
        // Validate context is not null
        let context_nonnull = NonNull::new(context).expect("CUDA context cannot be null");

        // Push the context onto the CUDA context stack
        let result = cuCtxPushCurrent_v2(context);
        check_cuda(result).expect("Failed to push CUDA context");

        let guard = Self {
            context: context_nonnull,
            _pin: std::marker::PhantomPinned,
            _not_send_sync: PhantomData,
        };

        Box::pin(guard)
    }

    /// Get the raw CUDA context.
    ///
    /// This method is safe because the guard ensures the context remains valid
    /// for its lifetime and cannot be moved or passed across async boundaries.
    ///
    /// # Returns
    /// The raw CUDA context handle
    pub fn context(&self) -> sys::CUcontext {
        self.context.as_ptr()
    }
}

impl Drop for DynamoCudaContextGuard {
    fn drop(&mut self) {
        // Pop the context from the CUDA context stack when the guard is dropped
        let mut popped_context: sys::CUcontext = std::ptr::null_mut();
        let result = unsafe { cuCtxPopCurrent_v2(&mut popped_context) };

        // Log errors but don't panic in Drop
        if result != cudaError_enum::CUDA_SUCCESS {
            eprintln!("Warning: Failed to pop CUDA context in drop: {:?}", result);
        }

        // Verify we popped the expected context
        if popped_context != self.context.as_ptr() {
            eprintln!(
                "Warning: Popped context {:?} does not match expected context {:?}",
                popped_context, self.context
            );
        }
    }
}
