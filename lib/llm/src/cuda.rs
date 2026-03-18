// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use cudarc::driver::{
    CudaContext, CudaStream,
    sys::{CUcontext, CUstream, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cudaError_enum},
};
use std::pin::Pin;
use std::{marker::PhantomData, sync::Arc};

pub trait DynamoCudaContextProvider {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA context.
    /// The caller must ensure that the context is valid and that the CUDA context is active.
    unsafe fn cu_context(&self) -> cudarc::driver::sys::CUcontext;

    fn bind_to_thread(&self) -> Pin<Box<DynamoCudaContextGuard>> {
        unsafe { DynamoCudaContextGuard::new(self.cu_context()) }
    }
}

pub trait DynamoCudaStreamProvider {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA stream.
    /// The caller must ensure that the stream is valid and that the CUDA context is active.
    ///
    /// Similarly, any pointers/references to data for which the stream will be accessed must
    /// have proper lifetimes and scoping, which is not guaranteed by this trait.
    unsafe fn cu_stream(&self) -> cudarc::driver::sys::CUstream;

    fn context(&self) -> Arc<dyn DynamoCudaContextProvider>;
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
    context: cudarc::driver::sys::CUcontext,
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
    pub unsafe fn new(context: CUcontext) -> Pin<Box<Self>> {
        // Push the context onto the CUDA context stack
        let result = cuCtxPushCurrent_v2(context);
        if result != cudaError_enum::CUDA_SUCCESS {
            panic!("Failed to push CUDA context: {:?}", result);
        }

        let guard = Self {
            context,
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
    pub fn context(&self) -> cudarc::driver::sys::CUcontext {
        self.context
    }
}

impl Drop for DynamoCudaContextGuard {
    fn drop(&mut self) {
        // Pop the context from the CUDA context stack when the guard is dropped
        let mut popped_context: CUcontext = std::ptr::null_mut();
        let result = unsafe { cuCtxPopCurrent_v2(&mut popped_context) };

        // Log errors but don't panic in Drop
        if result != cudaError_enum::CUDA_SUCCESS {
            eprintln!("Warning: Failed to pop CUDA context in drop: {:?}", result);
        }

        // Verify we popped the expected context
        if popped_context != self.context {
            eprintln!(
                "Warning: Popped context {:?} does not match expected context {:?}",
                popped_context, self.context
            );
        }
    }
}

/// A CUDA context provider that wraps an external CUDA context.
pub struct ExternalCudaContext {
    // SAFETY: CUcontext is thread-safe to pass between threads and can be used concurrently.
    context: CUcontext,
}

// SAFETY: See notes on CUcontext above.
unsafe impl Send for ExternalCudaContext {}
unsafe impl Sync for ExternalCudaContext {}

impl ExternalCudaContext {
    pub fn new(context: CUcontext) -> Arc<Self> {
        Arc::new(Self { context })
    }

    pub fn cu_context(&self) -> CUcontext {
        self.context
    }
}

impl DynamoCudaContextProvider for ExternalCudaContext {
    unsafe fn cu_context(&self) -> cudarc::driver::sys::CUcontext {
        self.cu_context()
    }
}

/// A CUDA stream provider that wraps an external CUDA stream.
pub struct ExternalCudaStream {
    stream: CUstream,
    context: Arc<dyn DynamoCudaContextProvider>,
}

impl ExternalCudaStream {
    pub fn new(stream: CUstream, context: Arc<dyn DynamoCudaContextProvider>) -> Self {
        Self { stream, context }
    }
}

impl DynamoCudaStreamProvider for ExternalCudaStream {
    unsafe fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.stream
    }

    fn context(&self) -> Arc<dyn DynamoCudaContextProvider> {
        self.context.clone()
    }
}

// The PhantomData<*const ()> field automatically makes this !Send and !Sync
// which prevents the guard from crossing async boundaries

// Implementations of this trait for the [`cudarc`] crate.

impl DynamoCudaContextProvider for CudaContext {
    unsafe fn cu_context(&self) -> cudarc::driver::sys::CUcontext {
        self.cu_ctx()
    }
}

impl DynamoCudaContextProvider for CudaStream {
    unsafe fn cu_context(&self) -> cudarc::driver::sys::CUcontext {
        self.context().cu_context()
    }
}

impl DynamoCudaStreamProvider for CudaStream {
    unsafe fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.cu_stream()
    }

    fn context(&self) -> Arc<dyn DynamoCudaContextProvider> {
        self.context().clone()
    }
}
