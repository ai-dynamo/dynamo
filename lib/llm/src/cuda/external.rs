// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    check_cuda,
    sys::{CUcontext, CUstream},
    CudaContext, CudaStream,
};

use cudarc::driver::{
    sys::{cuCtxGetCurrent, cuStreamGetCtx},
    DriverError,
};

use std::{ptr::NonNull, sync::Arc};

/// A CUDA context provider that wraps an external CUDA context.
///
/// This object is a convenience object that will provide reusable implementations of methods callable
/// with a [`CUcontext`] handle.
///
/// The object itself is not safe in anyway, it does not hold ownership of the context. Proper usage of
/// this requires the implementor to ensure that the context is valid and that the CUDA context is active.
pub struct ExternalCudaContext {
    // SAFETY: CUcontext is thread-safe to pass between threads and can be used concurrently.
    // Using NonNull ensures we never have null contexts.
    context: NonNull<cudarc::driver::sys::CUctx_st>,
}

// SAFETY: See notes on CUcontext above.
unsafe impl Send for ExternalCudaContext {}
unsafe impl Sync for ExternalCudaContext {}

impl ExternalCudaContext {
    fn new(context: CUcontext) -> Arc<Self> {
        let context_nonnull = NonNull::new(context).expect("CUDA context cannot be null");
        Arc::new(Self {
            context: context_nonnull,
        })
    }

    pub fn from_current() -> Result<Arc<Self>, DriverError> {
        let mut context: CUcontext = std::ptr::null_mut();
        check_cuda(unsafe { cuCtxGetCurrent(&mut context) })?;
        Ok(Self::new(context))
    }

    pub fn cu_context(&self) -> CUcontext {
        self.context.as_ptr()
    }
}

impl CudaContext for ExternalCudaContext {
    unsafe fn cu_context(&self) -> NonNull<cudarc::driver::sys::CUctx_st> {
        self.context
    }
}

/// A CUDA stream provider that wraps an external CUDA stream.
///
/// # Safety
///
/// It is the responsibility for the creator of an [`ExternalCudaStream`] that it must take ownership
/// of the cuda stream and it will be responsible for maintaining the stream.
pub struct ExternalCudaStream {
    stream: CUstream,
    context: Arc<dyn CudaContext>,
}

unsafe impl Send for ExternalCudaStream {}
unsafe impl Sync for ExternalCudaStream {}

impl ExternalCudaStream {
    /// # Safety
    ///
    /// This method is unsafe because it directly accesses the underlying CUDA stream.
    /// The caller must ensure that the stream is valid and that the CUDA context is active.
    ///
    /// Similarly, any pointers/references to data for which the stream will be accessed must
    /// have proper lifetimes and scoping, which is not guaranteed by this object.
    pub unsafe fn new(stream: CUstream) -> Result<Self, DriverError> {
        // Validate the stream by getting its context
        let mut context: CUcontext = std::ptr::null_mut();
        check_cuda(unsafe { cuStreamGetCtx(stream, &mut context) })?;

        // Create a new context provider for the stream
        let context = ExternalCudaContext::new(context);

        Ok(Self { stream, context })
    }
}

impl CudaStream for ExternalCudaStream {
    unsafe fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.stream
    }

    fn context(&self) -> &dyn CudaContext {
        self.context.as_ref()
    }
}

/*
// Note[oandreeva]: Disabling this for now

// The PhantomData<*const ()> field automatically makes this !Send and !Sync
// which prevents the guard from crossing async boundaries

/// A CUDA event provider that wraps an external CUDA event.
pub struct ExternalCudaEvent {
    event: CUevent,
    // todo: extract details on if the event is blocking, timed, etc.
    // todo: we will await blocking/async events differently
}

unsafe impl Send for ExternalCudaEvent {}
unsafe impl Sync for ExternalCudaEvent {}

impl ExternalCudaEvent {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(event: CUevent) -> Self {
        // TODO: extract flags from the event
        Self { event }
    }
}

impl CudaEvent for ExternalCudaEvent {
    unsafe fn cu_event(&self) -> CUevent {
        self.event
    }
}
*/
