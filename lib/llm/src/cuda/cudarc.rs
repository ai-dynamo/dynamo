// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Implementations of the Dynamo CUDA traits for [`cudarc`] objects.

use super::{CudaContext, CudaStream};

use cudarc::driver;
use std::ptr::NonNull;

impl CudaContext for driver::CudaContext {
    unsafe fn cu_context(&self) -> NonNull<cudarc::driver::sys::CUctx_st> {
        let context = self.cu_ctx();
        NonNull::new(context).expect("CudaContext returned null context")
    }
}

impl CudaContext for driver::CudaStream {
    unsafe fn cu_context(&self) -> NonNull<cudarc::driver::sys::CUctx_st> {
        self.context().cu_context()
    }
}

impl CudaStream for driver::CudaStream {
    unsafe fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.cu_stream()
    }

    fn context(&self) -> &dyn CudaContext {
        self.context().as_ref()
    }
}
