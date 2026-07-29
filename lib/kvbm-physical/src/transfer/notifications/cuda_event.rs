// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA event polling-based completion checker.

use anyhow::Result;
use cudarc::driver::{CudaEvent, DriverError, result as cuda_result, sys::CUresult};

use super::CompletionChecker;

/// Completion checker that polls CUDA event status.
pub struct CudaEventChecker {
    event: CudaEvent,
}

impl CudaEventChecker {
    pub fn new(event: CudaEvent) -> Self {
        Self { event }
    }
}

impl CompletionChecker for CudaEventChecker {
    fn is_complete(&self) -> Result<bool> {
        // Query the CUDA event to check if it's complete
        // cudaEventQuery returns cudaSuccess if complete, cudaErrorNotReady if still pending
        unsafe {
            match cuda_result::event::query(self.event.cu_event()) {
                Ok(()) => Ok(true), // Event is complete
                Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
                Err(e) => Err(anyhow::anyhow!("CUDA event query failed: {:?}", e)),
            }
        }
    }
}
