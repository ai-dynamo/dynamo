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

#[cfg(all(test, feature = "testing-cuda", feature = "testing-nixl"))]
mod tests {
    use crate::block_manager::v2::physical::manager::TransportManager;
    use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
    use crate::block_manager::v2::physical::transfer::tests::cuda::CudaSleep;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn test_cuda_event_delayed_notification() {
        let agent = NixlAgent::require_backends("test_agent", &[]).unwrap();
        let manager = TransportManager::builder()
            .worker_id(0)
            .cuda_device_id(0)
            .nixl_agent(agent)
            .build()
            .unwrap();

        let stream = manager.h2d_stream();
        let cuda_ctx = manager.cuda_context();

        // Get or create the CudaSleep utility (compiles kernel and calibrates on first use)
        let cuda_sleep = CudaSleep::for_context(cuda_ctx).unwrap();

        // Test 1: Launch sleep and wait via async notification
        let t0_queue_start = Instant::now();
        cuda_sleep
            .launch(Duration::from_millis(600), stream)
            .unwrap();
        let queue_time = t0_queue_start.elapsed();

        let event = stream.record_event(None).unwrap();
        let notification = manager.register_cuda_event(event);
        notification.await.unwrap();
        let wait_time = t0_queue_start.elapsed() - queue_time;

        println!(
            "GPU sleep test: queue {:?}, wait {:?}",
            queue_time, wait_time
        );

        assert!(
            queue_time < Duration::from_millis(10),
            "launching the sleep kernel should be fast: {:?}",
            queue_time
        );

        assert!(
            wait_time >= Duration::from_millis(500),
            "wait time should reflect >=500ms of GPU work: {:?}",
            wait_time
        );
    }
}
