// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device event polling-based completion checker (multi-backend).
//!
//! This replaces the CUDA-specific CudaEventChecker with a backend-agnostic
//! implementation that works with any DeviceEvent (CUDA, HPU, XPU).

use anyhow::Result;

use super::CompletionChecker;
use crate::block_manager::v2::device::DeviceEvent;

/// Completion checker that polls device event status (supports CUDA, HPU, XPU).
pub struct DeviceEventChecker {
    event: DeviceEvent,
}

impl DeviceEventChecker {
    pub fn new(event: DeviceEvent) -> Self {
        Self { event }
    }
}

impl CompletionChecker for DeviceEventChecker {
    fn is_complete(&self) -> Result<bool> {
        // Use the device abstraction's is_complete method
        self.event.is_complete()
    }
}

#[cfg(all(test, feature = "testing-cuda", feature = "testing-nixl"))]
mod tests {
    use crate::block_manager::v2::device::{DeviceBackend, DeviceContext};
    use crate::block_manager::v2::physical::manager::TransportManager;
    use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
    use crate::block_manager::v2::physical::transfer::tests::cuda::CudaSleep;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn test_device_event_delayed_notification() {
        // Auto-detect or use CUDA for testing
        let backend = DeviceBackend::auto_detect().unwrap();
        let device_ctx = DeviceContext::new(backend, 0).unwrap();

        let agent = NixlAgent::require_backends("test_agent", &[]).unwrap();
        let manager = TransportManager::builder()
            .worker_id(0)
            .device_backend(backend)
            .device_id(0)
            .nixl_agent(agent)
            .build()
            .unwrap();

        let stream = manager.h2d_stream();

        // Only run CUDA-specific sleep test if backend is CUDA
        if matches!(backend, DeviceBackend::Cuda) {
            let cuda_ctx_ops = device_ctx.as_cuda_context().unwrap();
            let cuda_ctx = cuda_ctx_ops.inner();

            // Get or create the CudaSleep utility
            let cuda_sleep = CudaSleep::for_context(cuda_ctx).unwrap();

            // Launch sleep and wait via async notification
            let t0_queue_start = Instant::now();
            cuda_sleep
                .launch(Duration::from_millis(600), stream.as_cuda_stream().unwrap())
                .unwrap();
            let queue_time = t0_queue_start.elapsed();

            let event = stream.record_event().unwrap();
            let notification = manager.register_device_event(event);
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
}
