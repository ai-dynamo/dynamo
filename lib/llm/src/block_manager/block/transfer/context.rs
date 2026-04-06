// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use cudarc::driver::{CudaContext, CudaStream};
use dynamo_memory::{BackendType, DeviceContext, ZeCommandQueue, ZeContext, pool::CudaMemPool};
use nixl_sys::Agent as NixlAgent;

use anyhow::Result;
use dynamo_runtime::utils::pool::{Returnable, SyncPool, SyncPoolItem};
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

// ============================================================================
// Legacy: Pinned Buffer Resource for Old Pooling (to be removed)
// ============================================================================

// Pinned Buffer Resource for Pooling
#[derive(Debug)]
pub struct PinnedBuffer {
    pub ptr: u64,
    pub size: usize,
    pub id: u64,
}

impl Returnable for PinnedBuffer {
    fn on_return(&mut self) {
        tracing::debug!(
            "Returning pinned buffer {} ({}KB) to pool",
            self.id,
            self.size / 1024
        );
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        tracing::debug!(
            "Dropping pinned buffer {} ({}KB) - freeing CUDA pinned memory",
            self.id,
            self.size / 1024
        );

        unsafe {
            if let Err(e) = cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void) {
                tracing::error!(
                    "Failed to free pinned buffer {} (0x{:x}): {}",
                    self.id,
                    self.ptr,
                    e
                );
            }
        }
    }
}

pub type SyncPinnedBufferPool = SyncPool<PinnedBuffer>;

pub struct TransferResources {
    src_buffer: SyncPoolItem<PinnedBuffer>,
    dst_buffer: SyncPoolItem<PinnedBuffer>,
}

impl TransferResources {
    /// Create TransferResources by acquiring 2 buffers from the context
    pub fn acquire_for_kernel_launch(
        ctx: &TransferContext,
        address_count: usize,
    ) -> Result<Self, TransferError> {
        tracing::debug!(
            "Acquiring TransferResources for {} addresses (need 2 buffers)",
            address_count
        );

        // Acquire 2 buffers: one for src addresses, one for dst addresses
        let src_buffer = ctx.acquire_resources_for_transfer_sync(address_count)?;
        let dst_buffer = ctx.acquire_resources_for_transfer_sync(address_count)?;

        tracing::debug!(
            "TransferResources ready: src=0x{:x}, dst=0x{:x}",
            src_buffer.ptr,
            dst_buffer.ptr
        );

        Ok(Self {
            src_buffer,
            dst_buffer,
        })
    }

    /// Copy address arrays into the pinned buffers
    pub fn copy_addresses_to_buffers(
        &self,
        src_addresses: &[u64],
        dst_addresses: &[u64],
    ) -> Result<(), TransferError> {
        if src_addresses.len() != dst_addresses.len() {
            return Err(TransferError::ExecutionError(format!(
                "Address array length mismatch: src={}, dst={}",
                src_addresses.len(),
                dst_addresses.len()
            )));
        }

        let required_size = std::mem::size_of_val(src_addresses);

        if self.src_buffer.size < required_size || self.dst_buffer.size < required_size {
            return Err(TransferError::ExecutionError(format!(
                "Buffer too small: {}B needed",
                required_size
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_addresses.as_ptr(),
                self.src_buffer.ptr as *mut u64,
                src_addresses.len(),
            );
            std::ptr::copy_nonoverlapping(
                dst_addresses.as_ptr(),
                self.dst_buffer.ptr as *mut u64,
                dst_addresses.len(),
            );
        }

        tracing::debug!(
            "Copied {} address pairs to pinned buffers",
            src_addresses.len()
        );

        Ok(())
    }

    /// Get the source buffer pointer (for kernel launch)
    pub fn src_ptr(&self) -> u64 {
        self.src_buffer.ptr
    }

    /// Get the destination buffer pointer (for kernel launch)
    pub fn dst_ptr(&self) -> u64 {
        self.dst_buffer.ptr
    }
}

impl Drop for TransferResources {
    fn drop(&mut self) {
        tracing::debug!(
            "Releasing TransferResources: buffers {} & {} returning to pool",
            self.src_buffer.id,
            self.dst_buffer.id
        );
    }
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub enable_pool: bool,
    pub max_concurrent_transfers: usize,
    pub max_transfer_batch_size: usize,
    pub num_outer_components: usize,
    pub num_layers: usize,
}

// ============================================================================
// Device memory pool wrapper
// ============================================================================

/// Backend-agnostic device stream.
#[derive(Clone)]
pub enum DeviceStream {
    /// CUDA stream.
    Cuda(Arc<CudaStream>),
    /// Level Zero command queue.
    Ze(Arc<ZeCommandQueue>),
}

/// Extension trait for DeviceContext — creates a DeviceStream by downcasting
/// to the concrete backend and calling its `new_stream()` method.
pub trait DeviceContextExt {
    fn new_stream(&self) -> anyhow::Result<DeviceStream>;
}

impl DeviceContextExt for DeviceContext {
    fn new_stream(&self) -> anyhow::Result<DeviceStream> {
        match self.backend_type() {
            BackendType::Cuda => {
                let cuda_ctx = self.backend.as_any()
                    .downcast_ref::<Arc<CudaContext>>()
                    .expect("BackendType::Cuda but downcast to Arc<CudaContext> failed");
                Ok(DeviceStream::Cuda(cuda_ctx.new_stream()?))
            }
            BackendType::Ze => {
                let ze_ctx = self.backend.as_any()
                    .downcast_ref::<Arc<ZeContext>>()
                    .expect("BackendType::Ze but downcast to Arc<ZeContext> failed");
                Ok(DeviceStream::Ze(ze_ctx.new_stream()?))
            }
        }
    }
}

/// Type-erased device memory pool.
pub enum DeviceMemPool {
    /// CUDA stream-ordered memory pool.
    Cuda(Arc<CudaMemPool>),
    /// Level Zero memory pool.
    Ze(Arc<super::ze::ZeMemPool>),
}

// ============================================================================
// TransferBackend trait
// ============================================================================

/// Backend-specific operations for transfer contexts.
pub(super) trait TransferBackend: Send + Sync {
    /// Get the device memory pool, if configured.
    fn device_mem_pool(&self) -> Option<DeviceMemPool>;

    /// Record a device event and notify `tx` when it completes.
    fn device_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError>;

    /// Acquire a pinned buffer from the legacy pool.
    fn acquire_resources_for_transfer_sync(
        &self,
        size: usize,
    ) -> Result<SyncPoolItem<PinnedBuffer>, TransferError>;

    /// Get the device stream.
    fn device_stream(&self) -> DeviceStream;

    /// Shut down the backend (cancel workers, join threads).
    fn shutdown(&mut self);
}

// ============================================================================
// TransferContext
// ============================================================================

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    async_rt_handle: Handle,
    backend: Box<dyn TransferBackend>,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: DeviceStream,
        async_rt_handle: Handle,
        config: Option<PoolConfig>,
    ) -> Result<Self, anyhow::Error> {
        let backend: Box<dyn TransferBackend> = match &stream {
            DeviceStream::Cuda(cuda_stream) => {
                Box::new(super::cuda::TransferBackendCuda::new(cuda_stream.clone(), config.as_ref())?)
            }
            DeviceStream::Ze(ze_queue) => {
                Box::new(super::ze::TransferBackendZe::new(ze_queue.clone(), config.as_ref()))
            }
        };
        Ok(Self {
            nixl_agent,
            async_rt_handle,
            backend,
        })
    }

    pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
        self.nixl_agent.clone()
    }

    /// Get the device stream.
    pub fn stream(&self) -> DeviceStream {
        self.backend.device_stream()
    }

    pub fn async_rt_handle(&self) -> &Handle {
        &self.async_rt_handle
    }

    /// Get the device memory pool for stream-ordered allocations.
    pub fn device_mem_pool(&self) -> Option<DeviceMemPool> {
        self.backend.device_mem_pool()
    }

    pub fn device_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        self.backend.device_event(tx)
    }

    pub fn acquire_resources_for_transfer_sync(
        &self,
        size: usize,
    ) -> Result<SyncPoolItem<PinnedBuffer>, TransferError> {
        self.backend.acquire_resources_for_transfer_sync(size)
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        self.backend.shutdown();
    }
}

pub mod v2 {
    use super::*;

    use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags};
    use nixl_sys::Agent as NixlAgent;

    use std::sync::Arc;
    use tokio::runtime::Handle;

    #[derive(Clone)]
    pub struct TransferContext {
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
    }

    pub struct EventSynchronizer {
        event: CudaEvent,
        async_rt_handle: Handle,
    }

    impl TransferContext {
        pub fn new(
            nixl_agent: Arc<Option<NixlAgent>>,
            stream: Arc<CudaStream>,
            async_rt_handle: Handle,
        ) -> Self {
            Self {
                nixl_agent,
                stream,
                async_rt_handle,
            }
        }

        pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
            self.nixl_agent.clone()
        }

        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        pub fn async_rt_handle(&self) -> &Handle {
            &self.async_rt_handle
        }

        pub fn record_event(&self) -> Result<EventSynchronizer, TransferError> {
            let event = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| TransferError::ExecutionError(e.to_string()))?;

            Ok(EventSynchronizer {
                event,
                async_rt_handle: self.async_rt_handle.clone(),
            })
        }
    }

    impl EventSynchronizer {
        pub fn synchronize_blocking(self) -> Result<(), TransferError> {
            self.event
                .synchronize()
                .map_err(|e| TransferError::ExecutionError(e.to_string()))
        }

        pub async fn synchronize(self) -> Result<(), TransferError> {
            let event = self.event;
            self.async_rt_handle
                .spawn_blocking(move || {
                    event
                        .synchronize()
                        .map_err(|e| TransferError::ExecutionError(e.to_string()))
                })
                .await
                .map_err(|e| TransferError::ExecutionError(format!("Task join error: {}", e)))?
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_transfer_context_is_cloneable() {
            // Compile-time test: TransferContext should implement Clone
            // This is important for concurrent usage scenarios
            fn assert_clone<T: Clone>() {}
            assert_clone::<TransferContext>();
        }

        #[test]
        fn test_event_synchronizer_consumes_on_use() {
            // Compile-time test: EventSynchronizer should be consumed by sync methods
            // This ensures proper resource management and prevents double-use

            // We can verify this by checking that EventSynchronizer doesn't implement Clone
            // (This is a documentation test since negative trait bounds aren't stable)
        }
    }

    #[cfg(all(test, feature = "testing-cuda"))]
    mod integration_tests {
        use super::*;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;
        use tokio_util::task::TaskTracker;

        fn setup_context() -> TransferContext {
            let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
            let stream = ctx.default_stream();
            let nixl_agent = Arc::new(None);
            let handle = tokio::runtime::Handle::current();

            TransferContext::new(nixl_agent, stream, handle)
        }

        #[tokio::test]
        async fn test_basic_event_synchronization() {
            let ctx = setup_context();

            // Test blocking synchronization
            let event = ctx.record_event().expect("Failed to record event");
            event.synchronize_blocking().expect("Blocking sync failed");

            // Test async synchronization
            let event = ctx.record_event().expect("Failed to record event");
            event.synchronize().await.expect("Async sync failed");
        }

        #[tokio::test]
        async fn test_context_cloning_works() {
            let ctx = setup_context();
            let ctx_clone = ctx.clone();

            // Both contexts should work independently
            let event1 = ctx
                .record_event()
                .expect("Failed to record event on original");
            let event2 = ctx_clone
                .record_event()
                .expect("Failed to record event on clone");

            // Both should synchronize successfully
            event1
                .synchronize_blocking()
                .expect("Original context sync failed");
            event2
                .synchronize()
                .await
                .expect("Cloned context sync failed");
        }

        #[tokio::test]
        async fn test_concurrent_synchronization() {
            let ctx = setup_context();
            let tracker = TaskTracker::new();

            // Spawn multiple concurrent synchronization tasks
            for i in 0..5 {
                let ctx_clone = ctx.clone();
                tracker.spawn(async move {
                    let event = ctx_clone
                        .record_event()
                        .unwrap_or_else(|_| panic!("Failed to record event {}", i));
                    event
                        .synchronize()
                        .await
                        .unwrap_or_else(|_| panic!("Failed to sync event {}", i));
                });
            }

            tracker.close();
            tracker.wait().await;
        }

        #[tokio::test]
        async fn test_error_handling() {
            let ctx = setup_context();

            // Test that we get proper error types on failure
            // Note: This test is limited since we can't easily force CUDA errors
            // in a controlled way, but we verify the error path exists

            let event = ctx.record_event().expect("Failed to record event");
            let result = event.synchronize().await;

            // In normal conditions this should succeed, but if it fails,
            // it should return a TransferError
            match result {
                Ok(_) => {}                                 // Expected in normal conditions
                Err(TransferError::ExecutionError(_)) => {} // Expected error type
                Err(other) => panic!("Unexpected error type: {:?}", other),
            }
        }

        #[tokio::test]
        async fn test_resource_cleanup() {
            // Test that contexts and events can be dropped properly
            let ctx = setup_context();

            // Create and immediately drop an event synchronizer
            {
                let _event = ctx.record_event().expect("Failed to record event");
                // _event goes out of scope here without being synchronized
            }

            // Context should still work after dropping unused events
            let event = ctx
                .record_event()
                .expect("Failed to record event after cleanup");
            event
                .synchronize()
                .await
                .expect("Sync after cleanup failed");
        }
    }
}
