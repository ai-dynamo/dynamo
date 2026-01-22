// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags};
use nixl_sys::Agent as NixlAgent;

use anyhow::Result;
use dynamo_memory::pool::CudaMemPool;
use dynamo_runtime::utils::pool::{Returnable, SyncPool, SyncPoolItem};
use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

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
        // Returns (), not pointers
        if src_addresses.len() != dst_addresses.len() {
            return Err(TransferError::ExecutionError(format!(
                "Address array length mismatch: src={}, dst={}",
                src_addresses.len(),
                dst_addresses.len()
            )));
        }

        let required_size = std::mem::size_of_val(src_addresses);

        // Check buffer sizes
        if self.src_buffer.size < required_size || self.dst_buffer.size < required_size {
            return Err(TransferError::ExecutionError(format!(
                "Buffer too small: {}B needed",
                required_size
            )));
        }

        // Copy addresses to pinned buffers
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
        // SyncPoolItem Drop handles returning buffers to pool automatically
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

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>,
    async_rt_handle: Handle,

    // NEW: CUDA memory pool for stream-ordered host memory allocation
    cuda_mem_pool: Option<Arc<CudaMemPool>>,

    // LEGACY: Old pinned buffer pool (still used by TransferResources)
    pinned_buffer_pool: Option<SyncPinnedBufferPool>,

    cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>)>,
    cuda_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
        config: Option<PoolConfig>,
    ) -> Result<Self, anyhow::Error> {
        let (cuda_event_tx, cuda_event_rx) =
            mpsc::unbounded_channel::<(CudaEvent, oneshot::Sender<()>)>();

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let cuda_event_worker = Self::setup_cuda_event_worker(cuda_event_rx, cancel_token_clone);

        let pool = {
            tracing::debug!(
                "Pinned buffer pool is no longer used for kernel transfers and will be removed in the future"
            );
            None
        };

        // Create CUDA memory pool for stream-ordered allocation
        let cuda_mem_pool = if let Some(ref cfg) = config {
            if cfg.enable_pool {
                // Calculate total reserve size for pre-warming
                let num_buffers = cfg.max_concurrent_transfers * 2 + 2;
                let buffer_size = cfg.max_transfer_batch_size
                    * cfg.num_outer_components
                    * cfg.num_layers
                    * std::mem::size_of::<u64>();
                let reserve_size = num_buffers * buffer_size;

                tracing::info!(
                    "Creating CUDA memory pool: {} buffers × {}KB = {}MB total",
                    num_buffers,
                    buffer_size / 1024,
                    reserve_size / (1024 * 1024)
                );

                let pool = CudaMemPool::builder(stream.context().clone(), reserve_size)
                    .release_threshold(128 * 1024 * 1024) // Release memory above 128MB back to OS
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to create CUDA memory pool: {}", e))?;

                tracing::info!(
                    "CUDA memory pool created successfully (DEVICE memory, stream-ordered allocation, pre-warmed with {}MB)",
                    reserve_size / (1024 * 1024)
                );
                Some(Arc::new(pool))
            } else {
                tracing::debug!("CUDA memory pool disabled by configuration");
                None
            }
        } else {
            tracing::debug!("No pool configuration provided - CUDA memory pool disabled");
            None
        };

        Ok(Self {
            nixl_agent,
            stream,
            async_rt_handle,
            cuda_mem_pool,
            pinned_buffer_pool: pool,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
            cancel_token,
        })
    }

    fn setup_cuda_event_worker(
        mut cuda_event_rx: mpsc::UnboundedReceiver<(CudaEvent, oneshot::Sender<()>)>,
        cancel_token: CancellationToken,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime for CUDA event worker.");

            runtime.block_on(async move {
                loop {
                    tokio::select! {
                        Some((event, tx)) = cuda_event_rx.recv() => {
                            if let Err(e) = event.synchronize() {
                                tracing::error!("Error synchronizing CUDA event: {}", e);
                            }
                            let _ = tx.send(());
                        }
                        _ = cancel_token.cancelled() => {
                            break;
                        }
                    }
                }
            });
        })
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

    /// Get the CUDA memory pool for stream-ordered allocations
    pub fn cuda_mem_pool(&self) -> Option<&Arc<CudaMemPool>> {
        self.cuda_mem_pool.as_ref()
    }

    pub fn cuda_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        let event = self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(|e| TransferError::ExecutionError(e.to_string()))?;

        self.cuda_event_tx
            .send((event, tx))
            .map_err(|_| TransferError::ExecutionError("CUDA event worker exited.".into()))?;
        Ok(())
    }

    pub fn acquire_resources_for_transfer_sync(
        &self,
        size: usize,
    ) -> Result<SyncPoolItem<PinnedBuffer>, TransferError> {
        let ptr_array_size = size * std::mem::size_of::<u64>();

        tracing::debug!(
            "Acquiring pinned buffer: need {} bytes for {} addresses",
            ptr_array_size,
            size
        );

        if let Some(pool) = &self.pinned_buffer_pool {
            tracing::debug!("Pool available - acquiring buffer (blocking)...");

            // All buffers are the same size, so just acquire one directly
            let buffer = pool.acquire_blocking();

            // Validate that the requested size fits in the buffer
            if buffer.size < ptr_array_size {
                return Err(TransferError::ExecutionError(format!(
                    "Buffer too small: need {}KB but buffer is only {}KB (addresses: {})",
                    ptr_array_size / 1024,
                    buffer.size / 1024,
                    size
                )));
            }

            Ok(buffer)
        } else {
            tracing::warn!(
                "No pinned buffer pool configured - this should not happen in production"
            );
            // No pool configured - this is a configuration error
            Err(TransferError::ExecutionError(
                "No sync pool configured - TransferContext must be created with a pool".into(),
            ))
        }
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        if let Some(handle) = self.cuda_event_worker.take()
            && let Err(e) = handle.join()
        {
            tracing::error!("Error joining CUDA event worker: {:?}", e);
        }
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
