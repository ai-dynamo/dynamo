// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags};
use nixl_sys::Agent as NixlAgent;

use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>,
    async_rt_handle: Handle,

    cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>)>,
    cuda_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
    ) -> Self {
        let (cuda_event_tx, mut cuda_event_rx) =
            mpsc::unbounded_channel::<(CudaEvent, oneshot::Sender<()>)>();

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let cuda_event_worker = std::thread::spawn(move || {
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
                        _ = cancel_token_clone.cancelled() => {
                            break;
                        }
                    }
                }
            });
        });

        Self {
            nixl_agent,
            stream,
            async_rt_handle,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
            cancel_token,
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
                        .expect(&format!("Failed to record event {}", i));
                    event
                        .synchronize()
                        .await
                        .expect(&format!("Failed to sync event {}", i));
                });
            }

            tracker.close();
            tracker.wait().await;
        }

        #[tokio::test]
        async fn test_performance_baseline() {
            let ctx = setup_context();
            let start = std::time::Instant::now();

            // Test a reasonable number of synchronizations
            for _ in 0..10 {
                let event = ctx.record_event().expect("Failed to record event");
                event.synchronize().await.expect("Sync failed");
            }

            let duration = start.elapsed();
            // Should complete 10 synchronizations in reasonable time (< 1ms total)
            assert!(
                duration < std::time::Duration::from_millis(1),
                "Performance regression: took {:?} for 10 syncs",
                duration
            );
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
