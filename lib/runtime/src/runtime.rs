// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [Runtime] module is the interface for [crate::component::Component]
//! to access shared resources. These include thread pool, memory allocators and other shared resources.
//!
//! The [Runtime] holds the primary [`CancellationToken`] which can be used to terminate all attached
//! [`crate::component::Component`].
//!
//! We expect in the future to offer topologically aware thread and memory resources, but for now the
//! set of resources is limited to the thread pool and cancellation token.
//!
//! Notes: We will need to do an evaluation on what is fully public, what is pub(crate) and what is
//! private; however, for now we are exposing most objects as fully public while the API is maturing.

use super::utils::GracefulShutdownTracker;
use crate::{
    compute,
    config::{self, RuntimeConfig},
};

use futures::Future;
use once_cell::sync::OnceCell;
use std::sync::{Arc, atomic::Ordering};
use tokio::{signal, sync::Mutex, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone, Debug)]
enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
    endpoint_shutdown_token: CancellationToken,
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
    compute_pool: Option<Arc<compute::ComputePool>>,
    block_in_place_permits: Option<Arc<tokio::sync::Semaphore>>,
}

impl Runtime {
    fn new(runtime: RuntimeType, secondary: Option<RuntimeType>) -> anyhow::Result<Runtime> {
        // worker id
        let id = Arc::new(uuid::Uuid::new_v4().to_string());

        // create a cancellation token
        let cancellation_token = CancellationToken::new();

        // create endpoint shutdown token as a child of the main token
        let endpoint_shutdown_token = cancellation_token.child_token();

        // secondary runtime for background ectd/nats tasks
        let secondary = match secondary {
            Some(secondary) => secondary,
            None => {
                tracing::debug!("Created secondary runtime with single thread");
                RuntimeType::Shared(Arc::new(RuntimeConfig::single_threaded().create_runtime()?))
            }
        };

        // Initialize compute pool with default config
        // This will be properly configured when created from RuntimeConfig
        let compute_pool = None;
        let block_in_place_permits = None;

        Ok(Runtime {
            id,
            primary: runtime,
            secondary,
            cancellation_token,
            endpoint_shutdown_token,
            graceful_shutdown_tracker: Arc::new(GracefulShutdownTracker::new()),
            compute_pool,
            block_in_place_permits,
        })
    }

    fn new_with_config(
        runtime: RuntimeType,
        secondary: Option<RuntimeType>,
        config: &RuntimeConfig,
    ) -> anyhow::Result<Runtime> {
        let mut rt = Self::new(runtime, secondary)?;

        // Create compute pool from configuration
        let compute_config = crate::compute::ComputeConfig {
            num_threads: config.compute_threads,
            stack_size: config.compute_stack_size,
            thread_prefix: config.compute_thread_prefix.clone(),
            pin_threads: false,
        };

        // Check if compute pool is explicitly disabled
        if config.compute_threads == Some(0) {
            tracing::info!("Compute pool disabled (compute_threads = 0)");
        } else {
            match crate::compute::ComputePool::new(compute_config) {
                Ok(pool) => {
                    rt.compute_pool = Some(Arc::new(pool));
                    tracing::debug!(
                        "Initialized compute pool with {} threads",
                        rt.compute_pool.as_ref().unwrap().num_threads()
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to create compute pool: {}. CPU-intensive operations will use spawn_blocking",
                        e
                    );
                }
            }
        }

        // Initialize block_in_place semaphore based on actual worker threads
        let num_workers = config
            .num_worker_threads
            .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get());
        // Reserve at least one thread for async work
        let permits = num_workers.saturating_sub(1).max(1);
        rt.block_in_place_permits = Some(Arc::new(tokio::sync::Semaphore::new(permits)));
        tracing::debug!(
            "Initialized block_in_place permits: {} (from {} worker threads)",
            permits,
            num_workers
        );

        Ok(rt)
    }

    /// Initialize thread-local compute context on the current thread
    /// This should be called on each Tokio worker thread
    pub fn initialize_thread_local(&self) {
        if let (Some(pool), Some(permits)) = (&self.compute_pool, &self.block_in_place_permits) {
            crate::compute::thread_local::initialize_context(Arc::clone(pool), Arc::clone(permits));
        }
    }

    /// Initialize thread-local compute context on all worker threads using a barrier
    /// This ensures every worker thread has its thread-local context initialized
    pub async fn initialize_all_thread_locals(&self) -> anyhow::Result<()> {
        if let (Some(pool), Some(permits)) = (&self.compute_pool, &self.block_in_place_permits) {
            // First, detect how many worker threads we actually have
            let num_workers = self.detect_worker_thread_count().await;

            if num_workers == 0 {
                return Err(anyhow::anyhow!("No worker threads detected"));
            }

            // Create a barrier that all threads must reach
            let barrier = Arc::new(std::sync::Barrier::new(num_workers));
            let init_pool = Arc::clone(pool);
            let init_permits = Arc::clone(permits);

            // Spawn exactly one blocking task per worker thread
            let mut handles = Vec::new();
            for i in 0..num_workers {
                let barrier_clone = Arc::clone(&barrier);
                let pool_clone = Arc::clone(&init_pool);
                let permits_clone = Arc::clone(&init_permits);

                let handle = tokio::task::spawn_blocking(move || {
                    // Wait at barrier - ensures all threads are participating
                    barrier_clone.wait();

                    // Now initialize thread-local storage
                    crate::compute::thread_local::initialize_context(pool_clone, permits_clone);

                    // Get thread ID for logging
                    let thread_id = std::thread::current().id();
                    tracing::trace!(
                        "Initialized thread-local compute context on thread {:?} (worker {})",
                        thread_id,
                        i
                    );
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle.await?;
            }

            tracing::info!(
                "Successfully initialized thread-local compute context on {} worker threads",
                num_workers
            );
        } else {
            tracing::debug!("No compute pool configured, skipping thread-local initialization");
        }
        Ok(())
    }

    /// Detect the number of worker threads in the runtime
    async fn detect_worker_thread_count(&self) -> usize {
        use parking_lot::Mutex;
        use std::collections::HashSet;

        let thread_ids = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();

        // Spawn many blocking tasks to ensure we hit all threads
        // We use spawn_blocking because it runs on worker threads
        let num_probes = 100;
        for _ in 0..num_probes {
            let ids = Arc::clone(&thread_ids);
            let handle = tokio::task::spawn_blocking(move || {
                let thread_id = std::thread::current().id();
                ids.lock().insert(thread_id);
            });
            handles.push(handle);
        }

        // Wait for all probes to complete
        for handle in handles {
            let _ = handle.await;
        }

        let count = thread_ids.lock().len();
        tracing::debug!("Detected {} worker threads in runtime", count);
        count
    }

    pub fn from_current() -> anyhow::Result<Runtime> {
        Runtime::from_handle(tokio::runtime::Handle::current())
    }

    pub fn from_handle(handle: tokio::runtime::Handle) -> anyhow::Result<Runtime> {
        let primary = RuntimeType::External(handle.clone());
        let secondary = RuntimeType::External(handle);
        Runtime::new(primary, Some(secondary))
    }

    /// Create a [`Runtime`] instance from the settings
    /// See [`config::RuntimeConfig::from_settings`]
    pub fn from_settings() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::from_settings()?;
        let runtime = Arc::new(config.create_runtime()?);
        let primary = RuntimeType::Shared(runtime.clone());
        let secondary = RuntimeType::External(runtime.handle().clone());
        Runtime::new_with_config(primary, Some(secondary), &config)
    }

    /// Create a [`Runtime`] with two single-threaded async tokio runtime
    pub fn single_threaded() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::single_threaded();
        let owned = RuntimeType::Shared(Arc::new(config.create_runtime()?));
        Runtime::new(owned, None)
    }

    /// Returns the unique identifier for the [`Runtime`]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns a [`tokio::runtime::Handle`] for the primary/application thread pool
    pub fn primary(&self) -> tokio::runtime::Handle {
        self.primary.handle()
    }

    /// Returns a [`tokio::runtime::Handle`] for the secondary/background thread pool
    pub fn secondary(&self) -> tokio::runtime::Handle {
        self.secondary.handle()
    }

    /// Access the primary [`CancellationToken`] for the [`Runtime`]
    pub fn primary_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Creates a child [`CancellationToken`] tied to the life-cycle of the [`Runtime`]'s endpoint shutdown token.
    pub fn child_token(&self) -> CancellationToken {
        self.endpoint_shutdown_token.child_token()
    }

    /// Get access to the graceful shutdown tracker
    pub(crate) fn graceful_shutdown_tracker(&self) -> Arc<GracefulShutdownTracker> {
        self.graceful_shutdown_tracker.clone()
    }

    /// Get access to the compute pool for CPU-intensive operations
    ///
    /// Returns None if the compute pool was not initialized (e.g., due to configuration error)
    pub fn compute_pool(&self) -> Option<&Arc<crate::compute::ComputePool>> {
        self.compute_pool.as_ref()
    }

    /// Shuts down the [`Runtime`] instance
    pub fn shutdown(&self) {
        tracing::info!("Runtime shutdown initiated");

        // Spawn the shutdown coordination task BEFORE cancelling tokens
        let tracker = self.graceful_shutdown_tracker.clone();
        let main_token = self.cancellation_token.clone();
        let endpoint_token = self.endpoint_shutdown_token.clone();

        // Use the runtime handle to spawn the task
        let handle = self.primary();
        handle.spawn(async move {
            // Phase 1: Cancel endpoint shutdown token to stop accepting new requests
            tracing::info!("Phase 1: Cancelling endpoint shutdown token");
            endpoint_token.cancel();

            // Phase 2: Wait for all graceful endpoints to complete
            tracing::info!("Phase 2: Waiting for graceful endpoints to complete");

            let count = tracker.get_count();
            tracing::info!("Active graceful endpoints: {}", count);

            if count != 0 {
                tracker.wait_for_completion().await;
            }

            // Phase 3: Now connections will be disconnected to NATS/ETCD by cancelling the main token
            tracing::info!(
                "Phase 3: All endpoints ended gracefully. Connections to NATS/ETCD will now be disconnected"
            );
            main_token.cancel();

            // TODO: We should likely call shutdown_background on tokio rt to stop it cleanly.
        });
    }
}

impl RuntimeType {
    /// Get [`tokio::runtime::Handle`] to runtime
    pub fn handle(&self) -> tokio::runtime::Handle {
        match self {
            RuntimeType::External(rt) => rt.clone(),
            RuntimeType::Shared(rt) => rt.handle().clone(),
        }
    }
}
