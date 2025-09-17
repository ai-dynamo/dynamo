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
use super::{Result, Runtime, RuntimeType, error};
use crate::config::{self, RuntimeConfig};

use futures::Future;
use once_cell::sync::OnceCell;
use std::sync::{Arc, atomic::Ordering};
use tokio::{signal, sync::Mutex, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

impl Runtime {
    fn new(runtime: RuntimeType, secondary: Option<RuntimeType>) -> Result<Runtime> {
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

        Ok(Runtime {
            id,
            primary: runtime,
            secondary,
            cancellation_token,
            endpoint_shutdown_token,
            graceful_shutdown_tracker: Arc::new(GracefulShutdownTracker::new()),
        })
    }

    pub fn from_current() -> Result<Runtime> {
        Runtime::from_handle(tokio::runtime::Handle::current())
    }

    pub fn from_handle(handle: tokio::runtime::Handle) -> Result<Runtime> {
        let primary = RuntimeType::External(handle.clone());
        let secondary = RuntimeType::External(handle);
        Runtime::new(primary, Some(secondary))
    }

    /// Create a [`Runtime`] instance from the settings
    /// See [`config::RuntimeConfig::from_settings`]
    pub fn from_settings() -> Result<Runtime> {
        let config = config::RuntimeConfig::from_settings()?;
        let runtime = Arc::new(config.create_runtime()?);
        let primary = RuntimeType::Shared(runtime.clone());
        let secondary = RuntimeType::External(runtime.handle().clone());
        Runtime::new(primary, Some(secondary))
    }

    /// Create a [`Runtime`] with two single-threaded async tokio runtime
    pub fn single_threaded() -> Result<Runtime> {
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

            // Phase 3: Now shutdown NATS/ETCD by cancelling the main token
            tracing::info!(
                "Phase 3: All graceful endpoints completed, shutting down NATS/ETCD connections"
            );
            main_token.cancel();
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

impl std::fmt::Debug for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::External(_) => write!(f, "RuntimeType::External"),
            RuntimeType::Shared(_) => write!(f, "RuntimeType::Shared"),
        }
    }
}
