// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use crate::config::{self, RuntimeConfig};
use loom_rs::LoomRuntime;
pub use loom_rs::{ComputeHint, current_runtime};

use futures::Future;
use once_cell::sync::OnceCell;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, atomic::Ordering},
};
use tokio::{signal, sync::Mutex, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone, Debug)]
enum RuntimeType {
    Shared(Arc<ManuallyDrop<tokio::runtime::Runtime>>),
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
    loom: LoomRuntime,
}

impl Runtime {
    pub(crate) fn new(loom: LoomRuntime) -> anyhow::Result<Runtime> {
        // worker id
        let id = Arc::new(uuid::Uuid::new_v4().to_string());

        // create a cancellation token
        let cancellation_token = CancellationToken::new();

        // create endpoint shutdown token as a child of the main token
        let endpoint_shutdown_token = cancellation_token.child_token();

        let primary = RuntimeType::External(loom.tokio_handle().clone());
        let secondary = RuntimeType::External(loom.tokio_handle().clone());

        Ok(Runtime {
            id,
            primary,
            secondary,
            cancellation_token,
            endpoint_shutdown_token,
            graceful_shutdown_tracker: Arc::new(GracefulShutdownTracker::new()),
            loom,
        })
    }

    pub fn from_current() -> anyhow::Result<Runtime> {
        let loom = loom_rs::current_runtime().expect("Failed to get current runtime");
        Runtime::new(loom)
    }

    /// Create a [`Runtime`] instance from the settings
    /// See [`config::RuntimeConfig::from_settings`]
    pub fn from_settings() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::from_settings()?;
        Runtime::new(config.create_runtime()?)
    }

    /// Create a [`Runtime`] with two single-threaded async tokio runtime
    pub fn single_threaded() -> anyhow::Result<Runtime> {
        let config = config::RuntimeConfig::single_threaded();
        Runtime::new(config.create_runtime()?)
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

            // Phase 3: Now connections will be disconnected to backend services (e.g. NATS/ETCD) by cancelling the main token
            tracing::info!(
                "Phase 3: All endpoints ended gracefully. Connections to backend services will now be disconnected"
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

/// Handle dropping a tokio runtime from an async context.
///
/// When used from the Python bindings the runtime will be dropped from (I think) Python's asyncio.
/// Tokio does not allow this and will panic. That panic prevents logging from printing it's last
/// messages, which makes knowing what went wrong very difficult.
///
/// This is the panic:
/// > pyo3_runtime.PanicException: Cannot drop a runtime in a context where blocking is not allowed.
/// > This happens when a runtime is dropped from within an asynchronous context.
///
/// Hence we wrap the runtime in a ManuallyDrop and use tokio's alternative shutdown if we detect
/// that we are inside an async runtime.
impl Drop for RuntimeType {
    fn drop(&mut self) {
        match self {
            RuntimeType::External(_) => {}
            RuntimeType::Shared(arc) => {
                let Some(md_runtime) = Arc::get_mut(arc) else {
                    // Only drop if we are the only owner of the shared pointer, meaning
                    // one strong count and no weak count.
                    return;
                };
                if tokio::runtime::Handle::try_current().is_ok() {
                    // We are inside an async runtime.
                    let tokio_runtime = unsafe { ManuallyDrop::take(md_runtime) };
                    tokio_runtime.shutdown_background();
                } else {
                    // We are not inside an async context, dropping the runtime is safe.
                    //
                    // We never reach this case. I'm not sure why, something about the interaction
                    // with pyo3 and Python lifetimes.
                    //
                    // Process is gone so doesn't really matter, but TODO now that we realize it.
                    unsafe { ManuallyDrop::drop(md_runtime) };
                }
            }
        }
    }
}
