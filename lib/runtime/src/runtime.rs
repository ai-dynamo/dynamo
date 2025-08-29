// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

use super::{Result, Runtime, RuntimeType, error};
use crate::config::{self, RuntimeConfig};

use futures::Future;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokio::{signal, task::JoinHandle};

pub use tokio_util::sync::CancellationToken;

/// Shutdown mode for the runtime
#[derive(Debug, Clone, Copy)]
pub enum ShutdownMode {
    /// Immediate shutdown - all components shutdown at once, then waits for requests if graceful is enabled when serving the endpoint
    Immediate,
    /// Graceful shutdown - endpoints shutdown first, then waits for requests, then infrastructure
    Graceful,
}

impl Runtime {
    fn new(runtime: RuntimeType, secondary: Option<RuntimeType>) -> Result<Runtime> {
        // worker id
        let id = Arc::new(uuid::Uuid::new_v4().to_string());

        // create a cancellation token
        let cancellation_token = CancellationToken::new();

        // create endpoint cancellation token as a child of the main token
        // this ensures endpoints are cancelled if runtime is forcefully shutdown
        let endpoint_cancellation_token = cancellation_token.child_token();

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
            endpoint_cancellation_token,
            request_tracker: crate::RequestTracker::new(),
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

    /// Creates a child [`CancellationToken`] tied to the life-cycle of the [`Runtime`]'s root [`CancellationToken::child_token`] method.
    pub fn child_token(&self) -> CancellationToken {
        self.cancellation_token.child_token()
    }

    /// Creates a child [`CancellationToken`] for endpoints, cancelled first during graceful shutdown
    pub fn endpoint_child_token(&self) -> CancellationToken {
        self.endpoint_cancellation_token.child_token()
    }

    /// Access the endpoint [`CancellationToken`]
    pub fn endpoint_token(&self) -> CancellationToken {
        self.endpoint_cancellation_token.clone()
    }

    /// Access the request tracker for tracking in-flight requests
    pub fn request_tracker(&self) -> crate::RequestTracker {
        self.request_tracker.clone()
    }

    /// Shuts down the [`Runtime`] instance
    pub fn shutdown(&self) {
        self.shutdown_with_mode(ShutdownMode::Immediate);
    }

    /// Shuts down the [`Runtime`] instance with the specified mode
    pub fn shutdown_with_mode(&self, mode: ShutdownMode) {
        match mode {
            ShutdownMode::Immediate => {
                // Cancel everything at once
                self.cancellation_token.cancel();
            }
            ShutdownMode::Graceful => {
                // First, cancel only endpoints
                self.endpoint_cancellation_token.cancel();

                // Schedule infrastructure shutdown after all requests complete
                let token = self.cancellation_token.clone();
                let tracker = self.request_tracker.clone();
                let handle = self.primary();  // Get the runtime handle
                handle.spawn(async move {
                    tracing::info!(
                        "Endpoints shutdown initiated, waiting for {} in-flight requests to complete",
                        tracker.inflight_count()
                    );

                    // Wait for all in-flight requests to complete
                    tracker.wait_for_completion().await;

                    tracing::info!(
                        "All in-flight requests completed, shutting down runtime infrastructure"
                    );
                    token.cancel();
                });
            }
        }
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
