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

//! The [Worker] class is a convenience wrapper around the construction of the [Runtime]
//! and execution of the users application.
//!
//! In the future, the [Worker] should probably be moved to a procedural macro similar
//! to the `#[tokio::main]` attribute, where we might annotate an async main function with
//! `#[dynamo::main]` or similar.
//!
//! The [Worker::execute] method is designed to be called once from main and will block
//! the calling thread until the application completes or is canceled. The method initialized
//! the signal handler used to trap `SIGINT` and `SIGTERM` signals and trigger a graceful shutdown.
//!
//! On termination, the user application is given a graceful shutdown period of controlled by
//! the [DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT] environment variable. If the application does not
//! shutdown in time, the worker will terminate the application with an exit code of 911.
//!
//! The default values of [DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT] differ between the development
//! and release builds. In development, the default is [DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG] and
//! in release, the default is [DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE].

use super::{error, CancellationToken, Result, Runtime, RuntimeConfig};

use futures::Future;
use once_cell::sync::OnceCell;
use std::{sync::Mutex, time::Duration};
use tokio::{signal, task::JoinHandle};

static RT: OnceCell<tokio::runtime::Runtime> = OnceCell::new();
static RTHANDLE: OnceCell<tokio::runtime::Handle> = OnceCell::new();
static INIT: OnceCell<Mutex<Option<tokio::task::JoinHandle<Result<()>>>>> = OnceCell::new();

const SHUTDOWN_MESSAGE: &str =
    "Application received shutdown signal; attempting to gracefully shutdown";
const SHUTDOWN_TIMEOUT_MESSAGE: &str =
    "Use DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT to control the graceful shutdown timeout";

/// Environment variable to control the graceful shutdown timeout
pub const DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT: &str = "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT";

/// Default graceful shutdown timeout in seconds in debug mode
pub const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG: u64 = 5;

/// Default graceful shutdown timeout in seconds in release mode
pub const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE: u64 = 30;

#[derive(Debug, Clone)]
pub struct Worker {
    runtime: Runtime,
}

impl Worker {
    /// Create a new [`Worker`] instance from [`RuntimeConfig`] settings which is sourced from the environment
    pub fn from_settings() -> Result<Worker> {
        let config = RuntimeConfig::from_settings()?;
        Worker::from_config(config)
    }

    /// Create a new [`Worker`] instance from a provided [`RuntimeConfig`]
    pub fn from_config(config: RuntimeConfig) -> Result<Worker> {
        // if the runtime is already initialized, return an error
        if RT.get().is_some() || RTHANDLE.get().is_some() {
            return Err(error!("Worker already initialized"));
        }

        // create a new runtime and insert it into the OnceCell
        // there is still a potential race-condition here, two threads cou have passed the first check
        // but only one will succeed in inserting the runtime
        let rt = RT.try_insert(config.create_runtime()?).map_err(|_| {
            error!("Failed to create worker; Only a single Worker should ever be created")
        })?;

        let runtime = Runtime::from_handle(rt.handle().clone())?;
        Ok(Worker { runtime })
    }

    pub fn tokio_runtime(&self) -> Result<&'static tokio::runtime::Runtime> {
        RT.get().ok_or_else(|| error!("Worker not initialized"))
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn execute<F, Fut>(self, f: F) -> Result<()>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        runtime.secondary().block_on(self.execute_internal(f))??;
        runtime.shutdown();
        Ok(())
    }

    pub async fn execute_async<F, Fut>(self, f: F) -> Result<()>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        let task = self.execute_internal(f);
        task.await??;
        runtime.shutdown();
        Ok(())
    }

    /// Executes the provided application/closure on the [`Runtime`].
    /// This is designed to be called once from main and will block the calling thread until the application completes.
    fn execute_internal<F, Fut>(self, f: F) -> JoinHandle<Result<()>>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        let primary = runtime.primary();
        let secondary = runtime.secondary();

        let timeout = std::env::var(DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or({
                if cfg!(debug_assertions) {
                    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG
                } else {
                    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE
                }
            });

        INIT.set(Mutex::new(Some(secondary.spawn(async move {
            // start signal handler
            tokio::spawn(signal_handler(runtime.cancellation_token.clone()));

            let cancel_token = runtime.child_token();
            let (mut app_tx, app_rx) = tokio::sync::oneshot::channel::<()>();

            // spawn a task to run the application
            let task: JoinHandle<Result<()>> = primary.spawn(async move {
                let _rx = app_rx;
                f(runtime).await
            });

            tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::debug!("{}", SHUTDOWN_MESSAGE);
                    tracing::debug!("{} {} seconds", SHUTDOWN_TIMEOUT_MESSAGE, timeout);
                }

                _ = app_tx.closed() => {
                }
            };

            let result = tokio::select! {
                result = task => {
                    result
                }

                _ = tokio::time::sleep(tokio::time::Duration::from_secs(timeout)) => {
                    tracing::debug!("Application did not shutdown in time; terminating");
                    std::process::exit(911);
                }
            }?;

            match &result {
                Ok(_) => {
                    tracing::debug!("Application shutdown successfully");
                }
                Err(e) => {
                    tracing::error!("Application shutdown with error: {:?}", e);
                }
            }

            result
        }))))
        .expect("Failed to spawn application task");

        let task = INIT
            .get()
            .expect("Application task not initialized")
            .lock()
            .unwrap()
            .take()
            .expect("Application initialized; but another thread is awaiting it; Worker.execute() can only be called once");
        task
    }

    pub fn from_current() -> Result<Worker> {
        if RT.get().is_some() || RTHANDLE.get().is_some() {
            return Err(error!("Worker already initialized"));
        }
        let runtime = Runtime::from_current()?;
        Ok(Worker { runtime })
    }
}

/// Catch signals and trigger a shutdown
async fn signal_handler(cancel_token: CancellationToken) -> Result<()> {
    let ctrl_c = async {
        signal::ctrl_c().await?;
        anyhow::Ok(())
    };

    let sigterm = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())?
            .recv()
            .await;
        anyhow::Ok(())
    };

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Ctrl+C received, starting graceful shutdown");
        },
        _ = sigterm => {
            tracing::info!("SIGTERM received, starting graceful shutdown");
        },
        _ = cancel_token.cancelled() => {
            tracing::debug!("CancellationToken triggered; shutting down");
        },
    }

    // trigger a shutdown
    cancel_token.cancel();

    Ok(())
}
