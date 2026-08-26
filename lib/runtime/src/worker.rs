// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
//! the `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` environment variable. If the application does not
//! shutdown in time, the worker will terminate the application with an exit code of 911.
//!
//! The default values of `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` differ between the development
//! and release builds. In development, the default is [DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG] and
//! in release, the default is [DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE].

use super::{CancellationToken, Runtime, RuntimeConfig};

use futures::Future;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::{signal, task::JoinHandle};

/// The one Tokio runtime for this process.
///
/// Holds the runtime itself rather than a `Handle`, because
/// `pyo3_async_runtimes::tokio::init_with_runtime` needs a `&'static Runtime`.
///
/// Set once, by whichever of [`Worker::from_config`] or [`Worker::ensure_process_runtime`] runs
/// first. Every path goes through this cell, so a process cannot end up with two runtimes.
static RT: OnceCell<tokio::runtime::Runtime> = OnceCell::new();

/// The config `RT` was built from, so [`Worker::runtime_from_existing`] can attach the matching
/// compute pool without re-reading the environment.
///
/// Only [`Worker::ensure_process_runtime`] fills this in; [`Worker::from_config`] brings its own
/// config and no pool.
static RTCONFIG: OnceCell<RuntimeConfig> = OnceCell::new();

/// Whether a [`Runtime`] has already taken the compute pool. One pool per process, however many
/// wrappers get built.
static COMPUTE_CLAIMED: AtomicBool = AtomicBool::new(false);

static INIT: OnceCell<Mutex<Option<tokio::task::JoinHandle<anyhow::Result<()>>>>> = OnceCell::new();

use crate::config::environment_names::worker as env_worker;

const SHUTDOWN_MESSAGE: &str =
    "Application received shutdown signal; attempting to gracefully shutdown";
const SHUTDOWN_TIMEOUT_MESSAGE: &str =
    "Use DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT to control the graceful shutdown timeout";

/// Default graceful shutdown timeout in seconds in debug mode
pub const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG: u64 = 5;

/// Default graceful shutdown timeout in seconds in release mode
pub const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE: u64 = 30;

#[derive(Debug, Clone)]
pub struct Worker {
    runtime: Runtime,
    config: RuntimeConfig,
}

impl Worker {
    /// Create a new [`Worker`] instance from [`RuntimeConfig`] settings which is sourced from the environment
    pub fn from_settings() -> anyhow::Result<Worker> {
        let config = RuntimeConfig::from_settings()?;
        Worker::from_config(config)
    }

    /// Create a new [`Worker`] instance from a provided [`RuntimeConfig`]
    pub fn from_config(config: RuntimeConfig) -> anyhow::Result<Worker> {
        // if the runtime is already initialized, return an error
        if RT.get().is_some() {
            return Err(anyhow::anyhow!("Worker already initialized"));
        }

        // create a new runtime and insert it into the OnceCell
        // there is still a potential race-condition here, two threads cou have passed the first check
        // but only one will succeed in inserting the runtime
        let rt = RT.try_insert(config.create_runtime()?).map_err(|_| {
            anyhow::anyhow!("Failed to create worker; Only a single Worker should ever be created")
        })?;

        let runtime = Runtime::from_handle(rt.handle().clone())?;
        Ok(Worker { runtime, config })
    }

    /// Share the process-wide runtime, creating it on first use.
    ///
    /// The returned [`Runtime`] only wraps a handle to `RT`, so every caller ends up on the same
    /// Tokio runtime. Creation goes through [`Worker::ensure_process_runtime`] rather than
    /// happening here, so the runtime stays reachable as a `&'static` for the pyo3 bridge.
    pub fn runtime_from_existing() -> anyhow::Result<Runtime> {
        let rt = Self::ensure_process_runtime()?;
        let handle = rt.handle().clone();

        // Only the first wrapper gets the compute pool and `block_in_place` permits: one Rayon
        // pool per process, not one per `DistributedRuntime`.
        //
        // An atomic swap rather than "did I just build `RT`?", because callers may call
        // `ensure_process_runtime` first — `DistributedRuntime::new` does — which would make
        // that question always answer no. The swap also settles the race between two threads.
        match RTCONFIG.get() {
            Some(config) if !COMPUTE_CLAIMED.swap(true, Ordering::SeqCst) => {
                Runtime::from_handle_with_config(handle, config)
            }
            _ => Runtime::from_handle(handle),
        }
    }

    /// Create the process-wide runtime if it does not exist yet, and return it. Idempotent.
    ///
    /// Exists because the pyo3 bridge needs a `&'static tokio::runtime::Runtime`, which
    /// [`Worker::from_config`] cannot provide — it errors when a runtime already exists.
    pub fn ensure_process_runtime() -> anyhow::Result<&'static tokio::runtime::Runtime> {
        // Fast path: already built by this function or by `Worker::from_config`.
        if let Some(rt) = RT.get() {
            return Ok(rt);
        }

        // If two threads arrive together, one builds and both observe the same runtime.
        RT.get_or_try_init(|| -> anyhow::Result<tokio::runtime::Runtime> {
            let config = RuntimeConfig::from_settings()?;
            // Logged so a running deployment can show whether DYN_RUNTIME_* was applied.
            tracing::info!("dynamo runtime configuration: {config}");
            let rt = config.create_runtime()?;
            let _ = RTCONFIG.set(config);
            Ok(rt)
        })
    }

    /// Whether the process-wide runtime already exists.
    ///
    /// Never creates one, unlike [`Worker::runtime_from_existing`], so a caller can use this to
    /// find out whether it would be the owner.
    pub fn has_existing_runtime() -> bool {
        RT.get().is_some()
    }

    pub fn tokio_runtime(&self) -> anyhow::Result<&'static tokio::runtime::Runtime> {
        RT.get()
            .ok_or_else(|| anyhow::anyhow!("Worker not initialized"))
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn execute<F, Fut>(self, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        runtime.secondary().block_on(self.execute_internal(f))??;
        runtime.shutdown();
        Ok(())
    }

    pub async fn execute_async<F, Fut>(self, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        let task = self.execute_internal(f);
        task.await??;
        runtime.shutdown();
        Ok(())
    }

    /// Executes the provided application/closure on the [`Runtime`].
    /// This is designed to be called once from main and will block the calling thread until the application completes.
    fn execute_internal<F, Fut>(self, f: F) -> JoinHandle<anyhow::Result<()>>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let runtime = self.runtime.clone();
        let primary = runtime.primary();
        let secondary = runtime.secondary();

        let timeout = std::env::var(env_worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT)
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
            tokio::spawn(signal_handler(runtime.primary_token().clone()));

            let cancel_token = runtime.child_token();
            let (mut app_tx, app_rx) = tokio::sync::oneshot::channel::<()>();

            // spawn a task to run the application
            let task: JoinHandle<anyhow::Result<()>> = primary.spawn(async move {
                let _rx = app_rx;
                f(runtime).await
            });

            tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::debug!("{SHUTDOWN_MESSAGE}");
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

        INIT
            .get()
            .expect("Application task not initialized")
            .lock()
            .take()
            .expect("Application initialized; but another thread is awaiting it; Worker.execute() can only be called once")
    }

    pub fn from_current() -> anyhow::Result<Worker> {
        if RT.get().is_some() {
            return Err(anyhow::anyhow!("Worker already initialized"));
        }
        let runtime = Runtime::from_current()?;
        let config = RuntimeConfig::from_settings()?;
        Ok(Worker { runtime, config })
    }
}

/// Catch signals and trigger a shutdown
async fn signal_handler(cancel_token: CancellationToken) -> anyhow::Result<()> {
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
