// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common entry point for unified backends.
//!
//! Each backend's `main.rs` parses CLI args, constructs its `LLMEngine`, and
//! hands the pair off to [`run`]:
//!
//! ```ignore
//! use std::sync::Arc;
//!
//! fn main() -> anyhow::Result<()> {
//!     let (engine, config) = MyEngine::from_args(None)?;
//!     dynamo_backend_common::run(Arc::new(engine), config)
//! }
//! ```
//!
//! `run` deliberately bypasses `dynamo_runtime::Worker::execute` and drives
//! the runtime directly. The reason is signal ownership: `Worker::execute`
//! spawns its own SIGTERM/SIGINT handler that immediately cancels the
//! primary cancellation token, which races [`crate::worker::Worker`]'s
//! shutdown orchestrator (discovery unregister → grace period → drain →
//! cleanup). Owning the signal flow here means the orchestrator runs
//! *before* any token cancellation tears down NATS / etcd, matching the
//! behaviour of the Python `graceful_shutdown_with_discovery` helper.

use std::sync::Arc;

use dynamo_runtime::{Runtime, logging};

use crate::engine::{LLMEngine, RawEngine};
use crate::worker::{Worker, WorkerConfig};

/// Drive the full lifecycle for an already-constructed token-pipeline engine.
pub fn run(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> anyhow::Result<()> {
    run_worker(config, |cfg| Worker::new(engine, cfg))
}

/// Drive the full lifecycle for an already-constructed raw media-pipeline
/// engine (image/video/audio generation). Identical orchestration to [`run`];
/// only the request adapter differs.
pub fn run_raw(engine: Arc<dyn RawEngine>, config: WorkerConfig) -> anyhow::Result<()> {
    run_worker(config, |cfg| Worker::new_raw(engine, cfg))
}

/// Shared runtime setup + shutdown for both engine modalities. `build`
/// constructs the [`Worker`] from the (env-applied) config once the runtime
/// is up.
fn run_worker(
    config: WorkerConfig,
    build: impl FnOnce(WorkerConfig) -> Worker,
) -> anyhow::Result<()> {
    logging::init();

    // Apply RuntimeConfig overrides to the env before the runtime reads
    // them. Done sync, before any tokio threads spawn, to match the
    // pattern used by `dynamo-runtime`'s own DistributedConfig::from_settings.
    config.runtime.apply_to_env();

    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();

    secondary.block_on(async move {
        let result = build(config)
            .run(runtime.clone())
            .await
            .map_err(anyhow::Error::from);

        // Trigger Phase 1/2/3 token cancellation + NATS/etcd disconnect.
        // Worker::run has already done discovery unregister, drain, and
        // engine.cleanup() at this point, so this is purely transport
        // teardown.
        runtime.shutdown();

        result
    })
}

/// Install signals before any asynchronous startup work. Dropping the returned
/// handle removes the listener task; the caller owns the full startup lifetime.
pub fn shutdown_signal(
    runtime: Runtime,
) -> Result<
    (
        tokio_util::sync::CancellationToken,
        tokio_util::task::AbortOnDropHandle<()>,
    ),
    crate::DynamoError,
> {
    use crate::{BackendError, DynamoError, ErrorType};
    use tokio_util::sync::CancellationToken;
    // Install the OS signal handlers synchronously, before spawning
    // anything, so a SIGTERM delivered between this point and the
    // task's first poll is captured by the kernel-side handler rather
    // than the OS default (which would terminate the process abruptly).
    // `Signal::recv` then drives the shared cancellation token.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("install SIGTERM handler: {e}"))
                .build()
        })?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("install SIGINT handler: {e}"))
                .build()
        })?;

    // Single shared shutdown signal observed across all phases. The
    // background task only flips the token; lifecycle transitions stay
    // on this owned Worker instance.
    let shutdown_token = CancellationToken::new();
    let signal_token = shutdown_token.clone();
    let signal_handle = tokio::spawn(async move {
        tokio::select! {
            _ = runtime.primary_token().cancelled_owned() => tracing::info!("Runtime shutdown received"),
            _ = sigterm.recv() => tracing::info!("SIGTERM received"),
            _ = sigint.recv() => tracing::info!("SIGINT received"),
        }
        runtime.mark_shutting_down();
        signal_token.cancel();
    });

    Ok((
        shutdown_token,
        tokio_util::task::AbortOnDropHandle::new(signal_handle),
    ))
}
