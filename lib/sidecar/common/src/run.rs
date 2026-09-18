// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use dynamo_backend_common::{DynamoError, LLMEngine, Worker, WorkerConfig};
use dynamo_runtime::system_status_server::SidecarStatusServer;
use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig, logging};
use tokio_util::sync::CancellationToken;

/// Start sidecar probes and runtime dependencies before discovering engine metadata.
/// CLI parsing must happen before constructing `bootstrap` so help and argument
/// errors do not require a listener or any runtime connections.
pub fn run<E: LLMEngine + 'static>(
    bootstrap: impl Future<Output = Result<(E, WorkerConfig), DynamoError>>,
) -> anyhow::Result<()> {
    logging::init();
    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();
    secondary.block_on(async move {
        let shutdown = CancellationToken::new();
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        let signal_token = shutdown.clone();
        let signal_handle = tokio::spawn(async move {
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("SIGTERM received"),
                _ = sigint.recv() => tracing::info!("SIGINT received"),
            }
            signal_token.cancel();
        });

        let result = run_until_shutdown(bootstrap, &runtime, shutdown.clone()).await;
        shutdown.cancel();
        signal_handle.abort();
        let _ = signal_handle.await;
        runtime.shutdown();
        result
    })
}

async fn run_until_shutdown<E: LLMEngine + 'static>(
    bootstrap: impl Future<Output = Result<(E, WorkerConfig), DynamoError>>,
    runtime: &Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let config = dynamo_runtime::config::RuntimeConfig::from_settings()?;
    let status = SidecarStatusServer::start(&config, shutdown.clone()).await?;
    let startup = async {
        let distributed = DistributedConfig::try_from_settings()?;
        let drt = DistributedRuntime::new_with_sidecar_status(
            runtime.clone(),
            distributed,
            status.as_ref(),
        )
        .await?;
        let (engine, config) = bootstrap.await?;
        // Sidecar CLI configuration uses env-based runtime settings. Reject a
        // future factory that tries to change them after connections are live.
        anyhow::ensure!(
            !config.runtime.has_overrides(),
            "sidecar runtime overrides must be configured before startup"
        );
        Ok::<_, anyhow::Error>((drt, engine, config))
    };
    let (drt, engine, config) = tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        result = startup => result?,
    };
    Worker::new(Arc::new(engine), config)
        .run_with_drt(drt, shutdown)
        .await
        .map_err(Into::into)
}
