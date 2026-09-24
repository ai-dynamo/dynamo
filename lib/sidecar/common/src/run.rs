// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use dynamo_backend_common::{DynamoError, LLMEngine, Worker, WorkerConfig};
use dynamo_runtime::system_status_server::SidecarStatusServer;
use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig, logging};
use tokio_util::sync::CancellationToken;

use crate::SidecarStartupError;

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
        let drt = DistributedRuntime::new(runtime.clone(), distributed).await?;
        if let Some(ref server) = status {
            server.attach(Arc::new(drt.clone()), drt.discovery_metadata())?;
            drt.set_system_status_server_info(server.info())?;
        }
        tracing::info!("Sidecar runtime connected; discovering engine metadata");
        // Keep engine discovery failures distinct from runtime/Worker failures
        // for embedded launchers' existing error contracts.
        let (engine, config) = bootstrap.await.map_err(SidecarStartupError::Dynamo)?;
        // Sidecar CLI configuration uses env-based runtime settings. Reject a
        // future factory that tries to change them after connections are live.
        anyhow::ensure!(
            !config.runtime.has_overrides(),
            "sidecar runtime overrides must be configured before startup"
        );
        Ok::<_, anyhow::Error>((drt, engine, config))
    };
    let runtime_shutdown = runtime.shutdown_started_token();
    let (drt, engine, config) = tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        _ = runtime_shutdown.cancelled() => {
            anyhow::bail!("runtime shut down during sidecar initialization");
        },
        result = startup => result?,
    };
    anyhow::ensure!(
        !runtime.is_shutting_down(),
        "runtime shut down during sidecar initialization"
    );
    Worker::new(Arc::new(engine), config)
        .run_with_drt(drt, shutdown)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{
        EngineConfig, GenerateContext, LLMEngineOutput, PreprocessedRequest,
    };
    use futures::stream::BoxStream;

    struct UnstartedEngine;

    #[async_trait::async_trait]
    impl LLMEngine for UnstartedEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            panic!("cancelled bootstrap must not start a worker");
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            unreachable!()
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    // Regression: losing the runtime while metadata discovery waits must cancel
    // bootstrap; a completed bootstrap must never start a worker on that runtime.
    #[tokio::test]
    async fn runtime_shutdown_cancels_pending_and_completed_bootstrap() {
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_PORT", None),
                ("DYN_DISCOVERY_BACKEND", Some("mem")),
                ("DYN_REQUEST_PLANE", Some("tcp")),
                ("DYN_EVENT_PLANE", Some("zmq")),
                ("NATS_SERVER", None),
            ],
            async {
                for complete in [false, true] {
                    let runtime = Runtime::from_current().unwrap();
                    let bootstrap = async {
                        runtime.mark_shutting_down();
                        if !complete {
                            std::future::pending::<()>().await;
                        }
                        Ok((UnstartedEngine, WorkerConfig::default()))
                    };
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        run_until_shutdown(bootstrap, &runtime, CancellationToken::new()),
                    )
                    .await
                    .expect("runtime shutdown cancels metadata discovery");
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("runtime shut down")
                    );
                    runtime.shutdown();
                }
            },
        )
        .await;
    }
}
