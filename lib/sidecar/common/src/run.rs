// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use dynamo_backend_common::{DynamoError, LLMEngine, RuntimeConfig, Worker, WorkerConfig};
use dynamo_runtime::system_status_server::SystemProbePolicy;
use dynamo_runtime::{DistributedRuntime, Runtime, logging};
use tokio_util::sync::CancellationToken;

use crate::SidecarStartupError;

/// Result of discovering an engine and preparing its worker configuration.
pub type EngineBootstrapResult<E> = Result<(E, WorkerConfig), DynamoError>;

/// Start sidecar probes and runtime dependencies before discovering engine metadata.
/// Runtime settings accompany the bootstrap future so they take effect before
/// the first connection, including in embedded Python launchers.
/// CLI parsing must happen before constructing `bootstrap` so help and argument
/// errors do not require a listener or any runtime connections.
pub fn run<E: LLMEngine + 'static>(
    (runtime_config, bootstrap): (
        RuntimeConfig,
        impl Future<Output = EngineBootstrapResult<E>>,
    ),
) -> anyhow::Result<()> {
    run_task(|runtime, shutdown| async move {
        run_until_shutdown(runtime_config, bootstrap, &runtime, shutdown).await
    })
}

/// Start runtime probes before a sidecar chooses whether to serve inference or
/// run a node-local telemetry task.
pub fn run_with_drt<T, B, F, Fut>(
    runtime_config: RuntimeConfig,
    bootstrap: B,
    task: F,
) -> anyhow::Result<()>
where
    B: Future<Output = anyhow::Result<T>>,
    F: FnOnce(DistributedRuntime, T, CancellationToken) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    run_task(|runtime, shutdown| async move {
        run_drt_task_until_shutdown(runtime_config, bootstrap, task, &runtime, shutdown).await
    })
}

/// Run a sidecar task with the shared runtime and process-signal lifecycle.
/// The task owns discovery and may serve inference or relay telemetry only.
pub fn run_task<F, Fut>(task: F) -> anyhow::Result<()>
where
    F: FnOnce(Runtime, CancellationToken) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    logging::init();
    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();
    secondary.block_on(async move {
        let shutdown = CancellationToken::new();
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        let signal_token = shutdown.clone();
        let signal_runtime = runtime.clone();
        let signal_handle = tokio::spawn(async move {
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("SIGTERM received"),
                _ = sigint.recv() => tracing::info!("SIGINT received"),
            }
            signal_token.cancel();
            signal_runtime.mark_shutting_down();
        });

        let result = task(runtime.clone(), shutdown.clone()).await;
        shutdown.cancel();
        signal_handle.abort();
        let _ = signal_handle.await;
        runtime.shutdown();
        result
    })
}

async fn run_until_shutdown<E: LLMEngine + 'static>(
    runtime_config: RuntimeConfig,
    bootstrap: impl Future<Output = EngineBootstrapResult<E>>,
    runtime: &Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let expected_runtime = runtime_config.clone();
    let bootstrap = async move {
        // Keep engine discovery failures distinct from runtime/Worker failures
        // for embedded launchers' existing error contracts.
        let (engine, config) = bootstrap.await.map_err(SidecarStartupError::Dynamo)?;
        anyhow::ensure!(
            config.runtime == expected_runtime,
            "sidecar runtime settings changed during engine discovery"
        );
        Ok::<_, anyhow::Error>((engine, config))
    };
    run_drt_task_until_shutdown(
        runtime_config,
        bootstrap,
        |drt, (engine, config), shutdown| async move {
            Worker::new(Arc::new(engine), config)
                .run_with_drt(drt, shutdown)
                .await
                .map_err(Into::into)
        },
        runtime,
        shutdown,
    )
    .await
}

async fn run_drt_task_until_shutdown<T, B, F, Fut>(
    runtime_config: RuntimeConfig,
    bootstrap: B,
    task: F,
    runtime: &Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()>
where
    B: Future<Output = anyhow::Result<T>>,
    F: FnOnce(DistributedRuntime, T, CancellationToken) -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    let startup = async {
        let distributed = runtime_config.to_distributed_config()?;
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            distributed,
            SystemProbePolicy::RuntimeOnly,
        )
        .await?;
        tracing::info!("Sidecar runtime connected; discovering engine metadata");
        let initialized = bootstrap.await?;
        Ok::<_, anyhow::Error>((drt, initialized))
    };
    let runtime_shutdown = runtime.shutdown_started_token();
    let result = tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        _ = runtime_shutdown.cancelled() => {
            Err(anyhow::anyhow!("runtime shut down during sidecar initialization"))
        },
        result = startup => result,
    };
    // The signal can arrive after its select arm was polled, including while
    // startup completes. Read runtime state before the signal token: the signal
    // handler always cancels that token before marking the runtime shutting down.
    if runtime.is_shutting_down() {
        if shutdown.is_cancelled() {
            return Ok(());
        }
        anyhow::bail!("runtime shut down during sidecar initialization");
    }
    let (drt, initialized) = result?;
    task(drt, initialized, shutdown).await
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

    // Cancellation and bootstrap completion may occur in the same poll. A
    // process signal is clean shutdown; independent runtime failure is an error.
    #[tokio::test]
    async fn shutdown_during_bootstrap_distinguishes_signals_from_runtime_failure() {
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_PORT", None),
                ("DYN_DISCOVERY_BACKEND", Some("mem")),
                ("DYN_REQUEST_PLANE", Some("tcp")),
                ("DYN_EVENT_PLANE", Some("zmq")),
                ("NATS_SERVER", None),
            ],
            async {
                for signal in [false, true] {
                    for outcome in ["pending", "success", "error"] {
                        let runtime = Runtime::from_current().unwrap();
                        let shutdown = CancellationToken::new();
                        let bootstrap = async {
                            if signal {
                                shutdown.cancel();
                            }
                            runtime.mark_shutting_down();
                            match outcome {
                                "pending" => std::future::pending().await,
                                "success" => Ok((UnstartedEngine, WorkerConfig::default())),
                                "error" => Err(DynamoError::msg("bootstrap failed")),
                                _ => unreachable!(),
                            }
                        };
                        let result = tokio::time::timeout(
                            std::time::Duration::from_secs(5),
                            run_until_shutdown(
                                RuntimeConfig::default(),
                                bootstrap,
                                &runtime,
                                shutdown.clone(),
                            ),
                        )
                        .await
                        .expect("runtime shutdown cancels metadata discovery");
                        if signal {
                            result.expect("a process signal exits cleanly");
                        } else {
                            assert!(
                                result
                                    .unwrap_err()
                                    .to_string()
                                    .contains("runtime shut down")
                            );
                        }
                        runtime.shutdown();
                    }
                }
            },
        )
        .await;
    }
}
