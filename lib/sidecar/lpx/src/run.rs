// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The executable's runner: the engine, a topology role, or both, on one
//! runtime.

use std::future::Future;
use std::sync::Arc;

use anyhow::{anyhow, bail};
use dynamo_backend_common::{CommonArgs, LLMEngine, RuntimeConfig, Worker};
use dynamo_runtime::system_status_server::SystemProbePolicy;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_sidecar_common::{EngineBootstrapResult, SidecarStartupError, run_task};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::engine::Launch;
use crate::topology::{self, Role};

/// A topology role with the runtime settings and Dynamo namespace it runs
/// with.
pub struct Topology {
    runtime: RuntimeConfig,
    namespace: String,
    role: Role,
}

impl Topology {
    pub(crate) fn new(common: &CommonArgs, role: Role) -> Self {
        Self {
            runtime: common.runtime.clone(),
            namespace: common.namespace.clone(),
            role,
        }
    }
}

/// Runs what the executable launched until SIGTERM or SIGINT.
///
/// Serving without a topology role is [`dynamo_sidecar_common::run`]'s. With
/// one, the runtime connects first, with probes that check only the
/// runtime's dependencies, and the role starts before the engine bootstrap
/// waits for the engine. A failing role is logged and never stops serving.
pub fn run<E, F>(launch: Launch<(RuntimeConfig, F)>) -> anyhow::Result<()>
where
    E: LLMEngine + 'static,
    F: Future<Output = EngineBootstrapResult<E>>,
{
    match launch {
        Launch::Serve(serve) => dynamo_sidecar_common::run(serve),
        Launch::ServeWithTopology(serve, topology) => {
            run_task(|runtime, shutdown| serve_with_topology(serve, topology, runtime, shutdown))
        }
        Launch::Topology(topology) => {
            run_task(|runtime, shutdown| topology_only(topology, runtime, shutdown))
        }
        Launch::Standby(rank) => crate::run_standby(rank),
    }
}

/// The connected runtime, or `None` when a signal arrives first.
async fn connect(
    runtime: &Runtime,
    config: &RuntimeConfig,
    shutdown: &CancellationToken,
) -> anyhow::Result<Option<DistributedRuntime>> {
    let runtime_shutdown = runtime.shutdown_started_token();
    let connecting = async {
        DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            config.to_distributed_config()?,
            SystemProbePolicy::RuntimeOnly,
        )
        .await
    };
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => Ok(None),
        _ = runtime_shutdown.cancelled() => {
            Err(anyhow!("runtime shut down during sidecar initialization"))
        }
        drt = connecting => drt.map(Some),
    }
}

/// A topology role and nothing else; its failure is the process's.
async fn topology_only(
    topology: Topology,
    runtime: Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let Some(drt) = connect(&runtime, &topology.runtime, &shutdown).await? else {
        return Ok(());
    };
    tracing::info!(role = ?topology.role, "Sidecar runtime connected; running only its LPX topology role");
    // The role also stops when the runtime shuts down without a signal.
    let stop = shutdown.child_token();
    let runtime_shutdown = runtime.shutdown_started_token();
    let watch = tokio::spawn({
        let stop = stop.clone();
        async move {
            runtime_shutdown.cancelled().await;
            stop.cancel();
        }
    });
    let result = topology::run(&topology.role, &drt, &topology.namespace, stop).await;
    watch.abort();
    result?;
    // The signal handler cancels the signal token before it marks the runtime
    // shutting down.
    if runtime.is_shutting_down() && !shutdown.is_cancelled() {
        bail!("runtime shut down while running the LPX topology role");
    }
    Ok(())
}

/// The engine, with a topology role that starts before the engine bootstrap.
async fn serve_with_topology<E, F>(
    (runtime_config, bootstrap): (RuntimeConfig, F),
    topology: Topology,
    runtime: Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()>
where
    E: LLMEngine + 'static,
    F: Future<Output = EngineBootstrapResult<E>>,
{
    let Some(drt) = connect(&runtime, &runtime_config, &shutdown).await? else {
        return Ok(());
    };
    // A conductor needs topology to become ready, and the bootstrap waits
    // until the engine is ready, so the role starts first.
    let role = BackgroundRole::start(topology, &drt, &shutdown);
    tracing::info!("Sidecar runtime connected; discovering engine metadata");
    let served = serve(runtime_config, bootstrap, drt, &runtime, shutdown).await;
    role.stop().await;
    served
}

/// The engine bootstrap and the Worker, as [`dynamo_sidecar_common::run`]
/// runs them once the runtime is connected.
async fn serve<E, F>(
    runtime_config: RuntimeConfig,
    bootstrap: F,
    drt: DistributedRuntime,
    runtime: &Runtime,
    shutdown: CancellationToken,
) -> anyhow::Result<()>
where
    E: LLMEngine + 'static,
    F: Future<Output = EngineBootstrapResult<E>>,
{
    let startup = async {
        // Keep engine discovery failures distinct from runtime/Worker failures.
        let (engine, config) = bootstrap.await.map_err(SidecarStartupError::Dynamo)?;
        // Engine discovery must not change settings after connections are live.
        anyhow::ensure!(
            config.runtime == runtime_config,
            "sidecar runtime settings changed during engine discovery"
        );
        Ok::<_, anyhow::Error>((engine, config))
    };
    let runtime_shutdown = runtime.shutdown_started_token();
    let result = tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        _ = runtime_shutdown.cancelled() => {
            Err(anyhow!("runtime shut down during sidecar initialization"))
        },
        result = startup => result,
    };
    // The signal can arrive after its select arm was polled. The signal
    // handler always cancels the signal token before it marks the runtime
    // shutting down.
    if runtime.is_shutting_down() {
        if shutdown.is_cancelled() {
            return Ok(());
        }
        bail!("runtime shut down during sidecar initialization");
    }
    let (engine, config) = result?;
    Worker::new(Arc::new(engine), config)
        .run_with_drt(drt, shutdown)
        .await
        .map_err(Into::into)
}

/// A topology role beside the engine. It fails alone: serving goes on.
struct BackgroundRole {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl BackgroundRole {
    fn start(topology: Topology, drt: &DistributedRuntime, shutdown: &CancellationToken) -> Self {
        let stop = shutdown.child_token();
        let task = tokio::spawn({
            let drt = drt.clone();
            let stop = stop.clone();
            async move {
                let Topology {
                    role, namespace, ..
                } = topology;
                if let Err(error) = topology::run(&role, &drt, &namespace, stop).await {
                    tracing::error!(
                        ?role,
                        error = format!("{error:#}"),
                        "The LPX topology role failed; serving continues without it"
                    );
                }
            }
        });
        Self { stop, task }
    }

    async fn stop(self) {
        self.stop.cancel();
        if let Err(error) = self.task.await {
            tracing::warn!(%error, "The LPX topology role did not stop cleanly");
        }
    }
}
