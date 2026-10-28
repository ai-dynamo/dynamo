// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example-local registration and lifecycle glue. Shared LLM workers are unchanged.

use std::{future::Future, sync::Arc, time::Duration};

use dynamo_backend_common::{
    ModelInput, Worker, WorkerConfig,
    worker::{grace_period_secs, graceful_shutdown_timeout},
};
use dynamo_llm::{
    local_model::{LocalModel, LocalModelBuilder},
    model_type::ModelType,
    worker_type::WorkerType,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime, pipeline::network::Ingress,
    system_status_server::SystemProbePolicy,
};
use dynamo_sidecar_common::GrpcChannelPool;
use tokio_util::sync::CancellationToken;

use crate::{
    asr::AsrAdapter,
    client,
    config::{Args, AsrConfig, Mode},
    tts::TtsEngine,
};

pub async fn run(args: Args, runtime: Runtime, shutdown: CancellationToken) -> anyhow::Result<()> {
    let worker = args.mode.worker();
    let startup = async {
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            worker.runtime.to_distributed_config()?,
            SystemProbePolicy::RuntimeOnly,
        )
        .await?;
        tracing::info!("Speech runtime connected; waiting for Speech NIM readiness");
        let channel = client::connect(&args.mode).await?;
        Ok::<_, anyhow::Error>((drt, channel))
    };
    let runtime_shutdown = runtime.shutdown_started_token();
    let result = tokio::select! {
        biased;
        _ = shutdown.cancelled() => return Ok(()),
        _ = runtime_shutdown.cancelled() => anyhow::bail!("runtime shut down during speech initialization"),
        result = startup => result,
    };
    if runtime.is_shutting_down() {
        if shutdown.is_cancelled() {
            return Ok(());
        }
        anyhow::bail!("runtime shut down during speech initialization");
    }
    let (drt, channel) = result?;
    serve(args.mode, channel, drt, shutdown).await
}

/// Serve a connected backend using an existing runtime, including in CPU tests.
pub async fn serve(
    mode: Mode,
    channel: Arc<GrpcChannelPool>,
    drt: DistributedRuntime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    match mode {
        Mode::Tts(config) => {
            let (namespace, component, endpoint) = config.worker.endpoint_parts("speech-tts");
            let worker_config = WorkerConfig {
                namespace,
                component,
                endpoint,
                // The engine supplies its public name; no tokenizer assets.
                model_input: ModelInput::Text,
                endpoint_types: "audios".into(),
                enable_kv_routing: false,
                enable_local_indexer: false,
                runtime: config.worker.runtime.clone(),
                ..Default::default()
            };
            Worker::new_raw(Arc::new(TtsEngine::new(config, channel)), worker_config)
                .run_with_drt(drt, shutdown)
                .await
                .map_err(Into::into)
        }
        Mode::Asr(config) => run_asr(config, channel, drt, shutdown).await,
    }
}

async fn run_asr(
    config: AsrConfig,
    channel: Arc<GrpcChannelPool>,
    drt: DistributedRuntime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (namespace, component, endpoint_name) = config.worker.endpoint_parts("speech-asr-realtime");
    let endpoint = drt
        .namespace(namespace)?
        .component(component)?
        .endpoint(endpoint_name);
    let mut model = LocalModelBuilder::default()
        .model_name(Some(config.model_name.clone()))
        .build()
        .await?;
    let adapter = Arc::new(AsrAdapter::new(config, channel));
    let ingress = Ingress::for_engine(adapter.clone())?;
    let started = endpoint
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start_with_registration()
        .await?;
    if let Err(error) = model
        .attach(
            &endpoint,
            ModelType::Realtime,
            ModelInput::Text,
            None,
            Some(WorkerType::Aggregated),
            Vec::new(),
        )
        .await
    {
        if let Err(shutdown_error) = started.shutdown().await {
            tracing::warn!(error = %shutdown_error, endpoint = %endpoint.id(), "Failed to shut down ASR endpoint after registration failure");
        }
        return Err(error);
    }
    tracing::info!(endpoint = %endpoint.id(), "Serving Speech NIM realtime transcription");
    let runtime_shutdown = drt.runtime().shutdown_started_token();
    let result = tokio::select! {
        biased;
        _ = shutdown.cancelled() => Ok(()),
        _ = runtime_shutdown.cancelled() => Err(anyhow::anyhow!("runtime shut down while serving speech")),
    };
    drt.runtime().mark_shutting_down();
    let withdraw = async {
        let unregister = endpoint.unregister_endpoint_instance().await;
        let detach = LocalModel::detach_from_endpoint(&endpoint, None).await;
        unregister.and(detach)
    };
    let stop = shutdown_asr(
        withdraw,
        started.shutdown(),
        Duration::from_secs_f64(grace_period_secs()),
        graceful_shutdown_timeout(),
    )
    .await;
    adapter.close();
    result.and(stop)
}

async fn shutdown_asr(
    withdraw: impl Future<Output = anyhow::Result<()>>,
    drain: impl Future<Output = anyhow::Result<()>>,
    grace: Duration,
    timeout: Duration,
) -> anyhow::Result<()> {
    let shutdown = async {
        let withdraw = withdraw.await;
        tokio::time::sleep(grace).await;
        let stop = drain.await;
        withdraw.and(stop)
    };
    // Match Worker: discovery and drain share a budget, plus propagation grace.
    tokio::time::timeout(timeout.saturating_add(grace), shutdown)
        .await
        .unwrap_or_else(|_| {
            tracing::warn!("ASR shutdown timed out; cancelling remaining sessions");
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, future::pending};

    #[tokio::test]
    async fn asr_shutdown_bounds_discovery_withdrawal_and_drain() {
        for is_discovery_stalled in [true, false] {
            let has_started_drain = Cell::new(false);
            let withdraw = async {
                if is_discovery_stalled {
                    pending::<()>().await;
                }
                Ok(())
            };
            let drain = async {
                has_started_drain.set(true);
                pending().await
            };
            tokio::time::timeout(
                Duration::from_secs(1),
                shutdown_asr(
                    withdraw,
                    drain,
                    Duration::from_millis(30),
                    Duration::from_millis(20),
                ),
            )
            .await
            .expect("discovery and drain must both be bounded")
            .unwrap();
            assert_eq!(has_started_drain.get(), !is_discovery_stalled);
        }
    }
}
