// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example-local registration and lifecycle glue. Shared LLM workers are unchanged.

use std::sync::Arc;

use dynamo_backend_common::{ModelInput, Worker, WorkerConfig};
use dynamo_llm::{
    local_model::{LocalModel, LocalModelBuilder},
    model_type::ModelType,
    worker_type::WorkerType,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime, pipeline::network::Ingress,
    system_status_server::SystemProbePolicy,
};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use crate::{
    asr::AsrAdapter,
    client,
    config::{Args, AsrConfig, Mode},
    tts::TtsEngine,
};

pub async fn run(args: Args, runtime: Runtime, shutdown: CancellationToken) -> anyhow::Result<()> {
    let worker = match &args.mode {
        Mode::Asr(c) => &c.worker,
        Mode::Tts(c) => &c.worker,
    };
    let startup = async {
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            worker.runtime.to_distributed_config()?,
            SystemProbePolicy::RuntimeOnly,
        )
        .await?;
        tracing::info!("Speech runtime connected; waiting for the NIM gRPC channel");
        let channel = client::connect(&worker.connection).await?;
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
    channel: Channel,
    drt: DistributedRuntime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    match mode {
        Mode::Tts(config) => {
            let (namespace, component, endpoint) = config.worker.endpoint_parts("speech-tts")?;
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
    channel: Channel,
    drt: DistributedRuntime,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (namespace, component, endpoint_name) =
        config.worker.endpoint_parts("speech-asr-realtime")?;
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
    adapter.close();
    let detach = LocalModel::detach_from_endpoint(&endpoint, None).await;
    let stop = started.shutdown().await;
    result.and(detach).and(stop)
}
