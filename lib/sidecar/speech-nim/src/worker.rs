// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Speech worker registration and lifecycle.

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
use dynamo_runtime::{DistributedRuntime, pipeline::network::Ingress};
use dynamo_sidecar_common::GrpcChannelPool;
use tokio_util::sync::CancellationToken;

use crate::{
    asr::AsrAdapter,
    config::{AsrConfig, Mode},
    tts::TtsEngine,
};

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
    let runtime_shutdown = drt.runtime().shutdown_started_token();
    let stop_requested = || async {
        tokio::select! {
            _ = shutdown.cancelled() => {},
            _ = runtime_shutdown.cancelled() => {},
        }
    };
    let stop_result = || {
        anyhow::ensure!(
            shutdown.is_cancelled(),
            "runtime shut down while serving speech"
        );
        Ok(())
    };
    let mut model = LocalModelBuilder::default()
        .model_name(Some(config.model_name.clone()))
        .build()
        .await?;
    let adapter = Arc::new(AsrAdapter::new(config, channel));
    let Some(started) = endpoint
        .endpoint_builder()
        .handler(Ingress::for_engine(adapter.clone())?)
        .graceful_shutdown(true)
        .start_with_registration_until(stop_requested(), graceful_shutdown_timeout())
        .await?
    else {
        return stop_result();
    };
    let attach = model.attach(
        &endpoint,
        ModelType::Realtime,
        ModelInput::Text,
        None,
        Some(WorkerType::Aggregated),
        Vec::new(),
    );
    let serving = async {
        attach.await?;
        tracing::info!(endpoint = %endpoint.id(), "Serving Speech NIM realtime transcription");
        std::future::pending::<anyhow::Result<()>>().await
    };
    let (result, grace) = tokio::select! {
        biased;
        _ = stop_requested() => {
            drt.runtime().mark_shutting_down();
            (stop_result(), Duration::from_secs_f64(grace_period_secs()))
        },
        error = serving => (error, Duration::ZERO),
    };
    let withdraw = async {
        let (unregister, detach) = tokio::join!(
            endpoint.unregister_endpoint_instance(),
            LocalModel::detach_from_endpoint(&endpoint, None),
        );
        unregister.and(detach)
    };
    let stop = shutdown_asr(
        withdraw,
        started.shutdown(),
        grace,
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
    let deadline = tokio::time::sleep(timeout.saturating_add(grace));
    tokio::pin!(deadline);
    let withdraw = tokio::select! {
        result = async {
            let withdraw = withdraw.await;
            tokio::time::sleep(grace).await;
            withdraw
        } => result,
        _ = &mut deadline => {
            tracing::warn!("ASR discovery withdrawal timed out");
            Ok(())
        },
    };
    // Poll shutdown even if withdrawal spent the budget, so it cancels admission.
    let stop = tokio::select! {
        biased;
        result = drain => result,
        _ = &mut deadline => {
            tracing::warn!("ASR shutdown timed out; cancelling remaining sessions");
            Ok(())
        },
    };
    withdraw.and(stop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, future::pending};

    #[tokio::test(start_paused = true)]
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
            let started = tokio::time::Instant::now();
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
            assert!(
                has_started_drain.get(),
                "endpoint cancellation must always start"
            );
            assert_eq!(started.elapsed(), Duration::from_millis(50));
        }
    }
}
