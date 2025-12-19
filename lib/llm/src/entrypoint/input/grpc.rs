// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::{EngineConfig, RouterConfig, input::common},
    grpc::service::kserve,
    http::service::metrics::Metrics,
    namespace::is_global_namespace,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::DistributedRuntime;

/// Build and run an KServe gRPC service
pub async fn run(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let mut grpc_service_builder = kserve::KserveService::builder()
        .port(engine_config.local_model().http_port()) // [WIP] generalize port..
        .with_request_template(engine_config.local_model().request_template());

    // Set HTTP metrics port if provided (for parallel test execution)
    if let Some(http_metrics_port) = engine_config.local_model().http_metrics_port() {
        grpc_service_builder = grpc_service_builder.http_metrics_port(http_metrics_port);
    }

    let grpc_service = match engine_config {
        EngineConfig::Dynamic { ref model, .. } => {
            let grpc_service = grpc_service_builder.build()?;
            let router_config = model.router_config();
            // Listen for models registering themselves, add them to gRPC service
            let namespace = model.namespace().unwrap_or("");
            let target_namespace = if is_global_namespace(namespace) {
                None
            } else {
                Some(namespace.to_string())
            };
            run_watcher(
                distributed_runtime.clone(),
                grpc_service.state().manager_clone(),
                router_config.clone(),
                target_namespace,
            )
            .await?;
            grpc_service
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let grpc_service = grpc_service_builder.build()?;
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = grpc_service.model_manager();
            let checksum = model.card().mdcsum();
            manager.add_completions_model(model.service_name(), checksum, engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), checksum, engine)?;
            grpc_service
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let grpc_service = grpc_service_builder.build()?;
            let manager = grpc_service.model_manager();
            let checksum = model.card().mdcsum();

            let tokenizer_hf = model.card().tokenizer_hf()?;
            let chat_pipeline =
                common::build_pipeline::<
                    NvCreateChatCompletionRequest,
                    NvCreateChatCompletionStreamResponse,
                >(model.card(), inner_engine.clone(), tokenizer_hf.clone())
                .await?;
            manager.add_chat_completions_model(model.service_name(), checksum, chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(model.card(), inner_engine, tokenizer_hf)
            .await?;
            manager.add_completions_model(model.service_name(), checksum, cmpl_pipeline)?;
            grpc_service
        }
    };

    // Run both HTTP (for metrics) and gRPC servers concurrently
    let http_service = grpc_service.http_service().clone();
    let shutdown_token = distributed_runtime.primary_token();

    // Wait for both servers to complete, propagating the first error if any occurs
    // Both tasks should run indefinitely until cancelled by the shutdown token
    tokio::try_join!(
        grpc_service.run(shutdown_token.clone()),
        http_service.run(shutdown_token)
    )?;

    distributed_runtime.shutdown(); // Cancel primary token
    Ok(())
}

/// Spawns a task that watches for new models in store,
/// and registers them with the ModelManager so that the HTTP service can use them.
async fn run_watcher(
    runtime: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    router_config: RouterConfig,
    target_namespace: Option<String>,
) -> anyhow::Result<()> {
    // Create metrics for migration tracking (not exposed via /metrics in gRPC mode)
    let metrics = Arc::new(Metrics::new());
    let watch_obj = ModelWatcher::new(runtime.clone(), model_manager, router_config, None, metrics);
    tracing::debug!("Waiting for remote model");
    let discovery = runtime.discovery();
    let discovery_stream = discovery
        .list_and_watch(
            dynamo_runtime::discovery::DiscoveryQuery::AllModels,
            Some(runtime.primary_token()),
        )
        .await?;

    // [gluo NOTE] This is different from http::run_watcher where it alters the HTTP service
    // endpoint being exposed, gRPC doesn't have the same concept as the KServe service
    // only has one kind of inference endpoint.

    // Pass the discovery stream to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj
            .watch(discovery_stream, target_namespace.as_deref())
            .await;
    });

    Ok(())
}
