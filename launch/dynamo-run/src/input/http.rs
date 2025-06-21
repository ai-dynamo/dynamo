// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::input::common;
use crate::{EngineConfig, Flags};
use dynamo_llm::kv_router::KvRouterConfig;
use dynamo_llm::{
    discovery::{ModelManager, ModelUpdate, ModelWatcher, MODEL_ROOT_PATH},
    engines::StreamingEngineAdapter,
    http::service::service_v2,
    http::service::service_v2::HttpService,
    model_type::ModelType,
    request_template::RequestTemplate,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{CompletionResponse, NvCreateCompletionRequest},
    },
};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::transports::etcd;
use dynamo_runtime::{DistributedRuntime, Runtime};

/// Build and run an HTTP service
pub async fn run(
    runtime: Runtime,
    flags: Flags,
    engine_config: EngineConfig,
    template: Option<RequestTemplate>,
) -> anyhow::Result<()> {
    let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
    let http_service = service_v2::HttpService::builder()
        .port(flags.http_port)
        .with_request_template(template)
        .runtime(Some(Arc::new(distributed_runtime)))
        .build()?;
    match engine_config {
        EngineConfig::Dynamic => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            match distributed_runtime.etcd_client() {
                Some(etcd_client) => {
                    // Listen for models registering themselves in etcd, add them to HTTP service
                    run_watcher(
                        distributed_runtime,
                        http_service.state().manager_clone(),
                        etcd_client.clone(),
                        MODEL_ROOT_PATH,
                        flags.router_mode.into(),
                        Some(flags.kv_router_config()),
                        Arc::new(http_service.clone()),
                    )
                    .await?;
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
        }
        EngineConfig::StaticFull { engine, model } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            manager.add_completions_model(model.service_name(), engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), engine)?;

            // Enable relevant endpoints
            http_service.enable_chat_endpoints(true).await;
            http_service.enable_cmpl_endpoints(true).await;
            http_service.enable_embeddings_endpoints(true).await;
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
        } => {
            let manager = http_service.model_manager();

            let chat_pipeline = common::build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine.clone())
            .await?;
            manager.add_chat_completions_model(model.service_name(), chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                CompletionResponse,
            >(model.card(), inner_engine)
            .await?;
            manager.add_completions_model(model.service_name(), cmpl_pipeline)?;

            // Enable relevant endpoints
            http_service.enable_chat_endpoints(true).await;
            http_service.enable_cmpl_endpoints(true).await;
            http_service.enable_embeddings_endpoints(true).await;
        }
    }
    tracing::debug!(
        "Supported routes: {:?}",
        http_service
            .route_docs()
            .iter()
            .map(|rd| rd.to_string())
            .collect::<Vec<String>>()
    );
    http_service.run(runtime.primary_token()).await?;
    runtime.shutdown(); // Cancel primary token
    Ok(())
}

/// Spawns a task that watches for new models in etcd at network_prefix,
/// and registers them with the ModelManager so that the HTTP service can use them.
async fn run_watcher(
    runtime: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    etcd_client: etcd::Client,
    network_prefix: &str,
    router_mode: RouterMode,
    kv_router_config: Option<KvRouterConfig>,
    http_service: Arc<HttpService>,
) -> anyhow::Result<()> {
    let mut watch_obj = ModelWatcher::new(runtime, model_manager, router_mode, kv_router_config);
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();

    // Create a channel to receive model type updates
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    watch_obj.set_notify_on_model_update(tx);

    // Spawn a task to watch for model type changes and update HTTP service endpoints
    let _endpoint_enabler_task = tokio::spawn(async move {
        while let Some(model_type) = rx.recv().await {
            tracing::debug!("Received model type update: {:?}", model_type);
            update_http_endpoints(http_service.clone(), model_type).await;
        }
    });

    // Pass the sender to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(receiver).await;
    });

    Ok(())
}

/// Updates HTTP service endpoints based on available model types
async fn update_http_endpoints(service: Arc<HttpService>, model_type: ModelUpdate) {
    tracing::debug!(
        "Updating HTTP service endpoints for model type: {:?}",
        model_type
    );
    match model_type {
        ModelUpdate::Added(model_type) => match model_type {
            ModelType::Chat => service.enable_chat_endpoints(true).await,
            ModelType::Completion => service.enable_cmpl_endpoints(true).await,
            ModelType::Embedding => service.enable_embeddings_endpoints(true).await,
            ModelType::Backend => {
                service.enable_chat_endpoints(true).await;
                service.enable_cmpl_endpoints(true).await;
            }
        },
        ModelUpdate::Removed(model_type) => match model_type {
            ModelType::Chat => service.enable_chat_endpoints(false).await,
            ModelType::Completion => service.enable_cmpl_endpoints(false).await,
            ModelType::Embedding => service.enable_embeddings_endpoints(false).await,
            ModelType::Backend => {
                service.enable_chat_endpoints(false).await;
                service.enable_cmpl_endpoints(false).await;
            }
        },
    }
}
