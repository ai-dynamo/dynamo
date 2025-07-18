// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelWatcher, MODEL_ROOT_PATH},
    engines::StreamingEngineAdapter,
    entrypoint::{input::common, EngineConfig},
    http::service::service_v2,
    kv_router::KvRouterConfig,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::transports::etcd;
use dynamo_runtime::{DistributedRuntime, Runtime};

/// Build and run an HTTP service
pub async fn run(runtime: Runtime, engine_config: EngineConfig) -> anyhow::Result<()> {
    let http_service = service_v2::HttpService::builder()
        .port(engine_config.local_model().http_port())
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .enable_embeddings_endpoints(true)
        .with_request_template(engine_config.local_model().request_template())
        .build()?;
    match engine_config {
        EngineConfig::Dynamic(_) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            match distributed_runtime.etcd_client() {
                Some(etcd_client) => {
                    let router_config = engine_config.local_model().router_config();
                    // Listen for models registering themselves in etcd, add them to HTTP service
                    run_watcher(
                        distributed_runtime,
                        http_service.state().manager_clone(),
                        etcd_client.clone(),
                        MODEL_ROOT_PATH,
                        router_config.router_mode,
                        Some(router_config.kv_router_config),
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
                NvCreateCompletionResponse,
            >(model.card(), inner_engine)
            .await?;
            manager.add_completions_model(model.service_name(), cmpl_pipeline)?;
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
) -> anyhow::Result<()> {
    let watch_obj = ModelWatcher::new(runtime, model_manager, router_mode, kv_router_config);
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(receiver).await;
    });
    Ok(())
}
