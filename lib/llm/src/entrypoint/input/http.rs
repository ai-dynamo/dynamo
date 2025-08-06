// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelWatcher, MODEL_ROOT_PATH},
    engines::StreamingEngineAdapter,
    entrypoint::{self, input::common, EngineConfig},
    http::service::service_v2,
    kv_router::KvRouterConfig,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::transports::etcd;
use dynamo_runtime::{distributed::DistributedConfig, pipeline::RouterMode};
use dynamo_runtime::{DistributedRuntime, Runtime};

/// Build and run an HTTP service
pub async fn run(runtime: Runtime, engine_config: EngineConfig) -> anyhow::Result<()> {
    let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
    let etcd_client = distributed_runtime.etcd_client().clone();

    let http_service = service_v2::HttpService::builder()
        .port(engine_config.local_model().http_port())
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .enable_embeddings_endpoints(true)
        .with_request_template(engine_config.local_model().request_template())
        .with_etcd_client(etcd_client.clone())
        .build()?;

    match engine_config {
        EngineConfig::Dynamic(_) => {
            match etcd_client {
                Some(ref etcd_client) => {
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
        EngineConfig::StaticRemote(local_model) => {
            let card = local_model.card();
            let router_mode = local_model.router_config().router_mode;

            let dst_config = DistributedConfig::from_settings(true);
            let distributed_runtime = DistributedRuntime::new(runtime.clone(), dst_config).await?;
            let manager = http_service.model_manager();

            let endpoint_id = local_model.endpoint_id();
            let component = distributed_runtime
                .namespace(&endpoint_id.namespace)?
                .component(&endpoint_id.component)?;
            let client = component.endpoint(&endpoint_id.name).client().await?;

            let kv_chooser = if router_mode == RouterMode::KV {
                Some(
                    manager
                        .kv_chooser_for(
                            local_model.display_name(),
                            &component,
                            card.kv_cache_block_size,
                            Some(local_model.router_config().kv_router_config),
                        )
                        .await?,
                )
            } else {
                None
            };

            let chat_engine = entrypoint::build_routed_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(card, &client, router_mode, kv_chooser.clone())
            .await?;
            manager.add_chat_completions_model(local_model.display_name(), chat_engine)?;

            let completions_engine = entrypoint::build_routed_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(card, &client, router_mode, kv_chooser)
            .await?;
            manager.add_completions_model(local_model.display_name(), completions_engine)?;
        }
        EngineConfig::StaticFull { engine, model, .. } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            manager.add_completions_model(model.service_name(), engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), engine)?;
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
            ..
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
