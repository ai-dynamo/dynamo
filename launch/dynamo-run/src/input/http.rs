// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::input::common;
use crate::{EngineConfig, Flags};
use dynamo_llm::http::service::ModelManager;
use dynamo_llm::{
    engines::StreamingEngineAdapter,
    http::service::{discovery, service_v2},
    request_template::RequestTemplate,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{CompletionRequest, CompletionResponse},
    },
};
use dynamo_runtime::component;
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
    let http_service = service_v2::HttpService::builder()
        .port(flags.http_port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .with_request_template(template)
        .build()?;
    match engine_config {
        EngineConfig::Dynamic => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            match distributed_runtime.etcd_client() {
                Some(etcd_client) => {
                    // Listen for models registering themselves in etcd, add them to HTTP service
                    run_watcher(
                        distributed_runtime,
                        http_service.model_manager().clone(),
                        etcd_client.clone(),
                        component::MODEL_ROOT_PATH,
                        flags.router_mode.into(),
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

            let cmpl_pipeline = common::build_pipeline::<CompletionRequest, CompletionResponse>(
                model.card(),
                inner_engine,
            )
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
    model_manager: ModelManager,
    etcd_client: etcd::Client,
    network_prefix: &str,
    router_mode: RouterMode,
) -> anyhow::Result<()> {
    let watch_obj =
        Arc::new(discovery::ModelWatcher::new(runtime, model_manager, router_mode).await?);
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
    let _watcher_task = tokio::spawn(watch_obj.watch(receiver));
    Ok(())
}
