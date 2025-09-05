// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{MODEL_ROOT_PATH, ModelManager, ModelWatcher},
    engines::StreamingEngineAdapter,
    entrypoint::{self, EngineConfig, input::common},
    grpc::service::kserve,
    kv_router::KvRouterConfig,
    namespace::is_global_namespace,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::transports::etcd;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_runtime::{distributed::DistributedConfig, pipeline::RouterMode};

/// Build and run an KServe gRPC service
pub async fn run(runtime: Runtime, engine_config: EngineConfig) -> anyhow::Result<()> {
    let mut grpc_service_builder = kserve::KserveService::builder()
        .port(engine_config.local_model().http_port()) // [WIP] generalize port..
        .with_request_template(engine_config.local_model().request_template());

    let grpc_service = match engine_config {
        EngineConfig::Dynamic(_) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            let etcd_client = distributed_runtime.etcd_client();
            // This allows the /health endpoint to query etcd for active instances
            grpc_service_builder = grpc_service_builder.with_etcd_client(etcd_client.clone());
            let grpc_service = grpc_service_builder.build()?;
            match etcd_client {
                Some(ref etcd_client) => {
                    let router_config = engine_config.local_model().router_config();
                    // Listen for models registering themselves in etcd, add them to gRPC service
                    let namespace = engine_config.local_model().namespace().unwrap_or("");
                    let target_namespace = if is_global_namespace(namespace) {
                        None
                    } else {
                        Some(namespace.to_string())
                    };
                    run_watcher(
                        distributed_runtime,
                        grpc_service.state().manager_clone(),
                        etcd_client.clone(),
                        MODEL_ROOT_PATH,
                        router_config.router_mode,
                        Some(router_config.kv_router_config),
                        router_config.busy_threshold,
                        target_namespace,
                    )
                    .await?;
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
            grpc_service
        }
        EngineConfig::StaticRemote(local_model) => {
            let card = local_model.card();
            let router_mode = local_model.router_config().router_mode;

            let dst_config = DistributedConfig::from_settings(true); // true means static
            let distributed_runtime = DistributedRuntime::new(runtime.clone(), dst_config).await?;
            let grpc_service = grpc_service_builder.build()?;
            let manager = grpc_service.model_manager();

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

            let tokenizer_hf = card.tokenizer_hf()?;
            let chat_engine = entrypoint::build_routed_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(
                card,
                &client,
                router_mode,
                None,
                kv_chooser.clone(),
                tokenizer_hf.clone(),
            )
            .await?;
            manager.add_chat_completions_model(local_model.display_name(), chat_engine)?;

            let completions_engine =
                entrypoint::build_routed_pipeline::<
                    NvCreateCompletionRequest,
                    NvCreateCompletionResponse,
                >(card, &client, router_mode, None, kv_chooser, tokenizer_hf)
                .await?;
            manager.add_completions_model(local_model.display_name(), completions_engine)?;

            grpc_service
        }
        EngineConfig::StaticFull { engine, model, .. } => {
            let grpc_service = grpc_service_builder.build()?;
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = grpc_service.model_manager();
            manager.add_completions_model(model.service_name(), engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), engine)?;
            grpc_service
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
            ..
        } => {
            let grpc_service = grpc_service_builder.build()?;
            let manager = grpc_service.model_manager();

            let tokenizer_hf = model.card().tokenizer_hf()?;
            let chat_pipeline =
                common::build_pipeline::<
                    NvCreateChatCompletionRequest,
                    NvCreateChatCompletionStreamResponse,
                >(model.card(), inner_engine.clone(), tokenizer_hf.clone())
                .await?;
            manager.add_chat_completions_model(model.service_name(), chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(model.card(), inner_engine, tokenizer_hf)
            .await?;
            manager.add_completions_model(model.service_name(), cmpl_pipeline)?;
            grpc_service
        }
    };
    grpc_service.run(runtime.primary_token()).await?;
    runtime.shutdown(); // Cancel primary token
    Ok(())
}

/// Spawns a task that watches for new models in etcd at network_prefix,
/// and registers them with the ModelManager so that the HTTP service can use them.
#[allow(clippy::too_many_arguments)]
async fn run_watcher(
    runtime: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    etcd_client: etcd::Client,
    network_prefix: &str,
    router_mode: RouterMode,
    kv_router_config: Option<KvRouterConfig>,
    busy_threshold: Option<f64>,
    target_namespace: Option<String>,
) -> anyhow::Result<()> {
    let watch_obj = ModelWatcher::new(
        runtime,
        model_manager,
        router_mode,
        kv_router_config,
        busy_threshold,
    );
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();

    // [gluo NOTE] This is different from http::run_watcher where it alters the HTTP service
    // endpoint being exposed, gRPC doesn't have the same concept as the KServe service
    // only has one kind of inference endpoint.

    // Pass the sender to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(receiver, target_namespace.as_deref()).await;
    });

    Ok(())
}
