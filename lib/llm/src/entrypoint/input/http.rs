// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{MODEL_ROOT_PATH, ModelManager, ModelUpdate, ModelWatcher},
    endpoint_type::EndpointType,
    engines::StreamingEngineAdapter,
    entrypoint::{self, EngineConfig, input::common},
    http::service::service_v2::{self, HttpService},
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

/// Build and run an HTTP service
pub async fn run(runtime: Runtime, engine_config: EngineConfig) -> anyhow::Result<()> {
    let local_model = engine_config.local_model();
    let mut http_service_builder = match (local_model.tls_cert_path(), local_model.tls_key_path()) {
        (Some(tls_cert_path), Some(tls_key_path)) => {
            if !tls_cert_path.exists() {
                anyhow::bail!("TLS certificate not found: {}", tls_cert_path.display());
            }
            if !tls_key_path.exists() {
                anyhow::bail!("TLS key not found: {}", tls_key_path.display());
            }
            service_v2::HttpService::builder()
                .enable_tls(true)
                .tls_cert_path(Some(tls_cert_path.to_path_buf()))
                .tls_key_path(Some(tls_key_path.to_path_buf()))
                .port(local_model.http_port())
        }
        (None, None) => service_v2::HttpService::builder().port(local_model.http_port()),
        (_, _) => {
            // CLI should prevent us ever getting here
            anyhow::bail!(
                "Both --tls-cert-path and --tls-key-path must be provided together to enable TLS"
            );
        }
    };
    if let Some(http_host) = local_model.http_host() {
        http_service_builder = http_service_builder.host(http_host);
    }
    http_service_builder =
        http_service_builder.with_request_template(engine_config.local_model().request_template());

    let http_service = match engine_config {
        EngineConfig::Dynamic(_) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            let etcd_client = distributed_runtime.etcd_client();
            // This allows the /health endpoint to query etcd for active instances
            http_service_builder = http_service_builder.with_etcd_client(etcd_client.clone());
            let http_service = http_service_builder.build()?;
            match etcd_client {
                Some(ref etcd_client) => {
                    let router_config = engine_config.local_model().router_config();
                    // Listen for models registering themselves in etcd, add them to HTTP service
                    // Check if we should filter by namespace (based on the local model's namespace)
                    // Get namespace from the model, fallback to endpoint_id namespace if not set
                    let namespace = engine_config.local_model().namespace().unwrap_or("");
                    let target_namespace = if is_global_namespace(namespace) {
                        None
                    } else {
                        Some(namespace.to_string())
                    };
                    run_watcher(
                        distributed_runtime,
                        http_service.state().manager_clone(),
                        etcd_client.clone(),
                        MODEL_ROOT_PATH,
                        router_config.router_mode,
                        Some(router_config.kv_router_config),
                        router_config.busy_threshold,
                        target_namespace,
                        Arc::new(http_service.clone()),
                        http_service.state().metrics_clone(),
                    )
                    .await?;
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
            http_service
        }
        EngineConfig::StaticRemote(local_model) => {
            let card = local_model.card();
            let router_mode = local_model.router_config().router_mode;

            let dst_config = DistributedConfig::from_settings(true); // true means static
            let distributed_runtime = DistributedRuntime::new(runtime.clone(), dst_config).await?;
            let http_service = http_service_builder.build()?;
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

            for endpoint_type in EndpointType::all() {
                http_service.enable_model_endpoint(endpoint_type, true);
            }

            http_service
        }
        EngineConfig::StaticFull { engine, model, .. } => {
            let http_service = http_service_builder.build()?;
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            manager.add_completions_model(model.service_name(), engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), engine)?;

            // Enable all endpoints
            for endpoint_type in EndpointType::all() {
                http_service.enable_model_endpoint(endpoint_type, true);
            }
            http_service
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
            ..
        } => {
            let http_service = http_service_builder.build()?;
            let manager = http_service.model_manager();

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
            // Enable all endpoints
            for endpoint_type in EndpointType::all() {
                http_service.enable_model_endpoint(endpoint_type, true);
            }
            http_service
        }
    };
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
    http_service: Arc<HttpService>,
    metrics: Arc<crate::http::service::metrics::Metrics>,
) -> anyhow::Result<()> {
    // Clone model_manager before it's moved into ModelWatcher
    let model_manager_clone = model_manager.clone();

    let mut watch_obj = ModelWatcher::new(
        runtime,
        model_manager,
        router_mode,
        kv_router_config,
        busy_threshold,
    );
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();

    // Create a channel to receive model type updates
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    watch_obj.set_notify_on_model_update(tx);

    // Spawn a task to watch for model type changes and update HTTP service endpoints and metrics
    let _endpoint_enabler_task = tokio::spawn(async move {
        while let Some(model_type) = rx.recv().await {
            tracing::debug!("Received model type update: {:?}", model_type);

            // Update HTTP endpoints (existing functionality)
            update_http_endpoints(http_service.clone(), model_type);

            // Update metrics (only for added models)
            update_model_metrics(
                model_type,
                model_manager_clone.clone(),
                metrics.clone(),
                Some(etcd_client.clone()),
            )
            .await;
        }
    });

    // Pass the sender to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(receiver, target_namespace.as_deref()).await;
    });

    Ok(())
}

/// Updates HTTP service endpoints based on available model types
fn update_http_endpoints(service: Arc<HttpService>, model_type: ModelUpdate) {
    tracing::debug!(
        "Updating HTTP service endpoints for model type: {:?}",
        model_type
    );
    match model_type {
        ModelUpdate::Added(model_type) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in model_type.as_endpoint_types() {
                service.enable_model_endpoint(endpoint_type, true);
            }
        }
        ModelUpdate::Removed(model_type) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in model_type.as_endpoint_types() {
                service.enable_model_endpoint(endpoint_type, false);
            }
        }
    }
}

/// Updates metrics for model type changes
async fn update_model_metrics(
    model_type: ModelUpdate,
    model_manager: Arc<ModelManager>,
    metrics: Arc<crate::http::service::metrics::Metrics>,
    etcd_client: Option<etcd::Client>,
) {
    match model_type {
        ModelUpdate::Added(model_type) => {
            tracing::debug!("Updating metrics for added model type: {:?}", model_type);

            // Get all model entries and update metrics for matching types
            let model_entries = model_manager.get_model_entries();
            for entry in model_entries {
                if entry.model_type == model_type {
                    // Update runtime config metrics if available
                    if let Some(runtime_config) = &entry.runtime_config {
                        metrics.update_runtime_config_metrics(&entry.name, runtime_config);
                    }

                    // Update MDC metrics if etcd is available
                    if let Some(ref etcd) = etcd_client
                        && let Err(e) = metrics
                            .update_metrics_from_model_entry_with_mdc(&entry, etcd)
                            .await
                    {
                        tracing::debug!(
                            model = %entry.name,
                            error = %e,
                            "Failed to update MDC metrics for newly added model"
                        );
                    }
                }
            }
        }
        ModelUpdate::Removed(model_type) => {
            tracing::debug!("Model type removed: {:?}", model_type);
            // Note: Metrics are typically not removed to preserve historical data
            // This matches the behavior in the polling task
        }
    }
}
