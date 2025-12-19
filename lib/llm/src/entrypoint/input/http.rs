// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelUpdate, ModelWatcher},
    endpoint_type::EndpointType,
    engines::StreamingEngineAdapter,
    entrypoint::{EngineConfig, EngineFactoryCallback, RouterConfig, input::common},
    http::service::service_v2::{self, HttpService},
    namespace::is_global_namespace,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::DistributedRuntime;

/// Build and run an HTTP service
pub async fn run(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
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

    // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
    // Pass the custom backend metrics endpoint as-is (already in namespace.component.endpoint format)
    http_service_builder = http_service_builder.with_custom_backend_config(
        local_model
            .custom_backend_metrics_endpoint()
            .map(|s| s.to_string()),
        local_model.custom_backend_metrics_polling_interval(),
    );

    let http_service = match engine_config {
        EngineConfig::Dynamic {
            ref model,
            ref engine_factory,
        } => {
            // This allows the /health endpoint to query store for active instances
            http_service_builder = http_service_builder.store(distributed_runtime.store().clone());
            let http_service = http_service_builder.build()?;

            let router_config = model.router_config();
            // Listen for models registering themselves, add them to HTTP service
            // Check if we should filter by namespace (based on the local model's namespace)
            // Get namespace from the model, fallback to endpoint_id namespace if not set
            let namespace = model.namespace().unwrap_or("");
            let target_namespace = if is_global_namespace(namespace) {
                None
            } else {
                Some(namespace.to_string())
            };
            run_watcher(
                distributed_runtime.clone(),
                http_service.state().manager_clone(),
                router_config.clone(),
                target_namespace,
                Arc::new(http_service.clone()),
                http_service.state().metrics_clone(),
                engine_factory.clone(),
            )
            .await?;
            http_service
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let http_service = http_service_builder.build()?;
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            let checksum = model.card().mdcsum();
            manager.add_completions_model(model.display_name(), checksum, engine.clone())?;
            manager.add_chat_completions_model(model.display_name(), checksum, engine)?;

            // Enable all endpoints
            for endpoint_type in EndpointType::all() {
                http_service.enable_model_endpoint(endpoint_type, true);
            }
            http_service
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let http_service = http_service_builder.build()?;
            let manager = http_service.model_manager();
            let checksum = model.card().mdcsum();

            let tokenizer_hf = model.card().tokenizer_hf()?;
            let chat_pipeline =
                common::build_pipeline::<
                    NvCreateChatCompletionRequest,
                    NvCreateChatCompletionStreamResponse,
                >(model.card(), inner_engine.clone(), tokenizer_hf.clone())
                .await?;
            manager.add_chat_completions_model(model.display_name(), checksum, chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(model.card(), inner_engine, tokenizer_hf)
            .await?;
            manager.add_completions_model(model.display_name(), checksum, cmpl_pipeline)?;
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

    // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
    // Start custom backend metrics polling if configured
    let polling_task =
        if let (Some(namespace_component_endpoint), Some(polling_interval), Some(registry)) = (
            http_service
                .custom_backend_namespace_component_endpoint
                .as_ref(),
            http_service.custom_backend_metrics_polling_interval,
            http_service.custom_backend_registry.as_ref(),
        ) {
            tracing::info!(
                namespace_component_endpoint=%namespace_component_endpoint,
                polling_interval_secs=polling_interval,
                "Starting custom backend metrics polling task"
            );
            // Spawn the polling task and keep the JoinHandle alive so it can be aborted during
            // shutdown. While graceful shutdown is not strictly necessary for this non-critical
            // metrics polling, explicitly aborting it prevents the task from running during the
            // shutdown phase.
            Some(
                crate::http::service::custom_backend_metrics::spawn_custom_backend_polling_task(
                    distributed_runtime.clone(),
                    namespace_component_endpoint.clone(),
                    polling_interval,
                    registry.clone(),
                ),
            )
        } else {
            None
        };

    http_service
        .run(distributed_runtime.primary_token())
        .await?;

    // Abort the polling task if it was started
    if let Some(task) = polling_task {
        task.abort();
    }

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
    http_service: Arc<HttpService>,
    metrics: Arc<crate::http::service::metrics::Metrics>,
    engine_factory: Option<EngineFactoryCallback>,
) -> anyhow::Result<()> {
    let mut watch_obj = ModelWatcher::new(
        runtime.clone(),
        model_manager,
        router_config,
        engine_factory,
        metrics.clone(),
    );
    tracing::debug!("Waiting for remote model");
    let discovery = runtime.discovery();
    let discovery_stream = discovery
        .list_and_watch(
            dynamo_runtime::discovery::DiscoveryQuery::AllModels,
            Some(runtime.primary_token()),
        )
        .await?;

    // Create a channel to receive model type updates
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    watch_obj.set_notify_on_model_update(tx);

    // Spawn a task to watch for model type changes and update HTTP service endpoints and metrics
    let _endpoint_enabler_task = tokio::spawn(async move {
        while let Some(model_update) = rx.recv().await {
            update_http_endpoints(http_service.clone(), model_update.clone());
            update_model_metrics(model_update, metrics.clone());
        }
    });

    // Pass the discovery stream to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj
            .watch(discovery_stream, target_namespace.as_deref())
            .await;
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
        ModelUpdate::Added(card) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in card.model_type.as_endpoint_types() {
                service.enable_model_endpoint(endpoint_type, true);
            }
        }
        ModelUpdate::Removed(card) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in card.model_type.as_endpoint_types() {
                service.enable_model_endpoint(endpoint_type, false);
            }
        }
    }
}

/// Updates metrics for model type changes
fn update_model_metrics(
    model_type: ModelUpdate,
    metrics: Arc<crate::http::service::metrics::Metrics>,
) {
    match model_type {
        ModelUpdate::Added(card) => {
            tracing::debug!("Updating metrics for added model: {}", card.display_name);
            if let Err(err) = metrics.update_metrics_from_mdc(&card) {
                tracing::warn!(%err, model_name=card.display_name, "update_metrics_from_mdc failed");
            }
        }
        ModelUpdate::Removed(card) => {
            tracing::debug!(model_name = card.display_name, "Model removed");
            // Note: Metrics are typically not removed to preserve historical data
            // This matches the behavior in the polling task
        }
    }
}
