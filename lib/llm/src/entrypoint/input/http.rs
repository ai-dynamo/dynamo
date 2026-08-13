// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelUpdate, ModelWatcher},
    endpoint_type::EndpointType,
    engines::StreamingEngineAdapter,
    entrypoint::{ChatEngineFactoryCallback, EngineConfig, RouterConfig, input::common},
    http::service::{
        FrontendRouteExtension,
        service_v2::{self, HttpService},
    },
    kv_router::WorkerSelectorFactory,
    local_model::runtime_config::{ModelRuntimeConfig, TokenizerBackend},
    namespace::NamespaceFilter,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerSelectionPolicy, WorkerType,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::metrics::MetricsHierarchy;

/// Dynamo's complete discovery-backed HTTP frontend.
///
/// The default frontend uses [`DefaultWorkerSelector`]. A statically linked external crate can
/// replace only worker selection with [`Self::worker_selection_policy_factory`].
#[derive(Default)]
pub struct HttpFrontend {
    frontend_route_extensions: Vec<FrontendRouteExtension>,
    worker_selection_policy_factory: Option<WorkerSelectorFactory<WorkerSelectionPolicy>>,
}

impl HttpFrontend {
    /// Add system route extensions to the frontend.
    pub fn frontend_route_extensions(
        mut self,
        frontend_route_extensions: Vec<FrontendRouteExtension>,
    ) -> Self {
        self.frontend_route_extensions = frontend_route_extensions;
        self
    }

    /// Replace the default worker selector with a statically linked native policy.
    ///
    /// The factory is called when each decode or prefill worker set is constructed, not per
    /// request. Dynamo continues to own discovery, scheduling, validation, and accounting.
    pub fn worker_selection_policy_factory<F>(mut self, factory: F) -> Self
    where
        F: for<'a> Fn(
                &KvRouterConfig,
                WorkerType,
                RoutingPartitionRef<'a>,
            ) -> WorkerSelectionPolicy
            + Send
            + Sync
            + 'static,
    {
        self.worker_selection_policy_factory = Some(Arc::new(factory));
        self
    }

    /// Run the frontend until it exits.
    pub async fn run(
        self,
        distributed_runtime: DistributedRuntime,
        engine_config: EngineConfig,
    ) -> anyhow::Result<()> {
        if self.worker_selection_policy_factory.is_some()
            && !matches!(&engine_config, EngineConfig::Dynamic { .. })
        {
            anyhow::bail!("custom worker-selection policies require a dynamic engine");
        }

        super::initialize_input(&distributed_runtime, &engine_config).await;

        match self.worker_selection_policy_factory {
            Some(factory) => {
                run_with_worker_selector_factory(
                    distributed_runtime,
                    engine_config,
                    self.frontend_route_extensions,
                    factory,
                )
                .await
            }
            None => {
                run_with_worker_selector_factory(
                    distributed_runtime,
                    engine_config,
                    self.frontend_route_extensions,
                    Arc::new(|config, worker_type, _partition| {
                        DefaultWorkerSelector::new(Some(config.clone()), worker_type.as_str())
                    }),
                )
                .await
            }
        }
    }
}

/// Build and run an HTTP service
pub async fn run(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    HttpFrontend::default()
        .run(distributed_runtime, engine_config)
        .await
}

/// Build and run an HTTP service with additional system route extensions.
pub async fn run_with_frontend_route_extensions(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
    frontend_route_extensions: Vec<FrontendRouteExtension>,
) -> anyhow::Result<()> {
    HttpFrontend::default()
        .frontend_route_extensions(frontend_route_extensions)
        .run(distributed_runtime, engine_config)
        .await
}

async fn run_with_worker_selector_factory<Sel>(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
    frontend_route_extensions: Vec<FrontendRouteExtension>,
    worker_selector_factory: WorkerSelectorFactory<Sel>,
) -> anyhow::Result<()>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
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
        http_service_builder.cancel_token(Some(distributed_runtime.primary_token()));
    http_service_builder =
        http_service_builder.with_request_template(engine_config.local_model().request_template());
    http_service_builder = http_service_builder
        .metrics_config(local_model.metrics_config().clone())
        .frontend_api_config(local_model.frontend_api_config().clone());
    // Inject the DRT's metrics registry so that component-scoped metrics
    // (e.g. KvIndexerMetrics) are exposed (default port 8000 if not overridden).
    http_service_builder =
        http_service_builder.drt_metrics(Some(distributed_runtime.get_metrics_registry().clone()));

    // Wire DRT discovery so that router metrics (dynamo_router_*) are registered
    // with the instance_id as the router_id label.
    http_service_builder =
        http_service_builder.drt_discovery(Some(distributed_runtime.discovery()));
    http_service_builder =
        http_service_builder.runtime(Some(Arc::new(distributed_runtime.clone())));
    for extension in frontend_route_extensions {
        http_service_builder = http_service_builder.add_frontend_route_extension_arc(extension);
    }

    let http_service = match engine_config {
        EngineConfig::Dynamic {
            ref model,
            ref chat_engine_factory,
            ref prefill_load_estimator,
        } => {
            // Pass the discovery client so the /health endpoint can query active instances
            http_service_builder =
                http_service_builder.discovery(Some(distributed_runtime.discovery()));
            let http_service = http_service_builder.build()?;

            let router_config = model.router_config();
            let migration_limit = model.migration_limit();
            let migration_max_seq_len = model.migration_max_seq_len();
            // Listen for models registering themselves, add them to HTTP service
            // Create namespace filter from model configuration
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                model.namespace(),
                model.namespace_prefix(),
            );
            let local_model_path =
                (!model.path().as_os_str().is_empty()).then(|| model.path().to_path_buf());
            let generate_engine_capabilities = http_service.generate_engine_capabilities();
            run_watcher(
                distributed_runtime.clone(),
                http_service.state().manager_clone(),
                router_config.clone(),
                migration_limit,
                migration_max_seq_len,
                namespace_filter,
                Arc::new(http_service.clone()),
                http_service.state().metrics_clone(),
                chat_engine_factory.clone(),
                prefill_load_estimator.clone(),
                local_model_path,
                model.runtime_config().tokenizer_backend,
                generate_engine_capabilities,
                worker_selector_factory.clone(),
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

            enable_in_process_model_endpoints(&http_service)?;
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

            let tokenizer = model.card().tokenizer()?;
            let chat_pipeline = common::build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine.clone(), tokenizer.clone())
            .await?;
            manager.add_chat_completions_model(model.display_name(), checksum, chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(model.card(), inner_engine, tokenizer)
            .await?;
            manager.add_completions_model(model.display_name(), checksum, cmpl_pipeline)?;
            enable_in_process_model_endpoints(&http_service)?;
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

    http_service
        .run(distributed_runtime.primary_token())
        .await?;

    distributed_runtime.shutdown(); // Cancel primary token
    Ok(())
}

fn enable_in_process_model_endpoints(http_service: &HttpService) -> anyhow::Result<()> {
    for endpoint_type in EndpointType::all() {
        if endpoint_type != EndpointType::Batch {
            http_service.enable_model_endpoint(endpoint_type, true)?;
        }
    }
    Ok(())
}

/// Spawns a task that watches for new models in store,
/// and registers them with the ModelManager so that the HTTP service can use them.
#[allow(clippy::too_many_arguments)]
async fn run_watcher<Sel>(
    runtime: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    router_config: RouterConfig,
    migration_limit: u32,
    migration_max_seq_len: Option<u32>,
    namespace_filter: NamespaceFilter,
    http_service: Arc<HttpService>,
    metrics: Arc<crate::http::service::metrics::Metrics>,
    chat_engine_factory: Option<ChatEngineFactoryCallback>,
    prefill_load_estimator: Option<Arc<dyn dynamo_kv_router::PrefillLoadEstimator>>,
    local_model_path: Option<PathBuf>,
    tokenizer_backend: Option<TokenizerBackend>,
    generate_engine_capabilities: Vec<&'static str>,
    worker_selector_factory: WorkerSelectorFactory<Sel>,
) -> anyhow::Result<()>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    // Start the LoRA allocation controller when LoRA serving is enabled. The
    // controller itself is additionally gated on the allocation config
    // (DYN_LORA_ALLOCATION_ENABLED) inside `start_lora_controller`.
    if crate::lora::lora_serving_enabled() {
        let cancel_token = runtime.primary_token();
        let _controller_handle = model_manager.start_lora_controller(cancel_token);
    }

    let mut watch_obj = ModelWatcher::new_with_worker_selector_factory(
        runtime.clone(),
        model_manager,
        router_config,
        migration_limit,
        migration_max_seq_len,
        chat_engine_factory,
        prefill_load_estimator,
        metrics.clone(),
        worker_selector_factory,
    );
    watch_obj.set_local_model_path(local_model_path);
    watch_obj.set_tokenizer_backend(tokenizer_backend);
    watch_obj.set_generate_engine_capabilities(generate_engine_capabilities);
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
    let watch_obj = Arc::new(watch_obj);

    // Spawn a task to watch for model type changes and update HTTP service endpoints and metrics
    let _endpoint_enabler_task = tokio::spawn(async move {
        while let Some(model_update) = rx.recv().await {
            if let Err(error) = update_http_endpoints(http_service.clone(), model_update.clone()) {
                tracing::error!(%error, "failed to update HTTP endpoints");
            }
            update_model_metrics(model_update, metrics.clone());
        }
    });

    // Pass the discovery stream to the watcher
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(discovery_stream, namespace_filter).await;
    });

    Ok(())
}

/// Updates HTTP service endpoints based on available model types
fn update_http_endpoints(service: Arc<HttpService>, model_type: ModelUpdate) -> anyhow::Result<()> {
    tracing::debug!(
        "Updating HTTP service endpoints for model type: {:?}",
        model_type
    );
    match model_type {
        ModelUpdate::Added(card) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in card
                .model_type
                .as_endpoint_types_with_anthropic(service.anthropic_api_enabled())
            {
                service.enable_model_endpoint(endpoint_type, true)?;
            }
        }
        ModelUpdate::Removed(card) => {
            // Handle all supported endpoint types, not just the first one
            for endpoint_type in card
                .model_type
                .as_endpoint_types_with_anthropic(service.anthropic_api_enabled())
            {
                service.enable_model_endpoint(endpoint_type, false)?;
            }
        }
    }
    Ok(())
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
