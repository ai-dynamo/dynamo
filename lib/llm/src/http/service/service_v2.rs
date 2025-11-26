// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env::var;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::Metrics;
use super::RouteDoc;
use super::metrics;
use crate::discovery::ModelManager;
use crate::endpoint_type::EndpointType;
use crate::request_template::RequestTemplate;
use anyhow::Result;
use axum_server::tls_rustls::RustlsConfig;
use derive_builder::Builder;
use dynamo_runtime::discovery::{Discovery, KVStoreDiscovery};
use dynamo_runtime::logging::make_request_span;
use dynamo_runtime::metrics::prometheus_names::name_prefix;
use dynamo_runtime::storage::kv;
use std::net::SocketAddr;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

/// HTTP service shared state
pub struct State {
    metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
    store: kv::Manager,
    discovery_client: Arc<dyn Discovery>,
    flags: StateFlags,
}

#[derive(Default, Debug)]
struct StateFlags {
    chat_endpoints_enabled: AtomicBool,
    cmpl_endpoints_enabled: AtomicBool,
    embeddings_endpoints_enabled: AtomicBool,
    responses_endpoints_enabled: AtomicBool,
}

impl StateFlags {
    pub fn get(&self, endpoint_type: &EndpointType) -> bool {
        match endpoint_type {
            EndpointType::Chat => self.chat_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Completion => self.cmpl_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Embedding => self.embeddings_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Responses => self.responses_endpoints_enabled.load(Ordering::Relaxed),
        }
    }

    pub fn set(&self, endpoint_type: &EndpointType, enabled: bool) {
        match endpoint_type {
            EndpointType::Chat => self
                .chat_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Completion => self
                .cmpl_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Embedding => self
                .embeddings_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Responses => self
                .responses_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
        }
    }
}

impl State {
    pub fn new(manager: Arc<ModelManager>, store: kv::Manager) -> Self {
        // Initialize discovery backed by KV store
        // Create a cancellation token for the discovery's watch streams
        let discovery_client = {
            let cancel_token = CancellationToken::new();
            Arc::new(KVStoreDiscovery::new(store.clone(), cancel_token)) as Arc<dyn Discovery>
        };

        Self {
            manager,
            metrics: Arc::new(Metrics::default()),
            store,
            discovery_client,
            flags: StateFlags {
                chat_endpoints_enabled: AtomicBool::new(false),
                cmpl_endpoints_enabled: AtomicBool::new(false),
                embeddings_endpoints_enabled: AtomicBool::new(false),
                responses_endpoints_enabled: AtomicBool::new(false),
            },
        }
    }

    /// Get the Prometheus [`Metrics`] object which tracks request counts and inflight requests
    pub fn metrics_clone(&self) -> Arc<Metrics> {
        self.metrics.clone()
    }

    pub fn manager(&self) -> &ModelManager {
        Arc::as_ref(&self.manager)
    }

    pub fn manager_clone(&self) -> Arc<ModelManager> {
        self.manager.clone()
    }

    pub fn store(&self) -> &kv::Manager {
        &self.store
    }

    pub fn discovery(&self) -> Arc<dyn Discovery> {
        self.discovery_client.clone()
    }

    // TODO
    pub fn sse_keep_alive(&self) -> Option<Duration> {
        None
    }
}

#[derive(Clone)]
pub struct HttpService {
    // The state we share with every request handler
    state: Arc<State>,

    router: axum::Router,
    port: u16,
    host: String,
    enable_tls: bool,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    route_docs: Vec<RouteDoc>,

    // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
    pub(crate) custom_backend_namespace_component_endpoint: Option<String>,
    pub(crate) custom_backend_metrics_polling_interval: Option<f64>,
    pub(crate) custom_backend_registry:
        Option<Arc<super::custom_backend_metrics::CustomBackendMetricsRegistry>>,
}

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct HttpServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

    #[builder(default = "false")]
    enable_tls: bool,

    #[builder(default = "None")]
    tls_cert_path: Option<PathBuf>,

    #[builder(default = "None")]
    tls_key_path: Option<PathBuf>,

    // #[builder(default)]
    // custom: Vec<axum::Router>
    #[builder(default = "false")]
    enable_chat_endpoints: bool,

    #[builder(default = "false")]
    enable_cmpl_endpoints: bool,

    #[builder(default = "true")]
    enable_embeddings_endpoints: bool,

    #[builder(default = "true")]
    enable_responses_endpoints: bool,

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,

    #[builder(default)]
    store: kv::Manager,

    // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
    #[builder(default = "None")]
    custom_backend_namespace_component_endpoint: Option<String>,

    #[builder(default = "None")]
    custom_backend_metrics_polling_interval: Option<f64>,
}

impl HttpService {
    pub fn builder() -> HttpServiceConfigBuilder {
        HttpServiceConfigBuilder::default()
    }

    pub fn state_clone(&self) -> Arc<State> {
        self.state.clone()
    }

    pub fn state(&self) -> &State {
        Arc::as_ref(&self.state)
    }

    pub fn model_manager(&self) -> &ModelManager {
        self.state().manager()
    }

    pub async fn spawn(&self, cancel_token: CancellationToken) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run(cancel_token).await })
    }

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let address = format!("{}:{}", self.host, self.port);
        let protocol = if self.enable_tls { "HTTPS" } else { "HTTP" };
        tracing::info!(protocol, address, "Starting HTTP(S) service");

        let router = self.router.clone();
        let observer = cancel_token.child_token();

        let addr: SocketAddr = address
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", address, e))?;

        if self.enable_tls {
            let cert_path = self
                .tls_cert_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS certificate path not provided"))?;
            let key_path = self
                .tls_key_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS private key path not provided"))?;

            // aws_lc_rs is the default but other crates pull in `ring` also,
            // so rustls doesn't know which one to use. Tell it.
            if let Err(e) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
                tracing::debug!("TLS crypto provider already installed: {e:?}");
            }

            let config = RustlsConfig::from_pem_file(cert_path, key_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create TLS config: {}", e))?;

            let handle = axum_server::Handle::new();
            let server = axum_server::bind_rustls(addr, config)
                .handle(handle.clone())
                .serve(router.into_make_service());

            tokio::select! {
                result = server => {
                    result.map_err(|e| anyhow::anyhow!("HTTPS server error: {}", e))?;
                }
                _ = observer.cancelled() => {
                    tracing::info!("HTTPS server shutdown requested");
                    handle.graceful_shutdown(Some(Duration::from_secs(5)));
                    // TODO: Do we need to wait?
                }
            }
        } else {
            let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
                tracing::error!(
                    protocol = %protocol,
                    address = %address,
                    error = %e,
                    "Failed to bind server to address"
                );
                match e.kind() {
                    std::io::ErrorKind::AddrInUse => anyhow::anyhow!(
                        "Failed to start {} server: port {} already in use. Use --http-port to specify a different port.",
                        protocol,
                        self.port
                    ),
                    _ => anyhow::anyhow!(
                        "Failed to start {} server on {}: {}",
                        protocol,
                        address,
                        e
                    ),
                }
            })?;

            axum::serve(listener, router)
                .with_graceful_shutdown(observer.cancelled_owned())
                .await
                .inspect_err(|_| cancel_token.cancel())?;
        }

        Ok(())
    }

    /// Documentation of exposed HTTP endpoints
    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }

    pub fn enable_model_endpoint(&self, endpoint_type: EndpointType, enable: bool) {
        self.state.flags.set(&endpoint_type, enable);
        tracing::info!(
            "{} endpoints {}",
            endpoint_type.as_str(),
            if enable { "enabled" } else { "disabled" }
        );
    }
}

/// Environment variable to set the metrics endpoint path (default: `/metrics`)
static HTTP_SVC_METRICS_PATH_ENV: &str = "DYN_HTTP_SVC_METRICS_PATH";
/// Environment variable to set the models endpoint path (default: `/v1/models`)
static HTTP_SVC_MODELS_PATH_ENV: &str = "DYN_HTTP_SVC_MODELS_PATH";
/// Environment variable to set the health endpoint path (default: `/health`)
static HTTP_SVC_HEALTH_PATH_ENV: &str = "DYN_HTTP_SVC_HEALTH_PATH";
/// Environment variable to set the live endpoint path (default: `/live`)
static HTTP_SVC_LIVE_PATH_ENV: &str = "DYN_HTTP_SVC_LIVE_PATH";
/// Environment variable to set the chat completions endpoint path (default: `/v1/chat/completions`)
static HTTP_SVC_CHAT_PATH_ENV: &str = "DYN_HTTP_SVC_CHAT_PATH";
/// Environment variable to set the completions endpoint path (default: `/v1/completions`)
static HTTP_SVC_CMP_PATH_ENV: &str = "DYN_HTTP_SVC_CMP_PATH";
/// Environment variable to set the embeddings endpoint path (default: `/v1/embeddings`)
static HTTP_SVC_EMB_PATH_ENV: &str = "DYN_HTTP_SVC_EMB_PATH";
/// Environment variable to set the responses endpoint path (default: `/v1/responses`)
static HTTP_SVC_RESPONSES_PATH_ENV: &str = "DYN_HTTP_SVC_RESPONSES_PATH";

impl HttpServiceConfigBuilder {
    pub fn build(self) -> Result<HttpService, anyhow::Error> {
        let config: HttpServiceConfig = self.build_internal()?;

        let model_manager = Arc::new(ModelManager::new());
        let state = Arc::new(State::new(model_manager, config.store));
        state
            .flags
            .set(&EndpointType::Chat, config.enable_chat_endpoints);
        state
            .flags
            .set(&EndpointType::Completion, config.enable_cmpl_endpoints);
        state
            .flags
            .set(&EndpointType::Embedding, config.enable_embeddings_endpoints);
        state
            .flags
            .set(&EndpointType::Responses, config.enable_responses_endpoints);

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        state.metrics_clone().register(&registry)?;

        // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
        // Setup custom backend metrics if configured
        let custom_backend_registry =
            if config.custom_backend_namespace_component_endpoint.is_some()
                && config.custom_backend_metrics_polling_interval.is_some()
            {
                Some(Arc::new(
                    super::custom_backend_metrics::CustomBackendMetricsRegistry::new(
                        name_prefix::COMPONENT.to_string(),
                        registry.clone(),
                    ),
                ))
            } else {
                None
            };

        let mut router = axum::Router::new();

        let mut all_docs = Vec::new();

        let mut routes = vec![
            metrics::router(registry, var(HTTP_SVC_METRICS_PATH_ENV).ok()),
            super::openai::list_models_router(state.clone(), var(HTTP_SVC_MODELS_PATH_ENV).ok()),
            super::health::health_check_router(state.clone(), var(HTTP_SVC_HEALTH_PATH_ENV).ok()),
            super::health::live_check_router(state.clone(), var(HTTP_SVC_LIVE_PATH_ENV).ok()),
        ];

        let endpoint_routes =
            HttpServiceConfigBuilder::get_endpoints_router(state.clone(), &config.request_template);
        routes.extend(endpoint_routes);
        for (route_docs, route) in routes {
            router = router.merge(route);
            all_docs.extend(route_docs);
        }

        // Add OpenAPI documentation routes (must be after all other routes so it can document them)
        // Note: The path parameter is currently unused as SwaggerUi requires static paths
        let (openapi_docs, openapi_route) =
            super::openapi_docs::openapi_router(all_docs.clone(), None);
        router = router.merge(openapi_route);
        all_docs.extend(openapi_docs);

        // Add span for tracing
        router = router.layer(TraceLayer::new_for_http().make_span_with(make_request_span));

        Ok(HttpService {
            state,
            router,
            port: config.port,
            host: config.host,
            enable_tls: config.enable_tls,
            tls_cert_path: config.tls_cert_path,
            tls_key_path: config.tls_key_path,
            route_docs: all_docs,
            custom_backend_namespace_component_endpoint: config
                .custom_backend_namespace_component_endpoint,
            custom_backend_metrics_polling_interval: config.custom_backend_metrics_polling_interval,
            custom_backend_registry,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }

    // DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
    pub fn with_custom_backend_config(
        mut self,
        namespace_component_endpoint: Option<String>,
        polling_interval: Option<f64>,
    ) -> Self {
        self.custom_backend_namespace_component_endpoint = Some(namespace_component_endpoint);
        self.custom_backend_metrics_polling_interval = Some(polling_interval);
        self
    }

    fn get_endpoints_router(
        state: Arc<State>,
        request_template: &Option<RequestTemplate>,
    ) -> Vec<(Vec<RouteDoc>, axum::Router)> {
        let mut routes = Vec::new();
        // Add chat completions route with conditional middleware
        let (chat_docs, chat_route) = super::openai::chat_completions_router(
            state.clone(),
            request_template.clone(),
            var(HTTP_SVC_CHAT_PATH_ENV).ok(),
        );
        let (cmpl_docs, cmpl_route) =
            super::openai::completions_router(state.clone(), var(HTTP_SVC_CMP_PATH_ENV).ok());
        let (embed_docs, embed_route) =
            super::openai::embeddings_router(state.clone(), var(HTTP_SVC_EMB_PATH_ENV).ok());
        let (responses_docs, responses_route) = super::openai::responses_router(
            state.clone(),
            request_template.clone(),
            var(HTTP_SVC_RESPONSES_PATH_ENV).ok(),
        );

        let mut endpoint_routes = HashMap::new();
        endpoint_routes.insert(EndpointType::Chat, (chat_docs, chat_route));
        endpoint_routes.insert(EndpointType::Completion, (cmpl_docs, cmpl_route));
        endpoint_routes.insert(EndpointType::Embedding, (embed_docs, embed_route));
        endpoint_routes.insert(EndpointType::Responses, (responses_docs, responses_route));

        for endpoint_type in EndpointType::all() {
            let state_route = state.clone();
            if !endpoint_routes.contains_key(&endpoint_type) {
                tracing::debug!("{} endpoints are disabled", endpoint_type.as_str());
                continue;
            }
            let (docs, route) = endpoint_routes.get(&endpoint_type).cloned().unwrap();
            let route = route.route_layer(axum::middleware::from_fn(
                move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
                    let state: Arc<State> = state_route.clone();
                    async move {
                        // Check if the endpoint is enabled
                        let enabled = state.flags.get(&endpoint_type);
                        if enabled {
                            Ok(next.run(req).await)
                        } else {
                            tracing::debug!("{} endpoints are disabled", endpoint_type.as_str());
                            Err(axum::http::StatusCode::SERVICE_UNAVAILABLE)
                        }
                    }
                },
            ));
            routes.push((docs, route));
        }
        routes
    }
}
