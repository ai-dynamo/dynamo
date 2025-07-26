// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env::var;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::metrics;
use super::Metrics;
use super::RouteDoc;
use crate::discovery::ModelManager;
use crate::endpoint_type::EndpointType;
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::DistributedRuntime;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// HTTP service shared state
#[derive(Default)]
pub struct State {
    metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
    runtime: Option<Arc<DistributedRuntime>>,
    flags: RwLock<StateFlags>,
}

#[derive(Default, Debug, Clone)]
struct StateFlags {
    chat_endpoints_enabled: bool,
    cmpl_endpoints_enabled: bool,
    embeddings_endpoints_enabled: bool,
    responses_endpoints_enabled: bool,
}

impl StateFlags {
    pub fn get(&self, endpoint_type: &EndpointType) -> bool {
        match endpoint_type {
            EndpointType::Chat => self.chat_endpoints_enabled,
            EndpointType::Completion => self.cmpl_endpoints_enabled,
            EndpointType::Embedding => self.embeddings_endpoints_enabled,
            EndpointType::Responses => self.responses_endpoints_enabled,
        }
    }

    pub fn set(&mut self, endpoint_type: &EndpointType, enabled: bool) {
        match endpoint_type {
            EndpointType::Chat => self.chat_endpoints_enabled = enabled,
            EndpointType::Completion => self.cmpl_endpoints_enabled = enabled,
            EndpointType::Embedding => self.embeddings_endpoints_enabled = enabled,
            EndpointType::Responses => self.responses_endpoints_enabled = enabled,
        }
    }
}

impl State {
    pub fn new(manager: Arc<ModelManager>) -> Self {
        Self {
            manager,
            metrics: Arc::new(Metrics::default()),
            runtime: None,
            flags: RwLock::new(StateFlags {
                chat_endpoints_enabled: false,
                cmpl_endpoints_enabled: false,
                embeddings_endpoints_enabled: false,
                responses_endpoints_enabled: false,
            }),
        }
    }

    pub fn with_runtime(manager: Arc<ModelManager>, runtime: Arc<DistributedRuntime>) -> Self {
        Self {
            manager,
            metrics: Arc::new(Metrics::default()),
            runtime: Some(runtime),
            flags: RwLock::new(StateFlags {
                chat_endpoints_enabled: false,
                cmpl_endpoints_enabled: false,
                embeddings_endpoints_enabled: false,
                responses_endpoints_enabled: false,
            }),
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
    route_docs: Vec<RouteDoc>,
}

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct HttpServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

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
        tracing::info!(address, "Starting HTTP service on: {address}");

        let listener = tokio::net::TcpListener::bind(address.as_str())
            .await
            .unwrap_or_else(|_| panic!("could not bind to address: {address}"));

        let router = self.router.clone();
        let observer = cancel_token.child_token();

        axum::serve(listener, router)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
            .inspect_err(|_| cancel_token.cancel())?;

        Ok(())
    }

    /// Documentation of exposed HTTP endpoints
    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }

    pub async fn enable_model_endpoint(&self, endpoint_type: EndpointType, enable: bool) {
        let mut state_flags: tokio::sync::RwLockWriteGuard<'_, StateFlags> =
            self.state.flags.write().await;
        state_flags.set(&endpoint_type, enable);
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
        let state = Arc::new(State::new(model_manager));

        // Set initial state flags based on config
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut state_flags = state.flags.write().await;
                state_flags.chat_endpoints_enabled = config.enable_chat_endpoints;
                state_flags.cmpl_endpoints_enabled = config.enable_cmpl_endpoints;
                state_flags.embeddings_endpoints_enabled = config.enable_embeddings_endpoints;
                state_flags.responses_endpoints_enabled = config.enable_responses_endpoints;
            });
        });

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        state.metrics_clone().register(&registry)?;

        let mut router = axum::Router::new();

        let mut all_docs = Vec::new();

        let mut routes = vec![
            metrics::router(registry, var(HTTP_SVC_METRICS_PATH_ENV).ok()),
            super::openai::list_models_router(state.clone(), var(HTTP_SVC_MODELS_PATH_ENV).ok()),
            super::health::health_check_router(state.clone(), var(HTTP_SVC_HEALTH_PATH_ENV).ok()),
            super::health::live_check_router(state.clone(), var(HTTP_SVC_LIVE_PATH_ENV).ok()),
        ];

        let endpoint_routes =
            HttpServiceConfigBuilder::get_endpoints_router(state.clone(), &config);
        routes.extend(endpoint_routes);
        for (route_docs, route) in routes {
            router = router.merge(route);
            all_docs.extend(route_docs);
        }

        Ok(HttpService {
            state,
            router,
            port: config.port,
            host: config.host,
            route_docs: all_docs,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }

    fn get_endpoints_router(
        state: Arc<State>,
        config: &HttpServiceConfig,
    ) -> Vec<(Vec<RouteDoc>, axum::Router)> {
        let mut routes = Vec::new();

        // Add chat completions route with conditional middleware
        let (chat_docs, chat_route) = super::openai::chat_completions_router(
            state.clone(),
            config.request_template.clone(),
            var(HTTP_SVC_CHAT_PATH_ENV).ok(),
        );
        let (cmpl_docs, cmpl_route) =
            super::openai::completions_router(state.clone(), var(HTTP_SVC_CMP_PATH_ENV).ok());
        let (embed_docs, embed_route) =
            super::openai::embeddings_router(state.clone(), var(HTTP_SVC_EMB_PATH_ENV).ok());
        let (responses_docs, responses_route) = super::openai::responses_router(
            state.clone(),
            config.request_template.clone(),
            var(HTTP_SVC_RESPONSES_PATH_ENV).ok(),
        );

        let mut endpoint_routes = HashMap::new();
        endpoint_routes.insert(EndpointType::Chat, (chat_docs, chat_route));
        endpoint_routes.insert(EndpointType::Completion, (cmpl_docs, cmpl_route));
        endpoint_routes.insert(EndpointType::Embedding, (embed_docs, embed_route));
        endpoint_routes.insert(EndpointType::Responses, (responses_docs, responses_route));

        for endpoint_type in EndpointType::all() {
            let state_route = state.clone();
            if endpoint_routes.get(&endpoint_type).is_none() {
                tracing::debug!("{} endpoints are disabled", endpoint_type.as_str());
                continue;
            }
            let (docs, route) = endpoint_routes.get(&endpoint_type).cloned().unwrap();
            let route = route.route_layer(axum::middleware::from_fn(
                move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
                    let state: Arc<State> = state_route.clone();
                    async move {
                        // Read the flag value and drop the lock before async operations
                        let guard = state.flags.read().await;
                        let enabled = guard.get(&endpoint_type);

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
        return routes;
    }
}
