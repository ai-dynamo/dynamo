// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::metrics;
use super::Metrics;
use super::RouteDoc;
use crate::discovery::ModelManager;
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

    /// Get the DistributedRuntime if available
    pub fn runtime(&self) -> Option<&DistributedRuntime> {
        self.runtime.as_ref().map(|r| r.as_ref())
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

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,

    #[builder(default = "None")]
    runtime: Option<Arc<DistributedRuntime>>,
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

    /// Enable or disable chat completion endpoints
    pub async fn enable_chat_endpoints(&self, enable: bool) {
        let mut state_flags = self.state.flags.write().await;
        state_flags.chat_endpoints_enabled = enable;
        tracing::info!(
            "Chat completion endpoints {}",
            if enable { "enabled" } else { "disabled" }
        );
    }

    /// Enable or disable completion endpoints
    pub async fn enable_cmpl_endpoints(&self, enable: bool) {
        let mut state_flags = self.state.flags.write().await;
        state_flags.cmpl_endpoints_enabled = enable;
        tracing::info!(
            "Completion endpoints {}",
            if enable { "enabled" } else { "disabled" }
        );
    }

    /// Enable or disable embeddings endpoints
    pub async fn enable_embeddings_endpoints(&self, enable: bool) {
        let mut state_flags = self.state.flags.write().await;
        state_flags.embeddings_endpoints_enabled = enable;
        tracing::info!(
            "Embeddings endpoints {}",
            if enable { "enabled" } else { "disabled" }
        );
    }
}

impl HttpServiceConfigBuilder {
    pub fn build(self) -> Result<HttpService, anyhow::Error> {
        let config: HttpServiceConfig = self.build_internal()?;

        let model_manager = Arc::new(ModelManager::new());
        let state = if let Some(runtime) = config.runtime {
            Arc::new(State::with_runtime(model_manager, runtime))
        } else {
            Arc::new(State::new(model_manager))
        };

        // Set initial state flags based on config
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut state_flags = state.flags.write().await;
                state_flags.chat_endpoints_enabled = config.enable_chat_endpoints;
                state_flags.cmpl_endpoints_enabled = config.enable_cmpl_endpoints;
                state_flags.embeddings_endpoints_enabled = config.enable_embeddings_endpoints;
            });
        });

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        state.metrics_clone().register(&registry)?;

        let mut router = axum::Router::new();

        let mut all_docs = Vec::new();

        let mut routes = vec![
            metrics::router(registry, None),
            super::openai::list_models_router(state.clone(), None),
            super::health::health_check_router(state.clone(), None),
            super::clear_kv_blocks::clear_kv_blocks_router(state.clone(), None),
        ];

        // Add chat completions route with conditional middleware
        let (chat_docs, chat_route) =
            super::openai::chat_completions_router(state.clone(), config.request_template, None);
        let state_chat_route = state.clone();
        let chat_route = chat_route.route_layer(axum::middleware::from_fn(
            move |req, next: axum::middleware::Next| {
                let state = state_chat_route.clone();
                async move {
                    // Read the flag value and drop the lock before async operations
                    let guard = state.flags.read().await;
                    let enabled = guard.chat_endpoints_enabled;

                    if enabled {
                        Ok(next.run(req).await)
                    } else {
                        tracing::debug!("Chat endpoints are disabled");
                        Err(axum::http::StatusCode::SERVICE_UNAVAILABLE)
                    }
                }
            },
        ));
        routes.push((chat_docs, chat_route));

        // Add completions route with conditional middleware
        let state_cmpl = state.clone();
        let (cmpl_docs, cmpl_route) = super::openai::completions_router(state_cmpl, None);

        let state_cmpl_route = state.clone();
        let cmpl_route = cmpl_route.route_layer(axum::middleware::from_fn(
            move |req, next: axum::middleware::Next| {
                let state_api = state_cmpl_route.clone();
                async move {
                    let guard = state_api.flags.read().await;
                    let enabled = guard.cmpl_endpoints_enabled;

                    if enabled {
                        Ok(next.run(req).await)
                    } else {
                        Err(axum::http::StatusCode::NOT_FOUND)
                    }
                }
            },
        ));
        routes.push((cmpl_docs, cmpl_route));

        // Add embeddings route with conditional middleware
        let (embed_docs, embed_route) = super::openai::embeddings_router(state.clone(), None);
        let state_embed_route = state.clone();
        let embed_route = embed_route.route_layer(axum::middleware::from_fn(
            move |req, next: axum::middleware::Next| {
                let state_api = state_embed_route.clone();
                async move {
                    let guard = state_api.flags.read().await;
                    let enabled = guard.embeddings_endpoints_enabled;

                    if enabled {
                        Ok(next.run(req).await)
                    } else {
                        Err(axum::http::StatusCode::NOT_FOUND)
                    }
                }
            },
        ));
        routes.push((embed_docs, embed_route));

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
}
