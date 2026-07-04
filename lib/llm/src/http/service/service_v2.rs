// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env::var;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::http::Response;
use axum::response::IntoResponse;

use super::RouteDoc;
use super::metrics;
use super::metrics::register_lora_allocation_metrics;
use crate::discovery::ModelManager;
use crate::endpoint_type::EndpointType;
use crate::kv_router::metrics::{
    RoutingOverheadMetrics, register_router_queue_metrics, register_worker_load_metrics,
};
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use dynamo_http_server::metrics::{Metrics, Registry, register_worker_timing_metrics};
use dynamo_http_server::{
    HttpServer as CoreHttpServer, HttpServerConfig as CoreHttpServerConfig, InflightPermit,
    ServiceObserver, ServiceStage, TlsConfig as CoreTlsConfig, echo_request_id_header,
    track_inflight_response,
};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::config::env_is_truthy;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::discovery::Discovery;
use dynamo_runtime::logging::{make_inference_request_span, make_system_request_span};
use dynamo_runtime::metrics::{
    frontend_perf::ensure_frontend_perf_metrics_registered_prometheus,
    request_plane::ensure_request_plane_metrics_registered_prometheus,
    tokio_perf::{ensure_tokio_perf_metrics_registered_prometheus, tokio_metrics_and_canary_loop},
    transport_metrics::ensure_transport_metrics_registered_prometheus,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

use crate::frontend_config::{FrontendApiConfig, MetricsConfig};

async fn track_inflight_inference(
    axum::extract::State(state): axum::extract::State<Arc<State>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let Some(permit) = state.service_observer.try_acquire_inflight() else {
        return super::openai::ErrorMessage::_service_unavailable().into_response();
    };

    let response = next.run(request).await;
    track_inflight_response(response, permit)
}

/// HTTP service shared state
pub struct State {
    metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
    discovery_client: Arc<dyn Discovery>,
    service_observer: Arc<ServiceObserver>,
    flags: StateFlags,
    cancel_token: CancellationToken,
    // Frontend API behavior read by request handlers after the service is built.
    frontend_api_config: FrontendApiConfig,
    nvext_enabled: bool,
}

/// Typed config needed only to construct HTTP shared state.
///
/// `MetricsConfig` initializes the per-service metrics object, while
/// `FrontendApiConfig` is retained in `State` for route and handler decisions.
struct StateConfig {
    metrics_config: MetricsConfig,
    frontend_api_config: FrontendApiConfig,
    nvext_enabled: bool,
}

#[derive(Default, Debug)]
struct StateFlags {
    chat_endpoints_enabled: AtomicBool,
    cmpl_endpoints_enabled: AtomicBool,
    embeddings_endpoints_enabled: AtomicBool,
    images_endpoints_enabled: AtomicBool,
    videos_endpoints_enabled: AtomicBool,
    audios_endpoints_enabled: AtomicBool,
    realtime_endpoints_enabled: AtomicBool,
    responses_endpoints_enabled: AtomicBool,
    anthropic_endpoints_enabled: AtomicBool,
    generate_endpoints_enabled: AtomicBool,
}

impl StateFlags {
    pub fn get(&self, endpoint_type: &EndpointType) -> bool {
        match endpoint_type {
            EndpointType::Chat => self.chat_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Completion => self.cmpl_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Embedding => self.embeddings_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Images => self.images_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Videos => self.videos_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Audios => self.audios_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Realtime => self.realtime_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::Responses => self.responses_endpoints_enabled.load(Ordering::Relaxed),
            EndpointType::AnthropicMessages => {
                self.anthropic_endpoints_enabled.load(Ordering::Relaxed)
            }
            EndpointType::Generate => self.generate_endpoints_enabled.load(Ordering::Relaxed),
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
            EndpointType::Images => self
                .images_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Videos => self
                .videos_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Audios => self
                .audios_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Realtime => self
                .realtime_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Responses => self
                .responses_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::AnthropicMessages => self
                .anthropic_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
            EndpointType::Generate => self
                .generate_endpoints_enabled
                .store(enabled, Ordering::Relaxed),
        }
    }
}

impl State {
    fn new(
        manager: Arc<ModelManager>,
        discovery_client: Arc<dyn Discovery>,
        cancel_token: CancellationToken,
        config: StateConfig,
    ) -> Self {
        Self {
            manager,
            metrics: Arc::new(Metrics::new_with_prefix(config.metrics_config.prefix())),
            discovery_client,
            service_observer: Arc::new(ServiceObserver::default()),
            nvext_enabled: config.nvext_enabled,
            flags: StateFlags {
                chat_endpoints_enabled: AtomicBool::new(false),
                cmpl_endpoints_enabled: AtomicBool::new(false),
                embeddings_endpoints_enabled: AtomicBool::new(false),
                images_endpoints_enabled: AtomicBool::new(false),
                videos_endpoints_enabled: AtomicBool::new(false),
                audios_endpoints_enabled: AtomicBool::new(false),
                realtime_endpoints_enabled: AtomicBool::new(false),
                responses_endpoints_enabled: AtomicBool::new(false),
                anthropic_endpoints_enabled: AtomicBool::new(false),
                generate_endpoints_enabled: AtomicBool::new(false),
            },
            cancel_token,
            frontend_api_config: config.frontend_api_config,
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

    pub fn discovery(&self) -> Arc<dyn Discovery> {
        self.discovery_client.clone()
    }

    pub fn service_observer(&self) -> Arc<ServiceObserver> {
        self.service_observer.clone()
    }

    pub fn service_stage(&self) -> ServiceStage {
        self.service_observer.stage()
    }

    pub fn is_ready(&self) -> bool {
        self.service_observer.is_ready()
    }

    pub fn start_draining(&self) {
        self.service_observer.start_draining();
    }

    pub fn start_stopping(&self) {
        self.service_observer.start_stopping();
    }

    pub fn acquire_inflight(&self) -> InflightPermit {
        self.service_observer.acquire_inflight()
    }

    pub fn inflight_count(&self) -> u64 {
        self.service_observer.inflight_count()
    }

    pub async fn wait_inflight_zero_or_timeout(&self, timeout: Duration) -> bool {
        self.service_observer
            .wait_inflight_zero_or_timeout(timeout)
            .await
    }

    /// Check if the service is shutting down
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Master switch for the `nvext` extension protocol (see
    /// [`environment_names::llm::DYN_DISABLE_FRONTEND_NVEXT`]).
    #[inline]
    pub fn nvext_enabled(&self) -> bool {
        self.nvext_enabled
    }

    /// Get the cancellation token
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    // TODO
    pub fn sse_keep_alive(&self) -> Option<Duration> {
        None
    }

    /// Returns true if Anthropic billing preamble stripping is enabled.
    pub fn strip_anthropic_preamble_enabled(&self) -> bool {
        self.frontend_api_config.anthropic().strip_preamble()
    }

    /// Returns true if the Anthropic Messages API is enabled by service config.
    pub fn anthropic_api_enabled(&self) -> bool {
        self.frontend_api_config.anthropic().enabled()
    }

    /// Returns true if streaming tool call dispatch is enabled.
    ///
    /// When enabled, the chat completions streaming path emits `event: tool_call_dispatch`
    /// SSE events for each complete tool call, letting clients start processing tool calls
    /// before `finish_reason="tool_calls"` arrives.
    pub fn streaming_tool_dispatch_enabled(&self) -> bool {
        self.frontend_api_config
            .streaming_dispatch()
            .tool_dispatch()
    }

    /// Returns true if streaming reasoning dispatch is enabled.
    ///
    /// When enabled, the chat completions streaming path accumulates reasoning tokens and
    /// emits a single `event: reasoning_dispatch` SSE event with the complete reasoning
    /// block once thinking ends (DeepSeek-R1, Qwen3, etc.).
    pub fn streaming_reasoning_dispatch_enabled(&self) -> bool {
        self.frontend_api_config
            .streaming_dispatch()
            .reasoning_dispatch()
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
    /// RL worker discovery router, served on a dedicated port when enabled.
    rl_router: Option<axum::Router>,
    rl_port: u16,
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

    /// Metrics naming config used when initializing the HTTP service metrics registry.
    #[builder(default)]
    metrics_config: MetricsConfig,

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

    /// Experimental token-in/token-out `Generate` API
    /// (`POST /inference/v1/generate`). Enabled by default, matching vLLM,
    /// which mounts this endpoint for any generate-capable model. The handler
    /// is a placeholder (HTTP 501) until request dispatch lands; set to
    /// `false` to hide the route entirely.
    #[builder(default = "true")]
    enable_generate_endpoints: bool,

    /// API behavior config retained in HTTP state for route and streaming decisions.
    #[builder(default)]
    frontend_api_config: FrontendApiConfig,

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,

    #[builder(default = "None")]
    discovery: Option<Arc<dyn Discovery>>,

    #[builder(default = "None")]
    cancel_token: Option<CancellationToken>,

    /// When set, the `/metrics` endpoint will also expose metrics from the
    /// DRT's registry tree (anything created via `metrics().create*()`).
    #[builder(default = "None")]
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,

    /// When set (e.g. DRT discovery), router metrics (dynamo_router_* with router_id label)
    /// are registered using discovery.instance_id() and exposed on /metrics.
    #[builder(default = "None")]
    drt_discovery: Option<Arc<dyn Discovery>>,

    /// When true, serve the RL worker discovery API on `rl_port`.
    #[builder(default = "false")]
    enable_rl: bool,

    /// Master switch for the `nvext` extension protocol. Default `true`;
    /// env-truthy `DYN_DISABLE_FRONTEND_NVEXT` overrides to `false`.
    #[builder(default = "true")]
    enable_nvext: bool,

    /// Master switch for the frontend admin API surface (`GET` /
    /// `POST /busy_threshold`). Default `true`; env-truthy
    /// `DYN_DISABLE_FRONTEND_ADMIN_API` overrides to `false`.
    #[builder(default = "true")]
    enable_admin_api: bool,

    /// Port for the RL worker discovery listener. Defaults to `DYN_RL_PORT` or 8001.
    #[builder(default = "default_rl_port()")]
    rl_port: u16,

    /// Distributed runtime used by the RL worker discovery API.
    #[builder(default = "None")]
    runtime: Option<Arc<DistributedRuntime>>,
}

fn default_rl_port() -> u16 {
    std::env::var("DYN_RL_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8001)
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

    pub fn anthropic_api_enabled(&self) -> bool {
        self.state().anthropic_api_enabled()
    }

    pub async fn spawn(&self, cancel_token: CancellationToken) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run(cancel_token).await })
    }

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        self.run_inner(cancel_token, None).await
    }

    /// Like [`spawn`], but uses a caller-provided pre-bound listener. Closes the TOCTOU
    /// port-allocation gap for tests that need to know the bound port up front. Not
    /// supported in TLS mode: TLS uses `axum_server::bind_rustls`, which owns its own
    /// bind, so a pre-bound listener cannot be threaded through and dropping it before
    /// `bind_rustls` would just re-open the same race. Returns an error if invoked on a
    /// service built with `enable_tls(true)`.
    ///
    /// [`spawn`]: HttpService::spawn
    pub async fn spawn_with_listener(
        &self,
        cancel_token: CancellationToken,
        listener: tokio::net::TcpListener,
    ) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run_with_listener(cancel_token, listener).await })
    }

    /// Like [`run`], but serves on a caller-provided pre-bound listener instead of
    /// binding `{host}:{port}` internally. See [`spawn_with_listener`] for the TLS
    /// restriction.
    ///
    /// [`run`]: HttpService::run
    /// [`spawn_with_listener`]: HttpService::spawn_with_listener
    pub async fn run_with_listener(
        &self,
        cancel_token: CancellationToken,
        listener: tokio::net::TcpListener,
    ) -> Result<()> {
        self.run_inner(cancel_token, Some(listener)).await
    }

    async fn run_inner(
        &self,
        cancel_token: CancellationToken,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<()> {
        if self.enable_tls && listener.is_some() {
            return Err(anyhow::anyhow!(
                "Pre-bound listener is not supported in TLS mode; \
                 axum_server::bind_rustls owns its own bind. \
                 Use run()/spawn() (which bind internally) when enable_tls is set."
            ));
        }
        let tls = if self.enable_tls {
            Some(CoreTlsConfig {
                cert_path: self
                    .tls_cert_path
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("TLS certificate path not provided"))?,
                key_path: self
                    .tls_key_path
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("TLS private key path not provided"))?,
            })
        } else {
            None
        };

        self.spawn_rl_listener_if_configured(&cancel_token).await?;
        tokio::spawn(tokio_metrics_and_canary_loop(cancel_token.clone()));

        let server = CoreHttpServer::new(
            self.router.clone(),
            CoreHttpServerConfig {
                host: self.host.clone(),
                port: self.port,
                tls,
                graceful_shutdown_timeout: Duration::from_secs(
                    get_graceful_shutdown_timeout() as u64
                ),
            },
            self.state.service_observer(),
            self.state.cancel_token().clone(),
        );
        let service_cancel = cancel_token.clone();
        let result = match listener {
            Some(listener) => server.run_with_listener(cancel_token, listener).await,
            None => server.run(cancel_token).await,
        };
        service_cancel.cancel();
        result
    }

    async fn spawn_rl_listener_if_configured(
        &self,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        let Some(rl_router) = self.rl_router.clone() else {
            return Ok(());
        };
        let rl_addr = format!("{}:{}", self.host, self.rl_port);
        // Bind eagerly and fail fast: when RL discovery is enabled, a bind failure
        // should abort service startup rather than silently leave RL discovery
        // unavailable while the main HTTP service keeps running.
        let listener = tokio::net::TcpListener::bind(&rl_addr).await.map_err(|e| {
            tracing::error!(
                address = %rl_addr,
                error = %e,
                "Failed to bind RL worker discovery listener"
            );
            anyhow::anyhow!("Failed to bind RL worker discovery listener on {rl_addr}: {e}")
        })?;
        tracing::info!(
            address = %rl_addr,
            "RL worker discovery listener started"
        );
        let rl_cancel = cancel_token.child_token();
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, rl_router)
                .with_graceful_shutdown(async move {
                    rl_cancel.cancelled_owned().await;
                })
                .await
            {
                tracing::error!("RL worker discovery listener error: {e}");
            }
        });
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

fn get_graceful_shutdown_timeout() -> usize {
    std::env::var(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
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
/// Environment variable to set the anthropic messages endpoint path (default: `/v1/messages`)
static HTTP_SVC_ANTHROPIC_PATH_ENV: &str = "DYN_HTTP_SVC_ANTHROPIC_PATH";
/// Environment variable to set the generate endpoint path (default: `/inference/v1/generate`)
static HTTP_SVC_GENERATE_PATH_ENV: &str = "DYN_HTTP_SVC_GENERATE_PATH";

impl HttpServiceConfigBuilder {
    pub fn build(self) -> Result<HttpService, anyhow::Error> {
        let config: HttpServiceConfig = self.build_internal()?;
        let metrics_config = config.metrics_config.clone();
        let frontend_api_config = config.frontend_api_config.clone();
        let anthropic_endpoints_enabled = frontend_api_config.anthropic().enabled();
        let generate_endpoints_enabled = config.enable_generate_endpoints;

        let model_manager = Arc::new(ModelManager::new());
        let cancel_token = config.cancel_token.unwrap_or_default();
        // Use the provided discovery client, or fall back to a no-op memory-backed one
        // (for in-process modes that don't need discovery)
        let discovery_client = config.discovery.unwrap_or_else(|| {
            use dynamo_runtime::discovery::KVStoreDiscovery;
            Arc::new(KVStoreDiscovery::new(
                dynamo_runtime::storage::kv::Manager::memory(),
                cancel_token.child_token(),
            )) as Arc<dyn Discovery>
        });
        // Both surfaces are on by default; an env-truthy DISABLE var turns them
        // off. The builder flag can also force off (e.g. tests), and wins.
        let nvext_enabled =
            config.enable_nvext && !env_is_truthy(env_llm::DYN_DISABLE_FRONTEND_NVEXT);
        let admin_api_enabled =
            config.enable_admin_api && !env_is_truthy(env_llm::DYN_DISABLE_FRONTEND_ADMIN_API);

        let state = Arc::new(State::new(
            model_manager,
            discovery_client,
            cancel_token,
            StateConfig {
                metrics_config,
                frontend_api_config,
                nvext_enabled,
            },
        ));
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
        state.flags.set(
            &EndpointType::AnthropicMessages,
            anthropic_endpoints_enabled,
        );
        state
            .flags
            .set(&EndpointType::Generate, generate_endpoints_enabled);

        // enable prometheus metrics
        let registry = Registry::new();
        state.metrics_clone().register(&registry)?;

        // Register worker load metrics (active_decode_blocks, active_prefill_tokens per worker)
        // These are updated by KvWorkerMonitor when receiving ActiveLoad events
        if let Err(e) = register_worker_load_metrics(&registry) {
            tracing::warn!("Failed to register worker load metrics: {}", e);
        }

        // Register worker timing metrics (last_ttft, last_itl per worker)
        // These are updated by ResponseMetricCollector when observing TTFT/ITL
        if let Err(e) = register_worker_timing_metrics(&registry) {
            tracing::warn!("Failed to register worker timing metrics: {}", e);
        }

        // Register router queue metrics (pending requests per worker_type)
        // These are updated by KvScheduler on enqueue/update/free
        if let Err(e) = register_router_queue_metrics(&registry) {
            tracing::warn!("Failed to register router queue metrics: {}", e);
        }

        if let Some(ref discovery) = config.drt_discovery {
            let instance_id = discovery.instance_id();
            if let Err(e) = RoutingOverheadMetrics::register(&registry, instance_id) {
                tracing::warn!("Failed to register routing overhead metrics: {}", e);
            }
        }

        if let Err(e) = ensure_request_plane_metrics_registered_prometheus(&registry) {
            tracing::warn!("Failed to register request-plane metrics: {}", e);
        }
        if let Err(e) = ensure_frontend_perf_metrics_registered_prometheus(&registry) {
            tracing::warn!("Failed to register frontend perf metrics: {}", e);
        }
        if let Err(e) = ensure_tokio_perf_metrics_registered_prometheus(&registry) {
            tracing::warn!("Failed to register tokio perf metrics: {}", e);
        }
        if let Err(e) = ensure_transport_metrics_registered_prometheus(&registry) {
            tracing::warn!("Failed to register transport metrics: {}", e);
        }
        if let Err(e) = register_lora_allocation_metrics(&registry) {
            tracing::warn!("Failed to register LoRA allocation metrics: {}", e);
        }

        let mut all_docs = Vec::new();

        // Shared on_response callback for both system and inference routes
        let on_response = |response: &Response<Body>, latency: Duration, _span: &tracing::Span| {
            let status = response.status();
            let latency_ms = latency.as_millis();
            if status.is_server_error() || status.is_client_error() {
                tracing::error!(status = %status.as_u16(), latency_ms = %latency_ms, "http response sent");
            } else {
                tracing::info!(status = %status.as_u16(), latency_ms = %latency_ms, "http response sent");
            }
        };

        // System routes (health, metrics, models) — debug-level spans
        let mut system_routes = vec![
            metrics::router(
                registry,
                var(HTTP_SVC_METRICS_PATH_ENV).ok(),
                config.drt_metrics,
            ),
            if anthropic_endpoints_enabled {
                super::anthropic::anthropic_models_router(
                    state.clone(),
                    var(HTTP_SVC_MODELS_PATH_ENV).ok(),
                )
            } else {
                super::openai::list_models_router(state.clone(), var(HTTP_SVC_MODELS_PATH_ENV).ok())
            },
            super::health::health_check_router(state.clone(), var(HTTP_SVC_HEALTH_PATH_ENV).ok()),
            super::health::live_check_router(state.clone(), var(HTTP_SVC_LIVE_PATH_ENV).ok()),
        ];
        if admin_api_enabled {
            system_routes.push(super::busy_threshold::busy_threshold_router(
                state.clone(),
                None,
            ));
        } else {
            tracing::info!(
                env = env_llm::DYN_DISABLE_FRONTEND_ADMIN_API,
                "frontend admin API disabled — busy_threshold routes not registered"
            );
        }
        let mut system_router = axum::Router::new();
        for (route_docs, route) in system_routes {
            system_router = system_router.merge(route);
            all_docs.extend(route_docs);
        }
        // Inference routes (completions, chat, embeddings, etc.) — info-level spans
        let endpoint_routes = HttpServiceConfigBuilder::get_endpoints_router(
            state.clone(),
            &config.request_template,
            anthropic_endpoints_enabled,
            generate_endpoints_enabled,
        );
        let mut inference_router = axum::Router::new();
        for (route_docs, route) in endpoint_routes {
            inference_router = inference_router.merge(route);
            all_docs.extend(route_docs);
        }
        inference_router = inference_router.layer(
            TraceLayer::new_for_http()
                .make_span_with(make_inference_request_span)
                .on_response(on_response),
        );
        inference_router = inference_router.layer(axum::middleware::from_fn_with_state(
            state.clone(),
            track_inflight_inference,
        ));

        // OpenAPI documentation routes (system)
        let (openapi_docs, openapi_route) =
            super::openapi_docs::openapi_router(all_docs.clone(), None);
        system_router = system_router.merge(openapi_route);
        all_docs.extend(openapi_docs);

        system_router = system_router.layer(
            TraceLayer::new_for_http()
                .make_span_with(make_system_request_span)
                .on_response(on_response),
        );

        let router = system_router.merge(inference_router);

        // Echo x-request-id from request to response headers for client correlation
        let router = router.layer(axum::middleware::from_fn(echo_request_id_header));

        let enable_rl_router = config.enable_rl || env_is_truthy("DYN_ENABLE_RL");
        let rl_router = if enable_rl_router {
            let Some(drt) = config.runtime.as_ref() else {
                return Err(anyhow::anyhow!(
                    "RL worker discovery was requested (DYN_ENABLE_RL=true \
                     or enable_rl) but HttpServiceConfig.runtime is not set."
                ));
            };
            let router = super::openai::rl_router(drt.clone())?;
            tracing::info!(
                rl_port = config.rl_port,
                "RL worker discovery enabled at /v1/rl/workers"
            );
            Some(
                router.layer(
                    TraceLayer::new_for_http()
                        .make_span_with(make_system_request_span)
                        .on_response(on_response),
                ),
            )
        } else {
            None
        };

        Ok(HttpService {
            state,
            router,
            port: config.port,
            host: config.host,
            enable_tls: config.enable_tls,
            tls_cert_path: config.tls_cert_path,
            tls_key_path: config.tls_key_path,
            route_docs: all_docs,
            rl_router,
            rl_port: config.rl_port,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }

    pub fn metrics_prefix(mut self, prefix: Option<String>) -> Self {
        self.metrics_config = Some(MetricsConfig::new(prefix));
        self
    }

    pub fn enable_anthropic_endpoints(mut self, enabled: bool) -> Self {
        self.frontend_api_config
            .get_or_insert_with(FrontendApiConfig::default)
            .anthropic_mut()
            .set_enabled(enabled);
        self
    }

    pub fn strip_anthropic_preamble(mut self, enabled: bool) -> Self {
        self.frontend_api_config
            .get_or_insert_with(FrontendApiConfig::default)
            .anthropic_mut()
            .set_strip_preamble(enabled);
        self
    }

    pub fn enable_streaming_tool_dispatch(mut self, enabled: bool) -> Self {
        self.frontend_api_config
            .get_or_insert_with(FrontendApiConfig::default)
            .streaming_dispatch_mut()
            .set_tool_dispatch(enabled);
        self
    }

    pub fn enable_streaming_reasoning_dispatch(mut self, enabled: bool) -> Self {
        self.frontend_api_config
            .get_or_insert_with(FrontendApiConfig::default)
            .streaming_dispatch_mut()
            .set_reasoning_dispatch(enabled);
        self
    }

    fn get_endpoints_router(
        state: Arc<State>,
        request_template: &Option<RequestTemplate>,
        enable_anthropic_endpoints: bool,
        enable_generate_endpoints: bool,
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
        let (images_docs, images_route) = super::openai::images_router(state.clone(), None);
        let (videos_docs, videos_route) = super::openai::videos_router(state.clone(), None);
        let (audios_docs, audios_route) = super::openai::audios_router(state.clone(), None);
        let (realtime_docs, realtime_route) = super::realtime::realtime_router(state.clone(), None);
        let (responses_docs, responses_route) = super::openai::responses_router(
            state.clone(),
            request_template.clone(),
            var(HTTP_SVC_RESPONSES_PATH_ENV).ok(),
        );
        let mut endpoint_routes = HashMap::new();
        endpoint_routes.insert(EndpointType::Chat, (chat_docs, chat_route));
        endpoint_routes.insert(EndpointType::Completion, (cmpl_docs, cmpl_route));
        endpoint_routes.insert(EndpointType::Embedding, (embed_docs, embed_route));
        endpoint_routes.insert(EndpointType::Images, (images_docs, images_route));
        endpoint_routes.insert(EndpointType::Videos, (videos_docs, videos_route));
        endpoint_routes.insert(EndpointType::Audios, (audios_docs, audios_route));
        endpoint_routes.insert(EndpointType::Realtime, (realtime_docs, realtime_route));
        endpoint_routes.insert(EndpointType::Responses, (responses_docs, responses_route));

        if enable_anthropic_endpoints {
            tracing::warn!("Anthropic Messages API (/v1/messages) is experimental.");
            let (anthropic_docs, anthropic_route) = super::anthropic::anthropic_messages_router(
                state.clone(),
                request_template.clone(),
                var(HTTP_SVC_ANTHROPIC_PATH_ENV).ok(),
            );
            endpoint_routes.insert(
                EndpointType::AnthropicMessages,
                (anthropic_docs, anthropic_route),
            );
        }

        if enable_generate_endpoints {
            tracing::warn!(
                "Generate API (/inference/v1/generate) is experimental and not implemented yet."
            );
            let (generate_docs, generate_route) = super::generate::generate_router(
                state.clone(),
                var(HTTP_SVC_GENERATE_PATH_ENV).ok(),
            );
            endpoint_routes.insert(EndpointType::Generate, (generate_docs, generate_route));
        }

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
                            Err(axum::http::StatusCode::NOT_FOUND)
                        }
                    }
                },
            ));
            routes.push((docs, route));
        }
        routes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    async fn wait_for_service_stage(state: &State, expected: ServiceStage) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        loop {
            if state.service_stage() == expected {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "service did not enter {expected} before timeout; current stage is {}",
                state.service_stage()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_liveness_endpoint_stays_live_while_draining() {
        temp_env::async_with_vars(
            [(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("1"))],
            async {
                let cancel_token = Arc::new(CancellationToken::new());
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("failed to bind ephemeral port");
                let port = listener.local_addr().unwrap().port();
                let service = HttpService::builder().port(port).build().unwrap();
                let state = service.state_clone();
                let inflight = state.acquire_inflight();

                let service_token = cancel_token.clone();
                let handle = tokio::spawn(async move {
                    service
                        .run_with_listener((*service_token).clone(), listener)
                        .await
                        .unwrap();
                });

                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                cancel_token.cancel();
                wait_for_service_stage(&state, ServiceStage::Draining).await;

                let resp = reqwest::Client::new()
                    .get(format!("http://localhost:{}/live", port))
                    .send()
                    .await
                    .expect("Request failed");

                assert_eq!(resp.status(), reqwest::StatusCode::OK);

                drop(inflight);
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn test_health_endpoint_reflects_draining_before_cancellation() {
        temp_env::async_with_vars(
            [(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("1"))],
            async {
                let cancel_token = Arc::new(CancellationToken::new());
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("failed to bind ephemeral port");
                let port = listener.local_addr().unwrap().port();
                let service = HttpService::builder().port(port).build().unwrap();
                let state = service.state_clone();
                let inflight = state.acquire_inflight();

                let service_token = cancel_token.clone();
                let handle = tokio::spawn(async move {
                    service
                        .run_with_listener((*service_token).clone(), listener)
                        .await
                        .unwrap();
                });

                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                cancel_token.cancel();
                wait_for_service_stage(&state, ServiceStage::Draining).await;

                assert_eq!(state.service_stage(), ServiceStage::Draining);

                let client = reqwest::Client::new();
                let health = client
                    .get(format!("http://localhost:{}/health", port))
                    .send()
                    .await
                    .expect("health request failed");
                assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

                let live = client
                    .get(format!("http://localhost:{}/live", port))
                    .send()
                    .await
                    .expect("live request failed");
                assert_eq!(live.status(), reqwest::StatusCode::OK);

                drop(inflight);
                handle.abort();
            },
        )
        .await;
    }

    /// `enable_admin_api=false` ⇒ `GET /busy_threshold` is not registered and
    /// returns 404, not 503 or 405. Inference is unaffected (covered by other
    /// tests).
    #[tokio::test]
    async fn test_admin_api_disabled_404s_busy_threshold() {
        let cancel_token = Arc::new(CancellationToken::new());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = HttpService::builder()
            .port(port)
            .enable_admin_api(false)
            .build()
            .unwrap();

        let service_token = cancel_token.clone();
        let handle = tokio::spawn(async move {
            service
                .run_with_listener((*service_token).clone(), listener)
                .await
                .unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let resp = reqwest::Client::new()
            .get(format!("http://localhost:{}/busy_threshold", port))
            .send()
            .await
            .expect("request failed");
        assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

        // And /live still works (sanity: only the admin surface is gated).
        let live = reqwest::Client::new()
            .get(format!("http://localhost:{}/live", port))
            .send()
            .await
            .expect("request failed");
        assert_eq!(live.status(), reqwest::StatusCode::OK);

        cancel_token.cancel();
        handle.abort();
    }

    /// `enable_nvext` is wired from the builder onto `State.nvext_enabled` and
    /// exposed via the accessor used by the openai handlers.
    #[test]
    #[serial_test::serial]
    fn test_enable_nvext_propagates_through_builder_to_state() {
        use dynamo_runtime::config::environment_names::llm::DYN_DISABLE_FRONTEND_NVEXT;

        // `build()` ANDs the builder flag with the env var, so this test must
        // pin the env to unset. Going through `temp_env` also serializes it
        // against `test_dyn_disable_frontend_nvext_env_var_mirror`, which mutates
        // the same process-global var in parallel.
        temp_env::with_var_unset(DYN_DISABLE_FRONTEND_NVEXT, || {
            let on = HttpService::builder().enable_nvext(true).build().unwrap();
            assert!(on.state.nvext_enabled());

            let off = HttpService::builder().enable_nvext(false).build().unwrap();
            assert!(!off.state.nvext_enabled());

            let default = HttpService::builder().build().unwrap();
            assert!(
                default.state.nvext_enabled(),
                "default should preserve current behavior (nvext on)"
            );
        });
    }

    /// `DYN_DISABLE_FRONTEND_NVEXT` is the env-var mirror of the builder
    /// flag. Unset -> builder default wins (on). Falsey strings -> on.
    /// Truthy strings (`1` / `true` / `yes` / `on`, case-insensitive) ->
    /// off, regardless of what the builder asked for.
    #[test]
    #[serial_test::serial]
    fn test_dyn_disable_frontend_nvext_env_var_mirror() {
        use dynamo_runtime::config::environment_names::llm::DYN_DISABLE_FRONTEND_NVEXT;

        // Unset -> builder default (true) wins.
        temp_env::with_var_unset(DYN_DISABLE_FRONTEND_NVEXT, || {
            let svc = HttpService::builder().build().unwrap();
            assert!(
                svc.state.nvext_enabled(),
                "unset env + default builder = on"
            );
        });

        // Explicit falsey -> still on (disable not requested).
        temp_env::with_var(DYN_DISABLE_FRONTEND_NVEXT, Some("false"), || {
            let svc = HttpService::builder().build().unwrap();
            assert!(
                svc.state.nvext_enabled(),
                "disable=false + default builder = on"
            );
        });

        // Explicit truthy -> off, even though the builder default is on.
        for truthy in ["true", "1", "yes", "on", "TRUE"] {
            temp_env::with_var(DYN_DISABLE_FRONTEND_NVEXT, Some(truthy), || {
                let svc = HttpService::builder().build().unwrap();
                assert!(
                    !svc.state.nvext_enabled(),
                    "disable={truthy:?} should override builder default to off"
                );
            });
        }

        // Builder=false short-circuits regardless of env.
        temp_env::with_var_unset(DYN_DISABLE_FRONTEND_NVEXT, || {
            let svc = HttpService::builder().enable_nvext(false).build().unwrap();
            assert!(
                !svc.state.nvext_enabled(),
                "builder=false wins even if disable is unset"
            );
        });
    }
}
