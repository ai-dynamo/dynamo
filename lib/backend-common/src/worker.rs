// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Worker` — runtime lifecycle driver for an [`LLMEngine`].
//!
//! Creates the `DistributedRuntime`, starts the engine, registers the
//! model, serves the endpoint, and runs cleanup on shutdown. Non-generic
//! over the engine type so a PyO3-wrapped engine can feed in through the
//! same `Arc<dyn LLMEngine>` path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::local_model::LocalModel;
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_llm::local_model::runtime_config::{
    DisaggregatedEndpoint, ModelRuntimeConfig, StructuralTagMode, StructuralTagSchemaMode,
    StructuralTagScope,
};
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::preprocessor::media::{MediaDecoder, MediaFetcher};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::engine_routes::EngineRouteCallback;
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{DistributedRuntime, Runtime};
use tokio_util::sync::CancellationToken;

use crate::adapter::{EngineAdapter, RawEngineAdapter};
use crate::disagg::DisaggregationMode;
use crate::engine::{
    EngineConfig, KvEventSource, LLMEngine, MetricsBindings, MetricsCtx, RawEngine,
};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::publisher::{PublisherHandles, setup_publishers};

/// Default grace-period in seconds between discovery unregister and engine drain.
/// Mirrors the Python `_DEFAULT_GRACE_PERIOD_SECS` constant.
const DEFAULT_GRACE_PERIOD_SECS: f64 = 5.0;

/// Environment variable name for overriding the grace-period.
/// Shared with the Python helper so a single env var controls both.
const GRACE_PERIOD_ENV: &str = "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS";

/// Default drain budget: max time spent polling `is_quiescent` before cleanup.
/// Capped at `graceful_shutdown_timeout - CLEANUP_RESERVE_S`.
const DEFAULT_DRAIN_TIMEOUT_S: f64 = 30.0;
const DRAIN_TIMEOUT_ENV: &str = "DYN_PREFILL_DRAIN_TIMEOUT_S";
/// Interval between `engine.is_quiescent()` polls during drain.
const DRAIN_POLL_INTERVAL_S: f64 = 0.5;
/// Cadence at which the drain loop emits a progress log.
const DRAIN_HEARTBEAT_INTERVAL_S: f64 = 5.0;
/// Budget reserved for `cleanup()` so the drain loop can't consume the whole
/// graceful-shutdown deadline and trip the hard-exit that skips cleanup.
const CLEANUP_RESERVE_S: f64 = 5.0;

/// Operator override for the health-check canary, mirrors the Python helper
/// in `lib/bindings/python/src/dynamo/health_check.py`.
const HEALTH_CHECK_PAYLOAD_ENV: &str = "DYN_HEALTH_CHECK_PAYLOAD";

/// Runtime / transport configuration applied to the process before the
/// distributed runtime is constructed.
///
/// `dynamo-runtime` reads these from environment variables in
/// [`DistributedConfig::from_settings`]. We mirror that by setting them
/// here before [`Runtime::from_settings`] runs, so a programmatic caller
/// can override per-process values without poking `std::env::set_var`
/// from user code.
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    /// Discovery backend selector — e.g. `"etcd"`, `"kubernetes"`, `"file"`,
    /// `"mem"`. Maps to `DYN_DISCOVERY_BACKEND`.
    pub discovery_backend: Option<String>,
    /// Request-plane transport — e.g. `"tcp"`, `"nats"`. Maps to `DYN_REQUEST_PLANE`.
    pub request_plane: Option<String>,
    /// Event-plane transport — `"nats"` or `"zmq"`. When `None` the runtime
    /// derives a default from the discovery backend. Maps to `DYN_EVENT_PLANE`.
    pub event_plane: Option<String>,
}

impl RuntimeConfig {
    /// `true` if any field is set. Used by the PyO3 binding to decide
    /// whether to warn that overrides will be dropped when reusing a
    /// runtime constructed by another caller.
    pub fn has_overrides(&self) -> bool {
        self.discovery_backend.is_some()
            || self.request_plane.is_some()
            || self.event_plane.is_some()
    }

    /// Apply each set field to the corresponding environment variable.
    /// Unset fields leave the existing environment value untouched.
    pub fn apply_to_env(&self) {
        // SAFETY: set_var is unsafe in edition 2024 because it can race with
        // other threads reading the environment. We call it before any
        // runtime threads spawn, matching the convention used by
        // `dynamo-runtime` itself in DistributedConfig::from_settings.
        unsafe {
            self.apply_with(|key, value| std::env::set_var(key, value));
        }
    }

    fn apply_with(&self, mut set: impl FnMut(&str, &str)) {
        if let Some(ref value) = self.discovery_backend {
            set("DYN_DISCOVERY_BACKEND", value);
        }
        if let Some(ref value) = self.request_plane {
            set("DYN_REQUEST_PLANE", value);
        }
        if let Some(ref value) = self.event_plane {
            set("DYN_EVENT_PLANE", value);
        }
    }
}

/// Per-worker runtime configuration.
#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Dynamo namespace for discovery routing.
    pub namespace: String,
    /// Component name within the namespace.
    pub component: String,
    /// Endpoint name exposed by this worker (e.g. `"generate"`).
    pub endpoint: String,
    /// HF repo name or local model path. Empty means name-only registration
    /// (no tokenizer / chat-template on the card).
    pub model_name: String,
    /// Public-facing model name (operator CLI override). When unset, the
    /// served name falls back to `EngineConfig.served_model_name`, then to
    /// `EngineConfig.model`.
    pub served_model_name: Option<String>,
    /// Whether the engine consumes tokens (`Tokens`) or raw text (`Text`).
    pub model_input: ModelInput,
    /// Comma-separated list, e.g. `"chat,completions"`.
    /// Accepted values: `chat`, `completions`, `embedding`/`embeddings`,
    /// `tensor`, `prefill` (see `parse_endpoint_types`).
    pub endpoint_types: String,
    /// Optional path to a custom Jinja chat template. When `None`, the
    /// template shipped with `model_name` is used.
    pub custom_jinja_template: Option<PathBuf>,
    /// Optional tool-call parser name written to model runtime metadata.
    pub tool_call_parser: Option<String>,
    /// Optional reasoning parser name written to model runtime metadata.
    pub reasoning_parser: Option<String>,
    /// Whether templates should omit tools when `tool_choice` is `none`.
    pub exclude_tools_when_tool_choice_none: bool,
    /// Whether this worker should keep an in-process KV indexer.
    pub enable_local_indexer: bool,
    /// Kill switch for KV-aware-routing publishers. When `false`, skip
    /// `engine.kv_event_sources()` and `SnapshotPublisher` setup.
    pub enable_kv_routing: bool,
    /// Per-endpoint Prometheus metric labels appended to every metric.
    /// Common labels: `("model", "<served-name>")`.
    pub metrics_labels: Vec<(String, String)>,
    /// Disaggregation role for this worker.
    ///
    /// `Aggregated` (default) registers the model with the parsed
    /// `endpoint_types`. `Prefill` registers with the legacy `ModelType::Prefill`
    /// marker bit (no OpenAI surface — dual-emitted for cross-version compat)
    /// and `WorkerType::Prefill`, so the frontend's prefill router targets it
    /// via `worker_type`. `Decode` keeps `endpoint_types` but force-disables the
    /// local KV indexer because decode workers do not host the indexer
    /// endpoint. `Encode` registers as `WorkerType::Encode` with topology needs
    /// `[[Prefill, Decode], [Aggregated]]`; it also force-disables the local KV
    /// indexer.
    pub disaggregation_mode: DisaggregationMode,
    /// Operator override. `Worker` resolves precedence: this field >
    /// `DYN_HEALTH_CHECK_PAYLOAD` env > `engine.health_check_payload()`.
    /// Python sets this via `--health-check-payload` / env; Rust-only
    /// engines leave it `None` and let `Worker` read the env directly.
    pub health_check_payload: Option<serde_json::Value>,
    /// Structural tag guided decoding mode.
    pub structural_tag_mode: StructuralTagMode,
    /// Structural tag activation scope.
    pub structural_tag_scope: StructuralTagScope,
    /// Structural tag schema strictness.
    pub structural_tag_schema: StructuralTagSchemaMode,
    /// Runtime / transport overrides applied via env vars before the
    /// `DistributedRuntime` is constructed.
    pub runtime: RuntimeConfig,
    /// When `true`, this worker declares an upstream `Encode` dependency in
    /// its topology `needs`. Meaningful only for `Prefill` and `Aggregated`
    /// roles -- setting it on `Decode` or `Encode` is rejected at
    /// `Worker::run` validation time with `BackendError::InvalidArgument`.
    pub route_to_encoder: bool,
    /// Optional frontend media decoding and fetch policy advertised on the
    /// model deployment card.
    pub media_decoder: Option<MediaDecoder>,
    pub media_fetcher: Option<MediaFetcher>,
    /// Deployment-level default thinking mode written to runtime metadata.
    pub default_thinking_mode: Option<String>,
}

impl WorkerConfig {
    /// Effective `enable_local_indexer`, accounting for disaggregation
    /// mode. Decode and Encode workers force this off because they don't
    /// host the in-process KV indexer endpoint and must not advertise it.
    pub(crate) fn effective_enable_local_indexer(&self) -> bool {
        self.enable_local_indexer
            && !self.disaggregation_mode.is_decode()
            && !self.disaggregation_mode.is_encode()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            model_name: String::new(),
            served_model_name: None,
            model_input: ModelInput::Tokens,
            endpoint_types: "chat,completions".to_string(),
            custom_jinja_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: true,
            enable_local_indexer: true,
            enable_kv_routing: true,
            metrics_labels: Vec::new(),
            disaggregation_mode: DisaggregationMode::Aggregated,
            health_check_payload: None,
            structural_tag_mode: StructuralTagMode::Off,
            structural_tag_scope: StructuralTagScope::Auto,
            structural_tag_schema: StructuralTagSchemaMode::Auto,
            runtime: RuntimeConfig::default(),
            route_to_encoder: false,
            media_decoder: None,
            media_fetcher: None,
            default_thinking_mode: None,
        }
    }
}

/// Lifecycle state for [`Worker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LifecycleState {
    /// `start_engine` has not been called (or shutdown arrived first and
    /// flipped us straight to `Stopped`).
    Init,
    /// `engine.start()` returned successfully; `engine.cleanup()` is owed.
    Running,
    /// `engine.start()` raised. The engine may have allocated partial
    /// state (inner LLM, sockets, background tasks) before failing, so
    /// `engine.cleanup()` is still owed exactly once.
    StartFailed,
    /// Cleanup done. `engine.cleanup()` will not be called again.
    Stopped,
}

/// The engine a [`Worker`] drives, tagged by request modality. Both variants
/// share the lifecycle (driven via the forwarders below); they differ only in
/// the serve-loop adapter: `Llm` → token pipeline ([`EngineAdapter`]), `Raw` →
/// JSON passthrough ([`RawEngineAdapter`]) for media. A new media modality is
/// a new `Raw` engine, not a new variant.
#[derive(Clone)]
pub(crate) enum EngineKind {
    Llm(Arc<dyn LLMEngine>),
    Raw(Arc<dyn RawEngine>),
}

impl EngineKind {
    async fn start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.start(worker_id).await,
            EngineKind::Raw(e) => e.start(worker_id).await,
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        match self {
            EngineKind::Llm(e) => e.cleanup().await,
            EngineKind::Raw(e) => e.cleanup().await,
        }
    }

    /// See [`LLMEngine::is_quiescent`].
    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.is_quiescent().await,
            EngineKind::Raw(e) => e.is_quiescent().await,
        }
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.setup_metrics(ctx).await,
            EngineKind::Raw(e) => e.setup_metrics(ctx).await,
        }
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.kv_event_sources().await,
            // Raw media engines have no block-structured KV cache to route on.
            EngineKind::Raw(_) => Ok(Vec::new()),
        }
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.health_check_payload().await,
            EngineKind::Raw(e) => e.health_check_payload().await,
        }
    }

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.supported_controls().await,
            // Raw media engines advertise no semantic engine controls.
            EngineKind::Raw(_) => Ok(Vec::new()),
        }
    }

    async fn engine_control(
        &self,
        control: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.engine_control(control, body).await,
            EngineKind::Raw(_) => Ok(serde_json::json!({
                "status": "error",
                "message": format!("unsupported engine control: {control}"),
            })),
        }
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.supported_updates().await,
            // Raw media engines advertise no semantic engine updates.
            EngineKind::Raw(_) => Ok(Vec::new()),
        }
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        match self {
            EngineKind::Llm(e) => e.engine_update(update, body).await,
            EngineKind::Raw(_) => Ok(serde_json::json!({
                "status": "error",
                "message": format!("unsupported engine update: {update}"),
            })),
        }
    }

    async fn on_endpoint_ready(
        &self,
        endpoint: dynamo_runtime::component::Endpoint,
    ) -> Result<(), DynamoError> {
        match self {
            EngineKind::Llm(e) => e.on_endpoint_ready(endpoint).await,
            // Raw media engines publish no discovery records of their own.
            EngineKind::Raw(_) => Ok(()),
        }
    }

    /// Raw media engines (image/video/audio) register name-only — the engine
    /// loads the model itself and the model has no LLM artifacts (tokenizer /
    /// chat template / config.json) for Dynamo to fetch.
    fn is_raw(&self) -> bool {
        matches!(self, EngineKind::Raw(_))
    }
}

/// Runtime host for an engine (an [`LLMEngine`] or a [`RawEngine`]).
///
/// `run()` creates the distributed runtime, calls `engine.start()`,
/// registers the model, serves the endpoint, and calls
/// `engine.cleanup()` on shutdown (guaranteed once `start()` succeeded).
pub struct Worker {
    engine: EngineKind,
    config: WorkerConfig,
    state: LifecycleState,
    /// KV-aware-routing publisher handles. Drained in `cleanup_once` while NATS is alive.
    publishers: Option<PublisherHandles>,
    /// Framework-owned lifecycle gauges. Set in `setup_publishing` after
    /// `engine.start()` succeeds; observed in `cleanup_once` and the drain
    /// step. Always present once `start()` returns Ok, independent of
    /// whether the engine returned a component publisher.
    lifecycle: Option<crate::metrics::LifecycleGauges>,
}

impl Worker {
    /// Build a `Worker` for a token-pipeline [`LLMEngine`].
    pub fn new(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> Self {
        Self::with_engine(EngineKind::Llm(engine), config)
    }

    /// Build a `Worker` for a raw media-pipeline [`RawEngine`]
    /// (image/video/audio generation).
    pub fn new_raw(engine: Arc<dyn RawEngine>, config: WorkerConfig) -> Self {
        Self::with_engine(EngineKind::Raw(engine), config)
    }

    fn with_engine(engine: EngineKind, config: WorkerConfig) -> Self {
        Self {
            engine,
            config,
            state: LifecycleState::Init,
            publishers: None,
            lifecycle: None,
        }
    }

    /// Lifecycle driver. Takes owned `self` — `Worker` is single-shot and
    /// cannot be reused after `run()` returns.
    ///
    /// Shutdown sequence (mirrors `graceful_shutdown_with_discovery` in
    /// `components/src/dynamo/common/utils/graceful_shutdown.py`):
    ///   1. `endpoint.unregister_endpoint_instance()` — router stops routing.
    ///   2. Sleep `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` (default 5s) to
    ///      let in-flight router decisions complete.
    ///   3. Poll `engine.is_quiescent()` until it returns true or the drain
    ///      budget (`DYN_PREFILL_DRAIN_TIMEOUT_S`, default 30s) expires.
    ///   4. `engine.cleanup()` — release engine resources while NATS / etcd
    ///      are still reachable.
    ///   5. Return — caller (`run.rs`) drives `runtime.shutdown()` for
    ///      request-plane drain and transport teardown.
    ///
    /// A SIGTERM/SIGINT listener is installed at the top of `run` and
    /// shared via a [`CancellationToken`]:
    ///   * Pre-start signal (during `DistributedRuntime` construction):
    ///     the post-DRT cancellation check returns `Ok(())` cleanly and
    ///     `engine.start()` is never called.
    ///   * Mid-start signal: `engine.start()` is allowed to complete (we
    ///     never cancel a partially-initialized engine mid-flight); the
    ///     post-start cancellation check then runs the orchestrator
    ///     directly without entering the serve loop.
    ///   * Mid-serve signal: the serve loop's [`tokio::select`] picks up
    ///     the same token and runs the orchestrator.
    ///
    /// `engine.cleanup()` is guaranteed to run exactly once if
    /// `engine.start()` succeeded, regardless of which path led to shutdown.
    pub async fn run(mut self, runtime: Runtime) -> Result<(), DynamoError> {
        // Validate the worker config up front so misconfiguration surfaces
        // before any signal handlers, tokio tasks, or runtime construction.
        // The same validation is also reachable via `run_inner`, but doing
        // it here means a user who passes an unsupported `model_input`
        // doesn't pay the cost of installing signal handlers and spawning
        // a listener task just to get an InvalidArgument error.
        validate_model_input(self.config.model_input, &self.engine)?;
        validate_route_to_encoder(&self.config)?;

        // Install the OS signal handlers synchronously, before spawning
        // anything, so a SIGTERM delivered between this point and the
        // task's first poll is captured by the kernel-side handler rather
        // than the OS default (which would terminate the process abruptly).
        // `Signal::recv` then drives the shared cancellation token.
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("install SIGTERM handler: {e}"),
                )
            })?;
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("install SIGINT handler: {e}"),
                )
            })?;

        // Single shared shutdown signal observed across all phases. The
        // background task only flips the token; lifecycle transitions stay
        // on this owned Worker instance.
        let shutdown_token = CancellationToken::new();
        let signal_token = shutdown_token.clone();
        let signal_handle = tokio::spawn(async move {
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("SIGTERM received"),
                _ = sigint.recv() => tracing::info!("SIGINT received"),
            }
            signal_token.cancel();
        });

        // Mirror `dynamo_runtime::Worker::execute`'s shutdown deadline:
        // once a signal arrives, the orchestrator + cleanup must finish
        // within `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` seconds (plus the
        // grace-period sleep, which is a fixed wait rather than a hang
        // risk), otherwise we exit(911). Healthy long-running workers
        // never hit this — the timer only starts after `shutdown_token`
        // is cancelled.
        let outcome = {
            let inner_fut = self.run_inner(runtime, &shutdown_token);
            tokio::pin!(inner_fut);

            tokio::select! {
                result = &mut inner_fut => result,
                _ = shutdown_token.cancelled() => {
                    let timeout = graceful_shutdown_timeout();
                    let grace = grace_period_secs();
                    let deadline = shutdown_deadline(timeout, grace);
                    tracing::debug!(
                        "graceful shutdown started; deadline {}s ({}s timeout + {:.2}s grace)",
                        deadline.as_secs(),
                        timeout.as_secs(),
                        grace,
                    );
                    match tokio::time::timeout(deadline, &mut inner_fut).await {
                        Ok(result) => result,
                        Err(_) => {
                            tracing::error!(
                                "Graceful shutdown exceeded {}s; force-exiting with code 911. \
                                 Set DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT to override.",
                                deadline.as_secs()
                            );
                            std::process::exit(911);
                        }
                    }
                }
            }
        };

        signal_handle.abort();
        let _ = signal_handle.await;

        // Final safety net: guarantee engine.cleanup() runs if start()
        // succeeded. No-op if cleanup already ran via the orchestrator.
        self.cleanup_once().await;

        outcome
    }

    async fn run_inner(
        &mut self,
        runtime: Runtime,
        shutdown: &CancellationToken,
    ) -> Result<(), DynamoError> {
        // model_input was already validated at the top of `run`; re-checking
        // here would double-error on misconfig.
        let drt = DistributedRuntime::from_settings(runtime)
            .await
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("distributed runtime: {e}"),
                )
            })?;
        tracing::debug!("distributed runtime connected");

        let component = drt
            .namespace(&self.config.namespace)
            .and_then(|ns| ns.component(&self.config.component))
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("component: {e}"),
                )
            })?;
        let endpoint = component.endpoint(&self.config.endpoint);
        tracing::debug!(
            namespace = %self.config.namespace,
            component = %self.config.component,
            endpoint = %self.config.endpoint,
            "component and endpoint resolved"
        );

        // Shutdown arrived during DRT construction; engine never started,
        // nothing to clean up.
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown signal observed before engine.start(); exiting cleanly");
            return Ok(());
        }

        // Pull the worker's unique runtime ID from the DRT before handing it
        // to the engine. Backed by `discovery_client.instance_id()` so it is
        // unique-per-replica by construction; engines see only an opaque
        // `worker_id`.
        let worker_id = drt.connection_id();
        let engine_start = std::time::Instant::now();
        let engine_config = self.start_engine(worker_id).await?;
        let model_load_time_seconds = engine_start.elapsed().as_secs_f64();
        tracing::debug!(
            model = %engine_config.model,
            worker_id,
            model_load_time_seconds,
            "engine.start() complete"
        );

        // Engine builds its EngineMetrics once. `setup_metrics` is the
        // single hook for both foreign-registry expfmt callbacks (side-
        // effect on engine_metrics) and the structured component publisher
        // (returned in MetricsBindings).
        let engine_metrics =
            crate::metrics::EngineMetrics::with_engine_config(endpoint.clone(), &engine_config);

        // Framework-owned lifecycle gauges (cleanup_time, drain_time,
        // model_load_time) — always emitted, regardless of engine opt-in.
        let lifecycle =
            crate::metrics::LifecycleGauges::new(&engine_metrics, model_load_time_seconds)?;

        self.setup_publishing(
            &component,
            &engine_config,
            &engine_metrics,
            model_load_time_seconds,
            lifecycle,
        )
        .await?;

        // Mid-start signal: engine.start() ran to completion but a signal
        // arrived during it. Skip the serve loop and run the orchestrator
        // directly so `engine.cleanup()` still runs while the runtime is
        // alive.
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown signal observed during engine.start(); running orchestrator");
            self.orchestrator_steps(&endpoint).await;
            return Ok(());
        }

        self.serve_with_orchestrator(&engine_config, endpoint, shutdown.clone())
            .await
    }

    /// Build KV-event publishers and the `SnapshotPublisher` from the
    /// engine's declarations. KV events flow on the engine's own threads
    /// (via Push or ZMQ); snapshot writes flow through the publisher
    /// inline (no polling, no GIL on the framework side). No-op if
    /// `enable_kv_routing` is off, the engine returned no sources +
    /// no dp_ranks, or `engine_config.kv_cache_block_size` is unset for
    /// KV events.
    async fn setup_publishing(
        &mut self,
        component: &dynamo_runtime::component::Component,
        engine_config: &EngineConfig,
        engine_metrics: &crate::metrics::EngineMetrics,
        model_load_time_seconds: f64,
        lifecycle: crate::metrics::LifecycleGauges,
    ) -> Result<(), DynamoError> {
        let ctx = crate::engine::MetricsCtx {
            model: &engine_config.model,
            component: &self.config.component,
            model_load_time_seconds,
            metrics: engine_metrics,
        };
        let bindings = self.engine.setup_metrics(ctx).await?;

        if !self.config.enable_kv_routing {
            tracing::debug!("enable_kv_routing=false; skipping kv/snapshot publishers");
            self.lifecycle = Some(lifecycle);
            return Ok(());
        }
        let kv_sources = self.engine.kv_event_sources().await?;
        if kv_sources.is_empty() && bindings.dp_ranks.is_empty() {
            tracing::debug!("engine returned no KV sources / dp_ranks; KV-aware routing disabled");
            self.lifecycle = Some(lifecycle);
            return Ok(());
        }
        let enable_local_indexer = self.config.effective_enable_local_indexer();
        // None for raw engines (no block-structured KV cache).
        let kv_cache_block_size = engine_config
            .llm
            .as_ref()
            .and_then(|l| l.kv_cache_block_size);
        tracing::debug!(
            kv_sources = kv_sources.len(),
            snapshot_dp_ranks = bindings.dp_ranks.len(),
            enable_local_indexer,
            kv_cache_block_size = ?kv_cache_block_size,
            "Starting KV-aware-routing publishers"
        );
        let handles = setup_publishers(
            component,
            engine_metrics,
            kv_sources,
            bindings.dp_ranks,
            bindings.on_publisher_ready,
            kv_cache_block_size,
            enable_local_indexer,
        )
        .await?;
        self.publishers = Some(handles);
        self.lifecycle = Some(lifecycle);
        Ok(())
    }

    /// Register advertised engine controls on the runtime system server.
    async fn register_engine_controls(
        &self,
        endpoint: &dynamo_runtime::component::Endpoint,
    ) -> Result<(), DynamoError> {
        let controls = self.engine.supported_controls().await?;
        if controls.is_empty() {
            tracing::debug!("engine returned no management controls");
            return Ok(());
        }

        let registry = endpoint.drt().engine_routes();
        let control_count = controls.len();
        // Serialize discovery-mutating controls so a concurrent resume cannot
        // re-register the endpoint between a pause control's unregister and
        // its engine-state mutation (and vice versa).
        let control_lock = Arc::new(tokio::sync::Mutex::new(()));
        for control_name in controls {
            let callback = engine_control_callback(control_name.clone(), self.engine.clone());
            let callback = wrap_engine_control_callback(
                control_name.clone(),
                callback,
                endpoint.clone(),
                control_lock.clone(),
            );
            // Namespace control routes under `/engine/control/<name>` so they
            // share the `/engine/{*path}` route without colliding with updates.
            registry.register(&format!("control/{control_name}"), callback);
        }
        tracing::info!(control_count, "registered engine management controls");
        Ok(())
    }

    /// Register advertised engine updates on the runtime system server.
    ///
    /// Updates are a sibling surface to controls for operations that mutate
    /// engine-managed assets (e.g. vLLM dynamic LoRA). They register under
    /// `/engine/update/<name>` and, unlike controls, never toggle discovery
    /// registration — so no quiesce/resume policy wrapper or serialization lock.
    async fn register_engine_updates(
        &self,
        endpoint: &dynamo_runtime::component::Endpoint,
    ) -> Result<(), DynamoError> {
        let updates = self.engine.supported_updates().await?;
        if updates.is_empty() {
            tracing::debug!("engine returned no management updates");
            return Ok(());
        }

        let registry = endpoint.drt().engine_routes();
        let update_count = updates.len();
        for update_name in updates {
            let callback = engine_update_callback(update_name.clone(), self.engine.clone());
            // Namespace update routes under `/engine/update/<name>`.
            registry.register(&format!("update/{update_name}"), callback);
        }
        tracing::info!(update_count, "registered engine management updates");
        Ok(())
    }

    /// Full graceful-shutdown orchestrator: discovery unregister →
    /// grace period → engine drain → cleanup. Shared by every shutdown path —
    /// pre-serve (mid-start signal) and the serve loop's signal arm.
    async fn orchestrator_steps(&mut self, endpoint: &dynamo_runtime::component::Endpoint) {
        if let Err(e) = endpoint.unregister_endpoint_instance().await {
            tracing::warn!(error = %e, "discovery unregister failed");
        } else {
            tracing::info!("Endpoint unregistered from discovery");
        }
        self.run_engine_shutdown_steps().await;
    }

    /// Start the engine exactly once. `Worker::run` consumes `self`, so all
    /// lifecycle transitions are single-threaded and do not need a mutex.
    async fn start_engine(&mut self, worker_id: u64) -> Result<EngineConfig, DynamoError> {
        // `start_engine` is called once from `run_inner`, which consumes
        // `self`. Hitting any other state is a programmer error worth
        // panicking over in release as well as debug builds.
        assert_eq!(
            self.state,
            LifecycleState::Init,
            "start_engine called in unexpected state {:?}",
            self.state
        );
        match self.engine.start(worker_id).await {
            Ok(cfg) => {
                self.state = LifecycleState::Running;
                Ok(cfg)
            }
            Err(e) => {
                // Engine.cleanup() still owed: start() may have built up
                // partial state (inner LLM, sockets, background tasks)
                // before raising, and the contract requires cleanup to be
                // safe against that. cleanup_once() picks up StartFailed.
                self.state = LifecycleState::StartFailed;
                Err(e)
            }
        }
    }

    /// Idempotent cleanup.
    async fn cleanup_once(&mut self) {
        match self.state {
            LifecycleState::Init | LifecycleState::Stopped => {
                // Pre-start shutdown, or cleanup already ran. Nothing
                // engine-side to do — `engine.start()` either never ran
                // or its allocations have already been released.
                self.state = LifecycleState::Stopped;
                return;
            }
            LifecycleState::Running | LifecycleState::StartFailed => {}
        }
        let cleanup_start = std::time::Instant::now();
        match self.engine.cleanup().await {
            Ok(()) => tracing::info!("Engine cleanup complete"),
            Err(e) => tracing::error!(error = %e, "engine cleanup failed"),
        }
        let cleanup_elapsed = cleanup_start.elapsed().as_secs_f64();
        // Record cleanup latency on dynamo_component_cleanup_time_seconds.
        // The gauge is operator-useful when scraped in the brief window
        // between cleanup-complete and pod-terminate.
        if let Some(lifecycle) = self.lifecycle.as_ref() {
            lifecycle.observe_cleanup_time(cleanup_elapsed);
        }
        // Drop publisher handles AFTER engine.cleanup so the engine's
        // last snapshot writes complete. There is no background task to
        // join — snapshot writes are event-driven (engine pushes
        // synchronously); KV-event publishers own their own threads.
        self.publishers = None;
        // Mark stopped even on failure so a follow-up call no-ops; engines
        // like vLLM/TRT-LLM tear down NCCL groups in cleanup() and a second
        // attempt can hang or raise.
        self.state = LifecycleState::Stopped;
    }

    /// Drive the serve loop and the shutdown orchestrator. Returns when
    /// either the serve loop exits or `shutdown` is cancelled.
    async fn serve_with_orchestrator(
        &mut self,
        engine_config: &EngineConfig,
        endpoint: dynamo_runtime::component::Endpoint,
        shutdown: CancellationToken,
    ) -> Result<(), DynamoError> {
        let model_type = resolve_model_type(&self.config)?;
        let (worker_type, needs) = resolve_worker_type_and_needs(&self.config);

        let mut local_model =
            build_local_model(&self.config, engine_config, self.engine.is_raw()).await?;
        tracing::debug!("local model built");

        // Hand the engine its serving endpoint before registering the model
        // with discovery. on_endpoint_ready is a fatal handoff: doing it first
        // means a failure leaves nothing published, so there is no stale
        // discovery entry to reclaim. Engines that publish their own discovery
        // records (e.g. vLLM dynamic LoRA) stash the endpoint here, and this
        // still runs before `register_engine_controls`, so `/engine/*` cannot
        // fire before the engine has the endpoint.
        self.engine.on_endpoint_ready(endpoint.clone()).await?;

        local_model
            .attach(
                &endpoint,
                model_type,
                self.config.model_input,
                None,
                Some(worker_type),
                needs,
            )
            .await
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("model attach: {e}"),
                )
            })?;
        tracing::debug!("model registered with discovery");

        self.register_engine_controls(&endpoint).await?;
        self.register_engine_updates(&endpoint).await?;

        let served = resolve_served_name(&self.config, engine_config)
            .unwrap_or_else(|| engine_config.model.clone());
        tracing::info!(
            "Serving {} on {}.{}.{}",
            served,
            self.config.namespace,
            self.config.component,
            self.config.endpoint
        );

        // Build the request adapter and a JSON-shaped health-check probe
        // engine for the worker's modality. The token pipeline
        // (`EngineAdapter`) needs a `JsonProbeAdapter` wrapper to expose a
        // `serde_json::Value` probe surface; the raw pipeline
        // (`RawEngineAdapter`) is already JSON-shaped, so it serves as its
        // own probe. The tuple annotation drives the trait-object coercions.
        let (ingress, probe_engine): (
            Arc<dyn dynamo_runtime::pipeline::network::PushWorkHandler>,
            dynamo_runtime::local_endpoint_registry::LocalAsyncEngine,
        ) = match &self.engine {
            EngineKind::Llm(engine) => {
                let engine_adapter = Arc::new(EngineAdapter::new(
                    engine.clone(),
                    self.config.disaggregation_mode,
                ));
                let ingress = Ingress::for_engine(engine_adapter.clone()).map_err(|e| {
                    err(
                        ErrorType::Backend(BackendError::Unknown),
                        format!("ingress: {e}"),
                    )
                })?;
                let probe = Arc::new(crate::adapter::JsonProbeAdapter::new(engine_adapter));
                (ingress, probe)
            }
            EngineKind::Raw(engine) => {
                let raw_adapter = Arc::new(RawEngineAdapter::new(engine.clone()));
                let ingress = Ingress::for_engine(raw_adapter.clone()).map_err(|e| {
                    err(
                        ErrorType::Backend(BackendError::Unknown),
                        format!("ingress: {e}"),
                    )
                })?;
                (ingress, raw_adapter)
            }
        };

        let metrics_labels = if self.config.metrics_labels.is_empty() {
            None
        } else {
            Some(self.config.metrics_labels.clone())
        };

        // Hold a registration with the DRT's graceful-shutdown tracker for
        // the entire serve + orchestrate window. If `Runtime::shutdown` is
        // initiated externally, its Phase 2 wait will block on this guard
        // (in addition to the endpoint's own registration), so Phase 3
        // (NATS/etcd teardown) doesn't fire until our `orchestrator_steps`
        // — discovery unregister, grace period, drain, cleanup — finishes.
        let _orchestrator_registration = endpoint.drt().register_graceful_task();

        // Precedence: WorkerConfig (Python argparse plumbs CLI/env here) >
        // DYN_HEALTH_CHECK_PAYLOAD env (backstop for Rust-only engines) >
        // engine default. Every override path stamps the `_HEALTH_CHECK`
        // marker so engines can branch on `is_probe(request)` regardless of
        // where the payload came from.
        let probe = match std::mem::take(&mut self.config.health_check_payload)
            .or_else(load_health_check_payload_from_env)
        {
            Some(p) => stamp_canary_marker(p),
            None => self
                .engine
                .health_check_payload()
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        error = %e,
                        "engine.health_check_payload() failed; canary disabled for this endpoint",
                    );
                    None
                })
                .and_then(stamp_canary_marker),
        };

        let mut builder = endpoint
            .endpoint_builder()
            .handler(ingress)
            .metrics_labels(metrics_labels)
            .graceful_shutdown(true);
        if let Some(payload) = probe {
            builder = builder.health_check_payload(payload);
            // The runtime's `HealthCheckManager` fires the canary by looking
            // up a `LocalAsyncEngine` for this endpoint name. Register the
            // modality's JSON-shaped probe engine so the probe exercises the
            // same `generate()` path as real traffic.
            builder = builder.register_local_engine(probe_engine).map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("register_local_engine: {e}"),
                )
            })?;
        }
        let serve_fut = builder.start();
        tokio::pin!(serve_fut);

        tokio::select! {
            biased;
            result = &mut serve_fut => {
                match result {
                    // Endpoint exited cleanly (e.g. DRT primary token
                    // cancelled it) — run the orchestrator so drain/cleanup
                    // don't race transport teardown.
                    Ok(()) => {
                        tracing::info!(
                            "Endpoint completed gracefully; running shutdown orchestration"
                        );
                    }
                    // Serve errored; cleanup_once in run() is the safety net.
                    Err(e) => {
                        return Err(err(
                            ErrorType::Backend(BackendError::Unknown),
                            format!("serve: {e}"),
                        ));
                    }
                }
            }
            _ = shutdown.cancelled() => {
                tracing::info!("Received shutdown signal; running graceful orchestration");
            }
        }

        self.orchestrator_steps(&endpoint).await;
        Ok(())
    }

    /// Engine-facing shutdown sequence: grace period sleep → drain loop on
    /// `engine.is_quiescent()` → `cleanup_once()`. Each engine step swallows
    /// non-fatal failures so a misbehaving engine can't block the worker
    /// from exiting.
    async fn run_engine_shutdown_steps(&mut self) {
        self.run_engine_shutdown_steps_with_grace(grace_period_secs())
            .await
    }

    /// Same as [`run_engine_shutdown_steps`] but with an explicit grace
    /// period. Lets unit tests assert on call ordering without setting
    /// `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` (which is process-global
    /// and would race other parallel tests).
    async fn run_engine_shutdown_steps_with_grace(&mut self, grace: f64) {
        if grace > 0.0 {
            tracing::info!("Grace period {:.2}s before drain", grace);
            tokio::time::sleep(Duration::from_secs_f64(grace)).await;
        }

        let drain_start = std::time::Instant::now();
        self.drain_until_idle_or_deadline().await;
        let drain_elapsed = drain_start.elapsed().as_secs_f64();
        if let Some(lifecycle) = self.lifecycle.as_ref() {
            lifecycle.observe_drain_time(drain_elapsed);
        }

        self.cleanup_once().await;
    }

    /// Hold a prefill worker open until its KV transfers finish, so cleanup
    /// doesn't free GPU memory a decode peer is still pulling.
    ///
    /// Prefill-only: aggregated/decode workers return immediately. Otherwise
    /// poll [`is_quiescent`](LLMEngine::is_quiescent) every
    /// `DRAIN_POLL_INTERVAL_S`, exiting on `Some(true)` or when the budget
    /// expires. Budget = `DYN_PREFILL_DRAIN_TIMEOUT_S` capped at
    /// `graceful_shutdown_timeout - CLEANUP_RESERVE_S`.
    async fn drain_until_idle_or_deadline(&self) {
        if !self.config.disaggregation_mode.is_prefill() {
            return;
        }
        let configured = drain_timeout_secs();
        let cap = (graceful_shutdown_timeout().as_secs_f64() - CLEANUP_RESERVE_S).max(0.0);
        let budget = configured.min(cap);
        let deadline = std::time::Instant::now() + Duration::from_secs_f64(budget);
        let start = std::time::Instant::now();
        let mut last_heartbeat = start;
        let mut announced = false;
        loop {
            match self.engine.is_quiescent().await {
                // Quiescent: in-flight transfers done, safe to exit drain.
                Ok(Some(true)) => {
                    if announced {
                        tracing::info!(
                            "drain: exited (quiescent, elapsed={:.1}s)",
                            start.elapsed().as_secs_f64()
                        );
                    }
                    return;
                }
                // Busy (Some(false)) or no introspection (None): keep polling.
                Ok(Some(false)) | Ok(None) => {}
                Err(e) => {
                    tracing::debug!(error = %e, "is_quiescent raised; treating as not quiescent")
                }
            }
            if !announced {
                // First non-quiescent poll: announce once that we're waiting.
                tracing::info!(
                    "drain: waiting for prefill to quiesce; polling is_quiescent (timeout={:.1}s)",
                    budget
                );
                announced = true;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    "drain: timed out at {:.1}s; proceeding with cleanup",
                    start.elapsed().as_secs_f64()
                );
                return;
            }
            if last_heartbeat.elapsed().as_secs_f64() >= DRAIN_HEARTBEAT_INTERVAL_S {
                tracing::info!(
                    "drain: heartbeat (elapsed={:.1}s)",
                    start.elapsed().as_secs_f64()
                );
                last_heartbeat = std::time::Instant::now();
            }
            tokio::time::sleep(Duration::from_secs_f64(DRAIN_POLL_INTERVAL_S)).await;
        }
    }
}

/// Drain-budget resolver: `DYN_PREFILL_DRAIN_TIMEOUT_S` with the same
/// validation policy as `grace_period_secs` (invalid → default, negative
/// → 0).
fn drain_timeout_secs() -> f64 {
    match std::env::var(DRAIN_TIMEOUT_ENV) {
        Err(_) => DEFAULT_DRAIN_TIMEOUT_S,
        Ok(s) if s.is_empty() => DEFAULT_DRAIN_TIMEOUT_S,
        Ok(s) => match s.parse::<f64>() {
            Ok(v) if !v.is_finite() => {
                tracing::warn!(
                    "Non-finite {}={:?}; using default {:.1}s",
                    DRAIN_TIMEOUT_ENV,
                    s,
                    DEFAULT_DRAIN_TIMEOUT_S
                );
                DEFAULT_DRAIN_TIMEOUT_S
            }
            Ok(v) if v < 0.0 => {
                tracing::warn!("Negative {}={:?}; clamping to 0", DRAIN_TIMEOUT_ENV, s);
                0.0
            }
            Ok(v) => v,
            Err(_) => {
                tracing::warn!(
                    "Invalid {}={:?}; using default {:.1}s",
                    DRAIN_TIMEOUT_ENV,
                    s,
                    DEFAULT_DRAIN_TIMEOUT_S
                );
                DEFAULT_DRAIN_TIMEOUT_S
            }
        },
    }
}

/// Read the post-signal shutdown deadline from
/// `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` (matching `dynamo_runtime::Worker`).
/// On expiry the worker hard-exits with code 911 — same contract as the
/// upstream `worker.execute` flow we bypass. Defaults are imported from
/// `dynamo_runtime::worker` so a default change there propagates here
/// without manual sync.
fn graceful_shutdown_timeout() -> Duration {
    use dynamo_runtime::config::environment_names::worker as env_worker;
    use dynamo_runtime::worker::{
        DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG, DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE,
    };

    let default = if cfg!(debug_assertions) {
        DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG
    } else {
        DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE
    };

    let value = std::env::var(env_worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT).ok();
    let secs = graceful_shutdown_timeout_secs(value.as_deref(), default);
    Duration::from_secs(secs)
}

fn graceful_shutdown_timeout_secs(value: Option<&str>, default: u64) -> u64 {
    value
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Compose the post-signal shutdown deadline from the drain+cleanup
/// timeout and the grace-period sleep that precedes them.
///
/// The grace sleep is a fixed wait (not a hang risk), so reserving its
/// duration on top of `timeout` ensures the drain loop and
/// `engine.cleanup()` always get the full timeout budget regardless of
/// how the operator configures the grace period. Without this reserve,
/// a grace period equal to the timeout (the debug default — both 5s)
/// consumes the whole budget and the deadline expires before drain or
/// cleanup get scheduled.
fn shutdown_deadline(timeout: Duration, grace_secs: f64) -> Duration {
    let grace = if grace_secs > 0.0 {
        Duration::from_secs_f64(grace_secs)
    } else {
        Duration::ZERO
    };
    timeout.saturating_add(grace)
}

/// Validate that `value` is a JSON object and stamp the canary marker on
/// it. Returns `None` for non-object payloads (logs a warning) so the
/// canary stays disabled rather than being registered with an invalid
/// shape. Operator overrides reach the engine's `generate()` with the
/// marker set so `is_probe(request)` detects them.
fn stamp_canary_marker(mut value: serde_json::Value) -> Option<serde_json::Value> {
    let Some(obj) = value.as_object_mut() else {
        tracing::warn!(
            ?value,
            "health_check_payload override is not a JSON object; canary disabled"
        );
        return None;
    };
    obj.insert(
        crate::engine::HEALTH_CHECK_KEY.to_string(),
        serde_json::Value::Bool(true),
    );
    Some(value)
}

/// Read `DYN_HEALTH_CHECK_PAYLOAD` (JSON object or `@/path/to/file.json`).
/// Returns `None` when the env is unset or the value is invalid; an invalid
/// value logs a warning so it can't silently disable the engine default.
fn load_health_check_payload_from_env() -> Option<serde_json::Value> {
    let raw = std::env::var(HEALTH_CHECK_PAYLOAD_ENV).ok();
    load_health_check_payload(raw.as_deref())
}

fn load_health_check_payload(raw: Option<&str>) -> Option<serde_json::Value> {
    let raw = raw.filter(|s| !s.is_empty())?;
    let parsed: Result<serde_json::Value, _> = if let Some(path) = raw.strip_prefix('@') {
        std::fs::read_to_string(path).map_or_else(
            |e| Err(format!("read {path}: {e}")),
            |s| serde_json::from_str(&s).map_err(|e| e.to_string()),
        )
    } else {
        serde_json::from_str(raw).map_err(|e| e.to_string())
    };
    match parsed {
        Ok(v) if v.is_object() => Some(v),
        Ok(_) => {
            tracing::warn!(
                env = HEALTH_CHECK_PAYLOAD_ENV,
                "value must be a JSON object"
            );
            None
        }
        Err(e) => {
            tracing::warn!(env = HEALTH_CHECK_PAYLOAD_ENV, error = %e, "parse failed");
            None
        }
    }
}

/// Read the grace-period seconds from `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`,
/// matching the Python helper. Negative values clamp to 0.
fn grace_period_secs() -> f64 {
    let value = std::env::var(GRACE_PERIOD_ENV).ok();
    grace_period_secs_from(value.as_deref())
}

fn grace_period_secs_from(value: Option<&str>) -> f64 {
    match value {
        Some(s) if !s.is_empty() => match s.parse::<f64>() {
            Ok(v) if v >= 0.0 => v,
            Ok(_) => 0.0,
            Err(_) => {
                tracing::warn!(
                    "Invalid {}={:?}; using default {}",
                    GRACE_PERIOD_ENV,
                    s,
                    DEFAULT_GRACE_PERIOD_SECS
                );
                DEFAULT_GRACE_PERIOD_SECS
            }
        },
        _ => DEFAULT_GRACE_PERIOD_SECS,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EngineControlPolicy {
    Direct,
    UnregisterBefore,
    RegisterAfter,
}

fn engine_control_policy(control: &str) -> EngineControlPolicy {
    // This policy only governs discovery (un)registration ordering. Draining
    // in-flight work before memory is freed is delegated to each backend's
    // pause controller: vLLM calls pause_generation() before native sleep(),
    // SGLang calls pause_generation() before release_memory_occupation(), and
    // TRT-LLM rejects new requests and waits for inflight requests to finish. The
    // UnregisterBefore step here is an additional guard (stop new routing), not
    // the drain itself.
    match control {
        // Pause controls make the engine unsafe for new requests, so remove
        // the endpoint before they mutate engine state. Resume controls make
        // the engine serving-safe again, so advertise it only after success.
        "sleep" | "release_memory_occupation" => EngineControlPolicy::UnregisterBefore,
        "wake_up" | "resume_memory_occupation" => EngineControlPolicy::RegisterAfter,
        _ => EngineControlPolicy::Direct,
    }
}

fn control_response_is_error(value: &serde_json::Value) -> bool {
    value
        .get("status")
        .and_then(|v| v.as_str())
        .is_some_and(|status| status.eq_ignore_ascii_case("error"))
        || value
            .get("success")
            .and_then(|v| v.as_bool())
            .is_some_and(|success| !success)
}

fn control_error_response(message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({"status": "error", "message": message.into()})
}

fn control_request_body_error(body: &serde_json::Value) -> Option<serde_json::Value> {
    if body.is_object() {
        None
    } else {
        Some(control_error_response(
            "engine control request body must be a JSON object",
        ))
    }
}

fn update_request_body_error(body: &serde_json::Value) -> Option<serde_json::Value> {
    if body.is_object() {
        None
    } else {
        Some(control_error_response(
            "engine update request body must be a JSON object",
        ))
    }
}

fn engine_control_callback(control_name: String, engine: EngineKind) -> EngineRouteCallback {
    Arc::new(move |body| {
        let engine = engine.clone();
        let control_name = control_name.clone();
        Box::pin(async move {
            engine
                .engine_control(control_name, body)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))
        })
    })
}

fn engine_update_callback(update_name: String, engine: EngineKind) -> EngineRouteCallback {
    Arc::new(move |body| {
        let engine = engine.clone();
        let update_name = update_name.clone();
        Box::pin(async move {
            if let Some(response) = update_request_body_error(&body) {
                return Ok(response);
            }
            engine
                .engine_update(update_name, body)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))
        })
    })
}

fn wrap_engine_control_callback(
    control_name: String,
    callback: EngineRouteCallback,
    endpoint: dynamo_runtime::component::Endpoint,
    control_lock: Arc<tokio::sync::Mutex<()>>,
) -> EngineRouteCallback {
    let policy = engine_control_policy(&control_name);
    Arc::new(move |body| {
        let callback = callback.clone();
        let endpoint = endpoint.clone();
        let control_name = control_name.clone();
        let control_lock = control_lock.clone();
        Box::pin(async move {
            match policy {
                EngineControlPolicy::Direct => callback(body).await,
                EngineControlPolicy::UnregisterBefore => {
                    if let Some(response) = control_request_body_error(&body) {
                        return Ok(response);
                    }

                    // Hold across unregister + callback so a concurrent resume
                    // cannot re-register between them.
                    let _guard = control_lock.lock().await;

                    if let Err(e) = endpoint.unregister_endpoint_instance().await {
                        return Ok(control_error_response(format!(
                            "failed to unregister endpoint before /engine/control/{control_name}: {e}"
                        )));
                    }

                    match callback(body).await {
                        Ok(response) => {
                            if control_response_is_error(&response) {
                                tracing::warn!(
                                    control = %control_name,
                                    "engine control returned an error after endpoint unregister; leaving endpoint unregistered"
                                );
                            }
                            Ok(response)
                        }
                        Err(e) => {
                            tracing::warn!(
                                control = %control_name,
                                error = %e,
                                "engine control callback failed after endpoint unregister; leaving endpoint unregistered"
                            );
                            Err(e)
                        }
                    }
                }
                EngineControlPolicy::RegisterAfter => {
                    // Hold across callback + register so a concurrent pause
                    // cannot unregister between them.
                    let _guard = control_lock.lock().await;

                    let response = callback(body).await?;
                    if !control_response_is_error(&response)
                        && let Err(e) = endpoint.register_endpoint_instance().await
                    {
                        // The engine is serving-safe but absent from discovery. The
                        // operation is idempotent: retrying /engine/control/{control_name}
                        // re-registers without repeating the wake/resume work (the
                        // controller short-circuits "already awake/resumed"), so surface
                        // that it is safe to retry.
                        return Ok(control_error_response(format!(
                            "engine resumed but re-registration failed after /engine/control/{control_name}: {e}; retry /engine/control/{control_name} to rejoin discovery"
                        )));
                    }
                    Ok(response)
                }
            }
        })
    })
}

/// Convenience shorthand for `DynamoError::builder().error_type(..).message(..).build()`.
fn err(error_type: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(error_type)
        .message(message)
        .build()
}

/// Resolve the public-facing served-model name.
///
/// Priority: `WorkerConfig.served_model_name` (operator CLI override) →
/// `EngineConfig.served_model_name` (engine's preferred advertise-as name).
/// Returns `None` if neither is set; callers fall back to
/// `EngineConfig.model`.
fn resolve_served_name(config: &WorkerConfig, engine_config: &EngineConfig) -> Option<String> {
    config
        .served_model_name
        .clone()
        .or_else(|| engine_config.served_model_name.clone())
}

/// Pick the `ModelType` to register with based on the worker's disaggregation
/// role. The prefill role is carried by `worker_type`; prefill workers expose
/// no OpenAI surface. They register the legacy `ModelType::Prefill` *marker*
/// bit (not a surface) so an OLD frontend, which detects prefill via that bit,
/// still routes disaggregated traffic during the cross-version rollout. A new
/// frontend ignores it and dispatches off `worker_type`.
///
/// Encode workers also expose no public OpenAI surface — they are reached
/// through encoder routing, not the frontend's public serving surface. They
/// register surface-less
/// (`ModelType::empty()`) so the discovery watcher registers them for
/// serving-readiness only and hides them from `/v1/models`; the role is
/// carried by `WorkerType::Encode`.
///
/// Everything else falls back to the parsed `endpoint_types`.
fn resolve_model_type(config: &WorkerConfig) -> Result<ModelType, DynamoError> {
    if config.disaggregation_mode.is_prefill() {
        return Ok(ModelType::Prefill);
    }
    if config.disaggregation_mode.is_encode() {
        return Ok(ModelType::empty());
    }
    parse_endpoint_types(&config.endpoint_types)
}

/// Derive the topology-readiness fields (`worker_type`, `needs`) for the
/// worker's disaggregation role. Prefill workers need a Decode peer, Decode
/// workers need a Prefill peer, Aggregated workers stand alone, and Encode
/// workers need either a Prefill+Decode pair or a single Aggregated peer.
///
/// `route_to_encoder` extends the `needs` of `Prefill`/`Aggregated` to
/// also require an `Encode` peer. Invalid combinations (`Decode` or
/// `Encode` with `route_to_encoder=true`) are rejected upstream in
/// `validate_route_to_encoder`; this function trusts that gate and only
/// applies the flag when it is meaningful.
fn resolve_worker_type_and_needs(config: &WorkerConfig) -> (WorkerType, Vec<Vec<WorkerType>>) {
    match config.disaggregation_mode {
        DisaggregationMode::Prefill => {
            let inner = if config.route_to_encoder {
                vec![WorkerType::Decode, WorkerType::Encode]
            } else {
                vec![WorkerType::Decode]
            };
            (WorkerType::Prefill, vec![inner])
        }
        DisaggregationMode::Decode => (WorkerType::Decode, vec![vec![WorkerType::Prefill]]),
        DisaggregationMode::Aggregated => {
            let needs = if config.route_to_encoder {
                vec![vec![WorkerType::Encode]]
            } else {
                Vec::new()
            };
            (WorkerType::Aggregated, needs)
        }
        DisaggregationMode::Encode => (
            WorkerType::Encode,
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
        ),
    }
}

/// Validate that `route_to_encoder` is meaningful for the worker's
/// disaggregation role. Setting the flag on `Decode` or `Encode` is a
/// configuration bug: Decode reads KV cache from a Prefill peer and never
/// sees the encoder output (the dependency is transitive through Prefill),
/// while Encode is the producer of the encoder result and has nothing
/// upstream to route to.
fn validate_route_to_encoder(config: &WorkerConfig) -> Result<(), DynamoError> {
    if !config.route_to_encoder {
        return Ok(());
    }
    match config.disaggregation_mode {
        DisaggregationMode::Aggregated | DisaggregationMode::Prefill => Ok(()),
        DisaggregationMode::Decode | DisaggregationMode::Encode => Err(err(
            ErrorType::Backend(BackendError::InvalidArgument),
            format!(
                "--route-to-encoder is meaningful only for --disaggregation-mode \
                 agg|prefill; got '{}'. Decode workers consume KV cache from a \
                 Prefill peer (encoder dependency propagates transitively through \
                 Prefill); Encode workers are the producer of the encoder result.",
                config.disaggregation_mode
            ),
        )),
    }
}

fn parse_endpoint_types(s: &str) -> Result<ModelType, DynamoError> {
    let mut out = ModelType::empty();
    let mut any = false;
    for raw in s.split(',') {
        let t = raw.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        let flag = match t.as_str() {
            "chat" => ModelType::Chat,
            "completions" => ModelType::Completions,
            "embedding" | "embeddings" => ModelType::Embedding,
            "tensor" => ModelType::TensorBased,
            // The prefill role is declared via `worker_type` (driven by the
            // disaggregation mode), not as an endpoint type. Reject
            // "prefill" here — it never made sense as one.
            // Raw media-generation modalities (served by a RawEngine).
            "images" | "image" => ModelType::Images,
            "videos" | "video" => ModelType::Videos,
            "audios" | "audio" => ModelType::Audios,
            other => {
                return Err(err(
                    ErrorType::Backend(BackendError::InvalidArgument),
                    format!("unknown endpoint type '{other}'"),
                ));
            }
        };
        out |= flag;
        any = true;
    }
    if !any {
        return Err(err(
            ErrorType::Backend(BackendError::InvalidArgument),
            "endpoint_types cannot be empty",
        ));
    }
    Ok(out)
}

/// Check `model_input` matches the engine modality: [`LLMEngine`] needs
/// `Tokens`; [`RawEngine`] needs `Text`/`Tensor` (no tokenizer stage).
fn validate_model_input(model_input: ModelInput, engine: &EngineKind) -> Result<(), DynamoError> {
    match engine {
        EngineKind::Llm(_) => {
            if model_input == ModelInput::Tokens {
                Ok(())
            } else {
                Err(err(
                    ErrorType::Backend(BackendError::InvalidArgument),
                    format!(
                        "LLMEngine (token pipeline) requires ModelInput::Tokens; got '{}'. \
                         Use a RawEngine for ModelInput::Text / Tensor.",
                        model_input.as_str()
                    ),
                ))
            }
        }
        EngineKind::Raw(_) => {
            if model_input == ModelInput::Tokens {
                Err(err(
                    ErrorType::Backend(BackendError::InvalidArgument),
                    "RawEngine (raw media pipeline) requires ModelInput::Text or ::Tensor; \
                     got 'tokens'. Use an LLMEngine for the token pipeline."
                        .to_string(),
                ))
            } else {
                Ok(())
            }
        }
    }
}

async fn build_local_model(
    config: &WorkerConfig,
    engine_config: &EngineConfig,
    name_only: bool,
) -> Result<LocalModel, DynamoError> {
    let served_name = resolve_served_name(config, engine_config)
        .or_else(|| Some(engine_config.model.clone()))
        .filter(|s| !s.is_empty());

    // Decode workers don't host the WorkerKvQuery endpoint, so they must not
    // advertise the local indexer regardless of the operator-supplied flag.
    // Mirrors the legacy non-unified vLLM path (worker_factory.py).
    let enable_local_indexer = config.effective_enable_local_indexer();

    // None for raw engines → all-`None` fields → no KV/DP/bootstrap hints.
    let llm = engine_config.llm.clone().unwrap_or_default();

    // Publish the disaggregated bootstrap endpoint when the engine
    // returned one. Only meaningful for prefill workers — decode/agg
    // engines leave both fields `None`. The frontend's `PrefillRouter`
    // reads this from `model_manager.get_disaggregated_endpoint(...)` to
    // take its optimised "Bootstrap path" (route decode concurrent with
    // prefill instead of waiting for prefill to drain).
    let disaggregated_endpoint = match (&llm.bootstrap_host, llm.bootstrap_port) {
        (Some(host), Some(port)) => {
            tracing::info!(
                bootstrap_host = %host,
                bootstrap_port = port,
                "Publishing disaggregated_endpoint for prefill worker"
            );
            Some(DisaggregatedEndpoint {
                bootstrap_host: Some(host.clone()),
                bootstrap_port: Some(port),
            })
        }
        _ => None,
    };

    let mut runtime_data = engine_config.runtime_data.clone();
    if let Some(default_thinking_mode) = config.default_thinking_mode.as_deref() {
        runtime_data.insert(
            "default_thinking_mode".to_string(),
            serde_json::json!(default_thinking_mode),
        );
    }

    let rt_cfg = ModelRuntimeConfig {
        context_length: llm.context_length,
        total_kv_blocks: llm.total_kv_blocks,
        max_num_seqs: llm.max_num_seqs,
        max_num_batched_tokens: llm.max_num_batched_tokens,
        data_parallel_size: llm.data_parallel_size.unwrap_or(1),
        data_parallel_start_rank: llm.data_parallel_start_rank.unwrap_or(0),
        tool_call_parser: config.tool_call_parser.clone(),
        reasoning_parser: config.reasoning_parser.clone(),
        exclude_tools_when_tool_choice_none: config.exclude_tools_when_tool_choice_none,
        structural_tag_mode: config.structural_tag_mode,
        structural_tag_scope: config.structural_tag_scope,
        structural_tag_schema: config.structural_tag_schema,
        enable_local_indexer,
        disaggregated_endpoint,
        runtime_data,
        ..ModelRuntimeConfig::default()
    };

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(served_name)
        .kv_cache_block_size(llm.kv_cache_block_size)
        .custom_template_path(config.custom_jinja_template.clone())
        .media_decoder(config.media_decoder.clone())
        .media_fetcher(config.media_fetcher.clone())
        .runtime_config(rt_cfg);

    // Resolve model_name to a local path. Empty string or a raw media engine
    // (`name_only`) → name-only card (no tokenizer/template): raw models carry
    // no LLM artifacts to fetch and load themselves (cf. the legacy
    // diffusion path's `ModelDeploymentCard::with_name_only()`).
    if !config.model_name.is_empty() && !name_only {
        let source = config.model_name.clone();
        let local_path = if std::fs::exists(&source).map_err(|e| {
            err(
                ErrorType::Backend(BackendError::InvalidArgument),
                format!("model path: {e}"),
            )
        })? {
            PathBuf::from(&source)
        } else {
            LocalModel::fetch(&source, false).await.map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("fetch '{source}': {e}"),
                )
            })?
        };
        builder.model_path(local_path);
        builder.source_path(PathBuf::from(source));
    }

    builder.build().await.map_err(|e| {
        err(
            ErrorType::Backend(BackendError::Unknown),
            format!("build local model: {e}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_type_of(result: Result<ModelType, DynamoError>) -> ErrorType {
        result.unwrap_err().error_type()
    }

    #[test]
    fn parse_endpoint_types_happy_path() {
        let got = parse_endpoint_types("chat,completions").unwrap();
        assert_eq!(got, ModelType::Chat | ModelType::Completions);
    }

    #[test]
    fn parse_endpoint_types_single() {
        assert_eq!(parse_endpoint_types("chat").unwrap(), ModelType::Chat);
        assert_eq!(
            parse_endpoint_types("completions").unwrap(),
            ModelType::Completions
        );
        assert_eq!(
            parse_endpoint_types("embedding").unwrap(),
            ModelType::Embedding
        );
    }

    #[test]
    fn parse_endpoint_types_trims_and_lowercases() {
        let got = parse_endpoint_types("  Chat , COMPLETIONS  ").unwrap();
        assert_eq!(got, ModelType::Chat | ModelType::Completions);
    }

    #[test]
    fn parse_endpoint_types_rejects_empty() {
        assert_eq!(
            error_type_of(parse_endpoint_types("")),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(
            error_type_of(parse_endpoint_types("   ,  ")),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn parse_endpoint_types_rejects_unknown() {
        let e = parse_endpoint_types("chat,bogus").unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(e.to_string().contains("bogus"));
    }

    /// Minimal `RawEngine` for validation tests — never started/served.
    struct ValidationRawMock;

    #[async_trait]
    impl RawEngine for ValidationRawMock {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            unreachable!("not used in validation tests")
        }
        async fn generate(
            &self,
            _request: serde_json::Value,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<BoxStream<'static, Result<serde_json::Value, DynamoError>>, DynamoError>
        {
            unreachable!("not used in validation tests")
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    fn llm_kind() -> EngineKind {
        let (engine, _) = StateMockEngine::new(false);
        EngineKind::Llm(engine)
    }

    #[test]
    fn engine_control_policy_wraps_discovery_mutating_controls() {
        assert_eq!(
            engine_control_policy("start_profile"),
            EngineControlPolicy::Direct
        );
        assert_eq!(
            engine_control_policy("stop_profile"),
            EngineControlPolicy::Direct
        );
        assert_eq!(
            engine_control_policy("update_weights_from_disk"),
            EngineControlPolicy::Direct
        );
        assert_eq!(
            engine_control_policy("clear_kv_blocks"),
            EngineControlPolicy::Direct
        );
        assert_eq!(
            engine_control_policy("sleep"),
            EngineControlPolicy::UnregisterBefore
        );
        assert_eq!(
            engine_control_policy("release_memory_occupation"),
            EngineControlPolicy::UnregisterBefore
        );
        assert_eq!(
            engine_control_policy("wake_up"),
            EngineControlPolicy::RegisterAfter
        );
        assert_eq!(
            engine_control_policy("resume_memory_occupation"),
            EngineControlPolicy::RegisterAfter
        );
    }

    #[test]
    fn control_request_body_validation_requires_json_object() {
        assert!(control_request_body_error(&serde_json::json!({})).is_none());
        assert!(control_request_body_error(&serde_json::json!({"tags": ["kv_cache"]})).is_none());

        for body in [
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!("bad"),
            serde_json::json!(["kv_cache"]),
        ] {
            let response = control_request_body_error(&body).unwrap();
            assert!(control_response_is_error(&response));
            assert_eq!(
                response.get("message").and_then(|value| value.as_str()),
                Some("engine control request body must be a JSON object")
            );
        }
    }

    #[test]
    fn update_request_body_validation_requires_json_object() {
        assert!(update_request_body_error(&serde_json::json!({})).is_none());
        assert!(update_request_body_error(&serde_json::json!({"lora_name": "a"})).is_none());

        for body in [
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!("bad"),
            serde_json::json!(["lora_name"]),
        ] {
            let response = update_request_body_error(&body).unwrap();
            assert!(control_response_is_error(&response));
            assert_eq!(
                response.get("message").and_then(|value| value.as_str()),
                Some("engine update request body must be a JSON object")
            );
        }
    }

    #[test]
    fn control_response_error_detection_matches_backend_conventions() {
        assert!(control_response_is_error(&serde_json::json!({
            "status": "error"
        })));
        assert!(control_response_is_error(&serde_json::json!({
            "status": "ERROR"
        })));
        assert!(control_response_is_error(&serde_json::json!({
            "success": false
        })));

        assert!(!control_response_is_error(&serde_json::json!({
            "status": "ok"
        })));
        assert!(!control_response_is_error(&serde_json::json!({
            "success": true
        })));
        assert!(!control_response_is_error(&serde_json::json!({
            "message": "ok"
        })));
    }

    fn raw_kind() -> EngineKind {
        EngineKind::Raw(Arc::new(ValidationRawMock))
    }

    #[test]
    fn validate_model_input_llm_accepts_tokens() {
        validate_model_input(ModelInput::Tokens, &llm_kind()).unwrap();
    }

    #[test]
    fn validate_model_input_llm_rejects_text_and_tensor() {
        for input in [ModelInput::Text, ModelInput::Tensor] {
            let e = validate_model_input(input, &llm_kind()).unwrap_err();
            assert_eq!(
                e.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(e.to_string().contains(input.as_str()));
        }
    }

    #[test]
    fn validate_model_input_raw_accepts_text_and_tensor() {
        validate_model_input(ModelInput::Text, &raw_kind()).unwrap();
        validate_model_input(ModelInput::Tensor, &raw_kind()).unwrap();
    }

    #[test]
    fn validate_model_input_raw_rejects_tokens() {
        let e = validate_model_input(ModelInput::Tokens, &raw_kind()).unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[tokio::test]
    async fn build_local_model_carries_runtime_parser_settings() {
        let config = WorkerConfig {
            tool_call_parser: Some("kimi_k2".to_string()),
            reasoning_parser: Some("kimi_k25".to_string()),
            default_thinking_mode: Some("disabled".to_string()),
            exclude_tools_when_tool_choice_none: false,
            enable_local_indexer: false,
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "nvidia/Kimi-K2.5-NVFP4".to_string(),
            runtime_data: [(
                "sglang_worker_group_id".to_string(),
                serde_json::json!("group-a"),
            )]
            .into(),
            llm: Some(crate::engine::LlmRegistration {
                context_length: Some(32_768),
                total_kv_blocks: Some(100),
                max_num_seqs: Some(16),
                max_num_batched_tokens: Some(8192),
                ..Default::default()
            }),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, false)
            .await
            .unwrap();
        let runtime_config = local_model.runtime_config();

        assert_eq!(runtime_config.context_length, Some(32_768));
        assert_eq!(runtime_config.total_kv_blocks, Some(100));
        assert_eq!(runtime_config.max_num_seqs, Some(16));
        assert_eq!(runtime_config.max_num_batched_tokens, Some(8192));
        assert_eq!(runtime_config.tool_call_parser.as_deref(), Some("kimi_k2"));
        assert_eq!(runtime_config.reasoning_parser.as_deref(), Some("kimi_k25"));
        assert_eq!(
            runtime_config
                .runtime_data
                .get("default_thinking_mode")
                .and_then(|value| value.as_str()),
            Some("disabled")
        );
        assert!(!runtime_config.exclude_tools_when_tool_choice_none);
        assert!(!runtime_config.enable_local_indexer);
        assert_eq!(
            runtime_config
                .runtime_data
                .get("sglang_worker_group_id")
                .and_then(|value| value.as_str()),
            Some("group-a")
        );
    }

    #[tokio::test]
    async fn build_local_model_name_only_skips_fetch() {
        // Raw media engines register name-only: a model_name that is neither a
        // local path nor a valid HF repo must NOT trigger a fetch (the engine
        // loads the model itself). If the gate regresses, this would attempt a
        // network fetch and fail with BackendCannotConnect.
        let bogus = "definitely/not-a-real-hf-model-xyz".to_string();
        let config = WorkerConfig {
            model_name: bogus.clone(),
            endpoint_types: "images".to_string(),
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: bogus,
            ..EngineConfig::default()
        };
        // name_only=true must succeed offline; name_only=false would fetch.
        build_local_model(&config, &engine_config, true)
            .await
            .expect("name-only build must not fetch");
    }

    #[tokio::test]
    async fn build_local_model_carries_media_configuration() {
        let config = WorkerConfig {
            media_decoder: Some(MediaDecoder::default()),
            media_fetcher: Some(MediaFetcher::default()),
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "media-config-test".to_string(),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, true)
            .await
            .expect("name-only model with media config must build");

        assert!(local_model.card().media_decoder.is_some());
        assert!(local_model.card().media_fetcher.is_some());
    }

    #[test]
    fn resolve_model_type_aggregated_uses_endpoint_types() {
        let config = WorkerConfig {
            endpoint_types: "chat,completions".to_string(),
            disaggregation_mode: DisaggregationMode::Aggregated,
            ..WorkerConfig::default()
        };
        assert_eq!(
            resolve_model_type(&config).unwrap(),
            ModelType::Chat | ModelType::Completions,
        );
    }

    #[test]
    fn resolve_model_type_decode_uses_endpoint_types() {
        // Decode workers register with the chat/completions surface; only
        // prefill workers short-circuit to an empty ModelType (their role
        // is carried by WorkerType::Prefill instead).
        let config = WorkerConfig {
            endpoint_types: "chat".to_string(),
            disaggregation_mode: DisaggregationMode::Decode,
            ..WorkerConfig::default()
        };
        assert_eq!(resolve_model_type(&config).unwrap(), ModelType::Chat);
    }

    #[test]
    fn resolve_model_type_prefill_uses_prefill_marker() {
        // The operator may have left endpoint_types at the default
        // "chat,completions"; --disaggregation-mode prefill forces the
        // ModelType to the legacy Prefill marker bit (no OpenAI surface) — the
        // prefill role is declared on `worker_type`, and the marker is
        // dual-emitted so an old frontend still detects it. It must expose no
        // OpenAI surface.
        let config = WorkerConfig {
            endpoint_types: "chat,completions".to_string(),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..WorkerConfig::default()
        };
        let mt = resolve_model_type(&config).unwrap();
        assert_eq!(mt, ModelType::Prefill);
        assert!(mt.supports_prefill());
        assert!(!mt.supports_chat());
        assert!(!mt.supports_completions());
    }

    #[test]
    fn resolve_model_type_encode_is_surface_less() {
        // Encode workers expose no public OpenAI surface: they are reached
        // through encoder routing, not the frontend. --disaggregation-mode encode
        // forces ModelType::empty() (even when endpoint_types is left at the
        // "chat,completions" default) so the discovery watcher registers them
        // for serving-readiness only and hides them from /v1/models. The role
        // is carried by WorkerType::Encode + topology needs at the discovery
        // layer.
        let config = WorkerConfig {
            endpoint_types: "chat,completions".to_string(),
            disaggregation_mode: DisaggregationMode::Encode,
            ..WorkerConfig::default()
        };
        let mt = resolve_model_type(&config).unwrap();
        assert_eq!(mt, ModelType::empty());
        assert!(mt.is_empty());
        assert!(!mt.supports_chat());
        assert!(!mt.supports_completions());
    }

    // -------------------------------------------------------------------
    // resolve_worker_type_and_needs: one test per row of the topology table
    // -------------------------------------------------------------------

    #[test]
    fn topology_aggregated_no_route_to_encoder() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: false,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Aggregated);
        assert!(needs.is_empty());
    }

    #[test]
    fn topology_aggregated_with_route_to_encoder() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: true,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Aggregated);
        assert_eq!(needs, vec![vec![WorkerType::Encode]]);
    }

    #[test]
    fn topology_prefill_no_route_to_encoder() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Prefill,
            route_to_encoder: false,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Prefill);
        assert_eq!(needs, vec![vec![WorkerType::Decode]]);
    }

    #[test]
    fn topology_prefill_with_route_to_encoder() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Prefill,
            route_to_encoder: true,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Prefill);
        assert_eq!(needs, vec![vec![WorkerType::Decode, WorkerType::Encode]]);
    }

    #[test]
    fn topology_decode_ignores_route_to_encoder_flag_in_needs() {
        // route_to_encoder=true on Decode is rejected by
        // validate_route_to_encoder; resolve_worker_type_and_needs only
        // sees the flag false case in production. But sanity-check that
        // even if it leaks through (e.g. internal callers bypassing the
        // validator), Decode's needs don't grow an encoder leg.
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Decode,
            route_to_encoder: false,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Decode);
        assert_eq!(needs, vec![vec![WorkerType::Prefill]]);
    }

    #[test]
    fn topology_encode_has_two_alternative_needs() {
        // Encode: needs `[[Prefill, Decode], [Aggregated]]` (DNF).
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Encode,
            route_to_encoder: false,
            ..WorkerConfig::default()
        };
        let (wt, needs) = resolve_worker_type_and_needs(&cfg);
        assert_eq!(wt, WorkerType::Encode);
        assert_eq!(
            needs,
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
        );
    }

    // -------------------------------------------------------------------
    // validate_route_to_encoder: one test per row of the rejection table
    // -------------------------------------------------------------------

    #[test]
    fn validate_route_to_encoder_accepts_aggregated_and_prefill_when_true() {
        for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Prefill] {
            let cfg = WorkerConfig {
                disaggregation_mode: mode,
                route_to_encoder: true,
                ..WorkerConfig::default()
            };
            validate_route_to_encoder(&cfg).unwrap_or_else(|e| {
                panic!("route_to_encoder=true should be accepted for {mode}; got {e}")
            });
        }
    }

    #[test]
    fn validate_route_to_encoder_rejects_decode_when_true() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Decode,
            route_to_encoder: true,
            ..WorkerConfig::default()
        };
        let e = validate_route_to_encoder(&cfg).unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(e.to_string().contains("decode"), "msg = {e}");
    }

    #[test]
    fn validate_route_to_encoder_rejects_encode_when_true() {
        let cfg = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Encode,
            route_to_encoder: true,
            ..WorkerConfig::default()
        };
        let e = validate_route_to_encoder(&cfg).unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(e.to_string().contains("encode"), "msg = {e}");
    }

    #[test]
    fn validate_route_to_encoder_accepts_any_mode_when_false() {
        for mode in [
            DisaggregationMode::Aggregated,
            DisaggregationMode::Prefill,
            DisaggregationMode::Decode,
            DisaggregationMode::Encode,
        ] {
            let cfg = WorkerConfig {
                disaggregation_mode: mode,
                route_to_encoder: false,
                ..WorkerConfig::default()
            };
            validate_route_to_encoder(&cfg).unwrap_or_else(|e| {
                panic!("route_to_encoder=false should be accepted for {mode}; got {e}")
            });
        }
    }

    // -------------------------------------------------------------------
    // effective_enable_local_indexer: Encode must force off
    // -------------------------------------------------------------------

    #[test]
    fn effective_enable_local_indexer_encode_force_disabled() {
        let cfg = WorkerConfig {
            enable_local_indexer: true,
            disaggregation_mode: DisaggregationMode::Encode,
            ..WorkerConfig::default()
        };
        assert!(!cfg.effective_enable_local_indexer());
    }

    #[tokio::test]
    async fn build_local_model_decode_disables_local_indexer() {
        let config = WorkerConfig {
            enable_local_indexer: true,
            disaggregation_mode: DisaggregationMode::Decode,
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "test/model".to_string(),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, false)
            .await
            .unwrap();
        // Decode workers cannot host the local indexer endpoint, so the
        // worker forces it off even when the operator-supplied flag is true.
        assert!(!local_model.runtime_config().enable_local_indexer);
    }

    #[tokio::test]
    async fn build_local_model_aggregated_keeps_local_indexer() {
        let config = WorkerConfig {
            enable_local_indexer: true,
            disaggregation_mode: DisaggregationMode::Aggregated,
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "test/model".to_string(),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, false)
            .await
            .unwrap();
        assert!(local_model.runtime_config().enable_local_indexer);
    }

    #[tokio::test]
    async fn build_local_model_publishes_disaggregated_endpoint_when_engine_provides_it() {
        // Prefill engines populate `EngineConfig.bootstrap_host/port` in
        // `start()`; `build_local_model` must surface that on the
        // `ModelRuntimeConfig` so the frontend's PrefillRouter can take
        // its optimised Bootstrap path.
        let config = WorkerConfig {
            disaggregation_mode: DisaggregationMode::Prefill,
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "test/model".to_string(),
            llm: Some(crate::engine::LlmRegistration {
                bootstrap_host: Some("10.0.0.5".to_string()),
                bootstrap_port: Some(12345),
                ..Default::default()
            }),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, false)
            .await
            .unwrap();
        let endpoint = local_model
            .runtime_config()
            .disaggregated_endpoint
            .as_ref()
            .expect("disaggregated_endpoint must be published");
        assert_eq!(endpoint.bootstrap_host.as_deref(), Some("10.0.0.5"));
        assert_eq!(endpoint.bootstrap_port, Some(12345));
    }

    #[tokio::test]
    async fn build_local_model_skips_disaggregated_endpoint_when_engine_omits_it() {
        // Aggregated/decode workers don't have a bootstrap address —
        // leaving both fields None on EngineConfig must keep the
        // disaggregated_endpoint slot empty so the router doesn't try to
        // route prefill traffic to them.
        let config = WorkerConfig::default();
        let engine_config = EngineConfig {
            model: "test/model".to_string(),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config, false)
            .await
            .unwrap();
        assert!(
            local_model
                .runtime_config()
                .disaggregated_endpoint
                .is_none()
        );
    }

    // -------------------------------------------------------------------
    // Lifecycle state machine tests
    // -------------------------------------------------------------------

    use crate::engine::PreprocessedRequest;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock engine that records `cleanup` calls and lets a test drive
    /// `start` success/failure via a flag.
    struct StateMockEngine {
        start_should_fail: bool,
        cleanup_calls: Arc<AtomicUsize>,
    }

    impl StateMockEngine {
        fn new(start_should_fail: bool) -> (Arc<Self>, Arc<AtomicUsize>) {
            let cleanup_calls = Arc::new(AtomicUsize::new(0));
            let eng = Arc::new(Self {
                start_should_fail,
                cleanup_calls: cleanup_calls.clone(),
            });
            (eng, cleanup_calls)
        }
    }

    #[async_trait]
    impl LLMEngine for StateMockEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            if self.start_should_fail {
                Err(err(
                    ErrorType::Backend(BackendError::EngineShutdown),
                    "synthetic start failure",
                ))
            } else {
                Ok(EngineConfig {
                    model: "mock".to_string(),
                    ..EngineConfig::default()
                })
            }
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<
            BoxStream<'static, Result<crate::engine::LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!("not used in state machine tests")
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn worker_with(engine: Arc<dyn LLMEngine>) -> Worker {
        Worker::new(engine, WorkerConfig::default())
    }

    /// Prefill worker — the drain loop only runs for prefill (the framework
    /// skips aggregated/decode), so drain-ordering tests must use this.
    fn worker_with_prefill(engine: Arc<dyn LLMEngine>) -> Worker {
        Worker::new(
            engine,
            WorkerConfig {
                disaggregation_mode: DisaggregationMode::Prefill,
                ..WorkerConfig::default()
            },
        )
    }

    #[tokio::test]
    async fn start_engine_init_to_running_on_success() {
        let (engine, _) = StateMockEngine::new(false);
        let mut worker = worker_with(engine);
        let cfg = worker.start_engine(0).await.expect("start");
        assert_eq!(cfg.model, "mock");
        assert_eq!(worker.state, LifecycleState::Running);
    }

    #[tokio::test]
    async fn start_engine_init_to_start_failed_on_failure() {
        let (engine, _) = StateMockEngine::new(true);
        let mut worker = worker_with(engine);
        let res = worker.start_engine(0).await;
        assert!(res.is_err(), "start should fail");
        // start() may have allocated partial state before raising; the
        // state machine keeps cleanup() owed by parking in StartFailed.
        assert_eq!(worker.state, LifecycleState::StartFailed);
    }

    #[tokio::test]
    async fn cleanup_once_runs_engine_cleanup_after_failed_start() {
        // Regression: previously, cleanup_once short-circuited on
        // `Stopped` after a failed start and engines were forced to
        // wrap their own start() in try/except to release partial
        // state. The state machine now owns the call.
        let (engine, cleanup_calls) = StateMockEngine::new(true);
        let mut worker = worker_with(engine);
        let _ = worker.start_engine(0).await; // intentional failure

        worker.cleanup_once().await;
        assert_eq!(
            cleanup_calls.load(Ordering::SeqCst),
            1,
            "engine.cleanup() must run exactly once after a failed start \
             so engines don't have to re-implement the guard"
        );
        assert_eq!(worker.state, LifecycleState::Stopped);

        // And still idempotent: a second call doesn't re-enter cleanup.
        worker.cleanup_once().await;
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
        assert_eq!(worker.state, LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn cleanup_once_is_idempotent() {
        let (engine, cleanup_calls) = StateMockEngine::new(false);
        let mut worker = worker_with(engine);
        worker.start_engine(0).await.unwrap();

        worker.cleanup_once().await;
        worker.cleanup_once().await;
        worker.cleanup_once().await;

        // engine.cleanup() runs at most once even though cleanup_once was
        // called three times — guards against the vLLM/TRT-LLM NCCL
        // double-teardown hang.
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
        assert_eq!(worker.state, LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn cleanup_once_noops_when_never_started() {
        let (engine, cleanup_calls) = StateMockEngine::new(false);
        let mut worker = worker_with(engine);
        // Pre-start signal path: cleanup before start completes.
        worker.cleanup_once().await;
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
        assert_eq!(worker.state, LifecycleState::Stopped);
    }

    // The pre-start shutdown path is handled in `run_inner` via a
    // `CancellationToken` cancellation check before `start_engine` is
    // called — not by flipping state to `Stopped` first. There is no
    // public path in the Worker that calls `start_engine` after state
    // was independently flipped to `Stopped`, so we don't test that
    // scenario at the state-machine level.

    // -------------------------------------------------------------------
    // Orchestrator step-ordering tests
    // -------------------------------------------------------------------

    use std::sync::Mutex as StdMutex;

    /// Serializes env-mutating drain tests in this module so cargo's parallel
    /// test threads don't race on `DYN_PREFILL_DRAIN_TIMEOUT_S`.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// Engine that records the order of `is_quiescent` and `cleanup` calls
    /// into a shared log so tests can assert on sequencing.
    struct OrderingMockEngine {
        log: Arc<StdMutex<Vec<&'static str>>>,
        is_quiescent_should_fail: bool,
    }

    impl OrderingMockEngine {
        fn new(is_quiescent_should_fail: bool) -> (Arc<Self>, Arc<StdMutex<Vec<&'static str>>>) {
            let log = Arc::new(StdMutex::new(Vec::new()));
            let eng = Arc::new(Self {
                log: log.clone(),
                is_quiescent_should_fail,
            });
            (eng, log)
        }
    }

    #[async_trait]
    impl LLMEngine for OrderingMockEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            self.log.lock().unwrap().push("start");
            Ok(EngineConfig {
                model: "mock".to_string(),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<
            BoxStream<'static, Result<crate::engine::LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!("not used in orchestrator tests")
        }

        async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
            self.log.lock().unwrap().push("is_quiescent");
            if self.is_quiescent_should_fail {
                Err(err(
                    ErrorType::Backend(BackendError::Unknown),
                    "synthetic is_quiescent failure",
                ))
            } else {
                Ok(Some(true))
            }
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            self.log.lock().unwrap().push("cleanup");
            Ok(())
        }
    }

    #[tokio::test]
    async fn shutdown_steps_run_drain_before_cleanup() {
        // Use the explicit-grace helper so we don't have to mutate the
        // process-global env var (which would race other parallel tests).
        let (engine, log) = OrderingMockEngine::new(false);
        let mut worker = worker_with_prefill(engine);
        worker.start_engine(0).await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["start", "is_quiescent", "cleanup"],
            "is_quiescent (drain) must run before cleanup"
        );
    }

    // ENV_LOCK must span the `.await` below: it serializes the env set/restore
    // window against other env-mutating drain tests, and the value must stay
    // pinned while the awaited drain loop reads it. No code reachable from the
    // await re-acquires ENV_LOCK, so there's no deadlock risk.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn shutdown_steps_drain_failure_does_not_block_cleanup() {
        // is_quiescent errors are treated as "not idle"; the drain loop keeps
        // polling until the budget expires, then cleanup still runs.
        let _guard = ENV_LOCK.lock().unwrap();
        let saved = std::env::var(DRAIN_TIMEOUT_ENV).ok();
        // SAFETY: tests in this mod serialize env access via ENV_LOCK.
        unsafe { std::env::set_var(DRAIN_TIMEOUT_ENV, "0") };

        let (engine, log) = OrderingMockEngine::new(true); // is_quiescent fails
        let mut worker = worker_with_prefill(engine);
        worker.start_engine(0).await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        // is_quiescent ran at least once (and errored), then cleanup ran.
        let recorded = log.lock().unwrap().clone();
        assert!(recorded.starts_with(&["start", "is_quiescent"]));
        assert_eq!(recorded.last().copied(), Some("cleanup"));
        assert_eq!(worker.state, LifecycleState::Stopped);

        // SAFETY: see above.
        unsafe {
            match saved {
                Some(v) => std::env::set_var(DRAIN_TIMEOUT_ENV, v),
                None => std::env::remove_var(DRAIN_TIMEOUT_ENV),
            }
        }
    }

    #[tokio::test]
    async fn shutdown_steps_skip_drain_for_non_prefill() {
        // Drain is prefill-only: an aggregated (or decode) worker must go
        // straight to cleanup without ever polling is_quiescent, regardless of
        // what the engine would report. Guards the mode-gate invariant that
        // makes the `Ok(None)` default safe (only prefill workers drain).
        let (engine, log) = OrderingMockEngine::new(false);
        let mut worker = worker_with(engine); // WorkerConfig::default() => Aggregated
        worker.start_engine(0).await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["start", "cleanup"],
            "non-prefill workers must not poll is_quiescent (no drain)"
        );
    }

    // The "drain skipped when engine never started" scenario isn't
    // reachable through the public `Worker::run` flow — pre-start
    // shutdown returns from `run_inner` before `serve_with_orchestrator`
    // (and therefore `run_engine_shutdown_steps`) ever runs. So we don't
    // pin a contract for run_engine_shutdown_steps in the Stopped state.

    #[test]
    fn grace_period_default_when_unset() {
        assert_eq!(grace_period_secs_from(None), DEFAULT_GRACE_PERIOD_SECS);
    }

    #[test]
    fn grace_period_parses_valid_value() {
        assert_eq!(grace_period_secs_from(Some("2.5")), 2.5);
    }

    #[test]
    fn grace_period_clamps_negative_to_zero() {
        assert_eq!(grace_period_secs_from(Some("-1")), 0.0);
    }

    #[test]
    fn grace_period_falls_back_to_default_on_parse_error() {
        assert_eq!(
            grace_period_secs_from(Some("not-a-number")),
            DEFAULT_GRACE_PERIOD_SECS
        );
    }

    #[test]
    fn grace_period_treats_empty_as_unset() {
        assert_eq!(grace_period_secs_from(Some("")), DEFAULT_GRACE_PERIOD_SECS);
    }

    // -------------------------------------------------------------------
    // load_health_check_payload_from_env
    // -------------------------------------------------------------------

    #[test]
    fn health_check_payload_env_returns_object() {
        let got = load_health_check_payload(Some(r#"{"token_ids":[1]}"#)).unwrap();
        assert_eq!(got["token_ids"], serde_json::json!([1]));
    }

    #[test]
    fn health_check_payload_env_rejects_non_object() {
        assert!(load_health_check_payload(Some("[1,2,3]")).is_none());
    }

    // -------------------------------------------------------------------
    // stamp_canary_marker
    // -------------------------------------------------------------------

    #[test]
    fn stamp_canary_marker_injects_into_object() {
        let stamped = stamp_canary_marker(serde_json::json!({"token_ids": [1]})).unwrap();
        assert_eq!(
            stamped[crate::engine::HEALTH_CHECK_KEY],
            serde_json::json!(true)
        );
        assert_eq!(stamped["token_ids"], serde_json::json!([1]));
    }

    #[test]
    fn stamp_canary_marker_rejects_non_object() {
        assert!(stamp_canary_marker(serde_json::json!([1, 2, 3])).is_none());
        assert!(stamp_canary_marker(serde_json::json!(42)).is_none());
    }

    #[test]
    fn stamp_canary_marker_overrides_falsy_marker() {
        // An operator can't disarm the marker by setting it false in their override.
        let stamped =
            stamp_canary_marker(serde_json::json!({crate::engine::HEALTH_CHECK_KEY: false}))
                .unwrap();
        assert_eq!(
            stamped[crate::engine::HEALTH_CHECK_KEY],
            serde_json::json!(true)
        );
    }

    // -------------------------------------------------------------------
    // graceful_shutdown_timeout env-var parsing
    // -------------------------------------------------------------------

    fn expected_default_timeout_secs() -> u64 {
        if cfg!(debug_assertions) {
            dynamo_runtime::worker::DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG
        } else {
            dynamo_runtime::worker::DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE
        }
    }

    #[test]
    fn shutdown_timeout_default_when_unset() {
        assert_eq!(
            graceful_shutdown_timeout_secs(None, expected_default_timeout_secs()),
            expected_default_timeout_secs()
        );
    }

    #[test]
    fn shutdown_timeout_parses_valid_value() {
        assert_eq!(
            graceful_shutdown_timeout_secs(Some("42"), expected_default_timeout_secs()),
            42
        );
    }

    #[test]
    fn shutdown_timeout_falls_back_to_default_on_parse_error() {
        assert_eq!(
            graceful_shutdown_timeout_secs(Some("not-a-number"), expected_default_timeout_secs()),
            expected_default_timeout_secs()
        );
    }

    #[test]
    fn shutdown_timeout_treats_empty_as_unset() {
        assert_eq!(
            graceful_shutdown_timeout_secs(Some(""), expected_default_timeout_secs()),
            expected_default_timeout_secs()
        );
    }

    // -------------------------------------------------------------------
    // shutdown_deadline composition + budget interaction with the grace
    // sleep. Regression coverage for the bug where deadline == timeout
    // and grace == timeout (the debug default) starves drain + cleanup.
    // -------------------------------------------------------------------

    #[test]
    fn shutdown_deadline_adds_grace_to_timeout() {
        assert_eq!(
            shutdown_deadline(Duration::from_secs(5), 5.0),
            Duration::from_secs(10)
        );
        assert_eq!(
            shutdown_deadline(Duration::from_secs(30), 0.0),
            Duration::from_secs(30)
        );
        assert_eq!(
            shutdown_deadline(Duration::from_secs(2), 0.5),
            Duration::from_millis(2_500)
        );
    }

    #[test]
    fn shutdown_deadline_clamps_negative_grace() {
        assert_eq!(
            shutdown_deadline(Duration::from_secs(5), -1.0),
            Duration::from_secs(5)
        );
    }

    /// Regression: with the buggy deadline (timeout only, no grace
    /// reserve), a grace period at or above the timeout consumes the
    /// whole budget and drain + cleanup never get scheduled. This is
    /// the default-env debug failure mode — DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG
    /// (5) equals DEFAULT_GRACE_PERIOD_SECS (5.0), and the unregister
    /// network call (~ms-scale) tips sleep past the deadline. We use
    /// grace > timeout to model that real-world latency deterministically
    /// in virtual time.
    #[tokio::test(start_paused = true)]
    async fn timeout_alone_starves_drain_cleanup_when_grace_meets_timeout() {
        let (engine, log) = OrderingMockEngine::new(false);
        let mut worker = worker_with(engine);
        worker.start_engine(0).await.unwrap();

        let timeout = Duration::from_secs(5);
        let grace = 5.1;

        // The pre-fix deadline (timeout, no grace reserve).
        let result =
            tokio::time::timeout(timeout, worker.run_engine_shutdown_steps_with_grace(grace)).await;
        assert!(
            result.is_err(),
            "buggy deadline must expire before drain/cleanup run"
        );

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["start"],
            "drain and cleanup must not have been observed"
        );
    }

    /// The fix: deadline = timeout + grace. Same scenario as above —
    /// grace exceeding the raw timeout — but drain and cleanup now both
    /// complete because the grace sleep is reserved on top of the
    /// timeout budget.
    #[tokio::test(start_paused = true)]
    async fn shutdown_deadline_reserves_grace_so_drain_cleanup_complete() {
        let (engine, log) = OrderingMockEngine::new(false);
        let mut worker = worker_with_prefill(engine);
        worker.start_engine(0).await.unwrap();

        let timeout = Duration::from_secs(5);
        let grace = 5.1;

        let deadline = shutdown_deadline(timeout, grace);
        let result =
            tokio::time::timeout(deadline, worker.run_engine_shutdown_steps_with_grace(grace))
                .await;
        assert!(
            result.is_ok(),
            "fixed deadline must allow drain + cleanup to finish"
        );

        let recorded = log.lock().unwrap().clone();
        assert_eq!(recorded, vec!["start", "is_quiescent", "cleanup"]);
    }

    // -------------------------------------------------------------------
    // RuntimeConfig env application
    // -------------------------------------------------------------------

    #[test]
    fn runtime_config_apply_to_env_writes_set_fields() {
        let cfg = RuntimeConfig {
            discovery_backend: Some("file".to_string()),
            request_plane: Some("tcp".to_string()),
            event_plane: Some("zmq".to_string()),
        };

        let mut applied = Vec::new();
        cfg.apply_with(|key, value| applied.push((key.to_string(), value.to_string())));
        assert_eq!(
            applied,
            vec![
                ("DYN_DISCOVERY_BACKEND".to_string(), "file".to_string()),
                ("DYN_REQUEST_PLANE".to_string(), "tcp".to_string()),
                ("DYN_EVENT_PLANE".to_string(), "zmq".to_string()),
            ]
        );
    }

    #[test]
    fn runtime_config_apply_to_env_leaves_unset_fields_untouched() {
        let cfg = RuntimeConfig {
            discovery_backend: Some("etcd".to_string()),
            request_plane: None,
            event_plane: None,
        };

        let mut applied = Vec::new();
        cfg.apply_with(|key, value| applied.push((key.to_string(), value.to_string())));
        assert_eq!(
            applied,
            vec![("DYN_DISCOVERY_BACKEND".to_string(), "etcd".to_string())]
        );
    }
}

// Integration tests for the `on_endpoint_ready` handoff. These need a real
// `DistributedRuntime`/`Endpoint` (NATS-backed), so they live behind the
// `integration` feature:
//   cargo test -p dynamo-backend-common --features integration on_endpoint_ready
#[cfg(all(test, feature = "integration"))]
mod handoff_integration_tests {
    use super::*;
    use crate::engine::PreprocessedRequest;
    use async_trait::async_trait;
    use dynamo_runtime::distributed_test_utils::create_test_drt_async;
    use futures::stream::BoxStream;
    use std::sync::Mutex as StdMutex;

    /// Build a real serving `Endpoint` from a test DRT, mirroring how
    /// `run_inner` resolves namespace → component → endpoint.
    async fn test_endpoint() -> dynamo_runtime::component::Endpoint {
        let drt = create_test_drt_async().await;
        drt.namespace("handoff_ns")
            .unwrap()
            .component("handoff_comp")
            .unwrap()
            .endpoint("generate")
    }

    /// Mock engine that records the order of `on_endpoint_ready`,
    /// `supported_controls`, and `supported_updates` calls, lets a test force
    /// `on_endpoint_ready` to fail, and advertises configurable control/update
    /// sets.
    struct HandoffMockEngine {
        log: Arc<StdMutex<Vec<&'static str>>>,
        endpoint_ready_should_fail: bool,
        controls: Vec<String>,
        updates: Vec<String>,
    }

    impl HandoffMockEngine {
        fn new(
            endpoint_ready_should_fail: bool,
            controls: Vec<String>,
            updates: Vec<String>,
        ) -> (Arc<Self>, Arc<StdMutex<Vec<&'static str>>>) {
            let log = Arc::new(StdMutex::new(Vec::new()));
            let eng = Arc::new(Self {
                log: log.clone(),
                endpoint_ready_should_fail,
                controls,
                updates,
            });
            (eng, log)
        }
    }

    #[async_trait]
    impl LLMEngine for HandoffMockEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig {
                model: "mock".to_string(),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<
            BoxStream<'static, Result<crate::engine::LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!("not used in handoff tests")
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }

        async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
            self.log.lock().unwrap().push("supported_controls");
            Ok(self.controls.clone())
        }

        async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
            self.log.lock().unwrap().push("supported_updates");
            Ok(self.updates.clone())
        }

        async fn on_endpoint_ready(
            &self,
            _endpoint: dynamo_runtime::component::Endpoint,
        ) -> Result<(), DynamoError> {
            self.log.lock().unwrap().push("on_endpoint_ready");
            if self.endpoint_ready_should_fail {
                Err(err(
                    ErrorType::Backend(BackendError::Unknown),
                    "synthetic on_endpoint_ready failure",
                ))
            } else {
                Ok(())
            }
        }
    }

    /// Engine that overrides only the required methods, so it inherits the
    /// trait-default `on_endpoint_ready` / `supported_controls`.
    struct DefaultsEngine;

    #[async_trait]
    impl LLMEngine for DefaultsEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: crate::engine::GenerateContext,
        ) -> Result<
            BoxStream<'static, Result<crate::engine::LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!("not used in handoff tests")
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    /// The trait default `on_endpoint_ready` is a no-op that succeeds against a
    /// real `Endpoint`.
    #[tokio::test]
    async fn default_on_endpoint_ready_is_noop() {
        let endpoint = test_endpoint().await;
        let engine = Arc::new(DefaultsEngine);
        engine
            .on_endpoint_ready(endpoint)
            .await
            .expect("default on_endpoint_ready must succeed");
    }

    /// `serve_with_orchestrator` runs `on_endpoint_ready` before
    /// `register_engine_controls` and `register_engine_updates`. Drive the same
    /// three production calls in that order and assert: (1) the handoff is
    /// observed before the engine is asked for its controls/updates, and (2) the
    /// advertised control lands under `control/<name>` and the advertised update
    /// under `update/<name>` in the DRT's engine-route registry, so
    /// `/engine/control/<name>` and `/engine/update/<name>` become routable.
    #[tokio::test]
    async fn handoff_precedes_registration_and_populates_namespaced_registry() {
        let endpoint = test_endpoint().await;
        let (engine, log) = HandoffMockEngine::new(
            false,
            vec!["start_profile".to_string()],
            vec!["load_lora".to_string()],
        );
        let worker = Worker::new(engine, WorkerConfig::default());

        // Mirror serve_with_orchestrator's handoff + registration calls exactly.
        worker
            .engine
            .on_endpoint_ready(endpoint.clone())
            .await
            .expect("handoff should succeed");
        worker
            .register_engine_controls(&endpoint)
            .await
            .expect("control registration should succeed");
        worker
            .register_engine_updates(&endpoint)
            .await
            .expect("update registration should succeed");

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![
                "on_endpoint_ready",
                "supported_controls",
                "supported_updates"
            ],
            "endpoint handoff must happen before controls/updates are enumerated/registered"
        );
        let routes = endpoint.drt().engine_routes();
        assert!(
            routes.get("control/start_profile").is_some(),
            "advertised control must be registered under control/<name>"
        );
        assert!(
            routes.get("update/load_lora").is_some(),
            "advertised update must be registered under update/<name>"
        );
        // Bare (unprefixed) keys must NOT be registered by the unified Worker.
        assert!(
            routes.get("start_profile").is_none(),
            "control must not be registered under its bare name"
        );
        assert!(
            routes.get("load_lora").is_none(),
            "update must not be registered under its bare name"
        );
    }

    /// A failing `on_endpoint_ready` aborts startup: the `?` in
    /// `serve_with_orchestrator` propagates the error before
    /// `register_engine_controls`/`register_engine_updates` run, so nothing is
    /// registered.
    #[tokio::test]
    async fn failed_handoff_is_fatal_and_skips_registration() {
        let endpoint = test_endpoint().await;
        let (engine, log) = HandoffMockEngine::new(
            true,
            vec!["start_profile".to_string()],
            vec!["load_lora".to_string()],
        );
        let worker = Worker::new(engine, WorkerConfig::default());

        let result = worker.engine.on_endpoint_ready(endpoint.clone()).await;
        assert!(result.is_err(), "failed handoff must surface as an error");

        // Production code returns here via `?`; we do NOT call
        // register_engine_controls/register_engine_updates. Confirm nothing
        // was registered.
        let recorded = log.lock().unwrap().clone();
        assert_eq!(recorded, vec!["on_endpoint_ready"]);
        let routes = endpoint.drt().engine_routes();
        assert!(
            routes.get("control/start_profile").is_none(),
            "no controls should be registered after a fatal handoff"
        );
        assert!(
            routes.get("update/load_lora").is_none(),
            "no updates should be registered after a fatal handoff"
        );
    }
}
