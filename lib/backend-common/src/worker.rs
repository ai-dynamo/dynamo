// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Worker` — runtime lifecycle driver for an [`LLMEngine`].
//!
//! Creates the `DistributedRuntime`, starts the engine, registers the
//! model, serves the endpoint, and runs cleanup on shutdown. Non-generic
//! over the engine type so a PyO3-wrapped engine (phase 2) can feed in
//! through the same `Arc<dyn LLMEngine>` path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::local_model::LocalModel;
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_llm::local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig};
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{DistributedRuntime, Runtime};
use tokio_util::sync::CancellationToken;

use crate::adapter::EngineAdapter;
use crate::disagg::DisaggregationMode;
use crate::engine::{EngineConfig, LLMEngine};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::publisher::{PublisherHandles, setup_publishers};

/// Default grace-period in seconds between discovery unregister and engine drain.
/// Mirrors the Python `_DEFAULT_GRACE_PERIOD_SECS` constant.
const DEFAULT_GRACE_PERIOD_SECS: f64 = 5.0;

/// Environment variable name for overriding the grace-period.
/// Shared with the Python helper so a single env var controls both.
const GRACE_PERIOD_ENV: &str = "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS";

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
    /// Request-plane transport — e.g. `"tcp"`, `"nats"`, `"http"`. Maps to
    /// `DYN_REQUEST_PLANE`.
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
            if let Some(ref v) = self.discovery_backend {
                std::env::set_var("DYN_DISCOVERY_BACKEND", v);
            }
            if let Some(ref v) = self.request_plane {
                std::env::set_var("DYN_REQUEST_PLANE", v);
            }
            if let Some(ref v) = self.event_plane {
                std::env::set_var("DYN_EVENT_PLANE", v);
            }
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
    /// `engine.kv_event_sources()` / `metrics_sources()` entirely.
    pub enable_kv_routing: bool,
    /// Per-endpoint Prometheus metric labels appended to every metric.
    /// Common labels: `("model", "<served-name>")`.
    pub metrics_labels: Vec<(String, String)>,
    /// Disaggregation role for this worker.
    ///
    /// `Aggregated` (default) registers the model with the parsed
    /// `endpoint_types`. `Prefill` registers with `ModelType::Prefill` so the
    /// frontend's prefill router targets it. `Decode` keeps `endpoint_types`
    /// but force-disables the local KV indexer because decode workers do not
    /// host the indexer endpoint.
    pub disaggregation_mode: DisaggregationMode,
    /// Runtime / transport overrides applied via env vars before the
    /// `DistributedRuntime` is constructed.
    pub runtime: RuntimeConfig,
}

impl WorkerConfig {
    /// Effective `enable_local_indexer`, accounting for disaggregation
    /// mode. Decode workers force this off because they don't host the
    /// in-process KV indexer endpoint and must not advertise it.
    pub(crate) fn effective_enable_local_indexer(&self) -> bool {
        self.enable_local_indexer && !self.disaggregation_mode.is_decode()
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
            runtime: RuntimeConfig::default(),
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

/// Runtime host for an [`LLMEngine`].
///
/// `run()` creates the distributed runtime, calls `engine.start()`,
/// registers the model, serves the endpoint, and calls
/// `engine.cleanup()` on shutdown (guaranteed once `start()` succeeded).
pub struct Worker {
    engine: Arc<dyn LLMEngine>,
    config: WorkerConfig,
    state: LifecycleState,
    /// KV-aware-routing publisher handles. Drained in `cleanup_once` while NATS is alive.
    publishers: Option<PublisherHandles>,
}

impl Worker {
    pub fn new(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> Self {
        Self {
            engine,
            config,
            state: LifecycleState::Init,
            publishers: None,
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
    ///   3. `engine.drain()` — backend-side drain (e.g. NIXL prefill).
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
        validate_model_input(self.config.model_input)?;

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
        let engine_config = self.start_engine(worker_id).await?;
        tracing::debug!(
            model = %engine_config.model,
            worker_id,
            "engine.start() complete"
        );

        self.setup_kv_aware_publishers(&component, &engine_config)
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

    /// Build KV-event and worker-metric publishers from the engine's source
    /// declarations. No-op if `enable_kv_routing` is off, the engine declares
    /// no sources, or `engine_config.kv_cache_block_size` is unset.
    async fn setup_kv_aware_publishers(
        &mut self,
        component: &dynamo_runtime::component::Component,
        engine_config: &EngineConfig,
    ) -> Result<(), DynamoError> {
        if !self.config.enable_kv_routing {
            tracing::debug!("enable_kv_routing=false; skipping kv_event_sources / metrics_sources");
            return Ok(());
        }
        let kv_sources = self.engine.kv_event_sources().await?;
        let metrics_snapshots = self.engine.metrics_sources().await?;
        if kv_sources.is_empty() && metrics_snapshots.is_empty() {
            tracing::debug!(
                "engine returned no KV/metrics sources; KV-aware routing disabled for this worker"
            );
            return Ok(());
        }
        let enable_local_indexer = self.config.effective_enable_local_indexer();
        tracing::debug!(
            kv_sources = kv_sources.len(),
            metrics_sources = metrics_snapshots.len(),
            enable_local_indexer,
            kv_cache_block_size = ?engine_config.kv_cache_block_size,
            "Starting KV-aware-routing publishers"
        );
        let handles = setup_publishers(
            component,
            kv_sources,
            metrics_snapshots,
            engine_config.kv_cache_block_size,
            enable_local_indexer,
        )
        .await?;
        self.publishers = Some(handles);
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
        match self.engine.cleanup().await {
            Ok(()) => tracing::info!("Engine cleanup complete"),
            Err(e) => tracing::error!(error = %e, "engine cleanup failed"),
        }
        // Stop publisher metric loops AFTER engine.cleanup so the engine's
        // last metric snapshots get published. Then drop the handles so the
        // publishers' own tokio tasks drain while NATS is still alive.
        if let Some(mut handles) = self.publishers.take() {
            handles.shutdown().await;
        }
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

        let mut local_model = build_local_model(&self.config, engine_config).await?;
        tracing::debug!("local model built");
        local_model
            .attach(&endpoint, model_type, self.config.model_input, None)
            .await
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("model attach: {e}"),
                )
            })?;
        tracing::debug!("model registered with discovery");

        let served = resolve_served_name(&self.config, engine_config)
            .unwrap_or_else(|| engine_config.model.clone());
        tracing::info!(
            "Serving {} on {}.{}.{}",
            served,
            self.config.namespace,
            self.config.component,
            self.config.endpoint
        );

        let ingress = Ingress::for_engine(Arc::new(EngineAdapter::new(
            self.engine.clone(),
            self.config.disaggregation_mode,
        )))
        .map_err(|e| {
            err(
                ErrorType::Backend(BackendError::Unknown),
                format!("ingress: {e}"),
            )
        })?;

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

        let serve_fut = endpoint
            .endpoint_builder()
            .handler(ingress)
            .metrics_labels(metrics_labels)
            .graceful_shutdown(true)
            .start();
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

    /// Engine-facing shutdown sequence: grace period sleep → `engine.drain()`
    /// → `cleanup_once()`. Each engine step swallows non-fatal failures so a
    /// misbehaving engine can't block the worker from exiting.
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

        if let Err(e) = self.engine.drain().await {
            tracing::warn!(error = %e, "engine drain failed");
        }

        self.cleanup_once().await;
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

    let secs = std::env::var(env_worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default);
    Duration::from_secs(secs)
}

/// Compose the post-signal shutdown deadline from the drain+cleanup
/// timeout and the grace-period sleep that precedes them.
///
/// The grace sleep is a fixed wait (not a hang risk), so reserving its
/// duration on top of `timeout` ensures `engine.drain()` and
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

/// Read the grace-period seconds from `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`,
/// matching the Python helper. Negative values clamp to 0.
fn grace_period_secs() -> f64 {
    match std::env::var(GRACE_PERIOD_ENV) {
        Ok(s) if !s.is_empty() => match s.parse::<f64>() {
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
/// role. `DisaggregationMode::Prefill` short-circuits to `ModelType::Prefill`
/// regardless of `endpoint_types`; everything else falls back to the parsed
/// `endpoint_types` so existing callers see no change.
fn resolve_model_type(config: &WorkerConfig) -> Result<ModelType, DynamoError> {
    if config.disaggregation_mode.is_prefill() {
        return Ok(ModelType::Prefill);
    }
    parse_endpoint_types(&config.endpoint_types)
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
            "prefill" => ModelType::Prefill,
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

fn validate_model_input(model_input: ModelInput) -> Result<(), DynamoError> {
    if model_input == ModelInput::Tokens {
        return Ok(());
    }

    Err(err(
        ErrorType::Backend(BackendError::InvalidArgument),
        format!(
            "dynamo_backend_common::Worker currently supports only ModelInput::Tokens; got '{}'. \
             ModelInput::Text and ModelInput::Tensor require dedicated raw-request adapters.",
            model_input.as_str()
        ),
    ))
}

async fn build_local_model(
    config: &WorkerConfig,
    engine_config: &EngineConfig,
) -> Result<LocalModel, DynamoError> {
    let served_name = resolve_served_name(config, engine_config)
        .or_else(|| Some(engine_config.model.clone()))
        .filter(|s| !s.is_empty());

    // Decode workers don't host the WorkerKvQuery endpoint, so they must not
    // advertise the local indexer regardless of the operator-supplied flag.
    // Mirrors the legacy non-unified vLLM path (worker_factory.py).
    let enable_local_indexer = config.effective_enable_local_indexer();

    // Publish the disaggregated bootstrap endpoint when the engine
    // returned one. Only meaningful for prefill workers — decode/agg
    // engines leave both fields `None`. The frontend's `PrefillRouter`
    // reads this from `model_manager.get_disaggregated_endpoint(...)` to
    // take its optimised "Bootstrap path" (route decode concurrent with
    // prefill instead of waiting for prefill to drain).
    let disaggregated_endpoint = match (&engine_config.bootstrap_host, engine_config.bootstrap_port)
    {
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

    let rt_cfg = ModelRuntimeConfig {
        total_kv_blocks: engine_config.total_kv_blocks,
        max_num_seqs: engine_config.max_num_seqs,
        max_num_batched_tokens: engine_config.max_num_batched_tokens,
        data_parallel_size: engine_config.data_parallel_size.unwrap_or(1),
        data_parallel_start_rank: engine_config.data_parallel_start_rank.unwrap_or(0),
        tool_call_parser: config.tool_call_parser.clone(),
        reasoning_parser: config.reasoning_parser.clone(),
        exclude_tools_when_tool_choice_none: config.exclude_tools_when_tool_choice_none,
        enable_local_indexer,
        disaggregated_endpoint,
        runtime_data: engine_config.runtime_data.clone(),
        ..ModelRuntimeConfig::default()
    };

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(served_name)
        .context_length(engine_config.context_length)
        .kv_cache_block_size(engine_config.kv_cache_block_size)
        .custom_template_path(config.custom_jinja_template.clone())
        .runtime_config(rt_cfg);

    // Resolve WorkerConfig.model_name into a local path. Empty string means
    // name-only mode (no tokenizer / chat template on the card).
    if !config.model_name.is_empty() {
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

    #[test]
    fn validate_model_input_accepts_tokens() {
        validate_model_input(ModelInput::Tokens).unwrap();
    }

    #[test]
    fn validate_model_input_rejects_text_and_tensor() {
        for input in [ModelInput::Text, ModelInput::Tensor] {
            let e = validate_model_input(input).unwrap_err();
            assert_eq!(
                e.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(e.to_string().contains(input.as_str()));
        }
    }

    #[tokio::test]
    async fn build_local_model_carries_runtime_parser_settings() {
        let config = WorkerConfig {
            tool_call_parser: Some("kimi_k2".to_string()),
            reasoning_parser: Some("kimi_k25".to_string()),
            exclude_tools_when_tool_choice_none: false,
            enable_local_indexer: false,
            ..WorkerConfig::default()
        };
        let engine_config = EngineConfig {
            model: "nvidia/Kimi-K2.5-NVFP4".to_string(),
            total_kv_blocks: Some(100),
            max_num_seqs: Some(16),
            max_num_batched_tokens: Some(8192),
            runtime_data: [(
                "sglang_worker_group_id".to_string(),
                serde_json::json!("group-a"),
            )]
            .into(),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config).await.unwrap();
        let runtime_config = local_model.runtime_config();

        assert_eq!(runtime_config.total_kv_blocks, Some(100));
        assert_eq!(runtime_config.max_num_seqs, Some(16));
        assert_eq!(runtime_config.max_num_batched_tokens, Some(8192));
        assert_eq!(runtime_config.tool_call_parser.as_deref(), Some("kimi_k2"));
        assert_eq!(runtime_config.reasoning_parser.as_deref(), Some("kimi_k25"));
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
        // prefill workers short-circuit to ModelType::Prefill.
        let config = WorkerConfig {
            endpoint_types: "chat".to_string(),
            disaggregation_mode: DisaggregationMode::Decode,
            ..WorkerConfig::default()
        };
        assert_eq!(resolve_model_type(&config).unwrap(), ModelType::Chat);
    }

    #[test]
    fn resolve_model_type_prefill_overrides_endpoint_types() {
        // The operator may have left endpoint_types at the default
        // "chat,completions"; --disaggregation-mode prefill forces the
        // registration to ModelType::Prefill regardless.
        let config = WorkerConfig {
            endpoint_types: "chat,completions".to_string(),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..WorkerConfig::default()
        };
        assert_eq!(resolve_model_type(&config).unwrap(), ModelType::Prefill);
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

        let local_model = build_local_model(&config, &engine_config).await.unwrap();
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

        let local_model = build_local_model(&config, &engine_config).await.unwrap();
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
            bootstrap_host: Some("10.0.0.5".to_string()),
            bootstrap_port: Some(12345),
            ..EngineConfig::default()
        };

        let local_model = build_local_model(&config, &engine_config).await.unwrap();
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

        let local_model = build_local_model(&config, &engine_config).await.unwrap();
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

    /// Engine that records the order of `drain` and `cleanup` calls into a
    /// shared log so tests can assert on sequencing.
    struct OrderingMockEngine {
        log: Arc<StdMutex<Vec<&'static str>>>,
        drain_should_fail: bool,
    }

    impl OrderingMockEngine {
        fn new(drain_should_fail: bool) -> (Arc<Self>, Arc<StdMutex<Vec<&'static str>>>) {
            let log = Arc::new(StdMutex::new(Vec::new()));
            let eng = Arc::new(Self {
                log: log.clone(),
                drain_should_fail,
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

        async fn drain(&self) -> Result<(), DynamoError> {
            self.log.lock().unwrap().push("drain");
            if self.drain_should_fail {
                Err(err(
                    ErrorType::Backend(BackendError::Unknown),
                    "synthetic drain failure",
                ))
            } else {
                Ok(())
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
        let mut worker = worker_with(engine);
        worker.start_engine(0).await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["start", "drain", "cleanup"],
            "drain must run before cleanup"
        );
    }

    #[tokio::test]
    async fn shutdown_steps_drain_failure_does_not_block_cleanup() {
        let (engine, log) = OrderingMockEngine::new(true); // drain fails
        let mut worker = worker_with(engine);
        worker.start_engine(0).await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        // Drain ran (and failed), but cleanup still ran exactly once.
        let recorded = log.lock().unwrap().clone();
        assert_eq!(recorded, vec!["start", "drain", "cleanup"]);
        assert_eq!(worker.state, LifecycleState::Stopped);
    }

    // The "drain skipped when engine never started" scenario isn't
    // reachable through the public `Worker::run` flow — pre-start
    // shutdown returns from `run_inner` before `serve_with_orchestrator`
    // (and therefore `run_engine_shutdown_steps`) ever runs. So we don't
    // pin a contract for run_engine_shutdown_steps in the Stopped state.

    // -------------------------------------------------------------------
    // grace_period_secs env-var parsing
    // -------------------------------------------------------------------
    //
    // These tests mutate process-wide environment state. tokio::test
    // marks them async (each runs on its own current-thread runtime) but
    // they are still serialized by `serial_test`-style discipline within
    // the test name space — keep them in this single mod and access the
    // env var only here.
    //
    // `ENV_LOCK` serializes all env-mutating tests in this module so cargo's
    // parallel runner can't interleave a `with_env` setup on one thread with
    // a read on another. Every helper that touches `std::env` acquires this
    // lock for the duration of its critical section.

    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce() -> R, R>(key: &str, value: Option<&str>, f: F) -> R {
        // Hold the lock for the entire snapshot → set → run → restore
        // window so concurrent tests can't observe our temporary value or
        // race the restore.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(key).ok();
        // SAFETY: ENV_LOCK serializes all env-mutating tests in this
        // module; no other test thread reads or writes env state while
        // this guard is held.
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        let out = f();
        unsafe {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        out
    }

    #[test]
    fn grace_period_default_when_unset() {
        with_env(GRACE_PERIOD_ENV, None, || {
            assert_eq!(grace_period_secs(), DEFAULT_GRACE_PERIOD_SECS);
        });
    }

    #[test]
    fn grace_period_parses_valid_value() {
        with_env(GRACE_PERIOD_ENV, Some("2.5"), || {
            assert_eq!(grace_period_secs(), 2.5);
        });
    }

    #[test]
    fn grace_period_clamps_negative_to_zero() {
        with_env(GRACE_PERIOD_ENV, Some("-1"), || {
            assert_eq!(grace_period_secs(), 0.0);
        });
    }

    #[test]
    fn grace_period_falls_back_to_default_on_parse_error() {
        with_env(GRACE_PERIOD_ENV, Some("not-a-number"), || {
            assert_eq!(grace_period_secs(), DEFAULT_GRACE_PERIOD_SECS);
        });
    }

    #[test]
    fn grace_period_treats_empty_as_unset() {
        with_env(GRACE_PERIOD_ENV, Some(""), || {
            assert_eq!(grace_period_secs(), DEFAULT_GRACE_PERIOD_SECS);
        });
    }

    // -------------------------------------------------------------------
    // graceful_shutdown_timeout env-var parsing
    // -------------------------------------------------------------------

    // Reference the same upstream constant the production code reads, so
    // a rename of the env var in `dynamo-runtime` doesn't silently leave
    // these tests pointing at a no-longer-honored name.
    const SHUTDOWN_TIMEOUT_ENV: &str =
        dynamo_runtime::config::environment_names::worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT;

    fn expected_default_timeout_secs() -> u64 {
        if cfg!(debug_assertions) {
            dynamo_runtime::worker::DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_DEBUG
        } else {
            dynamo_runtime::worker::DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_RELEASE
        }
    }

    #[test]
    fn shutdown_timeout_default_when_unset() {
        with_env(SHUTDOWN_TIMEOUT_ENV, None, || {
            assert_eq!(
                graceful_shutdown_timeout(),
                Duration::from_secs(expected_default_timeout_secs())
            );
        });
    }

    #[test]
    fn shutdown_timeout_parses_valid_value() {
        with_env(SHUTDOWN_TIMEOUT_ENV, Some("42"), || {
            assert_eq!(graceful_shutdown_timeout(), Duration::from_secs(42));
        });
    }

    #[test]
    fn shutdown_timeout_falls_back_to_default_on_parse_error() {
        with_env(SHUTDOWN_TIMEOUT_ENV, Some("not-a-number"), || {
            assert_eq!(
                graceful_shutdown_timeout(),
                Duration::from_secs(expected_default_timeout_secs())
            );
        });
    }

    #[test]
    fn shutdown_timeout_treats_empty_as_unset() {
        with_env(SHUTDOWN_TIMEOUT_ENV, Some(""), || {
            assert_eq!(
                graceful_shutdown_timeout(),
                Duration::from_secs(expected_default_timeout_secs())
            );
        });
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
        let mut worker = worker_with(engine);
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
        assert_eq!(recorded, vec!["start", "drain", "cleanup"]);
    }

    // -------------------------------------------------------------------
    // RuntimeConfig env application
    //
    // These tests touch DYN_DISCOVERY_BACKEND / DYN_REQUEST_PLANE /
    // DYN_EVENT_PLANE directly (without `with_env`), so they must
    // acquire `ENV_LOCK` themselves to keep parallel runs from racing
    // each other or the `with_env`-using tests above.
    // -------------------------------------------------------------------

    #[test]
    fn runtime_config_apply_to_env_writes_set_fields() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let cfg = RuntimeConfig {
            discovery_backend: Some("file".to_string()),
            request_plane: Some("tcp".to_string()),
            event_plane: Some("zmq".to_string()),
        };

        // Snapshot prior values so we don't leak state to other tests.
        let prev: Vec<_> = [
            "DYN_DISCOVERY_BACKEND",
            "DYN_REQUEST_PLANE",
            "DYN_EVENT_PLANE",
        ]
        .iter()
        .map(|k| (*k, std::env::var(k).ok()))
        .collect();

        cfg.apply_to_env();
        assert_eq!(std::env::var("DYN_DISCOVERY_BACKEND").unwrap(), "file");
        assert_eq!(std::env::var("DYN_REQUEST_PLANE").unwrap(), "tcp");
        assert_eq!(std::env::var("DYN_EVENT_PLANE").unwrap(), "zmq");

        for (k, v) in prev {
            unsafe {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    #[test]
    fn runtime_config_apply_to_env_leaves_unset_fields_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let key = "DYN_REQUEST_PLANE";
        let prev = std::env::var(key).ok();
        unsafe { std::env::set_var(key, "preexisting") };

        let cfg = RuntimeConfig {
            discovery_backend: Some("etcd".to_string()),
            request_plane: None,
            event_plane: None,
        };
        cfg.apply_to_env();

        // None field must not overwrite an existing value.
        assert_eq!(std::env::var(key).unwrap(), "preexisting");

        unsafe {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
