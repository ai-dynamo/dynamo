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
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::{DistributedRuntime, Runtime};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::adapter::EngineAdapter;
use crate::engine::{EngineConfig, LLMEngine};
use crate::error::{BackendError, DynamoError, ErrorType};

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
    /// Per-endpoint Prometheus metric labels appended to every metric.
    /// Common labels: `("model", "<served-name>")`.
    pub metrics_labels: Vec<(String, String)>,
    /// Runtime / transport overrides applied via env vars before the
    /// `DistributedRuntime` is constructed.
    pub runtime: RuntimeConfig,
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
            metrics_labels: Vec::new(),
            runtime: RuntimeConfig::default(),
        }
    }
}

/// Lifecycle state for [`Worker`].
///
/// Only three states are observable to callers because the lifecycle
/// `tokio::Mutex` serializes start and cleanup — `Starting` / `Stopping`
/// transitions live entirely inside the lock holder and never escape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LifecycleState {
    /// `start_engine` has not been called (or shutdown arrived first and
    /// flipped us straight to `Stopped`).
    Init,
    /// `engine.start()` returned successfully; `engine.cleanup()` is owed.
    Running,
    /// Cleanup done, never started, or start failed. `engine.cleanup()`
    /// will not be called again.
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
    state: Mutex<LifecycleState>,
}

impl Worker {
    pub fn new(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> Self {
        Self {
            engine,
            config,
            state: Mutex::new(LifecycleState::Init),
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
    ///      Phase 1/2/3 token-cancellation teardown.
    ///
    /// A SIGTERM/SIGINT listener is installed at the top of `run` and
    /// shared via a [`CancellationToken`]:
    ///   * Pre-start signal (during `DistributedRuntime` construction):
    ///     the post-DRT cancellation check returns `EngineShutdown` and
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
    pub async fn run(self, runtime: Runtime) -> Result<(), DynamoError> {
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
        // background task only flips the token — it intentionally does
        // not touch the lifecycle mutex, so `start_engine` cannot deadlock
        // on a signal-handler holding the lock.
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
        // within `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` seconds, otherwise
        // we exit(911). Healthy long-running workers never hit this — the
        // timer only starts after `shutdown_token` is cancelled.
        let inner_fut = self.run_inner(runtime, &shutdown_token);
        tokio::pin!(inner_fut);

        let outcome = tokio::select! {
            result = &mut inner_fut => result,
            _ = shutdown_token.cancelled() => {
                let timeout = graceful_shutdown_timeout();
                tracing::debug!(
                    "graceful shutdown started; deadline {}s",
                    timeout.as_secs()
                );
                match tokio::time::timeout(timeout, &mut inner_fut).await {
                    Ok(result) => result,
                    Err(_) => {
                        tracing::error!(
                            "Graceful shutdown exceeded {}s; force-exiting with code 911. \
                             Set DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT to override.",
                            timeout.as_secs()
                        );
                        std::process::exit(911);
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
        &self,
        runtime: Runtime,
        shutdown: &CancellationToken,
    ) -> Result<(), DynamoError> {
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

        // Pre-start: bail before doing real engine work if shutdown already
        // arrived during DRT construction.
        if shutdown.is_cancelled() {
            return Err(err(
                ErrorType::Backend(BackendError::EngineShutdown),
                "shutdown requested before engine start",
            ));
        }

        let engine_config = self.start_engine().await?;
        tracing::debug!(model = %engine_config.model, "engine.start() complete");

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

    /// Full graceful-shutdown orchestrator: discovery unregister →
    /// grace period → drain → cleanup. Shared by every shutdown path —
    /// pre-serve (mid-start signal) and the serve loop's signal arm.
    async fn orchestrator_steps(&self, endpoint: &dynamo_runtime::component::Endpoint) {
        if let Err(e) = endpoint.unregister_endpoint_instance().await {
            tracing::warn!(error = %e, "discovery unregister failed");
        } else {
            tracing::info!("Endpoint unregistered from discovery");
        }
        self.run_engine_shutdown_steps().await;
    }

    /// Mutex-guarded start. The lifecycle mutex is purely defensive:
    /// `start_engine` is called once and every `cleanup_once` invocation
    /// (whether from inside `run_inner` via the orchestrator paths or
    /// from `Worker::run`'s post-`run_inner` safety net) is strictly
    /// sequential within a single `Worker::run`. The lock provides
    /// memory ordering for state transitions but does not serialize
    /// concurrent callers (there are none).
    async fn start_engine(&self) -> Result<EngineConfig, DynamoError> {
        let mut guard = self.state.lock().await;
        // `start_engine` is called once from `run_inner`, which consumes
        // `self`. Hitting any other state is a programmer error worth
        // panicking over in release as well as debug builds.
        assert_eq!(
            *guard,
            LifecycleState::Init,
            "start_engine called in unexpected state {:?}",
            *guard
        );
        match self.engine.start().await {
            Ok(cfg) => {
                *guard = LifecycleState::Running;
                Ok(cfg)
            }
            Err(e) => {
                *guard = LifecycleState::Stopped;
                Err(e)
            }
        }
    }

    /// Mutex-guarded, idempotent cleanup.
    async fn cleanup_once(&self) {
        let mut guard = self.state.lock().await;
        match *guard {
            LifecycleState::Init | LifecycleState::Stopped => {
                // Pre-start shutdown, already cleaned up, or start failed —
                // nothing engine-side to do.
                *guard = LifecycleState::Stopped;
                return;
            }
            LifecycleState::Running => {}
        }
        match self.engine.cleanup().await {
            Ok(()) => tracing::info!("Engine cleanup complete"),
            Err(e) => tracing::error!(error = %e, "engine cleanup failed"),
        }
        // Mark stopped even on failure so a follow-up call no-ops; engines
        // like vLLM/TRT-LLM tear down NCCL groups in cleanup() and a second
        // attempt can hang or raise.
        *guard = LifecycleState::Stopped;
    }

    /// Drive the serve loop and the shutdown orchestrator. Returns when
    /// either the serve loop exits or `shutdown` is cancelled.
    async fn serve_with_orchestrator(
        &self,
        engine_config: &EngineConfig,
        endpoint: dynamo_runtime::component::Endpoint,
        shutdown: CancellationToken,
    ) -> Result<(), DynamoError> {
        let model_type = parse_endpoint_types(&self.config.endpoint_types)?;

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

        let ingress = Ingress::for_engine(Arc::new(EngineAdapter::new(self.engine.clone())))
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
                // Endpoint exited without a shutdown signal — usually an
                // error path. Skip the orchestrator; cleanup runs in run().
                return result.map_err(|e| {
                    err(
                        ErrorType::Backend(BackendError::Unknown),
                        format!("serve: {e}"),
                    )
                });
            }
            _ = shutdown.cancelled() => {
                tracing::info!("Received shutdown signal; running graceful orchestration");
            }
        }

        self.orchestrator_steps(&endpoint).await;
        Ok(())
    }

    /// Engine-facing shutdown sequence: grace period sleep → `engine.drain()`
    /// → `cleanup_once()`. Each step swallows non-fatal failures so a
    /// misbehaving engine can't block the worker from exiting.
    async fn run_engine_shutdown_steps(&self) {
        self.run_engine_shutdown_steps_with_grace(grace_period_secs())
            .await
    }

    /// Same as [`run_engine_shutdown_steps`] but with an explicit grace
    /// period. Lets unit tests assert on call ordering without setting
    /// `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` (which is process-global
    /// and would race other parallel tests).
    async fn run_engine_shutdown_steps_with_grace(&self, grace: f64) {
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
/// upstream `worker.execute` flow we bypass.
fn graceful_shutdown_timeout() -> Duration {
    use dynamo_runtime::config::environment_names::worker as env_worker;

    // `dynamo_runtime::worker::DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_*` are
    // re-implemented here rather than imported so backend-common's
    // contract is self-documenting.
    const DEFAULT_DEBUG: u64 = 5;
    const DEFAULT_RELEASE: u64 = 30;
    let default = if cfg!(debug_assertions) {
        DEFAULT_DEBUG
    } else {
        DEFAULT_RELEASE
    };

    let secs = std::env::var(env_worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default);
    Duration::from_secs(secs)
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

async fn build_local_model(
    config: &WorkerConfig,
    engine_config: &EngineConfig,
) -> Result<LocalModel, DynamoError> {
    let served_name = resolve_served_name(config, engine_config)
        .or_else(|| Some(engine_config.model.clone()))
        .filter(|s| !s.is_empty());

    let rt_cfg = ModelRuntimeConfig {
        total_kv_blocks: engine_config.total_kv_blocks,
        max_num_seqs: engine_config.max_num_seqs,
        max_num_batched_tokens: engine_config.max_num_batched_tokens,
        tool_call_parser: config.tool_call_parser.clone(),
        reasoning_parser: config.reasoning_parser.clone(),
        exclude_tools_when_tool_choice_none: config.exclude_tools_when_tool_choice_none,
        enable_local_indexer: config.enable_local_indexer,
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
        async fn start(&self) -> Result<EngineConfig, DynamoError> {
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
            _ctx: Arc<dyn crate::engine::AsyncEngineContext>,
        ) -> Result<BoxStream<'static, crate::engine::LLMEngineOutput>, DynamoError> {
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
        let worker = worker_with(engine);
        let cfg = worker.start_engine().await.expect("start");
        assert_eq!(cfg.model, "mock");
        assert_eq!(*worker.state.lock().await, LifecycleState::Running);
    }

    #[tokio::test]
    async fn start_engine_init_to_stopped_on_failure() {
        let (engine, cleanup_calls) = StateMockEngine::new(true);
        let worker = worker_with(engine);
        let res = worker.start_engine().await;
        assert!(res.is_err(), "start should fail");
        assert_eq!(*worker.state.lock().await, LifecycleState::Stopped);

        // cleanup_once is a no-op after a failed start: the state machine
        // skips engine.cleanup() because the engine never became running.
        worker.cleanup_once().await;
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
        assert_eq!(*worker.state.lock().await, LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn cleanup_once_is_idempotent() {
        let (engine, cleanup_calls) = StateMockEngine::new(false);
        let worker = worker_with(engine);
        worker.start_engine().await.unwrap();

        worker.cleanup_once().await;
        worker.cleanup_once().await;
        worker.cleanup_once().await;

        // engine.cleanup() runs at most once even though cleanup_once was
        // called three times — guards against the vLLM/TRT-LLM NCCL
        // double-teardown hang.
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*worker.state.lock().await, LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn cleanup_once_noops_when_never_started() {
        let (engine, cleanup_calls) = StateMockEngine::new(false);
        let worker = worker_with(engine);
        // Pre-start signal path: cleanup before start completes.
        worker.cleanup_once().await;
        assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
        assert_eq!(*worker.state.lock().await, LifecycleState::Stopped);
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
        async fn start(&self) -> Result<EngineConfig, DynamoError> {
            self.log.lock().unwrap().push("start");
            Ok(EngineConfig {
                model: "mock".to_string(),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: Arc<dyn crate::engine::AsyncEngineContext>,
        ) -> Result<BoxStream<'static, crate::engine::LLMEngineOutput>, DynamoError> {
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
        let worker = worker_with(engine);
        worker.start_engine().await.unwrap();

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
        let worker = worker_with(engine);
        worker.start_engine().await.unwrap();

        worker.run_engine_shutdown_steps_with_grace(0.0).await;

        // Drain ran (and failed), but cleanup still ran exactly once.
        let recorded = log.lock().unwrap().clone();
        assert_eq!(recorded, vec!["start", "drain", "cleanup"]);
        assert_eq!(*worker.state.lock().await, LifecycleState::Stopped);
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

    fn with_env<F: FnOnce() -> R, R>(key: &str, value: Option<&str>, f: F) -> R {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize env access on this key by convention; no
        // other test thread reads this var.
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
    // RuntimeConfig env application
    // -------------------------------------------------------------------

    #[test]
    fn runtime_config_apply_to_env_writes_set_fields() {
        let cfg = RuntimeConfig {
            discovery_backend: Some("file".to_string()),
            request_plane: Some("tcp".to_string()),
            event_plane: Some("zmq".to_string()),
        };

        // Snapshot prior values so we don't leak state to other tests.
        let prev: Vec<_> = ["DYN_DISCOVERY_BACKEND", "DYN_REQUEST_PLANE", "DYN_EVENT_PLANE"]
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
