// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-agnostic **direct-backend discoverability shim**.
//!
//! The direct-gRPC path does not run a Dynamo `Worker`: the frontend dials each
//! stock engine's gRPC server itself (see the per-engine `GrpcDispatch`). All a
//! replica needs is to be *discoverable* — a model card + a `TransportType::Grpc`
//! endpoint published in Dynamo discovery under its `DistributedRuntime` lease,
//! with a health gate that pulls the record when the engine dies.
//!
//! This crate owns that narrow responsibility as a thin shim on
//! `dynamo-runtime` + `dynamo-llm` — deliberately NOT on `backend-common`, and
//! NOT an `LLMEngine`/`Worker`. An engine contributes only a small
//! [`DirectBackend`] (connect / health / cleanup); the shim owns the DRT,
//! model-card build, endpoint registration, the hysteresis health loop, and
//! graceful shutdown.
//!
//! [`run_direct`] is the lifecycle driver — the standalone counterpart to
//! `backend-common`'s `register_direct_orchestrator`. It reimplements the
//! model-card build (`dynamo_llm::local_model`),
//! `Endpoint::register_direct_endpoint_instance`, and the
//! 3-fail-unregister / 2-success-reregister health loop (with a per-probe
//! timeout) directly on runtime + llm.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use dynamo_llm::discovery::DIRECT_BACKEND_KEY;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{DistributedRuntime, Runtime, logging};

/// Discovery key naming the direct dispatch endpoint the frontend dials for a
/// model. Written into the model card's `runtime_data` and used as the
/// `TransportType::Grpc` instance address. Owned here (not `backend-common`)
/// because the shim is the sole writer on the v2 path.
pub const DIRECT_GRPC_ENDPOINT_KEY: &str = "direct_grpc_endpoint";

/// The facts a direct backend resolves at connect time — everything the shim
/// needs to publish the model card and advertise the gRPC endpoint.
#[derive(Clone, Debug)]
pub struct DirectRegistration {
    /// Provider name written to the model card's `runtime_data["direct_backend"]`
    /// (e.g. `"trtllm"` / `"vllm"` / `"sglang"`); the frontend uses it to pick the
    /// matching `GrpcDispatch`.
    pub backend: String,
    /// The engine's own local gRPC address the shim health-checks. Absent a
    /// `DirectConfig::advertise_grpc_endpoint` override, this is also what the
    /// frontend dials; multi-node deployments override it with a routable
    /// address.
    pub grpc_endpoint: String,
    /// HF id or local path the frontend uses to build the tokenizer/template.
    pub model_path: String,
    /// Served / display model name, when the engine can report one (e.g. SGLang
    /// discovery). `None` lets the model card default to the model path.
    pub model_name: Option<String>,
    /// Model context length, when the engine can report it (TRT-LLM `GetModelInfo`
    /// returns 0, so it comes from `--context-length`); fills a default `max_tokens`.
    pub context_length: Option<u32>,
    /// Tool-call parser resolved from the engine (e.g. SGLang discovery). Takes
    /// precedence over [`DirectConfig::tool_call_parser`]; `None` falls back to it.
    pub tool_call_parser: Option<String>,
    /// Reasoning parser resolved from the engine. Takes precedence over
    /// [`DirectConfig::reasoning_parser`]; `None` falls back to it.
    pub reasoning_parser: Option<String>,
}

/// Deployment facts the host supplies from CLI args — everything the shim needs
/// that is NOT resolved from the running engine. Model identity and context are
/// resolved at connect time and travel on [`DirectRegistration`] instead.
#[derive(Clone, Debug)]
pub struct DirectConfig {
    /// Dynamo namespace for discovery routing.
    pub namespace: String,
    /// Component name within the namespace.
    pub component: String,
    /// Endpoint name exposed by this worker (e.g. `"generate"`).
    pub endpoint: String,
    /// Optional path to a custom Jinja chat template.
    pub custom_jinja_template: Option<PathBuf>,
    /// Tool-call parser from CLI. Used only when the engine
    /// ([`DirectRegistration::tool_call_parser`]) resolves none.
    pub tool_call_parser: Option<String>,
    /// Reasoning parser from CLI. Used only when the engine
    /// ([`DirectRegistration::reasoning_parser`]) resolves none.
    pub reasoning_parser: Option<String>,
    /// Address the frontend dials, when it differs from the engine's local gRPC
    /// endpoint (multi-node). `None` advertises the engine's own endpoint.
    pub advertise_grpc_endpoint: Option<String>,
}

/// An external engine reachable over gRPC that the shim makes discoverable.
///
/// This is the entire engine-facing contract for the direct path — no token
/// pipeline, no `generate`. `connect` resolves the registration facts; the shim
/// polls `health_check` on an interval and pulls the discovery record on failure
/// (re-adding on recovery); `cleanup` releases the client on shutdown.
#[async_trait]
pub trait DirectBackend: Send + Sync {
    /// Connect to the engine and resolve the model/endpoint facts to register.
    async fn connect(&self) -> Result<DirectRegistration>;

    /// Cheap liveness probe of the engine's gRPC. Drives the health gate.
    async fn health_check(&self) -> Result<()>;

    /// Release engine resources. Called once on shutdown.
    async fn cleanup(&self) -> Result<()>;
}

/// Lifecycle driver for the direct-gRPC discoverability path.
///
/// Standalone counterpart to `backend-common`'s `register_direct_orchestrator`,
/// built only on `dynamo-runtime` + `dynamo-llm`. Creates the runtime, installs
/// signal handling, resolves the endpoint, connects the backend, publishes the
/// model card + a `TransportType::Grpc` instance, runs the hysteresis health
/// loop, and tears the instance down on shutdown.
///
/// Sync (like `dynamo_backend_common::run`) so an engine's `main` can call it
/// directly: it owns the tokio runtime and blocks until shutdown.
pub fn run_direct(backend: Arc<dyn DirectBackend>, config: DirectConfig) -> Result<()> {
    logging::init();

    let runtime = Runtime::from_settings()?;
    let secondary = runtime.secondary();
    secondary.block_on(async move {
        let result = run_direct_inner(backend, config, runtime.clone()).await;
        // Trigger token cancellation + NATS/etcd disconnect. By this point the
        // discovery unregister + engine cleanup already ran, so this is purely
        // transport teardown (mirrors `backend-common`'s `run`).
        runtime.shutdown();
        result
    })
}

async fn run_direct_inner(
    backend: Arc<dyn DirectBackend>,
    config: DirectConfig,
    runtime: Runtime,
) -> Result<()> {
    // Install OS signal handlers synchronously so a SIGTERM delivered during
    // DRT construction is captured rather than defaulting to abrupt process
    // termination. The background task only flips the shared token.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("install SIGTERM handler")?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .context("install SIGINT handler")?;
    let shutdown = CancellationToken::new();
    let signal_token = shutdown.clone();
    let signal_handle = tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => tracing::info!("SIGTERM received"),
            _ = sigint.recv() => tracing::info!("SIGINT received"),
        }
        signal_token.cancel();
    });

    let outcome = run_direct_lifecycle(backend, config, runtime, &shutdown).await;

    signal_handle.abort();
    let _ = signal_handle.await;
    outcome
}

async fn run_direct_lifecycle(
    backend: Arc<dyn DirectBackend>,
    config: DirectConfig,
    runtime: Runtime,
    shutdown: &CancellationToken,
) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime)
        .await
        .context("distributed runtime")?;

    let endpoint = drt
        .namespace(&config.namespace)
        .and_then(|ns| ns.component(&config.component))
        .context("resolve component")?
        .endpoint(&config.endpoint);
    tracing::debug!(
        namespace = %config.namespace,
        component = %config.component,
        endpoint = %config.endpoint,
        "component and endpoint resolved"
    );

    // Shutdown arrived during DRT construction; the engine was never connected.
    if shutdown.is_cancelled() {
        tracing::info!("Shutdown signal observed before engine connect; exiting cleanly");
        return Ok(());
    }

    let reg = backend.connect().await.context("direct backend connect")?;
    tracing::debug!(backend = %reg.backend, "direct backend connected");

    // The address the frontend dials: the multi-node advertise override, else
    // the engine's own local gRPC endpoint.
    let grpc_endpoint = config
        .advertise_grpc_endpoint
        .clone()
        .unwrap_or_else(|| reg.grpc_endpoint.clone());

    // Publish the model card. `attach` registers a Model discovery record; the
    // frontend selects the direct GrpcDispatch off `runtime_data["direct_backend"]`.
    let mut local_model = build_local_model(&config, &reg, &grpc_endpoint)
        .await
        .context("build direct model card")?;
    local_model
        .attach(
            &endpoint,
            ModelType::Chat | ModelType::Completions,
            ModelInput::Tokens,
            None,
            Some(WorkerType::Aggregated),
            Vec::new(),
        )
        .await
        .context("attach direct model card")?;
    tracing::debug!("direct model card registered with discovery");

    // Register the direct-gRPC endpoint instance the frontend routes to.
    endpoint
        .register_direct_endpoint_instance(grpc_endpoint.clone())
        .await
        .context("register direct endpoint")?;

    let served = reg.model_name.clone().unwrap_or_else(|| reg.model_path.clone());
    tracing::info!(
        "Serving {} on {}.{}.{} via direct gRPC → {}",
        served,
        config.namespace,
        config.component,
        config.endpoint,
        grpc_endpoint,
    );

    // Hold the graceful-shutdown registration so runtime teardown waits for our
    // unregister + cleanup (mirrors `serve_with_orchestrator`).
    let _graceful = endpoint.drt().register_graceful_task();

    let registered = run_health_loop(
        backend.as_ref(),
        &endpoint,
        &grpc_endpoint,
        shutdown,
    )
    .await;

    tracing::info!("Received shutdown signal; tearing down direct endpoint");
    if registered {
        if let Err(error) = endpoint
            .unregister_direct_endpoint_instance(grpc_endpoint)
            .await
        {
            tracing::warn!(%error, "direct endpoint discovery unregister failed");
        } else {
            tracing::info!("Direct endpoint unregistered from discovery");
        }
    }
    if let Err(error) = backend.cleanup().await {
        tracing::warn!(%error, "direct backend cleanup failed");
    }
    Ok(())
}

/// Build the model card for the direct path. Replicates the essential subset of
/// `backend-common`'s `build_local_model`: fetch the model artifacts, stamp the
/// direct `runtime_data`, and carry the resolved context / parser settings.
async fn build_local_model(
    config: &DirectConfig,
    reg: &DirectRegistration,
    grpc_endpoint: &str,
) -> Result<LocalModel> {
    if reg.model_path.trim().is_empty() {
        anyhow::bail!("direct backend returned an empty model_path");
    }

    // Direct dispatch selectors, read by the frontend discovery watcher.
    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        DIRECT_BACKEND_KEY.to_string(),
        serde_json::Value::String(reg.backend.clone()),
    );
    runtime_data.insert(
        DIRECT_GRPC_ENDPOINT_KEY.to_string(),
        serde_json::Value::String(grpc_endpoint.to_string()),
    );

    let rt_cfg = ModelRuntimeConfig {
        context_length: reg.context_length,
        tool_call_parser: reg
            .tool_call_parser
            .clone()
            .or_else(|| config.tool_call_parser.clone()),
        reasoning_parser: reg
            .reasoning_parser
            .clone()
            .or_else(|| config.reasoning_parser.clone()),
        runtime_data,
        ..ModelRuntimeConfig::default()
    };

    // Resolve model_path to a local path. An existing filesystem path is used
    // as-is; otherwise fetch (HF repo id) the tokenizer/template artifacts.
    let source = reg.model_path.clone();
    let local_path = if std::fs::exists(&source).context("probe model path")? {
        PathBuf::from(&source)
    } else {
        LocalModel::fetch(&source, false)
            .await
            .with_context(|| format!("fetch '{source}'"))?
    };

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(reg.model_name.clone())
        .model_path(local_path)
        .source_path(PathBuf::from(&source))
        .namespace(Some(config.namespace.clone()))
        .custom_template_path(config.custom_jinja_template.clone())
        .runtime_config(rt_cfg);
    builder.build().await.context("build local model")
}

/// Health probe cadence for the direct-mode registrar.
const HEALTH_INTERVAL: Duration = Duration::from_secs(10);
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const FAILURES_BEFORE_UNREGISTER: u32 = 3;
const SUCCESSES_BEFORE_REGISTER: u32 = 2;

/// Proactively probe engine liveness and pull the instance from discovery when
/// it dies (re-register on recovery), complementing the frontend's reactive
/// `report_instance_down`. Consecutive-count thresholds stop a briefly-slow
/// engine from flapping the routing pool; a per-probe timeout stops a wedged
/// engine (TCP up, RPC hung) from defeating the unregister. Returns whether the
/// instance is still registered when shutdown is observed.
async fn run_health_loop(
    backend: &dyn DirectBackend,
    endpoint: &dynamo_runtime::component::Endpoint,
    grpc_endpoint: &str,
    shutdown: &CancellationToken,
) -> bool {
    let mut registered = true;
    let mut failures = 0u32;
    let mut successes = 0u32;
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            _ = tokio::time::sleep(HEALTH_INTERVAL) => {}
        }
        // Race the probe against shutdown so a hung RPC can't stall it.
        let health = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            result = tokio::time::timeout(HEALTH_PROBE_TIMEOUT, backend.health_check()) => result,
        };
        let is_healthy = match health {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                tracing::debug!(%error, "direct worker health probe failed");
                false
            }
            Err(_) => {
                tracing::debug!("direct worker health probe timed out");
                false
            }
        };

        if is_healthy {
            failures = 0;
            successes += 1;
            if !registered && successes >= SUCCESSES_BEFORE_REGISTER {
                match endpoint
                    .register_direct_endpoint_instance(grpc_endpoint.to_string())
                    .await
                {
                    Ok(()) => {
                        registered = true;
                        tracing::info!("direct worker recovered; re-registered with discovery");
                    }
                    Err(error) => {
                        tracing::warn!(%error, "failed to re-register recovered direct worker")
                    }
                }
            }
        } else {
            successes = 0;
            failures += 1;
            if registered && failures >= FAILURES_BEFORE_UNREGISTER {
                tracing::warn!(failures, "direct worker unhealthy; unregistering");
                match endpoint
                    .unregister_direct_endpoint_instance(grpc_endpoint.to_string())
                    .await
                {
                    Ok(()) => registered = false,
                    Err(error) => {
                        tracing::warn!(%error, "failed to unregister unhealthy direct worker")
                    }
                }
            }
        }
    }
    registered
}
