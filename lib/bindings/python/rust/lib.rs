// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::local_model::LocalModel;
use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::storage::kv;
use futures::StreamExt;
use once_cell::sync::OnceCell;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyStopAsyncIteration, PyTimeoutError, PyValueError};
use pyo3::types::PyCapsule;
use pyo3::types::{PyDict, PyString};
use pyo3::{exceptions::PyException, prelude::*};
use rs::pipeline::network::Ingress;
use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::{
    fmt::Display,
    sync::{Arc, Weak},
};
use tokio::sync::Mutex;
use tracing::Instrument;

use dynamo_runtime::config;
use dynamo_runtime::config::environment_names::logging::otlp as env_otlp;
use dynamo_runtime::{
    self as rs, logging,
    pipeline::{
        AsyncEngineContextProvider, EngineStream, ManyOut, SingleIn, context::Context as RsContext,
        network::egress::push_router::RouterMode as RsRouterMode,
    },
    protocols::annotated::Annotated as RsAnnotated,
    traits::DistributedRuntimeProvider,
};

use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::{self as llm_rs};

use crate::llm::entrypoint::RouterConfig as PyRouterConfig;

use crate::llm::local_model::{ModelRuntimeConfig, RoutingConstraints, parse_tensor_model_config};
use crate::llm::preprocessor::{MediaDecoder, MediaFetcher};

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum RouterMode {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    KV,
    /// Direct routing - reads worker ID from each request's routing hints.
    /// Used when an external orchestrator (e.g., EPP) handles worker selection.
    Direct,
    LeastLoaded,
    DeviceAwareWeighted,
}

impl From<RouterMode> for RsRouterMode {
    fn from(mode: RouterMode) -> Self {
        match mode {
            RouterMode::RoundRobin => Self::RoundRobin,
            RouterMode::Random => Self::Random,
            RouterMode::PowerOfTwoChoices => Self::PowerOfTwoChoices,
            RouterMode::KV => Self::KV,
            RouterMode::Direct => Self::Direct,
            RouterMode::LeastLoaded => Self::LeastLoaded,
            RouterMode::DeviceAwareWeighted => Self::DeviceAwareWeighted,
        }
    }
}

mod backend;
mod context;
mod engine;
pub mod errors;
mod http;
mod kserve_grpc;
mod llm;
mod parsers;
mod planner;
mod prometheus_metrics;

type JsonServerStreamingIngress =
    Ingress<SingleIn<serde_json::Value>, ManyOut<RsAnnotated<serde_json::Value>>>;

type JsonBidirectionalIngress =
    Ingress<rs::pipeline::ManyIn<serde_json::Value>, ManyOut<RsAnnotated<serde_json::Value>>>;

static INIT: OnceCell<()> = OnceCell::new();

const DEFAULT_ANNOTATED_SETTING: Option<bool> = Some(true);
const SKIP_PYTHON_LOG_INIT_ENV: &str = "DYNAMO_SKIP_PYTHON_LOG_INIT";

// Helper to get appropriate span for instrumentation - always emit spans
fn get_span_for_context(context: &context::Context, operation: &str) -> tracing::Span {
    logging::make_client_request_span(
        operation,
        context.inner().id(),
        context.trace_context(),
        None,
    )
}

// Helper to create span for direct method with instance_id
fn get_span_for_direct_context(
    context: &context::Context,
    operation: &str,
    instance_id: &str,
) -> tracing::Span {
    logging::make_client_request_span(
        operation,
        context.inner().id(),
        context.trace_context(),
        Some(instance_id),
    )
}

// Helper to create request context with proper linking and cancellation handling
fn create_request_context(
    request: serde_json::Value,
    parent_ctx: &Option<context::Context>,
) -> RsContext<serde_json::Value> {
    match parent_ctx {
        // If there is a parent context, link the request as a child context of it
        Some(parent_ctx) => {
            let child_ctx = RsContext::with_id_and_metadata(
                request,
                parent_ctx.inner().id().to_string(),
                parent_ctx.metadata_snapshot(),
            );
            parent_ctx.inner().link_child(child_ctx.context());
            if parent_ctx.inner().is_stopped() || parent_ctx.inner().is_killed() {
                // Let the server handle the cancellation for now since not all backends are
                // properly handling request exceptions
                // TODO: (DIS-830) Return an error if context is cancelled
                child_ctx.context().stop_generating();
            }
            child_ctx
        }
        // Otherwise if there is no parent context, use the request as-is
        _ => request.into(),
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging early unless OTEL export is enabled (which requires tokio runtime)
    if config::env_is_truthy(env_otlp::OTEL_EXPORT_ENABLED) {
        eprintln!(
            "Warning: OTEL_EXPORT_ENABLED detected. Logging initialization deferred until runtime is available. Early logs may be dropped."
        );
    } else if std::env::var_os(SKIP_PYTHON_LOG_INIT_ENV).is_none() {
        rs::logging::init();
    }

    m.add_function(wrap_pyfunction!(llm::kv::compute_block_hash_for_seq_py, m)?)?;
    m.add_function(wrap_pyfunction!(lora_name_to_id, m)?)?;
    #[cfg(feature = "mm-routing")]
    m.add_function(wrap_pyfunction!(resolve_routing_image_token_id, m)?)?;
    m.add_function(wrap_pyfunction!(log_message, m)?)?;
    m.add_function(wrap_pyfunction!(register_model, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_model, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_model, m)?)?;
    m.add_function(wrap_pyfunction!(run_kv_indexer, m)?)?;
    m.add_function(wrap_pyfunction!(run_slot_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(run_select_service, m)?)?;
    m.add_function(wrap_pyfunction!(llm::entrypoint::make_engine, m)?)?;
    m.add_function(wrap_pyfunction!(llm::replay::run_mocker_trace_replay, m)?)?;
    m.add_function(wrap_pyfunction!(
        llm::replay::run_mocker_synthetic_trace_replay,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(llm::entrypoint::run_input, m)?)?;

    m.add_class::<DistributedRuntime>()?;
    m.add_class::<Endpoint>()?;
    m.add_class::<ModelCardInstanceId>()?;
    m.add_class::<Client>()?;
    m.add_class::<AsyncResponseStream>()?;
    m.add_class::<PyAsyncRequestStream>()?;
    m.add_class::<llm::entrypoint::EntrypointArgs>()?;
    m.add_class::<llm::entrypoint::EngineConfig>()?;
    m.add_class::<llm::entrypoint::EngineType>()?;
    m.add_class::<llm::entrypoint::AicPerfConfig>()?;
    m.add_class::<llm::entrypoint::RouterConfig>()?;
    m.add_class::<llm::entrypoint::KvRouterConfig>()?;
    m.add_class::<llm::replay::ReasoningConfig>()?;
    m.add_class::<llm::replay::SglangArgs>()?;
    m.add_class::<llm::replay::TrtllmArgs>()?;
    m.add_class::<llm::replay::MockEngineArgs>()?;
    #[cfg(feature = "aic-forward-pass")]
    {
        m.add_class::<llm::engine_perf::AicEngineConfig>()?;
        m.add_class::<llm::engine_perf::EngineCapacity>()?;
        m.add_class::<llm::engine_perf::EngineCapacityRequest>()?;
        m.add_class::<llm::engine_perf::EnginePerfLimits>()?;
        m.add_class::<llm::engine_perf::OptimizationTarget>()?;
        m.add_class::<llm::engine_perf::RustEnginePerfModel>()?;
        m.add_class::<llm::engine_perf::RustEnginePerfOptions>()?;
    }
    m.add_class::<llm::replay::PlannerReplayBridge>()?;
    #[cfg(feature = "select-service")]
    m.add_class::<llm::kv::SelectionService>()?;
    m.add_class::<llm::kv::WorkerMetricsPublisher>()?;
    m.add_class::<llm::kv::MultimodalEmbeddingCachePublisher>()?;
    m.add_class::<llm::model_card::ModelDeploymentCard>()?; // Internal: only in _internal, not public API
    m.add_class::<llm::local_model::ModelRuntimeConfig>()?;
    m.add_class::<RoutingConstraints>()?;
    m.add_class::<llm::preprocessor::MediaDecoder>()?;
    m.add_class::<llm::preprocessor::MediaFetcher>()?;
    m.add_class::<llm::kv::OverlapScores>()?;
    m.add_class::<llm::kv::KvEventPublisher>()?;
    m.add_class::<llm::kv::RadixTree>()?;
    m.add_class::<llm::fpm::FpmEventRelay>()?;
    m.add_class::<llm::fpm::FpmDirectPublisher>()?;
    m.add_class::<llm::fpm::FpmEventSubscriber>()?;
    m.add_class::<llm::lora::LoRADownloader>()?;
    m.add_class::<http::HttpService>()?;
    m.add_class::<http::HttpAsyncEngine>()?;
    m.add_class::<context::Context>()?;
    m.add_class::<context::ContextMetadata>()?;
    m.add_class::<context::SpanProxy>()?;
    m.add_class::<ModelType>()?;
    m.add_class::<ModelInput>()?;
    m.add_class::<WorkerType>()?;
    m.add_class::<llm::kv::KvRouter>()?;
    m.add_class::<llm::routed_engine::RoutedEngine>()?;
    m.add_class::<RouterMode>()?;
    m.add_class::<kserve_grpc::KserveGrpcService>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<planner::VirtualConnectorCoordinator>()?;
    m.add_class::<planner::VirtualConnectorClient>()?;
    m.add_class::<planner::PlannerDecision>()?;

    engine::add_to_module(m)?;
    errors::register_exceptions(m)?;
    parsers::add_to_module(m)?;
    backend::add_to_module(m)?;

    m.add_class::<prometheus_metrics::RuntimeMetrics>()?;

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}

fn standalone_to_pyerr(err: anyhow::Error) -> PyErr {
    #[cfg(any(
        feature = "kv-indexer",
        feature = "slot-tracker",
        feature = "select-service"
    ))]
    if let Some(clap_error) = err.downcast_ref::<clap::Error>() {
        let _ = clap_error.print();
        return pyo3::exceptions::PySystemExit::new_err(clap_error.exit_code());
    }

    to_pyerr(err)
}

fn resolve_event_transport_kind(
    discovery_backend: &DiscoveryBackend,
    event_plane: Option<&str>,
) -> PyResult<EventTransportKind> {
    match event_plane {
        Some("nats") => Ok(EventTransportKind::Nats),
        Some("zmq") => Ok(EventTransportKind::Zmq),
        Some("") | None => Ok(discovery_backend.resolve_event_transport_kind()),
        Some(other) => Err(PyValueError::new_err(format!(
            "Invalid event_plane value '{other}'. Valid values: 'nats', 'zmq'"
        ))),
    }
}

#[pyfunction(name = "run_kv_indexer")]
#[pyo3(signature = (argv=None))]
fn run_kv_indexer(py: Python<'_>, argv: Option<Vec<String>>) -> PyResult<()> {
    let argv = argv.unwrap_or_default();
    py.allow_threads(move || llm::kv::run_kv_indexer_cli(argv))
        .map_err(standalone_to_pyerr)
}

#[pyfunction(name = "run_slot_tracker")]
#[pyo3(signature = (argv=None))]
fn run_slot_tracker(py: Python<'_>, argv: Option<Vec<String>>) -> PyResult<()> {
    let argv = argv.unwrap_or_default();
    py.allow_threads(move || llm::kv::run_slot_tracker_cli(argv))
        .map_err(standalone_to_pyerr)
}

#[pyfunction(name = "run_select_service")]
#[pyo3(signature = (argv=None))]
fn run_select_service(py: Python<'_>, argv: Option<Vec<String>>) -> PyResult<()> {
    let argv = argv.unwrap_or_default();
    py.allow_threads(move || llm::kv::run_select_service_cli(argv))
        .map_err(standalone_to_pyerr)
}

/// Log a message from Python with file and line info
#[pyfunction]
#[pyo3(text_signature = "(level, message, module, file, line)")]
fn log_message(level: &str, message: &str, module: &str, file: &str, line: u32) {
    logging::log_message(level, message, module, file, line);
}

/// Generate a deterministic signed int32 ID from a LoRA name using blake3 hash.
#[pyfunction]
#[pyo3(text_signature = "(lora_name)")]
fn lora_name_to_id(lora_name: &str) -> i32 {
    llm_rs::utils::lora_name_to_id(lora_name)
}

/// Resolve the routing-side image-placeholder token id for a model using the
/// same per-family logic the frontend's MM-aware KV routing uses (lightseek
/// `resolve_routing_tokens`). Returns `chat_placeholder_token_id` — the exact
/// id `OpenAIPreprocessor` substitutes `pad_value` over — so the vLLM worker's
/// KV-event normalizer keys on the identical token (no cross-process drift).
///
/// `model_id` is the HF id (used for registry matching); `model_dir` is the
/// local directory holding `config.json`/`tokenizer.json`. Returns `None` when
/// the model isn't in the MM-routing registry or its config can't be read.
#[cfg(feature = "mm-routing")]
#[pyfunction]
#[pyo3(text_signature = "(model_id, model_dir)")]
fn resolve_routing_image_token_id(model_id: &str, model_dir: &str) -> Option<u32> {
    let dir = std::path::Path::new(model_dir);
    llm_rs::preprocessor::lightseek_mm::resolve_routing_tokens(model_id, dir)
        .chat_placeholder_token_id
}

/// Create an engine and attach it to an endpoint to make it visible to the frontend.
/// This is the main way you create a Dynamo worker / backend.
///
/// If `lora_name` is provided, this function will publish a LoRA adapter instead of a base model:
/// - LoRA path: v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
/// - Base model path: v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}
///
/// For LoRA mode, both `lora_name` and `base_model_path` must be provided together.
/// Providing only one of them will result in an error.
#[pyfunction]
#[pyo3(signature = (model_input, model_type, endpoint, model_path, model_name=None, kv_cache_block_size=None, router_config=None, runtime_config=None, user_data=None, custom_template_path=None, media_decoder=None, media_fetcher=None, lora_name=None, base_model_path=None, worker_type=None, needs=None, self_host_metadata=None, *, tensor_model_config=None, ignore_weights=false, max_gpu_lora_count=None, model_aliases=None))]
#[allow(clippy::too_many_arguments)]
fn register_model<'p>(
    py: Python<'p>,
    model_input: ModelInput,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: &str,
    model_name: Option<&str>,
    kv_cache_block_size: Option<u32>,
    router_config: Option<PyRouterConfig>,
    runtime_config: Option<ModelRuntimeConfig>,
    user_data: Option<&Bound<'p, PyDict>>,
    custom_template_path: Option<&str>,
    media_decoder: Option<MediaDecoder>,
    media_fetcher: Option<MediaFetcher>,
    lora_name: Option<&str>,
    base_model_path: Option<&str>,
    worker_type: Option<WorkerType>,
    needs: Option<Vec<Vec<WorkerType>>>,
    self_host_metadata: Option<bool>,
    tensor_model_config: Option<&Bound<'p, PyDict>>,
    ignore_weights: bool,
    max_gpu_lora_count: Option<u32>,
    model_aliases: Option<Vec<String>>,
) -> PyResult<Bound<'p, PyAny>> {
    // Every worker registers with an explicit `worker_type`. Reject `None`
    // outright — a missing role would produce a card whose readiness math
    // is undefined and whose ws_key would collide with other Aggregated
    // workers in the same namespace.
    let Some(worker_type_unwrapped) = worker_type else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "register_model: `worker_type` is required. Pass one of \
             WorkerType.Prefill / Decode / Encode / Aggregated.",
        ));
    };

    // Prefill and Encode workers receive pre-tokenized requests (their engines
    // preprocess externally), so both require `ModelInput::Tokens`.
    if matches!(
        worker_type_unwrapped,
        WorkerType::Prefill | WorkerType::Encode
    ) && !matches!(model_input, ModelInput::Tokens)
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "register_model: worker_type={:?} requires model_input=ModelInput::Tokens",
            worker_type_unwrapped
        )));
    }

    // Prefill workers never expose an OpenAI surface — they are reached only
    // through the dedicated prefill router, never by the frontend. They MAY,
    // however, carry the legacy `ModelType.Prefill` *marker* bit, which new
    // prefill workers dual-emit so an old frontend still detects them during
    // the cross-version rollout (see `ModelType::Prefill`). So the only thing we
    // reject here is a genuine OpenAI *surface* on a prefill card. Encode
    // workers MAY carry a surface: an sglang multimodal encode worker is the
    // OpenAI front door that delegates generation to an internal worker,
    // whereas a vLLM-style encode helper registers Empty. Serving is driven by
    // `ModelType` (the OpenAI surface); the topology role is by `worker_type`.
    if matches!(worker_type_unwrapped, WorkerType::Prefill) {
        // Strip the allowed Prefill marker bit; whatever remains is a surface.
        let surface = model_type.inner - llm_rs::model_type::ModelType::Prefill;
        if !surface.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "register_model: worker_type={:?} must not expose an OpenAI surface \
                 (got model_type={:?}). Use ModelType.Empty or ModelType.Prefill; \
                 the prefill role is carried by worker_type, and ModelType only \
                 describes the OpenAI surface, which prefill workers don't expose.",
                worker_type_unwrapped, model_type.inner
            )));
        }
    }

    let model_input = match model_input {
        ModelInput::Text => llm_rs::model_type::ModelInput::Text,
        ModelInput::Tokens => llm_rs::model_type::ModelInput::Tokens,
        ModelInput::Tensor => llm_rs::model_type::ModelInput::Tensor,
    };

    let is_tensor_based = model_type.inner.supports_tensor();
    let is_images = model_type.inner.supports_images();
    let is_videos = model_type.inner.supports_videos();
    let is_realtime = model_type.inner.supports_realtime();

    let model_type_obj = model_type.inner;
    let tensor_model_config = parse_tensor_model_config(tensor_model_config)?;
    if tensor_model_config.is_some() && !is_tensor_based {
        return Err(PyValueError::new_err(
            "tensor_model_config is only valid for TensorBased models",
        ));
    }

    // Model-serving-readiness fields on the MDC. `worker_type` is required
    // (see the check above). Non-Aggregated workers must declare their peers
    // explicitly — an empty `needs` would make them immediately ready with
    // no dependencies, which is only correct for Aggregated.
    let worker_type_value: Option<llm_rs::worker_type::WorkerType> =
        Some(worker_type_unwrapped.into());
    let raw_needs: Vec<Vec<WorkerType>> = match (worker_type_unwrapped, needs) {
        (WorkerType::Aggregated, None) => Vec::new(),
        (WorkerType::Aggregated, Some(n)) => n,
        (_, None) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "register_model: worker_type={:?} requires a non-empty `needs` \
                 (at least one peer worker type the role depends on)",
                worker_type_unwrapped
            )));
        }
        (_, Some(n)) if n.is_empty() => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "register_model: worker_type={:?} requires a non-empty `needs` \
                 (at least one peer worker type the role depends on)",
                worker_type_unwrapped
            )));
        }
        (_, Some(n)) => n,
    };
    let needs_value: Vec<Vec<llm_rs::worker_type::WorkerType>> = raw_needs
        .into_iter()
        .map(|alt| alt.into_iter().map(|w| w.into()).collect())
        .collect();

    let inner_path = model_path.to_string();
    let model_name = model_name.map(|n| n.to_string());
    // Only embed router_config in the MDC when the caller explicitly provided it.
    // This preserves backward-compat: workers that don't specify router_config continue to
    // fall back to the frontend-level global router config via the watcher.
    let explicit_router_config: Option<RouterConfig> = router_config.map(|rc| rc.into());
    let model_aliases = model_aliases.unwrap_or_default();

    // Early validation of custom template path
    let custom_template_path_owned = custom_template_path
        .map(|s| {
            let path = PathBuf::from(s);
            if !path.exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Custom template file does not exist: {}", path.display()),
                ));
            }
            Ok(path)
        })
        .transpose()?;

    let user_data_json = user_data
        .map(|dict| pythonize::depythonize(dict))
        .transpose()
        .map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to convert user_data: {}", err))
        })?;

    // Validate LoRA parameters: both or neither must be provided
    if lora_name.is_some() ^ base_model_path.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lora_name and base_model_path must both be provided together, or neither",
        ));
    }

    // Determine source_path and lora_identifier based on registration mode
    let (source_path, lora_identifier) = match (lora_name, base_model_path) {
        (Some(lora), Some(base)) => (base.to_string(), Some(lora.to_string())),
        _ => (inner_path, None),
    };

    // Model name: use lora name if present, otherwise provided name or default to source path
    let model_name = lora_identifier
        .clone()
        .or(model_name)
        .or_else(|| Some(source_path.clone()));

    if let Some(cfg) = &runtime_config {
        cfg.validate_config()?;
    }

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let runtime_config = runtime_config.unwrap_or_default();

        // For TensorBased, Images, Videos, and Realtime models, skip
        // HuggingFace downloads and register directly. These model types
        // handle model loading internally; no tokenizer extraction is
        // needed and the source path is not required to be a HF repo.
        if is_tensor_based || is_images || is_videos || is_realtime {
            let model_name = model_name.unwrap_or_else(|| source_path.clone());
            let mut card = llm_rs::model_card::ModelDeploymentCard::with_name_only(&model_name);
            card.model_type = model_type_obj;
            card.model_input = model_input;
            card.worker_type = worker_type_value;
            card.needs = needs_value.clone();
            card.user_data = user_data_json;
            if !model_aliases.is_empty() {
                card.set_aliases(model_aliases);
            }

            card.runtime_config = runtime_config.inner;
            card.tensor_model_config = tensor_model_config;
            card.router_config = explicit_router_config.clone();

            // Register the Model Deployment Card via discovery interface
            let discovery = endpoint.inner.drt().discovery();
            let spec = rs::discovery::DiscoverySpec::from_model(
                endpoint.inner.component().namespace().name().to_string(),
                endpoint.inner.component().name().to_string(),
                endpoint.inner.name().to_string(),
                &card,
            )
            .map_err(to_pyerr)?;
            discovery.register(spec).await.map_err(to_pyerr)?;

            return Ok(());
        }

        // For non-TensorBased models, resolve the model path (local or fetch from HuggingFace).
        // ModelExpress load paths pass ignore_weights=true because the engine already owns
        // weight acquisition; other load paths keep the default full-fetch behavior.
        let model_path = if fs::exists(&source_path)? {
            PathBuf::from(&source_path)
        } else {
            LocalModel::fetch(&source_path, ignore_weights)
                .await
                .map_err(to_pyerr)?
        };

        let mut builder = dynamo_llm::local_model::LocalModelBuilder::default();
        builder
            // model path is the physical path on disk of the downloaded model
            .model_path(model_path)
            // source path is what the user gave as `--model-path`, either a real path (in which
            // case it matches model_path above), or an HF repo.
            .source_path(source_path.clone().into())
            // --served_model_name
            .model_name(model_name.clone())
            // --served_model_name aliases (additional names this model responds to)
            .model_aliases(model_aliases)
            .kv_cache_block_size(kv_cache_block_size)
            .router_config(explicit_router_config.clone())
            .runtime_config({
                let mut rc = runtime_config.inner;
                // The base worker registration carries the worker's LoRA slot capacity so the
                // frontend allocator sees idle-but-LoRA-capable workers before any adapter is
                // loaded. Adapter registrations (lora_name set) carry it via LoraInfo instead.
                if lora_identifier.is_none() {
                    rc.max_gpu_lora_count = max_gpu_lora_count;
                }
                rc
            })
            .user_data(user_data_json)
            .custom_template_path(custom_template_path_owned)
            .media_decoder(media_decoder.map(|m| m.inner))
            .media_fetcher(media_fetcher.map(|m| m.inner));
        // Absence falls through to the DYN_SELF_HOST_METADATA env var default.
        if let Some(enabled) = self_host_metadata {
            builder.self_host_metadata(enabled);
        }

        let mut local_model = builder.build().await.map_err(to_pyerr)?;

        // Convert lora_identifier (Option<String>) to Option<LoraInfo>
        let lora_info = lora_identifier
            .as_ref()
            .map(|name| llm_rs::model_card::LoraInfo {
                name: name.clone(),
                max_gpu_lora_count,
            });

        local_model
            .attach(
                &endpoint.inner,
                model_type_obj,
                model_input,
                lora_info,
                worker_type_value,
                needs_value,
            )
            .await
            .map_err(to_pyerr)?;

        if let Some(lora_name) = lora_identifier {
            tracing::info!("Registered LoRA '{}' MDC", lora_name);
        } else {
            tracing::info!(
                "Registered base model '{}' MDC",
                model_name.unwrap_or(source_path)
            );
        }

        Ok(())
    })
}

/// Unregister a Model Deployment Card (MDC) from the service registry
///
/// This removes an LLM deployment from the discovery system.
///
/// # Arguments
///
/// * `endpoint` - The endpoint where the model is registered
/// * `lora_name` - Optional LoRA adapter name (if unregistering a LoRA deployment)
///
/// # MDC Path Format
///
/// - Base model: `v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}`
/// - LoRA model: `v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}`
#[pyfunction]
#[pyo3(signature = (endpoint, lora_name=None))]
fn unregister_model<'p>(
    py: Python<'p>,
    endpoint: Endpoint,
    lora_name: Option<&str>,
) -> PyResult<Bound<'p, PyAny>> {
    let lora_name_owned = lora_name.map(|s| s.to_string());

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Unified detach method handles both base models and LoRA adapters
        LocalModel::detach_from_endpoint(&endpoint.inner, lora_name_owned.as_deref())
            .await
            .map_err(to_pyerr)?;
        Ok(())
    })
}

/// Download a model from Hugging Face, returning its local path
/// Example: `model_path = await fetch_model("Qwen/Qwen3-0.6B")`
#[pyfunction]
#[pyo3(signature = (remote_name, ignore_weights=false))]
fn fetch_model<'p>(
    py: Python<'p>,
    remote_name: &str,
    ignore_weights: bool,
) -> PyResult<Bound<'p, PyAny>> {
    let repo = remote_name.to_string();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        LocalModel::fetch(&repo, ignore_weights)
            .await
            .map_err(to_pyerr)
    })
}

#[pyclass]
#[derive(Clone)]
pub struct DistributedRuntime {
    inner: rs::DistributedRuntime,
    event_loop: PyObject,
}

impl DistributedRuntime {
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &rs::DistributedRuntime {
        &self.inner
    }
}

#[pyclass]
#[derive(Clone)]
struct CancellationToken {
    inner: rs::CancellationToken,
}

#[pyclass]
#[derive(Clone)]
struct Endpoint {
    inner: rs::component::Endpoint,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct ModelCardInstanceId {
    inner: rs::discovery::ModelCardInstanceId,
}

#[pyclass]
#[derive(Clone)]
struct Client {
    router: rs::pipeline::PushRouter<serde_json::Value, RsAnnotated<serde_json::Value>>,
    endpoint: rs::component::Endpoint,
}

#[pyclass]
#[derive(Clone, PartialEq)]
struct ModelType {
    inner: llm_rs::model_type::ModelType,
}

#[pymethods]
#[allow(non_upper_case_globals)]
impl ModelType {
    /// Empty value — no OpenAI surface. Used by prefill / encode workers
    /// whose role is carried by `WorkerType` rather than by a ModelType bit.
    /// (Name is `Empty` rather than `None` because `None` is reserved in
    /// Python; `Empty` is also symmetric with the other `ModelType.Foo`
    /// values.)
    #[classattr]
    const Empty: Self = ModelType {
        inner: llm_rs::model_type::ModelType::empty(),
    };

    #[classattr]
    const Chat: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Chat,
    };
    #[classattr]
    const Completions: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Completions,
    };
    #[classattr]
    const Embedding: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Embedding,
    };
    #[classattr]
    const TensorBased: Self = ModelType {
        inner: llm_rs::model_type::ModelType::TensorBased,
    };
    /// Legacy prefill marker (no OpenAI surface). The prefill role is
    /// expressed via `WorkerType::Prefill`; this bit is dual-emitted by new
    /// prefill workers for cross-version compatibility so an old frontend
    /// still detects them. Retained only for the cross-version compatibility window.
    #[classattr]
    const Prefill: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Prefill,
    };
    #[classattr]
    const Images: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Images,
    };
    #[classattr]
    const Audios: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Audios,
    };
    #[classattr]
    const Videos: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Videos,
    };
    #[classattr]
    const Realtime: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Realtime,
    };

    fn supports_chat(&self) -> bool {
        self.inner.supports_chat()
    }

    fn __or__(&self, other: &Self) -> Self {
        ModelType {
            inner: self.inner | other.inner,
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
enum ModelInput {
    Text = 1,
    Tokens = 2,
    Tensor = 3,
}

/// Processing stage a worker handles.
///
/// Each worker has exactly one role; values are not combinable. To express
/// "an encode worker needs Prefill+Decode OR Aggregated", `register_model`
/// takes `needs` in DNF form (a list of alternative AND-sets). See the Rust
/// `WorkerType` enum in `lib/llm/src/worker_type.rs` and
/// `docs/proposals/health-disagg-readiness.md`.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkerType {
    Prefill = 1,
    Decode = 2,
    Encode = 3,
    Aggregated = 4,
}

#[pymethods]
impl WorkerType {
    fn __str__(&self) -> &'static str {
        match self {
            WorkerType::Prefill => "prefill",
            WorkerType::Decode => "decode",
            WorkerType::Encode => "encode",
            WorkerType::Aggregated => "aggregated",
        }
    }

    fn __repr__(&self) -> String {
        format!("WorkerType.{:?}", self)
    }
}

impl From<WorkerType> for llm_rs::worker_type::WorkerType {
    fn from(w: WorkerType) -> Self {
        match w {
            WorkerType::Prefill => llm_rs::worker_type::WorkerType::Prefill,
            WorkerType::Decode => llm_rs::worker_type::WorkerType::Decode,
            WorkerType::Encode => llm_rs::worker_type::WorkerType::Encode,
            WorkerType::Aggregated => llm_rs::worker_type::WorkerType::Aggregated,
        }
    }
}

impl From<llm_rs::worker_type::WorkerType> for WorkerType {
    fn from(w: llm_rs::worker_type::WorkerType) -> Self {
        match w {
            llm_rs::worker_type::WorkerType::Prefill => WorkerType::Prefill,
            llm_rs::worker_type::WorkerType::Decode => WorkerType::Decode,
            llm_rs::worker_type::WorkerType::Encode => WorkerType::Encode,
            llm_rs::worker_type::WorkerType::Aggregated => WorkerType::Aggregated,
        }
    }
}

#[pymethods]
impl DistributedRuntime {
    #[new]
    #[pyo3(signature = (event_loop, discovery_backend, request_plane, enable_nats=None, *, event_plane=None))]
    fn new(
        event_loop: PyObject,
        discovery_backend: String,
        request_plane: String,
        enable_nats: Option<bool>,
        event_plane: Option<String>,
    ) -> PyResult<Self> {
        if enable_nats.is_some() {
            Python::with_gil(|py| {
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (
                        "The 'enable_nats' parameter is deprecated and will be removed in a future release. NATS enablement is now determined automatically from the event-plane configuration.",
                        py.import("builtins")?.getattr("DeprecationWarning")?,
                        2i32, // stacklevel
                    ),
                )?;
                Ok::<(), PyErr>(())
            })?;
        }
        let discovery_backend_config = match discovery_backend.as_str() {
            "kubernetes" => DiscoveryBackend::Kubernetes,
            other => {
                let selector: kv::Selector = other.parse().map_err(to_pyerr)?;
                DiscoveryBackend::KvStore(selector)
            }
        };
        let request_plane: RequestPlaneMode = request_plane.parse().map_err(to_pyerr)?;
        let explicit_event_plane = event_plane.as_deref().filter(|value| !value.is_empty());
        let event_transport_kind =
            resolve_event_transport_kind(&discovery_backend_config, event_plane.as_deref())?;

        // Try to get existing runtime first, create new Worker only if needed
        // This allows multiple DistributedRuntime instances to share the same tokio runtime
        let runtime = rs::Worker::runtime_from_existing()
            .or_else(|_| -> anyhow::Result<rs::Runtime> {
                // No existing Worker, create new one
                let worker = rs::Worker::from_settings()?;

                // Initialize pyo3 bridge (only happens once per process)
                INIT.get_or_try_init(|| -> anyhow::Result<()> {
                    let primary = worker.tokio_runtime()?;
                    pyo3_async_runtimes::tokio::init_with_runtime(primary).map_err(|e| {
                        anyhow::anyhow!("failed to initialize pyo3 static runtime: {:?}", e)
                    })?;
                    Ok(())
                })?;

                Ok(worker.runtime().clone())
            })
            .map_err(to_pyerr)?;

        // Initialize logging in context where tokio runtime is available
        // otel exporter requires it
        if config::env_is_truthy(env_otlp::OTEL_EXPORT_ENABLED) {
            runtime.secondary().block_on(async {
                rs::logging::init();
            });
        }

        let nats_enabled = request_plane.is_nats()
            || matches!(
                event_transport_kind,
                dynamo_runtime::discovery::EventTransportKind::Nats
            )
            || (explicit_event_plane.is_none()
                && std::env::var(config::environment_names::nats::NATS_SERVER).is_ok());

        let runtime_config = DistributedConfig {
            discovery_backend: discovery_backend_config,
            nats_config: if nats_enabled {
                Some(dynamo_runtime::transports::nats::ClientOptions::default())
            } else {
                None
            },
            request_plane,
            event_transport_kind,
        };
        let inner = runtime
            .secondary()
            .block_on(rs::DistributedRuntime::new(runtime, runtime_config))
            .map_err(to_pyerr)?;

        Ok(DistributedRuntime { inner, event_loop })
    }

    #[staticmethod]
    fn detached(py: Python) -> PyResult<Self> {
        let rt = rs::Worker::runtime_from_existing().map_err(to_pyerr)?;
        let handle = rt.primary();

        let inner = handle
            .block_on(rs::DistributedRuntime::from_settings(rt))
            .map_err(to_pyerr)?;

        Ok(DistributedRuntime {
            inner,
            event_loop: py.None(),
        })
    }

    /// Get an endpoint directly by path (e.g., "namespace.component.endpoint" or "dyn://...").
    fn endpoint(&self, path: String) -> PyResult<Endpoint> {
        let trimmed_path = path.trim_start_matches("dyn://");
        let parts: Vec<&str> = trimmed_path.split('.').collect();

        if parts.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid endpoint path '{}'. Expected format: 'namespace.component.endpoint' or 'dyn://namespace.component.endpoint'",
                path
            )));
        }

        let namespace_name = parts[0];
        let component_name = parts[1];
        let endpoint_name = parts[2];

        // Get endpoint using existing chain
        let namespace = self
            .inner
            .namespace(namespace_name.to_string())
            .map_err(to_pyerr)?;
        let component = namespace
            .component(component_name.to_string())
            .map_err(to_pyerr)?;
        let endpoint = component.endpoint(endpoint_name.to_string());

        Ok(Endpoint {
            inner: endpoint,
            event_loop: self.event_loop.clone(),
        })
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }

    fn event_loop(&self) -> PyObject {
        self.event_loop.clone()
    }

    /// Return the local system status server URL if this runtime started one.
    ///
    /// Workers use this in their RL request-plane route descriptor so the
    /// frontend does not need to derive worker system URLs from static env vars.
    fn system_status_server_url(&self) -> Option<String> {
        self.inner.system_status_server_info().map(|info| {
            let socket_addr = info.socket_addr;
            if socket_addr.ip().is_unspecified() {
                let host = dynamo_runtime::utils::ip_resolver::local_ip_for_advertise();
                format!("http://{host}:{}", socket_addr.port())
            } else {
                format!("http://{socket_addr}")
            }
        })
    }

    /// Register an async Python callback for /engine/{route_name}
    ///
    /// Args:
    ///     route_name: Route path (e.g., "control/start_profile" → /engine/control/start_profile)
    ///     callback: Async function with signature: async def(body: dict) -> dict
    ///
    /// Example:
    /// ```python
    /// async def start_profile(body: dict) -> dict:
    ///     await engine.start_profile(**body)
    ///     return {"status": "ok"}
    ///
    /// runtime.register_engine_route("control/start_profile", start_profile)
    /// ```
    #[pyo3(signature = (route_name, callback))]
    fn register_engine_route(
        &self,
        py: Python<'_>,
        route_name: String,
        callback: PyObject,
    ) -> PyResult<()> {
        // Capture TaskLocals at registration time when Python's event loop is running.
        // This is needed because later, when the callback is invoked from an HTTP request,
        // we'll be on a Rust thread without a running Python event loop.
        let locals =
            Arc::new(pyo3_async_runtimes::tokio::get_current_locals(py).map_err(to_pyerr)?);
        let callback = Arc::new(callback);

        // Wrap Python async callback in Rust async closure
        let rust_callback: rs::engine_routes::EngineRouteCallback =
            Arc::new(move |body: serde_json::Value| {
                let callback = callback.clone();
                let locals = locals.clone();

                // Return a boxed future
                Box::pin(async move {
                    // Acquire GIL to call Python callback and convert coroutine to future
                    let py_future = Python::with_gil(|py| {
                        // Convert body to Python dict
                        let py_body = pythonize::pythonize(py, &body).map_err(|e| {
                            anyhow::anyhow!("Failed to convert request body to Python: {}", e)
                        })?;

                        // Call Python async function to get a coroutine
                        let coroutine = callback.call1(py, (py_body,)).map_err(|e| {
                            anyhow::anyhow!("Failed to call Python callback: {}", e)
                        })?;

                        // Use the TaskLocals captured at registration time
                        pyo3_async_runtimes::into_future_with_locals(
                            &locals,
                            coroutine.into_bound(py),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to convert coroutine to future: {}", e)
                        })
                    })?;

                    // Await the Python coroutine (GIL is released during await)
                    let py_result = py_future
                        .await
                        .map_err(|e| anyhow::anyhow!("Python callback failed: {}", e))?;

                    // Convert result back to serde_json::Value
                    Python::with_gil(|py| {
                        pythonize::depythonize::<serde_json::Value>(py_result.bind(py))
                            .map_err(|e| anyhow::anyhow!("Failed to serialize response: {}", e))
                    })
                })
            });

        self.inner
            .engine_routes()
            .register(&route_name, rust_callback);
        tracing::debug!("Registered engine route: /engine/{}", route_name);
        Ok(())
    }

    /// Set the system-level health status (Ready / NotReady).
    fn set_health_status(&self, ready: bool) -> PyResult<()> {
        let status = if ready {
            config::HealthStatus::Ready
        } else {
            config::HealthStatus::NotReady
        };
        self.inner.system_health().lock().set_health_status(status);
        Ok(())
    }

    // This is used to pass the DistributedRuntime from the dynamo-runtime bindings
    // to the KVBM bindings, since KVBM cannot directly use the struct from this cdylib.
    // TODO: Create a separate crate "dynamo-python" so that all binding crates can import
    // from it and share the same crate path. This will allow PyO3 to automatically
    // recognize that both bindings use the same PyClass.
    #[pyo3(name = "to_capsule")]
    fn to_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let arc: Arc<rs::DistributedRuntime> = Arc::new(self.inner.clone());
        let weak: Weak<rs::DistributedRuntime> = Arc::downgrade(&arc);

        let name = CString::new("dynamo.runtime.weak").expect("valid capsule name");

        PyCapsule::new(py, weak, Some(name))
    }
}

#[pymethods]
impl Endpoint {
    #[pyo3(signature = (generator, graceful_shutdown = true, metrics_labels = None, health_check_payload = None))]
    fn serve_endpoint<'p>(
        &self,
        py: Python<'p>,
        generator: PyObject,
        graceful_shutdown: Option<bool>,
        metrics_labels: Option<Vec<(String, String)>>,
        health_check_payload: Option<&Bound<'p, PyDict>>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let engine = Arc::new(engine::PythonAsyncEngine::new(
            generator,
            self.event_loop.clone(),
        )?);
        let ingress = JsonServerStreamingIngress::for_engine(engine.clone()).map_err(to_pyerr)?;

        // Convert Python dict to serde_json::Value if provided and validate it's an object
        let health_payload_json = health_check_payload
            .map(|dict| pythonize::depythonize::<serde_json::Value>(dict))
            .transpose()
            .map_err(|err| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Failed to convert health_check_payload: {}",
                    err
                ))
            })?;

        // Require an object/dict
        if let Some(ref payload) = health_payload_json
            && !payload.is_object()
        {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "health_check_payload must be a JSON object (dict)",
            ));
        }

        let mut builder = self
            .inner
            .endpoint_builder()
            .metrics_labels(metrics_labels)
            .handler(ingress);

        if let Some(payload) = health_payload_json {
            builder = builder.health_check_payload(payload);
        }

        // Register the engine in the local endpoint registry for in-process calls
        builder = builder.register_local_engine(engine).map_err(to_pyerr)?;

        let graceful_shutdown = graceful_shutdown.unwrap_or(true);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            builder
                .graceful_shutdown(graceful_shutdown)
                .start()
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Serve a bidirectional (streaming-input, streaming-output) endpoint.
    ///
    /// The handler is a Python `async def generate(request_stream)` or
    /// `async def generate(request_stream, context)` coroutine that
    /// returns an async generator. `request_stream` is a
    /// [`PyAsyncRequestStream`] yielding inbound frames as plain Python
    /// objects (dicts/lists/etc., the depythonization of
    /// `serde_json::Value`). The generator yields response frames as
    /// plain Python objects that are then pythonized back to JSON values
    /// on the wire.
    ///
    /// Request-stream end (when `__anext__` raises `StopAsyncIteration`)
    /// is *not* a cancellation signal: the caller has merely stopped
    /// sending input. The engine must keep yielding response chunks until
    /// it chooses to return or observes `context.is_stopped()`.
    #[pyo3(signature = (generator, graceful_shutdown = true, metrics_labels = None))]
    fn serve_bidirectional_endpoint<'p>(
        &self,
        py: Python<'p>,
        generator: PyObject,
        graceful_shutdown: Option<bool>,
        metrics_labels: Option<Vec<(String, String)>>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let engine = Arc::new(engine::PythonBidirectionalEngine::new(
            generator,
            self.event_loop.clone(),
        )?);
        let ingress: Arc<JsonBidirectionalIngress> =
            Ingress::for_engine(engine).map_err(to_pyerr)?;

        let builder = self
            .inner
            .endpoint_builder()
            .metrics_labels(metrics_labels)
            .handler(ingress);

        // [gluo FIXME] skipping health check for now:
        // both `health_check_payload` and local in-process engine registration
        // are needed and that requires proper implementation of the bidirectional
        // engine type which is too much for this PR.
        //
        // Enabling them for bidirectional engines would require:
        //   1. A `ManyIn`-typed local-registry slot (e.g. a
        //      `LocalBidirectionalEngine` alias plus a
        //      `register_local_bidirectional_engine` builder method).
        //   2. A canary path that wraps the payload as a single-frame input
        //      stream and reads the first response (the current canary builds
        //      `SingleIn::new(payload)`).
        //   3. A `health_check_payload` that can be handled by the model
        //      (e.g. a realtime `session.update`); otherwise the engine yields
        //      an `error` frame and the probe is marked unhealthy.
        //      This needs extra caring because the bidirectional engine is likely
        //      to be stateful.

        // if let Some(payload) = health_payload_json {
        //     builder = builder.health_check_payload(payload);
        // }

        // // Register the engine in the local endpoint registry for in-process calls
        // builder = builder.register_local_engine(engine).map_err(to_pyerr)?;

        let graceful_shutdown = graceful_shutdown.unwrap_or(true);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            builder
                .graceful_shutdown(graceful_shutdown)
                .start()
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    #[pyo3(signature = (router_mode = None))]
    fn client<'p>(
        &self,
        py: Python<'p>,
        router_mode: Option<RouterMode>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let router_mode = router_mode.unwrap_or(RouterMode::RoundRobin);
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = inner.client().await.map_err(to_pyerr)?;
            let push_router = rs::pipeline::PushRouter::<
                serde_json::Value,
                RsAnnotated<serde_json::Value>,
            >::from_client(client, router_mode.into())
            .await
            .map_err(to_pyerr)?;
            Ok(Client {
                router: push_router,
                endpoint: inner,
            })
        })
    }

    // Opaque unique ID for this worker. May change over worker lifetime.
    fn connection_id(&self) -> u64 {
        self.inner.drt().connection_id()
    }

    /// Get a RuntimeMetrics helper for creating Prometheus metrics
    #[getter]
    fn metrics(&self) -> prometheus_metrics::RuntimeMetrics {
        prometheus_metrics::RuntimeMetrics::from_endpoint(self.inner.clone())
    }

    /// Unregister this endpoint instance from discovery.
    ///
    /// This removes the endpoint from the instances bucket, preventing the router
    /// from sending requests to this worker. Use this when a worker is sleeping
    /// and should not receive any requests.
    fn unregister_endpoint_instance<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .unregister_endpoint_instance()
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Re-register this endpoint instance to discovery.
    ///
    /// This adds the endpoint back to the instances bucket, allowing the router
    /// to send requests to this worker again. Use this when a worker wakes up
    /// and should start receiving requests.
    fn register_endpoint_instance<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.register_endpoint_instance().await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

#[pymethods]
impl ModelCardInstanceId {
    // (namespace, component, endpoint)
    // TODO: Can these be borrowed as &str?
    fn triple(&self) -> (String, String, String) {
        (
            self.inner.namespace.clone(),
            self.inner.component.clone(),
            self.inner.endpoint.clone(),
        )
    }
}

#[pymethods]
impl Client {
    /// Get list of current instances.
    /// Replaces endpoint_ids.
    fn instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids()
    }

    /// Wait for an instance to be available for work.
    /// Replaces wait_for_endpoints.
    fn wait_for_instances<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.router.client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .wait_for_instances()
                .await
                .map(|v| v.into_iter().map(|cei| cei.id()).collect::<Vec<u64>>())
                .map_err(to_pyerr)
        })
    }

    /// Wait for exactly one ready endpoint instance whose MDC runtime_data contains
    /// the requested JSON string value.
    #[pyo3(signature = (key, value, timeout_s=None))]
    fn wait_for_instance_by_runtime_data<'p>(
        &self,
        py: Python<'p>,
        key: String,
        value: String,
        timeout_s: Option<f64>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let endpoint = self.endpoint.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let last_matches = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
            let wait_state = last_matches.clone();
            let error_key = key.clone();
            let error_value = value.clone();
            let wait = async move {
                let mut rx = llm_rs::discovery::runtime_config_watch(&endpoint)
                    .await
                    .map_err(to_pyerr)?;

                loop {
                    let matches: Vec<u64> = rx
                        .borrow_and_update()
                        .iter()
                        .filter_map(|(worker_id, runtime_config)| {
                            let matched = runtime_config
                                .runtime_data
                                .get(&key)
                                .and_then(|value| value.as_str())
                                == Some(value.as_str());
                            matched.then_some(*worker_id)
                        })
                        .collect();

                    if let Ok(mut last) = wait_state.lock() {
                        *last = matches.clone();
                    }

                    if let [worker_id] = matches.as_slice() {
                        return Ok(*worker_id);
                    }

                    rx.changed().await.map_err(to_pyerr)?;
                }
            };

            if let Some(timeout_s) = timeout_s {
                if !timeout_s.is_finite() || timeout_s < 0.0 {
                    return Err(PyValueError::new_err(
                        "timeout_s must be a finite non-negative number",
                    ));
                }
                let timeout = std::time::Duration::from_secs_f64(timeout_s);
                tokio::time::timeout(timeout, wait).await.map_err(|_| {
                    let matches = last_matches
                        .lock()
                        .map(|matches| matches.clone())
                        .unwrap_or_default();
                    PyTimeoutError::new_err(format!(
                        "Timed out waiting for one endpoint instance with runtime_data[{error_key:?}] == {error_value:?}; last_match_count={}, matching_ids={matches:?}",
                        matches.len(),
                    ))
                })?
            } else {
                wait.await
            }
        })
    }

    /// Issue a request to the endpoint using the default routing strategy.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
        context: Option<context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        self.random(py, request, annotated, context)
    }

    /// Send a request to the next endpoint in a round-robin fashion.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn round_robin<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
        context: Option<context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let request_ctx = create_request_context(request, &context);
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = match context {
                Some(context) => {
                    // Always instrument with appropriate span (none if no trace context)
                    let span = get_span_for_context(&context, "round_robin");
                    client
                        .round_robin(request_ctx)
                        .instrument(span)
                        .await
                        .map_err(to_pyerr)?
                }
                _ => client.round_robin(request_ctx).await.map_err(to_pyerr)?,
            };
            tokio::spawn(process_stream(stream, tx));
            Ok(AsyncResponseStream::new(rx, annotated))
        })
    }

    /// Send a request to a random endpoint.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn random<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
        context: Option<context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let request_ctx = create_request_context(request, &context);
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = match context {
                Some(context) => {
                    // Always instrument with appropriate span (none if no trace context)
                    let span = get_span_for_context(&context, "random");
                    client
                        .random(request_ctx)
                        .instrument(span)
                        .await
                        .map_err(to_pyerr)?
                }
                _ => client.random(request_ctx).await.map_err(to_pyerr)?,
            };
            tokio::spawn(process_stream(stream, tx));
            Ok(AsyncResponseStream::new(rx, annotated))
        })
    }

    /// Send a request using device-aware weighted routing.
    /// Preferentially routes to GPU (CUDA) workers; CPU workers receive overflow
    /// only when GPU workers are sufficiently loaded (controlled by DYN_ENCODER_CUDA_TO_CPU_RATIO).
    /// With the default ratio of 8, all requests go to GPU workers unless they are
    /// handling 8x more load than CPU workers.
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn device_aware_weighted<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        annotated: Option<bool>,
        context: Option<context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let request_ctx = create_request_context(request, &context);
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = match context {
                Some(context) => {
                    let span = get_span_for_context(&context, "device_aware_weighted");
                    client
                        .device_aware_weighted(request_ctx)
                        .instrument(span)
                        .await
                        .map_err(to_pyerr)?
                }
                _ => client
                    .device_aware_weighted(request_ctx)
                    .await
                    .map_err(to_pyerr)?,
            };
            tokio::spawn(process_stream(stream, tx));
            Ok(AsyncResponseStream::new(rx, annotated))
        })
    }

    /// Directly send a request to a specific endpoint.
    #[pyo3(signature = (request, instance_id, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn direct<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
        instance_id: u64,
        annotated: Option<bool>,
        context: Option<context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: serde_json::Value = pythonize::depythonize(&request.into_bound(py))?;
        let request_ctx = create_request_context(request, &context);
        let annotated = annotated.unwrap_or(false);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let client = self.router.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = match context {
                Some(context) => {
                    // Always instrument with appropriate span (none if no trace context)
                    let span =
                        get_span_for_direct_context(&context, "direct", &instance_id.to_string());
                    client
                        .direct(request_ctx, instance_id)
                        .instrument(span)
                        .await
                        .map_err(to_pyerr)?
                }
                _ => client
                    .direct(request_ctx, instance_id)
                    .await
                    .map_err(to_pyerr)?,
            };

            tokio::spawn(process_stream(stream, tx));

            Ok(AsyncResponseStream::new(rx, annotated))
        })
    }
}

async fn process_stream(
    stream: EngineStream<RsAnnotated<serde_json::Value>>,
    tx: tokio::sync::mpsc::Sender<RsAnnotated<PyObject>>,
) {
    let mut stream = stream;
    while let Some(response) = stream.next().await {
        // Convert the response to a PyObject using Python's GIL
        let annotated: RsAnnotated<serde_json::Value> = response;
        let annotated: RsAnnotated<PyObject> = annotated.map_data(|data| {
            Python::with_gil(|py| match pythonize::pythonize(py, &data) {
                Ok(pyobj) => Ok(pyobj.into()),
                Err(e) => Err(e.to_string()),
            })
        });

        let is_error = annotated.is_error();

        // Send the PyObject through the channel or log an error
        if let Err(e) = tx.send(annotated).await {
            tracing::error!("Failed to send response: {:?}", e);
            break;
        }

        if is_error {
            break;
        }
    }
}

#[pyclass]
pub(crate) struct AsyncResponseStream {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>>>,
    annotated: bool,
}

impl AsyncResponseStream {
    pub(crate) fn new(
        rx: tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>,
        annotated: bool,
    ) -> Self {
        Self {
            rx: Arc::new(Mutex::new(rx)),
            annotated,
        }
    }
}

#[pymethods]
impl AsyncResponseStream {
    /// This method is required to implement the `AsyncIterator` protocol.
    #[pyo3(name = "__aiter__")]
    fn aiter(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        slf.into_py_any(py)
    }
    /// This method is required to implement the `AsyncIterator` protocol.
    #[pyo3(name = "__anext__")]
    fn next<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let rx = self.rx.clone();
        let annotated = self.annotated;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                let value = rx.lock().await.recv().await;
                match value {
                    Some(pyobj) => {
                        let pyobj = match pyobj.ok() {
                            Ok(pyobj) => pyobj,
                            Err(e) => {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e));
                            }
                        };

                        if annotated {
                            let object = Annotated { inner: pyobj };
                            #[allow(deprecated)]
                            let object = Python::with_gil(|py| object.into_py(py));
                            return Ok(object);
                        } else {
                            match pyobj.data {
                                Some(data) => return Ok(data),
                                None => continue,
                            }
                        }
                    }
                    None => return Err(PyStopAsyncIteration::new_err("Stream exhausted")),
                }
            }
        })
    }
}

/// Python-visible inbound iterator for bidirectional engines. Wraps an
/// mpsc receiver of pre-pythonized request frames; `__anext__` is a thin
/// `.recv()` that returns the next `PyObject` directly, with no per-frame
/// GIL acquisition on the consumer side. The producer (the forwarder
/// spawned by `PythonBidirectionalEngine::generate`) acquires the GIL
/// once per frame and pushes the converted `PyObject` onto the channel.
///
/// Termination follows the same shape as `AsyncResponseStream`: when the
/// channel returns `None`, `__anext__` raises `PyStopAsyncIteration` and
/// the iterator is exhausted. Note that input-stream end is *not* a
/// cancellation signal — engines must keep yielding response chunks until
/// they decide to return or observe `context.is_stopped()`.
#[pyclass]
pub(crate) struct PyAsyncRequestStream {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<PyObject>>>,
}

impl PyAsyncRequestStream {
    pub(crate) fn new(rx: tokio::sync::mpsc::Receiver<PyObject>) -> Self {
        Self {
            rx: Arc::new(Mutex::new(rx)),
        }
    }
}

#[pymethods]
impl PyAsyncRequestStream {
    /// Required by the `AsyncIterator` protocol.
    #[pyo3(name = "__aiter__")]
    fn aiter(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        slf.into_py_any(py)
    }

    /// Required by the `AsyncIterator` protocol. Returns an awaitable
    /// resolving to the next pre-pythonized frame, or raises
    /// `StopAsyncIteration` when the inbound channel is closed.
    #[pyo3(name = "__anext__")]
    fn next<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let rx = self.rx.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match rx.lock().await.recv().await {
                Some(pyobj) => Ok(pyobj),
                None => Err(PyStopAsyncIteration::new_err("Request stream exhausted")),
            }
        })
    }
}

#[pyclass]
struct Annotated {
    inner: RsAnnotated<PyObject>,
}

#[pymethods]
impl Annotated {
    #[new]
    fn new(data: PyObject) -> Self {
        Annotated {
            inner: RsAnnotated::from_data(data),
        }
    }

    fn is_error(&self) -> bool {
        self.inner.is_error()
    }

    fn data(&self) -> Option<PyObject> {
        self.inner.data.clone()
    }

    fn event(&self) -> Option<String> {
        self.inner.event.clone()
    }

    fn comments(&self) -> Option<Vec<String>> {
        self.inner.comment.clone()
    }

    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[pyo3(name = "__repr__")]
    fn _repr(&self, py: Python) -> String {
        let data = self.inner.data.clone().map(|obj| {
            obj.call_method0(py, "__repr__")
                .and_then(|repr_obj| repr_obj.extract::<Py<PyString>>(py))
                .map(|py_str| py_str.to_string_lossy(py).into_owned())
                .unwrap_or_else(|_| "<failed_repr>".to_string())
        });

        format!(
            "Annotated(data={}, event={}, comment={:?}, id={})",
            data.unwrap_or_else(|| "<no_data>".to_string()),
            self.inner.event.as_deref().unwrap_or("None"),
            self.inner.comment.as_deref().unwrap_or(&[]),
            self.inner.id.as_deref().unwrap_or("None")
        )
    }
}
