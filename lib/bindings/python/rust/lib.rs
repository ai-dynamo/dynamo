// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::local_model::LocalModel;
use dynamo_runtime::distributed::{DistributedConfig, RequestPlaneMode};
use dynamo_runtime::storage::key_value_store::KeyValueStoreSelect;
use futures::StreamExt;
use once_cell::sync::OnceCell;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyStopAsyncIteration;
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

use dynamo_runtime::{
    self as rs, logging,
    pipeline::{
        AsyncEngineContextProvider, EngineStream, ManyOut, SingleIn, context::Context as RsContext,
        network::egress::push_router::RouterMode as RsRouterMode,
    },
    protocols::annotated::Annotated as RsAnnotated,
    traits::DistributedRuntimeProvider,
};

use dynamo_llm::{self as llm_rs};
use dynamo_llm::{entrypoint::RouterConfig, kv_router::KvRouterConfig};

use crate::llm::local_model::ModelRuntimeConfig;
use crate::llm::preprocessor::{MediaDecoder, MediaFetcher};

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum RouterMode {
    RoundRobin,
    Random,
    KV,
}

impl From<RouterMode> for RsRouterMode {
    fn from(mode: RouterMode) -> Self {
        match mode {
            RouterMode::RoundRobin => Self::RoundRobin,
            RouterMode::Random => Self::Random,
            RouterMode::KV => Self::KV,
        }
    }
}

mod context;
mod engine;
mod http;
mod kserve_grpc;
mod llm;
mod parsers;
mod planner;
mod prometheus_metrics;

type JsonServerStreamingIngress =
    Ingress<SingleIn<serde_json::Value>, ManyOut<RsAnnotated<serde_json::Value>>>;

static INIT: OnceCell<()> = OnceCell::new();

const DEFAULT_ANNOTATED_SETTING: Option<bool> = Some(true);

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
            let child_ctx = RsContext::with_id(request, parent_ctx.inner().id().to_string());
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
    if rs::config::env_is_truthy("OTEL_EXPORT_ENABLED") {
        eprintln!(
            "Warning: OTEL_EXPORT_ENABLED detected. Logging initialization deferred until runtime is available. Early logs may be dropped."
        );
    } else {
        rs::logging::init();
    }

    m.add_function(wrap_pyfunction!(llm::kv::compute_block_hash_for_seq_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_message, m)?)?;
    m.add_function(wrap_pyfunction!(register_llm, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_llm, m)?)?;
    m.add_function(wrap_pyfunction!(llm::entrypoint::make_engine, m)?)?;
    m.add_function(wrap_pyfunction!(llm::entrypoint::run_input, m)?)?;

    m.add_class::<DistributedRuntime>()?;
    m.add_class::<CancellationToken>()?;
    m.add_class::<Namespace>()?;
    m.add_class::<Component>()?;
    m.add_class::<Endpoint>()?;
    m.add_class::<Client>()?;
    m.add_class::<AsyncResponseStream>()?;
    m.add_class::<llm::entrypoint::EntrypointArgs>()?;
    m.add_class::<llm::entrypoint::EngineConfig>()?;
    m.add_class::<llm::entrypoint::EngineType>()?;
    m.add_class::<llm::entrypoint::RouterConfig>()?;
    m.add_class::<llm::entrypoint::KvRouterConfig>()?;
    m.add_class::<llm::kv::WorkerMetricsPublisher>()?;
    m.add_class::<llm::model_card::ModelDeploymentCard>()?;
    m.add_class::<llm::local_model::ModelRuntimeConfig>()?;
    m.add_class::<llm::preprocessor::OAIChatPreprocessor>()?;
    m.add_class::<llm::preprocessor::MediaDecoder>()?;
    m.add_class::<llm::preprocessor::MediaFetcher>()?;
    m.add_class::<llm::backend::Backend>()?;
    m.add_class::<llm::kv::OverlapScores>()?;
    m.add_class::<llm::kv::KvIndexer>()?;
    m.add_class::<llm::kv::ApproxKvIndexer>()?;
    m.add_class::<llm::kv::KvEventPublisher>()?;
    m.add_class::<llm::kv::RadixTree>()?;
    m.add_class::<llm::kv::ZmqKvEventListener>()?;
    m.add_class::<llm::kv::ZmqKvEventPublisher>()?;
    m.add_class::<llm::kv::ZmqKvEventPublisherConfig>()?;
    m.add_class::<llm::kv::KvRecorder>()?;
    m.add_class::<http::HttpService>()?;
    m.add_class::<http::HttpAsyncEngine>()?;
    m.add_class::<context::Context>()?;
    m.add_class::<ModelType>()?;
    m.add_class::<ModelInput>()?;
    m.add_class::<llm::kv::ForwardPassMetrics>()?;
    m.add_class::<llm::kv::WorkerStats>()?;
    m.add_class::<llm::kv::KvStats>()?;
    m.add_class::<llm::kv::SpecDecodeStats>()?;
    m.add_class::<llm::kv::KvPushRouter>()?;
    m.add_class::<llm::kv::KvPushRouterStream>()?;
    m.add_class::<RouterMode>()?;
    m.add_class::<kserve_grpc::KserveGrpcService>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<planner::VirtualConnectorCoordinator>()?;
    m.add_class::<planner::VirtualConnectorClient>()?;
    m.add_class::<planner::PlannerDecision>()?;

    engine::add_to_module(m)?;
    parsers::add_to_module(m)?;

    m.add_class::<prometheus_metrics::RuntimeMetrics>()?;
    let prometheus_metrics = PyModule::new(m.py(), "prometheus_metrics")?;
    prometheus_metrics::add_to_module(&prometheus_metrics)?;
    m.add_submodule(&prometheus_metrics)?;

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}

/// Log a message from Python with file and line info
#[pyfunction]
#[pyo3(text_signature = "(level, message, module, file, line)")]
fn log_message(level: &str, message: &str, module: &str, file: &str, line: u32) {
    logging::log_message(level, message, module, file, line);
}

/// Create an engine and attach it to an endpoint to make it visible to the frontend.
/// This is the main way you create a Dynamo worker / backend.
#[pyfunction]
#[pyo3(signature = (model_input, model_type, endpoint, model_path, model_name=None, context_length=None, kv_cache_block_size=None, router_mode=None, migration_limit=0, runtime_config=None, user_data=None, custom_template_path=None, media_decoder=None, media_fetcher=None))]
#[allow(clippy::too_many_arguments)]
fn register_llm<'p>(
    py: Python<'p>,
    model_input: ModelInput,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: &str,
    model_name: Option<&str>,
    context_length: Option<u32>,
    kv_cache_block_size: Option<u32>,
    router_mode: Option<RouterMode>,
    migration_limit: u32,
    runtime_config: Option<ModelRuntimeConfig>,
    user_data: Option<&Bound<'p, PyDict>>,
    custom_template_path: Option<&str>,
    media_decoder: Option<MediaDecoder>,
    media_fetcher: Option<MediaFetcher>,
) -> PyResult<Bound<'p, PyAny>> {
    // Validate Prefill model type requirements
    if model_type.inner == llm_rs::model_type::ModelType::Prefill {
        if !matches!(model_input, ModelInput::Tokens) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ModelType::Prefill requires model_input to be ModelInput::Tokens",
            ));
        }
        if migration_limit != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ModelType::Prefill requires migration_limit to be 0",
            ));
        }
    }

    let model_input = match model_input {
        ModelInput::Text => llm_rs::model_type::ModelInput::Text,
        ModelInput::Tokens => llm_rs::model_type::ModelInput::Tokens,
        ModelInput::Tensor => llm_rs::model_type::ModelInput::Tensor,
    };

    let model_type_obj = model_type.inner;

    let inner_path = model_path.to_string();
    let mut model_name = model_name.map(|n| n.to_string());
    let router_mode = router_mode.unwrap_or(RouterMode::RoundRobin);
    let router_config = RouterConfig::new(router_mode.into(), KvRouterConfig::default());

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

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let model_path = if fs::exists(&inner_path)? {
            PathBuf::from(inner_path)
        } else {
            // Preserve the model name
            if model_name.is_none() {
                model_name = Some(inner_path.clone());
            }
            // Likely it's a Hugging Face repo, download it
            LocalModel::fetch(&inner_path, false)
                .await
                .map_err(to_pyerr)?
        };

        let mut builder = dynamo_llm::local_model::LocalModelBuilder::default();
        builder
            .model_path(model_path)
            .model_name(model_name)
            .context_length(context_length)
            .kv_cache_block_size(kv_cache_block_size)
            .router_config(Some(router_config))
            .migration_limit(Some(migration_limit))
            .runtime_config(runtime_config.unwrap_or_default().inner)
            .user_data(user_data_json)
            .custom_template_path(custom_template_path_owned)
            .media_decoder(media_decoder.map(|m| m.inner))
            .media_fetcher(media_fetcher.map(|m| m.inner));
        // Load the ModelDeploymentCard
        let mut local_model = builder.build().await.map_err(to_pyerr)?;
        // Advertise ourself so ingress can find us
        local_model
            .attach(&endpoint.inner, model_type_obj, model_input)
            .await
            .map_err(to_pyerr)?;

        Ok(())
    })
}

/// Download a model from Hugging Face, returning it's local path
/// Example: `model_path = await fetch_llm("Qwen/Qwen3-0.6B")`
#[pyfunction]
#[pyo3(signature = (remote_name))]
fn fetch_llm<'p>(py: Python<'p>, remote_name: &str) -> PyResult<Bound<'p, PyAny>> {
    let repo = remote_name.to_string();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        LocalModel::fetch(&repo, false).await.map_err(to_pyerr)
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
struct Namespace {
    inner: rs::component::Namespace,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Component {
    inner: rs::component::Component,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Endpoint {
    inner: rs::component::Endpoint,
    event_loop: PyObject,
}

#[pyclass]
#[derive(Clone)]
struct Client {
    router: rs::pipeline::PushRouter<serde_json::Value, RsAnnotated<serde_json::Value>>,
}

#[pyclass]
#[derive(Clone, PartialEq)]
struct ModelType {
    inner: llm_rs::model_type::ModelType,
}

#[pymethods]
#[allow(non_upper_case_globals)]
impl ModelType {
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
    #[classattr]
    const Prefill: Self = ModelType {
        inner: llm_rs::model_type::ModelType::Prefill,
    };

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

#[pymethods]
impl DistributedRuntime {
    #[new]
    fn new(event_loop: PyObject, store_kv: String, request_plane: String) -> PyResult<Self> {
        let selected_kv_store: KeyValueStoreSelect = store_kv.parse().map_err(to_pyerr)?;
        let request_plane: RequestPlaneMode = request_plane.parse().map_err(to_pyerr)?;

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
        if rs::config::env_is_truthy("OTEL_EXPORT_ENABLED") {
            runtime.secondary().block_on(async {
                rs::logging::init();
            });
        }

        let runtime_config = DistributedConfig {
            store_backend: selected_kv_store,
            nats_config: dynamo_runtime::transports::nats::ClientOptions::default(),
            request_plane,
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

    fn namespace(&self, name: String) -> PyResult<Namespace> {
        Ok(Namespace {
            inner: self.inner.namespace(name).map_err(to_pyerr)?,
            event_loop: self.event_loop.clone(),
        })
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }

    fn event_loop(&self) -> PyObject {
        self.event_loop.clone()
    }

    fn child_token(&self) -> CancellationToken {
        let inner = self.inner.runtime().child_token();
        CancellationToken { inner }
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
impl CancellationToken {
    fn cancel(&self) {
        self.inner.cancel();
    }

    fn cancelled<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let token = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            token.cancelled().await;
            Ok(())
        })
    }
}

#[pymethods]
impl Component {
    fn endpoint(&self, name: String) -> PyResult<Endpoint> {
        let inner = self.inner.endpoint(name);
        Ok(Endpoint {
            inner,
            event_loop: self.event_loop.clone(),
        })
    }

    /// NATS specific stats/metrics call
    fn create_service<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let mut inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.add_stats_service().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Get a RuntimeMetrics helper for creating Prometheus metrics
    #[getter]
    fn metrics(&self) -> prometheus_metrics::RuntimeMetrics {
        prometheus_metrics::RuntimeMetrics::from_component(self.inner.clone())
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
        let ingress = JsonServerStreamingIngress::for_engine(engine).map_err(to_pyerr)?;

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

    fn client<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = inner.client().await.map_err(to_pyerr)?;
            let push_router = rs::pipeline::PushRouter::<
                serde_json::Value,
                RsAnnotated<serde_json::Value>,
            >::from_client(client, Default::default())
            .await
            .map_err(to_pyerr)?;
            Ok(Client {
                router: push_router,
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
}

#[pymethods]
impl Namespace {
    fn component(&self, name: String) -> PyResult<Component> {
        let inner = self.inner.component(name).map_err(to_pyerr)?;
        Ok(Component {
            inner,
            event_loop: self.event_loop.clone(),
        })
    }

    /// Get a RuntimeMetrics helper for creating Prometheus metrics
    #[getter]
    fn metrics(&self) -> prometheus_metrics::RuntimeMetrics {
        prometheus_metrics::RuntimeMetrics::from_namespace(self.inner.clone())
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
            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
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
            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
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

            Ok(AsyncResponseStream {
                rx: Arc::new(Mutex::new(rx)),
                annotated,
            })
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
struct AsyncResponseStream {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>>>,
    annotated: bool,
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
