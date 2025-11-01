// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::local_model::LocalModel;
use futures::StreamExt;
use once_cell::sync::OnceCell;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::types::PyCapsule;
use pyo3::types::{PyDict, PyString};
use pyo3::{exceptions::PyException, prelude::*};
use rand::seq::IteratorRandom as _;
use rs::pipeline::network::Ingress;
use std::ffi::CString;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::time::Duration;
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
    if std::env::var("OTEL_EXPORT_ENABLED")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!(
            "Warning: OTEL_EXPORT_ENABLED=1 detected. Logging initialization deferred until runtime is available. Early logs may be dropped."
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
    m.add_class::<llm::disagg_router::DisaggregatedRouter>()?;
    m.add_class::<llm::entrypoint::EntrypointArgs>()?;
    m.add_class::<llm::entrypoint::EngineConfig>()?;
    m.add_class::<llm::entrypoint::EngineType>()?;
    m.add_class::<llm::entrypoint::RouterConfig>()?;
    m.add_class::<llm::entrypoint::KvRouterConfig>()?;
    m.add_class::<llm::kv::WorkerMetricsPublisher>()?;
    m.add_class::<llm::model_card::ModelDeploymentCard>()?;
    m.add_class::<llm::local_model::ModelRuntimeConfig>()?;
    m.add_class::<llm::preprocessor::OAIChatPreprocessor>()?;
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
#[pyo3(signature = (model_input, model_type, endpoint, model_path, model_name=None, context_length=None, kv_cache_block_size=None, router_mode=None, migration_limit=0, runtime_config=None, user_data=None, custom_template_path=None))]
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
            .custom_template_path(custom_template_path_owned);
        // Load the ModelDeploymentCard
        let mut local_model = builder.build().await.map_err(to_pyerr)?;
        // Advertise ourself on etcd so ingress can find us
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
    fn new(event_loop: PyObject, is_static: bool) -> PyResult<Self> {
        // Try to get existing runtime first, create new Worker only if needed
        // This allows multiple DistributedRuntime instances to share the same tokio runtime
        let runtime = rs::Worker::runtime_from_existing()
            .or_else(|_| {
                // No existing Worker, create new one
                let worker = rs::Worker::from_settings()?;

                // Initialize pyo3 bridge (only happens once per process)
                INIT.get_or_try_init(|| {
                    let primary = worker.tokio_runtime()?;
                    pyo3_async_runtimes::tokio::init_with_runtime(primary).map_err(|e| {
                        rs::error!("failed to initialize pyo3 static runtime: {:?}", e)
                    })?;
                    rs::OK(())
                })?;

                rs::OK(worker.runtime().clone())
            })
            .map_err(to_pyerr)?;

        // Initialize logging in context where tokio runtime is available
        // otel exporter requires it
        if std::env::var("OTEL_EXPORT_ENABLED")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            runtime.secondary().block_on(async {
                rs::logging::init();
            });
        }

        let inner =
            if is_static {
                runtime.secondary().block_on(
                    rs::DistributedRuntime::from_settings_without_discovery(runtime),
                )
            } else {
                runtime
                    .secondary()
                    .block_on(rs::DistributedRuntime::from_settings(runtime))
            };
        let inner = inner.map_err(to_pyerr)?;

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

    /// Allocate a contiguous block of ports from the specified range and atomically reserve them.
    /// Returns a list of all allocated ports in order.
    #[pyo3(signature = (namespace, port_min, port_max, block_size, context=None))]
    fn allocate_port_block<'p>(
        &self,
        py: Python<'p>,
        namespace: &str,
        port_min: u16,
        port_max: u16,
        block_size: u16,
        context: Option<String>, // Optional info to store alongside the reservation
    ) -> PyResult<Bound<'p, PyAny>> {
        const MAX_ALLOCATE_ATTEMPTS: usize = 100;
        if block_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Block size must be at least 1",
            ));
        }

        let Some(etcd_client) = self.inner.etcd_client() else {
            return Err(PyErr::new::<PyException, _>(
                "Static workers should not need to reserve ports",
            ));
        };

        let min = port_min;
        let max = port_max;

        // Compute maximum valid starting port (inclusive)
        let max_start_port = max.saturating_sub(block_size.saturating_sub(1));
        if max_start_port < min {
            return Err(PyErr::new::<PyException, _>(format!(
                "Port range {min}-{max} is too small for block size {block_size}",
            )));
        }

        // Randomize candidate starting ports to reduce contention/races
        let candidate_count =
            (max_start_port - port_min + 1).min(MAX_ALLOCATE_ATTEMPTS as u16) as usize;
        let mut rng = rand::rng();
        let candidate_ports: Vec<u16> =
            (port_min..=max_start_port).choose_multiple(&mut rng, candidate_count);

        let local_ip = match local_ip() {
            Ok(ip) => ip,
            Err(err) => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "Failed fetching local IP address: {err}"
                )));
            }
        };

        let context_bytes = context.map(|s| s.as_bytes().to_vec()).unwrap_or_default();
        let namespace = namespace.to_owned();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            for (attempt_idx, start_port) in candidate_ports.into_iter().enumerate() {
                let end_port_exclusive = start_port + block_size;
                let ports_to_reserve: Vec<u16> = (start_port..end_port_exclusive).collect();

                // Hold/bind all ports in the block
                let mut sockets = Vec::with_capacity(ports_to_reserve.len());
                let mut bind_failed = false;

                for &port in &ports_to_reserve {
                    match bind_tcp_port(port) {
                        Ok(sock) => sockets.push(sock),
                        Err(e) => {
                            tracing::error!(
                                "Failed to bind to port block starting at {start_port} (attempt {}): {e}",
                                attempt_idx + 1,
                            );
                            bind_failed = true;
                            break;
                        }
                    }
                }

                if bind_failed {
                    // Let previously bound sockets drop here
                    if attempt_idx < candidate_count - 1 {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    continue;
                }

                // With sockets held, reserve in ETCD
                let mut reserved_keys = Vec::with_capacity(ports_to_reserve.len());
                let mut reservation_failed = false;
                for port in &ports_to_reserve {
                    let key = make_port_key(&namespace, local_ip, *port).map_err(to_pyerr)?;

                    if let Err(e) = etcd_client
                        .kv_create(&key, context_bytes.clone(), None)
                        .await
                    {
                        tracing::error!(
                            "Failed to reserve port block starting at {start_port} (attempt {}): {e}",
                            attempt_idx + 1,
                        );
                        reservation_failed = true;
                        break;
                    }
                    reserved_keys.push(key);
                }

                if reservation_failed {
                    // Cleanup partial reservations
                    for key in reserved_keys {
                        if let Err(e) = etcd_client.kv_delete(key.as_str(), None).await {
                            tracing::warn!("Failed to cleanup reserved port {key}: {e}");
                        }
                    }

                    // Sockets automatically released via RAII
                    if attempt_idx < candidate_count - 1 {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    continue;
                }

                // Success - sockets will be released automatically
                tracing::debug!("Reserved port block {ports_to_reserve:?}");
                return Ok(ports_to_reserve);
            }

            Err(PyErr::new::<PyException, _>(format!(
                "Failed to allocate and reserve a port block of size {block_size} from range {min}-{max} after {candidate_count} attempts"
            )))
        })
    }

    fn shutdown(&self) {
        self.inner.runtime().shutdown();
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

// Bind a TCP port and return a socket held until dropped.
fn bind_tcp_port(port: u16) -> std::io::Result<socket2::Socket> {
    let sock = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    sock.set_reuse_address(true)?;
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    sock.bind(&addr.into())?;
    Ok(sock)
}

fn make_port_key(namespace: &str, node_ip: IpAddr, port: u16) -> anyhow::Result<String> {
    Ok(format!("v1/{namespace}/ports/{node_ip}/{port}"))
}

fn local_ip() -> Result<IpAddr, local_ip_address::Error> {
    local_ip_address::local_ip().or_else(|err| match err {
        local_ip_address::Error::LocalIpAddressNotFound => {
            // Fall back to IPv6 if no IPv4 addresses are found
            local_ip_address::local_ipv6()
        }
        _ => Err(err),
    })
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
        if self.router.client.is_static() {
            self.r#static(py, request, annotated, context)
        } else {
            self.random(py, request, annotated, context)
        }
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

    /// Directly send a request to a pre-defined static worker
    #[pyo3(signature = (request, annotated=DEFAULT_ANNOTATED_SETTING, context=None))]
    fn r#static<'p>(
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
                    let span = get_span_for_context(&context, "static");
                    client
                        .r#static(request_ctx)
                        .instrument(span)
                        .await
                        .map_err(to_pyerr)?
                }
                _ => client.r#static(request_ctx).await.map_err(to_pyerr)?,
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
