// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bridge for `dynamo_backend_common::Worker`.
//!
//! Lets a Python `LLMEngine` ABC subclass plug into the Rust `Worker`
//! through a thin `PyLLMEngine` adapter. All lifecycle work — signal
//! handling, discovery unregister, grace period, drain, cleanup, and
//! 3-phase runtime shutdown — lives in Rust; Python only owns engine
//! semantics.
//!
//! Exposed under `dynamo._core.backend` as `Worker`, `WorkerConfig`,
//! `EngineConfig`, and `RuntimeConfig`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, Capability as RsCapability, ComponentSnapshot,
    DisaggregationMode as RsDisaggregationMode, DynamoError, EngineConfig as RsEngineConfig,
    ErrorType, KvEventSource as RsKvEventSource, LLMEngine, LLMEngineOutput, MetricsBindings,
    MetricsCtx, OnPublisherReady, PreprocessedRequest, RuntimeConfig as RsRuntimeConfig,
    SnapshotPublisher as RsSnapshotPublisher,
    UnsupportedFieldPolicy as RsUnsupportedFieldPolicy, Worker as RsWorker,
    WorkerConfig as RsWorkerConfig, list_request_fields as rs_list_request_fields,
};
use dynamo_llm::local_model::runtime_config::{
    StructuralTagMode as RsStructuralTagMode, StructuralTagSchemaMode as RsStructuralTagSchemaMode,
    StructuralTagScope as RsStructuralTagScope,
};
use dynamo_llm::model_type::ModelInput as RsModelInput;
use dynamo_runtime as rs;
use dynamo_runtime::logging::{DistributedTraceContext, get_distributed_tracing_context};
use futures::stream::{BoxStream, StreamExt};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};

use crate::ModelInput;
use crate::context::Context as PyContext;
use crate::errors::py_exception_to_backend_error;
use crate::llm::kv::KvEventPublisher as PyKvEventPublisher;
use crate::to_pyerr;

/// Snapshot of the unified-backend schema registry: `(field_name, status)`
/// pairs where status is one of `"supported"`, `"experimental"`,
/// `"forwarded"`. Used by Python tests/tooling to introspect the
/// contract without re-encoding the list.
#[pyfunction]
fn list_request_fields() -> Vec<(String, String)> {
    rs_list_request_fields()
        .into_iter()
        .map(|(name, status)| (name.to_string(), status.to_string()))
        .collect()
}

/// Register `dynamo._core.backend` and its classes on the parent `_core` module.
pub fn add_to_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "backend")?;
    m.add_class::<Capability>()?;
    m.add_class::<DisaggregationMode>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<UnsupportedFieldPolicy>()?;
    m.add_class::<WorkerConfig>()?;
    m.add_class::<Worker>()?;
    m.add_class::<PySnapshotPublisher>()?;
    m.add_class::<crate::prometheus_metrics::EngineMetrics>()?;
    m.add("HEALTH_CHECK_KEY", dynamo_backend_common::HEALTH_CHECK_KEY)?;
    m.add_function(wrap_pyfunction!(list_request_fields, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("dynamo._core.backend", &m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Capability — mirror of `dynamo_backend_common::schema::Capability`.
// ---------------------------------------------------------------------------

#[pyclass(
    module = "dynamo._core.backend",
    name = "Capability",
    eq,
    eq_int,
    hash,
    frozen
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    PromptEmbeds = 1,
    MultiModalData = 2,
    MmRoutingInfo = 3,
    MmProcessorKwargs = 4,
    RouterConfigOverride = 5,
    AgentContext = 6,
    ExtraArgs = 7,
}

impl From<Capability> for RsCapability {
    fn from(value: Capability) -> Self {
        match value {
            Capability::PromptEmbeds => RsCapability::PromptEmbeds,
            Capability::MultiModalData => RsCapability::MultiModalData,
            Capability::MmRoutingInfo => RsCapability::MmRoutingInfo,
            Capability::MmProcessorKwargs => RsCapability::MmProcessorKwargs,
            Capability::RouterConfigOverride => RsCapability::RouterConfigOverride,
            Capability::AgentContext => RsCapability::AgentContext,
            Capability::ExtraArgs => RsCapability::ExtraArgs,
        }
    }
}

impl From<RsCapability> for Capability {
    fn from(value: RsCapability) -> Self {
        match value {
            RsCapability::PromptEmbeds => Capability::PromptEmbeds,
            RsCapability::MultiModalData => Capability::MultiModalData,
            RsCapability::MmRoutingInfo => Capability::MmRoutingInfo,
            RsCapability::MmProcessorKwargs => Capability::MmProcessorKwargs,
            RsCapability::RouterConfigOverride => Capability::RouterConfigOverride,
            RsCapability::AgentContext => Capability::AgentContext,
            RsCapability::ExtraArgs => Capability::ExtraArgs,
        }
    }
}

// ---------------------------------------------------------------------------
// UnsupportedFieldPolicy — mirror of
// `dynamo_backend_common::schema::UnsupportedFieldPolicy`.
// ---------------------------------------------------------------------------

#[pyclass(
    module = "dynamo._core.backend",
    name = "UnsupportedFieldPolicy",
    eq,
    eq_int
)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum UnsupportedFieldPolicy {
    Reject = 1,
    #[default]
    Warn = 2,
    Ignore = 3,
}

impl From<UnsupportedFieldPolicy> for RsUnsupportedFieldPolicy {
    fn from(value: UnsupportedFieldPolicy) -> Self {
        match value {
            UnsupportedFieldPolicy::Reject => RsUnsupportedFieldPolicy::Reject,
            UnsupportedFieldPolicy::Warn => RsUnsupportedFieldPolicy::Warn,
            UnsupportedFieldPolicy::Ignore => RsUnsupportedFieldPolicy::Ignore,
        }
    }
}

impl From<RsUnsupportedFieldPolicy> for UnsupportedFieldPolicy {
    fn from(value: RsUnsupportedFieldPolicy) -> Self {
        match value {
            RsUnsupportedFieldPolicy::Reject => UnsupportedFieldPolicy::Reject,
            RsUnsupportedFieldPolicy::Warn => UnsupportedFieldPolicy::Warn,
            RsUnsupportedFieldPolicy::Ignore => UnsupportedFieldPolicy::Ignore,
        }
    }
}

// ---------------------------------------------------------------------------
// DisaggregationMode — mirror of `dynamo_backend_common::DisaggregationMode`.
//
// Variant names and integer values are stable wire format for the Python
// side. `eq_int` enables `mode == DisaggregationMode.Prefill` plus integer
// comparisons in tests.
// ---------------------------------------------------------------------------

#[pyclass(
    module = "dynamo._core.backend",
    name = "DisaggregationMode",
    eq,
    eq_int
)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum DisaggregationMode {
    #[default]
    Aggregated = 1,
    Prefill = 2,
    Decode = 3,
}

impl From<DisaggregationMode> for RsDisaggregationMode {
    fn from(value: DisaggregationMode) -> Self {
        match value {
            DisaggregationMode::Aggregated => RsDisaggregationMode::Aggregated,
            DisaggregationMode::Prefill => RsDisaggregationMode::Prefill,
            DisaggregationMode::Decode => RsDisaggregationMode::Decode,
        }
    }
}

// ---------------------------------------------------------------------------
// EngineConfig — mirror of `dynamo_backend_common::EngineConfig`.
//
// Engines are free to return either a `dynamo._core.backend.EngineConfig`
// or any plain Python dataclass with the canonical attribute names; the
// bridge accepts both. We expose this pyclass mainly so engines that want
// strong typing can opt in.
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "EngineConfig")]
#[derive(Clone, Default)]
pub struct EngineConfig {
    inner: RsEngineConfig,
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (
        model,
        served_model_name = None,
        context_length = None,
        kv_cache_block_size = None,
        total_kv_blocks = None,
        max_num_seqs = None,
        max_num_batched_tokens = None,
        data_parallel_size = None,
        data_parallel_start_rank = None,
        bootstrap_host = None,
        bootstrap_port = None,
        runtime_data = None,
        capabilities = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: String,
        served_model_name: Option<String>,
        context_length: Option<u32>,
        kv_cache_block_size: Option<u32>,
        total_kv_blocks: Option<u64>,
        max_num_seqs: Option<u64>,
        max_num_batched_tokens: Option<u64>,
        data_parallel_size: Option<u32>,
        data_parallel_start_rank: Option<u32>,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
        runtime_data: Option<&Bound<'_, PyDict>>,
        capabilities: Option<Vec<Capability>>,
    ) -> PyResult<Self> {
        let runtime_data = runtime_data
            .map(|dict| depythonize::<HashMap<String, serde_json::Value>>(dict))
            .transpose()
            .map_err(to_pyerr)?
            .unwrap_or_default();

        Ok(Self {
            inner: RsEngineConfig {
                model,
                served_model_name,
                context_length,
                kv_cache_block_size,
                total_kv_blocks,
                max_num_seqs,
                max_num_batched_tokens,
                data_parallel_size,
                data_parallel_start_rank,
                bootstrap_host,
                bootstrap_port,
                runtime_data,
                capabilities: capabilities
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            },
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn served_model_name(&self) -> Option<&str> {
        self.inner.served_model_name.as_deref()
    }
    #[getter]
    fn context_length(&self) -> Option<u32> {
        self.inner.context_length
    }
    #[getter]
    fn kv_cache_block_size(&self) -> Option<u32> {
        self.inner.kv_cache_block_size
    }
    #[getter]
    fn total_kv_blocks(&self) -> Option<u64> {
        self.inner.total_kv_blocks
    }
    #[getter]
    fn max_num_seqs(&self) -> Option<u64> {
        self.inner.max_num_seqs
    }
    #[getter]
    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.inner.max_num_batched_tokens
    }
    #[getter]
    fn data_parallel_size(&self) -> Option<u32> {
        self.inner.data_parallel_size
    }
    #[getter]
    fn data_parallel_start_rank(&self) -> Option<u32> {
        self.inner.data_parallel_start_rank
    }
    #[getter]
    fn bootstrap_host(&self) -> Option<&str> {
        self.inner.bootstrap_host.as_deref()
    }
    #[getter]
    fn bootstrap_port(&self) -> Option<u16> {
        self.inner.bootstrap_port
    }
    #[getter]
    fn runtime_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.inner.runtime_data)
            .map(|value| value.unbind())
            .map_err(to_pyerr)
    }
    #[getter]
    fn capabilities(&self) -> Vec<Capability> {
        // Sorted by integer value for stable test assertions.
        let mut v: Vec<Capability> = self
            .inner
            .capabilities
            .iter()
            .copied()
            .map(Capability::from)
            .collect();
        v.sort_unstable_by_key(|c| *c as u8);
        v
    }
}

// ---------------------------------------------------------------------------
// RuntimeConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "RuntimeConfig")]
#[derive(Clone, Default)]
pub struct RuntimeConfig {
    inner: RsRuntimeConfig,
}

#[pymethods]
impl RuntimeConfig {
    #[new]
    #[pyo3(signature = (discovery_backend = None, request_plane = None, event_plane = None))]
    fn new(
        discovery_backend: Option<String>,
        request_plane: Option<String>,
        event_plane: Option<String>,
    ) -> Self {
        Self {
            inner: RsRuntimeConfig {
                discovery_backend,
                request_plane,
                event_plane,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerConfig
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "WorkerConfig")]
#[derive(Clone)]
pub struct WorkerConfig {
    inner: RsWorkerConfig,
}

#[pymethods]
impl WorkerConfig {
    #[new]
    #[pyo3(signature = (
        namespace,
        component = "backend".to_string(),
        endpoint = "generate".to_string(),
        model_name = String::new(),
        served_model_name = None,
        model_input = ModelInput::Tokens,
        endpoint_types = "chat,completions".to_string(),
        custom_jinja_template = None,
        tool_call_parser = None,
        reasoning_parser = None,
        exclude_tools_when_tool_choice_none = true,
        enable_local_indexer = true,
        enable_kv_routing = true,
        metrics_labels = Vec::new(),
        runtime = None,
        disaggregation_mode = DisaggregationMode::Aggregated,
        health_check_payload = None,
        structural_tag_mode = "off".to_string(),
        structural_tag_scope = "auto".to_string(),
        structural_tag_schema = "auto".to_string(),
        unsupported_field_policy = UnsupportedFieldPolicy::Warn,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        namespace: String,
        component: String,
        endpoint: String,
        model_name: String,
        served_model_name: Option<String>,
        model_input: ModelInput,
        endpoint_types: String,
        custom_jinja_template: Option<String>,
        tool_call_parser: Option<String>,
        reasoning_parser: Option<String>,
        exclude_tools_when_tool_choice_none: bool,
        enable_local_indexer: bool,
        enable_kv_routing: bool,
        metrics_labels: Vec<(String, String)>,
        runtime: Option<RuntimeConfig>,
        disaggregation_mode: DisaggregationMode,
        health_check_payload: Option<PyObject>,
        structural_tag_mode: String,
        structural_tag_scope: String,
        structural_tag_schema: String,
        unsupported_field_policy: UnsupportedFieldPolicy,
    ) -> PyResult<Self> {
        // Delegating to the same conversion used by `register_model`.
        let model_input_rs = match model_input {
            ModelInput::Text => RsModelInput::Text,
            ModelInput::Tokens => RsModelInput::Tokens,
            ModelInput::Tensor => RsModelInput::Tensor,
        };
        // Accept a Python dict or None; depythonize to serde_json::Value
        // and require an object — engines branch on a dict marker, and the
        // runtime canary registers a dict-shaped payload.
        let health_check_payload = match health_check_payload {
            Some(obj) if !obj.is_none(py) => {
                let bound = obj.bind(py);
                let value: serde_json::Value = depythonize(bound).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "health_check_payload must be a JSON-serializable dict: {e}"
                    ))
                })?;
                if !value.is_object() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "health_check_payload must be a JSON object (dict)",
                    ));
                }
                Some(value)
            }
            _ => None,
        };
        let st_mode = match structural_tag_mode.as_str() {
            "off" => RsStructuralTagMode::Off,
            "on" => RsStructuralTagMode::On,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid structural_tag_mode: {other}. Expected 'off' or 'on'."
                )));
            }
        };
        let st_scope = match structural_tag_scope.as_str() {
            "auto" => RsStructuralTagScope::Auto,
            "always" => RsStructuralTagScope::Always,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid structural_tag_scope: {other}. Expected 'auto' or 'always'."
                )));
            }
        };
        let st_schema = match structural_tag_schema.as_str() {
            "auto" => RsStructuralTagSchemaMode::Auto,
            "strict" => RsStructuralTagSchemaMode::Strict,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid structural_tag_schema: {other}. Expected 'auto' or 'strict'."
                )));
            }
        };
        Ok(Self {
            inner: RsWorkerConfig {
                namespace,
                component,
                endpoint,
                model_name,
                served_model_name,
                model_input: model_input_rs,
                endpoint_types,
                custom_jinja_template: custom_jinja_template.map(PathBuf::from),
                tool_call_parser,
                reasoning_parser,
                exclude_tools_when_tool_choice_none,
                enable_local_indexer,
                enable_kv_routing,
                metrics_labels,
                disaggregation_mode: disaggregation_mode.into(),
                health_check_payload,
                structural_tag_mode: st_mode,
                structural_tag_scope: st_scope,
                structural_tag_schema: st_schema,
                unsupported_field_policy: unsupported_field_policy.into(),
                runtime: runtime.map(|r| r.inner).unwrap_or_default(),
            },
        })
    }

    #[getter]
    fn unsupported_field_policy(&self) -> UnsupportedFieldPolicy {
        self.inner.unsupported_field_policy.into()
    }
}

// ---------------------------------------------------------------------------
// Worker — the entry point Python users `await`.
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "Worker")]
pub struct Worker {
    engine: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    config: RsWorkerConfig,
    /// `true` if this `Worker` instance constructed the dynamo runtime
    /// itself (no `DistributedRuntime` already existed in this process).
    /// Determines whether `run()` should call `runtime.shutdown()` at the
    /// end — we only want to tear down a runtime we own.
    owns_runtime: bool,
    /// Single-shot guard — flipped to `true` on the first `run()` call.
    /// The Rust `Worker` underneath consumes `self`; calling `run()`
    /// twice from Python would build a second `RsWorker` and call
    /// `engine.start()` again, which most engines (vLLM, sglang, trtllm)
    /// don't tolerate. We surface a clear `RuntimeError` instead.
    consumed: AtomicBool,
}

#[pymethods]
impl Worker {
    #[new]
    fn new(engine: PyObject, config: WorkerConfig, event_loop: PyObject) -> PyResult<Self> {
        // True existing-only check — `runtime_from_existing()` would
        // synthesize a fresh runtime here and falsely mark us as shared.
        let owns_runtime = !rs::Worker::has_existing_runtime();

        if owns_runtime {
            // Apply RuntimeConfig env overrides synchronously, on the
            // calling thread, before any tokio worker threads spawn.
            // Setting env vars from inside the future-into-py block would
            // race with concurrent env reads in already-running tokio
            // tasks (NATS / etcd setup).
            config.inner.runtime.apply_to_env();

            let worker = rs::Worker::from_settings().map_err(to_pyerr)?;
            let primary = worker.tokio_runtime().map_err(to_pyerr)?;
            // `init_with_runtime` errors if already initialized; that case
            // means someone called us in a process where the OnceCell was
            // populated between our check and now. Idempotent — ignore.
            let _ = pyo3_async_runtimes::tokio::init_with_runtime(primary);
        } else if config.inner.runtime.has_overrides() {
            // The shared runtime was constructed before our caller, so its
            // env-driven config (`DYN_DISCOVERY_BACKEND` etc.) is already
            // baked in. Setting env vars now wouldn't change the runtime
            // — surface the silent-drop loudly so operators don't assume
            // their override took effect.
            tracing::warn!(
                "Worker received RuntimeConfig overrides but the dynamo \
                 runtime was already constructed elsewhere; overrides ignored. \
                 Set DYN_DISCOVERY_BACKEND / DYN_REQUEST_PLANE / DYN_EVENT_PLANE \
                 in the environment instead."
            );
        }

        Ok(Self {
            engine: Arc::new(engine),
            event_loop: Arc::new(event_loop),
            config: config.inner,
            owns_runtime,
            consumed: AtomicBool::new(false),
        })
    }

    /// Drive the full lifecycle: start engine → register model → serve →
    /// (on signal) orchestrate graceful shutdown → cleanup → return.
    fn run<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        // Worker is single-shot — flip the consumed flag atomically so
        // a second `await worker.run()` raises clearly instead of
        // re-initializing the engine.
        if self
            .consumed
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Worker.run() can only be called once per Worker instance; \
                 construct a fresh engine + Worker to run again (most LLM \
                 engines do not tolerate re-initialization)",
            ));
        }

        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let config = self.config.clone();
        let owns_runtime = self.owns_runtime;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let runtime = rs::Worker::runtime_from_existing()
                .or_else(|_| {
                    let worker = rs::Worker::from_settings()?;
                    Ok::<_, anyhow::Error>(worker.runtime().clone())
                })
                .map_err(to_pyerr)?;

            // Initialize logging now that tokio context is available. Mirrors
            // the DistributedRuntime init path — required so workers using
            // `dynamo.common.backend.Worker` directly (without constructing a
            // DistributedRuntime first) install the tracing + OTLP exporter
            // layers. Without this, OTEL_EXPORT_ENABLED workers emit no
            // logs and no spans.
            if dynamo_runtime::config::env_is_truthy(
                dynamo_runtime::config::environment_names::logging::otlp::OTEL_EXPORT_ENABLED,
            ) {
                rs::logging::init();
                // Runtime canary: if a future refactor drops or breaks the
                // init above, this loud stderr message fires once per worker
                // startup so operators discover the regression without
                // needing to chase silent missing-spans symptoms. We use
                // `eprintln!` (not tracing::warn!) deliberately — a missing
                // subscriber means tracing calls are silent no-ops, so the
                // signal MUST bypass tracing.
                if !tracing::dispatcher::has_been_set() {
                    eprintln!(
                        "ERROR: OTEL_EXPORT_ENABLED=1 but no tracing subscriber \
                         installed after `Worker::run` init. Worker telemetry \
                         (spans, logs) will be SILENT — check the conditional \
                         `rs::logging::init()` call in `Worker::run`."
                    );
                }
            }

            let py_engine = PyLLMEngine::new(engine, event_loop);
            let worker = RsWorker::new(Arc::new(py_engine), config);

            let result = worker.run(runtime.clone()).await.map_err(to_pyerr);

            // Only tear the runtime down if we constructed it. When a
            // `DistributedRuntime` was already in scope (HTTP frontend,
            // tests, etc.) it owns the shutdown lifecycle and we'd be
            // pulling the rug out from other tasks if we called shutdown.
            if owns_runtime {
                runtime.shutdown();
            } else {
                tracing::debug!(
                    "Worker.run skipping runtime.shutdown(); runtime is \
                     shared with another caller"
                );
            }

            result
        })
    }
}

// ---------------------------------------------------------------------------
// PyLLMEngine — the actual bridge. Not a `#[pyclass]`; lives only in Rust.
// ---------------------------------------------------------------------------

struct PyLLMEngine {
    // Wrapped in `Arc` so we can clone refcount-style without acquiring
    // the GIL — `PyObject::clone` would otherwise need to bump Python's
    // own refcount, which requires the GIL. Same pattern as
    // `PythonAsyncEngine` in `engine.rs`.
    engine: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    trace_contexts: Arc<StdMutex<HashMap<String, DistributedTraceContext>>>,
    request_metadata: Arc<StdMutex<HashMap<String, BTreeMap<String, String>>>>,
}

impl PyLLMEngine {
    fn new(engine: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        Self {
            engine,
            event_loop,
            trace_contexts: Arc::new(StdMutex::new(HashMap::new())),
            request_metadata: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    /// Call a no-arg async method on `self.engine` and await it on
    /// `self.event_loop`. Used for `start`, `drain`, `cleanup`.
    async fn call_method0_async(&self, method: &'static str) -> Result<PyObject, PyErr> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();

        // Acquiring the GIL inside an async task can stall the tokio
        // worker; spawn_blocking matches the existing `PythonAsyncEngine`
        // pattern in `engine.rs`.
        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<_> {
                let bound = engine.bind(py);
                let coroutine = bound.call_method0(method)?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("offload error: {e}"))
        })??;

        py_future.await
    }
}

struct RequestStateGuard {
    request_id: String,
    trace_contexts: Arc<StdMutex<HashMap<String, DistributedTraceContext>>>,
    request_metadata: Arc<StdMutex<HashMap<String, BTreeMap<String, String>>>>,
}

impl Drop for RequestStateGuard {
    fn drop(&mut self) {
        self.trace_contexts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.request_id);
        self.request_metadata
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.request_id);
    }
}

#[async_trait]
impl LLMEngine for PyLLMEngine {
    async fn start(&self, worker_id: u64) -> Result<RsEngineConfig, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();

        // Forward worker_id to Python `start(worker_id)`. spawn_blocking
        // around the GIL section matches `call_method0_async`.
        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<_> {
                let bound = engine.bind(py);
                let coroutine = bound.call_method1("start", (worker_id,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("start offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        let result = py_future.await.map_err(py_err_to_dynamo)?;

        Python::with_gil(|py| -> PyResult<RsEngineConfig> {
            let bound = result.bind(py);
            // Accept either the Rust EngineConfig pyclass or any Python
            // object exposing the canonical attribute names (e.g. the
            // `dynamo.common.backend.EngineConfig` dataclass).
            if let Ok(cfg) = bound.extract::<EngineConfig>() {
                return Ok(cfg.inner);
            }
            // capabilities is optional on the duck-typed object. Missing
            // attribute or `None` → empty. Items extract as the typed
            // `Capability` PyO3 enum; anything else raises TypeError.
            let capabilities: HashSet<RsCapability> = match bound.getattr("capabilities") {
                Ok(value) if !value.is_none() => value
                    .extract::<Vec<Capability>>()?
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                _ => HashSet::new(),
            };
            Ok(RsEngineConfig {
                model: bound.getattr("model")?.extract()?,
                served_model_name: opt_attr::<String>(bound, "served_model_name")?,
                context_length: opt_attr::<u32>(bound, "context_length")?,
                kv_cache_block_size: opt_attr::<u32>(bound, "kv_cache_block_size")?,
                total_kv_blocks: opt_attr::<u64>(bound, "total_kv_blocks")?,
                max_num_seqs: opt_attr::<u64>(bound, "max_num_seqs")?,
                max_num_batched_tokens: opt_attr::<u64>(bound, "max_num_batched_tokens")?,
                data_parallel_size: opt_attr::<u32>(bound, "data_parallel_size")?,
                data_parallel_start_rank: opt_attr::<u32>(bound, "data_parallel_start_rank")?,
                bootstrap_host: opt_attr::<String>(bound, "bootstrap_host")?,
                bootstrap_port: opt_attr::<u16>(bound, "bootstrap_port")?,
                runtime_data: match bound.getattr("runtime_data") {
                    Ok(value) if !value.is_none() => depythonize(&value).map_err(to_pyerr)?,
                    _ => HashMap::new(),
                },
                capabilities,
            })
        })
        .map_err(py_err_to_dynamo)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: dynamo_backend_common::GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let trace_context = get_distributed_tracing_context();
        let request_id = ctx.id().to_string();
        self.request_metadata
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id.clone(), ctx.metadata().clone());
        if let Some(trace_context) = trace_context.as_ref() {
            self.trace_contexts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(request_id.clone(), trace_context.clone());
        }
        let request_state_guard = RequestStateGuard {
            request_id,
            trace_contexts: self.trace_contexts.clone(),
            request_metadata: self.request_metadata.clone(),
        };

        let first_token = ctx.first_token_sender().cloned();
        let inner_ctx = ctx.inner_arc();
        // **Invariant**: `tracing::Span::current()` here MUST be the
        // `engine.generate` span opened by `EngineAdapter::generate`. The
        // capture must happen BEFORE `spawn_blocking` because inside the
        // blocking closure, `Span::current()` is the worker-thread root,
        // not the auto-span.
        //
        // If anyone refactors this dispatch (extra task hop, different
        // scheduler), they must re-verify the captured span. `Context`
        // stores this span and routes engine telemetry calls to it via
        // `current_span` / `start_span` — wrong span = wrong attributes
        // silently. See the `auto_span_records_*` tests in
        // `lib/backend-common/src/adapter.rs` for the assertions that depend
        // on this invariant.
        debug_assert_eq!(
            tracing::Span::current().metadata().map(|m| m.name()),
            Some("engine.generate"),
            "Span::current() must be engine.generate at PyLLMEngine boundary; \
             a dispatch refactor likely broke the capture point"
        );
        let engine_span = tracing::Span::current();

        // Pythonize the request, call generate(request, context=ctx), and
        // turn the resulting Python async generator into a Rust stream.
        let stream = tokio::task::spawn_blocking(move || -> PyResult<_> {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let py_ctx = Py::new(
                    py,
                    PyContext::new(
                        inner_ctx,
                        trace_context,
                        first_token,
                        ctx.metadata().clone(),
                    )
                    .with_span(engine_span),
                )?;

                let kwargs = PyDict::new(py);
                kwargs.set_item("context", &py_ctx)?;

                let bound = engine.bind(py);
                let gen_obj = bound.call_method("generate", (py_request,), Some(&kwargs))?;

                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(locals, gen_obj)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("generate offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        let mapped = async_stream::stream! {
            let _request_state_guard = request_state_guard;
            let mut inner = std::pin::pin!(stream);
            while let Some(item) = inner.next().await {
                let py_obj = match item {
                    Ok(obj) => obj,
                    Err(e) => {
                        yield Err(py_err_to_dynamo(e));
                        return;
                    }
                };

                // Depythonize the chunk dict on a blocking thread — same
                // GIL-contention rationale as the request side.
                let parsed = tokio::task::spawn_blocking(move || {
                    Python::with_gil(|py| -> PyResult<LLMEngineOutput> {
                        let bound = py_obj.into_bound(py);
                        let mut out: LLMEngineOutput = depythonize(&bound).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "invalid chunk shape: {e}"
                            ))
                        })?;
                        // Match the Python `Worker.generate` default of
                        // `index = 0` for single-choice streams so the
                        // OpenAI frontend keeps choices stable.
                        if out.index.is_none() {
                            out.index = Some(0);
                        }
                        Ok(out)
                    })
                })
                .await;

                match parsed {
                    Ok(Ok(chunk)) => yield Ok(chunk),
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "failed to parse chunk from python engine");
                        yield Err(py_err_to_dynamo(e));
                        return;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "chunk parse offload error");
                        yield Err(DynamoError::builder()
                            .error_type(ErrorType::Backend(BackendError::Unknown))
                            .message(format!("chunk parse offload error: {e}"))
                            .build());
                        return;
                    }
                }
            }
        };

        Ok(Box::pin(mapped))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let trace_context = self
            .trace_contexts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(ctx.id())
            .cloned()
            .or_else(get_distributed_tracing_context);
        let metadata = self
            .request_metadata
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(ctx.id())
            .unwrap_or_default();

        let res: Result<(), PyErr> = async move {
            let py_future = tokio::task::spawn_blocking(move || {
                Python::with_gil(|py| -> PyResult<_> {
                    let bound = engine.bind(py);
                    let py_ctx = Py::new(py, PyContext::new(ctx, trace_context, None, metadata))?;
                    let coroutine = bound.call_method1("abort", (py_ctx,))?;
                    let locals = TaskLocals::new(event_loop.bind(py).clone());
                    pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
                })
            })
            .await
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("offload error: {e}"))
            })??;
            py_future.await?;
            Ok(())
        }
        .await;

        if let Err(e) = res {
            // Aborts are best-effort — log and swallow so cancellation
            // bookkeeping isn't blocked by a misbehaving engine.
            tracing::debug!(error = %e, "engine.abort raised; ignoring");
        }
    }

    async fn drain(&self) -> Result<(), DynamoError> {
        self.call_method0_async("drain")
            .await
            .map_err(py_err_to_dynamo)?;
        Ok(())
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.call_method0_async("cleanup")
            .await
            .map_err(py_err_to_dynamo)?;
        Ok(())
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        let py_obj = self
            .call_method0_async("health_check_payload")
            .await
            .map_err(py_err_to_dynamo)?;
        Python::with_gil(|py| -> PyResult<Option<serde_json::Value>> {
            let bound = py_obj.bind(py);
            if bound.is_none() {
                return Ok(None);
            }
            let value: serde_json::Value = depythonize(bound).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "health_check_payload must return a JSON-serializable dict or None: {e}"
                ))
            })?;
            if !value.is_object() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "health_check_payload must return a JSON object (dict) or None",
                ));
            }
            Ok(Some(value))
        })
        .map_err(py_err_to_dynamo)
    }

    async fn kv_event_sources(&self) -> Result<Vec<RsKvEventSource>, DynamoError> {
        let py_list = self
            .call_method0_async("kv_event_sources")
            .await
            .map_err(py_err_to_dynamo)?;
        Python::with_gil(|py| -> PyResult<Vec<RsKvEventSource>> {
            let bound = py_list.bind(py);
            let list = bound.downcast::<pyo3::types::PyList>()?;
            let mut sources = Vec::with_capacity(list.len());
            for item in list.iter() {
                sources.push(depythonize_kv_source(&item)?);
            }
            Ok(sources)
        })
        .map_err(py_err_to_dynamo)
    }

    async fn setup_metrics(&self, _ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        // Step 1: call Python's `register_prometheus` for vendor-registry
        // bridging (side effect on ctx.metrics; nothing flows back). Engines
        // that don't override the ABC default no-op return immediately.
        self.call_python_register_prometheus(_ctx.metrics).await?;

        // Step 2: read the engine's declared dp_ranks. Empty = opt out.
        let dp_ranks = self.read_dp_ranks().await?;
        if dp_ranks.is_empty() {
            return Ok(MetricsBindings {
                dp_ranks,
                on_publisher_ready: None,
            });
        }

        // Step 3: build the on_publisher_ready closure. Framework calls
        // it with the constructed `Arc<SnapshotPublisher>`; we hand it
        // back to Python via `attach_snapshot_publisher`. Engine stashes
        // and calls `publisher.publish(rank, snap)` from its stat-logger
        // thereafter.
        let engine = self.engine.clone();
        let on_publisher_ready: dynamo_backend_common::OnSnapshotPublisherReady =
            Box::new(move |publisher: Arc<RsSnapshotPublisher>| {
                Python::with_gil(|py| -> PyResult<()> {
                    let py_pub = Py::new(py, PySnapshotPublisher { inner: publisher })?;
                    engine
                        .bind(py)
                        .call_method1("attach_snapshot_publisher", (py_pub,))?;
                    Ok(())
                })
                .map_err(py_err_to_dynamo)
            });

        Ok(MetricsBindings {
            dp_ranks,
            on_publisher_ready: Some(on_publisher_ready),
        })
    }
}

impl PyLLMEngine {
    async fn call_python_register_prometheus(
        &self,
        metrics: &dynamo_backend_common::EngineMetrics,
    ) -> Result<(), DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let py_metrics_state = crate::prometheus_metrics::EngineMetrics::from_rust(metrics);

        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<_> {
                let bound = engine.bind(py);
                let py_metrics = Py::new(py, py_metrics_state)?;
                let coroutine = bound.call_method1("register_prometheus", (py_metrics,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("register_prometheus offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        py_future.await.map_err(py_err_to_dynamo)?;
        Ok(())
    }

    /// Call Python `component_metrics_dp_ranks()` → `Vec<u32>`. Empty list
    /// is the default (engine opts out of per-rank metrics).
    async fn read_dp_ranks(&self) -> Result<Vec<u32>, DynamoError> {
        let engine = self.engine.clone();
        let join = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<u32>> {
                let result = engine.bind(py).call_method0("component_metrics_dp_ranks")?;
                let list = result.downcast::<pyo3::types::PyList>()?;
                list.iter().map(|item| item.extract::<u32>()).collect()
            })
        })
        .await;
        match join {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(py_err_to_dynamo(e)),
            Err(join_err) => Err(DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!(
                    "component_metrics_dp_ranks spawn_blocking join failed: {join_err}"
                ))
                .build()),
        }
    }
}

/// PyO3 binding for [`SnapshotPublisher`]. Engines call `publish(rank, snap)`
/// from their stat-logger thread; the call acquires no extra GIL (it's
/// invoked from Python which already holds it), copies the snapshot
/// fields out via attribute access, then releases the GIL while writing
/// the Rust gauges + NATS signal.
#[pyclass(module = "dynamo._core.backend", name = "SnapshotPublisher", frozen)]
pub struct PySnapshotPublisher {
    inner: Arc<RsSnapshotPublisher>,
}

#[pymethods]
impl PySnapshotPublisher {
    /// Push a snapshot for `dp_rank`. Fields are read from the
    /// `ComponentSnapshot` Python dataclass via attribute access; the
    /// Rust-side write is performed without holding the GIL.
    fn publish(&self, py: Python<'_>, dp_rank: u32, snapshot: &Bound<'_, PyAny>) -> PyResult<()> {
        let kv_used_blocks: u64 = snapshot.getattr("kv_used_blocks")?.extract()?;
        let kv_total_blocks: u64 = snapshot.getattr("kv_total_blocks")?.extract()?;
        let gpu_cache_usage: f32 = snapshot.getattr("gpu_cache_usage")?.extract()?;
        let kv_cache_hit_rate: Option<f32> = match snapshot.getattr("kv_cache_hit_rate") {
            Ok(v) if v.is_none() => None,
            Ok(v) => v.extract().ok(),
            Err(_) => None,
        };
        let snap = ComponentSnapshot {
            kv_used_blocks,
            kv_total_blocks,
            gpu_cache_usage,
            kv_cache_hit_rate,
            dp_rank,
        };
        let inner = self.inner.clone();
        py.allow_threads(move || inner.publish(dp_rank, snap));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Source-descriptor depythonization
//
// Each Python descriptor maps to one arm of the Rust source enum. Identify
// by class name (`type(x).__name__`) — the Python module exports each as a
// distinct dataclass, so the name is the unambiguous discriminator. Falling
// back to attribute-presence checks would silently accept unrelated objects.
// ---------------------------------------------------------------------------

fn class_name(item: &Bound<'_, PyAny>) -> PyResult<String> {
    item.get_type().getattr("__name__")?.extract::<String>()
}

fn depythonize_kv_source(item: &Bound<'_, PyAny>) -> PyResult<RsKvEventSource> {
    let cls = class_name(item)?;
    let dp_rank: u32 = item.getattr("dp_rank")?.extract()?;
    match cls.as_str() {
        "ZmqSource" => Ok(RsKvEventSource::Zmq {
            endpoint: item.getattr("endpoint")?.extract()?,
            topic: item.getattr("topic")?.extract()?,
            dp_rank,
        }),
        "PushSource" => {
            // Capture the Python callable as a `PyObject` and wrap in a
            // Rust closure. The closure runs once when Worker has the
            // publisher ready: it acquires the GIL, wraps the Rust
            // `Arc<KvEventPublisher>` as the existing Python pyclass, and
            // invokes the engine-supplied callback. The callback is
            // declared sync on the Python side (see `PushSource.on_ready`
            // in `dynamo.common.backend.publisher`), so no asyncio
            // round-trip is needed here.
            let on_ready_obj: PyObject = item.getattr("on_ready")?.into();
            let on_ready: OnPublisherReady = Box::new(move |publisher| {
                Python::with_gil(|py| -> PyResult<()> {
                    let py_pub = Py::new(py, PyKvEventPublisher::from_arc(publisher, dp_rank))?;
                    on_ready_obj.call1(py, (py_pub,))?;
                    Ok(())
                })
                .map_err(py_err_to_dynamo)
            });
            Ok(RsKvEventSource::Push { on_ready, dp_rank })
        }
        other => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "kv_event_sources() returned unknown descriptor type {other:?}; \
             expected ZmqSource or PushSource"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract an optional attribute from a Python object.
///
/// Returns:
///   * `Ok(None)` when the attribute is missing or set to `None`.
///   * `Ok(Some(v))` when present and convertible.
///   * `Err(PyErr)` when present and non-`None` but the conversion fails
///     — surfaces engine-author bugs (e.g. `context_length="not-a-int"`)
///     rather than silently dropping them.
fn opt_attr<T>(bound: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<T>>
where
    T: for<'py> FromPyObject<'py>,
{
    let attr = match bound.getattr(name) {
        Ok(v) => v,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(bound.py()) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    if attr.is_none() {
        return Ok(None);
    }
    Ok(Some(attr.extract()?))
}

/// Map a Python exception to a `BackendError` variant. `DynamoException`
/// subclasses go through the shared mapping table; built-in Python
/// exceptions fall back to the closest category.
fn py_err_to_dynamo(err: PyErr) -> DynamoError {
    let (backend, message) = Python::with_gil(|py| {
        if let Some(mapped) = py_exception_to_backend_error(py, &err) {
            return mapped;
        }
        let backend = if err.is_instance_of::<pyo3::exceptions::PyValueError>(py)
            || err.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
        {
            BackendError::InvalidArgument
        } else if err.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
            BackendError::ConnectionTimeout
        } else if err.is_instance_of::<pyo3::exceptions::PyConnectionRefusedError>(py) {
            BackendError::CannotConnect
        } else if err.is_instance_of::<pyo3::exceptions::PyConnectionResetError>(py)
            || err.is_instance_of::<pyo3::exceptions::PyBrokenPipeError>(py)
            || err.is_instance_of::<pyo3::exceptions::PyConnectionError>(py)
        {
            BackendError::Disconnected
        } else if err.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py) {
            BackendError::Cancelled
        } else if err.is_instance_of::<pyo3::exceptions::PyGeneratorExit>(py) {
            BackendError::EngineShutdown
        } else {
            BackendError::Unknown
        };
        (backend, err.to_string())
    });
    DynamoError::builder()
        .error_type(ErrorType::Backend(backend))
        .message(message)
        .build()
}
