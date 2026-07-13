// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bridge for `dynamo_backend_common::Worker`.
//!
//! Lets a Python `BaseEngine` ABC subclass plug into the Rust `Worker`
//! through a thin adapter: `PyLLMEngine` for token engines, `PyRawEngine`
//! for raw-media engines, both composing the shared `PyEngineCore`. All
//! lifecycle work — signal handling, discovery unregister, grace period,
//! drain, cleanup, and 3-phase runtime shutdown — lives in Rust; Python
//! only owns engine semantics.
//!
//! Exposed under `dynamo._core.backend` as `Worker`, `WorkerConfig`,
//! `EngineConfig`, `RuntimeConfig`, and `PreRuntimeContext`.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, ComponentSnapshot,
    DisaggregationMode as RsDisaggregationMode, DynamoError, EngineConfig as RsEngineConfig,
    ErrorType, KvEventSource as RsKvEventSource, LLMEngine, LLMEngineOutput,
    LlmRegistration as RsLlmRegistration, MetricsBindings, MetricsCtx, OnPublisherReady,
    PreRuntimeContext as RsPreRuntimeContext, PreRuntimeOutcome as RsPreRuntimeOutcome,
    PreprocessedRequest, RawEngine, RestoredRuntimeConfig as RsRestoredRuntimeConfig,
    RuntimeConfig as RsRuntimeConfig, SnapshotPublisher as RsSnapshotPublisher, Worker as RsWorker,
    WorkerConfig as RsWorkerConfig, WorkerExit as RsWorkerExit,
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
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};

use crate::ModelInput;
use crate::context::Context as PyContext;
use crate::errors::{extract_http_like_error, py_exception_to_backend_error};
use crate::llm::kv::KvEventPublisher as PyKvEventPublisher;
use crate::llm::preprocessor::{MediaDecoder, MediaFetcher};
use crate::to_pyerr;

/// Register `dynamo._core.backend` and its classes on the parent `_core` module.
pub fn add_to_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "backend")?;
    m.add_class::<DisaggregationMode>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<LlmRegistration>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<PreRuntimeContext>()?;
    m.add_class::<WorkerConfig>()?;
    m.add_class::<Worker>()?;
    m.add_class::<PySnapshotPublisher>()?;
    m.add_class::<crate::prometheus_metrics::EngineMetrics>()?;
    m.add("HEALTH_CHECK_KEY", dynamo_backend_common::HEALTH_CHECK_KEY)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("dynamo._core.backend", &m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PreRuntimeContext — cancellation view handed to Python BaseEngine.
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "PreRuntimeContext", frozen)]
#[derive(Clone)]
pub struct PreRuntimeContext {
    inner: RsPreRuntimeContext,
}

#[pymethods]
impl PreRuntimeContext {
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    fn cancelled<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let context = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            context.cancelled().await;
            Ok(())
        })
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
    Encode = 4,
}

impl From<DisaggregationMode> for RsDisaggregationMode {
    fn from(value: DisaggregationMode) -> Self {
        match value {
            DisaggregationMode::Aggregated => RsDisaggregationMode::Aggregated,
            DisaggregationMode::Prefill => RsDisaggregationMode::Prefill,
            DisaggregationMode::Decode => RsDisaggregationMode::Decode,
            DisaggregationMode::Encode => RsDisaggregationMode::Encode,
        }
    }
}

// ---------------------------------------------------------------------------
// EngineConfig — mirror of `dynamo_backend_common::EngineConfig`.
//
// Engines may return either this pyclass or any object with the canonical
// attributes `model` / `served_model_name` / `runtime_data` / `llm`; the
// bridge's `start()` extraction accepts both. Note `llm` is a nested record
// (LlmRegistration), NOT flat fields — an object exposing flat `context_length`
// etc. (the pre-split shape) registers with `llm=None`, i.e. no KV/DP/bootstrap
// hints. We expose this pyclass mainly so engines that want strong typing can
// opt in.
// ---------------------------------------------------------------------------

#[pyclass(module = "dynamo._core.backend", name = "LlmRegistration")]
#[derive(Clone, Default)]
pub struct LlmRegistration {
    inner: RsLlmRegistration,
}

#[pymethods]
impl LlmRegistration {
    #[new]
    #[pyo3(signature = (
        context_length = None,
        kv_cache_block_size = None,
        total_kv_blocks = None,
        max_num_seqs = None,
        max_num_batched_tokens = None,
        data_parallel_size = None,
        data_parallel_start_rank = None,
        bootstrap_host = None,
        bootstrap_port = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        context_length: Option<u32>,
        kv_cache_block_size: Option<u32>,
        total_kv_blocks: Option<u64>,
        max_num_seqs: Option<u64>,
        max_num_batched_tokens: Option<u64>,
        data_parallel_size: Option<u32>,
        data_parallel_start_rank: Option<u32>,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) -> Self {
        Self {
            inner: RsLlmRegistration {
                context_length,
                kv_cache_block_size,
                total_kv_blocks,
                max_num_seqs,
                max_num_batched_tokens,
                data_parallel_size,
                data_parallel_start_rank,
                bootstrap_host,
                bootstrap_port,
            },
        }
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
}

#[pyclass(module = "dynamo._core.backend", name = "EngineConfig")]
#[derive(Clone, Default)]
pub struct EngineConfig {
    inner: RsEngineConfig,
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (model, served_model_name = None, runtime_data = None, llm = None))]
    fn new(
        model: String,
        served_model_name: Option<String>,
        runtime_data: Option<&Bound<'_, PyDict>>,
        llm: Option<LlmRegistration>,
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
                runtime_data,
                llm: llm.map(|l| l.inner),
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
    fn llm(&self) -> Option<LlmRegistration> {
        self.inner
            .llm
            .clone()
            .map(|inner| LlmRegistration { inner })
    }
    #[getter]
    fn runtime_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize(py, &self.inner.runtime_data)
            .map(|value| value.unbind())
            .map_err(to_pyerr)
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
        route_to_encoder = false,
        media_decoder = None,
        media_fetcher = None,
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
        route_to_encoder: bool,
        media_decoder: Option<MediaDecoder>,
        media_fetcher: Option<MediaFetcher>,
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
                runtime: runtime.map(|r| r.inner).unwrap_or_default(),
                route_to_encoder,
                media_decoder: media_decoder.map(|decoder| decoder.inner),
                media_fetcher: media_fetcher.map(|fetcher| fetcher.inner),
            },
        })
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
    /// `true` when `engine` is a `DiffusionEngine` (raw media pipeline).
    /// Set by the Python `Worker` shim via `isinstance`. Selects the raw
    /// JSON request adapter (`RsWorker::new_raw`) instead of the token
    /// adapter (`RsWorker::new`).
    raw: bool,
}

#[pymethods]
impl Worker {
    #[new]
    #[pyo3(signature = (engine, config, event_loop, raw = false))]
    fn new(
        engine: PyObject,
        config: WorkerConfig,
        event_loop: PyObject,
        raw: bool,
    ) -> PyResult<Self> {
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
            raw,
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
        let raw = self.raw;

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

            // Select the engine modality. A `DiffusionEngine` (raw media
            // pipeline) routes through `PyRawEngine` + `RsWorker::new_raw`
            // (JSON adapter); everything else is a token-pipeline
            // `LLMEngine`.
            let worker = if raw {
                let py_engine = PyRawEngine::new(engine, event_loop);
                RsWorker::new_raw(Arc::new(py_engine), config)
            } else {
                let py_engine = PyLLMEngine::new(engine, event_loop);
                RsWorker::new(Arc::new(py_engine), config)
            };

            let result = worker
                .run_with_outcome(runtime.clone())
                .await
                .map_err(to_pyerr);
            let snapshot_complete = matches!(result, Ok(RsWorkerExit::ExitSuccess));

            // Only tear the runtime down if we constructed it. When a
            // `DistributedRuntime` was already in scope (HTTP frontend,
            // tests, etc.) it owns the shutdown lifecycle and we'd be
            // pulling the rug out from other tasks if we called shutdown.
            if owns_runtime && !snapshot_complete {
                runtime.shutdown();
            } else if !snapshot_complete {
                tracing::debug!(
                    "Worker.run skipping runtime.shutdown(); runtime is \
                     shared with another caller"
                );
            }

            result.map(|outcome| matches!(outcome, RsWorkerExit::ExitSuccess))
        })
    }
}

// ---------------------------------------------------------------------------
// PyEngineCore — the Python-bridge state and lifecycle shared by both wrappers
// (`PyLLMEngine`, `PyRawEngine`): everything that doesn't depend on the request
// modality. The modality-specific `generate` shaping and `kv_event_sources`
// live on the wrappers. Not a `#[pyclass]`; Rust-only.
// ---------------------------------------------------------------------------

struct PyEngineCore {
    // Wrapped in `Arc` so we can clone refcount-style without acquiring
    // the GIL — `PyObject::clone` would otherwise need to bump Python's
    // own refcount, which requires the GIL. Same pattern as
    // `PythonAsyncEngine` in `engine.rs`.
    engine: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    trace_contexts: Arc<StdMutex<HashMap<String, DistributedTraceContext>>>,
    request_metadata: Arc<StdMutex<HashMap<String, BTreeMap<String, String>>>>,
}

impl PyEngineCore {
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

    /// Shared dispatch for the token and raw `generate` paths: records
    /// per-request trace/metadata state, pythonizes the request (any
    /// `Serialize` — `PreprocessedRequest` or `serde_json::Value`), calls
    /// Python `generate(request, context=ctx)`, and returns the resulting
    /// stream of Python chunk objects + the state guard. The caller maps each
    /// `PyObject` to its output type.
    ///
    /// Invariant: must be awaited within the `engine.generate` span (both
    /// adapters `.instrument()` it) so the span capture below sees the right
    /// parent.
    async fn dispatch_generate<T: serde::Serialize + Send + 'static>(
        &self,
        request: T,
        ctx: dynamo_backend_common::GenerateContext,
    ) -> Result<(BoxStream<'static, PyResult<PyObject>>, RequestStateGuard), DynamoError> {
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
        // `engine.generate` span opened by the adapter. The capture must
        // happen BEFORE `spawn_blocking` because inside the blocking closure,
        // `Span::current()` is the worker-thread root, not the auto-span.
        //
        // If anyone refactors this dispatch (extra task hop, different
        // scheduler), they must re-verify the captured span. `Context` stores
        // this span and routes engine telemetry calls to it via
        // `current_span` / `start_span` — wrong span = wrong attributes
        // silently. See the `auto_span_records_*` tests in
        // `lib/backend-common/src/adapter.rs` for the assertions that depend
        // on this invariant.
        debug_assert_eq!(
            tracing::Span::current().metadata().map(|m| m.name()),
            Some("engine.generate"),
            "Span::current() must be engine.generate at PyEngineCore boundary; \
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

        Ok((stream.boxed(), request_state_guard))
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

impl PyEngineCore {
    async fn prepare_for_runtime(
        &self,
        context: RsPreRuntimeContext,
    ) -> Result<RsPreRuntimeOutcome, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();

        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<_> {
                let py_context = Py::new(py, PreRuntimeContext { inner: context })?;
                let bound = engine.bind(py);
                let coroutine = bound.call_method1("prepare_for_runtime", (py_context,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("prepare_for_runtime offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        let result = py_future.await.map_err(py_err_to_dynamo)?;
        Python::with_gil(|py| depythonize_pre_runtime_outcome(result.bind(py)))
            .map_err(py_err_to_dynamo)
    }

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
            // Token-pipeline metadata lives in the optional `.llm` sub-record
            // (LlmRegistration). LLMEngines set it; RawEngines leave it None.
            // Missing attr = the duck-typed default (pre-split / raw engines);
            // any other getattr error is a real bug → propagate, don't swallow.
            let llm = match bound.getattr("llm") {
                Ok(v) if !v.is_none() => Some(RsLlmRegistration {
                    context_length: opt_attr::<u32>(&v, "context_length")?,
                    kv_cache_block_size: opt_attr::<u32>(&v, "kv_cache_block_size")?,
                    total_kv_blocks: opt_attr::<u64>(&v, "total_kv_blocks")?,
                    max_num_seqs: opt_attr::<u64>(&v, "max_num_seqs")?,
                    max_num_batched_tokens: opt_attr::<u64>(&v, "max_num_batched_tokens")?,
                    data_parallel_size: opt_attr::<u32>(&v, "data_parallel_size")?,
                    data_parallel_start_rank: opt_attr::<u32>(&v, "data_parallel_start_rank")?,
                    bootstrap_host: opt_attr::<String>(&v, "bootstrap_host")?,
                    bootstrap_port: opt_attr::<u16>(&v, "bootstrap_port")?,
                }),
                Ok(_) => None,
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => None,
                Err(e) => return Err(e),
            };
            Ok(RsEngineConfig {
                model: bound.getattr("model")?.extract()?,
                served_model_name: opt_attr::<String>(bound, "served_model_name")?,
                runtime_data: match bound.getattr("runtime_data") {
                    Ok(value) if !value.is_none() => depythonize(&value).map_err(to_pyerr)?,
                    Ok(_) => HashMap::new(),
                    Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                        HashMap::new()
                    }
                    Err(e) => return Err(e),
                },
                llm,
            })
        })
        .map_err(py_err_to_dynamo)
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

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        let py_obj = self
            .call_method0_async("is_quiescent")
            .await
            .map_err(py_err_to_dynamo)?;
        Python::with_gil(|py| -> PyResult<Option<bool>> {
            let bound = py_obj.bind(py);
            // Python `None` maps to `Ok(None)`; a bool to `Some(bool)`.
            if bound.is_none() {
                return Ok(None);
            }
            Ok(Some(bound.extract::<bool>()?))
        })
        .map_err(py_err_to_dynamo)
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

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        let engine = self.engine.clone();
        let join = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<String>> {
                let result = engine.bind(py).call_method0("supported_controls")?;
                let mut controls = Vec::new();
                for item in result.try_iter()? {
                    controls.push(item?.extract()?);
                }
                Ok(controls)
            })
        })
        .await;

        match join {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(py_err_to_dynamo(e)),
            Err(join_err) => Err(DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!(
                    "supported_controls spawn_blocking join failed: {join_err}"
                ))
                .build()),
        }
    }

    async fn engine_control(
        &self,
        control: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_body = pythonize(py, &body).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert engine control request body to Python: {e}"
                    ))
                })?;
                let coroutine = engine
                    .bind(py)
                    .call_method1("engine_control", (control, py_body))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("engine_control offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        let py_result = py_future.await.map_err(py_err_to_dynamo)?;

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                depythonize::<serde_json::Value>(py_result.bind(py)).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to serialize engine control response: {e}"
                    ))
                })
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("engine_control response offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        let engine = self.engine.clone();
        let join = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Vec<String>> {
                let result = engine.bind(py).call_method0("supported_updates")?;
                let mut updates = Vec::new();
                for item in result.try_iter()? {
                    updates.push(item?.extract()?);
                }
                Ok(updates)
            })
        })
        .await;

        match join {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(py_err_to_dynamo(e)),
            Err(join_err) => Err(DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!(
                    "supported_updates spawn_blocking join failed: {join_err}"
                ))
                .build()),
        }
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_body = pythonize(py, &body).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert engine update request body to Python: {e}"
                    ))
                })?;
                let coroutine = engine
                    .bind(py)
                    .call_method1("engine_update", (update, py_body))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("engine_update offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        let py_result = py_future.await.map_err(py_err_to_dynamo)?;

        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                depythonize::<serde_json::Value>(py_result.bind(py)).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to serialize engine update response: {e}"
                    ))
                })
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("engine_update response offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)
    }

    async fn on_endpoint_ready(
        &self,
        endpoint: rs::component::Endpoint,
    ) -> Result<(), DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();

        let py_future = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<_> {
                let py_endpoint = Py::new(
                    py,
                    crate::Endpoint {
                        inner: endpoint,
                        event_loop: event_loop.bind(py).clone().unbind(),
                    },
                )?;
                let coroutine = engine
                    .bind(py)
                    .call_method1("on_endpoint_ready", (py_endpoint,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(&locals, coroutine)
            })
        })
        .await
        .map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("on_endpoint_ready offload error: {e}"))
                .build()
        })?
        .map_err(py_err_to_dynamo)?;

        py_future.await.map_err(py_err_to_dynamo)?;
        Ok(())
    }
}

impl PyEngineCore {
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

// ---------------------------------------------------------------------------
// PyLLMEngine — token-pipeline bridge: `PyEngineCore` plus the token `generate`
// (chunks → `LLMEngineOutput`) and `kv_event_sources`.
// ---------------------------------------------------------------------------

struct PyLLMEngine {
    core: PyEngineCore,
}

impl PyLLMEngine {
    fn new(engine: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        Self {
            core: PyEngineCore::new(engine, event_loop),
        }
    }
}

#[async_trait]
impl LLMEngine for PyLLMEngine {
    async fn prepare_for_runtime(
        &self,
        context: RsPreRuntimeContext,
    ) -> Result<RsPreRuntimeOutcome, DynamoError> {
        self.core.prepare_for_runtime(context).await
    }

    async fn start(&self, worker_id: u64) -> Result<RsEngineConfig, DynamoError> {
        self.core.start(worker_id).await
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: dynamo_backend_common::GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let (stream, request_state_guard) = self.core.dispatch_generate(request, ctx).await?;

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
        self.core.abort(ctx).await
    }

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        self.core.is_quiescent().await
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.core.cleanup().await
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        self.core.health_check_payload().await
    }

    async fn kv_event_sources(&self) -> Result<Vec<RsKvEventSource>, DynamoError> {
        let py_list = self
            .core
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

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        self.core.setup_metrics(ctx).await
    }

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        self.core.supported_controls().await
    }

    async fn engine_control(
        &self,
        control: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        self.core.engine_control(control, body).await
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        self.core.supported_updates().await
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        self.core.engine_update(update, body).await
    }

    async fn on_endpoint_ready(
        &self,
        endpoint: rs::component::Endpoint,
    ) -> Result<(), DynamoError> {
        self.core.on_endpoint_ready(endpoint).await
    }
}

// ---------------------------------------------------------------------------
// PyRawEngine — raw-media bridge for Python `DiffusionEngine`: `PyEngineCore`
// plus a `generate` that passes the request/response through as
// `serde_json::Value` (no token/`LLMEngineOutput` shaping).
// ---------------------------------------------------------------------------

struct PyRawEngine {
    core: PyEngineCore,
}

impl PyRawEngine {
    fn new(engine: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        Self {
            core: PyEngineCore::new(engine, event_loop),
        }
    }
}

#[async_trait]
impl RawEngine for PyRawEngine {
    async fn start(&self, worker_id: u64) -> Result<RsEngineConfig, DynamoError> {
        self.core.start(worker_id).await
    }

    async fn generate(
        &self,
        request: serde_json::Value,
        ctx: dynamo_backend_common::GenerateContext,
    ) -> Result<BoxStream<'static, Result<serde_json::Value, DynamoError>>, DynamoError> {
        let (stream, request_state_guard) = self.core.dispatch_generate(request, ctx).await?;

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

                // Depythonize the response object to JSON on a blocking
                // thread — same GIL-contention rationale as the LLM path.
                // No `LLMEngineOutput` shaping: the media response body
                // flows through verbatim.
                let parsed = tokio::task::spawn_blocking(move || {
                    Python::with_gil(|py| -> PyResult<serde_json::Value> {
                        let bound = py_obj.into_bound(py);
                        depythonize(&bound).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "raw engine chunk must be a JSON-serializable object: {e}"
                            ))
                        })
                    })
                })
                .await;

                match parsed {
                    Ok(Ok(value)) => yield Ok(value),
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "failed to parse chunk from python raw engine");
                        yield Err(py_err_to_dynamo(e));
                        return;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "raw chunk parse offload error");
                        yield Err(DynamoError::builder()
                            .error_type(ErrorType::Backend(BackendError::Unknown))
                            .message(format!("raw chunk parse offload error: {e}"))
                            .build());
                        return;
                    }
                }
            }
        };

        Ok(Box::pin(mapped))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        self.core.abort(ctx).await
    }

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        self.core.is_quiescent().await
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.core.cleanup().await
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        self.core.setup_metrics(ctx).await
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        self.core.health_check_payload().await
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

fn depythonize_pre_runtime_outcome(bound: &Bound<'_, PyAny>) -> PyResult<RsPreRuntimeOutcome> {
    let action_obj = bound.getattr("action")?;
    let action: String = match action_obj.getattr("value") {
        Ok(value) => value.extract()?,
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(bound.py()) => {
            action_obj.extract()?
        }
        Err(e) => return Err(e),
    };

    match action.as_str() {
        "continue" => Ok(RsPreRuntimeOutcome::Continue),
        "prepared" => {
            let restored_obj = bound.getattr("restored_runtime")?;
            if restored_obj.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "prepared pre-runtime outcome requires restored_runtime",
                ));
            }
            let restored_runtime = RsRestoredRuntimeConfig {
                namespace: restored_obj.getattr("namespace")?.extract()?,
                discovery_backend: restored_obj.getattr("discovery_backend")?.extract()?,
                request_plane: restored_obj.getattr("request_plane")?.extract()?,
                event_plane: opt_attr::<String>(&restored_obj, "event_plane")?,
            };
            Ok(RsPreRuntimeOutcome::Prepared { restored_runtime })
        }
        "exit_success" => Ok(RsPreRuntimeOutcome::ExitSuccess),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid pre-runtime outcome action {other:?}"
        ))),
    }
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
        // See engine.rs::process_item — emit JSON-shaped message so the OpenAI
        // frontend can read the status code instead of defaulting to 500.
        if let Some((code, message)) = extract_http_like_error(py, &err) {
            let backend = if (400..500).contains(&code) {
                BackendError::InvalidArgument
            } else {
                BackendError::Unknown
            };
            let json_msg = serde_json::json!({ "message": message, "code": code }).to_string();
            return (backend, json_msg);
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
