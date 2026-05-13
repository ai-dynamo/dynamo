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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DisaggregationMode as RsDisaggregationMode, DynamoError,
    EngineConfig as RsEngineConfig, ErrorType, LLMEngine, LLMEngineOutput, PreprocessedRequest,
    RuntimeConfig as RsRuntimeConfig, Worker as RsWorker, WorkerConfig as RsWorkerConfig,
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
use crate::errors::py_exception_to_backend_error;
use crate::to_pyerr;

/// Register `dynamo._core.backend` and its classes on the parent `_core` module.
pub fn add_to_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "backend")?;
    m.add_class::<DisaggregationMode>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<WorkerConfig>()?;
    m.add_class::<Worker>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("dynamo._core.backend", &m)?;
    Ok(())
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
        bootstrap_host = None,
        bootstrap_port = None,
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
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) -> Self {
        Self {
            inner: RsEngineConfig {
                model,
                served_model_name,
                context_length,
                kv_cache_block_size,
                total_kv_blocks,
                max_num_seqs,
                max_num_batched_tokens,
                bootstrap_host,
                bootstrap_port,
            },
        }
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
    fn bootstrap_host(&self) -> Option<&str> {
        self.inner.bootstrap_host.as_deref()
    }
    #[getter]
    fn bootstrap_port(&self) -> Option<u16> {
        self.inner.bootstrap_port
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
        metrics_labels = Vec::new(),
        runtime = None,
        disaggregation_mode = DisaggregationMode::Aggregated,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        metrics_labels: Vec<(String, String)>,
        runtime: Option<RuntimeConfig>,
        disaggregation_mode: DisaggregationMode,
    ) -> Self {
        // Delegating to the same conversion used by `register_model`.
        let model_input_rs = match model_input {
            ModelInput::Text => RsModelInput::Text,
            ModelInput::Tokens => RsModelInput::Tokens,
            ModelInput::Tensor => RsModelInput::Tensor,
        };
        Self {
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
                metrics_labels,
                disaggregation_mode: disaggregation_mode.into(),
                runtime: runtime.map(|r| r.inner).unwrap_or_default(),
            },
        }
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
}

impl PyLLMEngine {
    fn new(engine: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        Self {
            engine,
            event_loop,
            trace_contexts: Arc::new(StdMutex::new(HashMap::new())),
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

struct TraceContextGuard {
    request_id: String,
    trace_contexts: Arc<StdMutex<HashMap<String, DistributedTraceContext>>>,
}

impl Drop for TraceContextGuard {
    fn drop(&mut self) {
        self.trace_contexts
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
            Ok(RsEngineConfig {
                model: bound.getattr("model")?.extract()?,
                served_model_name: opt_attr::<String>(bound, "served_model_name")?,
                context_length: opt_attr::<u32>(bound, "context_length")?,
                kv_cache_block_size: opt_attr::<u32>(bound, "kv_cache_block_size")?,
                total_kv_blocks: opt_attr::<u64>(bound, "total_kv_blocks")?,
                max_num_seqs: opt_attr::<u64>(bound, "max_num_seqs")?,
                max_num_batched_tokens: opt_attr::<u64>(bound, "max_num_batched_tokens")?,
                bootstrap_host: opt_attr::<String>(bound, "bootstrap_host")?,
                bootstrap_port: opt_attr::<u16>(bound, "bootstrap_port")?,
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
        let trace_guard = trace_context.as_ref().map(|trace_context| {
            self.trace_contexts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(request_id.clone(), trace_context.clone());
            TraceContextGuard {
                request_id,
                trace_contexts: self.trace_contexts.clone(),
            }
        });

        let first_token = ctx.first_token_sender().cloned();
        let inner_ctx = ctx.inner_arc();

        // Pythonize the request, call generate(request, context=ctx), and
        // turn the resulting Python async generator into a Rust stream.
        let stream = tokio::task::spawn_blocking(move || -> PyResult<_> {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let py_ctx = Py::new(py, PyContext::new(inner_ctx, trace_context, first_token))?;

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
            let _trace_guard = trace_guard;
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

        let res: Result<(), PyErr> = async move {
            let py_future = tokio::task::spawn_blocking(move || {
                Python::with_gil(|py| -> PyResult<_> {
                    let bound = engine.bind(py);
                    let py_ctx = Py::new(py, PyContext::new(ctx, trace_context, None))?;
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
