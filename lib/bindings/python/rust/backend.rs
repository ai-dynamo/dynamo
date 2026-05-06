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

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DynamoError, EngineConfig as RsEngineConfig, ErrorType,
    FinishReason, LLMEngine, LLMEngineOutput, PreprocessedRequest,
    RuntimeConfig as RsRuntimeConfig, Worker as RsWorker, WorkerConfig as RsWorkerConfig,
};
use dynamo_llm::model_type::ModelInput as RsModelInput;
use dynamo_runtime as rs;
use futures::stream::{BoxStream, StreamExt};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};

use crate::ModelInput;
use crate::context::Context as PyContext;
use crate::to_pyerr;

/// Register `dynamo._core.backend` and its classes on the parent `_core` module.
pub fn add_to_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "backend")?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<WorkerConfig>()?;
    m.add_class::<Worker>()?;
    parent.add_submodule(&m)?;
    Ok(())
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
    ))]
    fn new(
        model: String,
        served_model_name: Option<String>,
        context_length: Option<u32>,
        kv_cache_block_size: Option<u32>,
        total_kv_blocks: Option<u64>,
        max_num_seqs: Option<u64>,
        max_num_batched_tokens: Option<u64>,
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
}

#[pymethods]
impl Worker {
    #[new]
    fn new(
        engine: PyObject,
        config: WorkerConfig,
        event_loop: PyObject,
    ) -> PyResult<Self> {
        // Ensure the dynamo Runtime + pyo3-async-runtimes bridge are
        // initialized exactly once per process. Mirrors what
        // `DistributedRuntime.__new__` does in `lib.rs`.
        rs::Worker::runtime_from_existing()
            .or_else(|_| {
                let worker = rs::Worker::from_settings()?;
                let primary = worker.tokio_runtime()?;
                // `init_with_runtime` is idempotent across pyo3-async-runtimes
                // versions — ignore the "already initialized" error.
                let _ = pyo3_async_runtimes::tokio::init_with_runtime(primary);
                Ok::<_, anyhow::Error>(worker.runtime().clone())
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            engine: Arc::new(engine),
            event_loop: Arc::new(event_loop),
            config: config.inner,
        })
    }

    /// Drive the full lifecycle: start engine → register model → serve →
    /// (on signal) orchestrate graceful shutdown → cleanup → return.
    fn run<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();
        let config = self.config.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Apply runtime overrides before constructing the runtime
            // (matches the sync `apply_to_env` call in `run.rs`).
            config.runtime.apply_to_env();

            let runtime = rs::Worker::runtime_from_existing()
                .or_else(|_| {
                    let worker = rs::Worker::from_settings()?;
                    Ok::<_, anyhow::Error>(worker.runtime().clone())
                })
                .map_err(to_pyerr)?;

            let py_engine = PyLLMEngine::new(engine, event_loop);
            let worker = RsWorker::new(Arc::new(py_engine), config);

            let result = worker.run(runtime.clone()).await.map_err(to_pyerr);

            // Phase 1/2/3 token-cancellation + NATS/etcd disconnect.
            // `Worker::run` already did discovery unregister, drain, and
            // engine.cleanup() at this point — this is purely transport
            // teardown. Skipped on shared-runtime processes that own
            // shutdown elsewhere (e.g. tests using DistributedRuntime).
            runtime.shutdown();

            result
        })
    }
}

// ---------------------------------------------------------------------------
// PyLLMEngine — the actual bridge. Not a `#[pyclass]`; lives only in Rust.
// ---------------------------------------------------------------------------

struct PyLLMEngine {
    engine: Arc<PyObject>,
    event_loop: Arc<PyObject>,
}

impl PyLLMEngine {
    fn new(engine: Arc<PyObject>, event_loop: Arc<PyObject>) -> Self {
        Self { engine, event_loop }
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

#[async_trait]
impl LLMEngine for PyLLMEngine {
    async fn start(&self) -> Result<RsEngineConfig, DynamoError> {
        let result = self
            .call_method0_async("start")
            .await
            .map_err(py_err_to_dynamo)?;

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
                served_model_name: opt_attr::<String>(bound, "served_model_name"),
                context_length: opt_attr::<u32>(bound, "context_length"),
                kv_cache_block_size: opt_attr::<u32>(bound, "kv_cache_block_size"),
                total_kv_blocks: opt_attr::<u64>(bound, "total_kv_blocks"),
                max_num_seqs: opt_attr::<u64>(bound, "max_num_seqs"),
                max_num_batched_tokens: opt_attr::<u64>(bound, "max_num_batched_tokens"),
            })
        })
        .map_err(py_err_to_dynamo)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        let engine = self.engine.clone();
        let event_loop = self.event_loop.clone();

        // Pythonize the request, call generate(request, context=ctx), and
        // turn the resulting Python async generator into a Rust stream.
        let stream = tokio::task::spawn_blocking(move || -> PyResult<_> {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let py_ctx = Py::new(py, PyContext::new(ctx, None))?;

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
            let mut inner = std::pin::pin!(stream);
            while let Some(item) = inner.next().await {
                let py_obj = match item {
                    Ok(obj) => obj,
                    Err(e) => {
                        // Python engine raised mid-stream — terminate with
                        // an Error chunk so downstream clients see a
                        // well-formed terminal frame.
                        let msg = e.to_string();
                        yield LLMEngineOutput {
                            finish_reason: Some(FinishReason::Error(msg)),
                            ..LLMEngineOutput::default()
                        };
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
                    Ok(Ok(chunk)) => yield chunk,
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "failed to parse chunk from python engine");
                        yield LLMEngineOutput {
                            finish_reason: Some(FinishReason::Error(e.to_string())),
                            ..LLMEngineOutput::default()
                        };
                        return;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "chunk parse offload error");
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

        let res: Result<(), PyErr> = async move {
            let py_future = tokio::task::spawn_blocking(move || {
                Python::with_gil(|py| -> PyResult<_> {
                    let bound = engine.bind(py);
                    let py_ctx = Py::new(py, PyContext::new(ctx, None))?;
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
        // Treat `drain` as optional even if the ABC defines a default
        // no-op: engines that never override it can short-circuit so we
        // don't pay a GIL hop during shutdown.
        let has_drain = Python::with_gil(|py| {
            self.engine
                .bind(py)
                .hasattr("drain")
                .unwrap_or(false)
        });
        if !has_drain {
            return Ok(());
        }
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

fn opt_attr<T>(bound: &Bound<'_, PyAny>, name: &str) -> Option<T>
where
    T: for<'py> FromPyObject<'py>,
{
    bound
        .getattr(name)
        .ok()
        .and_then(|v| if v.is_none() { None } else { v.extract().ok() })
}

/// Map a Python exception class to the closest `BackendError` variant.
/// Mirrors the existing logic in `engine.rs::process_item` so engines
/// running through the new bridge produce the same error categories.
fn py_err_to_dynamo(err: PyErr) -> DynamoError {
    let backend = Python::with_gil(|py| {
        if err.is_instance_of::<pyo3::exceptions::PyValueError>(py)
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
        }
    });
    DynamoError::builder()
        .error_type(ErrorType::Backend(backend))
        .message(err.to_string())
        .build()
}
