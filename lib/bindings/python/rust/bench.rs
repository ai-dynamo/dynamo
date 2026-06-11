// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process microbenchmark entries, gated behind the `bench-harness` feature
//! (NOT in production wheels). Both drive [`run_load`] through the production
//! `EngineAdapter` with no NATS / etcd and return stats as a dict:
//!
//! - [`bench_unified_python_engine`]: a Python engine via the real
//!   `PyLLMEngine` bridge — measures bridge + GIL cost.
//! - [`bench_unified_rust_floor`]: the GIL-free `BenchFloorEngine` baseline.

use std::sync::Arc;

use dynamo_backend_common::testing::bench::{BenchFloorEngine, BenchStats, BenchWorkload, run_load};
use dynamo_backend_common::{DisaggregationMode, LLMEngine};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backend::PyLLMEngine;
use crate::to_pyerr;

/// Init the pyo3-async-runtimes tokio runtime if no `DistributedRuntime`
/// already wired one. Mirrors `backend::Worker::new`.
fn ensure_tokio_runtime() -> PyResult<()> {
    if dynamo_runtime::Worker::has_existing_runtime() {
        return Ok(());
    }
    let worker = dynamo_runtime::Worker::from_settings().map_err(to_pyerr)?;
    let primary = worker.tokio_runtime().map_err(to_pyerr)?;
    let _ = pyo3_async_runtimes::tokio::init_with_runtime(primary);
    Ok(())
}

/// Convert aggregated stats into a Python dict.
fn stats_to_py(py: Python<'_>, stats: &BenchStats) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("requests", stats.requests)?;
    d.set_item("total_output_tokens", stats.total_output_tokens)?;
    d.set_item("wall_seconds", stats.wall_seconds)?;
    d.set_item("tokens_per_sec", stats.tokens_per_sec)?;
    d.set_item("ttft_p50_ms", stats.ttft_p50_ms)?;
    d.set_item("ttft_p99_ms", stats.ttft_p99_ms)?;
    d.set_item("itl_p50_ms", stats.itl_p50_ms)?;
    d.set_item("itl_p99_ms", stats.itl_p99_ms)?;
    Ok(d.into_any().unbind())
}

/// Awaitable for the unified path: wrap the `LLMEngine` in `EngineAdapter`.
fn run_load_to_py<'p>(
    py: Python<'p>,
    engine: Arc<dyn LLMEngine>,
    workload: BenchWorkload,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let stats = run_load(engine, DisaggregationMode::Aggregated, workload).await;
        Python::with_gil(|py| stats_to_py(py, &stats))
    })
}

/// Drive a Python `LLMEngine` through the `PyLLMEngine` bridge + `EngineAdapter`.
/// `engine` is constructed but NOT started; `event_loop` is the running asyncio
/// loop (pass `asyncio.get_running_loop()`).
#[pyfunction]
#[pyo3(signature = (
    engine, event_loop, model, prompt_len, max_tokens,
    logprobs_k, concurrency, total_requests,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_unified_python_engine<'p>(
    py: Python<'p>,
    engine: PyObject,
    event_loop: PyObject,
    model: String,
    prompt_len: usize,
    max_tokens: u32,
    logprobs_k: Option<u32>,
    concurrency: usize,
    total_requests: usize,
) -> PyResult<Bound<'p, PyAny>> {
    ensure_tokio_runtime()?;
    let engine: Arc<dyn LLMEngine> =
        Arc::new(PyLLMEngine::new(Arc::new(engine), Arc::new(event_loop)));
    run_load_to_py(
        py,
        engine,
        BenchWorkload {
            model,
            prompt_len,
            max_tokens,
            logprobs_k,
            concurrency,
            total_requests,
        },
    )
}

/// Drive the GIL-free `BenchFloorEngine` through the same path.
/// `per_token_delay_ms` should match the Python engine's delay (0.0 = no pacing).
#[pyfunction]
#[pyo3(signature = (
    model, prompt_len, max_tokens, logprobs_k,
    per_token_delay_ms, concurrency, total_requests,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_unified_rust_floor<'p>(
    py: Python<'p>,
    model: String,
    prompt_len: usize,
    max_tokens: u32,
    logprobs_k: Option<u32>,
    per_token_delay_ms: f64,
    concurrency: usize,
    total_requests: usize,
) -> PyResult<Bound<'p, PyAny>> {
    ensure_tokio_runtime()?;
    let engine: Arc<dyn LLMEngine> = Arc::new(BenchFloorEngine::new(per_token_delay_ms));
    run_load_to_py(
        py,
        engine,
        BenchWorkload {
            model,
            prompt_len,
            max_tokens,
            logprobs_k,
            concurrency,
            total_requests,
        },
    )
}
