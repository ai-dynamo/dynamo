// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Dynamo's NVTX timeline annotations.
//!
//! These thin wrappers expose the runtime's NVTX helpers to Python so that
//! Python workers (the TRT-LLM / vLLM / SGLang backends) can annotate their own
//! hot paths on the same Nsight Systems timeline as the Rust core, under one
//! shared gate: build with `--features nvtx` and set `DYN_ENABLE_RUST_NVTX=1`.
//! When either is off every function below is a cheap no-op.
//!
//! The ergonomic Python wrapper lives in `dynamo/nvtx.py`; these are the raw
//! primitives it builds on.

use pyo3::prelude::*;

use dynamo_runtime::nvtx as rt_nvtx;

/// Whether NVTX annotations are active (the `nvtx` feature is compiled in AND
/// `DYN_ENABLE_RUST_NVTX` is set). Lets callers skip building expensive span
/// names when profiling is off.
#[pyfunction]
fn nvtx_enabled() -> bool {
    rt_nvtx::enabled()
}

/// Open a correlated NVTX range and return an id to pass to `nvtx_range_end`.
///
/// Returns `0` when NVTX is disabled. Correlated ranges are not tied to the
/// thread-local push/pop stack, so they are safe to open and close across
/// `await` points and interleaved coroutines — unlike a scope guard.
#[pyfunction]
fn nvtx_range_start(name: &str) -> i64 {
    rt_nvtx::range_start_impl(name)
}

/// Close a correlated NVTX range opened by `nvtx_range_start`. No-op for id `0`.
#[pyfunction]
fn nvtx_range_end(id: i64) {
    rt_nvtx::range_end_impl(id);
}

/// Register the NVTX primitives on the top-level `_core` module.
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nvtx_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(nvtx_range_start, m)?)?;
    m.add_function(wrap_pyfunction!(nvtx_range_end, m)?)?;
    Ok(())
}
