// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python↔Rust bridge for the AIC (AI Configurator) perf model.
//!
//! [`RustAicCallback`] wraps a compiled `aiconfigurator_core::AicEngine` and
//! answers the mocker/router latency predictions purely in Rust — no GIL on the
//! predict hot path. Requires the `aic-forward-pass` feature; a build failure is
//! a hard error (no Python fallback). KV-block sizing still crosses into Python
//! via [`estimate_aic_num_gpu_blocks`].

#[cfg(feature = "aic-forward-pass")]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "aic-forward-pass")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "aic-forward-pass")]
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "aic-forward-pass")]
use aiconfigurator_core::{AicEngine, build_aic_engine};
use dynamo_kv_router::PrefillLoadEstimator;
use dynamo_mocker::common::perf_model::AicCallback;

/// Pure-Rust AIC callback: wraps an `aiconfigurator_core::AicEngine`
/// compiled once at startup and answers predict calls with NO PyO3 / GIL on the
/// hot path — `AicEngine::{prefill,decode}_latency_ms` are pure Rust.
///
/// `AicEngine` is `Send + Sync` (it is an `Arc<Engine>` over an
/// `Arc<PerfDatabase>`), so no `unsafe impl` is needed, unlike `PyAicCallback`.
#[cfg(feature = "aic-forward-pass")]
pub(super) struct RustAicCallback {
    engine: Arc<AicEngine>,
}

#[cfg(feature = "aic-forward-pass")]
impl AicCallback for RustAicCallback {
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64 {
        // The engine's predict_prefill_latency takes the FULL isl and subtracts
        // `prefix` internally, while the mocker gives us the post-prefix
        // `effective_isl`. Pass `effective_isl + prefix` so the engine recovers
        // the same effective length (and keeps `prefix` for the KV-cache-aware
        // context-attention cost). Mirrors the Python AicSession adapter.
        self.engine
            .prefill_latency_ms(
                batch_size as u32,
                (effective_isl + prefix) as u32,
                prefix as u32,
            )
            .unwrap_or_else(|e| panic!("AIC predict_prefill (rust) failed: {e}"))
    }

    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64 {
        self.engine
            .decode_latency_ms(batch_size as u32, isl as u32, osl as u32)
            .unwrap_or_else(|e| panic!("AIC predict_decode (rust) failed: {e}"))
    }
}

#[cfg(feature = "aic-forward-pass")]
impl PrefillLoadEstimator for RustAicCallback {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration> {
        let latency_ms = self
            .engine
            .prefill_latency_ms(
                batch_size as u32,
                (effective_isl + prefix) as u32,
                prefix as u32,
            )
            .map_err(|e| anyhow::anyhow!("AIC predict_prefill (rust) failed: {e}"))?;
        Ok(Duration::from_secs_f64(latency_ms / 1000.0))
    }
}

/// Point the Rust AIC core at the data bundled in the installed Python wheel.
#[cfg(feature = "aic-forward-pass")]
pub(super) fn configure_default_data_roots(py: Python<'_>) {
    if let Err(error) = py
        .import("aiconfigurator.sdk.rust_engine_step")
        .and_then(|module| module.call_method0("_configure_default_data_roots"))
    {
        tracing::warn!(
            error = %error,
            "AIC could not configure data roots; relying on the build-time default"
        );
    }
}

/// Build the pure-Rust AIC engine ONCE at startup and cache it per identity.
/// `build_aic_engine` crosses into Python once here (shared pyo3 interpreter) to
/// run `compile_engine`; the returned engine's predict hot path is pure Rust.
///
/// A build failure is a HARD ERROR — there is no Python fallback. The requested
/// model/system/backend must be supported by the Rust engine (aiconfigurator's
/// `compile_engine` covers every supported config), so a failure means a real
/// problem (missing perf data, bad config) and should surface, not silently
/// degrade to the slower GIL-bound Python op-walk.
#[cfg(feature = "aic-forward-pass")]
#[allow(clippy::too_many_arguments)]
fn build_rust_engine(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    backend_version: Option<&str>,
    moe_tp_size: Option<usize>,
    moe_ep_size: Option<usize>,
    attention_dp_size: Option<usize>,
    gemm_dtype: Option<&str>,
    moe_dtype: Option<&str>,
    fmha_dtype: Option<&str>,
    kv_cache_dtype: Option<&str>,
    comm_dtype: Option<&str>,
    nextn: Option<usize>,
    nextn_accept_rates: Option<&str>,
) -> PyResult<Arc<AicEngine>> {
    // Speculative (MTP) decoding: forward the mocker's nextn / accept-rates to
    // the engine build, mirroring the Python AicSession path. Dense models pass
    // nextn=0 and no rates. accept-rates arrive comma-separated from the caller.
    let nextn = nextn.unwrap_or(0) as u32;
    let nextn_accept_rates: Option<Vec<f64>> = match nextn_accept_rates {
        Some(s) if !s.trim().is_empty() => Some(
            s.split(',')
                .map(|x| x.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "AIC: invalid nextn_accept_rates {s:?}: {e}"
                    ))
                })?,
        ),
        _ => None,
    };
    // Resolve each quant-mode string through the single Python source of truth
    // (`dynamo._internal.aic._resolve_quant_mode_name`) so this latency-engine
    // path matches the Python paths (`create_session`/`estimate_num_gpu_blocks`)
    // exactly: it normalizes the vocabulary (`auto`/`none`/`null` -> default,
    // `int4` -> `int4_wo`) AND validates per field, rejecting unsupported
    // field/dtype combos (e.g. `kvcache=int4`) up front with a clear error
    // instead of a generic failure from `build_aic_engine`. Done before the
    // cache key so equivalent spellings share one compiled engine.
    let aic_module = py.import("dynamo._internal.aic")?;
    let resolve_quant_mode = |field: &str, value: Option<&str>| -> PyResult<Option<String>> {
        aic_module
            .call_method1("_resolve_quant_mode_name", (field, value))?
            .extract()
    };
    let gemm_dtype = resolve_quant_mode("gemm", gemm_dtype)?;
    let moe_dtype = resolve_quant_mode("moe", moe_dtype)?;
    let fmha_dtype = resolve_quant_mode("fmha", fmha_dtype)?;
    let kv_cache_dtype = resolve_quant_mode("kvcache", kv_cache_dtype)?;
    let comm_dtype = resolve_quant_mode("comm", comm_dtype)?;

    // Cache the compiled engine per identity. build_aic_engine compiles the
    // model (Python) and loads the perf DB (Rust parquet) — a one-time startup
    // cost, but callers may construct several callbacks (per-worker,
    // prefill+decode). Mirror the Python `_cached_engine_handle` so the build is
    // paid once per unique config (speculative config included).
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<AicEngine>>>> = OnceLock::new();
    let key = format!(
        "{backend_name}|{system}|{backend_version:?}|{model_path}|{tp_size}|{moe_tp_size:?}|{moe_ep_size:?}|{attention_dp_size:?}|{gemm_dtype:?}|{moe_dtype:?}|{fmha_dtype:?}|{kv_cache_dtype:?}|{comm_dtype:?}|{nextn}|{nextn_accept_rates:?}"
    );
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(existing) = cache.lock().unwrap().get(&key) {
        return Ok(Arc::clone(existing));
    }
    // Reuse aiconfigurator's own systems-path resolution: this sets
    // AICONFIGURATOR_SYSTEMS_PATH in the process env, which build_aic_engine
    // reads for the Rust-side perf-DB load.
    configure_default_data_roots(py);
    let engine = build_aic_engine(
        model_path,
        system,
        backend_name,
        backend_version,
        tp_size as u32,
        1, // pp_size
        attention_dp_size.unwrap_or(1) as u32,
        moe_tp_size.map(|x| x as u32),
        moe_ep_size.map(|x| x as u32),
        gemm_dtype.as_deref(),     // gemm_quant_mode
        moe_dtype.as_deref(),      // moe_quant_mode
        kv_cache_dtype.as_deref(), // kvcache_quant_mode
        fmha_dtype.as_deref(),     // fmha_quant_mode
        comm_dtype.as_deref(),     // comm_quant_mode
        nextn,                     // speculative (MTP) tokens; 0 for dense
        nextn_accept_rates,        // per-position accept rates
        None,                      // kv_block_size
        None,                      // systems_path (resolved via env above / build-time default)
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "AIC: failed to build the Rust engine for {model_path} / {system} / {backend_name}: {e}"
        ))
    })?;
    tracing::info!("AIC: using pure-Rust RustAicCallback (no GIL on the predict hot path)");
    let arc = Arc::new(engine);
    cache.lock().unwrap().insert(key, Arc::clone(&arc));
    Ok(arc)
}

/// Build the AIC latency callback. Called once at mocker startup when
/// `--aic-perf-model` is requested. Requires the `aic-forward-pass` feature.
#[cfg_attr(not(feature = "aic-forward-pass"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(super) fn create_aic_callback(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    backend_version: Option<&str>,
    moe_tp_size: Option<usize>,
    moe_ep_size: Option<usize>,
    attention_dp_size: Option<usize>,
    gemm_dtype: Option<&str>,
    moe_dtype: Option<&str>,
    fmha_dtype: Option<&str>,
    kv_cache_dtype: Option<&str>,
    comm_dtype: Option<&str>,
    nextn: Option<usize>,
    nextn_accept_rates: Option<&str>,
) -> PyResult<Arc<dyn AicCallback>> {
    #[cfg(feature = "aic-forward-pass")]
    {
        let engine = build_rust_engine(
            py,
            backend_name,
            system,
            model_path,
            tp_size,
            backend_version,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            gemm_dtype,
            moe_dtype,
            fmha_dtype,
            kv_cache_dtype,
            comm_dtype,
            nextn,
            nextn_accept_rates,
        )?;
        Ok(Arc::new(RustAicCallback { engine }))
    }
    #[cfg(not(feature = "aic-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIC perf model requires the `aic-forward-pass` feature; rebuild the dynamo bindings with `--features aic-forward-pass`",
    ))
}

/// Build the AIC prefill-load estimator for the KV router / live path. Requires
/// the `aic-forward-pass` feature; a build failure is a hard error (no fallback).
#[cfg_attr(not(feature = "aic-forward-pass"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(super) fn create_aic_prefill_load_estimator(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    backend_version: Option<&str>,
    moe_tp_size: Option<usize>,
    moe_ep_size: Option<usize>,
    attention_dp_size: Option<usize>,
    gemm_dtype: Option<&str>,
    moe_dtype: Option<&str>,
    fmha_dtype: Option<&str>,
    kv_cache_dtype: Option<&str>,
    comm_dtype: Option<&str>,
    nextn: Option<usize>,
    nextn_accept_rates: Option<&str>,
) -> PyResult<Arc<dyn PrefillLoadEstimator>> {
    #[cfg(feature = "aic-forward-pass")]
    {
        let engine = build_rust_engine(
            py,
            backend_name,
            system,
            model_path,
            tp_size,
            backend_version,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            gemm_dtype,
            moe_dtype,
            fmha_dtype,
            kv_cache_dtype,
            comm_dtype,
            nextn,
            nextn_accept_rates,
        )?;
        Ok(Arc::new(RustAicCallback { engine }))
    }
    #[cfg(not(feature = "aic-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIC perf model requires the `aic-forward-pass` feature; rebuild the dynamo bindings with `--features aic-forward-pass`",
    ))
}

/// Estimate the KV block pool size from AIC's base-model memory model.
#[allow(clippy::too_many_arguments)]
pub(super) fn estimate_aic_num_gpu_blocks(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    block_size: usize,
    max_num_batched_tokens: usize,
    gpu_memory_utilization: f64,
    mem_fraction_static: Option<f64>,
    free_gpu_memory_fraction: Option<f64>,
    backend_version: Option<&str>,
    moe_tp_size: Option<usize>,
    moe_ep_size: Option<usize>,
    attention_dp_size: Option<usize>,
    gemm_dtype: Option<&str>,
    moe_dtype: Option<&str>,
    fmha_dtype: Option<&str>,
    kv_cache_dtype: Option<&str>,
    comm_dtype: Option<&str>,
) -> PyResult<usize> {
    let module = py.import("dynamo._internal.aic")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("backend_name", backend_name)?;
    kwargs.set_item("system", system)?;
    kwargs.set_item("model_path", model_path)?;
    kwargs.set_item("tp_size", tp_size)?;
    kwargs.set_item("block_size", block_size)?;
    kwargs.set_item("max_num_batched_tokens", max_num_batched_tokens)?;
    kwargs.set_item("gpu_memory_utilization", gpu_memory_utilization)?;
    kwargs.set_item("mem_fraction_static", mem_fraction_static)?;
    kwargs.set_item("free_gpu_memory_fraction", free_gpu_memory_fraction)?;
    kwargs.set_item("backend_version", backend_version)?;
    kwargs.set_item("moe_tp_size", moe_tp_size)?;
    kwargs.set_item("moe_ep_size", moe_ep_size)?;
    kwargs.set_item("attention_dp_size", attention_dp_size)?;
    kwargs.set_item("gemm_dtype", gemm_dtype)?;
    kwargs.set_item("moe_dtype", moe_dtype)?;
    kwargs.set_item("fmha_dtype", fmha_dtype)?;
    kwargs.set_item("kv_cache_dtype", kv_cache_dtype)?;
    kwargs.set_item("comm_dtype", comm_dtype)?;
    let blocks = module.call_method("estimate_num_gpu_blocks", (), Some(&kwargs))?;
    blocks.extract()
}
