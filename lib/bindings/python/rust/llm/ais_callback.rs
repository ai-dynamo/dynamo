// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical AISimulate model construction and pure Rust scheduler adapters.
#[cfg(feature = "ais-forward-pass")]
use aisimulate_core::{
    ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassPerfReadiness,
    ScheduledRequestMetrics,
};
use dynamo_kv_router::PrefillLoadEstimator;
use dynamo_mocker::common::perf_model::AisCallback;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
#[cfg(feature = "ais-forward-pass")]
use std::time::Duration;

pub(super) fn ais_worker_type(
    worker_type: dynamo_mocker::common::protocols::WorkerType,
) -> &'static str {
    use dynamo_mocker::common::protocols::WorkerType;
    match worker_type {
        WorkerType::Aggregated => "aggregated",
        WorkerType::Prefill => "prefill",
        WorkerType::Decode => "decode",
    }
}

#[cfg(feature = "ais-forward-pass")]
pub(super) struct RustAisCallback {
    model: ForwardPassPerfModel,
    nextn: u32,
}

#[cfg(feature = "ais-forward-pass")]
fn checked_count(value: usize, field: &str) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("AIS {field} exceeds u32"))
}

#[cfg(feature = "ais-forward-pass")]
fn prefill_metrics(
    batch_size: usize,
    effective_isl: usize,
    prefix: usize,
) -> anyhow::Result<ForwardPassMetrics> {
    let batch = checked_count(batch_size, "batch size")?;
    let tokens = checked_count(effective_isl, "prefill tokens")?;
    let prefix = checked_count(prefix, "prefix tokens")?;
    Ok(ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: batch,
            sum_prefill_tokens: batch
                .checked_mul(tokens)
                .ok_or_else(|| anyhow::anyhow!("AIS prefill total exceeds u32"))?,
            sum_prefill_kv_tokens: batch
                .checked_mul(prefix)
                .ok_or_else(|| anyhow::anyhow!("AIS prefix total exceeds u32"))?,
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(feature = "ais-forward-pass")]
fn decode_metrics(
    batch_size: usize,
    context: usize,
    nextn: u32,
) -> anyhow::Result<ForwardPassMetrics> {
    // FPM counts are already packed. Preserve the former static-engine adapter's
    // verification width explicitly, without applying attention-DP a second time.
    let batch = checked_count(batch_size, "batch size")?
        .checked_mul(nextn + 1)
        .ok_or_else(|| anyhow::anyhow!("AIS verification batch exceeds u32"))?;
    let context = checked_count(context, "decode context")?;
    Ok(ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: batch,
            sum_decode_kv_tokens: batch
                .checked_mul(context)
                .ok_or_else(|| anyhow::anyhow!("AIS decode KV total exceeds u32"))?,
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(feature = "ais-forward-pass")]
impl RustAisCallback {
    fn estimate(&self, metrics: ForwardPassMetrics) -> anyhow::Result<f64> {
        let value = self
            .model
            .estimate_forward_pass_time_ms(&[metrics])?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "AIS estimator is not ready; Router/Mocker has no FPM training source"
                )
            })?;
        anyhow::ensure!(
            value.is_finite() && value >= 0.0,
            "AIS returned invalid latency {value}"
        );
        Ok(value)
    }
}

#[cfg(feature = "ais-forward-pass")]
impl AisCallback for RustAisCallback {
    fn predict_prefill(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<f64> {
        self.estimate(prefill_metrics(batch_size, effective_isl, prefix)?)
    }
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> anyhow::Result<f64> {
        // Legacy API returns a trajectory sum. Mocker uses osl=2, i.e. one
        // generation step at isl+1; preserve this exact context convention.
        let mut total = 0.0;
        let stride = aisimulate_core::perfmodel::engine::DEFAULT_STATIC_STRIDE as usize;
        for step in (1..osl).step_by(stride) {
            let context = isl
                .checked_add(step)
                .ok_or_else(|| anyhow::anyhow!("AIS decode context overflow"))?;
            total += self.estimate(decode_metrics(batch_size, context, self.nextn)?)?
                * (osl - step).min(stride) as f64;
        }
        Ok(total)
    }
}

#[cfg(feature = "ais-forward-pass")]
impl PrefillLoadEstimator for RustAisCallback {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration> {
        Ok(Duration::try_from_secs_f64(
            self.predict_prefill(batch_size, effective_isl, prefix)? / 1000.0,
        )?)
    }
}

#[cfg(feature = "ais-forward-pass")]
fn build_model(config: &serde_json::Value) -> PyResult<RustAisCallback> {
    let config: ForwardPassPerfModelConfig =
        serde_json::from_value(config.clone()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid AIS perf config: {e}"))
        })?;
    let nextn = config
        .speculation
        .as_ref()
        .map_or(config.nextn, |spec| spec.num_speculative_tokens());
    let model = ForwardPassPerfModel::best_available(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("AIS model construction failed: {e}"))
    })?;
    if model.diagnostics().readiness != ForwardPassPerfReadiness::Ready {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "AIS estimator is not ready; Router/Mocker requires a ready model because it has no FPM training source",
        ));
    }
    if nextn > 0
        && model.provenance().is_some_and(|p| {
            p.selected_estimation_mode == aisimulate_core::EstimationMode::FpmInterpolation
        })
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "AIS canonical FPM queries do not yet support speculative FPM interpolation; use op_level",
        ));
    }
    Ok(RustAisCallback { model, nextn })
}

#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
pub(super) fn create_ais_callback(
    _py: Python<'_>,
    config: &serde_json::Value,
) -> PyResult<Arc<dyn AisCallback>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        Ok(Arc::new(build_model(config)?))
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the ais-forward-pass feature",
    ))
}

#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
pub(super) fn create_ais_prefill_load_estimator(
    _py: Python<'_>,
    config: &serde_json::Value,
) -> PyResult<Arc<dyn PrefillLoadEstimator>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        Ok(Arc::new(build_model(config)?))
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the ais-forward-pass feature",
    ))
}

#[cfg(feature = "ais-forward-pass")]
#[allow(clippy::too_many_arguments)]
fn build_legacy_model(
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
    worker_type: &str,
) -> PyResult<(ForwardPassPerfModel, u32)> {
    let module = py.import("dynamo._internal.ais")?;
    if nextn.unwrap_or(0) > 0 {
        module.call_method1("_pad_nextn_accept_rates", (nextn_accept_rates,))?;
    }
    let mut config = serde_json::json!({"model": model_path, "system": system, "backend": backend_name,
        "worker_type": worker_type, "tp": tp_size, "attention_dp": attention_dp_size.unwrap_or(1),
        "backend_version": backend_version, "moe_tp_size": moe_tp_size, "moe_ep_size": moe_ep_size,
        "nextn": nextn.unwrap_or(0)});
    for (field, key, value) in [
        ("gemm", "gemm_quant_mode", gemm_dtype),
        ("moe", "moe_quant_mode", moe_dtype),
        ("fmha", "fmha_quant_mode", fmha_dtype),
        ("kvcache", "kvcache_quant_mode", kv_cache_dtype),
        ("comm", "comm_quant_mode", comm_dtype),
    ] {
        let normalized: Option<String> = module
            .call_method1("_resolve_quant_mode_name", (field, value))?
            .extract()?;
        config[key] = serde_json::to_value(normalized).expect("string serialization");
    }
    let callback = build_model(&config)?;
    Ok((callback.model, callback.nextn))
}

/// Adapt legacy flat inputs once, then construct the canonical AIS model.
#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
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
    worker_type: &str,
) -> PyResult<Arc<dyn AisCallback>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        let engine = build_legacy_model(
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
            worker_type,
        )?;
        Ok(Arc::new(RustAisCallback {
            model: engine.0,
            nextn: engine.1,
        }))
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the `ais-forward-pass` feature; rebuild the dynamo bindings with `--features ais-forward-pass`",
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
    let module = py.import("dynamo._internal.ais")?;
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

#[cfg(all(test, feature = "ais-forward-pass"))]
mod tests {
    use super::{decode_metrics, prefill_metrics};

    #[test]
    fn scheduler_queries_preserve_prefix_and_local_attention_dp_batch() {
        let prefill = prefill_metrics(7, 96, 32).unwrap().scheduled_requests;
        assert_eq!(prefill.num_prefill_requests, 7);
        assert_eq!(prefill.sum_prefill_tokens, 672);
        assert_eq!(prefill.sum_prefill_kv_tokens, 224);
        let decode = decode_metrics(7, 129, 2).unwrap().scheduled_requests;
        assert_eq!(decode.num_decode_requests, 21);
        assert_eq!(decode.sum_decode_kv_tokens, 2709);
    }

    #[test]
    fn oversized_scheduler_work_is_rejected_instead_of_wrapping() {
        assert!(prefill_metrics(u32::MAX as usize, 2, 0).is_err());
        assert!(prefill_metrics(u32::MAX as usize, 1, 2).is_err());
        assert!(decode_metrics(u32::MAX as usize, 1, 1).is_err());
        assert!(decode_metrics(u32::MAX as usize, 2, 0).is_err());
    }
}
