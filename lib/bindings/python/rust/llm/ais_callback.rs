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
use std::sync::Arc;
#[cfg(feature = "ais-forward-pass")]
use std::time::Duration;

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
