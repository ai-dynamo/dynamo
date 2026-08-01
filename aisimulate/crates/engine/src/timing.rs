// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral forward-pass timing models.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::common::perf_model::{polynomial_decode_time, polynomial_prefill_time};

/// Serializable timing-provider selection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TimingModelConfig {
    /// Current-main polynomial fallback.
    #[default]
    Polynomial,
    /// Deterministic constant latency, primarily useful for parity fixtures.
    Fixed { prefill_ms: f64, decode_ms: f64 },
    /// Process-local provider loaded by a Runner or binding.
    External {
        provider: String,
        #[serde(default)]
        config: Value,
    },
}

/// Runtime latency model injected at the engine boundary.
///
/// Implementations may call AIC, interpolate profiler data, or use another
/// provider without adding that dependency to `aisimulate-engine`.
pub trait TimingModel: Send + Sync {
    /// Predict one prefill batch's latency in milliseconds.
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64>;

    /// Predict one decode batch's latency in milliseconds.
    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64>;
}

struct PolynomialTimingModel;

impl TimingModel for PolynomialTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        Ok(polynomial_prefill_time(
            batch_size,
            mean_isl.saturating_sub(mean_prefix),
        ))
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        _mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        if batch_size == 0 {
            return Ok(0.0);
        }
        Ok(polynomial_decode_time(active_kv_tokens, total_kv_tokens))
    }
}

struct FixedTimingModel {
    prefill_ms: f64,
    decode_ms: f64,
}

impl TimingModel for FixedTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> Result<f64> {
        Ok(if batch_size == 0 {
            0.0
        } else {
            self.prefill_ms
        })
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        Ok(if batch_size == 0 { 0.0 } else { self.decode_ms })
    }
}

pub(crate) fn built_in_timing_model(config: &TimingModelConfig) -> Result<Arc<dyn TimingModel>> {
    match config {
        TimingModelConfig::Polynomial => Ok(Arc::new(PolynomialTimingModel)),
        TimingModelConfig::Fixed {
            prefill_ms,
            decode_ms,
        } => Ok(Arc::new(FixedTimingModel {
            prefill_ms: *prefill_ms,
            decode_ms: *decode_ms,
        })),
        TimingModelConfig::External { provider, .. } => {
            bail!("timing provider '{provider}' requires EngineFactory::with_timing_model")
        }
    }
}

pub(crate) fn modeled_duration_ms(raw_ms: f64, speedup_ratio: f64) -> Result<f64> {
    ensure!(
        raw_ms.is_finite() && raw_ms >= 0.0,
        "timing provider returned invalid duration {raw_ms}ms"
    );
    ensure!(
        speedup_ratio.is_finite() && speedup_ratio >= 0.0,
        "modeled speedup ratio must be finite and non-negative, got {speedup_ratio}"
    );
    let unscaled = Duration::try_from_secs_f64(raw_ms / 1_000.0)
        .map_err(|error| anyhow::anyhow!("timing duration {raw_ms}ms is out of range: {error}"))?;
    let modeled = if speedup_ratio > 0.0 && unscaled > Duration::ZERO {
        Duration::try_from_secs_f64(unscaled.as_secs_f64() / speedup_ratio).map_err(|error| {
            anyhow::anyhow!(
                "scaled timing duration is out of range for speedup {speedup_ratio}: {error}"
            )
        })?
    } else {
        unscaled
    };
    Ok(modeled.as_secs_f64() * 1_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modeled_duration_applies_speedup_and_zero_means_unscaled() {
        assert_eq!(modeled_duration_ms(12.0, 3.0).unwrap(), 4.0);
        assert_eq!(modeled_duration_ms(12.0, 0.0).unwrap(), 12.0);
        assert_eq!(modeled_duration_ms(0.0, 3.0).unwrap(), 0.0);
    }

    #[test]
    fn modeled_duration_rejects_invalid_provider_values() {
        for raw_ms in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(modeled_duration_ms(raw_ms, 1.0).is_err());
        }
        for speedup in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(modeled_duration_ms(1.0, speedup).is_err());
        }
    }

    #[test]
    fn fixed_model_returns_zero_for_empty_batches() {
        let model = built_in_timing_model(&TimingModelConfig::Fixed {
            prefill_ms: 7.0,
            decode_ms: 3.0,
        })
        .unwrap();
        assert_eq!(model.predict_prefill_ms(0, 128, 0).unwrap(), 0.0);
        assert_eq!(model.predict_decode_ms(0, 128, 64, 1024).unwrap(), 0.0);
        assert_eq!(model.predict_prefill_ms(2, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_ms(2, 128, 64, 1024).unwrap(), 3.0);
    }

    #[test]
    fn external_provider_requires_runner_resolution() {
        let error = built_in_timing_model(&TimingModelConfig::External {
            provider: "example".to_string(),
            config: Value::Null,
        })
        .err()
        .expect("external descriptor cannot be materialized in the neutral crate");
        assert!(
            error
                .to_string()
                .contains("EngineFactory::with_timing_model")
        );
    }
}
