// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! Scheduler algorithms keep their historical polynomial fallback while
//! provider-backed timing enters through the runtime-neutral engine contract.

use anyhow::{Context, Result};
use std::sync::Arc;

/// Performance model for predicting prefill and decode timing
#[derive(Default)]
pub enum PerfModel {
    /// Default polynomial-based model using hardcoded formulas
    #[default]
    Polynomial,
    /// Runtime-resolved provider supplied through the neutral engine boundary.
    External { timing: Arc<dyn crate::TimingModel> },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::External { timing } => PerfModel::External {
                timing: Arc::clone(timing),
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::External { .. } => write!(f, "PerfModel::External"),
        }
    }
}

impl PerfModel {
    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// The polynomial fallback uses total new tokens across the batch. An
    /// injected provider receives the original batch-local inputs.
    pub fn predict_prefill_time(
        &self,
        batch_size: usize,
        isl: usize,
        prefix: usize,
    ) -> Result<f64> {
        let new_tokens_per_req = isl.saturating_sub(prefix);
        if batch_size == 0 || new_tokens_per_req == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => polynomial_prefill_time(batch_size, new_tokens_per_req),
            PerfModel::External { timing } => timing
                .predict_prefill_ms(batch_size, prefix + new_tokens_per_req, prefix)
                .context("external prefill prediction failed")?,
        };
        Ok(time.max(0.0))
    }

    /// Predict decode time in milliseconds.
    ///
    /// `active_kv_tokens` is the sum of logical context lengths in the scheduled
    /// batch, not the number of distinct physically resident tokens.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial uses logical active KV tokens relative to total capacity,
    ///   clamped to full utilization.
    /// - An injected provider receives the full scheduler-local context.
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        if batch_size == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => polynomial_decode_time(active_kv_tokens, total_kv_tokens),
            PerfModel::External { timing } => timing
                .predict_decode_ms(
                    batch_size,
                    active_kv_tokens,
                    context_length,
                    total_kv_tokens,
                )
                .context("external decode prediction failed")?,
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={result:.2}ms"
        );
        Ok(result)
    }
}

pub(crate) fn polynomial_prefill_time(batch_size: usize, new_tokens_per_request: usize) -> f64 {
    if batch_size == 0 || new_tokens_per_request == 0 {
        return 0.0;
    }
    // Total tokens across the batch — GPU processes them in parallel.
    let tokens = (batch_size * new_tokens_per_request) as f64;
    4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
}

pub(crate) fn polynomial_decode_time(active_kv_tokens: usize, total_kv_tokens: usize) -> f64 {
    let active_perc = if total_kv_tokens > 0 {
        (active_kv_tokens as f64 / total_kv_tokens as f64).min(1.0)
    } else {
        tracing::warn!("Total KV tokens is 0, using 1.0 as capacity");
        1.0
    };
    (-25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74).max(1.0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{PerfModel, polynomial_decode_time};
    use crate::TimingModel;

    struct EchoBatchTiming;

    impl TimingModel for EchoBatchTiming {
        fn predict_prefill_ms(
            &self,
            batch_size: usize,
            _mean_isl: usize,
            _mean_prefix: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }

        fn predict_decode_ms(
            &self,
            batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }
    }

    struct FailingTiming;

    impl TimingModel for FailingTiming {
        fn predict_prefill_ms(
            &self,
            _batch_size: usize,
            _mean_isl: usize,
            _mean_prefix: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing prefill point")
        }

        fn predict_decode_ms(
            &self,
            _batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing decode point")
        }
    }

    #[test]
    fn fully_cached_prompt_skips_prefill() {
        assert_eq!(
            PerfModel::default()
                .predict_prefill_time(1, 128, 128)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn external_provider_receives_scheduler_local_batch() {
        let model = PerfModel::External {
            timing: Arc::new(EchoBatchTiming),
        };
        assert_eq!(model.predict_prefill_time(7, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_time(9, 0, 128, 0).unwrap(), 9.0);
    }

    #[test]
    fn external_prefill_errors_propagate_with_context() {
        let error = PerfModel::External {
            timing: Arc::new(FailingTiming),
        }
        .predict_prefill_time(2, 128, 32)
        .unwrap_err();
        assert_eq!(error.to_string(), "external prefill prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing prefill point");
    }

    #[test]
    fn external_decode_errors_propagate_with_context() {
        let error = PerfModel::External {
            timing: Arc::new(FailingTiming),
        }
        .predict_decode_time(2, 64, 128, 1024)
        .unwrap_err();
        assert_eq!(error.to_string(), "external decode prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing decode point");
    }

    #[test]
    fn polynomial_decode_utilization_is_clamped_to_capacity() {
        assert_eq!(
            polynomial_decode_time(2_048, 1_024),
            polynomial_decode_time(1_024, 1_024)
        );
    }
}
