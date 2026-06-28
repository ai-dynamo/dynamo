// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! This module provides two timing models:
//! 1. Polynomial: Hardcoded polynomial formulas (default, backward compatible)
//! 2. Interpolated: Grid-based interpolation from profiler data (loaded from NPZ files)

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_interp::InterpolateError;
use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use ndarray_interp::interp2d::{Bilinear, Interp2DBuilder};
use std::path::Path;
use std::sync::Arc;

/// Trait to abstract over 1D interpolation for prefill timing
pub trait PrefillInterpolator: Send + Sync {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError>;
}

/// Trait to abstract over 2D interpolation for decode timing
pub trait DecodeInterpolator: Send + Sync {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError>;
}

/// Callback trait for direct AIC SDK calls.
/// Implementors call the Python AIC SDK via PyO3 GIL.
pub trait AicCallback: Send + Sync {
    /// Predict prefill latency in ms.
    /// Parameters: (batch_size, effective_isl, prefix)
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64;

    /// Predict decode (generation) latency in ms.
    /// Parameters: (batch_size, isl, osl)
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64;
}

/// Wrapper to implement PrefillInterpolator for the concrete Interp1D type
struct PrefillInterp1D {
    inner: ndarray_interp::interp1d::Interp1D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix1,
        Linear,
    >,
}

impl PrefillInterpolator for PrefillInterp1D {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError> {
        self.inner.interp_scalar(x)
    }
}

/// Wrapper to implement DecodeInterpolator for the concrete Interp2D type
struct DecodeInterp2D {
    inner: ndarray_interp::interp2d::Interp2D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix2,
        Bilinear,
    >,
}

impl DecodeInterpolator for DecodeInterp2D {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError> {
        self.inner.interp_scalar(x, y)
    }
}

/// Performance model for predicting prefill and decode timing
#[derive(Default)]
pub enum PerfModel {
    /// Default polynomial-based model using hardcoded formulas
    #[default]
    Polynomial,
    /// Interpolation-based model using profiler data
    /// Decode axes: (active_kv_tokens, context_length)
    Interpolated {
        prefill_interp: Arc<dyn PrefillInterpolator>,
        decode_interp: Arc<dyn DecodeInterpolator>,
    },
    /// AI Configurator SDK calls via Python callback.
    /// Passes the reduced prefill inputs (batch_size, effective_isl, prefix).
    ///
    /// `attention_dp_size` is the number of attention data-parallel ranks this
    /// engine aggregates. The offline-replay aggregate engine holds the GLOBAL
    /// batch across all ranks, but the AIC SDK expects a PER-RANK batch
    /// (`global_bs = bs * attention_dp_size`), so the scheduled batch is divided
    /// by this value before each perf query. It is 1 for the live path (which
    /// replicates one scheduler per rank, so each already sees a per-rank batch)
    /// and for non-DP configs — making the division a no-op there.
    Aiconfigurator {
        callback: Arc<dyn AicCallback>,
        attention_dp_size: usize,
    },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::Interpolated {
                prefill_interp,
                decode_interp,
            } => PerfModel::Interpolated {
                prefill_interp: Arc::clone(prefill_interp),
                decode_interp: Arc::clone(decode_interp),
            },
            PerfModel::Aiconfigurator {
                callback,
                attention_dp_size,
            } => PerfModel::Aiconfigurator {
                callback: Arc::clone(callback),
                attention_dp_size: *attention_dp_size,
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::Interpolated { .. } => write!(f, "PerfModel::Interpolated {{ .. }}"),
            PerfModel::Aiconfigurator { .. } => write!(f, "PerfModel::Aiconfigurator"),
        }
    }
}

impl PerfModel {
    /// Load performance model from NPZ file
    ///
    /// Expected arrays in NPZ file:
    /// - prefill_isl: 1D array of input sequence lengths
    /// - prefill_ttft_ms: 1D array of time to first token in milliseconds
    /// - decode_active_kv_tokens: 1D array of active KV token counts
    /// - decode_context_length: 1D array of context lengths
    /// - decode_itl: 2D array of inter-token latencies in milliseconds
    pub fn from_npz(path: &Path) -> Result<Self> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        tracing::info!("Loading performance model from NPZ file: {:?}", path);

        let file =
            File::open(path).with_context(|| format!("Failed to open NPZ file: {:?}", path))?;

        let mut npz = NpzReader::new(file)
            .with_context(|| format!("Failed to create NPZ reader for: {:?}", path))?;

        // Load prefill arrays
        let prefill_isl: Array1<f64> = npz
            .by_name("prefill_isl")
            .with_context(|| "Failed to load prefill_isl from NPZ")?;
        let prefill_ttft_ms: Array1<f64> = npz
            .by_name("prefill_ttft_ms")
            .with_context(|| "Failed to load prefill_ttft_ms from NPZ")?;

        // Load decode arrays
        let decode_active_kv_tokens: Array1<f64> = npz
            .by_name("decode_active_kv_tokens")
            .with_context(|| "Failed to load decode_active_kv_tokens from NPZ")?;
        let decode_context_length: Array1<f64> = npz
            .by_name("decode_context_length")
            .with_context(|| "Failed to load decode_context_length from NPZ")?;
        let decode_itl: Array2<f64> = npz
            .by_name("decode_itl")
            .with_context(|| "Failed to load decode_itl from NPZ")?;

        // Validate dimensions
        if prefill_isl.len() != prefill_ttft_ms.len() {
            anyhow::bail!(
                "Prefill array length mismatch: isl={}, ttft={}",
                prefill_isl.len(),
                prefill_ttft_ms.len()
            );
        }

        if decode_itl.nrows() != decode_active_kv_tokens.len()
            || decode_itl.ncols() != decode_context_length.len()
        {
            anyhow::bail!(
                "Decode array dimension mismatch: itl shape=({}, {}), active_kv={}, context={}",
                decode_itl.nrows(),
                decode_itl.ncols(),
                decode_active_kv_tokens.len(),
                decode_context_length.len()
            );
        }

        tracing::info!(
            "Loaded performance model: prefill_points={}, decode_grid={}x{}",
            prefill_isl.len(),
            decode_itl.nrows(),
            decode_itl.ncols()
        );

        // Build interpolators once during loading
        let prefill_interp = Interp1DBuilder::new(prefill_ttft_ms)
            .x(prefill_isl)
            .strategy(Linear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build prefill interpolator")?;

        let decode_interp = Interp2DBuilder::new(decode_itl)
            .x(decode_active_kv_tokens)
            .y(decode_context_length)
            .strategy(Bilinear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build decode interpolator")?;

        Ok(PerfModel::Interpolated {
            prefill_interp: Arc::new(PrefillInterp1D {
                inner: prefill_interp,
            }),
            decode_interp: Arc::new(DecodeInterp2D {
                inner: decode_interp,
            }),
        })
    }

    /// Create an Aiconfigurator perf model from a callback.
    ///
    /// `attention_dp_size` defaults to 1, so the per-rank batch division is a
    /// no-op. Use [`PerfModel::from_aic_callback_with_attention_dp`] from the
    /// offline-replay aggregate path, which holds the global multi-rank batch.
    pub fn from_aic_callback(callback: Arc<dyn AicCallback>) -> Self {
        PerfModel::Aiconfigurator {
            callback,
            attention_dp_size: 1,
        }
    }

    /// Like [`PerfModel::from_aic_callback`], but records the attention-DP degree
    /// so the aggregated offline-replay engine queries the AIC SDK with the
    /// per-rank batch (`scheduled_batch / attention_dp_size`) it expects. The
    /// live path must NOT use this (it already replicates one scheduler per rank).
    pub fn from_aic_callback_with_attention_dp(
        callback: Arc<dyn AicCallback>,
        attention_dp_size: usize,
    ) -> Self {
        PerfModel::Aiconfigurator {
            callback,
            attention_dp_size: attention_dp_size.max(1),
        }
    }

    /// Global batch -> per-rank batch for the AIC SDK; see the
    /// `Aiconfigurator { attention_dp_size }` doc. `div_ceil` bounds the step by
    /// the busiest rank, and dp == 1 (live / non-DP) is a no-op.
    fn aic_per_rank_batch(batch_size: usize, attention_dp_size: usize) -> usize {
        batch_size.div_ceil(attention_dp_size.max(1))
    }

    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial/Interpolated: uses total new tokens across the batch
    ///   (`batch_size * (isl - prefix)`), modeling GPU processing total tokens in parallel
    /// - Aiconfigurator: passes (batch_size, isl - prefix, prefix) to the AIC SDK
    pub fn predict_prefill_time(&self, batch_size: usize, isl: usize, prefix: usize) -> f64 {
        let new_tokens_per_req = isl.saturating_sub(prefix);
        if batch_size == 0 || new_tokens_per_req == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                // Total tokens across the batch — GPU processes them in parallel
                let tokens = (batch_size * new_tokens_per_req) as f64;
                4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
            }
            PerfModel::Interpolated { prefill_interp, .. } => {
                let tokens = (batch_size * new_tokens_per_req) as f64;
                prefill_interp.interp(tokens).unwrap_or(0.0)
            }
            PerfModel::Aiconfigurator {
                callback,
                attention_dp_size,
            } => callback.predict_prefill(
                Self::aic_per_rank_batch(batch_size, *attention_dp_size),
                new_tokens_per_req,
                prefix,
            ),
        };
        time.max(0.0)
    }

    /// Predict decode time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial: uses (active_kv_tokens, total_kv_tokens) as utilization
    /// - Interpolated: uses (active_kv_tokens, context_length)
    /// - Aiconfigurator: uses (batch_size, context_length)
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> f64 {
        if batch_size == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                let active_perc = if total_kv_tokens > 0 {
                    active_kv_tokens as f64 / total_kv_tokens as f64
                } else {
                    tracing::warn!("Total KV tokens is 0, using 1.0 as capacity");
                    1.0
                };
                -25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74
            }
            PerfModel::Interpolated { decode_interp, .. } => decode_interp
                .interp(active_kv_tokens as f64, context_length as f64)
                .unwrap_or(0.0),
            PerfModel::Aiconfigurator {
                callback,
                attention_dp_size,
            } => callback.predict_decode(
                Self::aic_per_rank_batch(batch_size, *attention_dp_size),
                context_length,
                2,
            ),
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={result:.2}ms"
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{AicCallback, PerfModel};
    use std::sync::Arc;

    #[test]
    fn fully_cached_prompt_skips_prefill() {
        assert_eq!(PerfModel::default().predict_prefill_time(1, 128, 128), 0.0);
    }

    /// Echoes back the batch_size it is called with, so tests can assert exactly
    /// what batch reached the AIC SDK after any per-rank division.
    struct EchoBatchCallback;
    impl AicCallback for EchoBatchCallback {
        fn predict_prefill(&self, batch_size: usize, _effective_isl: usize, _prefix: usize) -> f64 {
            batch_size as f64
        }
        fn predict_decode(&self, batch_size: usize, _isl: usize, _osl: usize) -> f64 {
            batch_size as f64
        }
    }

    // The AIC SDK expects a per-rank batch (global_bs = bs * attention_dp_size).
    // Offline replay holds the global batch in one engine, so the perf model must
    // divide by attention_dp_size before the AIC call. attention_dp_size=1 (live /
    // non-DP / `from_aic_callback`) must be a strict no-op.

    #[test]
    fn aic_decode_attention_dp_1_is_noop() {
        let m = PerfModel::from_aic_callback(Arc::new(EchoBatchCallback));
        // callback sees the full global batch unchanged
        assert_eq!(m.predict_decode_time(128, 0, 1024, 0), 128.0);
        assert_eq!(m.predict_decode_time(1, 0, 1024, 0), 1.0);
    }

    #[test]
    fn aic_decode_divides_batch_by_attention_dp() {
        let m = PerfModel::from_aic_callback_with_attention_dp(Arc::new(EchoBatchCallback), 8);
        // 128 sequences across 8 DP ranks -> 16 per rank
        assert_eq!(m.predict_decode_time(128, 0, 1024, 0), 16.0);
        // div_ceil: 130/8 = 17 (the busiest rank bounds the step)
        assert_eq!(m.predict_decode_time(130, 0, 1024, 0), 17.0);
        // fewer sequences than ranks -> at least 1 per active rank
        assert_eq!(m.predict_decode_time(4, 0, 1024, 0), 1.0);
    }

    #[test]
    fn aic_prefill_attention_dp_1_is_noop() {
        let m = PerfModel::from_aic_callback(Arc::new(EchoBatchCallback));
        assert_eq!(m.predict_prefill_time(8, 1024, 0), 8.0);
    }

    #[test]
    fn aic_prefill_divides_batch_by_attention_dp() {
        let m = PerfModel::from_aic_callback_with_attention_dp(Arc::new(EchoBatchCallback), 8);
        assert_eq!(m.predict_prefill_time(8, 1024, 0), 1.0);
        assert_eq!(m.predict_prefill_time(128, 1024, 0), 16.0);
    }
}
