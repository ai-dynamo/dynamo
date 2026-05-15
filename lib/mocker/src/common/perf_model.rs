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
    Aiconfigurator { callback: Arc<dyn AicCallback> },
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
            PerfModel::Aiconfigurator { callback } => PerfModel::Aiconfigurator {
                callback: Arc::clone(callback),
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
    pub fn from_aic_callback(callback: Arc<dyn AicCallback>) -> Self {
        PerfModel::Aiconfigurator { callback }
    }

    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Polynomial/Interpolated: uses total new tokens across the batch
    ///   (`batch_size * (isl - prefix)`), modeling GPU processing total tokens in parallel
    /// - Aiconfigurator: passes (batch_size, isl - prefix, prefix) to the AIC SDK
    pub fn predict_prefill_time(&self, batch_size: usize, isl: usize, prefix: usize) -> f64 {
        let new_tokens_per_req = isl.saturating_sub(prefix);
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
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_prefill(batch_size, new_tokens_per_req, prefix)
            }
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
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_decode(batch_size, context_length, 2)
            }
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={result:.2}ms"
        );
        result
    }
}
