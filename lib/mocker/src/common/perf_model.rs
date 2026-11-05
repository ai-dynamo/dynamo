// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! `Polynomial` remains a legacy configuration marker. Its implementation is
//! owned by `aisimulate_core::engine`; this module only implements external NPZ and
//! AI Configurator providers.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_interp::InterpolateError;
use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use ndarray_interp::interp2d::{Bilinear, Interp2DBuilder};
use std::path::Path;
use std::sync::Arc;

/// Trait to abstract over interpolation for prefill timing.
pub trait PrefillInterpolator: Send + Sync {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError>;

    /// Predict using total new tokens and mean cached prefix length.
    /// Existing one-dimensional interpolators ignore the prefix.
    fn interp_with_prefix(&self, x: f64, _prefix: f64) -> Result<f64, InterpolateError> {
        self.interp(x)
    }
}

/// Trait to abstract over 2D interpolation for decode timing
pub trait DecodeInterpolator: Send + Sync {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError>;
}

/// Callback trait for direct AIS SDK calls.
/// Implementors call the Rust AIS core API.
pub trait AisCallback: Send + Sync {
    /// Predict prefill latency in ms.
    /// Parameters: (batch_size, effective_isl, prefix)
    fn predict_prefill(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> Result<f64>;

    /// Predict decode (generation) latency in ms.
    /// Parameters: (batch_size, isl, osl)
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> Result<f64>;
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

struct PrefillInterp2D {
    inner: ndarray_interp::interp2d::Interp2D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix2,
        Bilinear,
    >,
    min_new_tokens: f64,
    min_prefix: f64,
    max_prefix: f64,
}

impl PrefillInterpolator for PrefillInterp2D {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError> {
        self.interp_with_prefix(x, 0.0)
    }

    fn interp_with_prefix(&self, x: f64, prefix: f64) -> Result<f64, InterpolateError> {
        // Clamp to the measured prefix range and minimum sampled workload.
        // Larger new-token counts use the final segment's slope.
        self.inner.interp_scalar(
            x.max(self.min_new_tokens),
            prefix.clamp(self.min_prefix, self.max_prefix),
        )
    }
}

fn validate_prefill_axis(name: &str, axis: &Array1<f64>) -> Result<()> {
    anyhow::ensure!(axis.len() >= 2, "{name} must contain at least two points");
    anyhow::ensure!(
        axis.iter().all(|value| value.is_finite() && *value >= 0.0),
        "{name} must contain finite, nonnegative token counts"
    );
    anyhow::ensure!(
        axis.iter().zip(axis.iter().skip(1)).all(|(a, b)| a < b),
        "{name} must be strictly increasing"
    );
    Ok(())
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
    /// Select the built-in AISimulate polynomial timing model.
    #[default]
    Polynomial,
    /// Constant per-pass latencies owned by the AISimulate engine.
    Fixed { prefill_ms: f64, decode_ms: f64 },
    /// Interpolation-based model using profiler data
    /// Decode axes: (scheduled logical KV tokens, mean context length)
    Interpolated {
        prefill_interp: Arc<dyn PrefillInterpolator>,
        decode_interp: Arc<dyn DecodeInterpolator>,
    },
    /// AI Configurator SDK calls through the configured callback.
    /// Passes the reduced prefill inputs (batch_size, effective_isl, prefix).
    Ais { callback: Arc<dyn AisCallback> },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::Fixed {
                prefill_ms,
                decode_ms,
            } => PerfModel::Fixed {
                prefill_ms: *prefill_ms,
                decode_ms: *decode_ms,
            },
            PerfModel::Interpolated {
                prefill_interp,
                decode_interp,
            } => PerfModel::Interpolated {
                prefill_interp: Arc::clone(prefill_interp),
                decode_interp: Arc::clone(decode_interp),
            },
            PerfModel::Ais { callback } => PerfModel::Ais {
                callback: Arc::clone(callback),
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::Fixed {
                prefill_ms,
                decode_ms,
            } => write!(
                f,
                "PerfModel::Fixed {{ prefill_ms: {prefill_ms}, decode_ms: {decode_ms} }}"
            ),
            PerfModel::Interpolated { .. } => write!(f, "PerfModel::Interpolated {{ .. }}"),
            PerfModel::Ais { .. } => write!(f, "PerfModel::Ais"),
        }
    }
}

impl PerfModel {
    /// Load performance model from NPZ file
    ///
    /// Expected arrays in NPZ file:
    /// - prefill_isl: 1D array of input sequence lengths
    /// - prefill_ttft_ms: 1D array of time to first token in milliseconds
    /// - decode_active_kv_tokens: 1D array of scheduled logical KV token counts
    /// - decode_context_length: 1D array of context lengths
    /// - decode_itl: 2D array of inter-token latencies in milliseconds
    ///
    /// Optional arrays (all three must be present) override 1D prefill timing:
    /// - prefill2d_new: total new-token counts across the batch
    /// - prefill2d_prefix: mean cached prefix lengths per request
    /// - prefill2d_ms: latencies with shape (new-token counts, prefix lengths)
    ///
    /// All arrays use f64. Both 2D axes must be strictly increasing with at least
    /// two finite, nonnegative points; latencies must be finite and nonnegative.
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
        let prefill_1d = Interp1DBuilder::new(prefill_ttft_ms)
            .x(prefill_isl)
            .strategy(Linear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build prefill interpolator")?;
        let mut prefill_interp: Arc<dyn PrefillInterpolator> =
            Arc::new(PrefillInterp1D { inner: prefill_1d });

        let names = npz.names().context("Failed to list NPZ arrays")?;
        let prefill_2d_names = ["prefill2d_new", "prefill2d_prefix", "prefill2d_ms"];
        let has_prefill_2d = prefill_2d_names.map(|name| names.iter().any(|entry| entry == name));
        if has_prefill_2d.iter().any(|present| *present) {
            anyhow::ensure!(
                has_prefill_2d.iter().all(|present| *present),
                "prefill2d_new, prefill2d_prefix, and prefill2d_ms must be provided together"
            );
            let new_tokens: Array1<f64> = npz
                .by_name("prefill2d_new")
                .context("Failed to load prefill2d_new from NPZ")?;
            let prefixes: Array1<f64> = npz
                .by_name("prefill2d_prefix")
                .context("Failed to load prefill2d_prefix from NPZ")?;
            let times: Array2<f64> = npz
                .by_name("prefill2d_ms")
                .context("Failed to load prefill2d_ms from NPZ")?;
            validate_prefill_axis("prefill2d_new", &new_tokens)?;
            validate_prefill_axis("prefill2d_prefix", &prefixes)?;
            anyhow::ensure!(
                times.dim() == (new_tokens.len(), prefixes.len()),
                "prefill2d_ms shape must be ({}, {}), got {:?}",
                new_tokens.len(),
                prefixes.len(),
                times.dim()
            );
            anyhow::ensure!(
                times.iter().all(|time| time.is_finite() && *time >= 0.0),
                "prefill2d_ms must contain finite, nonnegative latencies"
            );

            let min_new_tokens = new_tokens[0];
            let min_prefix = prefixes[0];
            let max_prefix = prefixes[prefixes.len() - 1];
            tracing::info!(
                "Using prefix-aware prefill grid: {}x{}",
                new_tokens.len(),
                prefixes.len()
            );
            let inner = Interp2DBuilder::new(times)
                .x(new_tokens)
                .y(prefixes)
                .strategy(Bilinear::new().extrapolate(true))
                .build()
                .context("Failed to build 2D prefill interpolator")?;
            prefill_interp = Arc::new(PrefillInterp2D {
                inner,
                min_new_tokens,
                min_prefix,
                max_prefix,
            });
        }

        let decode_interp = Interp2DBuilder::new(decode_itl)
            .x(decode_active_kv_tokens)
            .y(decode_context_length)
            .strategy(Bilinear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build decode interpolator")?;

        Ok(PerfModel::Interpolated {
            prefill_interp,
            decode_interp: Arc::new(DecodeInterp2D {
                inner: decode_interp,
            }),
        })
    }

    /// Create an Ais perf model from a callback.
    pub fn from_ais_callback(callback: Arc<dyn AisCallback>) -> Self {
        PerfModel::Ais { callback }
    }

    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each external variant uses what it needs:
    /// - Interpolated uses total new tokens across the batch
    ///   (`batch_size * (isl - prefix)`) and, when a 2D table is present,
    ///   the mean cached prefix length (`prefix`).
    /// - Ais: passes (batch_size, isl - prefix, prefix) to the AIS SDK
    pub fn predict_prefill_time(
        &self,
        batch_size: usize,
        isl: usize,
        prefix: usize,
    ) -> Result<f64> {
        if matches!(self, Self::Polynomial | Self::Fixed { .. }) {
            anyhow::bail!("built-in timing is implemented by the AISimulate engine");
        }
        let new_tokens_per_req = isl.saturating_sub(prefix);
        if batch_size == 0 || new_tokens_per_req == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => unreachable!("polynomial handled above"),
            PerfModel::Fixed { .. } => unreachable!("fixed timing handled above"),
            PerfModel::Interpolated { prefill_interp, .. } => {
                let tokens = (batch_size * new_tokens_per_req) as f64;
                prefill_interp
                    .interp_with_prefix(tokens, prefix as f64)
                    .unwrap_or(0.0)
            }
            PerfModel::Ais { callback } => callback
                .predict_prefill(batch_size, new_tokens_per_req, prefix)
                .context("AIS prefill prediction failed")?,
        };
        Ok(time.max(0.0))
    }

    /// Predict decode time in milliseconds.
    ///
    /// `active_kv_tokens` is the sum of logical context lengths in the scheduled
    /// batch, not the number of distinct physically resident tokens.
    ///
    /// Callers always pass all parameters; each variant uses what it needs:
    /// - Interpolated: uses (active_kv_tokens, context_length)
    /// - Ais: uses (batch_size, context_length)
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        if matches!(self, Self::Polynomial | Self::Fixed { .. }) {
            anyhow::bail!("built-in timing is implemented by the AISimulate engine");
        }
        if batch_size == 0 {
            return Ok(0.0);
        }
        let time = match self {
            PerfModel::Polynomial => unreachable!("polynomial handled above"),
            PerfModel::Fixed { .. } => unreachable!("fixed timing handled above"),
            PerfModel::Interpolated { decode_interp, .. } => decode_interp
                .interp(active_kv_tokens as f64, context_length as f64)
                .unwrap_or(0.0),
            PerfModel::Ais { callback } => callback
                .predict_decode(batch_size, context_length, 2)
                .context("AIS decode prediction failed")?,
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={context_length}, time={result:.2}ms"
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{AisCallback, PerfModel};
    use ndarray::{Array1, Array2, array};
    use ndarray_npy::NpzWriter;
    use std::fs::File;
    use std::sync::Arc;

    fn load_npz(add_arrays: impl FnOnce(&mut NpzWriter<File>)) -> anyhow::Result<PerfModel> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut npz = NpzWriter::new(file.reopen().unwrap());
        npz.add_array("prefill_isl", &array![100.0, 300.0]).unwrap();
        npz.add_array("prefill_ttft_ms", &array![10.0, 30.0])
            .unwrap();
        npz.add_array("decode_active_kv_tokens", &array![100.0, 300.0])
            .unwrap();
        npz.add_array("decode_context_length", &array![10.0, 30.0])
            .unwrap();
        npz.add_array("decode_itl", &array![[1.0, 3.0], [3.0, 5.0]])
            .unwrap();
        add_arrays(&mut npz);
        npz.finish().unwrap();
        PerfModel::from_npz(file.path())
    }

    fn load_prefill_2d(
        new_tokens: &Array1<f64>,
        prefixes: &Array1<f64>,
        times: &Array2<f64>,
    ) -> anyhow::Result<PerfModel> {
        load_npz(|npz| {
            npz.add_array("prefill2d_new", new_tokens).unwrap();
            npz.add_array("prefill2d_prefix", prefixes).unwrap();
            npz.add_array("prefill2d_ms", times).unwrap();
        })
    }

    #[test]
    fn npz_prefill_depends_on_cached_prefix() {
        let model = load_prefill_2d(
            &array![4096.0, 8192.0],
            &array![0.0, 16384.0, 65536.0],
            &array![[150.0, 230.0, 480.0], [303.0, 469.0, 968.0]],
        )
        .unwrap();

        for (prefix, expected) in [(0, 303.0), (16384, 469.0), (65536, 968.0)] {
            assert_eq!(
                model
                    .predict_prefill_time(1, prefix + 8192, prefix)
                    .unwrap(),
                expected,
                "prefix={prefix}"
            );
        }
    }

    #[test]
    fn legacy_npz_uses_total_new_tokens_and_ignores_prefix() {
        let model = load_npz(|_| {}).unwrap();
        assert_eq!(model.predict_prefill_time(1, 200, 0).unwrap(), 20.0);
        assert_eq!(model.predict_prefill_time(1, 1200, 1000).unwrap(), 20.0);
        assert_eq!(model.predict_prefill_time(2, 1100, 1000).unwrap(), 20.0);
        assert_eq!(model.predict_prefill_time(1, 400, 0).unwrap(), 40.0);
        assert_eq!(model.predict_decode_time(1, 200, 20, 200).unwrap(), 3.0);
    }

    #[test]
    fn npz_prefill_interpolates_both_axes_and_uses_batch_total() {
        let model = load_prefill_2d(
            &array![100.0, 300.0],
            &array![0.0, 200.0],
            &array![[10.0, 30.0], [30.0, 70.0]],
        )
        .unwrap();

        assert_eq!(model.predict_prefill_time(1, 300, 100).unwrap(), 35.0);
        assert_eq!(model.predict_prefill_time(2, 200, 100).unwrap(), 35.0);
        assert_eq!(model.predict_prefill_time(1, 500, 200).unwrap(), 70.0);
        // The optional prefill table must not change decode predictions.
        assert_eq!(model.predict_decode_time(1, 200, 20, 200).unwrap(), 3.0);
    }

    #[test]
    fn npz_prefill_clamps_prefix_and_small_workloads_but_extrapolates_larger_workloads() {
        let model = load_prefill_2d(
            &array![100.0, 300.0],
            &array![50.0, 200.0],
            &array![[10.0, 30.0], [30.0, 70.0]],
        )
        .unwrap();

        assert_eq!(model.predict_prefill_time(1, 100, 0).unwrap(), 10.0);
        assert_eq!(model.predict_prefill_time(1, 500, 400).unwrap(), 30.0);
        assert_eq!(model.predict_prefill_time(1, 175, 125).unwrap(), 20.0);
        assert_eq!(model.predict_prefill_time(1, 525, 125).unwrap(), 65.0);
        assert_eq!(model.predict_prefill_time(0, 300, 100).unwrap(), 0.0);
        assert_eq!(model.predict_prefill_time(1, 100, 100).unwrap(), 0.0);
        assert_eq!(model.predict_prefill_time(1, 50, 100).unwrap(), 0.0);
    }

    #[test]
    fn npz_prefill_tables_belong_to_each_model_and_survive_cloning() {
        let first = load_prefill_2d(
            &array![100.0, 300.0],
            &array![0.0, 200.0],
            &array![[10.0, 30.0], [30.0, 70.0]],
        )
        .unwrap();
        let cloned = first.clone();
        drop(first);
        let second = load_prefill_2d(
            &array![100.0, 300.0],
            &array![0.0, 200.0],
            &array![[20.0, 60.0], [60.0, 140.0]],
        )
        .unwrap();

        assert_eq!(cloned.predict_prefill_time(1, 300, 100).unwrap(), 35.0);
        assert_eq!(second.predict_prefill_time(1, 300, 100).unwrap(), 70.0);
    }

    #[test]
    fn npz_prefill_rejects_partial_2d_tables() {
        for mask in 1..7 {
            let error = load_npz(|npz| {
                if mask & 1 != 0 {
                    npz.add_array("prefill2d_new", &array![100.0, 300.0])
                        .unwrap();
                }
                if mask & 2 != 0 {
                    npz.add_array("prefill2d_prefix", &array![0.0, 200.0])
                        .unwrap();
                }
                if mask & 4 != 0 {
                    npz.add_array("prefill2d_ms", &array![[10.0, 30.0], [30.0, 70.0]])
                        .unwrap();
                }
            })
            .unwrap_err();
            assert!(error.to_string().contains("must be provided together"));
        }
    }

    #[test]
    fn npz_prefill_rejects_invalid_axes() {
        for invalid in [
            array![],
            array![0.0],
            array![100.0, 100.0],
            array![300.0, 100.0],
            array![-1.0, 100.0],
            array![0.0, f64::NAN],
            array![0.0, f64::INFINITY],
        ] {
            let times = array![[10.0, 30.0], [30.0, 70.0]];
            let new_error = load_prefill_2d(&invalid, &array![0.0, 200.0], &times).unwrap_err();
            assert!(new_error.to_string().contains("prefill2d_new"));
            let prefix_error =
                load_prefill_2d(&array![100.0, 300.0], &invalid, &times).unwrap_err();
            assert!(prefix_error.to_string().contains("prefill2d_prefix"));
        }
    }

    #[test]
    fn npz_prefill_rejects_invalid_latencies_and_shape() {
        for times in [
            array![[10.0, 30.0]],
            array![[10.0], [30.0]],
            array![[10.0, -1.0], [30.0, 70.0]],
            array![[10.0, f64::NAN], [30.0, 70.0]],
            array![[10.0, f64::INFINITY], [30.0, 70.0]],
        ] {
            let error =
                load_prefill_2d(&array![100.0, 300.0], &array![0.0, 200.0], &times).unwrap_err();
            assert!(error.to_string().contains("prefill2d_ms"));
        }
    }

    #[test]
    fn npz_prefill_reports_wrong_array_types() {
        let error = load_npz(|npz| {
            npz.add_array("prefill2d_new", &array![100_i64, 300])
                .unwrap();
            npz.add_array("prefill2d_prefix", &array![0.0, 200.0])
                .unwrap();
            npz.add_array("prefill2d_ms", &array![[10.0, 30.0], [30.0, 70.0]])
                .unwrap();
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "Failed to load prefill2d_new from NPZ");
    }

    struct EchoBatchCallback;

    impl AisCallback for EchoBatchCallback {
        fn predict_prefill(
            &self,
            batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }

        fn predict_decode(
            &self,
            batch_size: usize,
            _isl: usize,
            _osl: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }
    }

    struct FailingCallback;

    impl AisCallback for FailingCallback {
        fn predict_prefill(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing AIS prefill point")
        }

        fn predict_decode(
            &self,
            _batch_size: usize,
            _isl: usize,
            _osl: usize,
        ) -> anyhow::Result<f64> {
            anyhow::bail!("missing AIS decode point")
        }
    }

    #[test]
    fn polynomial_is_only_a_builtin_engine_marker() {
        let model = PerfModel::default();
        assert!(matches!(model, PerfModel::Polynomial));
        assert!(model.predict_prefill_time(1, 128, 0).is_err());
        assert!(model.predict_decode_time(1, 128, 128, 1024).is_err());
    }

    #[test]
    fn ais_forwards_scheduler_local_batch() {
        let model = PerfModel::from_ais_callback(Arc::new(EchoBatchCallback));

        assert_eq!(model.predict_prefill_time(7, 128, 0).unwrap(), 7.0);
        assert_eq!(model.predict_decode_time(9, 0, 128, 0).unwrap(), 9.0);
    }

    #[test]
    fn ais_prefill_prediction_errors_propagate() {
        let error = PerfModel::from_ais_callback(Arc::new(FailingCallback))
            .predict_prefill_time(2, 128, 32)
            .unwrap_err();

        assert_eq!(error.to_string(), "AIS prefill prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing AIS prefill point");
    }

    #[test]
    fn ais_decode_prediction_errors_propagate() {
        let error = PerfModel::from_ais_callback(Arc::new(FailingCallback))
            .predict_decode_time(2, 64, 128, 1024)
            .unwrap_err();

        assert_eq!(error.to_string(), "AIS decode prediction failed");
        assert_eq!(error.root_cause().to_string(), "missing AIS decode point");
    }
}
