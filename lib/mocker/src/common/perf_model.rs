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
use ndarray_npy::ReadNpyExt;
use std::io::{Read, Seek};
use std::path::Path;
use std::sync::Arc;
use zip::ZipArchive;
use zip::result::ZipError;

/// Trait to abstract over 1D interpolation for prefill timing
pub trait PrefillInterpolator: Send + Sync {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError>;
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

/// Reads arrays from a NumPy `.npz` archive. `ndarray_npy::NpzReader` needs zip 2.x, so this
/// opens the archive with `zip` and parses each entry with `ReadNpyExt`.
struct NpzReader<R>(ZipArchive<R>);

impl<R: Read + Seek> NpzReader<R> {
    fn new(reader: R) -> Result<Self> {
        Ok(Self(ZipArchive::new(reader)?))
    }

    /// Reads entry `name`, else `{name}.npy`, the name that `numpy.savez` writes.
    fn by_name<A: ReadNpyExt>(&mut self, name: &str) -> Result<A> {
        match self.0.by_name(name) {
            Ok(file) => return Ok(A::read_npy(file)?),
            Err(ZipError::FileNotFound) => {}
            Err(err) => return Err(err.into()),
        }
        Ok(A::read_npy(self.0.by_name(&format!("{name}.npy"))?)?)
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
    pub fn from_npz(path: &Path) -> Result<Self> {
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

    /// Create an Ais perf model from a callback.
    pub fn from_ais_callback(callback: Arc<dyn AisCallback>) -> Self {
        PerfModel::Ais { callback }
    }

    /// Predict prefill time in milliseconds.
    ///
    /// Callers always pass all parameters; each external variant uses what it needs:
    /// - Interpolated uses total new tokens across the batch
    ///   (`batch_size * (isl - prefix)`).
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
                prefill_interp.interp(tokens).unwrap_or(0.0)
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
    use super::{AisCallback, NpzReader, PerfModel};
    use ndarray::{Array1, array};
    use ndarray_npy::WriteNpyExt;
    use rstest::rstest;
    use std::io::{Cursor, Seek, Write};
    use std::sync::Arc;
    use zip::CompressionMethod;
    use zip::result::ZipError;
    use zip::write::SimpleFileOptions;

    fn npy(array: &impl WriteNpyExt) -> Vec<u8> {
        let mut bytes = Vec::new();
        array.write_npy(&mut bytes).unwrap();
        bytes
    }

    /// Writes entries with the compression of `numpy.savez` (stored) or `numpy.savez_compressed`
    /// (deflated).
    fn write_npz(
        writer: impl Write + Seek,
        method: CompressionMethod,
        entries: &[(&str, Vec<u8>)],
    ) {
        let mut npz = zip::ZipWriter::new(writer);
        let options = SimpleFileOptions::default().compression_method(method);
        for (name, bytes) in entries {
            npz.start_file(*name, options).unwrap();
            npz.write_all(bytes).unwrap();
        }
        npz.finish().unwrap();
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

    #[rstest]
    #[case::stored(CompressionMethod::Stored)]
    #[case::deflated(CompressionMethod::Deflated)]
    fn npz_profile_loads_numpy_entry_names(#[case] method: CompressionMethod) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.npz");
        write_npz(
            std::fs::File::create(&path).unwrap(),
            method,
            &[
                ("prefill_isl.npy", npy(&array![0.0, 128.0])),
                ("prefill_ttft_ms.npy", npy(&array![0.0, 64.0])),
                ("decode_active_kv_tokens.npy", npy(&array![0.0, 128.0])),
                ("decode_context_length.npy", npy(&array![0.0, 128.0])),
                ("decode_itl.npy", npy(&array![[2.0, 4.0], [6.0, 8.0]])),
            ],
        );

        let model = PerfModel::from_npz(&path).unwrap();
        assert_eq!(model.predict_prefill_time(1, 100, 0).unwrap(), 50.0);
        assert_eq!(model.predict_decode_time(1, 128, 0, 0).unwrap(), 6.0);
        assert_eq!(model.predict_decode_time(1, 0, 128, 0).unwrap(), 4.0);
        assert_eq!(model.predict_decode_time(1, 64, 64, 0).unwrap(), 5.0);
    }

    #[test]
    fn npz_lookup_prefers_the_exact_name_over_the_npy_suffix() {
        let mut bytes = Cursor::new(Vec::new());
        write_npz(
            &mut bytes,
            CompressionMethod::Stored,
            &[
                ("a", npy(&array![1.0])),
                ("a.npy", npy(&array![2.0])),
                ("b.npy", npy(&array![3.0])),
            ],
        );

        let mut npz = NpzReader::new(bytes).unwrap();
        assert_eq!(npz.by_name::<Array1<f64>>("a").unwrap(), array![1.0]);
        assert_eq!(npz.by_name::<Array1<f64>>("b").unwrap(), array![3.0]);
        let missing = npz.by_name::<Array1<f64>>("c").unwrap_err();
        assert!(matches!(
            missing.downcast_ref(),
            Some(ZipError::FileNotFound)
        ));
    }
}
