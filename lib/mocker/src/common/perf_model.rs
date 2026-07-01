// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! Built-in timing comes from polynomial, interpolated, or AI Configurator
//! models. Embedding applications can provide a native Rust model instead.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_interp::InterpolateError;
use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use ndarray_interp::interp2d::{Bilinear, Interp2DBuilder};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Exact request shapes for one replay prefill forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayPrefillInput<'a> {
    /// Per-request sequence lengths in forward-pass order.
    pub sequence_lengths: &'a [usize],
    /// Per-request cached-prefix lengths in the same order.
    pub prefix_lengths: &'a [usize],
}

impl<'a> ReplayPrefillInput<'a> {
    /// Validate and construct an exact prefill batch description.
    pub fn new(sequence_lengths: &'a [usize], prefix_lengths: &'a [usize]) -> Result<Self> {
        if sequence_lengths.is_empty() {
            anyhow::bail!("replay prefill input requires at least one request");
        }
        if sequence_lengths.len() != prefix_lengths.len() {
            anyhow::bail!(
                "replay prefill input length mismatch: sequence_lengths={}, prefix_lengths={}",
                sequence_lengths.len(),
                prefix_lengths.len()
            );
        }
        if let Some((index, (prefix, sequence))) = prefix_lengths
            .iter()
            .zip(sequence_lengths)
            .enumerate()
            .find(|(_, (prefix, sequence))| prefix > sequence)
        {
            anyhow::bail!(
                "replay prefill prefix length exceeds sequence length at index {index}: prefix={prefix}, sequence={sequence}"
            );
        }
        Ok(Self {
            sequence_lengths,
            prefix_lengths,
        })
    }

    /// Number of requests in the forward pass.
    pub fn batch_size(&self) -> usize {
        self.sequence_lengths.len()
    }

    /// Integer average of the exact sequence lengths.
    pub fn avg_sequence_length(&self) -> usize {
        average_length(self.sequence_lengths)
    }

    /// Integer average of the exact cached-prefix lengths.
    pub fn avg_prefix_length(&self) -> usize {
        average_length(self.prefix_lengths)
    }

    /// AIC-compatible average sequence length minus average prefix length.
    pub fn avg_effective_input_length(&self) -> usize {
        self.avg_sequence_length()
            .saturating_sub(self.avg_prefix_length())
    }
}

/// Exact request shapes and KV occupancy for one replay decode forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayDecodeInput<'a> {
    /// Per-request context lengths in forward-pass order.
    pub sequence_lengths: &'a [usize],
    /// KV tokens active for this forward pass.
    pub active_kv_tokens: usize,
    /// Total KV-token capacity of the worker.
    pub total_kv_tokens: usize,
    /// Number of output tokens attempted per request.
    pub output_length: usize,
}

impl ReplayDecodeInput<'_> {
    /// Number of requests in the forward pass.
    pub fn batch_size(&self) -> usize {
        self.sequence_lengths.len()
    }

    /// Integer average of the exact context lengths.
    pub fn avg_context_length(&self) -> usize {
        average_length(self.sequence_lengths)
    }
}

fn average_length(lengths: &[usize]) -> usize {
    if lengths.is_empty() {
        return 0;
    }
    lengths.iter().sum::<usize>() / lengths.len()
}

/// Prefill latency model used by replay schedulers.
pub trait ReplayPrefillLatencyModel: Send + Sync {
    /// Predict prefill latency in milliseconds for one exact forward pass.
    fn prefill_latency_ms(&self, input: ReplayPrefillInput<'_>) -> f64;
}

/// Decode latency model used by replay schedulers.
pub trait ReplayDecodeLatencyModel: Send + Sync {
    /// Predict decode latency in milliseconds for one exact forward pass.
    fn decode_latency_ms(&self, input: ReplayDecodeInput<'_>) -> f64;
}

/// Combined latency model for applications that use one model for both phases.
pub trait ReplayLatencyModel: ReplayPrefillLatencyModel + ReplayDecodeLatencyModel {}

impl<T> ReplayLatencyModel for T where T: ReplayPrefillLatencyModel + ReplayDecodeLatencyModel {}

pub(crate) fn normalize_replay_latency_ms(
    latency_ms: f64,
    minimum_ms: f64,
    phase: &'static str,
) -> f64 {
    if latency_ms.is_finite() && latency_ms >= 0.0 {
        return latency_ms.max(minimum_ms);
    }

    tracing::warn!(
        phase,
        latency_ms,
        minimum_ms,
        "Replay latency model returned an invalid latency; using the minimum"
    );
    minimum_ms
}

pub(crate) fn replay_latency_duration(
    latency_ms: f64,
    minimum_ms: f64,
    phase: &'static str,
) -> Duration {
    let seconds = normalize_replay_latency_ms(latency_ms, minimum_ms, phase) / 1000.0;
    duration_from_seconds(seconds, phase)
}

pub(crate) fn scale_replay_duration(
    duration: Duration,
    speedup_ratio: f64,
    phase: &'static str,
) -> Duration {
    if duration.is_zero() || !speedup_ratio.is_finite() || speedup_ratio <= 0.0 {
        return duration;
    }
    duration_from_seconds(duration.as_secs_f64() / speedup_ratio, phase)
}

fn duration_from_seconds(seconds: f64, phase: &'static str) -> Duration {
    if seconds.is_finite() && seconds >= 0.0 && seconds < Duration::MAX.as_secs_f64() {
        return Duration::from_secs_f64(seconds);
    }
    tracing::warn!(
        phase,
        seconds,
        "Replay latency exceeds the representable duration; clamping to Duration::MAX"
    );
    Duration::MAX
}

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
    /// Native replay models supplied by an embedding Rust application.
    Custom {
        prefill: Arc<dyn ReplayPrefillLatencyModel>,
        decode: Arc<dyn ReplayDecodeLatencyModel>,
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
            PerfModel::Custom { prefill, decode } => PerfModel::Custom {
                prefill: Arc::clone(prefill),
                decode: Arc::clone(decode),
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
            PerfModel::Custom { .. } => write!(f, "PerfModel::Custom {{ .. }}"),
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

    /// Use one native Rust model for both replay phases.
    pub fn from_replay_latency_model<M>(model: Arc<M>) -> Self
    where
        M: ReplayLatencyModel + 'static,
    {
        let prefill: Arc<dyn ReplayPrefillLatencyModel> = model.clone();
        let decode: Arc<dyn ReplayDecodeLatencyModel> = model;
        Self::Custom { prefill, decode }
    }

    /// Use independent native Rust models for prefill and decode replay.
    pub fn from_replay_latency_models<P, D>(prefill: Arc<P>, decode: Arc<D>) -> Self
    where
        P: ReplayPrefillLatencyModel + 'static,
        D: ReplayDecodeLatencyModel + 'static,
    {
        Self::Custom { prefill, decode }
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
        if batch_size == 0 {
            return 0.0;
        }
        if matches!(self, Self::Custom { .. }) {
            let sequence_lengths = vec![isl; batch_size];
            let prefix_lengths = vec![prefix; batch_size];
            return self.prefill_latency_ms(
                ReplayPrefillInput::new(&sequence_lengths, &prefix_lengths)
                    .expect("aggregate prefill shape must be valid"),
            );
        }
        self.predict_prefill_aggregates(batch_size, isl.saturating_sub(prefix), prefix)
    }

    fn predict_prefill_aggregates(
        &self,
        batch_size: usize,
        avg_effective_input_length: usize,
        avg_prefix_length: usize,
    ) -> f64 {
        if batch_size == 0 || avg_effective_input_length == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                // Total tokens across the batch — GPU processes them in parallel
                let tokens = (batch_size * avg_effective_input_length) as f64;
                4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
            }
            PerfModel::Interpolated { prefill_interp, .. } => {
                let tokens = (batch_size * avg_effective_input_length) as f64;
                prefill_interp.interp(tokens).unwrap_or(0.0)
            }
            PerfModel::Aiconfigurator {
                callback,
                attention_dp_size,
            } => callback.predict_prefill(
                Self::aic_per_rank_batch(batch_size, *attention_dp_size),
                avg_effective_input_length,
                avg_prefix_length,
            ),
            PerfModel::Custom { .. } => unreachable!("custom model handled before aggregation"),
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
        if matches!(self, Self::Custom { .. }) {
            let sequence_lengths = vec![context_length; batch_size];
            return self.decode_latency_ms(ReplayDecodeInput {
                sequence_lengths: &sequence_lengths,
                active_kv_tokens,
                total_kv_tokens,
                output_length: 2,
            });
        }
        self.predict_decode_aggregates(
            batch_size,
            active_kv_tokens,
            context_length,
            total_kv_tokens,
        )
    }

    fn predict_decode_aggregates(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        avg_context_length: usize,
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
                .interp(active_kv_tokens as f64, avg_context_length as f64)
                .unwrap_or(0.0),
            PerfModel::Aiconfigurator {
                callback,
                attention_dp_size,
            } => callback.predict_decode(
                Self::aic_per_rank_batch(batch_size, *attention_dp_size),
                avg_context_length,
                2,
            ),
            PerfModel::Custom { .. } => unreachable!("custom model handled before aggregation"),
        };
        // Token-emitting decode steps should not collapse onto the same timestamp.
        let result = time.max(1.0);
        tracing::trace!(
            "Decode time prediction: batch_size={batch_size}, active_kv_tokens={active_kv_tokens}, context_length={avg_context_length}, time={result:.2}ms"
        );
        result
    }
}

impl ReplayPrefillLatencyModel for PerfModel {
    fn prefill_latency_ms(&self, input: ReplayPrefillInput<'_>) -> f64 {
        match self {
            Self::Custom { prefill, .. } => prefill.prefill_latency_ms(input),
            _ => self.predict_prefill_aggregates(
                input.batch_size(),
                input.avg_effective_input_length(),
                input.avg_prefix_length(),
            ),
        }
    }
}

impl ReplayDecodeLatencyModel for PerfModel {
    fn decode_latency_ms(&self, input: ReplayDecodeInput<'_>) -> f64 {
        match self {
            Self::Custom { decode, .. } => decode.decode_latency_ms(input),
            _ => self.predict_decode_aggregates(
                input.batch_size(),
                input.active_kv_tokens,
                input.avg_context_length(),
                input.total_kv_tokens,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingAicCallback {
        prefill_calls: Mutex<Vec<(usize, usize, usize)>>,
        decode_calls: Mutex<Vec<(usize, usize, usize)>>,
    }

    impl AicCallback for RecordingAicCallback {
        fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64 {
            self.prefill_calls
                .lock()
                .unwrap()
                .push((batch_size, effective_isl, prefix));
            2.0
        }

        fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64 {
            self.decode_calls
                .lock()
                .unwrap()
                .push((batch_size, isl, osl));
            1.0
        }
    }

    #[derive(Default)]
    struct RecordingPrefillInterpolator {
        calls: Mutex<Vec<f64>>,
    }

    impl PrefillInterpolator for RecordingPrefillInterpolator {
        fn interp(&self, x: f64) -> Result<f64, InterpolateError> {
            self.calls.lock().unwrap().push(x);
            Ok(7.0)
        }
    }

    #[derive(Default)]
    struct RecordingDecodeInterpolator {
        calls: Mutex<Vec<(f64, f64)>>,
    }

    impl DecodeInterpolator for RecordingDecodeInterpolator {
        fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError> {
            self.calls.lock().unwrap().push((x, y));
            Ok(11.0)
        }
    }

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

    #[test]
    fn normalize_replay_latency_ms_enforces_contract() {
        for (latency_ms, minimum_ms, expected_ms) in [
            (f64::NAN, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (2.0, 1.0, 2.0),
        ] {
            assert_eq!(
                normalize_replay_latency_ms(latency_ms, minimum_ms, "test"),
                expected_ms
            );
        }
    }

    #[test]
    fn replay_latency_duration_clamps_unrepresentable_values() {
        assert_eq!(
            replay_latency_duration(f64::MAX, 0.0, "test"),
            Duration::MAX
        );
        assert_eq!(
            scale_replay_duration(Duration::from_secs(1), f64::MIN_POSITIVE, "test"),
            Duration::MAX
        );
    }

    #[test]
    fn replay_prefill_input_validates_request_shapes() {
        assert!(ReplayPrefillInput::new(&[], &[]).is_err());
        assert!(ReplayPrefillInput::new(&[8, 12], &[0]).is_err());
        assert!(ReplayPrefillInput::new(&[8, 12], &[0, 13]).is_err());
    }

    #[test]
    fn replay_inputs_derive_legacy_averages_from_exact_lengths() {
        let prefill = ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap();
        assert_eq!(prefill.batch_size(), 2);
        assert_eq!(prefill.avg_sequence_length(), 10);
        assert_eq!(prefill.avg_prefix_length(), 4);
        assert_eq!(prefill.avg_effective_input_length(), 6);

        let decode = ReplayDecodeInput {
            sequence_lengths: &[9, 14],
            active_kv_tokens: 23,
            total_kv_tokens: 128,
            output_length: 3,
        };
        assert_eq!(decode.batch_size(), 2);
        assert_eq!(decode.avg_context_length(), 11);
        assert_eq!(decode.output_length, 3);
    }

    #[test]
    fn polynomial_model_uses_legacy_aggregates_from_exact_lengths() {
        let model = PerfModel::Polynomial;
        let prefill = ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap();
        let decode = ReplayDecodeInput {
            sequence_lengths: &[9, 14],
            active_kv_tokens: 23,
            total_kv_tokens: 128,
            output_length: 3,
        };

        assert_eq!(
            model.prefill_latency_ms(prefill),
            model.predict_prefill_time(2, 10, 4)
        );
        assert_eq!(
            model.decode_latency_ms(decode),
            model.predict_decode_time(2, 23, 11, 128)
        );
    }

    #[test]
    fn interpolated_model_uses_legacy_aggregates_from_exact_lengths() {
        let prefill_interp = Arc::new(RecordingPrefillInterpolator::default());
        let decode_interp = Arc::new(RecordingDecodeInterpolator::default());
        let model = PerfModel::Interpolated {
            prefill_interp: prefill_interp.clone(),
            decode_interp: decode_interp.clone(),
        };

        assert_eq!(
            model.prefill_latency_ms(ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap()),
            7.0
        );
        assert_eq!(
            model.decode_latency_ms(ReplayDecodeInput {
                sequence_lengths: &[9, 14],
                active_kv_tokens: 23,
                total_kv_tokens: 128,
                output_length: 3,
            }),
            11.0
        );
        assert_eq!(*prefill_interp.calls.lock().unwrap(), vec![12.0]);
        assert_eq!(*decode_interp.calls.lock().unwrap(), vec![(23.0, 11.0)]);
    }

    #[test]
    fn aic_model_derives_legacy_aggregates_from_exact_lengths() {
        let callback = Arc::new(RecordingAicCallback::default());
        let model = PerfModel::from_aic_callback(callback.clone());

        model.prefill_latency_ms(ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap());
        model.decode_latency_ms(ReplayDecodeInput {
            sequence_lengths: &[9, 14],
            active_kv_tokens: 23,
            total_kv_tokens: 128,
            output_length: 3,
        });

        assert_eq!(*callback.prefill_calls.lock().unwrap(), vec![(2, 6, 4)]);
        assert_eq!(*callback.decode_calls.lock().unwrap(), vec![(2, 11, 2)]);
    }
}
