// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-worker prefill-performance calibration: wire types, request
//! resolution (default-filling + clamping), and the regression analysis.
//!
//! The calibration sweep is a single-stream ISL sweep with a fixed OSL
//! (default 64), `ignore_eos` so the engine produces exactly `osl` tokens,
//! and per-request token IDs whose first token is a cache-buster (so the
//! prefix cache never serves a calibration prompt). Python drives the
//! sweep and emits a `RawCalibrationPayload` JSON blob; this module
//! resolves user knobs against framework-captured defaults and fits two
//! regressions:
//!
//! - **TTFT vs ISL** — *quadratic*: `TTFT = a + b·ISL + c·ISL²`. Modern
//!   transformer prefill is O(N²) because attention is O(N²) in input
//!   length, so a single linear slope is the wrong model. The linear
//!   coefficient `b` represents per-token compute that grows linearly
//!   (MLP, projections) and the quadratic coefficient `c` represents the
//!   per-pair attention cost. The original "effective FLOPS = TTFT
//!   slope" interpretation does not survive this — at any ISL the
//!   instantaneous slope is `b + 2·c·ISL`, so quoting a single FLOPS
//!   figure requires picking an ISL.
//! - **ITL vs sequence position** — *linear*: `ITL = i + s·position`,
//!   where `position = ISL + 2 + k` for the k-th retained ITL. The
//!   y-intercept `i` reflects memory-bandwidth-bound per-token cost; the
//!   slope `s` reflects attention cost per active KV token.
//!
//! Two derived crossovers:
//!
//! - `N_OPT` — compute/memory crossover. Solves
//!   `b + 2·c·N = t2tl_intercept` for the ISL at which the marginal
//!   prefill cost per token equals the per-token decode cost. Falls back
//!   to `t2tl_intercept / b` when `c ≈ 0` (engines with no quadratic
//!   attention term measured, e.g. very short sweeps).
//! - `N_ATT = t2tl_intercept / t2tl_slope` — number of in-flight KV
//!   tokens at which the forward pass becomes attention-dominated
//!   (unchanged from the original linear-ITL model).

use anyhow::{Context, Result, anyhow};
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use serde::{Deserialize, Serialize};

/// Velo unary handler name. Worker bindings register the matching
/// handler from `PrefillRouterHandler::new` when a `calibrate_lambda`
/// is supplied.
pub const CALIBRATE_HANDLER: &str = "kvbm.prefill_router.calibrate";

/// Default OSL when the caller does not specify one. 64 generated tokens
/// is enough to drive a stable ITL regression while keeping the sweep
/// short.
pub const DEFAULT_OSL: u32 = 64;

/// Default RNG seed for body tokens.
pub const DEFAULT_SEED: u64 = 0xC0FFEE;

/// Default upper bound on `isl + osl` as a fraction of `max_seq_len`.
pub const DEFAULT_MAX_ISL_FRACTION: f64 = 0.80;

/// Default vocab range for random body tokens. Narrow on purpose —
/// matches the original TRT-LLM calibration's `[1000, 2000)`.
pub const DEFAULT_BODY_VOCAB: (u32, u32) = (1000, 2000);

/// Default vocab range for the per-request cache-buster first token.
/// Wider than [`DEFAULT_BODY_VOCAB`] so a single process can run many
/// calibration sweeps without first-token collisions.
pub const DEFAULT_BUSTER_VOCAB: (u32, u32) = (1000, 30000);

/// ISL ladder used when the caller does not specify `seq`. Walks
/// power-of-two ISLs from 1k up; the resolver keeps only the entries
/// that fit under the framework's `max_seq_len * max_isl_fraction` cap.
pub const DEFAULT_SEQ_LADDER: &[u32] = &[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

/// Knobs the caller can steer. Every field is optional — None means
/// "use framework default"; out-of-range values are silently clamped and
/// reported in `ResolvedCalibrationRequest::clamps`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CalibrationRequest {
    /// If true, re-run even when a cached result is available.
    #[serde(default)]
    pub force: bool,
    /// Explicit ISL ladder. If None, derived from [`DEFAULT_SEQ_LADDER`]
    /// and clamped by `max_isl_fraction` * `max_seq_len`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<Vec<u32>>,
    /// Output sequence length (tokens generated per request, with
    /// `ignore_eos`). Default [`DEFAULT_OSL`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl: Option<u32>,
    /// Body-token RNG seed. Default [`DEFAULT_SEED`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Fraction of `max_seq_len` the deepest ISL+OSL may consume. Only
    /// used when `seq` is None. Default [`DEFAULT_MAX_ISL_FRACTION`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_isl_fraction: Option<f64>,
    /// Whether to issue a tiny warmup request before the sweep starts.
    /// Default `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<bool>,
    /// `[lo, hi)` range for random body tokens. Default
    /// [`DEFAULT_BODY_VOCAB`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_vocab: Option<(u32, u32)>,
    /// `[lo, hi)` range for the per-request cache-buster first token.
    /// Default [`DEFAULT_BUSTER_VOCAB`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buster_vocab: Option<(u32, u32)>,
}

/// Framework-captured defaults the resolver uses to fill None values
/// and clamp out-of-range overrides. Captured Python-side from the live
/// vLLM engine + tokenizer at handler construction time and passed to
/// `PrefillRouterHandler::new`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationDefaults {
    pub model_id: String,
    /// Engine cap on `isl + osl`.
    pub max_seq_len: u32,
    /// Engine cap on `isl` alone (often equal to `max_seq_len`).
    pub max_input_len: u32,
    /// Engine cap on `osl` alone (often equal to `max_seq_len`).
    pub max_output_len: u32,
    /// Tokenizer vocab size; clamps body/buster ranges.
    pub vocab_size: u32,
    /// First token id past special/added tokens (lower bound for
    /// vocab clamps).
    pub safe_vocab_lo: u32,
}

impl CalibrationDefaults {
    /// Conservative fallback used when the handler is constructed
    /// without explicit framework defaults. Calibration with these
    /// defaults still works on most modern small models.
    pub const FALLBACK: Self = Self {
        model_id: String::new(),
        max_seq_len: 8192,
        max_input_len: 8192,
        max_output_len: 8192,
        vocab_size: 32_000,
        safe_vocab_lo: 1000,
    };
}

/// The deterministic, fully-resolved sweep parameters. Produced by
/// [`CalibrationRequest::resolve`]. This is what the Python lambda
/// actually sees (pythonized).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedCalibrationRequest {
    pub seq: Vec<u32>,
    pub osl: u32,
    pub seed: u64,
    pub max_isl_fraction: f64,
    pub warmup: bool,
    pub body_vocab: (u32, u32),
    pub buster_vocab: (u32, u32),
    /// Human-readable list of clamps that were applied — one entry per
    /// dropped or shrunken knob, so callers can see "what they asked
    /// for" vs "what actually ran".
    #[serde(default)]
    pub clamps: Vec<String>,
}

impl CalibrationRequest {
    /// Fill None values from `defaults` and clamp out-of-range overrides.
    /// Returns an error only if the resolved `seq` is empty (model too
    /// small for any sample at the requested OSL).
    pub fn resolve(&self, defaults: &CalibrationDefaults) -> Result<ResolvedCalibrationRequest> {
        let mut clamps: Vec<String> = Vec::new();

        let osl_raw = self.osl.unwrap_or(DEFAULT_OSL).max(1);
        let osl = osl_raw.min(defaults.max_output_len.max(1));
        if osl != osl_raw {
            clamps.push(format!(
                "osl clamped {} -> {} (max_output_len)",
                osl_raw, osl
            ));
        }

        let frac_raw = self.max_isl_fraction.unwrap_or(DEFAULT_MAX_ISL_FRACTION);
        let frac = frac_raw.clamp(0.10, 0.95);
        if (frac - frac_raw).abs() > f64::EPSILON {
            clamps.push(format!(
                "max_isl_fraction clamped {:.3} -> {:.3} (allowed range [0.10, 0.95])",
                frac_raw, frac
            ));
        }

        let cap = ((defaults.max_seq_len as f64) * frac) as u32;

        let candidate_seq: Vec<u32> = match &self.seq {
            Some(user_seq) => user_seq.clone(),
            None => DEFAULT_SEQ_LADDER.to_vec(),
        };

        let mut seq: Vec<u32> = Vec::with_capacity(candidate_seq.len());
        for &isl in &candidate_seq {
            if isl < 2 {
                clamps.push(format!("isl {} dropped (must be >= 2)", isl));
                continue;
            }
            if isl > defaults.max_input_len {
                clamps.push(format!(
                    "isl {} dropped (exceeds max_input_len {})",
                    isl, defaults.max_input_len
                ));
                continue;
            }
            if isl.saturating_add(osl) > defaults.max_seq_len {
                clamps.push(format!(
                    "isl {} dropped (isl+osl {} > max_seq_len {})",
                    isl,
                    isl.saturating_add(osl),
                    defaults.max_seq_len
                ));
                continue;
            }
            // When user_seq is None, we *also* honor the soft cap; explicit
            // user lists are only constrained by the hard caps above.
            if self.seq.is_none() && isl.saturating_add(osl) > cap {
                clamps.push(format!(
                    "isl {} dropped (isl+osl {} > max_isl_fraction cap {})",
                    isl,
                    isl.saturating_add(osl),
                    cap
                ));
                continue;
            }
            seq.push(isl);
        }
        if seq.is_empty() {
            return Err(anyhow!(
                "resolved seq is empty (max_seq_len={}, max_input_len={}, osl={}); \
                 model too small for any calibration sample",
                defaults.max_seq_len,
                defaults.max_input_len,
                osl
            ));
        }
        // The quadratic TTFT fit needs ≥ 4 distinct ISLs (3 params + a
        // residual DoF). Bail here so the caller sees a clear knob-level
        // error instead of a downstream linregress failure.
        let distinct = {
            let mut v = seq.clone();
            v.sort();
            v.dedup();
            v.len()
        };
        if distinct < 4 {
            return Err(anyhow!(
                "resolved seq has {} distinct ISL(s); need >= 4 for a quadratic TTFT fit. \
                 Either widen `seq` or relax `max_isl_fraction` (current: {:.2}, max_seq_len={})",
                distinct,
                frac,
                defaults.max_seq_len
            ));
        }

        let body_vocab = clamp_vocab_range(
            "body_vocab",
            self.body_vocab.unwrap_or(DEFAULT_BODY_VOCAB),
            defaults,
            &mut clamps,
        )?;
        let buster_vocab = clamp_vocab_range(
            "buster_vocab",
            self.buster_vocab.unwrap_or(DEFAULT_BUSTER_VOCAB),
            defaults,
            &mut clamps,
        )?;

        Ok(ResolvedCalibrationRequest {
            seq,
            osl,
            seed: self.seed.unwrap_or(DEFAULT_SEED),
            max_isl_fraction: frac,
            warmup: self.warmup.unwrap_or(true),
            body_vocab,
            buster_vocab,
            clamps,
        })
    }
}

fn clamp_vocab_range(
    label: &str,
    (lo_raw, hi_raw): (u32, u32),
    defaults: &CalibrationDefaults,
    clamps: &mut Vec<String>,
) -> Result<(u32, u32)> {
    let safe_lo = defaults.safe_vocab_lo;
    let safe_hi = defaults.vocab_size;
    if safe_hi <= safe_lo {
        return Err(anyhow!(
            "{label}: framework defaults degenerate (safe_vocab_lo={safe_lo}, vocab_size={safe_hi})"
        ));
    }
    let lo = lo_raw.max(safe_lo).min(safe_hi - 1);
    let hi = hi_raw.max(lo + 1).min(safe_hi);
    if (lo, hi) != (lo_raw, hi_raw) {
        clamps.push(format!(
            "{label} clamped ({}, {}) -> ({}, {}) (vocab [{}, {}))",
            lo_raw, hi_raw, lo, hi, safe_lo, safe_hi
        ));
    }
    Ok((lo, hi))
}

/// Per-request trace emitted by the Python lambda. `itl_us` is the list
/// of inter-token latencies (token N → token N+1), so its length is
/// `osl - 1`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawTrace {
    pub isl: u32,
    pub osl: u32,
    pub ttft_us: u64,
    pub itl_us: Vec<u64>,
    pub first_token: u32,
}

/// Full payload Python returns over the velo wire (as a JSON string,
/// decoded here).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawCalibrationPayload {
    pub traces: Vec<RawTrace>,
}

/// Scatter dataset for the two fitted regressions.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ScatterData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Fitted performance model. TTFT is quadratic in ISL (`a + b·ISL +
/// c·ISL²`) to capture the attention O(N²) term; ITL is linear in
/// sequence position.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PerformanceModel {
    /// TTFT(ISL) = `t2ft_intercept + t2ft_linear·ISL + t2ft_quadratic·ISL²`
    /// (μs).
    pub t2ft_intercept: f64,
    /// Linear coefficient (μs/token). Per-token compute that grows
    /// linearly in ISL (MLP, projections, KV write-out).
    pub t2ft_linear: f64,
    /// Quadratic coefficient (μs/token²). Attention's O(N²) prefill
    /// cost. Modern engines that use Flash-Attention still pay this in
    /// flops even if memory traffic is collapsed.
    pub t2ft_quadratic: f64,
    /// ITL(position) = `t2tl_intercept + t2tl_slope·position` (μs).
    pub t2tl_intercept: f64,
    /// μs of ITL added per additional active KV token in the
    /// generation phase.
    pub t2tl_slope: f64,
    pub t2ft_fit_r2: f64,
    pub t2tl_fit_r2: f64,
}

/// Final analysis output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationResults {
    pub performance_model: PerformanceModel,
    pub n_opt: u32,
    pub n_att: u32,
    pub traces: Vec<RawTrace>,
    pub t2ft_scatter_data: ScatterData,
    pub t2tl_scatter_data: ScatterData,
}

/// On-handler cache entry. Pairs the analyzed results with the resolved
/// request and the defaults snapshot that produced them, so a cache-hit
/// response can be reconstructed faithfully.
#[derive(Debug, Clone)]
pub struct CalibrationSnapshot {
    pub results: CalibrationResults,
    pub resolved: ResolvedCalibrationRequest,
    pub defaults: CalibrationDefaults,
}

/// Wire response. `resolved` + `defaults` make it possible to see what
/// knobs the framework picked and which user steers were clamped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationResponse {
    pub results: CalibrationResults,
    pub from_cache: bool,
    pub resolved: ResolvedCalibrationRequest,
    pub defaults: CalibrationDefaults,
}

/// Fit the performance model to a vector of raw traces. TTFT vs ISL is
/// fit with a quadratic (modern transformer prefill is O(N²) in input
/// length because of attention); ITL vs sequence position is fit with a
/// line. The first and last ITL of each trace are dropped to avoid
/// prefill-tail and shutdown noise.
pub fn analyze(raw: Vec<RawTrace>) -> Result<CalibrationResults> {
    if raw.is_empty() {
        return Err(anyhow!("no traces to analyze"));
    }

    // --- TTFT vs ISL (quadratic: Y = a + b·X + c·X²) ---
    let mut t2ft_x: Vec<f64> = Vec::with_capacity(raw.len());
    let mut t2ft_y: Vec<f64> = Vec::with_capacity(raw.len());
    for trace in &raw {
        t2ft_x.push(trace.isl as f64);
        t2ft_y.push(trace.ttft_us as f64);
    }
    let t2ft_scatter_data = ScatterData {
        x: t2ft_x.clone(),
        y: t2ft_y.clone(),
    };
    // Need at least three distinct ISLs *plus* a residual degree of
    // freedom for the quadratic fit, i.e. ≥ 4 distinct ISL points.
    // The default ISL ladder gives 6+ on a 32k-context model.
    let distinct_x = {
        let mut v = t2ft_x.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.dedup();
        v.len()
    };
    if distinct_x < 4 {
        return Err(anyhow!(
            "need at least 4 distinct ISLs for a quadratic TTFT fit; got {}",
            distinct_x
        ));
    }
    let (t2ft_intercept, t2ft_linear, t2ft_quadratic, t2ft_fit_r2) =
        fit_quadratic(&t2ft_x, &t2ft_y).context("fit TTFT vs ISL (quadratic)")?;

    // --- ITL vs sequence position ---
    let mut t2tl_x: Vec<f64> = Vec::new();
    let mut t2tl_y: Vec<f64> = Vec::new();
    for trace in &raw {
        // Position at the first ITL entry is `isl + 2` (prompt + 1st
        // generated token + 1). Increment per entry. Drop the first and
        // last ITL to remove prefill-tail and shutdown noise (matches
        // the original calibration's behavior).
        if trace.itl_us.len() <= 2 {
            continue;
        }
        let mut position = trace.isl.saturating_add(1) as f64;
        let slice = &trace.itl_us[1..trace.itl_us.len() - 1];
        for &latency in slice {
            position += 1.0;
            t2tl_x.push(position);
            t2tl_y.push(latency as f64);
        }
    }
    if t2tl_x.is_empty() {
        return Err(anyhow!(
            "no ITL samples to fit (each trace had <= 2 ITLs after dropping ends)"
        ));
    }
    let t2tl_scatter_data = ScatterData {
        x: t2tl_x.clone(),
        y: t2tl_y.clone(),
    };
    let (t2tl_intercept, t2tl_slope, t2tl_fit_r2) =
        fit_line(&t2tl_x, &t2tl_y).context("fit ITL vs position")?;

    // N_OPT: marginal prefill cost per token = decode cost per token.
    // Linear-marginal-of-quadratic is `b + 2·c·N`; set equal to
    // t2tl_intercept and solve for N. When c ≈ 0 fall back to the
    // original linear formula so single-line fits stay meaningful.
    let n_opt = if t2ft_quadratic.abs() > 1e-12 {
        saturating_u32((t2tl_intercept - t2ft_linear) / (2.0 * t2ft_quadratic))
    } else if t2ft_linear.abs() > 1e-12 {
        saturating_u32(t2tl_intercept / t2ft_linear)
    } else {
        0
    };
    let n_att = saturating_u32(t2tl_intercept / t2tl_slope);

    Ok(CalibrationResults {
        performance_model: PerformanceModel {
            t2ft_intercept,
            t2ft_linear,
            t2ft_quadratic,
            t2tl_intercept,
            t2tl_slope,
            t2ft_fit_r2,
            t2tl_fit_r2,
        },
        n_opt,
        n_att,
        traces: raw,
        t2ft_scatter_data,
        t2tl_scatter_data,
    })
}

fn fit_line(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    let data =
        RegressionDataBuilder::new().build_from(vec![("Y", y.to_vec()), ("X", x.to_vec())])?;
    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .data_columns("Y", ["X"])
        .fit()?;
    let params = model.parameters();
    if params.len() < 2 {
        return Err(anyhow!("linregress returned <2 params"));
    }
    Ok((params[0], params[1], model.rsquared()))
}

/// Fit Y = a + b·X + c·X² and return `(a, b, c, r²)`.
fn fit_quadratic(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64, f64)> {
    let x2: Vec<f64> = x.iter().map(|v| v * v).collect();
    let data = RegressionDataBuilder::new().build_from(vec![
        ("Y", y.to_vec()),
        ("X", x.to_vec()),
        ("X2", x2),
    ])?;
    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .data_columns("Y", ["X", "X2"])
        .fit()?;
    let params = model.parameters();
    if params.len() < 3 {
        return Err(anyhow!("linregress returned <3 params for quadratic fit"));
    }
    Ok((params[0], params[1], params[2], model.rsquared()))
}

fn saturating_u32(v: f64) -> u32 {
    if !v.is_finite() || v <= 0.0 {
        0
    } else if v >= u32::MAX as f64 {
        u32::MAX
    } else {
        v as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults sized so the default ladder yields ≥ 4 distinct ISLs
    /// (the quadratic-fit minimum). max_seq_len=32k, max_input_len=32k.
    fn defaults_32k() -> CalibrationDefaults {
        CalibrationDefaults {
            model_id: "test/model".into(),
            max_seq_len: 32_768,
            max_input_len: 32_768,
            max_output_len: 1024,
            vocab_size: 32_000,
            safe_vocab_lo: 1000,
        }
    }

    #[test]
    fn resolve_all_defaults() {
        let req = CalibrationRequest::default();
        let resolved = req.resolve(&defaults_32k()).unwrap();
        // max_seq_len=32768, frac=0.80 -> cap = 26214. With osl=64,
        // eligible ladder entries from DEFAULT_SEQ_LADDER are
        // [1024, 2048, 4096, 8192, 16384] (32768+64 > 26214).
        assert_eq!(resolved.seq, vec![1024, 2048, 4096, 8192, 16384]);
        assert_eq!(resolved.osl, 64);
        assert_eq!(resolved.seed, DEFAULT_SEED);
        assert!(resolved.warmup);
        assert_eq!(resolved.body_vocab, DEFAULT_BODY_VOCAB);
        assert_eq!(resolved.buster_vocab, DEFAULT_BUSTER_VOCAB);
    }

    #[test]
    fn resolve_drops_out_of_range_user_seq() {
        let req = CalibrationRequest {
            seq: Some(vec![512, 1024, 2048, 4096, 999_999]),
            ..Default::default()
        };
        let resolved = req.resolve(&defaults_32k()).unwrap();
        assert_eq!(resolved.seq, vec![512, 1024, 2048, 4096]);
        assert!(
            resolved.clamps.iter().any(|c| c.contains("999999")),
            "missing clamp note: {:?}",
            resolved.clamps
        );
    }

    #[test]
    fn resolve_clamps_osl() {
        let req = CalibrationRequest {
            osl: Some(10_000_000),
            ..Default::default()
        };
        let resolved = req.resolve(&defaults_32k()).unwrap();
        assert_eq!(resolved.osl, 1024); // == defaults.max_output_len
        assert!(
            resolved.clamps.iter().any(|c| c.starts_with("osl clamped")),
            "missing osl clamp: {:?}",
            resolved.clamps
        );
    }

    #[test]
    fn resolve_clamps_vocab_above_size() {
        let req = CalibrationRequest {
            buster_vocab: Some((1000, 999_999)),
            ..Default::default()
        };
        let resolved = req.resolve(&defaults_32k()).unwrap();
        assert_eq!(resolved.buster_vocab.1, 32_000);
        assert!(
            resolved
                .clamps
                .iter()
                .any(|c| c.starts_with("buster_vocab clamped")),
            "missing buster_vocab clamp: {:?}",
            resolved.clamps
        );
    }

    /// Resolved seq with < 4 distinct ISLs must error (quadratic fit
    /// downstream needs ≥ 4 distinct points).
    #[test]
    fn resolve_errors_when_too_few_isls() {
        let defaults = CalibrationDefaults {
            model_id: "small".into(),
            max_seq_len: 8192,
            max_input_len: 4096,
            max_output_len: 1024,
            vocab_size: 32_000,
            safe_vocab_lo: 1000,
        };
        let req = CalibrationRequest::default();
        // Default ladder against this small model yields [1024, 2048, 4096]
        // → 3 distinct ISLs, below the 4 required.
        let err = req.resolve(&defaults).unwrap_err();
        let s = err.to_string();
        assert!(
            s.contains("need >= 4"),
            "expected distinct-isl error, got: {s}"
        );
    }

    #[test]
    fn resolve_errors_when_model_too_small() {
        // Tiny model: no ladder entry fits.
        let defaults = CalibrationDefaults {
            model_id: "tiny".into(),
            max_seq_len: 512,
            max_input_len: 512,
            max_output_len: 128,
            vocab_size: 32_000,
            safe_vocab_lo: 1000,
        };
        let req = CalibrationRequest::default();
        let err = req.resolve(&defaults).unwrap_err();
        assert!(err.to_string().contains("model too small"));
    }

    /// Synthetic payload: TTFT = a + b·ISL + c·ISL² (quadratic),
    /// ITL_i = i + s·(ISL + 2 + k). Both regressions should recover the
    /// constants with R²≈1.
    #[test]
    fn analyze_roundtrip_quadratic_ttft_linear_itl() {
        const A: f64 = 1500.0; // TTFT intercept (μs)
        const B: f64 = 2.5; // linear TTFT coef (μs/token)
        const C: f64 = 5e-5; // quadratic TTFT coef (μs/token²) — attention
        const ITL_INT: f64 = 800.0;
        const ITL_SLOPE: f64 = 0.05;

        let mut traces = Vec::new();
        for &isl in &[1024u32, 2048, 4096, 8192, 16384, 32768] {
            let n = isl as f64;
            let ttft_us = (A + B * n + C * n * n) as u64;
            let osl = 32u32;
            let mut itl_us: Vec<u64> = Vec::with_capacity(osl as usize - 1);
            for i in 0..(osl as usize - 1) {
                let position = (isl as usize + 2 + i) as f64;
                itl_us.push((ITL_INT + ITL_SLOPE * position) as u64);
            }
            traces.push(RawTrace {
                isl,
                osl,
                ttft_us,
                itl_us,
                first_token: 12345 + isl,
            });
        }

        let results = analyze(traces).unwrap();
        let pm = results.performance_model;
        assert!(pm.t2ft_fit_r2 > 0.999, "t2ft R² = {}", pm.t2ft_fit_r2);
        assert!(pm.t2tl_fit_r2 > 0.999, "t2tl R² = {}", pm.t2tl_fit_r2);
        assert!(
            (pm.t2ft_linear - B).abs() / B < 0.10,
            "t2ft_linear {} vs {}",
            pm.t2ft_linear,
            B
        );
        assert!(
            (pm.t2ft_quadratic - C).abs() / C < 0.10,
            "t2ft_quadratic {} vs {}",
            pm.t2ft_quadratic,
            C
        );
        assert!(
            (pm.t2tl_slope - ITL_SLOPE).abs() / ITL_SLOPE < 0.05,
            "t2tl_slope {} vs {}",
            pm.t2tl_slope,
            ITL_SLOPE
        );
        // N_OPT analytically = (ITL_INT - B) / (2·C).
        let expected_n_opt = ((ITL_INT - B) / (2.0 * C)) as u32;
        let n_opt = results.n_opt;
        let rel_err = if expected_n_opt > 0 {
            ((n_opt as i64 - expected_n_opt as i64).unsigned_abs() as f64) / (expected_n_opt as f64)
        } else {
            0.0
        };
        assert!(
            rel_err < 0.10,
            "n_opt {} vs expected {} (rel_err {})",
            n_opt,
            expected_n_opt,
            rel_err
        );
        assert!(results.n_att > 0);
    }

    /// Quadratic TTFT fit requires ≥ 4 distinct ISLs (3 params + a
    /// residual DoF). Fewer points must produce a clear error rather
    /// than a silently-degraded fit.
    #[test]
    fn analyze_errors_with_too_few_isls() {
        let mut traces = Vec::new();
        for &isl in &[1024u32, 2048, 4096] {
            traces.push(RawTrace {
                isl,
                osl: 8,
                ttft_us: 1000 + (isl as u64) * 2,
                itl_us: vec![500, 600, 650, 700, 750, 800, 850],
                first_token: isl,
            });
        }
        let err = analyze(traces).unwrap_err();
        assert!(
            err.to_string().contains("at least 4 distinct ISLs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn analyze_errors_on_empty() {
        let err = analyze(Vec::new()).unwrap_err();
        assert!(err.to_string().contains("no traces"));
    }

    #[test]
    fn request_roundtrips_json_with_optional_fields_omitted() {
        let req = CalibrationRequest::default();
        let s = serde_json::to_string(&req).unwrap();
        // None fields skip; force defaults to false.
        assert_eq!(s, "{\"force\":false}");
        let back: CalibrationRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn raw_trace_roundtrips_json() {
        let t = RawTrace {
            isl: 1024,
            osl: 64,
            ttft_us: 1234,
            itl_us: vec![10, 11, 12],
            first_token: 4242,
        };
        let s = serde_json::to_string(&t).unwrap();
        let back: RawTrace = serde_json::from_str(&s).unwrap();
        assert_eq!(back, t);
    }
}
