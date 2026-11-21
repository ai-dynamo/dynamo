// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Size-interval task assignment (SITA) worker selection, with optional score shaping.
//!
//! On agentic traffic, request sizes span orders of magnitude: a short tool-call follow-up and a
//! 100k-token context reload arrive at the same pool. Under the default cost they share every
//! worker, so a short request can queue behind a multi-thousand-token prefill. SITA partitions
//! the eligible workers into contiguous size bands and confines each request to the band matching
//! its estimated size, which keeps short requests out of long requests' queues.
//!
//! ```text
//! size = (isl_tokens - best_cached_tokens_in_band) + osl_weight * expected_output_tokens
//! band = 0 if size <= boundary_1, 1 if size <= boundary_2 (or boundary_2 == 0), else 2
//! ```
//!
//! Band 0 holds `ceil(small_band_share * N)` workers, ordered by worker ID; the longer bands split
//! the rest. Each request then picks the minimum-cost worker inside its band, with the default's
//! uniform tie-breaking and temperature sampling.
//!
//! Strict confinement has costs, so three relief valves widen the band, all driven by
//! `spill_threshold`:
//!
//! - **Spill.** A band whose relative occupancy exceeds `spill_threshold` widens into an adjacent,
//!   quieter band: upward when one exists, otherwise the top band widens down into band 1, never
//!   into band 0.
//! - **Borrow.** A saturated long band may also borrow band 0 while band 0 is measurably idle.
//!   The borrowed band 0 is added as its own range, so it never re-admits a band in between.
//! - **Uncached widening.** A long request with no cached prefix in its band has no affinity to
//!   preserve, so it may use every non-short worker, and band 0 too under a looser idleness bar.
//!
//! `spill_threshold: 1.0` turns off spill and borrowing. Uncached widening still sends an uncached
//! long request to the whole non-short pool, but band 0 is never lent at `1.0`.
//!
//! Cache only shrinks a request's size when the band's own workers hold it. The request goes to
//! the smallest band it still fits after counting that band's own cache, so a prefix cached on
//! another band's worker never sends it cold to a small-band worker.
//!
//! The scorer reproduces the default cost from the same `SelectionConfig` weights, optionally
//! reshaped by the `score_*` parameters:
//!
//! ```text
//! overlap_frac = effective_overlap_blocks / request_blocks
//! local_credit = 0                                         if overlap_frac < score_min_overlap_frac
//!              = overlap_frac ^ (score_overlap_gamma - 1) * (device + host + disk credit)  otherwise
//! credit       = score_overlap_weight * (local_credit + shared_cache_credit)
//! load_scale   = score_load_weight * (decode_load / mean_decode_load) ^ (score_load_gamma - 1)
//! ```
//!
//! Shared-cache credit is not part of `overlap_frac`, so the cutoff and the curve leave it alone.
//! The neutral setting (both weights and both gammas `1.0`, `score_min_overlap_frac: 0.0`) is the
//! default; with it the scorer reproduces the default cost bit for bit, so any routing difference
//! from `dynamo-default-cost-fn` comes from the band restriction alone.

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

use crate::default::softmax_sample_index;

/// Policy type selected under a `worker_selection` role.
pub const POLICY_TYPE: &str = "dynamo-sita";

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional and keeps its default when omitted. Unknown keys are rejected at
/// startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Upper size bound of band 0, in tokens.
    boundary_1: usize,
    /// Upper size bound of band 1, in tokens. `0` disables the third band, leaving a two-band
    /// split at `boundary_1`.
    boundary_2: usize,
    /// Fraction of the pool assigned to band 0, strictly between 0 and 1.
    small_band_share: f64,
    /// Weight on the caller-declared `expected_output_tokens` in the size estimate. `0.0` sizes
    /// requests by prefill alone.
    osl_weight: f64,
    /// Relative occupancy above which a band may spill or borrow, in `[0.5, 1.0]`. `1.0` disables
    /// spill and borrowing; uncached long requests still widen to the non-short pool.
    spill_threshold: f64,
    /// Multiplier on the overlap credit. Above `1.0` chases cache locality harder.
    score_overlap_weight: f64,
    /// Exponent on the overlap fraction. Above `1.0` concentrates credit on near-full hits.
    score_overlap_gamma: f64,
    /// Overlap fraction below which a worker earns no local cache credit.
    score_min_overlap_frac: f64,
    /// Multiplier on the decode-side load term.
    score_load_weight: f64,
    /// Convexity of the decode-side load relative to the candidate mean. Above `1.0` charges an
    /// already-busy worker superlinearly.
    score_load_gamma: f64,
    // The default cost's parameters, with the same names, meaning and resolution as on
    // `dynamo-default-cost-fn`. Each falls back to the router setting when omitted.
    overlap_score_credit: Option<f64>,
    overlap_score_credit_decay: Option<f64>,
    prefill_load_scale: Option<f64>,
    decode_active_request_weight: Option<f64>,
    host_cache_hit_weight: Option<f64>,
    disk_cache_hit_weight: Option<f64>,
    shared_cache_multiplier: Option<f64>,
    router_temperature: Option<f64>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            boundary_1: 1024,
            boundary_2: 8192,
            small_band_share: 0.5,
            osl_weight: 0.0,
            spill_threshold: 0.85,
            score_overlap_weight: 1.0,
            score_overlap_gamma: 1.0,
            score_min_overlap_frac: 0.0,
            score_load_weight: 1.0,
            score_load_gamma: 1.0,
            overlap_score_credit: None,
            overlap_score_credit_decay: None,
            prefill_load_scale: None,
            decode_active_request_weight: None,
            host_cache_hit_weight: None,
            disk_cache_hit_weight: None,
            shared_cache_multiplier: None,
            router_temperature: None,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.boundary_1 == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "boundary_1 must be greater than 0",
            ));
        }
        if self.boundary_2 != 0 && self.boundary_2 <= self.boundary_1 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "boundary_2 must be 0 (two-band split) or greater than boundary_1",
            ));
        }
        if !(self.small_band_share > 0.0 && self.small_band_share < 1.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "small_band_share must be between 0.0 and 1.0, exclusive",
            ));
        }
        if !self.osl_weight.is_finite() || self.osl_weight < 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "osl_weight must be finite and non-negative",
            ));
        }
        for (name, value, low, high) in [
            ("spill_threshold", self.spill_threshold, 0.5, 1.0),
            ("score_overlap_weight", self.score_overlap_weight, 0.25, 4.0),
            ("score_overlap_gamma", self.score_overlap_gamma, 0.5, 2.0),
            (
                "score_min_overlap_frac",
                self.score_min_overlap_frac,
                0.0,
                0.5,
            ),
            ("score_load_weight", self.score_load_weight, 0.25, 4.0),
            ("score_load_gamma", self.score_load_gamma, 1.0, 2.0),
        ] {
            if !value.is_finite() || !(low..=high).contains(&value) {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be a finite number between {low} and {high}"
                )));
            }
        }
        // Validated as `dynamo-default-cost-fn` validates them.
        for (name, value) in [
            ("overlap_score_credit", self.overlap_score_credit),
            (
                "overlap_score_credit_decay",
                self.overlap_score_credit_decay,
            ),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("shared_cache_multiplier", self.shared_cache_multiplier),
            ("router_temperature", self.router_temperature),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Band placement
// ---------------------------------------------------------------------------------------------

/// Half-open index range into the sorted worker list.
type Slice = (usize, usize);

/// Map an estimated request size in tokens to a band index.
fn band_for_size(size: usize, boundary_1: usize, boundary_2: usize) -> usize {
    if size <= boundary_1 {
        0
    } else if boundary_2 == 0 || size <= boundary_2 {
        1
    } else {
        2
    }
}

/// Partition `worker_count >= 2` workers into contiguous, non-empty per-band slices. A two-band
/// split (or a pool too small for three) repeats the tail slice for band 2.
fn band_slices(worker_count: usize, small_band_share: f64, band_count: usize) -> [Slice; 3] {
    debug_assert!(worker_count >= 2);
    let band_0 =
        ((small_band_share * worker_count as f64).ceil() as usize).clamp(1, worker_count - 1);
    let remaining = worker_count - band_0;
    if band_count < 3 || remaining == 1 {
        let tail = (band_0, worker_count);
        return [(0, band_0), tail, tail];
    }
    // Keep at least one worker in the largest band so huge requests never share the whole tail
    // with medium ones.
    let band_1 = remaining.div_ceil(2).clamp(1, remaining - 1);
    [
        (0, band_0),
        (band_0, band_0 + band_1),
        (band_0 + band_1, worker_count),
    ]
}

fn slice_mean(loads: &[f64], (start, end): Slice) -> f64 {
    if end <= start {
        return 0.0;
    }
    loads[start..end].iter().sum::<f64>() / (end - start) as f64
}

/// Share of the pool's per-worker load that sits in `slice`, in `[0, 1]`.
///
/// An evenly loaded pool scores 0.5, rising toward 1.0 as the band's workers get busier than
/// everyone else's. Expressing occupancy relatively is what makes `spill_threshold`'s
/// `[0.5, 1.0]` range meaningful: an absolute KV-capacity fraction is a few percent under any
/// realistic serving load, so no threshold in that range would ever trip.
fn band_occupancy(loads: &[f64], slice: Slice) -> f64 {
    let (start, end) = slice;
    let outside_count = loads.len() - (end - start);
    if outside_count == 0 {
        return 0.5;
    }
    let inside = slice_mean(loads, slice);
    let outside_total = loads.iter().sum::<f64>() - loads[start..end].iter().sum::<f64>();
    let denominator = inside + outside_total / outside_count as f64;
    if denominator <= 0.0 {
        return 0.5;
    }
    inside / denominator
}

/// Widen a saturated band into a quieter neighbor, then let it borrow an idle band 0.
///
/// Only a band above `spill_threshold` needs relief. A band below it keeps its own workers even
/// when band 0 is idle: parking a long request there would delay later short requests without
/// relieving anything.
///
/// The neighbor is asymmetric. Sending a request *up* into a longer band costs roughly its own
/// service time; sending a long request *down* parks a multi-thousand-token prefill in front of
/// short ones, the head-of-line blocking SITA exists to prevent. So the band above is preferred,
/// and the top band falls back to band 1, never band 0. Without that downward valve the top band
/// would be a saturation sink, receiving spill from below and never shedding it.
///
/// Returns up to two ranges: the band, widened by its neighbor when that neighbor qualifies, and
/// band 0 when it is borrowed. They stay separate so borrowing band 0 never re-admits a band that
/// the spill check rejected in between.
fn apply_spill(
    band: usize,
    band_count: usize,
    slices: &[Slice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> Vec<Slice> {
    let target = slices[band];
    if spill_threshold >= 1.0 || band_occupancy(loads, target) <= spill_threshold {
        return vec![target];
    }
    let neighbor_band = if band + 1 < band_count {
        Some(band + 1)
    } else if band >= 2 {
        Some(band - 1)
    } else {
        None
    };
    let widened = match neighbor_band.map(|next| slices[next]) {
        Some(neighbor)
            if neighbor != target && slice_mean(loads, neighbor) < slice_mean(loads, target) =>
        {
            (target.0.min(neighbor.0), target.1.max(neighbor.1))
        }
        _ => target,
    };
    let mut ranges = vec![widened];
    if borrows_idle_short_band(widened, slices, loads, spill_threshold) {
        ranges.push(slices[0]);
    }
    ranges
}

/// Whether a saturated long band may borrow band 0's workers: only while band 0 is measurably
/// idle.
///
/// The short-band reservation is only free while short requests use it. Fencing it off
/// unconditionally starves the long bands of KV capacity, and their tail end-to-end latency blows
/// up even though time-to-first-token is unchanged. `spill_threshold` sets both sides of the
/// dial: a band is saturated above it and idle below its complement.
fn borrows_idle_short_band(
    target: Slice,
    slices: &[Slice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> bool {
    target.0 > slices[0].0 && band_occupancy(loads, slices[0]) < 1.0 - spill_threshold
}

/// Widen a long request with no cached prefix in its band to every non-short worker, and to band
/// 0 too while band 0 is idle under a looser bar.
///
/// Band confinement pays for itself through cache affinity: same-sized requests keep revisiting
/// the same workers, so their shared prefixes stay resident. A request with nothing cached in its
/// band has no affinity to preserve, and the long band's requests carry the largest contexts and
/// longest decodes, so confining them to a slice multiplies the decode contention there. Such a
/// request is also the cheapest possible borrower of band 0: lending it a worker costs the short
/// band only the load it brings, never a lost cache hit, so it gets a doubled idleness allowance.
///
/// This applies at every `spill_threshold`, including `1.0`: an uncached long request always gets
/// the whole non-short pool. At `1.0` the idleness allowance is zero, so band 0 is never lent.
fn widen_uncached_long_band(
    band: usize,
    slices: &[Slice; 3],
    worker_count: usize,
    best_cached_tokens: usize,
    loads: &[f64],
    spill_threshold: f64,
) -> Slice {
    if band == 0 || best_cached_tokens > 0 {
        return slices[band];
    }
    let idle_allowance = (1.0 - spill_threshold) * 2.0;
    if band_occupancy(loads, slices[0]) < idle_allowance {
        return (0, worker_count);
    }
    (slices[0].1, worker_count)
}

/// Per-worker signals SITA places a request with, aggregated across a worker's dp ranks.
struct Pool {
    /// Ascending, deduplicated worker IDs.
    worker_ids: Vec<WorkerId>,
    /// Mean queued work per rank, in tokens: queued prefill plus resident decode.
    loads: Vec<f64>,
    /// Best cached prompt tokens across the worker's ranks.
    cached_tokens: Vec<usize>,
}

impl Pool {
    fn new(
        candidates: &[ScoredWorkerCandidate],
        cache: WorkerCacheInputs<'_>,
        load: &[WorkerLoadInput],
        block_size: u32,
    ) -> Self {
        let mut worker_ids: Vec<WorkerId> =
            candidates.iter().map(|c| c.worker().worker_id).collect();
        worker_ids.sort_unstable();
        worker_ids.dedup();
        let mut load_sums = vec![0.0; worker_ids.len()];
        let mut ranks = vec![0usize; worker_ids.len()];
        let mut cached_tokens = vec![0usize; worker_ids.len()];
        for (row, candidate) in candidates.iter().enumerate() {
            let index = worker_ids
                .binary_search(&candidate.worker().worker_id)
                .expect("every row's worker is in the pool");
            if let Some(rank_load) = load.get(row) {
                // Queued prefill dominates TTFT, so it counts directly; resident decode blocks
                // are converted to tokens so both contribute in the same unit.
                load_sums[index] += rank_load.active_prefill_tokens() as f64
                    + (rank_load.active_decode_blocks() * block_size as usize) as f64;
                ranks[index] += 1;
            }
            if let Some(rank_cache) = cache.get(row) {
                cached_tokens[index] =
                    cached_tokens[index].max(rank_cache.accounting_cache_estimate().1);
            }
        }
        let loads = load_sums
            .into_iter()
            .zip(ranks)
            .map(|(sum, ranks)| if ranks == 0 { 0.0 } else { sum / ranks as f64 })
            .collect();
        Self {
            worker_ids,
            loads,
            cached_tokens,
        }
    }

    fn best_cached_in(&self, (start, end): Slice) -> usize {
        self.cached_tokens[start..end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Whether `worker_id` falls in one of `ranges`, which index into the sorted worker list.
    fn contains(&self, ranges: &[Slice], worker_id: WorkerId) -> bool {
        self.worker_ids
            .binary_search(&worker_id)
            .is_ok_and(|index| {
                ranges
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&index))
            })
    }

    /// Ranges of the sorted worker list this request may route to, or `None` when the pool is too
    /// small to partition.
    fn band_ranges(
        &self,
        parameters: &Parameters,
        context: &WorkerSelectionContext<'_>,
    ) -> Option<Vec<Slice>> {
        let worker_count = self.worker_ids.len();
        if worker_count < 2 {
            return None;
        }
        let output_tokens = context
            .expected_output_tokens()
            .map_or(0.0, |tokens| parameters.osl_weight * f64::from(tokens));
        let size_with_cache = |cached: usize| -> usize {
            let isl = context.isl_tokens();
            isl.saturating_sub(cached.min(isl))
                .saturating_add(output_tokens as usize)
        };
        let band_for =
            |size: usize| band_for_size(size, parameters.boundary_1, parameters.boundary_2);

        let band_count = if parameters.boundary_2 == 0 { 2 } else { 3 };
        let slices = band_slices(worker_count, parameters.small_band_share, band_count);

        // Cache only shrinks a request in a band whose own workers hold it, so a band qualifies
        // when the request still fits it after counting that band's best cache. Starting from the
        // smallest band the best cache anywhere could allow, take the first band that qualifies:
        // the request stays in the smallest band it genuinely fits, which keeps it out of longer
        // bands even when one of them holds a deeper prefix. The top band always qualifies.
        let smallest = band_for(size_with_cache(self.best_cached_in((0, worker_count))));
        let band = (smallest..band_count)
            .find(|&band| band_for(size_with_cache(self.best_cached_in(slices[band]))) <= band)
            .unwrap_or(band_count - 1);
        let best_cached_tokens = self.best_cached_in(slices[band]);
        let size = size_with_cache(best_cached_tokens);

        let mut ranges = apply_spill(
            band,
            band_count,
            &slices,
            &self.loads,
            parameters.spill_threshold,
        );
        ranges.push(widen_uncached_long_band(
            band,
            &slices,
            worker_count,
            best_cached_tokens,
            &self.loads,
            parameters.spill_threshold,
        ));
        let start = ranges.iter().map(|range| range.0).min().unwrap_or(0);
        let end = ranges
            .iter()
            .map(|range| range.1)
            .max()
            .unwrap_or(worker_count);
        tracing::debug!(
            request_id = context.request_id().unwrap_or("-"),
            size,
            band,
            min_worker_id = self.worker_ids[start],
            max_worker_id = self.worker_ids[end - 1],
            allowed_workers = (start..end)
                .filter(|index| ranges.iter().any(|&(s, e)| (s..e).contains(index)))
                .count(),
            "SITA restricted candidate workers"
        );
        Some(ranges)
    }
}

// ---------------------------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------------------------

/// The default cost's weights, resolved from `SelectionConfig` the way the default policy
/// resolves them, so the neutral scorer stays in lockstep with it.
#[derive(Debug, Clone, Copy)]
struct CostWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    decode_active_request_weight: f64,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    shared_cache_multiplier: f64,
}

impl CostWeights {
    /// Resolve each weight from the instance parameter when set, else from the router setting,
    /// exactly as `dynamo-default-cost-fn` does.
    fn new(config: &SelectionConfig, role: WorkerType, parameters: &Parameters) -> Self {
        // Plain disaggregated decode is load-only; conditional decode retains cache credit.
        if role == WorkerType::Decode && !config.conditional_disagg_enabled {
            return Self {
                overlap_score_credit: 0.0,
                host_cache_hit_weight: 0.0,
                disk_cache_hit_weight: 0.0,
                shared_cache_multiplier: 0.0,
                ..Self::new(config, WorkerType::Aggregated, parameters)
            };
        }
        let overlap_score_credit = parameters
            .overlap_score_credit
            .unwrap_or(config.overlap_score_credit);
        // Legacy load-only configs retain tier weights for host accounting only. Only explicit
        // parameters enable tier scoring independently of device credit.
        let inherited = |weight: f64| {
            if overlap_score_credit == 0.0 {
                0.0
            } else {
                weight
            }
        };
        Self {
            overlap_score_credit,
            overlap_score_credit_decay: parameters
                .overlap_score_credit_decay
                .unwrap_or(config.overlap_score_credit_decay),
            prefill_load_scale: parameters
                .prefill_load_scale
                .unwrap_or(config.prefill_load_scale),
            decode_active_request_weight: parameters
                .decode_active_request_weight
                .unwrap_or(config.decode_active_request_weight),
            host_cache_hit_weight: parameters
                .host_cache_hit_weight
                .unwrap_or(inherited(config.host_cache_hit_weight)),
            disk_cache_hit_weight: parameters
                .disk_cache_hit_weight
                .unwrap_or(inherited(config.disk_cache_hit_weight)),
            shared_cache_multiplier: parameters.shared_cache_multiplier.unwrap_or_else(|| {
                inherited(config.shared_cache_multiplier.unwrap_or(
                    match config.shared_cache_type {
                        SharedCacheType::None => 0.0,
                        SharedCacheType::Hicache => 0.5,
                    },
                ))
            }),
        }
    }
}

/// Values shared by every worker score in one selection.
struct PreparedRequest {
    min_prefill: usize,
    needs_decay: bool,
    needs_decode_subtraction: bool,
    /// Mean decode-side load across candidates; zero when the load curve is neutral.
    mean_decode_load: f64,
}

/// The default cost, with the overlap credit and decode-side load reshaped by the `score_*`
/// parameters.
struct ShapedCostScorer {
    weights: CostWeights,
    parameters: Parameters,
    is_decode: bool,
}

impl ShapedCostScorer {
    fn request_cost(&self, load: &WorkerLoadInput) -> f64 {
        if self.weights.decode_active_request_weight != 0.0 {
            self.weights.decode_active_request_weight * load.active_requests() as f64
        } else {
            // Preserve signed zero when the configured weight is -0.0.
            self.weights.decode_active_request_weight
        }
    }

    fn prepare(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
    ) -> Result<PreparedRequest, WorkerSelectionPolicyError> {
        let needs_decay =
            context.tracks_prefill_tokens() && self.weights.overlap_score_credit_decay > 0.0;
        let needs_mean = self.parameters.score_load_gamma != 1.0;
        let mut min_prefill = usize::MAX;
        let mut total_load = 0.0;
        for candidate in candidates.iter() {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            min_prefill = min_prefill.min(load.active_prefill_tokens());
            if needs_mean {
                total_load += load.decode_cost_blocks() + self.request_cost(load);
            }
        }
        Ok(PreparedRequest {
            min_prefill: if needs_decay { min_prefill } else { 0 },
            needs_decay,
            needs_decode_subtraction: self.is_decode && !context.tracks_prefill_tokens(),
            mean_decode_load: if needs_mean && !candidates.is_empty() {
                total_load / candidates.len() as f64
            } else {
                0.0
            },
        })
    }

    fn score_worker(
        &self,
        prepared: &PreparedRequest,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let parameters = &self.parameters;
        let weights = &self.weights;
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let block_size = f64::from(context.block_size());
        // An empty prompt has no overlap to decay; clamp so a floor worker does not compute 0/0.
        let request_blocks = context.request_blocks().max(1) as f64;
        let (cached_tokens, credit) = if let Some(cache) = candidate.cache() {
            let (estimated_overlap, cached_tokens) = cache.accounting_cache_estimate();
            let device = if cache.has_tier_matches() {
                cache.device_overlap_blocks()
            } else {
                estimated_overlap
            };
            let shared_credit = if weights.shared_cache_multiplier != 0.0 {
                // Host and disk blocks earn their own tier credit; the shared cache extends the
                // prefix from where the worker's ends.
                let shared = cache.shared_hits().map_or(0, |hits| {
                    let local = device + cache.host_overlap_blocks() + cache.disk_overlap_blocks();
                    hits.loadable_beyond(local.round().max(0.0) as u32)
                });
                weights.shared_cache_multiplier * shared as f64
            } else {
                weights.shared_cache_multiplier
            };
            let decay = if prepared.needs_decay {
                let excess = load
                    .active_prefill_tokens()
                    .saturating_sub(prepared.min_prefill) as f64
                    / block_size;
                1.0 / (1.0 + weights.overlap_score_credit_decay * (excess / request_blocks))
            } else {
                1.0
            };
            // Worker-local credit: the tiers that make up `effective_overlap_blocks`, and
            // therefore `overlap_frac`.
            let local_credit = weights.overlap_score_credit * decay * device
                + weights.host_cache_hit_weight * cache.host_overlap_blocks()
                + weights.disk_cache_hit_weight * cache.disk_overlap_blocks();
            // `overlap_frac^gamma * request_blocks` equals `overlap_frac^(gamma-1)` times the
            // linear credit, so shaping multiplicatively keeps host and disk on the same curve.
            // Shared-cache hits are outside `overlap_frac`, so neither the cutoff nor the curve
            // touches them. At the neutral knobs every factor is exactly 1.0 and the sum below
            // keeps the default's addition order, so the credit is bit-identical.
            let overlap_frac = (estimated_overlap / request_blocks).clamp(0.0, 1.0);
            let shaped_local = if overlap_frac < parameters.score_min_overlap_frac {
                0.0
            } else if parameters.score_overlap_gamma == 1.0 {
                local_credit
            } else if overlap_frac > 0.0 {
                overlap_frac.powf(parameters.score_overlap_gamma - 1.0) * local_credit
            } else {
                // `0^negative` is infinite; a zero-overlap worker earns no local credit.
                0.0
            };
            (
                cached_tokens,
                parameters.score_overlap_weight * (shaped_local + shared_credit),
            )
        } else {
            (0, 0.0)
        };

        let decode = load.decode_cost_blocks();
        let request_cost = self.request_cost(load);
        // Charge decode-side backlog convexly relative to the candidate mean, so a worker
        // already hotter than its peers pays more than its linear share. Computed before the
        // decode branch so both return paths honor it.
        let load_term = decode + request_cost;
        let load_scale = parameters.score_load_weight
            * if parameters.score_load_gamma == 1.0
                || prepared.mean_decode_load <= 0.0
                || load_term <= 0.0
            {
                1.0
            } else {
                (load_term / prepared.mean_decode_load).powf(parameters.score_load_gamma - 1.0)
            };

        let logit = if prepared.needs_decode_subtraction {
            // Multiplying by 1.0 is exact, so the neutral case matches the default.
            (load_scale * decode - credit).max(0.0) + load_scale * request_cost
        } else {
            let raw_tokens = if !context.tracks_prefill_tokens() {
                0
            } else if load.is_available() {
                let uncached = context.isl_tokens().saturating_sub(cached_tokens);
                (load.active_prefill_tokens() + uncached).saturating_add(cached_tokens)
            } else {
                context.isl_tokens()
            };
            let prefill = (raw_tokens as f64 / block_size - credit).max(0.0);
            if load_scale == 1.0 {
                // Keep the default's summation order: float addition is not associative.
                weights.prefill_load_scale * prefill + decode + request_cost
            } else {
                weights.prefill_load_scale * prefill + load_scale * load_term
            }
        };
        Ok(logit * candidate.preferred_taint_multiplier().unwrap_or(1.0))
    }
}

impl WorkerScorer for ShapedCostScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let prepared = self.prepare(context, candidates)?;
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = self.score_worker(&prepared, context, candidate)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------------------------

/// Restricts the rows to the request's SITA band, then picks the minimum cost among them with the
/// default's uniform tie-breaking, or samples them by temperature.
struct SitaPicker {
    parameters: Parameters,
    temperature: f64,
}

impl WorkerPicker for SitaPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        // A pinned request has exactly one row; SITA must not move it.
        if context.pinned_worker().is_some() {
            return Ok(0);
        }
        // Without KV state there is nothing to size the request by, so every row stays eligible.
        let pool = match (input.cache(), input.load()) {
            (Some(cache), Some(load)) => {
                Some(Pool::new(candidates, cache, load, context.block_size()))
            }
            _ => None,
        };
        let ranges = pool
            .as_ref()
            .and_then(|pool| pool.band_ranges(&self.parameters, context));
        let in_band = |row: &usize| match (&pool, &ranges) {
            (Some(pool), Some(ranges)) => {
                pool.contains(ranges, candidates[*row].worker().worker_id)
            }
            _ => true,
        };

        if self.temperature != 0.0 {
            let rows: Vec<usize> = (0..candidates.len()).filter(in_band).collect();
            if rows.is_empty() {
                return Err(WorkerSelectionPolicyError::failed(
                    "no eligible worker in band",
                ));
            }
            let mut probabilities = Vec::new();
            let index = softmax_sample_index(
                &rows,
                |&row| candidates[row].cost(),
                self.temperature,
                fastrand::f64(),
                &mut probabilities,
            );
            return Ok(rows[index]);
        }
        let mut best_row = None;
        let mut best_cost = f64::INFINITY;
        let mut ties = 0;
        for row in (0..candidates.len()).filter(in_band) {
            let cost = candidates[row].cost();
            if cost < best_cost {
                best_row = Some(row);
                best_cost = cost;
                ties = 1;
            } else if cost == best_cost {
                ties += 1;
                if fastrand::usize(0..ties) == 0 {
                    best_row = Some(row);
                }
            }
        }
        best_row.ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker in band"))
    }
}

// ---------------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------------

fn policy(
    config: &SelectionConfig,
    parameters: Parameters,
    role: WorkerType,
) -> WorkerSelectionPolicy {
    let worker_label = role.default_selector_label();
    WorkerSelectionPolicy::new(
        config.clone(),
        worker_label,
        vec![Box::new(ShapedCostScorer {
            weights: CostWeights::new(config, role, &parameters),
            parameters,
            is_decode: worker_label == "decode",
        })],
        Box::new(SitaPicker {
            parameters,
            temperature: parameters
                .router_temperature
                .unwrap_or(config.router_temperature),
        }),
    )
    .with_exclusive_affinity(true)
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &SelectionConfig, role, _partition| policy(config, parameters, role),
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_scheduler::host::{WorkerSelectionInput, WorkerSelector};
    use parking_lot::Mutex;

    use super::*;
    use crate::default::default_policy;

    const BLOCK_SIZE: u32 = 16;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1_000_000)
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            accounting_attribution: Default::default(),
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    fn set_cache(request: &mut SchedulingRequest, id: u64, cached_tokens: usize) {
        request
            .overlap
            .effective_cached_tokens
            .insert(worker(id), cached_tokens);
        request
            .overlap
            .effective_overlap_blocks
            .insert(worker(id), cached_tokens as f64 / f64::from(BLOCK_SIZE));
    }

    fn set_decode(request: &mut SchedulingRequest, id: u64, active_decode_blocks: usize) {
        request.worker_loads.insert(
            worker(id),
            WorkerLoadProjection {
                active_decode_blocks,
                ..Default::default()
            },
        );
    }

    /// A request with a tiny cached prefix on every worker of an 8-worker pool, so whichever band
    /// it lands in has cache affinity and the uncached widening does not mask band placement.
    fn cached_request(isl_tokens: usize) -> SchedulingRequest {
        let mut request = base_request(isl_tokens);
        for id in 0..8 {
            set_cache(&mut request, id, 64);
        }
        request
    }

    fn workers(count: u64) -> HashMap<u64, TestWorker> {
        (0..count).map(|id| (id, TestWorker)).collect()
    }

    /// Strict confinement: no spill, borrow, or uncached widening.
    fn strict() -> Parameters {
        Parameters {
            spill_threshold: 1.0,
            ..Parameters::default()
        }
    }

    fn select_on(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        request: &SchedulingRequest,
    ) -> u64 {
        policy
            .select_worker(WorkerSelectionInput::configured(
                workers,
                request,
                request.eligibility(),
                BLOCK_SIZE,
            ))
            .unwrap()
            .worker
            .worker_id
    }

    /// Route `request` across an 8-worker pool: band 0 = 0..4, band 1 = 4..6, band 2 = 6..8.
    fn pick(parameters: Parameters, request: &SchedulingRequest) -> u64 {
        select_on(
            &policy(
                &SelectionConfig::default(),
                parameters,
                WorkerType::Aggregated,
            ),
            &workers(8),
            request,
        )
    }

    /// Picks the lowest cost, and records every row's cost for inspection.
    struct RecordingPicker(Arc<Mutex<Vec<(WorkerWithDpRank, f64)>>>);

    impl WorkerPicker for RecordingPicker {
        fn pick(
            &self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let candidates = input.candidates();
            *self.0.lock() = candidates.iter().map(|c| (c.worker(), c.cost())).collect();
            Ok((0..candidates.len())
                .min_by(|&a, &b| candidates[a].cost().total_cmp(&candidates[b].cost()))
                .unwrap())
        }
    }

    /// Score `request` with the shaped scorer alone, returning `(chosen worker, cost of worker 0)`.
    fn score(
        config: &SelectionConfig,
        role: WorkerType,
        parameters: Parameters,
        request: &SchedulingRequest,
        worker_ids: &[u64],
    ) -> (u64, f64) {
        let costs = Arc::new(Mutex::new(Vec::new()));
        let label = role.default_selector_label();
        let policy = WorkerSelectionPolicy::new(
            config.clone(),
            label,
            vec![Box::new(ShapedCostScorer {
                weights: CostWeights::new(config, role, &parameters),
                parameters,
                is_decode: label == "decode",
            })],
            Box::new(RecordingPicker(costs.clone())),
        );
        let workers = worker_ids.iter().map(|&id| (id, TestWorker)).collect();
        let chosen = select_on(&policy, &workers, request);
        let cost = costs
            .lock()
            .iter()
            .find(|(w, _)| w.worker_id == worker_ids[0])
            .unwrap()
            .1;
        (chosen, cost)
    }

    #[test]
    fn registers_its_policy_type() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();
        assert!(register(&mut registry).is_err());
    }

    #[test]
    fn band_mapping_and_slices() {
        assert_eq!(band_for_size(0, 1024, 8192), 0);
        assert_eq!(band_for_size(1024, 1024, 8192), 0);
        assert_eq!(band_for_size(1025, 1024, 8192), 1);
        assert_eq!(band_for_size(8193, 1024, 8192), 2);
        assert_eq!(band_for_size(100_000, 1024, 0), 1);

        assert_eq!(band_slices(8, 0.5, 3), [(0, 4), (4, 6), (6, 8)]);
        assert_eq!(band_slices(8, 0.5, 2), [(0, 4), (4, 8), (4, 8)]);
        // Every slice stays non-empty, however extreme the share.
        for count in 2..20 {
            for share in [0.01, 0.5, 0.99] {
                for (start, end) in band_slices(count, share, 3) {
                    assert!(start < end && end <= count, "{count} {share}");
                }
            }
        }
    }

    #[test]
    fn neutral_scorer_reproduces_the_default_choice() {
        // Cases where cache overlap and decode footprint disagree, so drift in the re-implemented
        // cost would show up as a different worker.
        for [(a, a_cached, a_decode), (b, b_cached, b_decode)] in [
            [(29, 0, 0), (41, 128, 40)],
            [(29, 96, 30), (41, 0, 0)],
            [(29, 160, 100), (41, 32, 5)],
            [(29, 48, 12), (41, 64, 20)],
            [(29, 0, 7), (41, 80, 9)],
        ] {
            let mut request = base_request(160);
            set_cache(&mut request, a, a_cached);
            set_cache(&mut request, b, b_cached);
            set_decode(&mut request, a, a_decode);
            set_decode(&mut request, b, b_decode);
            let workers = HashMap::from([(a, TestWorker), (b, TestWorker)]);
            // Stock weights, then cost overrides: a weight set as a `dynamo-sita` parameter must
            // rank workers as the same weight does on the default policy.
            for (credit, prefill_scale) in [(None, None), (Some(0.0), Some(2.0)), (Some(2.5), None)]
            {
                let default_config = SelectionConfig {
                    overlap_score_credit: credit
                        .unwrap_or(SelectionConfig::default().overlap_score_credit),
                    prefill_load_scale: prefill_scale
                        .unwrap_or(SelectionConfig::default().prefill_load_scale),
                    ..SelectionConfig::default()
                };
                let stock = select_on(&default_policy(default_config, "test"), &workers, &request);
                let parameters = Parameters {
                    overlap_score_credit: credit,
                    prefill_load_scale: prefill_scale,
                    ..Parameters::default()
                };
                let (shaped, _) = score(
                    &SelectionConfig::default(),
                    WorkerType::Aggregated,
                    parameters,
                    &request,
                    &[a, b],
                );
                assert_eq!(
                    shaped, stock,
                    "diverged from the default on {a}/{b} with credit {credit:?}, \
                     prefill scale {prefill_scale:?}"
                );
            }
        }
    }

    #[test]
    fn short_and_long_requests_use_disjoint_bands() {
        assert!(pick(strict(), &cached_request(256)) < 4);
        assert!(pick(strict(), &cached_request(4_096)) >= 4);
        assert!(pick(strict(), &cached_request(16_384)) >= 6);
    }

    #[test]
    fn cache_shrinks_size_only_when_the_band_holds_it() {
        // 1,800 cached tokens on worker 0 leave a band-0 prefill, and worker 0 is in band 0.
        let mut in_band = base_request(2_048);
        set_cache(&mut in_band, 0, 1_800);
        assert!(pick(strict(), &in_band) < 4);

        // The same cache on worker 7 is outside band 0, so band 0's workers would prefill the
        // request cold: it must not be sent there.
        let mut out_of_band = base_request(2_048);
        set_cache(&mut out_of_band, 7, 1_800);
        let chosen = pick(strict(), &out_of_band);
        assert!(chosen >= 4, "routed to cold band-0 worker {chosen}");
    }

    /// A pool of eight workers with the given per-worker loads and cached tokens.
    fn pool(loads: [f64; 8], cached_tokens: [usize; 8]) -> Pool {
        Pool {
            worker_ids: (0..8).collect(),
            loads: loads.to_vec(),
            cached_tokens: cached_tokens.to_vec(),
        }
    }

    fn allowed(pool: &Pool, parameters: Parameters, isl_tokens: usize) -> Vec<u64> {
        let context = WorkerSelectionContext::new(isl_tokens, BLOCK_SIZE);
        let ranges = pool.band_ranges(&parameters, &context).unwrap();
        (0..8).filter(|&id| pool.contains(&ranges, id)).collect()
    }

    #[test]
    fn request_goes_to_the_smallest_band_its_own_cache_fits() {
        // 10,000 tokens. Band 0 holds nothing (cold size 10,000 maps to band 2). Band 1's worker 4
        // holds 2,000 (size 8,000 fits band 1). Band 2's worker 6 holds 9,500. The first band that
        // fits is band 1, so the request must not skip it for band 2's deeper hit.
        let pool = pool([0.0; 8], [0, 0, 0, 0, 2_000, 0, 9_500, 0]);
        assert_eq!(allowed(&pool, strict(), 10_000), [4, 5]);

        // Without band 1's cache, the request falls through to band 2.
        let pool = super::Pool {
            cached_tokens: vec![0, 0, 0, 0, 0, 0, 9_500, 0],
            ..pool
        };
        assert_eq!(allowed(&pool, strict(), 10_000), [6, 7]);
    }

    #[test]
    fn borrowing_band_0_does_not_readmit_a_rejected_middle_band() {
        // share 0.75 over 8 workers => band 0 = 0..6, band 1 = 6..7, band 2 = 7..8. Band 2
        // occupancy = 950 / (950 + 1000/7) = 0.87 > 0.85, so it seeks relief. Band 1 is busier
        // (1,000), so spill rejects it; band 0 is idle, so it is borrowed as a separate range.
        let mut loads = [0.0; 8];
        loads[6] = 1_000.0;
        loads[7] = 950.0;
        let pool = pool(loads, [64; 8]);
        let parameters = Parameters {
            small_band_share: 0.75,
            ..Parameters::default()
        };
        assert_eq!(allowed(&pool, parameters, 9_000), [0, 1, 2, 3, 4, 5, 7]);
    }

    #[test]
    fn uncached_long_request_widens_beyond_its_band() {
        // Uncached widening is intentionally kept at `spill_threshold: 1.0` (`strict()`): an
        // uncached long request always gets the whole non-short pool, and only band 0 stays
        // reserved.
        // Saturate the top band; spilling is off, so only the uncached widening can move it.
        let saturate = |request: &mut SchedulingRequest| {
            for id in 6..8 {
                set_decode(request, id, 900);
            }
        };
        let mut uncached = base_request(16_384);
        saturate(&mut uncached);
        let mut cached = cached_request(16_384);
        saturate(&mut cached);

        assert!((4..6).contains(&pick(strict(), &uncached)));
        assert!(pick(strict(), &cached) >= 6);
    }

    #[test]
    fn unsaturated_long_band_does_not_borrow_idle_band_0() {
        // Band 0 idle; band 2 occupancy = 100 / (100 + 200/6) = 0.75, under 0.85.
        let mut request = cached_request(9_000);
        for id in 4..8 {
            set_decode(&mut request, id, 100);
        }
        let chosen = pick(Parameters::default(), &request);
        assert!(
            chosen >= 6,
            "an unsaturated top band borrowed worker {chosen}"
        );
    }

    #[test]
    fn saturated_long_band_borrows_band_0_only_while_it_is_idle() {
        // Band 1 at 300, band 2 at 950. With band 0 idle, band 2 occupancy = 950 / (950 + 100)
        // = 0.90 > 0.85: it spills into band 1 and borrows band 0, whose workers are cheapest.
        // With band 0 at 950, band 2 occupancy = 950 / (950 + 733) = 0.56: no relief needed.
        let saturate_long = |band_0_load: usize| {
            let mut request = cached_request(9_000);
            for id in 0..8 {
                let load = match id {
                    0..=3 => band_0_load,
                    4..=5 => 300,
                    _ => 950,
                };
                set_decode(&mut request, id, load);
            }
            pick(Parameters::default(), &request)
        };
        assert!(saturate_long(0) < 4, "an idle band 0 should be lent out");
        assert!(saturate_long(950) >= 4, "a busy band 0 stays reserved");
    }

    #[test]
    fn top_band_spills_down_but_never_into_band_0() {
        // Band 2 occupancy = 950 / (950 + 100) = 0.90 > 0.85, so it spills; band 0 occupancy
        // = 150 / (150 + 475) = 0.24 >= 0.15, so it is not lendable.
        let mut request = cached_request(9_000);
        for id in 0..8 {
            let load = match id {
                0..=3 => 150,
                4..=5 => 0,
                _ => 950,
            };
            set_decode(&mut request, id, load);
        }
        assert!(pick(strict(), &request) >= 6);
        assert!((4..6).contains(&pick(Parameters::default(), &request)));
    }

    #[test]
    fn respects_pinned_workers_and_tiny_pools() {
        let mut pinned = base_request(256);
        pinned.pinned_worker = Some(worker(7));
        assert_eq!(pick(strict(), &pinned), 7);

        let single = policy(
            &SelectionConfig::default(),
            strict(),
            WorkerType::Aggregated,
        );
        assert_eq!(select_on(&single, &workers(1), &base_request(16_384)), 0);
    }

    #[test]
    fn osl_weight_promotes_long_output_requests() {
        let mut request = cached_request(512);
        request.expected_output_tokens = Some(4_096);
        assert!(pick(strict(), &request) < 4);
        let weighted = Parameters {
            osl_weight: 1.0,
            ..strict()
        };
        assert!(pick(weighted, &request) >= 4);
    }

    #[test]
    fn min_overlap_cutoff_drops_thin_local_matches() {
        // 2 of 10 request blocks cached: 0.2.
        let mut request = base_request(160);
        set_cache(&mut request, 0, 32);
        let config = SelectionConfig::default();
        let cost = |min_overlap_frac| {
            let parameters = Parameters {
                score_min_overlap_frac: min_overlap_frac,
                ..Parameters::default()
            };
            score(&config, WorkerType::Aggregated, parameters, &request, &[0]).1
        };
        assert_eq!(cost(0.1), cost(0.0));
        assert!(
            cost(0.3) > cost(0.0),
            "cutting the credit must raise the cost"
        );
    }

    #[test]
    fn overlap_shaping_keeps_shared_cache_credit() {
        // No local overlap, four loadable shared-cache blocks.
        let mut request = base_request(64);
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..4]);
        request.shared_cache_hits = Some(hits);
        let config = |multiplier| SelectionConfig {
            shared_cache_multiplier: Some(multiplier),
            ..SelectionConfig::default()
        };
        let cost = |config: &SelectionConfig, parameters| {
            score(config, WorkerType::Aggregated, parameters, &request, &[0]).1
        };

        let stock = cost(&config(0.5), Parameters::default());
        assert!(
            stock < cost(&config(0.0), Parameters::default()),
            "the fixture must earn shared-cache credit"
        );
        for parameters in [
            Parameters {
                score_min_overlap_frac: 0.5,
                ..Parameters::default()
            },
            Parameters {
                score_overlap_gamma: 2.0,
                ..Parameters::default()
            },
        ] {
            assert_eq!(cost(&config(0.5), parameters), stock, "{parameters:?}");
        }
    }

    #[test]
    fn load_shaping_applies_to_the_decode_overlap_branch() {
        // Conditional disaggregation keeps cache credit on decode workers.
        let config = SelectionConfig {
            conditional_disagg_enabled: true,
            ..SelectionConfig::default()
        };
        let mut request = base_request(64);
        request.track_prefill_tokens = false;
        set_cache(&mut request, 0, 48);
        set_decode(&mut request, 0, 10);
        let cost = |load_weight| {
            let parameters = Parameters {
                score_load_weight: load_weight,
                ..Parameters::default()
            };
            score(&config, WorkerType::Decode, parameters, &request, &[0]).1
        };
        assert!(
            cost(2.0) > cost(1.0),
            "load weight ignored on the decode branch"
        );
    }

    #[test]
    fn convex_load_charges_a_hot_worker_more() {
        let mut request = base_request(160);
        set_decode(&mut request, 0, 200);
        let config = SelectionConfig::default();
        let cost = |gamma| {
            let parameters = Parameters {
                score_load_gamma: gamma,
                ..Parameters::default()
            };
            score(
                &config,
                WorkerType::Aggregated,
                parameters,
                &request,
                &[0, 1],
            )
            .1
        };
        let costs: Vec<f64> = [1.0, 1.5, 2.0].into_iter().map(cost).collect();
        assert!(costs.windows(2).all(|pair| pair[1] > pair[0]), "{costs:?}");
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let check = |f: fn(&mut Parameters)| {
            let mut parameters = Parameters::default();
            f(&mut parameters);
            parameters.validate()
        };
        assert!(Parameters::default().validate().is_ok());
        assert!(check(|p| p.boundary_1 = 0).is_err());
        assert!(check(|p| p.boundary_2 = 512).is_err());
        assert!(check(|p| p.boundary_2 = 0).is_ok());
        assert!(check(|p| p.small_band_share = 1.0).is_err());
        assert!(check(|p| p.osl_weight = -1.0).is_err());
        assert!(check(|p| p.spill_threshold = 0.4).is_err());
        assert!(check(|p| p.score_overlap_gamma = f64::NAN).is_err());
        assert!(check(|p| p.score_load_gamma = 0.5).is_err());
        assert!(check(|p| p.score_min_overlap_frac = 0.6).is_err());
    }
}
