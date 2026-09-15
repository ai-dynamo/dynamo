// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::collections::HashMap;
#[cfg(any(test, feature = "bench"))]
use std::sync::Arc;

use parking_lot::Mutex;

use super::policy::WorkerSelectionPolicyStateRef;
use super::{
    LogitWeights, MaterializedSelectionInput, WorkerCandidate, WorkerInputs,
    WorkerSelectionContext, WorkerSelectionInput, WorkerSelector, select_worker_with_policy,
};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};
use crate::scheduling::config::KvRouterConfig;
use crate::scheduling::filter::RoutingEligibility;
use crate::scheduling::types::{KvSchedulerError, SchedulingRequest};

#[cfg(any(test, feature = "bench"))]
fn softmax_sample_entries<T: Copy>(
    entries: Vec<(T, f64)>,
    temperature: f64,
    sample: f64,
) -> (T, f64) {
    assert!(!entries.is_empty(), "Empty logits for softmax sampling");

    let mut probabilities = Vec::with_capacity(entries.len());
    let row = softmax_sample_index(
        &entries,
        |(_, cost)| *cost,
        temperature,
        sample,
        &mut probabilities,
    );
    entries[row]
}

fn softmax_sample_index<T>(
    entries: &[T],
    cost: impl Fn(&T) -> f64,
    temperature: f64,
    sample: f64,
    probabilities: &mut Vec<f64>,
) -> usize {
    assert!(!entries.is_empty(), "Empty entries for softmax sampling");
    debug_assert_ne!(temperature, 0.0);

    let (min_cost, max_cost) = entries
        .iter()
        .map(&cost)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(cost), hi.max(cost))
        });

    probabilities.clear();
    if min_cost == max_cost {
        probabilities.resize(entries.len(), 1.0 / entries.len() as f64);
    } else {
        let range = max_cost - min_cost;
        let magnitude = if range.is_finite() {
            1.0
        } else {
            min_cost.abs().max(max_cost.abs())
        };
        let min_normalized = min_cost / magnitude;
        let scale = -1.0 / ((max_cost / magnitude - min_normalized) * temperature);
        let max_scaled = min_normalized * scale;
        probabilities.extend(
            entries
                .iter()
                .map(|entry| (cost(entry) / magnitude * scale - max_scaled).exp()),
        );
    }

    let sum: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= sum;
    }
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if sample <= cumulative {
            return row;
        }
    }
    entries.len() - 1
}

/// Half-open index range into the sorted eligible worker-id list.
type SitaSlice = (usize, usize);

/// Map an estimated request size in tokens to a SITA band index.
///
/// `boundary_2 == 0` collapses the policy to a two-band split at `boundary_1`.
pub(super) fn sita_band_for_size(size: usize, boundary_1: usize, boundary_2: usize) -> usize {
    if size <= boundary_1 {
        return 0;
    }
    if boundary_2 == 0 || size <= boundary_2 {
        return 1;
    }
    2
}

/// Number of bands implied by the configured boundaries.
fn sita_band_count(boundary_2: usize) -> usize {
    if boundary_2 == 0 { 2 } else { 3 }
}

/// Partition `worker_count` workers into contiguous per-band slices.
///
/// Band 0 receives `ceil(small_band_share * worker_count)` workers. The
/// remaining workers are split evenly across the longer bands. Every returned
/// slice is non-empty, so a band always has somewhere to route.
pub(super) fn sita_band_slices(
    worker_count: usize,
    small_band_share: f64,
    band_count: usize,
) -> [SitaSlice; 3] {
    debug_assert!(worker_count >= 2);
    let band_0 =
        ((small_band_share * worker_count as f64).ceil() as usize).clamp(1, worker_count - 1);
    let remaining = worker_count - band_0;
    if band_count < 3 || remaining == 1 {
        let tail = (band_0, worker_count);
        return [(0, band_0), tail, tail];
    }
    // Keep at least one worker in the largest band so huge requests never
    // share the whole tail with medium ones.
    let band_1 = remaining.div_ceil(2).clamp(1, remaining - 1);
    [
        (0, band_0),
        (band_0, band_0 + band_1),
        (band_0 + band_1, worker_count),
    ]
}

/// Per-worker queued work, in tokens, for the workers in `worker_ids`.
///
/// Queued prefill dominates TTFT, so it is counted directly; resident decode
/// blocks are converted to tokens so both contribute in the same unit.
fn sita_worker_loads<C: WorkerConfigLike>(
    worker_ids: &[WorkerId],
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    block_size: usize,
) -> Vec<f64> {
    worker_ids
        .iter()
        .map(|&worker_id| {
            let Some(config) = workers.get(&worker_id) else {
                return 0.0;
            };
            let dp_start = config.data_parallel_start_rank();
            let dp_size = config.data_parallel_size().max(1);
            let mut total = 0.0;
            for dp_rank in dp_start..dp_start + dp_size {
                let load = request.worker_load_for(WorkerWithDpRank::new(worker_id, dp_rank));
                total += load.active_prefill_tokens as f64
                    + (load.potential_decode_blocks() * block_size) as f64;
            }
            total / dp_size as f64
        })
        .collect()
}

/// Mean queued work per worker across `slice`.
fn sita_slice_mean(loads: &[f64], slice: SitaSlice) -> f64 {
    let (start, end) = slice;
    if end <= start {
        return 0.0;
    }
    loads[start..end].iter().sum::<f64>() / (end - start) as f64
}

/// Share of the pool's per-worker load that sits in `slice`, in `[0, 1]`.
///
/// An evenly loaded pool scores 0.5, and the score rises toward 1.0 as the
/// band's workers get busier than everyone else's. Expressing occupancy
/// relatively is what makes `sita_spill_threshold`'s [0.5, 1.0] range
/// meaningful: an absolute KV-capacity fraction is a few percent under any
/// realistic serving load, so no threshold in that range would ever trip.
fn sita_band_occupancy(loads: &[f64], slice: SitaSlice) -> f64 {
    let (start, end) = slice;
    let inside = sita_slice_mean(loads, slice);
    let outside_count = loads.len() - (end - start);
    if outside_count == 0 {
        return 0.5;
    }
    let outside_total = loads.iter().sum::<f64>() - loads[start..end].iter().sum::<f64>();
    let outside = outside_total / outside_count as f64;
    let denominator = inside + outside;
    if denominator <= 0.0 {
        return 0.5;
    }
    inside / denominator
}

/// Widen `band` into an adjacent band when the target holds more than
/// `spill_threshold` of the pool's load and that neighbor is genuinely quieter.
///
/// Which neighbor is allowed is asymmetric, because the two directions have very
/// different costs. Sending a request *up* into a longer band costs roughly its
/// own service time. Sending a long request *down* parks a multi-thousand-token
/// prefill in front of everything queued behind it — the head-of-line blocking
/// SITA exists to prevent. So band 0 is not a spill target merely because it is
/// the nearest neighbor: the short band's isolation is the entire source of the
/// mean-TTFT win. See `sita_borrow_idle_short_band` for the one exception.
///
/// Every other band still needs a relief valve. Without one the largest band is
/// a saturation sink — it receives spill from below and can never shed it — and
/// the resulting queue on its few workers wrecks tail latency (p99 end-to-end
/// blows up even as the mean improves). The top band may therefore widen
/// downward into band 1, which holds medium requests.
fn sita_apply_spill(
    band: usize,
    band_count: usize,
    slices: &[SitaSlice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> SitaSlice {
    let target = slices[band];
    if spill_threshold >= 1.0 {
        return target;
    }

    let widened = if sita_band_occupancy(loads, target) > spill_threshold {
        // Prefer the band above; the top band falls back to the one below, which
        // is band 1 (never band 0, since reaching here needs `band >= 2`).
        let neighbor_band = if band + 1 < band_count {
            Some(band + 1)
        } else if band >= 2 {
            Some(band - 1)
        } else {
            None
        };
        match neighbor_band.map(|next| slices[next]) {
            Some(neighbor)
                if neighbor != target
                    && sita_slice_mean(loads, neighbor) < sita_slice_mean(loads, target) =>
            {
                (target.0.min(neighbor.0), target.1.max(neighbor.1))
            }
            _ => target,
        }
    } else {
        target
    };

    sita_borrow_idle_short_band(widened, slices, loads, spill_threshold)
}

/// Last-resort valve: let a saturated long band borrow band 0's workers, but
/// only while band 0 is measurably *idle*.
///
/// Reserving workers for short requests is what makes SITA work, but the
/// reservation is only free when the short band is actually using them. A
/// statically fenced-off band 0 starves the long bands of KV capacity: the huge
/// requests that must share the remaining workers pile up their multi-thousand
/// token contexts on a fraction of the pool, and end-to-end tail latency blows
/// up even though their time-to-first-token is unchanged. Lending idle short-band
/// workers hands that capacity back exactly when doing so costs nothing.
///
/// `spill_threshold` sets both directions of the dial: a band is "saturated"
/// above it and "idle" below its complement, so a high threshold means both a
/// reluctance to spill and a strict definition of idle.
fn sita_borrow_idle_short_band(
    target: SitaSlice,
    slices: &[SitaSlice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> SitaSlice {
    let short = slices[0];
    if target.0 <= short.0 {
        return target;
    }
    if sita_band_occupancy(loads, short) >= 1.0 - spill_threshold {
        return target;
    }
    (short.0, target.1)
}

/// Widen a long request's band to every non-short worker when it has no cached
/// prefix to come back to.
///
/// Band confinement pays for itself through *cache affinity*: repeatedly sending
/// same-sized requests to the same few workers concentrates their shared
/// prefixes there, so the blocks are still resident on the next hit. That is a
/// real effect for the short and medium bands, whose requests share prompt
/// templates and corpus chunks.
///
/// A request that matched nothing in any worker's cache has no affinity to
/// preserve — it will prefill from scratch wherever it lands. Confining it buys
/// nothing and costs plenty: the long band's requests are exactly the ones with
/// multi-thousand-token contexts and the longest decodes, so pinning them to a
/// slice of the pool multiplies the resident KV and decode-batch contention on
/// those workers. That shows up as inter-token latency on the requests that
/// already have the worst end-to-end latency, which is what wrecks the tail.
///
/// So a zero-overlap request always gets the whole non-short pool, and it may
/// additionally reach band 0 while band 0 is idle.
///
/// That last part matters because the short band is systematically the *least*
/// contended slice of the pool: its requests are short in output as well as
/// input, so they retire quickly and leave decode capacity free on their
/// workers. Meanwhile the uncached long requests — the ones with multi-thousand
/// token contexts and the longest decodes — are packed onto the remaining
/// workers, where their inter-token latency (not their time-to-first-token)
/// sets end-to-end tail latency.
///
/// `sita_borrow_idle_short_band` already lends band 0 out, but it applies one
/// idleness bar to every borrower, and at useful spill thresholds that bar is
/// so strict it effectively never clears. A request with no cached prefix is the
/// cheapest possible borrower: it has no affinity to any worker, so lending it a
/// band-0 worker costs the short band only the load it brings, never a lost
/// cache hit. It therefore gets a proportionally larger idleness allowance. As
/// with every other spill rule here `sita_spill_threshold` sets the dial, and at
/// 1.0 band 0 is never lent out and this reduces to plain confinement.
fn sita_widen_uncached_long_band(
    band: usize,
    slices: &[SitaSlice; 3],
    worker_count: usize,
    best_cached_tokens: usize,
    loads: &[f64],
    spill_threshold: f64,
) -> SitaSlice {
    if band == 0 || best_cached_tokens > 0 {
        return slices[band];
    }
    let idle_allowance = (1.0 - spill_threshold) * 2.0;
    if sita_band_occupancy(loads, slices[0]) < idle_allowance {
        return (0, worker_count);
    }
    (slices[0].1, worker_count)
}

/// Inclusive worker-id bounds of the SITA band this request may route to.
///
/// Returns `None` whenever SITA must not change routing: the knob is off, the
/// pool is too small to partition, or the request is pinned.
fn sita_worker_id_bounds<C: WorkerConfigLike>(
    kv_router_config: &KvRouterConfig,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    block_size: usize,
) -> Option<(WorkerId, WorkerId)> {
    if !kv_router_config.sita_enabled || eligibility.pinned_worker().is_some() {
        return None;
    }

    let mut worker_ids: Vec<WorkerId> = workers
        .iter()
        .filter(|(worker_id, config)| eligibility.allows_worker(**worker_id, *config))
        .map(|(worker_id, _)| *worker_id)
        .collect();
    if worker_ids.len() < 2 {
        return None;
    }
    worker_ids.sort_unstable();

    // Reuse the scoring path's notion of cache credit: the best overlap any
    // eligible worker offers is the prefill this request can actually skip.
    let mut best_cached_tokens = 0;
    eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
        best_cached_tokens = best_cached_tokens.max(request.effective_cached_tokens_for(worker));
    });
    let effective_prefill_tokens = crate::scheduling::prefill_load::effective_prefill_tokens(
        request.isl_tokens,
        best_cached_tokens,
    );
    let output_tokens = request
        .expected_output_tokens
        .map_or(0.0, |tokens| kv_router_config.sita_osl_weight * tokens as f64);
    let size = effective_prefill_tokens.saturating_add(output_tokens as usize);

    let band_count = sita_band_count(kv_router_config.sita_boundary_2);
    let band = sita_band_for_size(
        size,
        kv_router_config.sita_boundary_1,
        kv_router_config.sita_boundary_2,
    );
    let slices = sita_band_slices(
        worker_ids.len(),
        kv_router_config.sita_small_band_share,
        band_count,
    );
    let loads = sita_worker_loads(&worker_ids, workers, request, block_size);
    let spilled = sita_apply_spill(
        band,
        band_count,
        &slices,
        &loads,
        kv_router_config.sita_spill_threshold,
    );
    // A request with no cache to return to gains nothing from confinement, so
    // prefer the widest of the two candidate slices.
    let widened = sita_widen_uncached_long_band(
        band,
        &slices,
        worker_ids.len(),
        best_cached_tokens,
        &loads,
        kv_router_config.sita_spill_threshold,
    );
    let (start, end) = if widened.1 - widened.0 > spilled.1 - spilled.0 {
        widened
    } else {
        spilled
    };

    tracing::debug!(
        request_id = request.mode.request_id().unwrap_or("-"),
        size,
        band,
        "SITA band restricted routing to worker ids [{}, {}]",
        worker_ids[start],
        worker_ids[end - 1],
    );
    Some((worker_ids[start], worker_ids[end - 1]))
}

/// Default implementation matching the Python _cost_function.
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
    picker: DefaultWorkerPicker,
}

pub(super) struct DefaultWorkerScorer<C = KvRouterConfig> {
    pub(super) kv_router_config: C,
    pub(super) worker_type: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct DefaultScoringContext {
    min_active_prefill_tokens: usize,
    has_tier_overlap_blocks: bool,
    /// Mean decode-side load across eligible workers, used as the reference
    /// point for `score_load_gamma`. Zero whenever the shaping is inactive, in
    /// which case `worker_logit` never reads it.
    mean_pool_load: f64,
}

pub(super) struct DefaultWorkerPicker {
    // Preserve DefaultWorkerSelector's Sync contract. Zero-temperature selection never locks.
    softmax_scratch: Mutex<DefaultSoftmaxScratch>,
    #[cfg(any(test, feature = "bench"))]
    deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

#[derive(Default)]
struct DefaultSoftmaxScratch {
    entries: Vec<(WorkerWithDpRank, f64)>,
    probabilities: Vec<f64>,
}

impl std::fmt::Debug for DefaultWorkerSelector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DefaultWorkerSelector")
            .field("kv_router_config", &self.kv_router_config)
            .field("worker_type", &self.worker_type)
            .finish_non_exhaustive()
    }
}

impl Clone for DefaultWorkerSelector {
    fn clone(&self) -> Self {
        #[cfg(any(test, feature = "bench"))]
        let deterministic_rng = self.picker.deterministic_rng.clone();
        Self::from_parts(
            self.kv_router_config.clone(),
            self.worker_type,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        )
    }
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self::from_parts(
            kv_router_config.unwrap_or_default(),
            worker_type,
            #[cfg(any(test, feature = "bench"))]
            None,
        )
    }

    #[cfg(any(test, feature = "bench"))]
    pub fn new_seeded(
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self::from_parts(
            kv_router_config.unwrap_or_default(),
            worker_type,
            Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        )
    }

    fn from_parts(
        kv_router_config: KvRouterConfig,
        worker_type: &'static str,
        #[cfg(any(test, feature = "bench"))] deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        let picker = DefaultWorkerPicker::from_parts(
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        );
        Self {
            kv_router_config,
            worker_type,
            picker,
        }
    }
}

#[cfg(test)]
impl DefaultWorkerScorer<KvRouterConfig> {
    pub(super) fn new(kv_router_config: KvRouterConfig, worker_type: &'static str) -> Self {
        Self {
            kv_router_config,
            worker_type,
        }
    }
}

pub(super) fn selection_weights(
    kv_router_config: &KvRouterConfig,
    request: &SchedulingRequest,
) -> LogitWeights {
    LogitWeights {
        overlap_score_credit: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.overlap_score_credit)
            .unwrap_or(kv_router_config.overlap_score_credit),
        overlap_score_credit_decay: kv_router_config.overlap_score_credit_decay,
        prefill_load_scale: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.prefill_load_scale)
            .unwrap_or(kv_router_config.prefill_load_scale),
        shared_cache_multiplier: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.shared_cache_multiplier)
            .unwrap_or(kv_router_config.shared_cache_multiplier),
    }
}

impl DefaultScoringContext {
    fn new<C: WorkerConfigLike>(
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        weights: LogitWeights,
        kv_router_config: &KvRouterConfig,
    ) -> Self {
        // Only the convex branch of the load shaping needs a pool reference
        // point, so skip the extra pass in every other configuration.
        let mean_pool_load = if kv_router_config.score_enabled
            && kv_router_config.score_load_gamma != 1.0
        {
            let mut total = 0.0;
            let mut count = 0usize;
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                let load = request.worker_load_for(worker);
                total += load.potential_decode_blocks() as f64
                    + kv_router_config.decode_active_request_weight * load.active_requests as f64;
                count += 1;
            });
            if count == 0 {
                0.0
            } else {
                total / count as f64
            }
        } else {
            0.0
        };
        let min_active_prefill_tokens =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let mut minimum = usize::MAX;
                eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                    minimum = minimum.min(request.worker_load_for(worker).active_prefill_tokens);
                });
                if minimum == usize::MAX { 0 } else { minimum }
            } else {
                0
            };
        let has_tier_overlap_blocks = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        Self {
            min_active_prefill_tokens,
            has_tier_overlap_blocks,
            mean_pool_load,
        }
    }

    fn device_overlap(self, effective_overlap_blocks: f64, device_overlap_blocks: f64) -> f64 {
        if self.has_tier_overlap_blocks {
            device_overlap_blocks
        } else {
            effective_overlap_blocks
        }
    }
}

fn default_row(
    input: &MaterializedSelectionInput<'_>,
    context: DefaultScoringContext,
    worker: WorkerWithDpRank,
    preferred_taint_multiplier: Option<f64>,
) -> WorkerCandidate {
    input.row_with_device_overlap(
        worker,
        preferred_taint_multiplier,
        WorkerInputs::ALL,
        |effective_overlap_blocks, device_overlap_blocks| {
            context.device_overlap(effective_overlap_blocks, device_overlap_blocks)
        },
    )
}

impl<C: Borrow<KvRouterConfig>> DefaultWorkerScorer<C> {
    fn worker_logit(
        &self,
        context: &WorkerSelectionContext<'_>,
        default_context: DefaultScoringContext,
        row: &WorkerCandidate,
        formula_name: &'static str,
    ) -> f64 {
        let kv_router_config = self.kv_router_config.borrow();
        let weights = context.weights;
        let worker = row.worker;
        let cache = &row.cache;
        let load = &row.load;
        let effective_overlap_blocks = cache.effective_overlap_blocks;
        let device_overlap_blocks = cache.device_overlap_blocks;
        let shared_beyond_device_blocks = cache.shared_beyond_device_blocks;
        let shared_overlap_blocks =
            weights.shared_cache_multiplier * shared_beyond_device_blocks as f64;
        // Normalize backlog above the least-loaded eligible worker by this request's
        // size. The rational decay softly trades cache locality for prefill balance,
        // while leaving workers at the load floor with their full device credit.
        let overlap_credit_decay =
            if context.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let excess_active_prefill_blocks = load
                    .active_prefill_tokens
                    .saturating_sub(default_context.min_active_prefill_tokens)
                    as f64
                    / context.block_size as f64;
                let normalized_prefill_load =
                    excess_active_prefill_blocks / context.request_blocks as f64;
                1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
            } else {
                1.0
            };
        let effective_overlap_score_credit = weights.overlap_score_credit * overlap_credit_decay;
        let stock_overlap_credit_blocks = effective_overlap_score_credit * device_overlap_blocks
            + kv_router_config.host_cache_hit_weight * cache.host_overlap_blocks
            + kv_router_config.disk_cache_hit_weight * cache.disk_overlap_blocks
            + shared_overlap_blocks;
        // How much of this request's prompt the worker already holds. The stock
        // credit is linear in the absolute overlap; expressing it as a fraction
        // of the request lets `score_overlap_gamma` bend that response curve and
        // `score_min_overlap_frac` drop matches too thin to be worth chasing.
        let overlap_frac = if context.request_blocks == 0 {
            0.0
        } else {
            (effective_overlap_blocks / context.request_blocks as f64).clamp(0.0, 1.0)
        };
        // `overlap_frac^gamma * request_blocks` from the contract equals
        // `overlap_frac^(gamma-1)` times the linear credit, so shaping the stock
        // credit multiplicatively reproduces it while keeping the host, disk and
        // shared tiers on the same curve. At the neutral knobs the factor is
        // exactly 1.0 and this reduces to the stock credit bit for bit.
        let overlap_credit_blocks = if kv_router_config.score_enabled {
            if overlap_frac < kv_router_config.score_min_overlap_frac {
                0.0
            } else {
                let shape = if kv_router_config.score_overlap_gamma == 1.0 {
                    1.0
                } else if overlap_frac > 0.0 {
                    overlap_frac.powf(kv_router_config.score_overlap_gamma - 1.0)
                } else {
                    // `0^negative` is infinite; a zero-overlap worker earns
                    // nothing under any gamma, so pin the shaped credit to zero.
                    0.0
                };
                kv_router_config.score_overlap_weight * shape * stock_overlap_credit_blocks
            }
        } else {
            stock_overlap_credit_blocks
        };
        let decode_cost_blocks = load.decode_cost_blocks;
        let active_request_cost_blocks =
            kv_router_config.decode_active_request_weight * load.active_requests as f64;

        // Decode routers normally force `overlap_score_credit=0` through the
        // per-request override, which preserves load-only disagg routing. When
        // conditional disagg leaves a positive overlap credit in place, prefer
        // cache-hot decode workers while still charging decode backlog.
        if self.worker_type == "decode"
            && !context.track_prefill_tokens
            && weights.overlap_score_credit > 0.0
        {
            // Clamp at zero because downstream taint multipliers assume non-negative scores.
            // This loses ordering between workers whose overlap fully offsets decode load, but
            // avoids inverting taint preference among negative-score workers.
            let overlap_adjusted_decode_blocks =
                (decode_cost_blocks - overlap_credit_blocks).max(0.0);
            let logit = overlap_adjusted_decode_blocks + active_request_cost_blocks;
            // Stamped for the same reason as the two rows below: this row is emitted from the
            // `SchedulerQueueActor` task, so the logging layer cannot attach request identity to
            // it, and this branch returns early without reaching them.
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = max(0, decode_blocks - overlap_credit_blocks) + active_request_cost_blocks \
                 = max(0, {decode_cost_blocks:.3} - {overlap_credit_blocks:.3}) + {active_request_cost_blocks:.3}",
                worker.worker_id,
                worker.dp_rank,
            );
            return logit;
        }

        let adjusted_prefill_blocks = (load.raw_prefill_blocks - overlap_credit_blocks).max(0.0);
        let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
        // Charge the decode-side backlog convexly relative to the pool average so
        // a worker that is already hotter than its peers pays more than its
        // linear share.
        let load_scale = if kv_router_config.score_enabled {
            let load_term = decode_cost_blocks + active_request_cost_blocks;
            let shape = if kv_router_config.score_load_gamma == 1.0
                || default_context.mean_pool_load <= 0.0
                || load_term <= 0.0
            {
                1.0
            } else {
                (load_term / default_context.mean_pool_load)
                    .powf(kv_router_config.score_load_gamma - 1.0)
            };
            kv_router_config.score_load_weight * shape
        } else {
            1.0
        };
        // A unit scale is the stock case (shaping off, or neutral knobs). Keep
        // the original summation order there so the result is bit-identical
        // rather than merely close: float addition is not associative.
        let logit = if load_scale == 1.0 {
            prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks
        } else {
            prefill_cost_blocks + load_scale * (decode_cost_blocks + active_request_cost_blocks)
        };

        // These rows are emitted from the `SchedulerQueueActor` task, which `scheduling::queue`
        // spawns without the caller's request span, so the logging layer cannot attach
        // `x_request_id`/`trace_id` to them. Stamp the identity the row needs to be self-joining:
        // `request_id` is the same value `[ROUTING] Best` logs, and `worker_type` separates the
        // prefill-pool and decode-pool decisions that interleave into one log. Both are evaluated
        // inside the macro so they cost nothing when DEBUG is disabled.
        if shared_beyond_device_blocks > 0 {
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks, \
                 {} shared blocks beyond device (multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                shared_beyond_device_blocks,
                load.raw_prefill_blocks,
                shared_cache_multiplier = weights.shared_cache_multiplier,
                prefill_load_scale = weights.prefill_load_scale
            );
        } else {
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                load.raw_prefill_blocks,
                prefill_load_scale = weights.prefill_load_scale
            );
        }

        logit
    }

    #[inline]
    fn worker_cost(
        &self,
        context: &WorkerSelectionContext<'_>,
        default_context: DefaultScoringContext,
        row: &WorkerCandidate,
    ) -> f64 {
        let base_score = self.worker_logit(context, default_context, row, "Formula");
        match row.preferred_taint_multiplier {
            // NOTE: This multiplicative bias assumes a non-negative score. Negative
            // overlap scores expose its pre-existing sign sensitivity; keep it for now.
            Some(multiplier) => base_score * multiplier,
            None => base_score,
        }
    }
}

impl DefaultWorkerPicker {
    pub(super) fn new() -> Self {
        Self::from_parts(
            #[cfg(any(test, feature = "bench"))]
            None,
        )
    }
}

#[inline(always)]
pub(super) fn pick_default_worker<C: WorkerConfigLike>(
    scorer: &DefaultWorkerScorer<&KvRouterConfig>,
    picker: &DefaultWorkerPicker,
    input: &MaterializedSelectionInput<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
) -> Option<(WorkerWithDpRank, f64)> {
    let default_context = DefaultScoringContext::new(
        workers,
        request,
        eligibility,
        input.context.weights,
        scorer.kv_router_config,
    );
    if let Some(worker) = eligibility.pinned_worker() {
        let row = default_row(input, default_context, worker, None);
        return Some((
            worker,
            scorer.worker_logit(&input.context, default_context, &row, "Pinned formula"),
        ));
    }

    // `None` whenever SITA is disabled or inapplicable, which keeps every
    // candidate in the running and preserves stock selection exactly.
    let sita_bounds = sita_worker_id_bounds(
        scorer.kv_router_config,
        workers,
        request,
        eligibility,
        input.context.block_size as usize,
    );
    let in_sita_band = |worker: WorkerWithDpRank| {
        sita_bounds.is_none_or(|(first, last)| {
            worker.worker_id >= first && worker.worker_id <= last
        })
    };

    let temperature = input
        .context
        .router_temperature_override
        .unwrap_or(scorer.kv_router_config.router_temperature);
    let get_score = |worker, config: &C| {
        let preferred_taint_multiplier = request
            .routing_constraints
            .preferred_taint_multiplier(config.taints());
        scorer.worker_cost(
            &input.context,
            default_context,
            &default_row(input, default_context, worker, preferred_taint_multiplier),
        )
    };

    #[cfg(any(test, feature = "bench"))]
    if let Some(rng) = &picker.deterministic_rng {
        let mut candidates = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            if in_sita_band(worker) {
                candidates.push(worker);
            }
        });
        candidates.sort_unstable_by_key(|worker| (worker.worker_id, worker.dp_rank));
        if candidates.is_empty() {
            return None;
        }

        let mut rng = rng.lock();
        let get_candidate_score = |worker| get_score(worker, &workers[&worker.worker_id]);
        if temperature == 0.0 {
            let mut best_worker = None;
            let mut best_cost = f64::INFINITY;
            let mut tie_count = 0;
            for worker in candidates {
                let cost = get_candidate_score(worker);
                if cost < best_cost {
                    best_worker = Some(worker);
                    best_cost = cost;
                    tie_count = 1;
                } else if cost == best_cost {
                    tie_count += 1;
                    if rng.usize(0..tie_count) == 0 {
                        best_worker = Some(worker);
                    }
                }
            }
            return best_worker.map(|worker| (worker, best_cost));
        }

        let entries = candidates
            .into_iter()
            .map(|worker| (worker, get_candidate_score(worker)))
            .collect();
        return Some(softmax_sample_entries(entries, temperature, rng.f64()));
    }

    if temperature == 0.0 {
        let mut best_worker = None;
        let mut best_cost = f64::INFINITY;
        let mut tie_count = 0;
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            if !in_sita_band(worker) {
                return;
            }
            let cost = get_score(worker, config);
            if cost < best_cost {
                best_worker = Some(worker);
                best_cost = cost;
                tie_count = 1;
            } else if cost == best_cost {
                tie_count += 1;
                if fastrand::usize(0..tie_count) == 0 {
                    best_worker = Some(worker);
                }
            }
        });
        return best_worker.map(|worker| (worker, best_cost));
    }

    let mut scratch = picker.softmax_scratch.lock();
    scratch.entries.clear();
    eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
        if in_sita_band(worker) {
            scratch.entries.push((worker, get_score(worker, config)));
        }
    });
    if scratch.entries.is_empty() {
        None
    } else {
        let DefaultSoftmaxScratch {
            entries,
            probabilities,
        } = &mut *scratch;
        let row = softmax_sample_index(
            entries,
            |(_, cost)| *cost,
            temperature,
            fastrand::f64(),
            probabilities,
        );
        Some(entries[row])
    }
}

impl DefaultWorkerPicker {
    fn from_parts(
        #[cfg(any(test, feature = "bench"))] deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        Self {
            softmax_scratch: Mutex::default(),
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn uses_exclusive_affinity_target(&self) -> bool {
        true
    }

    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    #[inline(always)]
    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let (workers, request, eligibility, block_size) = input.into_configured()?;
        select_worker_with_policy(
            &self.kv_router_config,
            self.worker_type,
            WorkerSelectionPolicyStateRef::Default(&self.picker),
            workers,
            request,
            eligibility,
            block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rustc_hash::FxHashMap;

    use super::super::test_support::*;
    use super::*;
    use crate::config::RouterConfigOverride;
    use crate::protocols::SharedCacheHits;
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    fn worker_logit(
        selector: &DefaultWorkerSelector,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        weights: LogitWeights,
    ) -> f64 {
        let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
        let input = MaterializedSelectionInput::new(request, block_size, weights);
        let default_context = DefaultScoringContext::new(
            &workers,
            request,
            request.eligibility(),
            weights,
            &selector.kv_router_config,
        );
        DefaultWorkerScorer::new(selector.kv_router_config.clone(), selector.worker_type)
            .worker_logit(
                &input.context,
                default_context,
                &default_row(&input, default_context, worker, None),
                "test",
            )
    }

    #[test]
    fn default_scoring_context_only_computes_minimum_when_decay_uses_it() {
        let workers = HashMap::from([
            (0, TaintedWorkerConfig::default()),
            (1, TaintedWorkerConfig::default()),
        ]);
        let mut request = base_request(64);
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(0),
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(1),
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 11,
                ..Default::default()
            },
        );
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 1.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };

        let config = KvRouterConfig::default();
        let default_context = DefaultScoringContext::new(
            &workers,
            &request,
            request.eligibility(),
            weights,
            &config,
        );
        assert_eq!(default_context.min_active_prefill_tokens, 7);

        let weights_without_decay = LogitWeights {
            overlap_score_credit_decay: 0.0,
            ..weights
        };
        assert_eq!(
            DefaultScoringContext::new(
                &workers,
                &request,
                request.eligibility(),
                weights_without_decay,
                &config,
            )
            .min_active_prefill_tokens,
            0
        );
    }

    #[test]
    fn softmax_sample_orders_extreme_finite_costs() {
        let result = softmax_sample_entries(vec![(0, -f64::MAX), (1, f64::MAX)], 1.0, 0.6);
        assert_eq!(result.0, 0);
    }

    #[test]
    fn test_default_selector_randomizes_zero_temperature_ties() {
        use crate::test_utils::SimpleWorkerConfig;

        let config = KvRouterConfig {
            router_temperature: 0.0,
            ..Default::default()
        };
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let workers = HashMap::from([
            (10, SimpleWorkerConfig::default()),
            (20, SimpleWorkerConfig::default()),
            (30, SimpleWorkerConfig::default()),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        };
        let mut selected = [false; 3];

        for _ in 0..120 {
            let result = selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .unwrap();
            match result.worker.worker_id {
                10 => selected[0] = true,
                20 => selected[1] = true,
                30 => selected[2] = true,
                worker_id => panic!("unexpected worker id: {worker_id}"),
            }
        }

        let selected_count = selected.into_iter().filter(|seen| *seen).count();
        assert!(
            selected_count > 1,
            "zero-temperature tie-breaking should not always select the same worker"
        );
    }

    #[test]
    fn seeded_selector_is_stable_for_ties_and_temperature_sampling() {
        use crate::test_utils::SimpleWorkerConfig;

        for (temperature, expected_prefix) in [
            (
                0.0,
                [
                    10, 30, 10, 10, 30, 20, 10, 10, 10, 20, 10, 20, 30, 10, 20, 20,
                ],
            ),
            (
                0.7,
                [
                    30, 20, 30, 10, 20, 20, 30, 20, 30, 10, 10, 30, 30, 20, 30, 30,
                ],
            ),
        ] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let mut first = DefaultWorkerSelector::new_seeded(
                Some(KvRouterConfig {
                    router_temperature: 0.0,
                    ..Default::default()
                }),
                "test",
                42,
            );
            first.kv_router_config.router_temperature = temperature;
            let second = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
            let first_workers = HashMap::from([
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
                (20, SimpleWorkerConfig::default()),
            ]);
            let second_workers = HashMap::from([
                (20, SimpleWorkerConfig::default()),
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
            ]);
            let request = base_request(16);

            let first_sequence = (0..64)
                .map(|_| {
                    first
                        .select_worker(WorkerSelectionInput::configured(
                            &first_workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();
            let second_sequence = (0..64)
                .map(|_| {
                    second
                        .select_worker(WorkerSelectionInput::configured(
                            &second_workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();

            assert_eq!(first_sequence, second_sequence);
            assert_eq!(
                first_sequence
                    .iter()
                    .take(expected_prefix.len())
                    .map(|worker| worker.worker_id)
                    .collect::<Vec<_>>(),
                expected_prefix,
            );
        }
    }

    #[test]
    fn per_request_overrides_change_selection() {
        use crate::test_utils::SimpleWorkerConfig;

        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(4);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 2);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 2);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 1,
                ..Default::default()
            },
        );
        #[allow(clippy::single_range_in_vec_init)]
        let shared_cache_hits = SharedCacheHits::from_ranges(vec![0..4]);
        request.shared_cache_hits = Some(shared_cache_hits);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
            router_temperature: 0.0,
            ..Default::default()
        };
        let select = |request: &SchedulingRequest| {
            DefaultWorkerSelector::new_seeded(Some(config.clone()), "test", 42)
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    request,
                    request.eligibility(),
                    1,
                ))
                .unwrap()
                .worker
        };

        assert_eq!(select(&request), warm_worker);

        for (name, config_override) in [
            (
                "overlap_score_credit",
                RouterConfigOverride {
                    overlap_score_credit: Some(0.0),
                    ..Default::default()
                },
            ),
            (
                "prefill_load_scale",
                RouterConfigOverride {
                    prefill_load_scale: Some(0.0),
                    ..Default::default()
                },
            ),
            (
                "shared_cache_multiplier",
                RouterConfigOverride {
                    shared_cache_multiplier: Some(1.0),
                    ..Default::default()
                },
            ),
            (
                "router_temperature",
                RouterConfigOverride {
                    router_temperature: Some(1.0),
                    ..Default::default()
                },
            ),
        ] {
            request.router_config_override = Some(config_override);
            assert_eq!(select(&request), cold_worker, "{name} override was ignored");
        }
    }

    #[test]
    fn test_overloaded_high_overlap_worker_is_skipped() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request
            .overlap
            .effective_overlap_blocks
            .insert(worker0, 4.0);
        request.overlap.effective_cached_tokens.insert(worker0, 64);

        let overloaded_worker_ids = HashSet::from([0]);
        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
                16,
            ))
            .unwrap();

        assert_eq!(result.worker.worker_id, 1);
    }

    #[test]
    fn test_all_eligible_workers_overloaded_returns_overload_error() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 1.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let request = base_request(16);
        let overloaded_worker_ids = HashSet::from([0, 1]);

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        ));

        assert!(matches!(
            result,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[test]
    fn default_policy_retains_eligible_affinity_and_falls_back_when_overloaded() {
        use crate::protocols::WorkerAffinityTarget;
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (
                0,
                SimpleWorkerConfig {
                    data_parallel_size: 2,
                    ..Default::default()
                },
            ),
            (1, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(16);
        request.affinity_target = Some(worker1.into());
        request.worker_loads =
            worker_loads_with_active_decode(FxHashMap::from_iter([(worker0, 0), (worker1, 100)]));
        let eligibility = request
            .eligibility()
            .with_affinity_target(request.affinity_target.unwrap());

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                eligibility,
                16,
            ))
            .unwrap();

        assert_eq!(result.worker, worker1);

        request.affinity_target = Some(WorkerAffinityTarget::new(0, None));
        let eligibility = request
            .eligibility()
            .with_affinity_target(request.affinity_target.unwrap());

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                eligibility,
                16,
            ))
            .unwrap();

        assert_eq!(result.worker.worker_id, 0);
        assert!(result.worker.dp_rank < 2);

        request.affinity_target = Some(worker1.into());
        let overloaded_worker_ids = HashSet::from([1]);

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
                16,
            ))
            .unwrap();

        assert_eq!(result.worker.worker_id, worker0.worker_id);
    }

    #[test]
    fn test_overloaded_pinned_worker_is_not_rerouted() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(16);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        let overloaded_worker_ids = HashSet::from([0]);

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        ));

        assert!(matches!(
            result,
            Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
        ));
    }

    #[test]
    fn test_required_taints_return_no_endpoints_when_no_worker_matches() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([(
            10,
            TaintedWorkerConfig {
                taints: HashSet::from(["mdc-a".to_string()]),
            },
        )]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ));
        assert!(matches!(result, Err(KvSchedulerError::NoEndpoints)));
    }

    #[test]
    fn test_required_taints_filter_out_incompatible_workers() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    #[test]
    fn test_required_taints_switch_matching_worker_sets_by_label() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let name_a = "mdc-a".to_string();
        let name_b = "mdc-b".to_string();
        let name_c = "mdc-c".to_string();
        let taint_a = TaintedWorkerConfig {
            taints: HashSet::from([name_a.clone()]),
        };
        let taint_b = TaintedWorkerConfig {
            taints: HashSet::from([name_b.clone()]),
        };
        let taint_c = TaintedWorkerConfig {
            taints: HashSet::from([name_c.clone()]),
        };
        let workers = HashMap::from([
            (10, taint_a.clone()),
            (11, taint_a),
            (20, taint_b.clone()),
            (21, taint_b),
            (30, taint_c.clone()),
            (31, taint_c),
        ]);

        for (required_taint, expected_worker_id, noisy_worker_id) in [
            (name_a, 10_u64, 11_u64),
            (name_b, 20_u64, 21_u64),
            (name_c, 30_u64, 31_u64),
        ] {
            let mut decode_blocks = FxHashMap::default();
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(expected_worker_id), 0);
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(noisy_worker_id), 400_000);

            let request = SchedulingRequest {
                mode: ScheduleMode::QueryOnly {
                    request_id: Some("test".into()),
                },
                token_seq: None,
                isl_tokens: 16,
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: HashMap::default(),
                    effective_cached_tokens: HashMap::default(),
                },
                kv_transfer_candidates: None,
                retain_kv_transfer_chain: false,
                worker_loads: worker_loads_with_active_decode(decode_blocks),
                track_prefill_tokens: true,
                router_config_override: None,
                lora_name: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                expected_output_tokens: None,
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: crate::protocols::RoutingConstraints {
                    required_taints: HashSet::from([required_taint.clone()]),
                    preferred_taints: HashMap::new(),
                },
                shared_cache_hits: None,
                resp_tx: None,
            };

            let result = selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .unwrap();
            assert_eq!(
                result.worker.worker_id, expected_worker_id,
                "required taint {required_taint} should route only within its compatible worker set"
            );
        }
    }

    #[test]
    fn test_preferred_taints_prefer_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 100);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 90);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), 0.85)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 10);
    }

    #[test]
    fn test_negative_preferred_taints_avoid_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 90);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 100);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), -0.25)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    /// Test the scoring formula with shared cache hits.
    ///
    /// Request [A, B, C, D], shared_cache_multiplier=0.5, block_size=1
    /// - Worker 0: device=[A,B] (overlap=2), shared has [A,B,C,D] -> shared_beyond=2
    ///   adjusted_prefill = isl - 2 - 0.5*2 = 4-2-1 = 1, logit = 1.0 * 1 + 0 = 1.0
    /// - Worker 1: device=[] (overlap=0), shared has [A,B,C,D] -> shared_beyond=4
    ///   adjusted_prefill = isl - 0.5*4 = 4-2 = 2, logit = 1.0 * 2 + 0 = 2.0
    ///
    /// Worker 0 has lower logit (less work), so it wins.
    #[test]
    fn test_shared_cache_hits_scoring() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 1u32;
        let isl = 4usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);
        // worker1 has 0 overlap (not in map)

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 2);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            shared_cache_multiplier: 0.5,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        // Worker 0 should win: logit 1.0 < 2.0
        assert_eq!(
            result.worker, worker0,
            "Worker 0 should be selected (lower logit due to device and shared cache)"
        );
    }

    #[test]
    fn test_prefill_load_scale_applies_after_overlap_credits() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 32);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            prefill_load_scale: 2.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 3);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks: HashMap::new(),
                effective_cached_tokens,
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "prefill load scale should apply before adding decode block load"
        );
    }

    #[test]
    fn test_overlap_credit_above_one_can_prefer_colocated_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(128);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 5,
                ..Default::default()
            },
        );

        let normal_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                ..Default::default()
            }),
            "test",
        );
        let amplified_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.5,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            normal_credit
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            cold_worker
        );
        assert_eq!(
            amplified_credit
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            warm_worker
        );
    }

    #[test]
    fn test_worker_logit_clamps_non_decode_overlap_credit() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        request.overlap.tier_overlap_blocks.device.insert(worker, 6);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 16,
                active_decode_blocks: 2,
                active_requests: 0,
                additional_active_blocks: 3,
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 2.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 7.0);

        request.track_prefill_tokens = false;
        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 5.0);
    }

    #[test]
    fn test_worker_logit_can_charge_active_requests() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(0);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 100,
                active_requests: 4,
                ..Default::default()
            },
        );
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };
        let default = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weighted = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                decode_active_request_weight: 32.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(worker_logit(&default, &request, worker, 16, weights), 100.0);
        assert_eq!(
            worker_logit(&weighted, &request, worker, 16, weights),
            228.0
        );
    }

    #[test]
    fn test_decode_worker_logit_credits_overlap_without_prefill_tracking() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.track_prefill_tokens = false;
        request.overlap.tier_overlap_blocks.device.insert(worker, 3);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 10,
                ..Default::default()
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "decode");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 7.0);
    }

    #[test]
    fn test_overlap_credit_decay_can_prefer_less_loaded_cold_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(64);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 48,
                ..Default::default()
            },
        );

        let no_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let with_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 1.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            no_decay
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            warm_worker
        );
        assert_eq!(
            with_decay
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            cold_worker
        );
    }

    #[test]
    fn test_effective_overlap_falls_back_when_tier_blocks_are_absent() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 4.0);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 1);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "effective overlap should still credit older callers without tier maps"
        );
    }

    #[test]
    fn summary_overlap_fallback_preserves_default_shared_prefix() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
        let mut request = base_request(64);
        request.overlap.effective_overlap_blocks.insert(worker, 2.0);
        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);
        request.shared_cache_hits = Some(shared_hits);
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 1.0,
        };
        let input = MaterializedSelectionInput::new(&request, 16, weights);
        let default_context = DefaultScoringContext::new(
            &workers,
            &request,
            request.eligibility(),
            weights,
            &KvRouterConfig::default(),
        );
        let custom_row = input.row(worker, None, WorkerInputs::CACHE);
        let default_row = default_row(&input, default_context, worker, None);

        assert_eq!(custom_row.cache.device_overlap_blocks, 0.0);
        assert_eq!(custom_row.cache.shared_beyond_device_blocks, 4);
        assert_eq!(default_row.cache.device_overlap_blocks, 2.0);
        assert_eq!(default_row.cache.shared_beyond_device_blocks, 2);
    }

    fn sita_config(worker_count_share: f64) -> KvRouterConfig {
        KvRouterConfig {
            sita_enabled: true,
            sita_boundary_1: 1024,
            sita_boundary_2: 8192,
            sita_small_band_share: worker_count_share,
            sita_osl_weight: 0.0,
            // Disable spilling so band placement is observable on its own.
            sita_spill_threshold: 1.0,
            router_temperature: 0.0,
            ..Default::default()
        }
    }

    /// A request that already has a cached prefix somewhere, so it keeps the
    /// cache affinity that band confinement exists to protect. Tests about band
    /// *placement* need this: a request with no overlap anywhere is
    /// deliberately allowed out of its band (see
    /// `sita_widen_uncached_long_band`), which would otherwise mask the
    /// behavior under test.
    /// The overlap is deliberately tiny so it does not move the request across a
    /// band boundary — only its presence matters here.
    fn sita_cached_request(isl_tokens: usize) -> SchedulingRequest {
        let mut request = base_request(isl_tokens);
        request
            .overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::from_worker_id(0), 64);
        request
    }

    /// Workers that report KV capacity, so band occupancy is computable.
    fn sita_workers(count: u64, total_kv_blocks: u64) -> HashMap<WorkerId, SitaWorkerConfig> {
        (0..count)
            .map(|worker_id| {
                (
                    worker_id,
                    SitaWorkerConfig {
                        total_kv_blocks,
                        taints: HashSet::new(),
                    },
                )
            })
            .collect()
    }

    #[derive(Clone)]
    struct SitaWorkerConfig {
        total_kv_blocks: u64,
        taints: HashSet<String>,
    }

    impl crate::protocols::WorkerConfigLike for SitaWorkerConfig {
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
            Some(self.total_kv_blocks)
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    #[test]
    fn sita_band_mapping_uses_boundaries() {
        // Three-band configuration.
        assert_eq!(sita_band_for_size(0, 1024, 8192), 0);
        assert_eq!(sita_band_for_size(1024, 1024, 8192), 0);
        assert_eq!(sita_band_for_size(1025, 1024, 8192), 1);
        assert_eq!(sita_band_for_size(8192, 1024, 8192), 1);
        assert_eq!(sita_band_for_size(8193, 1024, 8192), 2);
        assert_eq!(sita_band_for_size(usize::MAX, 1024, 8192), 2);

        // boundary_2 == 0 collapses to a two-band split.
        assert_eq!(sita_band_for_size(1024, 1024, 0), 0);
        assert_eq!(sita_band_for_size(1025, 1024, 0), 1);
        assert_eq!(sita_band_for_size(usize::MAX, 1024, 0), 1);
    }

    #[test]
    fn sita_band_slices_are_contiguous_and_non_empty() {
        for (worker_count, share, band_count) in [
            (16, 0.5, 3),
            (16, 0.5, 2),
            (16, 0.125, 3),
            (16, 0.75, 3),
            (2, 0.5, 3),
            (3, 0.5, 3),
            (5, 0.125, 3),
            (7, 0.75, 3),
        ] {
            let slices = sita_band_slices(worker_count, share, band_count);
            let used = &slices[..band_count];
            assert_eq!(used[0].0, 0, "band 0 must start at the first worker");
            assert_eq!(
                used[band_count - 1].1,
                worker_count,
                "the last band must reach the final worker"
            );
            for (band, &(start, end)) in used.iter().enumerate() {
                assert!(start < end, "band {band} must be non-empty: {slices:?}");
            }
            for pair in used.windows(2) {
                // Adjacent bands are contiguous, or identical when the pool is
                // too small to give every band its own workers.
                assert!(pair[0].1 == pair[1].0 || pair[0] == pair[1], "{slices:?}");
            }
        }

        // Band 0 gets ceil(share * N).
        assert_eq!(sita_band_slices(16, 0.5, 3)[0], (0, 8));
        assert_eq!(sita_band_slices(16, 0.125, 3)[0], (0, 2));
        assert_eq!(sita_band_slices(10, 0.25, 3)[0], (0, 3));
    }

    #[test]
    fn sita_routes_short_and_long_requests_to_disjoint_bands() {
        let workers = sita_workers(8, 1_000);
        let selector = DefaultWorkerSelector::new(Some(sita_config(0.5)), "test");

        let short = sita_cached_request(256);
        let long = sita_cached_request(16_384);

        let short_worker = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &short,
                short.eligibility(),
                64,
            ))
            .unwrap()
            .worker;
        let long_worker = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &long,
                long.eligibility(),
                64,
            ))
            .unwrap()
            .worker;

        // share=0.5 over 8 workers: band 0 = ids 0..4, band 1 = 4..6, band 2 = 6..8.
        assert!(short_worker.worker_id < 4, "short request left band 0");
        assert!(long_worker.worker_id >= 6, "long request left band 2");
    }

    /// Band confinement is paid for by cache affinity, so a request with no
    /// overlap anywhere gets the whole pool above band 0 instead of its own
    /// narrow slice — but band 0 stays reserved while short requests are using it.
    #[test]
    fn sita_uncached_long_request_widens_beyond_its_band() {
        // share=0.5 over 8 workers: band 0 = 0..4, band 1 = 4..6, band 2 = 6..8.
        let slices = sita_band_slices(8, 0.5, 3);
        assert_eq!(slices, [(0, 4), (4, 6), (6, 8)]);

        // Band 0 as busy as the rest of the pool, so it is not lendable.
        let even = [1.0; 8];
        let widen = |band, cached, loads: &[f64]| {
            sita_widen_uncached_long_band(band, &slices, 8, cached, loads, 0.85)
        };

        // A cached request stays inside its own band.
        assert_eq!(widen(2, 64, &even), (6, 8));
        assert_eq!(widen(1, 64, &even), (4, 6));

        // With no cached prefix, bands 1 and 2 open up to every non-short worker.
        assert_eq!(widen(2, 0, &even), (4, 8));
        assert_eq!(widen(1, 0, &even), (4, 8));

        // Band 0 is never widened: short requests keep their reservation, and a
        // zero-overlap short request must not escape into the long workers.
        assert_eq!(widen(0, 0, &even), (0, 4));
    }

    /// An uncached long request has no affinity to lose, so it is the cheapest
    /// possible borrower of band 0 — but only while band 0 is genuinely idle,
    /// and never when spilling is switched off.
    #[test]
    fn sita_uncached_long_request_borrows_band_0_only_while_it_is_idle() {
        let slices = sita_band_slices(8, 0.5, 3);
        let widen = |loads: &[f64], spill| {
            sita_widen_uncached_long_band(2, &slices, 8, 0, loads, spill)
        };

        // Band 0 idle (occupancy 0.02 < 2 * (1 - 0.85) = 0.30): lend it out.
        let idle = [0.02, 0.02, 0.02, 0.02, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(widen(&idle, 0.85), (0, 8));

        // Band 0 carrying its share of the pool: reservation holds.
        let busy = [1.0; 8];
        assert_eq!(widen(&busy, 0.85), (4, 8));

        // Spilling off means band 0 is never lent, however idle it is.
        assert_eq!(widen(&idle, 1.0), (4, 8));
    }

    /// End-to-end through the selector: the same long request lands outside its
    /// band when it has nothing cached, and inside it when it does.
    #[test]
    fn sita_uncached_long_request_can_use_the_medium_band() {
        let workers = sita_workers(8, 1_000);
        let selector = DefaultWorkerSelector::new(Some(sita_config(0.5)), "test");

        // Saturate the top band so the widened slice is genuinely preferred;
        // spilling is off in `sita_config`, so only the widening can move it.
        let saturate_top_band = |request: &mut SchedulingRequest| {
            for worker_id in 6..8 {
                request.worker_loads.insert(
                    WorkerWithDpRank::from_worker_id(worker_id),
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks: 900,
                        ..Default::default()
                    },
                );
            }
        };
        let mut uncached = base_request(16_384);
        saturate_top_band(&mut uncached);
        let mut cached = sita_cached_request(16_384);
        saturate_top_band(&mut cached);

        let pick = |request: &SchedulingRequest| {
            selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    request,
                    request.eligibility(),
                    64,
                ))
                .unwrap()
                .worker
                .worker_id
        };

        assert!(
            (4..6).contains(&pick(&uncached)),
            "an uncached long request should reach the idle medium band"
        );
        assert!(
            pick(&cached) >= 6,
            "a request with cache affinity stays in its own band"
        );
    }

    #[test]
    fn sita_osl_weight_promotes_long_output_requests() {
        let workers = sita_workers(8, 1_000);
        // ISL alone lands in band 0; the output estimate must push it past
        // boundary_1 and out of the short band.
        let mut request = base_request(512);
        request.expected_output_tokens = Some(2_048);

        let without_osl = DefaultWorkerSelector::new(Some(sita_config(0.5)), "test")
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                64,
            ))
            .unwrap()
            .worker;
        let with_osl = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_osl_weight: 1.0,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            64,
        ))
        .unwrap()
        .worker;

        assert!(without_osl.worker_id < 4);
        assert!(with_osl.worker_id >= 4);
    }

    #[test]
    fn sita_cache_overlap_shrinks_effective_size() {
        let workers = sita_workers(8, 1_000);
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        // ISL is a band-1 request, but a cache hit on worker 0 leaves only a
        // band-0 amount of prefill to actually do.
        let mut request = base_request(2_048);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 1_800);

        let selector = DefaultWorkerSelector::new(Some(sita_config(0.5)), "test");
        let worker = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                64,
            ))
            .unwrap()
            .worker;

        assert!(
            worker.worker_id < 4,
            "overlap-adjusted size should map to band 0"
        );
    }

    #[test]
    fn sita_spills_into_adjacent_band_when_target_is_saturated() {
        let workers = sita_workers(8, 1_000);
        let mut saturated = base_request(256);
        // Fill band 0 (ids 0..4) to 95% and leave band 1 (ids 4..6) empty.
        for worker_id in 0..4 {
            saturated.worker_loads.insert(
                WorkerWithDpRank::from_worker_id(worker_id),
                crate::sequences::WorkerLoadProjection {
                    active_decode_blocks: 950,
                    ..Default::default()
                },
            );
        }

        let no_spill = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_spill_threshold: 1.0,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &saturated,
            saturated.eligibility(),
            64,
        ))
        .unwrap()
        .worker;
        let with_spill = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_spill_threshold: 0.85,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &saturated,
            saturated.eligibility(),
            64,
        ))
        .unwrap()
        .worker;

        assert!(
            no_spill.worker_id < 4,
            "a saturated band still confines routing when spilling is disabled"
        );
        assert!(
            with_spill.worker_id >= 4,
            "spilling should reach the idle adjacent band"
        );
    }

    #[test]
    fn sita_never_spills_into_a_busier_band() {
        let workers = sita_workers(8, 1_000);
        let mut request = base_request(256);
        // Every band is over the threshold; the neighbor is no better, so the
        // request must stay in its own band.
        for worker_id in 0..8 {
            request.worker_loads.insert(
                WorkerWithDpRank::from_worker_id(worker_id),
                crate::sequences::WorkerLoadProjection {
                    active_decode_blocks: if worker_id < 4 { 900 } else { 980 },
                    ..Default::default()
                },
            );
        }

        let worker = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_spill_threshold: 0.5,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            64,
        ))
        .unwrap()
        .worker;

        assert!(worker.worker_id < 4);
    }

    /// The largest band has no band above it, so without a downward relief
    /// valve it accumulates spill it can never shed and its queue wrecks tail
    /// latency. It may widen into band 1 — but never into the protected band 0.
    #[test]
    fn sita_top_band_spills_down_but_never_into_the_short_band() {
        // share 0.5 over 8 workers => band 0 = 0..4, band 1 = 4..6, band 2 = 6..8.
        let workers = sita_workers(8, 1_000);
        let mut request = sita_cached_request(9_000);
        // Saturate the top band while keeping band 0 busy enough not to be
        // lendable, so the only relief available is the step down into band 1.
        // Band 2 occupancy = 950/(950+100) = 0.90 > 0.85, so it spills; band 0
        // occupancy = 150/(150+475) = 0.24 >= 1 - 0.85, so it stays reserved.
        for worker_id in 0..8 {
            let active_decode_blocks = match worker_id {
                0..=3 => 150, // band 0: in use, not lendable
                4..=5 => 0,   // band 1: idle
                _ => 950,     // band 2: saturated
            };
            request.worker_loads.insert(
                WorkerWithDpRank::from_worker_id(worker_id),
                crate::sequences::WorkerLoadProjection {
                    active_decode_blocks,
                    ..Default::default()
                },
            );
        }

        let no_spill = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_spill_threshold: 1.0,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            64,
        ))
        .unwrap()
        .worker;
        let with_spill = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                sita_spill_threshold: 0.85,
                ..sita_config(0.5)
            }),
            "test",
        )
        .select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            64,
        ))
        .unwrap()
        .worker;

        assert!(
            no_spill.worker_id >= 6,
            "with spilling off the top band stays confined to its own slice"
        );
        assert!(
            (4..6).contains(&with_spill.worker_id),
            "a saturated top band must reach band 1, and must not touch band 0"
        );
    }

    /// Band 0's reservation is only free while band 0 is idle. A saturated long
    /// band may borrow it then, but must not touch it while short requests are
    /// actually using those workers.
    #[test]
    fn sita_long_band_borrows_band_0_only_while_it_is_idle() {
        // share 0.5 over 8 workers => band 0 = 0..4, band 1 = 4..6, band 2 = 6..8.
        let workers = sita_workers(8, 1_000);
        let config = KvRouterConfig {
            sita_spill_threshold: 0.85,
            ..sita_config(0.5)
        };

        let saturate_long = |band_0_load: usize| {
            let mut request = base_request(9_000);
            for worker_id in 0..8 {
                request.worker_loads.insert(
                    WorkerWithDpRank::from_worker_id(worker_id),
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks: if worker_id < 4 { band_0_load } else { 950 },
                        ..Default::default()
                    },
                );
            }
            DefaultWorkerSelector::new(Some(config.clone()), "test")
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    64,
                ))
                .unwrap()
                .worker
        };

        assert!(
            saturate_long(0).worker_id < 4,
            "an idle band 0 should be lent to a saturated long band"
        );
        assert!(
            saturate_long(950).worker_id >= 4,
            "a busy band 0 keeps its workers reserved for short requests"
        );
    }

    #[test]
    fn sita_respects_pinned_workers_and_tiny_pools() {
        // A pinned worker outside the request's band still wins.
        let workers = sita_workers(8, 1_000);
        let mut request = base_request(256);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(7));
        let selector = DefaultWorkerSelector::new(Some(sita_config(0.5)), "test");
        assert_eq!(
            selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    64,
                ))
                .unwrap()
                .worker
                .worker_id,
            7
        );

        // A single-worker pool cannot be partitioned, so routing must succeed.
        let single = sita_workers(1, 1_000);
        let long = base_request(16_384);
        assert_eq!(
            selector
                .select_worker(WorkerSelectionInput::configured(
                    &single,
                    &long,
                    long.eligibility(),
                    64,
                ))
                .unwrap()
                .worker
                .worker_id,
            0
        );
    }

    /// A request with partial device overlap and decode backlog, enough for
    /// both the overlap credit and the load term to be non-trivial.
    fn score_request() -> SchedulingRequest {
        let worker = WorkerWithDpRank::from_worker_id(0);
        // 512 tokens at block_size 64 = 8 request blocks; 2 blocks overlap = 0.25.
        let mut request = base_request(512);
        request.overlap.effective_overlap_blocks.insert(worker, 2.0);
        request
            .overlap
            .effective_cached_tokens
            .insert(worker, 2 * 64);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 40,
                active_requests: 3,
                ..Default::default()
            },
        );
        request
    }

    fn score_weights() -> LogitWeights {
        LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        }
    }

    fn score_logit(config: KvRouterConfig) -> f64 {
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        worker_logit(
            &selector,
            &score_request(),
            WorkerWithDpRank::from_worker_id(0),
            64,
            score_weights(),
        )
    }

    /// The whole shaping is multiplicative around 1.0, so the neutral knob set
    /// must land on the stock logit exactly — not merely close.
    #[test]
    fn score_neutral_values_are_identical_to_stock() {
        let neutral = KvRouterConfig {
            score_enabled: true,
            score_overlap_weight: 1.0,
            score_overlap_gamma: 1.0,
            score_load_weight: 1.0,
            score_load_gamma: 1.0,
            score_min_overlap_frac: 0.0,
            decode_active_request_weight: 2.0,
            ..Default::default()
        };
        let stock = KvRouterConfig {
            score_enabled: false,
            ..neutral.clone()
        };

        assert_eq!(score_logit(neutral), score_logit(stock));
    }

    /// The A/B mechanism probe replays every claim with `score_enabled=false`,
    /// so disabled shaping must not perturb a single selection.
    #[test]
    fn score_disabled_is_identical_to_stock() {
        let workers = sita_workers(8, 1_000);
        let disabled = KvRouterConfig {
            score_enabled: false,
            // Non-default score knobs must stay inert while disabled.
            score_overlap_weight: 3.5,
            score_overlap_gamma: 0.5,
            score_load_weight: 0.25,
            score_load_gamma: 2.0,
            score_min_overlap_frac: 0.4,
            ..Default::default()
        };
        let stock = KvRouterConfig::default();

        for isl in [64, 256, 1_024, 4_096, 16_384] {
            let mut request = base_request(isl);
            for worker_id in 0..8 {
                let worker = WorkerWithDpRank::from_worker_id(worker_id);
                request.worker_loads.insert(
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks: (worker_id as usize) * 37,
                        active_prefill_tokens: (worker_id as usize) * 11,
                        active_requests: worker_id as usize,
                        ..Default::default()
                    },
                );
                request
                    .overlap
                    .effective_cached_tokens
                    .insert(worker, (worker_id as usize) * 64);
                request
                    .overlap
                    .effective_overlap_blocks
                    .insert(worker, worker_id as f64);
            }

            for temperature in [0.0, 0.7] {
                let select = |config: &KvRouterConfig| {
                    let selector = DefaultWorkerSelector::new_seeded(
                        Some(KvRouterConfig {
                            router_temperature: temperature,
                            ..config.clone()
                        }),
                        "test",
                        7,
                    );
                    (0..32)
                        .map(|_| {
                            selector
                                .select_worker(WorkerSelectionInput::configured(
                                    &workers,
                                    &request,
                                    request.eligibility(),
                                    64,
                                ))
                                .unwrap()
                                .worker
                        })
                        .collect::<Vec<_>>()
                };

                assert_eq!(
                    select(&disabled),
                    select(&stock),
                    "score_enabled=false changed selection at isl={isl} temperature={temperature}"
                );
            }
        }
    }

    /// A prefix match thinner than the cutoff earns nothing, which raises the
    /// logit to the no-credit baseline.
    #[test]
    fn score_min_overlap_frac_drops_thin_matches() {
        let base = KvRouterConfig {
            score_enabled: true,
            ..Default::default()
        };
        // The fixture overlaps 2 of 8 request blocks.
        let below_cutoff = score_logit(KvRouterConfig {
            score_min_overlap_frac: 0.2,
            ..base.clone()
        });
        let above_cutoff = score_logit(KvRouterConfig {
            score_min_overlap_frac: 0.3,
            ..base.clone()
        });
        let no_credit = score_logit(KvRouterConfig {
            score_overlap_weight: 0.25,
            score_min_overlap_frac: 0.5,
            ..base.clone()
        });

        // 0.25 >= 0.2 keeps the credit and matches the unshaped logit.
        assert_eq!(below_cutoff, score_logit(base));
        // 0.25 < 0.3 suppresses it, so the request pays full prefill.
        assert!(
            above_cutoff > below_cutoff,
            "cutting overlap credit must raise the cost: {above_cutoff} vs {below_cutoff}"
        );
        // Once suppressed, the credit weight no longer matters.
        assert_eq!(above_cutoff, no_credit);
    }

    /// The fixture's overlap fraction is below 1.0, so raising the exponent
    /// shrinks the credit and raising load gamma above 1.0 charges a
    /// hotter-than-average worker more.
    #[test]
    fn score_gammas_are_monotone() {
        let base = KvRouterConfig {
            score_enabled: true,
            ..Default::default()
        };
        let overlap_logit = |gamma: f64| {
            score_logit(KvRouterConfig {
                score_overlap_gamma: gamma,
                ..base.clone()
            })
        };

        // overlap_frac = 0.25 < 1, so frac^gamma decreases as gamma grows,
        // shrinking the credit and increasing the cost.
        let costs: Vec<f64> = [0.5, 1.0, 1.5, 2.0].into_iter().map(overlap_logit).collect();
        for pair in costs.windows(2) {
            assert!(
                pair[1] > pair[0],
                "overlap cost must rise with gamma: {pair:?}"
            );
        }

        // Load shaping needs a pool where this worker is above the mean, so
        // score the busy worker against an idle peer.
        let workers = sita_workers(2, 1_000);
        let mut request = base_request(512);
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(0),
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 200,
                ..Default::default()
            },
        );
        let load_cost = |gamma: f64| {
            let config = KvRouterConfig {
                score_load_gamma: gamma,
                ..base.clone()
            };
            let weights = score_weights();
            let input = MaterializedSelectionInput::new(&request, 64, weights);
            let default_context = DefaultScoringContext::new(
                &workers,
                &request,
                request.eligibility(),
                weights,
                &config,
            );
            let worker = WorkerWithDpRank::from_worker_id(0);
            DefaultWorkerScorer::new(config, "test").worker_logit(
                &input.context,
                default_context,
                &default_row(&input, default_context, worker, None),
                "test",
            )
        };

        // Worker 0 carries 200 blocks against a pool mean of 100, so the
        // load/mean ratio is 2.0 and convexity must charge it more.
        let load_costs: Vec<f64> = [1.0, 1.25, 1.5, 2.0].into_iter().map(load_cost).collect();
        for pair in load_costs.windows(2) {
            assert!(
                pair[1] > pair[0],
                "an above-average worker must be charged more as load gamma rises: {pair:?}"
            );
        }
    }

    /// The A/B mechanism probe compares `sita_enabled=false` against stock, so
    /// disabled SITA must not perturb a single selection.
    #[test]
    fn sita_disabled_is_identical_to_stock() {
        let workers = sita_workers(8, 1_000);
        let disabled = KvRouterConfig {
            sita_enabled: false,
            // Non-default SITA knobs must stay inert while disabled.
            sita_boundary_1: 256,
            sita_boundary_2: 512,
            sita_small_band_share: 0.25,
            sita_osl_weight: 1.5,
            sita_spill_threshold: 0.6,
            ..Default::default()
        };
        let stock = KvRouterConfig::default();

        for isl in [64, 256, 1_024, 4_096, 16_384, 65_536] {
            let mut request = base_request(isl);
            request.expected_output_tokens = Some(1_024);
            for worker_id in 0..8 {
                let worker = WorkerWithDpRank::from_worker_id(worker_id);
                request.worker_loads.insert(
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks: (worker_id as usize) * 37,
                        active_prefill_tokens: (worker_id as usize) * 11,
                        ..Default::default()
                    },
                );
                request
                    .overlap
                    .effective_cached_tokens
                    .insert(worker, (worker_id as usize) * 64);
            }

            // Seeded selectors make the full decision sequence comparable,
            // including tie-breaks and temperature sampling.
            for temperature in [0.0, 0.7] {
                let disabled_selector = DefaultWorkerSelector::new_seeded(
                    Some(KvRouterConfig {
                        router_temperature: temperature,
                        ..disabled.clone()
                    }),
                    "test",
                    7,
                );
                let stock_selector = DefaultWorkerSelector::new_seeded(
                    Some(KvRouterConfig {
                        router_temperature: temperature,
                        ..stock.clone()
                    }),
                    "test",
                    7,
                );
                let select = |selector: &DefaultWorkerSelector| {
                    (0..32)
                        .map(|_| {
                            selector
                                .select_worker(WorkerSelectionInput::configured(
                                    &workers,
                                    &request,
                                    request.eligibility(),
                                    64,
                                ))
                                .unwrap()
                                .worker
                        })
                        .collect::<Vec<_>>()
                };

                assert_eq!(
                    select(&disabled_selector),
                    select(&stock_selector),
                    "sita_enabled=false changed selection at isl={isl} temperature={temperature}"
                );
            }
        }
    }

    /// Without shared cache hits, the scoring should be unchanged.
    #[test]
    fn test_no_shared_cache_unchanged() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);

        let config = KvRouterConfig::default();
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(result.worker, worker0);
    }
}
