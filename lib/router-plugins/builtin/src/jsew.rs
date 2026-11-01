// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Join-shortest-expected-work (JSEW) worker selection.
//!
//! Dynamo's default scorer charges decode load as *footprint*: the KV blocks a worker's active
//! requests occupy. On long-context agentic traces footprint is a poor proxy for the *work* those
//! requests still owe. A 300k-token prompt that emits 24 tokens occupies an enormous footprint
//! while costing almost no decode work; a 45k prompt that emits 60k tokens pins a worker for
//! minutes on a modest footprint. Routing on footprint therefore steers new requests away from
//! workers that are nearly done and toward workers that are about to generate for a long time,
//! which shows up as tail time-to-last-token.
//!
//! This policy computes the default cost and rescales only its decode term:
//!
//! ```text
//! unshared_blocks = decode_blocks * active_requests ^ decode_share_exponent
//! charged_blocks  = unshared_blocks * (unshared_blocks / mean_unshared_blocks) ^ (decode_load_exponent - 1)
//! decode_term     = decode_footprint_weight * charged_blocks
//!                 + decode_work_scale * (predicted_decode_blocks / request_blocks) ^ decode_work_exponent
//!                   * charged_blocks
//! ```
//!
//! `decode_footprint_weight` retains the stock footprint belief and `decode_work_scale` adds the
//! predicted-work belief, normalized by the incoming request's own size so the term is a
//! dimensionless ratio. `decode_footprint_weight: 1.0`, `decode_work_scale: 0.0`,
//! `decode_load_exponent: 1.0` and `decode_share_exponent: 0.0` reproduce the default
//! selector's decision exactly, which is what an A/B against the default leans on.
//!
//! - `decode_work_exponent` bends the size ratio. The ratio's denominator is the incoming
//!   request's own block count, so the term charges a short prompt more per block of a worker's
//!   backlog than a long one; the exponent sets how sharply that falls off with prompt size.
//! - `decode_load_exponent` bends the charge in the worker's own backlog. Decode step time does
//!   not grow linearly in resident KV: past the batch size where the step goes memory-bound,
//!   another long session costs the whole batch more than the one before it. A convex charge
//!   (`> 1.0`) makes the last increment of backlog on an already-loaded worker dominate.
//!   Dividing by the candidate mean keeps the term comparable to the prefill term it is summed
//!   with, so bending the curve does not re-weight the whole cost.
//! - `decode_share_exponent` corrects a units mismatch. The reported decode footprint counts
//!   physically resident blocks, which deduplicates the prefix that sibling sessions share;
//!   decode step time scales with the sum of their logical context lengths. A worker holding
//!   several sharing sessions under-reports its load by roughly that count, and cache-affinity
//!   routing sends it more of exactly those sessions. This factor charges the un-shared sum back.
//!
//! Two predictors supply `predicted_decode_blocks`:
//!
//! - `ema_uncached_tokens` (default) tracks an exponential moving average of the *uncached*
//!   prompt tokens the router has placed. It uses only what the router knows at admission time.
//! - `ground_truth_future_decode` reads the caller-declared `expected_output_tokens`, falling back
//!   to the EMA when absent. In replay that is the recorded output length, which makes it an
//!   oracle upper bound on what any predictor can reach.
//!
//! The belief is resolved once per request, not once per candidate: the EMA is stateful, so
//! folding it per candidate would make the prediction depend on how many workers are eligible.
//! State lives in the policy instance, one per routing pool, like the default's picker state.

use std::sync::Arc;

use dynamo_kv_router::config::SharedCacheType;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerCandidates, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerType};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-jsew";

/// Defaults tuned jointly on a 1,200-session Claude Code agentic trace (16 workers, DynoSim
/// replay, TTLT objective): 290 random points over the four tunables, the leaders re-verified
/// at 10 seeds. Treat them as a set, not as independent dials: the objective surface is stepped
/// rather than smooth because what moves the tail is whether a handful of very long sessions
/// land on distinct workers. The predicted-work term wants to be loud and the belief that feeds
/// it slow (`ema_alpha` is a horizon of about 60 requests), the convex load exponent is the knob
/// that moves the tail, and the footprint weight is the linear counterweight that keeps the
/// convexity from over-draining lightly loaded workers. The share exponent sits far below the
/// `1.0` the units argument alone suggests because `active_requests` counts every session, not
/// only the ones that share a prefix; `0.13` took nearly all of the gain while leaving TTFT
/// headroom that survived a seed change. Re-tune the set against a target trace.
const DEFAULT_DECODE_WORK_SCALE: f64 = 17.601;
const DEFAULT_DECODE_FOOTPRINT_WEIGHT: f64 = 1.526;
const DEFAULT_EMA_ALPHA: f64 = 0.01737;
const DEFAULT_DECODE_WORK_EXPONENT: f64 = 1.0;
const DEFAULT_DECODE_LOAD_EXPONENT: f64 = 1.2;
const DEFAULT_DECODE_SHARE_EXPONENT: f64 = 0.13;

/// Which belief supplies the predicted remaining decode work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum EffectiveWorkPredictor {
    /// Exponential moving average over the uncached prompt tokens already routed.
    #[default]
    EmaUncachedTokens,
    /// The caller-declared `expected_output_tokens`, falling back to the EMA when absent.
    GroundTruthFutureDecode,
}

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional and keeps the tuned default when omitted. Unknown keys are rejected at
/// startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Which belief supplies the predicted remaining decode work.
    effective_work_predictor: EffectiveWorkPredictor,
    /// Weight on the predicted-work term, relative to the request's own block count. `0.0`
    /// disables the hypothesis and leaves the stock footprint belief in charge.
    decode_work_scale: f64,
    /// Weight retained on the stock decode-footprint term. `1.0` keeps it whole.
    decode_footprint_weight: f64,
    /// Smoothing factor for the EMA, in `(0, 1]`. `1.0` degenerates to "last routed request".
    ema_alpha: f64,
    /// Exponent on the predicted-work-to-request-size ratio. `1.0` leaves the ratio linear.
    decode_work_exponent: f64,
    /// Exponent on the worker's decode backlog, normalized by the candidate mean. `1.0` charges
    /// backlog linearly; above `1.0` charges an already-loaded worker superlinearly.
    decode_load_exponent: f64,
    /// Exponent on the worker's active-request count, which un-shares the deduplicated decode
    /// footprint. `0.0` leaves the footprint as reported.
    decode_share_exponent: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            effective_work_predictor: EffectiveWorkPredictor::default(),
            decode_work_scale: DEFAULT_DECODE_WORK_SCALE,
            decode_footprint_weight: DEFAULT_DECODE_FOOTPRINT_WEIGHT,
            ema_alpha: DEFAULT_EMA_ALPHA,
            decode_work_exponent: DEFAULT_DECODE_WORK_EXPONENT,
            decode_load_exponent: DEFAULT_DECODE_LOAD_EXPONENT,
            decode_share_exponent: DEFAULT_DECODE_SHARE_EXPONENT,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.decode_work_scale.is_finite() || !(0.0..=20.0).contains(&self.decode_work_scale) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_work_scale must be a finite number between 0.0 and 20.0",
            ));
        }
        if !self.decode_footprint_weight.is_finite()
            || !(0.0..=2.0).contains(&self.decode_footprint_weight)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_footprint_weight must be a finite number between 0.0 and 2.0",
            ));
        }
        if !self.ema_alpha.is_finite() || self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "ema_alpha must be a finite number in (0.0, 1.0]",
            ));
        }
        // Both exponents are bounded away from zero: an exponent of zero collapses its factor to a
        // constant, which silently deletes the term rather than reshaping it.
        if !self.decode_work_exponent.is_finite()
            || !(0.1..=4.0).contains(&self.decode_work_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_work_exponent must be a finite number between 0.1 and 4.0",
            ));
        }
        if !self.decode_load_exponent.is_finite()
            || !(0.1..=4.0).contains(&self.decode_load_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_load_exponent must be a finite number between 0.1 and 4.0",
            ));
        }
        // Zero is the neutral value here, not a degenerate one: it leaves the reported footprint
        // untouched rather than deleting a factor.
        if !self.decode_share_exponent.is_finite()
            || !(0.0..=2.0).contains(&self.decode_share_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_share_exponent must be a finite number between 0.0 and 2.0",
            ));
        }
        Ok(())
    }
}

/// Cross-request belief for the EMA predictor. Seeded from the first observation rather than
/// from zero, so the first requests are not routed against a belief no traffic has informed.
#[derive(Debug, Default)]
struct EmaBelief {
    uncached_tokens: Option<f64>,
}

impl EmaBelief {
    fn observe(&mut self, uncached_tokens: f64, alpha: f64) -> f64 {
        let updated = match self.uncached_tokens {
            Some(previous) => alpha * uncached_tokens + (1.0 - alpha) * previous,
            None => uncached_tokens,
        };
        self.uncached_tokens = Some(updated);
        updated
    }
}

/// The decode-cost rescaling this policy applies, resolved once per request.
#[derive(Debug, Clone, Copy)]
struct DecodeWorkBelief {
    predicted_decode_blocks: f64,
    mean_decode_blocks: f64,
    decode_work_scale: f64,
    decode_footprint_weight: f64,
    decode_work_exponent: f64,
    decode_load_exponent: f64,
    decode_share_exponent: f64,
}

impl DecodeWorkBelief {
    /// Undo the deduplication in the reported decode footprint: `0.0` trusts the reported
    /// footprint, `1.0` charges the fully un-shared sum.
    fn unshared_blocks(self, decode_cost_blocks: f64, active_requests: usize) -> f64 {
        if self.decode_share_exponent == 0.0 || decode_cost_blocks <= 0.0 {
            return decode_cost_blocks;
        }
        decode_cost_blocks * (active_requests.max(1) as f64).powf(self.decode_share_exponent)
    }

    /// Bend the worker's decode backlog around the candidate mean. An empty queue costs nothing
    /// at any exponent, and short-circuiting it keeps `0 ^ negative` out of the normalization.
    fn charged_blocks(self, decode_cost_blocks: f64) -> f64 {
        if self.decode_load_exponent == 1.0
            || decode_cost_blocks <= 0.0
            || self.mean_decode_blocks <= 0.0
        {
            return decode_cost_blocks;
        }
        decode_cost_blocks
            * (decode_cost_blocks / self.mean_decode_blocks).powf(self.decode_load_exponent - 1.0)
    }

    fn decode_cost_blocks(
        self,
        decode_cost_blocks: f64,
        request_blocks: u64,
        active_requests: usize,
    ) -> f64 {
        let charged =
            self.charged_blocks(self.unshared_blocks(decode_cost_blocks, active_requests));
        let ratio = self.predicted_decode_blocks / request_blocks.max(1) as f64;
        let shaped_ratio = if self.decode_work_exponent == 1.0 {
            ratio
        } else {
            ratio.powf(self.decode_work_exponent)
        };
        self.decode_footprint_weight * charged + self.decode_work_scale * shaped_ratio * charged
    }
}

/// Values shared by every worker score in one selection.
#[derive(Default)]
struct PreparedRequest {
    min_prefill: usize,
    overlap_credit: f64,
    needs_decay: bool,
    needs_decode_subtraction: bool,
    belief: Option<DecodeWorkBelief>,
}

/// Tokens of the prompt this worker already holds, on the same basis the default scorer uses
/// for its prefill estimate: device-tier matches when the host reports tiers, else the host's
/// accounting estimate.
fn cached_tokens(context: &WorkerSelectionContext<'_>, cache: WorkerCacheInput<'_>) -> usize {
    if cache.has_tier_matches() {
        (cache.device_overlap_blocks() * f64::from(context.block_size()))
            .round()
            .max(0.0) as usize
    } else {
        cache.accounting_cache_estimate().1
    }
}

use dynamo_kv_router::plugins::worker_selection::WorkerCacheInput;

/// Reproduces the default scorer's cost with the decode term rescaled by the JSEW belief.
///
/// Every host weight is read from `KvRouterConfig` so the two costs stay in lockstep: any drift
/// would make an A/B measure this re-implementation rather than the decode-work hypothesis.
struct JsewScorer {
    parameters: Parameters,
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    shared_cache_multiplier: f64,
    decode_active_request_weight: f64,
    prefill_load_scale: f64,
    is_decode: bool,
    is_plain_decode: bool,
    prepared: PreparedRequest,
    ema: EmaBelief,
}

impl JsewScorer {
    fn new(
        config: &KvRouterConfig,
        parameters: Parameters,
        worker_label: &'static str,
        is_plain_decode: bool,
    ) -> Self {
        Self {
            parameters,
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            // Resolved the way the default policy resolves it: an unset multiplier means the
            // shared-cache type's own default.
            shared_cache_multiplier: config.shared_cache_multiplier.unwrap_or(
                match config.shared_cache_type {
                    SharedCacheType::None => 0.0,
                    SharedCacheType::Hicache => 0.5,
                },
            ),
            decode_active_request_weight: config.decode_active_request_weight,
            prefill_load_scale: config.prefill_load_scale,
            is_decode: worker_label == "decode",
            is_plain_decode,
            prepared: PreparedRequest::default(),
            ema: EmaBelief::default(),
        }
    }

    fn prepare(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
    ) -> Result<(), WorkerSelectionPolicyError> {
        // Plain disaggregated decode is load-only. Conditional decode retains cache credit.
        let overlap_credit = if self.is_plain_decode && !context.tracks_prefill_tokens() {
            0.0
        } else {
            self.overlap_score_credit
        };
        let needs_decay = context.tracks_prefill_tokens() && self.overlap_score_credit_decay > 0.0;
        let mut min_prefill = usize::MAX;
        let mut best_cached = 0usize;
        let mut unshared_sum = 0.0;
        for candidate in candidates.iter() {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            min_prefill = min_prefill.min(load.active_prefill_tokens());
            if let Some(cache) = candidate.cache() {
                best_cached = best_cached.max(cached_tokens(context, cache));
            }
            let blocks = load.decode_cost_blocks();
            unshared_sum += if self.parameters.decode_share_exponent == 0.0 || blocks <= 0.0 {
                blocks
            } else {
                blocks
                    * (load.active_requests().max(1) as f64)
                        .powf(self.parameters.decode_share_exponent)
            };
        }
        // The belief is folded unconditionally so the EMA stays current even when the oracle
        // supplies the prediction, keeping the fallback usable for undeclared requests.
        let uncached = context.prompt_tokens().saturating_sub(best_cached) as f64;
        let ema_tokens = self.ema.observe(uncached, self.parameters.ema_alpha);
        let predicted_tokens = match self.parameters.effective_work_predictor {
            EffectiveWorkPredictor::GroundTruthFutureDecode => context
                .expected_output_tokens()
                .map(f64::from)
                .unwrap_or(ema_tokens),
            EffectiveWorkPredictor::EmaUncachedTokens => ema_tokens,
        };
        let mean_decode_blocks = if candidates.is_empty() {
            0.0
        } else {
            unshared_sum / candidates.len() as f64
        };
        self.prepared = PreparedRequest {
            min_prefill: if needs_decay { min_prefill } else { 0 },
            overlap_credit,
            needs_decay,
            needs_decode_subtraction: self.is_decode
                && !context.tracks_prefill_tokens()
                && overlap_credit > 0.0,
            belief: Some(DecodeWorkBelief {
                predicted_decode_blocks: predicted_tokens / f64::from(context.block_size().max(1)),
                mean_decode_blocks,
                decode_work_scale: self.parameters.decode_work_scale,
                decode_footprint_weight: self.parameters.decode_footprint_weight,
                decode_work_exponent: self.parameters.decode_work_exponent,
                decode_load_exponent: self.parameters.decode_load_exponent,
                decode_share_exponent: self.parameters.decode_share_exponent,
            }),
        };
        Ok(())
    }

    fn score_worker(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let block_size = f64::from(context.block_size());
        let request_blocks = context.request_blocks().max(1);
        let (cached_tokens, credit) = if let Some(cache) = candidate.cache() {
            let (estimated_overlap, cached_tokens) = cache.accounting_cache_estimate();
            let device = if cache.has_tier_matches() {
                cache.device_overlap_blocks()
            } else {
                estimated_overlap
            };
            let shared_credit = if self.shared_cache_multiplier != 0.0 {
                let shared = cache
                    .shared_hits()
                    .map_or(0, |hits| hits.hits_beyond(device.round().max(0.0) as u32));
                self.shared_cache_multiplier * shared as f64
            } else {
                self.shared_cache_multiplier
            };
            let decay = if self.prepared.needs_decay {
                let excess = load
                    .active_prefill_tokens()
                    .saturating_sub(self.prepared.min_prefill) as f64
                    / block_size;
                1.0 / (1.0 + self.overlap_score_credit_decay * (excess / request_blocks as f64))
            } else {
                1.0
            };
            let credit = self.prepared.overlap_credit * decay * device
                + self.host_cache_hit_weight * cache.host_overlap_blocks()
                + self.disk_cache_hit_weight * cache.disk_overlap_blocks()
                + shared_credit;
            (cached_tokens, credit)
        } else {
            (0, 0.0)
        };
        let request_cost = if self.decode_active_request_weight != 0.0 {
            self.decode_active_request_weight * load.active_requests() as f64
        } else {
            self.decode_active_request_weight
        };
        // The only departure from the default cost.
        let decode = self
            .prepared
            .belief
            .expect("prepared before scoring")
            .decode_cost_blocks(
                load.decode_cost_blocks(),
                request_blocks,
                load.active_requests(),
            );
        let logit = if self.prepared.needs_decode_subtraction {
            (decode - credit).max(0.0) + request_cost
        } else {
            let raw_tokens = if !context.tracks_prefill_tokens() {
                0
            } else if load.is_available() {
                let uncached = context.prompt_tokens().saturating_sub(cached_tokens);
                (load.active_prefill_tokens() + uncached).saturating_add(cached_tokens)
            } else {
                context.prompt_tokens()
            };
            let prefill = (raw_tokens as f64 / block_size - credit).max(0.0);
            self.prefill_load_scale * prefill + decode + request_cost
        };
        Ok(logit * candidate.preferred_taint_multiplier().unwrap_or(1.0))
    }
}

impl WorkerScorer for JsewScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.prepare(context, candidates)?;
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = self.score_worker(context, candidate)?;
        }
        Ok(())
    }
}

/// Minimum cost with uniform tie-breaking, or temperature sampling, as the default picker does.
struct MinCostPicker {
    temperature: f64,
    probabilities: Vec<f64>,
}

impl WorkerPicker for MinCostPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        if context.pinned_worker().is_some() {
            return Ok(0);
        }
        let temperature = context
            .router_temperature_override()
            .unwrap_or(self.temperature);
        if temperature != 0.0 {
            return Ok(softmax_sample_row(
                candidates.iter().map(|c| c.cost()),
                temperature,
                fastrand::f64(),
                &mut self.probabilities,
            ));
        }
        let mut best_row = 0;
        let mut best_cost = f64::INFINITY;
        let mut ties = 0;
        for (row, candidate) in candidates.iter().enumerate() {
            let cost = candidate.cost();
            if cost < best_cost {
                best_row = row;
                best_cost = cost;
                ties = 1;
            } else if cost == best_cost {
                ties += 1;
                if fastrand::usize(0..ties) == 0 {
                    best_row = row;
                }
            }
        }
        Ok(best_row)
    }
}

fn softmax_sample_row(
    costs: impl Iterator<Item = f64> + Clone,
    temperature: f64,
    sample: f64,
    probabilities: &mut Vec<f64>,
) -> usize {
    let (min_cost, max_cost) = costs
        .clone()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(cost), hi.max(cost))
        });
    probabilities.clear();
    let count = costs.clone().count();
    if min_cost == max_cost {
        probabilities.resize(count, 1.0 / count as f64);
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
        probabilities.extend(costs.map(|cost| (cost / magnitude * scale - max_scaled).exp()));
    }
    let sum: f64 = probabilities.iter().sum();
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability / sum;
        if sample <= cumulative {
            return row;
        }
    }
    count - 1
}

fn policy(
    config: &KvRouterConfig,
    parameters: Parameters,
    role: WorkerType,
) -> WorkerSelectionPolicy {
    // Label pools the way the default does, and treat plain disaggregated decode as load-only
    // the way the default factory does, so the reproduced cost matches it role for role.
    let worker_label = role.default_selector_label();
    let is_plain_decode = role == WorkerType::Decode && !config.conditional_disagg_enabled;
    WorkerSelectionPolicy::new(
        config.clone(),
        worker_label,
        vec![Box::new(JsewScorer::new(
            config,
            parameters,
            worker_label,
            is_plain_decode,
        ))],
        Box::new(MinCostPicker {
            temperature: config.router_temperature,
            probabilities: Vec::new(),
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
        move |config: &KvRouterConfig, role, _partition| policy(config, parameters, role),
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

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;
    use crate::default::default_policy;

    const BLOCK_SIZE: u32 = 16;
    /// Ten blocks of prompt.
    const TEN_BLOCKS: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;

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

    /// One worker's signals: `(worker_id, overlap_blocks, active_decode_blocks, active_requests)`.
    type WorkerSpec = (u64, usize, usize, usize);

    /// Widen `(id, overlap, decode)` to one active request per worker, which is what the stock
    /// cost assumes when it charges a deduplicated footprint.
    fn unshared(workers: [(u64, usize, usize); 2]) -> [WorkerSpec; 2] {
        workers.map(|(id, overlap, decode)| (id, overlap, decode, 1))
    }

    fn request(workers: [WorkerSpec; 2], expected_output_tokens: Option<u32>) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: TEN_BLOCKS,
            lora_name: None,
            expected_output_tokens,
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
        };
        for (id, overlap_blocks, active_decode_blocks, active_requests) in workers {
            // Offline replay reports no per-tier breakdown, only the accounting estimate.
            request
                .overlap
                .effective_overlap_blocks
                .insert(worker(id), overlap_blocks as f64);
            request
                .overlap
                .effective_cached_tokens
                .insert(worker(id), overlap_blocks * BLOCK_SIZE as usize);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_decode_blocks,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    fn configs(workers: [WorkerSpec; 2]) -> HashMap<u64, TestWorker> {
        HashMap::from(workers.map(|(id, _, _, _)| (id, TestWorker)))
    }

    fn jsew_policy(parameters: Parameters) -> WorkerSelectionPolicy {
        policy(
            &KvRouterConfig::default(),
            parameters,
            WorkerType::Aggregated,
        )
    }

    /// Route a sequence of requests through one policy instance, so the EMA carries across them.
    fn select_seq<const N: usize>(
        policy: &WorkerSelectionPolicy,
        requests: [[WorkerSpec; 2]; N],
        expected_output_tokens: Option<u32>,
    ) -> Vec<WorkerWithDpRank> {
        requests
            .into_iter()
            .map(|workers| {
                let request = request(workers, expected_output_tokens);
                let configs = configs(workers);
                policy
                    .select_worker(WorkerSelectionInput::configured(
                        &configs,
                        &request,
                        request.eligibility(),
                        BLOCK_SIZE,
                    ))
                    .unwrap()
                    .worker
            })
            .collect()
    }

    fn select_with(parameters: Parameters, workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_seq(&jsew_policy(parameters), [unshared(workers)], None)[0]
    }

    /// Route the same request through Dynamo's default policy.
    fn select_stock(workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        let policy = default_policy(KvRouterConfig::default(), "test");
        select_seq(&policy, [unshared(workers)], None)[0]
    }

    /// Neutral parameters: keep the stock footprint term whole, add nothing on top. The exponents
    /// are pinned to their inert values since the tuned defaults bend the charge.
    fn neutral() -> Parameters {
        Parameters {
            decode_work_scale: 0.0,
            decode_footprint_weight: 1.0,
            decode_work_exponent: 1.0,
            decode_load_exponent: 1.0,
            decode_share_exponent: 0.0,
            ..Parameters::default()
        }
    }

    #[test]
    fn registers_its_policy_type() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();
        assert_eq!(POLICY_TYPE, "dynamo-jsew");
        // Re-registering the same type is what a duplicated catalog entry would do.
        assert!(register(&mut registry).is_err());
    }

    #[test]
    fn neutral_parameters_reproduce_the_default_choice() {
        // Cases where cache overlap and decode footprint disagree, so any drift in the ported cost
        // would show up as a different worker.
        for workers in [
            [(A, 0, 0), (B, 8, 40)],
            [(A, 6, 30), (B, 0, 0)],
            [(A, 10, 100), (B, 2, 5)],
            [(A, 3, 12), (B, 4, 20)],
            [(A, 0, 7), (B, 5, 9)],
        ] {
            assert_eq!(
                select_with(neutral(), workers),
                select_stock(workers),
                "diverged from the default policy on {workers:?}"
            );
        }
    }

    #[test]
    fn predicted_work_outweighs_cache_affinity() {
        // B holds five of the request's ten blocks but carries more decode backlog than A. Stock
        // cache credit outweighs those four extra decode blocks, and the neutral policy agrees.
        let workers = [(A, 0, 30), (B, 5, 34)];
        assert_eq!(select_with(neutral(), workers), worker(B));

        // Scaling the predicted-work term amplifies the decode gap until it swamps the five blocks
        // of cache credit, so the request goes to the worker that owes less work.
        let tuned = Parameters {
            decode_work_scale: 20.0,
            ..neutral()
        };
        assert_eq!(select_with(tuned, workers), worker(A));
    }

    #[test]
    fn dropping_the_footprint_weight_restores_cache_affinity() {
        // With the footprint term whole, A's empty decode queue outweighs B's full cache hit.
        let workers = [(A, 0, 0), (B, 10, 12)];
        assert_eq!(select_with(neutral(), workers), worker(A));

        // Discarding the footprint belief leaves only prefill load, so the cache hit decides.
        let tuned = Parameters {
            decode_footprint_weight: 0.0,
            decode_work_scale: 0.0,
            ..Parameters::default()
        };
        assert_eq!(select_with(tuned, workers), worker(B));
    }

    #[test]
    fn the_ema_belief_carries_across_requests() {
        // One request, routed twice against identical worker state but different prior traffic.
        let request = [(A, 0, 30), (B, 5, 34)];
        // Fully cached: this warm-up owes almost no prefill, so it teaches the EMA that traffic is
        // cheap. Fully uncached: the same shape teaches it that traffic is expensive.
        let warm_cache = [(A, 10, 0), (B, 10, 0)];
        let cold_cache = [(A, 0, 0), (B, 0, 0)];
        let parameters = Parameters {
            decode_work_scale: 20.0,
            ..neutral()
        };

        // After cheap traffic the predicted-work term stays small and cache credit decides.
        assert_eq!(
            select_seq(
                &jsew_policy(parameters),
                [unshared(warm_cache), unshared(request)],
                None
            )[1],
            worker(B)
        );
        // After expensive traffic the same term dominates, and the request avoids B's backlog.
        assert_eq!(
            select_seq(
                &jsew_policy(parameters),
                [unshared(cold_cache), unshared(request)],
                None
            )[1],
            worker(A)
        );
    }

    #[test]
    fn ground_truth_predictor_reads_the_declared_output_length() {
        let workers = [(A, 0, 30), (B, 5, 34)];
        let ground_truth = Parameters {
            effective_work_predictor: EffectiveWorkPredictor::GroundTruthFutureDecode,
            decode_work_scale: 20.0,
            ..neutral()
        };

        // A short declared completion leaves the predicted-work term small, so cache credit wins.
        assert_eq!(
            select_seq(&jsew_policy(ground_truth), [unshared(workers)], Some(1))[0],
            worker(B)
        );
        // A long one charges decode backlog hard enough to flip the choice.
        assert_eq!(
            select_seq(
                &jsew_policy(ground_truth),
                [unshared(workers)],
                Some(100_000)
            )[0],
            worker(A)
        );
    }

    #[test]
    fn a_convex_load_exponent_avoids_the_deeper_backlog() {
        // A holds no cache and a small backlog; B holds half the prompt and three more blocks of
        // backlog. Charged linearly, B's five blocks of cache credit cover those three blocks.
        let workers = [(A, 0, 8), (B, 5, 11)];
        assert_eq!(select_with(neutral(), workers), worker(B));

        // Bending the charge in the worker's own backlog costs B's deeper queue disproportionately
        // more than A's, which is enough to outweigh the cache hit.
        let convex = Parameters {
            decode_load_exponent: 3.0,
            ..neutral()
        };
        assert_eq!(select_with(convex, workers), worker(A));
    }

    #[test]
    fn the_share_exponent_un_shares_a_deduplicated_footprint() {
        // Both workers report the same decode footprint, but B's is the deduplicated total of six
        // sessions sharing a prefix while A's belongs to a single session. B also holds half this
        // request's prompt, so cache credit sends the request to B on the reported numbers.
        let workers: [WorkerSpec; 2] = [(A, 0, 20, 1), (B, 5, 20, 6)];
        assert_eq!(
            select_seq(&jsew_policy(neutral()), [workers], None)[0],
            worker(B)
        );

        // Charging the un-shared sum reveals that B's six sessions each carry that prefix through
        // every decode step, so the request avoids piling a seventh onto it.
        let unshare = Parameters {
            decode_share_exponent: 1.0,
            ..neutral()
        };
        assert_eq!(
            select_seq(&jsew_policy(unshare), [workers], None)[0],
            worker(A)
        );
    }

    #[test]
    fn neutral_exponents_leave_the_charge_untouched() {
        let belief = DecodeWorkBelief {
            predicted_decode_blocks: 37.0,
            mean_decode_blocks: 22.0,
            decode_work_scale: DEFAULT_DECODE_WORK_SCALE,
            decode_footprint_weight: DEFAULT_DECODE_FOOTPRINT_WEIGHT,
            decode_work_exponent: 1.0,
            decode_load_exponent: 1.0,
            decode_share_exponent: 0.0,
        };
        for blocks in [0.0, 7.0, 12.0, 40.0, 100.0] {
            let expected = DEFAULT_DECODE_FOOTPRINT_WEIGHT * blocks
                + DEFAULT_DECODE_WORK_SCALE * (37.0 / 10.0) * blocks;
            for active_requests in [0, 1, 7] {
                assert_eq!(
                    belief.decode_cost_blocks(blocks, 10, active_requests),
                    expected
                );
            }
        }
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let check = |f: fn(&mut Parameters, f64), v: f64| {
            let mut p = Parameters::default();
            f(&mut p, v);
            p.validate()
        };
        let scale = |p: &mut Parameters, v| p.decode_work_scale = v;
        let footprint = |p: &mut Parameters, v| p.decode_footprint_weight = v;
        let alpha = |p: &mut Parameters, v| p.ema_alpha = v;
        let work_exp = |p: &mut Parameters, v| p.decode_work_exponent = v;
        let load_exp = |p: &mut Parameters, v| p.decode_load_exponent = v;
        let share_exp = |p: &mut Parameters, v| p.decode_share_exponent = v;

        assert!(
            check(scale, -0.1).is_err()
                && check(scale, 20.1).is_err()
                && check(scale, f64::NAN).is_err()
        );
        assert!(check(footprint, -0.1).is_err() && check(footprint, 2.1).is_err());
        assert!(
            check(alpha, 0.0).is_err()
                && check(alpha, 1.1).is_err()
                && check(alpha, f64::NAN).is_err()
        );
        assert!(check(work_exp, 0.0).is_err() && check(work_exp, 4.1).is_err());
        assert!(check(load_exp, 0.0).is_err() && check(load_exp, f64::NAN).is_err());
        // Zero is the share exponent's neutral value, so it must be accepted.
        assert!(check(share_exp, 0.0).is_ok());
        assert!(check(share_exp, -0.1).is_err() && check(share_exp, 2.1).is_err());
        assert!(Parameters::default().validate().is_ok());
    }
}
