// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache and load cost calculation. The host owns snapshots and validates finite scores.

use super::parameters::PolicyParameters;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Resolve optional terms once so disabled weights add neither branches nor conversions to
/// each worker score. The shared-cache specialization also omits range traversal code.
pub(super) fn build(
    config: &PolicyParameters,
    worker_label: &'static str,
    plain_decode: bool,
) -> Box<dyn WorkerScorer> {
    match (
        config.decode_active_request_weight != 0.0,
        config.shared_cache_multiplier != 0.0,
    ) {
        (false, false) => Box::new(DefaultScorer::<false, false>::new(
            config,
            worker_label,
            plain_decode,
        )),
        (false, true) => Box::new(DefaultScorer::<false, true>::new(
            config,
            worker_label,
            plain_decode,
        )),
        (true, false) => Box::new(DefaultScorer::<true, false>::new(
            config,
            worker_label,
            plain_decode,
        )),
        (true, true) => Box::new(DefaultScorer::<true, true>::new(
            config,
            worker_label,
            plain_decode,
        )),
    }
}

struct DefaultScorer<const REQUEST_COST: bool, const SHARED_CREDIT: bool> {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    shared_cache_multiplier: f64,
    decode_active_request_weight: f64,
    prefill_load_scale: f64,
    is_decode: bool,
    plain_decode: bool,
    prepared: PreparedRequest,
}

/// Values shared by every worker score in one selection. Preparation keeps their conversions
/// and role checks outside the per-worker trait call without changing floating-point arithmetic.
#[derive(Default)]
struct PreparedRequest {
    min_prefill: usize,
    block_size: IntegerDivisor,
    request_blocks: IntegerDivisor,
    overlap_credit: f64,
    use_decay: bool,
    subtract_from_decode: bool,
}

/// Division by a power-of-two integer has an exact reciprocal. Both inputs to this helper
/// originate as unsigned token/block counts, so neither result can underflow or overflow.
/// Other divisors retain ordinary division to preserve the score's rounding exactly.
#[derive(Clone, Copy)]
struct IntegerDivisor {
    value: f64,
    reciprocal: Option<f64>,
}

impl IntegerDivisor {
    fn new(value: u64) -> Self {
        Self {
            value: value as f64,
            reciprocal: value.is_power_of_two().then(|| 1.0 / value as f64),
        }
    }

    #[inline]
    fn divide(self, numerator: f64) -> f64 {
        match self.reciprocal {
            Some(reciprocal) => numerator * reciprocal,
            None => numerator / self.value,
        }
    }
}

impl Default for IntegerDivisor {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<const REQUEST_COST: bool, const SHARED_CREDIT: bool>
    DefaultScorer<REQUEST_COST, SHARED_CREDIT>
{
    pub(super) fn new(
        config: &PolicyParameters,
        worker_label: &'static str,
        plain_decode: bool,
    ) -> Self {
        Self {
            overlap_score_credit: config.overlap_score_credit,
            overlap_score_credit_decay: config.overlap_score_credit_decay,
            host_cache_hit_weight: config.host_cache_hit_weight,
            disk_cache_hit_weight: config.disk_cache_hit_weight,
            shared_cache_multiplier: config.shared_cache_multiplier,
            decode_active_request_weight: config.decode_active_request_weight,
            prefill_load_scale: config.prefill_load_scale,
            is_decode: worker_label == "decode",
            plain_decode,
            prepared: PreparedRequest::default(),
        }
    }
}

impl<const REQUEST_COST: bool, const SHARED_CREDIT: bool>
    DefaultScorer<REQUEST_COST, SHARED_CREDIT>
{
    fn prepare(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: &[WorkerCandidate],
    ) -> Result<(), WorkerSelectionPolicyError> {
        // Plain disaggregated decode is load-only. Conditional decode retains cache credit.
        let overlap_credit = if self.plain_decode && !context.tracks_prefill_tokens() {
            0.0
        } else {
            self.overlap_score_credit
        };
        self.prepared = PreparedRequest {
            min_prefill: 0,
            block_size: IntegerDivisor::new(u64::from(context.block_size())),
            request_blocks: IntegerDivisor::new(context.request_blocks()),
            overlap_credit,
            use_decay: context.tracks_prefill_tokens() && self.overlap_score_credit_decay > 0.0,
            subtract_from_decode: self.is_decode
                && !context.tracks_prefill_tokens()
                && overlap_credit > 0.0,
        };
        if self.prepared.use_decay {
            let mut minimum = usize::MAX;
            for candidate in candidates {
                let load = candidate
                    .load()
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
                minimum = minimum.min(load.active_prefill_tokens());
            }
            self.prepared.min_prefill = minimum;
        }
        Ok(())
    }

    fn score_worker(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let (estimated_overlap, cached_tokens) = cache.accounting_cache_estimate();
        let device = if context.has_tier_matches() {
            cache.device_overlap_blocks()
        } else {
            estimated_overlap
        };
        let shared_credit = if SHARED_CREDIT {
            let shared = if context.has_tier_matches() {
                cache.shared_beyond_device_blocks()
            } else {
                context
                    .shared_cache_hits()
                    .map_or(0, |hits| hits.hits_beyond(device.round().max(0.0) as u32))
            };
            self.shared_cache_multiplier * shared as f64
        } else {
            // Preserve signed zero when the configured weight is -0.0.
            self.shared_cache_multiplier
        };
        let decay = if self.prepared.use_decay {
            let excess = self.prepared.block_size.divide(
                load.active_prefill_tokens()
                    .saturating_sub(self.prepared.min_prefill) as f64,
            );
            1.0 / (1.0
                + self.overlap_score_credit_decay * self.prepared.request_blocks.divide(excess))
        } else {
            1.0
        };
        let credit = self.prepared.overlap_credit * decay * device
            + self.host_cache_hit_weight * cache.host_overlap_blocks()
            + self.disk_cache_hit_weight * cache.disk_overlap_blocks()
            + shared_credit;
        let request_cost = if REQUEST_COST {
            self.decode_active_request_weight * load.active_requests() as f64
        } else {
            self.decode_active_request_weight
        };
        let logit = if self.prepared.subtract_from_decode {
            (load.decode_cost_blocks() - credit).max(0.0) + request_cost
        } else {
            let raw_tokens = if !context.tracks_prefill_tokens() {
                0
            } else if load.is_available() {
                let uncached = context.prompt_tokens().saturating_sub(cached_tokens);
                (load.active_prefill_tokens() + uncached).saturating_add(cached_tokens)
            } else {
                context.prompt_tokens()
            };
            let prefill = (self.prepared.block_size.divide(raw_tokens as f64) - credit).max(0.0);
            self.prefill_load_scale * prefill + load.decode_cost_blocks() + request_cost
        };
        let cost = logit * candidate.preferred_taint_multiplier().unwrap_or(1.0);
        Ok(cost)
    }
}

impl<const REQUEST_COST: bool, const SHARED_CREDIT: bool> WorkerScorer
    for DefaultScorer<REQUEST_COST, SHARED_CREDIT>
{
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: &[WorkerCandidate],
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.prepare(context, candidates)?;
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost = self.score_worker(context, candidate)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::IntegerDivisor;

    #[test]
    fn prepared_division_preserves_score_rounding() {
        let counts = [0, 1, 7, 127, 2048, (1 << 53) - 1, 1 << 53, u64::MAX];
        for block_size in (0..32)
            .map(|shift| 1u64 << shift)
            .chain([3, 17, 63, u32::MAX as u64])
        {
            for request_blocks in (0..64).map(|shift| 1u64 << shift).chain([3, 127, u64::MAX]) {
                for count in counts {
                    let blocks = IntegerDivisor::new(block_size).divide(count as f64);
                    let expected = count as f64 / block_size as f64;
                    assert_eq!(blocks.to_bits(), expected.to_bits());
                    assert_eq!(
                        IntegerDivisor::new(request_blocks).divide(blocks).to_bits(),
                        (expected / request_blocks as f64).to_bits(),
                    );
                }
            }
        }
    }
}
