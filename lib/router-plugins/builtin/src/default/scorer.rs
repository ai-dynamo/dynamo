// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache and load cost calculation. The host owns snapshots and validates finite scores.

use dynamo_kv_router::{
    KvRouterConfig, WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

pub(super) struct DefaultScorer {
    config: KvRouterConfig,
    worker_label: &'static str,
    plain_decode: bool,
    min_prefill: usize,
}

impl DefaultScorer {
    pub(super) fn new(
        config: KvRouterConfig,
        worker_label: &'static str,
        plain_decode: bool,
    ) -> Self {
        Self {
            config,
            worker_label,
            plain_decode,
            min_prefill: 0,
        }
    }
}

impl WorkerScorer for DefaultScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn prepare(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: &[WorkerCandidate],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.min_prefill = 0;
        if context.tracks_prefill_tokens() && self.config.overlap_score_credit_decay > 0.0 {
            let mut minimum = usize::MAX;
            for candidate in candidates {
                let load = candidate
                    .load()
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
                minimum = minimum.min(load.active_prefill_tokens());
            }
            self.min_prefill = minimum;
        }
        Ok(())
    }

    fn score(
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
        let shared = if context.has_tier_matches() {
            cache.shared_beyond_device_blocks()
        } else {
            context
                .shared_cache_hits()
                .map_or(0, |hits| hits.hits_beyond(device.round().max(0.0) as u32))
        };
        let decay =
            if context.tracks_prefill_tokens() && self.config.overlap_score_credit_decay > 0.0 {
                let excess = load
                    .active_prefill_tokens()
                    .saturating_sub(self.min_prefill) as f64
                    / context.block_size() as f64;
                1.0 / (1.0
                    + self.config.overlap_score_credit_decay
                        * (excess / context.request_blocks() as f64))
            } else {
                1.0
            };
        // Plain disaggregated decode is load-only. Conditional decode retains cache credit.
        let overlap_credit = if self.plain_decode && !context.tracks_prefill_tokens() {
            0.0
        } else {
            self.config.overlap_score_credit
        };
        let credit = overlap_credit * decay * device
            + self.config.host_cache_hit_weight * cache.host_overlap_blocks()
            + self.config.disk_cache_hit_weight * cache.disk_overlap_blocks()
            + self.config.shared_cache_multiplier * shared as f64;
        let request_cost = self.config.decode_active_request_weight * load.active_requests() as f64;
        let logit = if self.worker_label == "decode"
            && !context.tracks_prefill_tokens()
            && overlap_credit > 0.0
        {
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
            let prefill = (raw_tokens as f64 / context.block_size() as f64 - credit).max(0.0);
            self.config.prefill_load_scale * prefill + load.decode_cost_blocks() + request_cost
        };
        let cost = logit * candidate.preferred_taint_multiplier().unwrap_or(1.0);
        Ok(cost)
    }
}
