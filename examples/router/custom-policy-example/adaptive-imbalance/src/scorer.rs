// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scores candidates with the current weights and feeds the per-worker load snapshot the loop reads.
//! Per request it takes one guard read of the weights; it stores a new load sample at most once per
//! `interval`, the rate at which the loop reads it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::{
    WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

use crate::math::cost_for;
use crate::state::SharedState;

pub struct AdaptiveScorer {
    shared: Arc<SharedState>,
    /// Reused per request so the hot path does not allocate.
    active: Vec<f64>,
    /// The loop samples once per `interval`, so publishing more often only costs stores nobody reads.
    interval: Duration,
    last_publish: Option<Instant>,
}

impl AdaptiveScorer {
    pub fn new(shared: Arc<SharedState>, interval: Duration) -> Self {
        Self {
            shared,
            active: Vec::new(),
            interval,
            last_publish: None,
        }
    }
}

impl WorkerScorer for AdaptiveScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let weights = self.shared.weights();
        let track_prefill = context.tracks_prefill_tokens();
        let block_size = context.block_size() as f64;
        let prompt_tokens = context.prompt_tokens();
        self.active.clear();

        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            let active_requests = load.active_requests() as f64;
            // A worker without a load projection is scored as idle, but kept out of the loop's sample
            // so a missing observation cannot read as imbalance.
            if load.is_available() {
                self.active.push(active_requests);
            }
            // Same token accounting as the default selector: an observed worker counts its prefill
            // backlog plus the larger of the prompt and its cached-token estimate; an unobserved one
            // counts the prompt only. Both are divided by the block size without rounding.
            let (request_tokens, backlog_tokens) = if load.is_available() {
                let (_, cached_tokens) = cache.accounting_cache_estimate();
                (
                    prompt_tokens.max(cached_tokens),
                    load.active_prefill_tokens(),
                )
            } else {
                (prompt_tokens, 0)
            };
            *cost = cost_for(
                weights.set,
                request_tokens as f64 / block_size,
                backlog_tokens as f64 / block_size,
                track_prefill,
                cache.device_overlap_blocks(),
                load.decode_cost_blocks(),
                active_requests,
            );
        }

        // The loop reads this; publishing here needs no new router hook and costs one Arc store per interval.
        // A pinned request or one with fewer than two observed workers says nothing about pool balance, so it
        // neither publishes nor resets the interval clock: the next representative request publishes instead.
        if self.active.len() < 2 || context.pinned_worker().is_some() {
            return Ok(());
        }
        let now = Instant::now();
        if self
            .last_publish
            .is_none_or(|last| now.saturating_duration_since(last) >= self.interval)
        {
            self.shared.publish_load(self.active.clone());
            self.last_publish = Some(now);
        }
        Ok(())
    }
}
