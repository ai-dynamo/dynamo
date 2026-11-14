// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scores candidates with the current weights and feeds the per-worker load snapshot the loop reads.
//! Per request it takes one guard read of the weights; it stores a new load sample at most once per
//! `interval`, the rate at which the loop reads it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dynamo_kv_router::plugins::prelude::*;

use crate::math::cost_for;
use crate::state::SharedState;

/// `last_publish_ns` value before the first sample.
const NEVER: u64 = u64::MAX;

pub struct AdaptiveScorer {
    shared: Arc<SharedState>,
    /// The loop samples once per `interval`, so publishing more often only costs stores nobody reads.
    interval: Duration,
    /// Reference point for `last_publish_ns`.
    epoch: Instant,
    /// Nanoseconds after `epoch` of the last published sample, or `NEVER`. Concurrent selections
    /// race for each interval with a compare-and-swap, so only one of them builds and stores the sample.
    last_publish_ns: AtomicU64,
}

impl AdaptiveScorer {
    pub fn new(shared: Arc<SharedState>, interval: Duration) -> Self {
        Self {
            shared,
            interval,
            epoch: Instant::now(),
            last_publish_ns: AtomicU64::new(NEVER),
        }
    }

    /// Claims the current interval for this selection. Returns `false` when another selection
    /// published less than `interval` ago or won the race for this interval.
    fn claim_publish(&self) -> bool {
        let now = self.epoch.elapsed().as_nanos().min(u128::from(NEVER - 1)) as u64;
        let interval = self.interval.as_nanos().min(u128::from(NEVER)) as u64;
        let last = self.last_publish_ns.load(Ordering::Relaxed);
        if last != NEVER && now.saturating_sub(last) < interval {
            return false;
        }
        self.last_publish_ns
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
}

impl WorkerScorer for AdaptiveScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let weights = self.shared.weights();
        let track_prefill = context.tracks_prefill_tokens();
        let block_size = context.block_size() as f64;
        let isl_tokens = context.isl_tokens();
        let mut observed = 0usize;

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
                observed += 1;
            }
            // Same token accounting as the default selector: an observed worker counts its prefill
            // backlog plus the larger of the prompt and its cached-token estimate; an unobserved one
            // counts the prompt only. Both are divided by the block size without rounding.
            let (request_tokens, backlog_tokens) = if load.is_available() {
                let (_, cached_tokens) = cache.accounting_cache_estimate();
                (isl_tokens.max(cached_tokens), load.active_prefill_tokens())
            } else {
                (isl_tokens, 0)
            };
            // Preferred routing constraints scale the cost as in the default selector.
            *cost = cost_for(
                weights.set,
                request_tokens as f64 / block_size,
                backlog_tokens as f64 / block_size,
                track_prefill,
                cache.device_overlap_blocks(),
                load.decode_cost_blocks(),
                active_requests,
            ) * candidate.preferred_taint_multiplier().unwrap_or(1.0);
        }

        // The loop reads this; publishing here needs no new router hook and costs one Arc store per interval.
        // A pinned request, a preview, or one with fewer than two observed workers says nothing about pool
        // balance, so it neither publishes nor resets the interval clock: the next representative request
        // publishes instead.
        if observed < 2 || context.pinned_worker().is_some() || context.is_preview() {
            return Ok(());
        }
        if self.claim_publish() {
            let active = candidates
                .iter()
                .filter_map(|candidate| candidate.load())
                .filter(|load| load.is_available())
                .map(|load| load.active_requests() as f64)
                .collect();
            self.shared.publish_load(active);
        }
        Ok(())
    }
}
