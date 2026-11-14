// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quadratic prefill-cost scorer for the `prefill-cost-score-pick` policy.

use dynamo_kv_router::plugins::prelude::*;

/// `cost(q, c) = uncached_squared * q * q + uncached_times_cached * q * c + uncached * q`.
#[derive(Clone, Copy)]
pub(crate) struct PrefillCostModel {
    pub(crate) uncached_squared: f64,
    pub(crate) uncached_times_cached: f64,
    pub(crate) uncached: f64,
}

impl PrefillCostModel {
    /// Cost of one prefill with `uncached` new and `cached` reused prompt tokens.
    pub(crate) fn request_cost(&self, uncached: f64, cached: f64) -> f64 {
        self.uncached_squared * uncached * uncached
            + self.uncached_times_cached * uncached * cached
            + self.uncached * uncached
    }

    /// Summed cost of a worker's active prefills. The model is linear in its terms, so the
    /// worker's sums give the exact total without the individual requests.
    pub(crate) fn active_cost(&self, sums: ActivePrefillSums) -> f64 {
        self.uncached_squared * sums.uncached_tokens_squared() as f64
            + self.uncached_times_cached * sums.uncached_times_cached_tokens() as f64
            + self.uncached * sums.uncached_tokens() as f64
    }
}

/// Scores each worker by its active prefill cost plus the incoming request's cost on it.
pub(crate) struct PrefillCostScorer {
    pub(crate) model: PrefillCostModel,
}

impl WorkerScorer for PrefillCostScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::ACTIVE_PREFILL_SUMS
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let prompt = context.isl_tokens() as f64;
        for (candidate, cost) in candidates.iter().zip(costs) {
            let sums = candidate.active_prefill_sums().ok_or_else(|| {
                WorkerSelectionPolicyError::failed("active prefill sums unavailable")
            })?;
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            // The router's cached-token estimate for this worker, clamped to the prompt.
            let cached = (cache.accounting_cache_estimate().1 as f64).min(prompt);
            *cost = self.model.active_cost(sums) + self.model.request_cost(prompt - cached, cached);
        }
        Ok(())
    }
}
