// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `kvcr` cost: the default cost, plus credit for the blocks a worker could copy from a peer
//! at the shared-cache multiplier. It takes the default policy's parameters, picker and runtime
//! configuration; only the scorer differs.

use super::parameters::PolicyParameters;
use super::scorer::DefaultScorer;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

/// [`super::scorer::build`] for the `kvcr` preset: the same cost plus credit for the blocks a
/// worker could copy from a peer, at the shared-cache multiplier.
pub(super) fn build(
    config: &PolicyParameters,
    worker_label: &'static str,
) -> Box<dyn WorkerScorer> {
    match (
        config.decode_active_request_weight != 0.0,
        config.shared_cache_multiplier != 0.0,
    ) {
        // Without copy credit, the cost is the default policy's.
        (_, false) => super::scorer::build(config, worker_label),
        (false, true) => Box::new(CopyCreditScorer(DefaultScorer::<false, true>::new(
            config,
            worker_label,
        ))),
        (true, true) => Box::new(CopyCreditScorer(DefaultScorer::<true, true>::new(
            config,
            worker_label,
        ))),
    }
}

/// The default cost, with the blocks a worker could copy from its offered source counted toward
/// the shared-cache credit. It reads [`WorkerInputs::KV_COPY`], so the router commits the selected
/// worker to that copy. Built only when the multiplier gives copies credit.
struct CopyCreditScorer<const REQUEST_COST: bool>(DefaultScorer<REQUEST_COST, true>);

impl<const REQUEST_COST: bool> WorkerScorer for CopyCreditScorer<REQUEST_COST> {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.0.required_worker_inputs() | WorkerInputs::KV_COPY
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let prepared = self.0.prepare(context, candidates)?;
        for (candidate, cost) in candidates.iter().zip(costs) {
            *cost =
                self.0
                    .score_worker(&prepared, context, candidate, copyable_blocks(candidate))?;
        }
        Ok(())
    }
}

/// Blocks a copy would add beyond everything this worker already holds.
fn copyable_blocks(candidate: WorkerCandidate<'_>) -> f64 {
    let (Some(cache), Some(source)) = (
        candidate.cache(),
        candidate
            .kv_copy()
            .and_then(|copy| copy.source_prefix_blocks()),
    ) else {
        return 0.0;
    };
    let local =
        cache.device_overlap_blocks() + cache.host_overlap_blocks() + cache.disk_overlap_blocks();
    (f64::from(source) - local).max(0.0)
}
