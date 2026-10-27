// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Uncached-block cost for the stacked scorer stage.

use dynamo_kv_router::{
    WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

pub(crate) struct UncachedBlocksScorer;

impl WorkerScorer for UncachedBlocksScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let reusable = if cache.prefix_cache_read_allowed() == Some(false) {
                0.0
            } else {
                cache.device_overlap_blocks()
            };
            *cost = (context.request_blocks() as f64 - reusable).max(0.0);
        }
        Ok(())
    }
}
