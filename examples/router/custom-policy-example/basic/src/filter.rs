// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-affinity filter for the `least-busy` policy.

use dynamo_kv_router::{
    WorkerCandidate, WorkerFilter, WorkerInputs, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Keeps workers whose effective cache overlap meets the configured minimum.
pub(crate) struct MinimumEffectiveOverlapFilter {
    pub(crate) min_effective_overlap_blocks: f64,
}

impl WorkerFilter for MinimumEffectiveOverlapFilter {
    /// Requests cache inputs before the scorer runs.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    /// Keeps a candidate when its effective cache overlap meets the minimum.
    fn keep(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        Ok(cache.effective_overlap_blocks() >= self.min_effective_overlap_blocks)
    }
}
