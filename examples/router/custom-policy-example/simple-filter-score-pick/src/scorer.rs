// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active-request scorer for the `simple-filter-score-pick` policy.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Scores active requests above the least-loaded surviving candidate.
#[derive(Default)]
pub(crate) struct ActiveRequestsScorer {
    minimum: usize,
}

impl WorkerScorer for ActiveRequestsScorer {
    /// Requests load inputs for the active-request count.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    /// Prepare a request-local minimum after all filters have run.
    fn prepare(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidates: &[WorkerCandidate],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.minimum = usize::MAX;
        for candidate in candidates {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            self.minimum = self.minimum.min(load.active_requests());
        }
        Ok(())
    }

    /// Return a relative load cost. Subtracting the minimum preserves the ranking.
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok((load.active_requests() - self.minimum) as f64)
    }
}
