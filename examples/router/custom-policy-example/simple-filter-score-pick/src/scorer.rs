// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active-request scorer for the `simple-filter-score-pick` policy.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

/// Scores active requests above the least-loaded surviving candidate.
pub(crate) struct ActiveRequestsScorer;

impl WorkerScorer for ActiveRequestsScorer {
    /// Requests load inputs for the active-request count.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    /// Find the batch minimum and write one relative load cost per worker.
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let mut minimum = usize::MAX;
        for candidate in candidates.iter() {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            minimum = minimum.min(load.active_requests());
        }
        for (candidate, cost) in candidates.iter().zip(costs) {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            *cost = (load.active_requests() - minimum) as f64;
        }
        Ok(())
    }
}
