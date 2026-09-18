// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active-request scorer for the `simple-filter-score-pick` policy.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Scores each candidate by its current number of active requests.
pub(crate) struct ActiveRequestsScorer;

impl WorkerScorer for ActiveRequestsScorer {
    /// Requests load inputs for the active-request count.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    /// Returns the active-request count as a lower-is-better cost.
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok(load.active_requests() as f64)
    }
}
