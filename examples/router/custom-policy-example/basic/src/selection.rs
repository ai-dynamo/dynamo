// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load scorer and lowest-cost picker for the `least-busy` policy.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Scores each candidate by its current number of active requests.
pub(crate) struct LeastBusyScorer;

impl WorkerScorer for LeastBusyScorer {
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

/// Picks the candidate with the lowest accumulated scorer cost.
pub(crate) struct LowestCostPicker;

impl WorkerPicker for LowestCostPicker {
    /// Returns the row with the lowest cost.
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}
