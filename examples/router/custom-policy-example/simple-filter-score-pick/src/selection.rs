// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scorer and request-aware picker for the `simple-filter-score-pick` policy.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
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

/// Uses device affinity for tool results and total cost for other requests.
pub(crate) struct RequestAwarePicker;

impl WorkerPicker for RequestAwarePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if context
            .session_context()
            .and_then(|session| session.input_trigger())
            == Some(WorkerSelectionInputTrigger::ToolResult)
        {
            return input
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.device_overlap_blocks()
                        .total_cmp(&right.device_overlap_blocks())
                })
                .map(|(row, _)| row)
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}
