// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-aware picker for the `simple-filter-score-pick` policy.

use dynamo_kv_router::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};

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
