// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pickers for prefill and decode routing partitions.

use dynamo_kv_router::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Finds the candidate row with the lowest accumulated scorer cost.
fn lowest_cost_row(input: WorkerInputView<'_>) -> Result<usize, WorkerSelectionPolicyError> {
    input
        .candidates()
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
        .map(|(row, _)| row)
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
}

/// Picks the prefill candidate with the lowest total cost.
pub(crate) struct PrefillPicker;

impl WorkerPicker for PrefillPicker {
    /// Returns the prefill candidate row with the lowest cost.
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        lowest_cost_row(input)
    }
}

/// Picks the decode candidate with the lowest total cost.
pub(crate) struct DecodePicker;

impl WorkerPicker for DecodePicker {
    /// Returns the decode candidate row with the lowest cost.
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        lowest_cost_row(input)
    }
}

/// Rejects worker-pool roles that this disaggregated example does not support.
pub(crate) struct UnsupportedWorkerTypePicker {
    worker_type: &'static str,
}

impl UnsupportedWorkerTypePicker {
    pub(crate) fn new(worker_type: &'static str) -> Self {
        Self { worker_type }
    }
}

impl WorkerPicker for UnsupportedWorkerTypePicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        Err(WorkerSelectionPolicyError::failed(format!(
            "disagg-filter-score-pick does not support worker type {:?}; expected prefill or decode",
            self.worker_type
        )))
    }
}
