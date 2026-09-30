// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduling profiles from the llm-d inference scheduler
//! (<https://github.com/llm-d/llm-d-inference-scheduler>).

pub(crate) mod optimized_baseline;
pub(crate) mod precise_prefix;

use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::WorkerSelectionPolicyRegistryError;
use dynamo_kv_router::plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerInputView, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

use crate::signals::rotate_by_worker;

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    optimized_baseline::register(registry)?;
    precise_prefix::register(registry)
}

/// llm-d's max-score picker: the lowest total cost, rotating among ties where llm-d shuffles.
#[derive(Default)]
struct MaxScorePicker {
    rotation: usize,
    tied: Vec<usize>,
}

impl MaxScorePicker {
    fn pick_lowest(
        &mut self,
        candidates: &[ScoredWorkerCandidate],
        rows: impl Iterator<Item = usize> + Clone,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let lowest = rows
            .clone()
            .map(|row| candidates[row].cost())
            .min_by(f64::total_cmp)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        self.tied.clear();
        self.tied
            .extend(rows.filter(|&row| candidates[row].cost() == lowest));
        let row = rotate_by_worker(
            &mut self.tied,
            |row| candidates[row].worker(),
            self.rotation,
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        self.rotation = self.rotation.wrapping_add(1);
        Ok(row)
    }
}

impl WorkerPicker for MaxScorePicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        self.pick_lowest(candidates, 0..candidates.len())
    }
}
