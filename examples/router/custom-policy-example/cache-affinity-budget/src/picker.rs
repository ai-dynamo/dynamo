// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Effective-cache-first selection within a bounded active-load cost delta.

use dynamo_kv_router::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

pub(crate) struct CacheAffinityBudgetPicker {
    pub(crate) max_load_cost_delta_blocks: f64,
}

impl WorkerPicker for CacheAffinityBudgetPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let min_cost = candidates
            .iter()
            .map(|candidate| candidate.cost())
            .min_by(f64::total_cmp)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        candidates
            .iter()
            .zip(cache)
            .enumerate()
            .filter(|(_, (candidate, _))| {
                candidate.cost() - min_cost <= self.max_load_cost_delta_blocks
            })
            .max_by(|(_, (left, left_cache)), (_, (right, right_cache))| {
                left_cache
                    .effective_overlap_blocks()
                    .total_cmp(&right_cache.effective_overlap_blocks())
                    .then_with(|| right.cost().total_cmp(&left.cost()))
                    .then_with(|| right.worker().cmp(&left.worker()))
            })
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no worker within load budget"))
    }
}
