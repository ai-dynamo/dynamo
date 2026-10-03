// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Picker for the `fetch-to-idle` policy.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Keeps a request on the worker that caches most of its prefix, unless that worker is busy.
/// Otherwise selects the least-loaded worker, which the fetch policy can then fill from the
/// busy holder.
pub(crate) struct FetchToIdlePicker {
    /// A cache holder with at least this many active requests is busy.
    pub(crate) busy_active_requests: usize,
}

impl WorkerPicker for FetchToIdlePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
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
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let cached_blocks = |row: usize| {
            cache.get(row).map_or(0.0, |cache| {
                cache.device_overlap_blocks() + cache.host_overlap_blocks()
            })
        };
        let active_requests = |row: usize| loads[row].active_requests();

        // Ties go to the lower worker ID, so the choice does not depend on row order.
        let holder = (0..candidates.len()).max_by(|&left, &right| {
            cached_blocks(left)
                .total_cmp(&cached_blocks(right))
                .then_with(|| candidates[right].worker().cmp(&candidates[left].worker()))
        });
        let least_loaded = (0..candidates.len())
            .min_by_key(|&row| (active_requests(row), candidates[row].worker()));

        match (holder, least_loaded) {
            (Some(holder), _)
                if cached_blocks(holder) > 0.0
                    && active_requests(holder) < self.busy_active_requests =>
            {
                Ok(holder)
            }
            (_, Some(least_loaded)) => Ok(least_loaded),
            _ => Err(WorkerSelectionPolicyError::failed("no eligible worker")),
        }
    }
}
