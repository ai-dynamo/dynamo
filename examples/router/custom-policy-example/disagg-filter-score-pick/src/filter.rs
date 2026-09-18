// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-cache filter shared by prefill and decode workers.

use dynamo_kv_router::{
    WorkerCandidate, WorkerFilter, WorkerInputs, WorkerSelectionContext, WorkerSelectionPolicyError,
};

pub(crate) struct MinimumDeviceOverlapFilter {
    pub(crate) min_device_overlap_blocks: f64,
}

impl WorkerFilter for MinimumDeviceOverlapFilter {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn keep(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<bool, WorkerSelectionPolicyError> {
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        Ok(cache.device_overlap_blocks() >= self.min_device_overlap_blocks)
    }
}
