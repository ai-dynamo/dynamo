// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active-request cost for the stacked scorer stage.

use dynamo_kv_router::{
    WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicyError,
};

pub(crate) struct ActiveRequestsScorer;

impl WorkerScorer for ActiveRequestsScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
            *cost = load.active_requests() as f64;
        }
        Ok(())
    }
}
