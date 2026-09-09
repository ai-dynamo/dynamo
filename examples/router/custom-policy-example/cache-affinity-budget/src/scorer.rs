// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Modeled active-load cost for cache-affinity budgeting.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

pub(crate) struct ActiveLoadScorer {
    pub(crate) prefill_load_scale: f64,
    pub(crate) active_request_weight: f64,
}

impl WorkerScorer for ActiveLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let active_prefill_blocks =
            load.active_prefill_tokens() as f64 / context.block_size() as f64;
        Ok(self.prefill_load_scale * active_prefill_blocks
            + load.decode_cost_blocks()
            + self.active_request_weight * load.active_requests() as f64)
    }
}
