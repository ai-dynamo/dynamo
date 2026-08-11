// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load scorers for prefill and decode routing partitions.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Scores prefill workers by active work and uncached request blocks.
pub(crate) struct PrefillLoadScorer;

impl WorkerScorer for PrefillLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let cache = candidate
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let active_prefill_blocks =
            load.active_prefill_tokens() as f64 / context.block_size() as f64;
        let uncached_request_blocks =
            (context.request_blocks() as f64 - cache.device_overlap_blocks()).max(0.0);
        Ok(active_prefill_blocks + uncached_request_blocks)
    }
}

/// Scores decode workers by their projected decode cost in blocks.
pub(crate) struct DecodeLoadScorer;

impl WorkerScorer for DecodeLoadScorer {
    /// Requests load inputs for projected decode cost.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    /// Returns the projected decode cost as a lower-is-better cost.
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok(load.decode_cost_blocks())
    }
}
