// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load scorers for prefill and decode routing partitions.

use dynamo_kv_router::{
    WorkerCandidate, WorkerInputs, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// Scores prefill workers by active prefill tokens.
pub(crate) struct PrefillLoadScorer;

impl WorkerScorer for PrefillLoadScorer {
    /// Requests load inputs for active prefill tokens.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    /// Returns the active prefill-token count as a lower-is-better cost.
    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok(load.active_prefill_tokens() as f64)
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
