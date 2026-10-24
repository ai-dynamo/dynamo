// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicitly configured two-worker progress/affinity experiment.
//! Thresholds are workload calibration, not a completion-time predictor.

use dynamo_kv_router::plugins::WorkerSelectionPolicyProviderError;
use dynamo_kv_router::plugins::worker_selection::{WorkerInputView, WorkerSelectionContext};
use std::time::Duration;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProgressGroupParameters {
    pub(crate) source_active_requests: usize,
    pub(crate) min_prefix_tokens: usize,
    pub(crate) min_remaining_tokens: usize,
    pub(crate) max_progress_age_ms: u64,
}

impl ProgressGroupParameters {
    pub(crate) fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.source_active_requests == 0
            || self.source_active_requests >= 4096
            || self.min_prefix_tokens == 0
            || self.min_remaining_tokens == 0
            || !(1..=60_000).contains(&self.max_progress_age_ms)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "progress_group requires active requests in 1..4096, positive token thresholds, and age in 1..=60000 ms",
            ));
        }
        Ok(())
    }
}

pub(crate) fn select(
    parameters: ProgressGroupParameters,
    context: &WorkerSelectionContext<'_>,
    input: WorkerInputView<'_>,
) -> Option<usize> {
    context.session_context()?.parent_session_id()?;
    let progress = context.decode_progress()?;
    let candidates = input.candidates();
    let cache = input.cache()?;
    let load = input.load()?;
    // Wider topologies and disaggregated progress are not validated by this experiment.
    if candidates.len() != 2
        || cache.len() != 2
        || load.len() != 2
        || candidates[0].worker().worker_id == candidates[1].worker().worker_id
        || context.pinned_worker().is_some()
        || context.block_size() == 0
    {
        return None;
    }
    let target = context.affinity_target()?;
    let mut targets = candidates
        .iter()
        .enumerate()
        .filter_map(|(row, candidate)| {
            let worker = candidate.worker();
            (worker.worker_id == target.worker_id
                && target.dp_rank.is_none_or(|rank| rank == worker.dp_rank))
            .then_some(row)
        });
    let target = targets.next()?;
    if targets.next().is_some() {
        return None;
    }
    let overlaps = [
        cache.get(0)?.device_overlap_blocks(),
        cache.get(1)?.device_overlap_blocks(),
    ];
    if overlaps.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return None;
    }
    let required = context
        .prompt_tokens()
        .saturating_sub(context.block_size() as usize)
        .max(parameters.min_prefix_tokens) as f64;
    let warm = overlaps.map(|blocks| blocks * f64::from(context.block_size()) >= required);
    if warm == [true, true] {
        // There is no cold-prefix tradeoff; let the existing load/cache policy decide.
        return None;
    }
    let source = match (warm, overlaps) {
        ([true, false], [_, 0.0]) => 0,
        ([false, true], [0.0, _]) => 1,
        _ => return None,
    };
    let other = 1 - source;
    if load[source].active_requests() != parameters.source_active_requests
        || load[source].active_prefill_tokens() != 0
    {
        return None;
    }
    if target == source {
        if load[other].active_requests() != 0 {
            return None;
        }
    } else if load[target].active_requests() != 1 || load[target].active_prefill_tokens() == 0 {
        return None;
    }
    let source_worker = candidates[source].worker();
    let source_progress = progress
        .entries()
        .iter()
        .find(|row| row.worker() == source_worker)?;
    if source_progress.observed_requests() != load[source].active_requests()
        || source_progress.observed_age() > Duration::from_millis(parameters.max_progress_age_ms)
    {
        return None;
    }
    let remaining = source_progress.remaining_token_budget();
    // Preserve an existing move only after validating the source observations too.
    if target != source {
        return Some(target);
    }
    Some(if remaining > parameters.min_remaining_tokens {
        other
    } else {
        source
    })
}
