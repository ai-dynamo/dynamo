// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_kv_router::plugins::prelude::*;
use dynamo_kv_router::plugins::worker_selection::experimental::decode_progress::{
    DECODE_PROGRESS, DecodeProgressByRow,
};

use crate::{Parameters, fallback};

pub(crate) struct HotspotProgressPicker {
    pub(crate) parameters: Parameters,
    pub(crate) aggregated: bool,
}

impl WorkerPicker for HotspotProgressPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.aggregated {
            WorkerInputs::CACHE | WorkerInputs::LOAD | DECODE_PROGRESS
        } else {
            WorkerInputs::NONE
        }
    }

    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if !self.aggregated {
            return Err(WorkerSelectionPolicyError::failed(
                "hotspot-progress supports aggregated workers only",
            ));
        }
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        hotspot_row(self.parameters, context, input, cache, load)
            .or_else(|| fallback::select(cache, load, context.request_blocks()))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn hotspot_row(
    parameters: Parameters,
    context: &WorkerSelectionContext<'_>,
    input: WorkerInputView<'_>,
    cache: WorkerCacheInputs<'_>,
    load: &[WorkerLoadInput],
) -> Option<usize> {
    context.session_context()?.parent_session_id()?;
    let candidates = input.candidates();
    if candidates.len() != 2
        || cache.len() != 2
        || load.len() != 2
        || candidates[0].worker().worker_id == candidates[1].worker().worker_id
        || context.pinned_worker().is_some()
        || context.block_size() == 0
        || load.iter().any(|row| !row.is_available())
    {
        return None;
    }
    let affinity = context.affinity_target()?;
    let target = candidates.iter().position(|candidate| {
        let worker = candidate.worker();
        worker.worker_id == affinity.worker_id
            && affinity.dp_rank.is_none_or(|rank| rank == worker.dp_rank)
    })?;
    let overlaps = [
        cache.get(0)?.device_overlap_blocks(),
        cache.get(1)?.device_overlap_blocks(),
    ];
    if overlaps
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }
    let required = context
        .isl_tokens()
        .saturating_sub(context.block_size() as usize)
        .max(parameters.min_prefix_tokens) as f64;
    let warm = overlaps.map(|blocks| blocks * f64::from(context.block_size()) >= required);
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
    let progress = input.decode_progress(source)?;
    if progress.observed_requests() != load[source].active_requests()
        || progress.observed_age()? > Duration::from_millis(parameters.max_progress_age_ms)
    {
        return None;
    }
    // A sibling's cold target stays valid only while its source observations are valid too.
    if target != source {
        return Some(target);
    }
    Some(
        if progress.remaining_token_budget() > parameters.min_remaining_tokens {
            other
        } else {
            source
        },
    )
}
