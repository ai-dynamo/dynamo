// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `least-loaded`: the worker with the fewest active requests.

use super::{Samples, ThreadRandom, active_requests, no_candidates};
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// The worker with the fewest active requests, ties broken at random.
pub const LEAST_LOADED: &str = "least-loaded";

pub(super) struct LeastLoadedPicker;

impl WorkerPicker for LeastLoadedPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        least_loaded(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
}

/// The lowest load, with ties broken uniformly by reservoir sampling. Reads every load once.
#[inline]
pub(super) fn least_loaded(
    len: usize,
    load: impl Fn(usize) -> u64,
    samples: &mut impl Samples,
) -> Option<usize> {
    let mut best = None;
    let mut best_load = u64::MAX;
    let mut ties = 0usize;
    for index in 0..len {
        let candidate_load = load(index);
        if candidate_load < best_load {
            best = Some(index);
            best_load = candidate_load;
            ties = 1;
        } else if candidate_load == best_load {
            ties += 1;
            if samples.index(ties) == 0 {
                best = Some(index);
            }
        }
    }
    best
}
