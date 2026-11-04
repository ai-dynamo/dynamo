// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `power-of-two`: the less loaded of two random workers.

use super::{Samples, ThreadRandom, active_requests, no_candidates};
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// The less loaded of two random workers.
pub const POWER_OF_TWO: &str = "power-of-two";

pub(super) struct PowerOfTwoPicker;

impl WorkerPicker for PowerOfTwoPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        power_of_two(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
}

/// Two distinct uniform draws; the lower load wins, and the first on ties. Reads two loads.
#[inline]
pub(super) fn power_of_two(
    len: usize,
    load: impl Fn(usize) -> u64,
    samples: &mut impl Samples,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let first = samples.index(len);
    if len == 1 {
        return Some(first);
    }
    let second = (first + 1 + samples.index(len - 1)) % len;
    Some(if load(first) <= load(second) {
        first
    } else {
        second
    })
}
