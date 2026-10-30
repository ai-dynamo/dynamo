// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `power-of-two`: the less loaded of two random workers.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use dynamo_routing_algorithms::{self as algorithms, ThreadRandom};

use super::{active_requests, no_candidates};

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
        algorithms::power_of_two(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
}
