// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `least-loaded`: the worker with the fewest active requests.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use dynamo_routing_algorithms::{self as algorithms, ThreadRandom};

use super::{active_requests, no_candidates};

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
        algorithms::least_loaded(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
}
