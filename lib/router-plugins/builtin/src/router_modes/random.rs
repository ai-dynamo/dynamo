// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `random`: a uniform random worker.

use super::{Samples, ThreadRandom, no_candidates};
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};

/// A uniform random worker.
pub const RANDOM: &str = "random";

pub(super) struct RandomPicker;

impl WorkerPicker for RandomPicker {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        random(input.len(), &mut ThreadRandom).ok_or_else(no_candidates)
    }
}

/// One uniform draw.
#[inline]
pub(super) fn random(len: usize, samples: &mut impl Samples) -> Option<usize> {
    (len > 0).then(|| samples.index(len))
}
