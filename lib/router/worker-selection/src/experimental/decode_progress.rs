// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local decode observations, read on demand by a worker picker.
//!
//! Declaring [`DECODE_PROGRESS`] enables collection for supported aggregated requests in
//! the routing partition. Each read locks only the requested rank's native slot tracker;
//! prepared rows retain no progress. Filters, scorers and request classifiers receive none.
//!
//! A summary covers every active request on the rank. It is `None` for an undeclared input,
//! an unavailable host, an idle rank, incomplete or invalid observations, or a rank with
//! requests still prefilling. Supported streams have one choice, a positive explicit output
//! limit, `ignore_eos=true`, and no other stop conditions. Counts are unweighted raw tokens
//! observed by this frontend, and are not shared between frontend replicas. Collection is
//! independent of output-block accounting.
//!
//! Remaining output budget is a bound, not a completion-time estimate. Use
//! [`DecodeProgress::observed_age`] to reject old data; it includes time waiting for the
//! lookup lock. Consecutive reads may observe different updates.

use crate::{WorkerInputView, WorkerInputs};

pub use dynamo_slot_tracker::host::DecodeProgress;

/// Request live decode progress during the picker callback.
pub const DECODE_PROGRESS: WorkerInputs = WorkerInputs::DECODE_PROGRESS;

mod sealed {
    pub trait Sealed {}
}

/// Reads a candidate's progress without building a snapshot of the other candidates.
pub trait DecodeProgressByRow: sealed::Sealed {
    /// Read the summary for candidate `row`, or `None` for missing data or an invalid row.
    fn decode_progress(self, row: usize) -> Option<DecodeProgress>;
}

impl sealed::Sealed for WorkerInputView<'_> {}

impl DecodeProgressByRow for WorkerInputView<'_> {
    fn decode_progress(self, row: usize) -> Option<DecodeProgress> {
        self.decode_progress_for_row(row)
    }
}
