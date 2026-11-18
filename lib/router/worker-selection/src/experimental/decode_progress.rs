// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local output budgets, read on demand by a worker picker.
//!
//! Declare [`DECODE_PROGRESS`] and import [`DecodeProgressByRow`] to read a candidate.
//! Supported aggregated requests have one choice and a positive effective output limit.
//! A limit derived by the frontend from the remaining context also counts.
//! A rank returns `None` while any active request is prefilling, unobserved or invalid,
//! or when the input or host is unavailable. Filters and scorers receive none.
//!
//! Budgets are raw-token upper bounds, not completion-time estimates. Observations stay
//! local to this frontend. Use [`DecodeProgress::observed_age`] for a freshness limit.

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
