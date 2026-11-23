// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local output budgets, read on demand by a worker picker.
//!
//! Declare [`DECODE_PROGRESS`] and import [`DecodeProgressByRow`] to read a candidate.
//! Supported aggregated requests have one choice and a positive effective output limit.
//! A limit derived by the frontend from the remaining context also counts.
//! The complete accessor requires a valid observation for every active request and
//! completed prefill for every request. The partial accessor covers the active requests
//! with valid output observations and counts the remaining requests as unknown. Both
//! accessors require a supported host and a picker that declared the input.
//!
//! Budgets are raw-token upper bounds over the reported requests. Observations stay
//! local to this frontend. Use the reported observation age to apply a freshness limit.

use crate::{WorkerInputView, WorkerInputs};

pub use dynamo_slot_tracker::host::{DecodeProgress, PartialDecodeProgress};

/// Request live decode progress during the picker callback.
pub const DECODE_PROGRESS: WorkerInputs = WorkerInputs::DECODE_PROGRESS;

mod sealed {
    pub trait Sealed {}
}

/// Reads a candidate's progress without building a snapshot of the other candidates.
pub trait DecodeProgressByRow: sealed::Sealed {
    /// Read the summary for candidate `row`, or `None` for missing data or an invalid row.
    fn decode_progress(self, row: usize) -> Option<DecodeProgress>;

    /// Read the observed subset for candidate `row`, or `None` when no valid observation is available.
    fn partial_decode_progress(self, row: usize) -> Option<PartialDecodeProgress>;
}

impl sealed::Sealed for WorkerInputView<'_> {}

impl DecodeProgressByRow for WorkerInputView<'_> {
    fn decode_progress(self, row: usize) -> Option<DecodeProgress> {
        self.decode_progress_for_row(row)
    }

    fn partial_decode_progress(self, row: usize) -> Option<PartialDecodeProgress> {
        self.partial_decode_progress_for_row(row)
    }
}
