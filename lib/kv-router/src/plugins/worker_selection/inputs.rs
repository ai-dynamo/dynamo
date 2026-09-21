// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker input groups, stored snapshots, and component-scoped borrowed views.

use crate::protocols::WorkerWithDpRank;
use std::ops::BitOr;

/// Host-owned row materialized once for the union of component input requirements.
pub(crate) struct CandidateData {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) inputs: WorkerInputs,
    pub(crate) cache: WorkerCacheInput,
    pub(crate) load: WorkerLoadInput,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// A borrowed view of one worker, restricted to this component's declared inputs.
#[derive(Clone, Copy)]
pub struct WorkerCandidate<'a> {
    data: &'a CandidateData,
    inputs: WorkerInputs,
}

impl<'a> WorkerCandidate<'a> {
    pub(crate) fn new(data: &'a CandidateData, inputs: WorkerInputs) -> Self {
        Self { data, inputs }
    }

    /// Worker identity is always available.
    pub fn worker(self) -> WorkerWithDpRank {
        self.data.worker
    }

    /// Read this worker's cache snapshot only when this component declared CACHE.
    pub fn cache(self) -> Option<&'a WorkerCacheInput> {
        self.inputs
            .contains(WorkerInputs::CACHE)
            .then_some(&self.data.cache)
    }

    /// Read this worker's load snapshot only when this component declared LOAD.
    pub fn load(self) -> Option<&'a WorkerLoadInput> {
        self.inputs
            .contains(WorkerInputs::LOAD)
            .then_some(&self.data.load)
    }

    /// Preferred-taint cost multiplier, only when this component declared PREFERRED_TAINT.
    /// None means no preference was supplied, access was not requested, or the host pinned the worker.
    pub fn preferred_taint_multiplier(self) -> Option<f64> {
        if self.inputs.contains(WorkerInputs::PREFERRED_TAINT) {
            self.data.preferred_taint_multiplier
        } else {
            None
        }
    }
}

/// Borrowed candidate batch restricted to one scorer's declared inputs.
/// All scorers see the same surviving workers in the same order. Creating or iterating this
/// view does not copy rows, allocate, or change another component's access.
#[derive(Clone, Copy)]
pub struct WorkerCandidates<'a> {
    rows: &'a [CandidateData],
    inputs: WorkerInputs,
}

impl<'a> WorkerCandidates<'a> {
    pub(crate) fn new(rows: &'a [CandidateData], inputs: WorkerInputs) -> Self {
        Self { rows, inputs }
    }

    /// Number of surviving workers, equal to the scorer's cost-buffer length.
    pub fn len(self) -> usize {
        self.rows.len()
    }

    /// Whether the batch contains no workers.
    pub fn is_empty(self) -> bool {
        self.rows.is_empty()
    }

    /// Borrow one row with this scorer's input permissions, or None for an out-of-range index.
    pub fn get(self, row: usize) -> Option<WorkerCandidate<'a>> {
        self.rows
            .get(row)
            .map(|data| WorkerCandidate::new(data, self.inputs))
    }

    /// Iterate over the surviving workers without copying their data.
    pub fn iter(
        self,
    ) -> impl ExactSizeIterator<Item = WorkerCandidate<'a>> + DoubleEndedIterator + Clone {
        self.rows
            .iter()
            .map(move |data| WorkerCandidate::new(data, self.inputs))
    }
}

/// One eligible worker and its total cost after all scorers run.
#[derive(Clone, Copy)]
pub struct ScoredWorkerCandidate {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) cost: f64,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// Optional worker-signal groups requested by scorers and pickers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerInputs(u8);

impl WorkerInputs {
    /// Request no optional worker inputs.
    pub const NONE: Self = Self(0);
    /// Request worker KV-cache overlap inputs and request-level cache context.
    pub const CACHE: Self = Self(1 << 0);
    /// Request active-load inputs.
    pub const LOAD: Self = Self(1 << 1);
    /// Request preferred-taint routing metadata.
    pub const PREFERRED_TAINT: Self = Self(1 << 2);
    /// Request host-owned active-request counts.
    pub const OCCUPANCY: Self = Self(1 << 5);
    #[cfg(any(test, feature = "bench"))]
    pub(crate) const ALL: Self = Self(Self::CACHE.0 | Self::LOAD.0 | Self::PREFERRED_TAINT.0);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for WorkerInputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// KV-cache overlap values for one worker.
#[derive(Clone, Copy, Default)]
pub struct WorkerCacheInput {
    pub(crate) effective_overlap_blocks: f64,
    pub(crate) estimated_cached_tokens: usize,
    pub(crate) device_overlap_blocks: f64,
    pub(crate) host_overlap_blocks: f64,
    pub(crate) disk_overlap_blocks: f64,
}

/// Active-load values for one worker.
#[derive(Clone, Copy, Default)]
pub struct WorkerLoadInput {
    pub(crate) available: bool,
    pub(crate) active_prefill_tokens: usize,
    pub(crate) decode_cost_blocks: f64,
    pub(crate) active_requests: usize,
}

/// Borrowed, index-aligned view of one custom picker's requested worker inputs.
#[derive(Clone, Copy)]
pub struct WorkerInputView<'a> {
    pub(crate) candidates: &'a [ScoredWorkerCandidate],
    pub(crate) cache: Option<&'a [WorkerCacheInput]>,
    pub(crate) load: Option<&'a [WorkerLoadInput]>,
}

impl CandidateData {
    pub(crate) fn with_inputs_from(&self, additional: &Self, inputs: WorkerInputs) -> Self {
        debug_assert_eq!(self.worker, additional.worker);
        Self {
            worker: self.worker,
            inputs,
            cache: if inputs.contains(WorkerInputs::CACHE) {
                if self.inputs.contains(WorkerInputs::CACHE) {
                    self.cache
                } else {
                    additional.cache
                }
            } else {
                WorkerCacheInput::default()
            },
            load: if inputs.contains(WorkerInputs::LOAD) {
                if self.inputs.contains(WorkerInputs::LOAD) {
                    self.load
                } else {
                    additional.load
                }
            } else {
                WorkerLoadInput::default()
            },
            preferred_taint_multiplier: self
                .preferred_taint_multiplier
                .or(additional.preferred_taint_multiplier),
        }
    }
}

impl ScoredWorkerCandidate {
    /// Return this candidate's worker ID and data-parallel rank.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Return the sum of all scorer contributions for this candidate.
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Return the optional cost multiplier from preferred routing constraints when the picker
    /// requested [`WorkerInputs::PREFERRED_TAINT`].
    pub fn preferred_taint_multiplier(&self) -> Option<f64> {
        self.preferred_taint_multiplier
    }
}

impl WorkerCacheInput {
    /// Host accounting estimate for this worker, in weighted KV blocks and rounded tokens.
    /// Lower-tier matches use the host's cache weights. Missing estimates are zero;
    /// neither value is clamped to prompt length. This is the current lookup snapshot,
    /// not a count of physically resident GPU tokens. Policy scores do not alter it.
    pub fn accounting_cache_estimate(&self) -> (f64, usize) {
        (self.effective_overlap_blocks, self.estimated_cached_tokens)
    }

    /// Return device-resident prefix overlap in KV blocks.
    pub fn device_overlap_blocks(&self) -> f64 {
        self.device_overlap_blocks
    }

    /// Return host-pinned prefix overlap in KV blocks.
    pub fn host_overlap_blocks(&self) -> f64 {
        self.host_overlap_blocks
    }

    /// Return disk prefix overlap in KV blocks.
    pub fn disk_overlap_blocks(&self) -> f64 {
        self.disk_overlap_blocks
    }
}

impl WorkerLoadInput {
    /// Whether the host supplied a load projection for this worker in this selection.
    /// False distinguishes a missing observation from an observed idle worker.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Return the tokens active in this worker's prefill stage.
    pub fn active_prefill_tokens(&self) -> usize {
        self.active_prefill_tokens
    }

    /// Return the projected active decode footprint in KV blocks.
    pub fn decode_cost_blocks(&self) -> f64 {
        self.decode_cost_blocks
    }

    /// Return this worker's active request count.
    pub fn active_requests(&self) -> usize {
        self.active_requests
    }
}

impl<'a> WorkerInputView<'a> {
    /// Return the eligible candidates and their total costs.
    pub fn candidates(self) -> &'a [ScoredWorkerCandidate] {
        self.candidates
    }

    /// Return index-aligned KV-cache inputs when the picker requested them.
    pub fn cache(self) -> Option<&'a [WorkerCacheInput]> {
        self.cache
    }

    /// Return index-aligned active-load inputs when the picker requested them.
    pub fn load(self) -> Option<&'a [WorkerLoadInput]> {
        self.load
    }
}
