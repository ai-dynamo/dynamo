// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker input groups, stored snapshots, and component-scoped borrowed views.

use crate::protocols::{SharedCacheHits, WorkerWithDpRank};
use std::ops::BitOr;

/// Host-owned row materialized once for the union of component input requirements.
pub(crate) struct CandidateData {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) inputs: WorkerInputs,
    pub(crate) cache: WorkerCacheData,
    pub(crate) load: WorkerLoadInput,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// A borrowed view of one worker, restricted to this component's declared inputs.
#[derive(Clone, Copy)]
pub struct WorkerCandidate<'a> {
    data: &'a CandidateData,
    inputs: WorkerInputs,
    cache_snapshot: &'a CacheSnapshot<'a>,
}

impl<'a> WorkerCandidate<'a> {
    pub(crate) fn new(
        data: &'a CandidateData,
        inputs: WorkerInputs,
        cache_snapshot: &'a CacheSnapshot<'a>,
    ) -> Self {
        Self {
            data,
            inputs,
            cache_snapshot,
        }
    }

    /// Worker identity is always available.
    pub fn worker(self) -> WorkerWithDpRank {
        self.data.worker
    }

    /// Read this worker's cache snapshot only when this component declared CACHE.
    pub fn cache(self) -> Option<WorkerCacheInput<'a>> {
        self.inputs
            .contains(WorkerInputs::CACHE)
            .then_some(WorkerCacheInput {
                data: &self.data.cache,
                snapshot: self.cache_snapshot,
            })
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
    cache_snapshot: &'a CacheSnapshot<'a>,
}

impl<'a> WorkerCandidates<'a> {
    pub(crate) fn new(
        rows: &'a [CandidateData],
        inputs: WorkerInputs,
        cache_snapshot: &'a CacheSnapshot<'a>,
    ) -> Self {
        Self {
            rows,
            inputs,
            cache_snapshot,
        }
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
            .map(|data| WorkerCandidate::new(data, self.inputs, self.cache_snapshot))
    }

    /// Iterate over the surviving workers without copying their data.
    pub fn iter(
        self,
    ) -> impl ExactSizeIterator<Item = WorkerCandidate<'a>> + DoubleEndedIterator + Clone {
        self.rows
            .iter()
            .map(move |data| WorkerCandidate::new(data, self.inputs, self.cache_snapshot))
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
    /// Request worker KV-cache overlaps and shared-cache lookup results.
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

/// Request-wide facts borrowed once for all cache views in a selection.
pub(crate) struct CacheSnapshot<'a> {
    pub(crate) shared_hits: Option<&'a SharedCacheHits>,
    pub(crate) has_tier_matches: bool,
}

/// Numeric cache row retained in host buffers between selections.
#[derive(Clone, Copy, Default)]
pub(crate) struct WorkerCacheData {
    pub(crate) effective_overlap_blocks: f64,
    pub(crate) estimated_cached_tokens: usize,
    pub(crate) device_overlap_blocks: f64,
    pub(crate) host_overlap_blocks: f64,
    pub(crate) disk_overlap_blocks: f64,
}

/// Borrowed worker overlaps and shared-cache facts from one lookup snapshot.
/// Available only to components that declare [`WorkerInputs::CACHE`].
#[derive(Clone, Copy)]
pub struct WorkerCacheInput<'a> {
    data: &'a WorkerCacheData,
    snapshot: &'a CacheSnapshot<'a>,
}

/// Borrowed cache rows in the same order as a picker's candidates.
#[derive(Clone, Copy)]
pub struct WorkerCacheInputs<'a> {
    pub(crate) rows: &'a [WorkerCacheData],
    pub(crate) snapshot: &'a CacheSnapshot<'a>,
}

impl<'a> WorkerCacheInputs<'a> {
    /// Number of cache rows, equal to the picker candidate count.
    pub fn len(self) -> usize {
        self.rows.len()
    }

    /// Whether there are no cache rows.
    pub fn is_empty(self) -> bool {
        self.rows.is_empty()
    }

    /// Borrow one cache row, or None for an out-of-range index.
    pub fn get(self, row: usize) -> Option<WorkerCacheInput<'a>> {
        self.rows.get(row).map(|data| WorkerCacheInput {
            data,
            snapshot: self.snapshot,
        })
    }

    /// Iterate over cache rows without copying their data.
    pub fn iter(
        self,
    ) -> impl ExactSizeIterator<Item = WorkerCacheInput<'a>> + DoubleEndedIterator + Clone {
        self.rows.iter().map(move |data| WorkerCacheInput {
            data,
            snapshot: self.snapshot,
        })
    }
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
    pub(crate) cache: Option<WorkerCacheInputs<'a>>,
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
                WorkerCacheData::default()
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

impl<'a> WorkerCacheInput<'a> {
    /// Unweighted shared-cache ranges in KV block positions, or None if no result was supplied.
    /// The host owns this snapshot. It does not change during the callback, and may lag engine state.
    /// Use `hits_beyond(prefix)` to exclude hits already covered by the policy's chosen prefix.
    pub fn shared_hits(self) -> Option<&'a SharedCacheHits> {
        self.snapshot.shared_hits
    }

    /// Whether the request snapshot contains any tier-specific matches before worker filtering.
    /// False means only accounting estimates (or no cache data) were supplied. This describes
    /// observed matches, not worker cache capacity or whether a particular worker has a match.
    pub fn has_tier_matches(self) -> bool {
        self.snapshot.has_tier_matches
    }

    /// Host accounting estimate for this worker, in weighted KV blocks and rounded tokens.
    /// Lower-tier matches use the host's cache weights. Missing estimates are zero;
    /// neither value is clamped to prompt length. This is the current lookup snapshot,
    /// not a count of physically resident GPU tokens. Policy scores do not alter it.
    pub fn accounting_cache_estimate(&self) -> (f64, usize) {
        (
            self.data.effective_overlap_blocks,
            self.data.estimated_cached_tokens,
        )
    }

    /// Return device-resident prefix overlap in KV blocks.
    pub fn device_overlap_blocks(&self) -> f64 {
        self.data.device_overlap_blocks
    }

    /// Return host-pinned prefix overlap in KV blocks.
    pub fn host_overlap_blocks(&self) -> f64 {
        self.data.host_overlap_blocks
    }

    /// Return disk prefix overlap in KV blocks.
    pub fn disk_overlap_blocks(&self) -> f64 {
        self.data.disk_overlap_blocks
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
    pub fn cache(self) -> Option<WorkerCacheInputs<'a>> {
        self.cache
    }

    /// Return index-aligned active-load inputs when the picker requested them.
    pub fn load(self) -> Option<&'a [WorkerLoadInput]> {
        self.load
    }
}
