// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker input groups, stored snapshots, and component-scoped borrowed views.

use crate::protocols::{SharedCacheHits, WorkerConfigLike, WorkerWithDpRank};
use crate::sequences::WorkerLoadProjection;
use rustc_hash::FxHashMap;
use std::ops::BitOr;

/// Host-owned row materialized once for the union of component input requirements.
pub(crate) struct CandidateData {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) inputs: WorkerInputs,
    pub(crate) cache: WorkerCacheData,
    pub(crate) load: WorkerLoadInput,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// Static worker info by view row. Hosts look it up only when a component asks, so rows
/// never carry it.
pub(crate) type WorkerInfoLookup<'a> = &'a dyn Fn(usize) -> WorkerInfo;

/// A borrowed view of one worker, restricted to this component's declared inputs.
#[derive(Clone, Copy)]
pub struct WorkerCandidate<'a> {
    data: &'a CandidateData,
    inputs: WorkerInputs,
    // Row passed to `info`. A u32 packs beside `inputs`, keeping the view at five words.
    row: u32,
    cache_snapshot: &'a CacheSnapshot<'a>,
    info: WorkerInfoLookup<'a>,
}

impl<'a> WorkerCandidate<'a> {
    /// A single-worker view; `info` is called with row 0.
    pub(crate) fn new(
        data: &'a CandidateData,
        inputs: WorkerInputs,
        cache_snapshot: &'a CacheSnapshot<'a>,
        info: WorkerInfoLookup<'a>,
    ) -> Self {
        Self {
            data,
            inputs,
            row: 0,
            cache_snapshot,
            info,
        }
    }

    /// Worker identity is always available.
    pub fn worker(self) -> WorkerWithDpRank {
        self.data.worker
    }

    /// This worker's identity and advertised capacity. Always available, with no input group
    /// to declare; the host looks it up on each call.
    pub fn info(self) -> WorkerInfo {
        (self.info)(self.row as usize)
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
    // Indexed by row in `rows`.
    info: WorkerInfoLookup<'a>,
}

impl<'a> WorkerCandidates<'a> {
    pub(crate) fn new(
        rows: &'a [CandidateData],
        inputs: WorkerInputs,
        cache_snapshot: &'a CacheSnapshot<'a>,
        info: WorkerInfoLookup<'a>,
    ) -> Self {
        debug_assert!(u32::try_from(rows.len()).is_ok());
        Self {
            rows,
            inputs,
            cache_snapshot,
            info,
        }
    }

    fn candidate(self, row: usize, data: &'a CandidateData) -> WorkerCandidate<'a> {
        WorkerCandidate {
            data,
            inputs: self.inputs,
            row: row as u32,
            cache_snapshot: self.cache_snapshot,
            info: self.info,
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
        self.rows.get(row).map(|data| self.candidate(row, data))
    }

    /// Iterate over the surviving workers without copying their data.
    pub fn iter(
        self,
    ) -> impl ExactSizeIterator<Item = WorkerCandidate<'a>> + DoubleEndedIterator + Clone {
        self.rows
            .iter()
            .enumerate()
            .map(move |(row, data)| self.candidate(row, data))
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

/// Rows and request-wide cache facts owned by a reader outside worker selection.
pub(crate) struct WorkerInputRows {
    inputs: WorkerInputs,
    rows: Vec<CandidateData>,
    // Index-aligned with `rows`. The rows outlive the worker map, so they carry each rank's
    // capacity; the rank itself is already in the row.
    capacities: Vec<WorkerCapacity>,
    shared_hits: Option<SharedCacheHits>,
    has_tier_matches: bool,
}

impl WorkerInputRows {
    pub(crate) fn new(
        inputs: WorkerInputs,
        rows: Vec<CandidateData>,
        capacities: Vec<WorkerCapacity>,
        cache_snapshot: CacheSnapshot<'_>,
    ) -> Self {
        debug_assert_eq!(rows.len(), capacities.len());
        Self {
            inputs,
            rows,
            capacities,
            shared_hits: if inputs.contains(WorkerInputs::CACHE) {
                cache_snapshot.shared_hits.cloned()
            } else {
                None
            },
            has_tier_matches: cache_snapshot.has_tier_matches,
        }
    }

    /// Rows for a reader that declared no inputs.
    pub(crate) fn empty() -> Self {
        Self {
            inputs: WorkerInputs::NONE,
            rows: Vec::new(),
            capacities: Vec::new(),
            shared_hits: None,
            has_tier_matches: false,
        }
    }

    /// Lend the rows as a view restricted to the declared inputs.
    pub(crate) fn read<R>(&self, read: impl FnOnce(WorkerCandidates<'_>) -> R) -> R {
        let cache_snapshot = CacheSnapshot {
            shared_hits: self.shared_hits.as_ref(),
            has_tier_matches: self.has_tier_matches,
        };
        let info = |row: usize| WorkerInfo::new(self.rows[row].worker, self.capacities[row]);
        read(WorkerCandidates::new(
            &self.rows,
            self.inputs,
            &cache_snapshot,
            &info,
        ))
    }
}

impl std::fmt::Debug for WorkerInputRows {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerInputRows")
            .field("inputs", &self.inputs)
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
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
    pub(crate) active_decode_blocks: usize,
    pub(crate) decode_cost_blocks: f64,
    pub(crate) active_requests: usize,
}

// Every policy copies these rows per candidate per selection. `active_decode_blocks` took the
// row from 32 to 40 bytes; the selection benches put that within noise at 10k workers, while
// narrowing the counters to `u32` cost 2-5% in per-row saturating conversions. Keep the sizes
// from drifting further without a measurement.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<WorkerLoadInput>() == 40);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CandidateData>() == 120);

/// Prefill budget assumed for a rank that does not report `max_num_batched_tokens`. Large
/// enough that a fractional prefill threshold never fires on an unreported budget.
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Advertised capacity of one worker rank, from its runtime config. It changes only when
/// the worker registers again, so hosts serve it from cached discovery state, not per request.
/// Components read it through [`WorkerInfo::capacity`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerCapacity {
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
}

impl WorkerCapacity {
    /// A reported zero means unreported. vLLM registers `total_kv_blocks = 0` as a placeholder
    /// under the Ray data-parallel backend, and a zero budget would mark every rank full.
    pub fn new(max_num_batched_tokens: Option<u64>, total_kv_blocks: Option<u64>) -> Self {
        Self {
            max_num_batched_tokens: max_num_batched_tokens.filter(|&value| value > 0),
            total_kv_blocks: total_kv_blocks.filter(|&value| value > 0),
        }
    }

    /// Read the capacity a worker advertises in its runtime config.
    pub fn of<C: WorkerConfigLike>(config: &C) -> Self {
        Self::new(config.max_num_batched_tokens(), config.total_kv_blocks())
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    pub fn max_num_batched_tokens(self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    pub fn total_kv_blocks(self) -> Option<u64> {
        self.total_kv_blocks
    }

    /// The budget fractional prefill thresholds are taken of: `max_num_batched_tokens`, or
    /// [`DEFAULT_MAX_BATCHED_TOKENS`] when the rank does not report it.
    pub(crate) fn prefill_budget(self) -> u64 {
        self.max_num_batched_tokens
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS)
    }
}

/// Static facts about one worker rank: its identity and the capacity its runtime config
/// advertises. Every component can read it without declaring an input group. It changes
/// only when the worker registers again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerInfo {
    worker: WorkerWithDpRank,
    capacity: WorkerCapacity,
}

impl WorkerInfo {
    pub fn new(worker: WorkerWithDpRank, capacity: WorkerCapacity) -> Self {
        Self { worker, capacity }
    }

    /// Read a rank's info from its worker's runtime config.
    pub fn of<C: WorkerConfigLike>(worker: WorkerWithDpRank, config: &C) -> Self {
        Self::new(worker, WorkerCapacity::of(config))
    }

    /// Worker ID and data-parallel rank.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    pub fn max_num_batched_tokens(&self) -> Option<u64> {
        self.capacity.max_num_batched_tokens()
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    pub fn total_kv_blocks(&self) -> Option<u64> {
        self.capacity.total_kv_blocks()
    }

    /// Both advertised values, as [`crate::scheduling::BusyThresholds::has_room`] takes them.
    pub fn capacity(&self) -> WorkerCapacity {
        self.capacity
    }
}

/// Borrowed, index-aligned view of one custom picker's requested worker inputs.
#[derive(Clone, Copy)]
pub struct WorkerInputView<'a> {
    pub(crate) candidates: &'a [ScoredWorkerCandidate],
    pub(crate) cache: Option<WorkerCacheInputs<'a>>,
    pub(crate) load: Option<&'a [WorkerLoadInput]>,
    // Indexed by row in `candidates`.
    pub(crate) info: WorkerInfoLookup<'a>,
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
    /// Convert an observed load projection into a `LOAD` row.
    pub(crate) fn from_projection(load: WorkerLoadProjection) -> Self {
        Self {
            available: true,
            active_prefill_tokens: load.active_prefill_tokens,
            active_decode_blocks: load.active_decode_blocks,
            decode_cost_blocks: load.potential_decode_blocks() as f64,
            active_requests: load.active_requests,
        }
    }

    /// The `LOAD` row for `worker`: its projection in `loads` when the host observed one,
    /// otherwise `unobserved()` flagged unavailable. Worker selection, the policy-class
    /// queue, and offline replay all build rows here, so they read the same row for the
    /// same worker by construction.
    pub(crate) fn for_worker(
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        worker: WorkerWithDpRank,
        unobserved: impl FnOnce() -> WorkerLoadProjection,
    ) -> Self {
        match loads.get(&worker) {
            Some(&load) => Self::from_projection(load),
            None => Self {
                available: false,
                ..Self::from_projection(unobserved())
            },
        }
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

    /// Return the KV blocks this worker's active requests hold. Unlike
    /// [`decode_cost_blocks`](Self::decode_cost_blocks), this excludes the incoming request.
    /// Counts each request's prompt blocks, plus the blocks generation adds only when the
    /// router runs with `--router-track-output-blocks` (off by default). Zero when the router
    /// runs with `--no-router-track-active-blocks`, as the prefill KV router always does.
    pub fn active_decode_blocks(&self) -> usize {
        self.active_decode_blocks
    }

    /// Return the projected active decode footprint in KV blocks, including the incoming
    /// request's blocks not already active on this worker.
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

    /// Return static info for the candidate at `row`, or None for an out-of-range row.
    /// Always available, with no input group to declare; the host looks it up on each call.
    pub fn info(self, row: usize) -> Option<WorkerInfo> {
        (row < self.candidates.len()).then(|| (self.info)(row))
    }
}
