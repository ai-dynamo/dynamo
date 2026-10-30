// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker input groups, stored snapshots, and component-scoped borrowed views.

use std::ops::BitOr;

use dynamo_router_protocol::worker::WorkerWithDpRank;

use super::shared_cache::SharedCacheHits;

/// Host-owned row materialized once for the union of component input requirements.
/// One eligible rank's input columns: the row a [`CandidateRank`](super::CandidateRank)
/// yields and a policy's views read. The kernel fills it; a column outside the row's
/// `inputs` holds its default and no view exposes it.
pub struct CandidateData {
    pub worker: WorkerWithDpRank,
    /// The columns this row holds.
    pub inputs: WorkerInputs,
    /// Device, host, and disk overlap in tokens, for the best eligible cache reuse a
    /// tracked request reports; zero for untracked requests. A `u32` packs beside
    /// `inputs`, keeping the row at 128 bytes.
    pub raw_cached_tokens: u32,
    pub cache: WorkerCacheData,
    pub load: WorkerLoadInput,
    /// `Some` once the fetch column is filled, which needs the request's transfer candidates.
    pub transfer: Option<WorkerTransferInput>,
    /// The request's preference for this rank's taints, when asked for and not pinned.
    pub preferred_taint_multiplier: Option<f64>,
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

    /// The KV prefix this worker could fetch, only for a scorer that declared
    /// KV_TRANSFER; filters and pickers do not receive it. None also means the
    /// host has no transfer data for this request: no worker in the partition
    /// consumes router hints, or the indexer cannot retain the matched block chain.
    pub fn kv_transfer(self) -> Option<WorkerTransferInput> {
        if self.inputs.contains(WorkerInputs::KV_TRANSFER) {
            self.data.transfer
        } else {
            None
        }
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
    /// Request the KV prefix each worker could fetch from another worker or cache.
    pub const KV_TRANSFER: Self = Self(1 << 6);
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
pub struct CacheSnapshot<'a> {
    pub shared_hits: Option<&'a SharedCacheHits>,
    pub has_tier_matches: bool,
}

/// Rows and request-wide cache facts owned by a reader outside worker selection.
pub struct WorkerInputRows {
    inputs: WorkerInputs,
    rows: Vec<CandidateData>,
    // Index-aligned with `rows`. The rows outlive the worker map, so they carry each rank's
    // capacity; the rank itself is already in the row.
    capacities: Vec<WorkerCapacity>,
    shared_hits: Option<SharedCacheHits>,
    has_tier_matches: bool,
}

impl WorkerInputRows {
    pub fn new(
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
    pub fn empty() -> Self {
        Self {
            inputs: WorkerInputs::NONE,
            rows: Vec::new(),
            capacities: Vec::new(),
            shared_hits: None,
            has_tier_matches: false,
        }
    }

    /// Lend the rows as a view restricted to the declared inputs.
    pub fn read<R>(&self, read: impl FnOnce(WorkerCandidates<'_>) -> R) -> R {
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
pub struct WorkerCacheData {
    pub effective_overlap_blocks: f64,
    pub estimated_cached_tokens: usize,
    pub device_overlap_blocks: f64,
    pub host_overlap_blocks: f64,
    pub disk_overlap_blocks: f64,
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
    /// Whether the host observed a projection for this rank; false flags a default row.
    pub available: bool,
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    pub decode_cost_blocks: f64,
    pub active_requests: usize,
}

// Every policy copies these rows per candidate per selection. `active_decode_blocks` took the
// row from 32 to 40 bytes; the selection benches put that within noise at 10k workers, while
// narrowing the counters to `u32` cost 2-5% in per-row saturating conversions. Keep the sizes
// from drifting further without a measurement.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<WorkerLoadInput>() == 40);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CandidateData>() == 128);

pub use dynamo_router_protocol::load::{DEFAULT_MAX_BATCHED_TOKENS, WorkerCapacity};

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

    /// Both advertised values, as a fullness check takes them.
    pub fn capacity(&self) -> WorkerCapacity {
        self.capacity
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerTransferInput {
    /// Blocks of the prompt a fetch source holds beyond this rank's local prefix, when one
    /// was chosen for the rank.
    pub source_prefix_blocks: Option<u32>,
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
            transfer: if inputs.contains(WorkerInputs::KV_TRANSFER) {
                if self.inputs.contains(WorkerInputs::KV_TRANSFER) {
                    self.transfer
                } else {
                    additional.transfer
                }
            } else {
                None
            },
            raw_cached_tokens: self.raw_cached_tokens,
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

impl dynamo_router_protocol::load::RankLoad for WorkerLoadInput {
    fn active_prefill_tokens(&self) -> usize {
        self.active_prefill_tokens
    }

    fn active_decode_blocks(&self) -> usize {
        self.active_decode_blocks
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

impl WorkerTransferInput {
    /// Root-aligned prefix, in blocks, held by the source this worker would
    /// fetch from, and so the prefix it holds after the fetch. None when no
    /// eligible source holds more than this worker's device and host-pinned
    /// prefix, the `CACHE` inputs' `device_overlap_blocks() + host_overlap_blocks()`.
    ///
    /// The host chooses the source by its router-hint rules: the same hint
    /// worker type, neither this worker nor a cache owner attached to it, and,
    /// when this worker sets a KV transfer domain, its topology island
    /// (`preferred` enforcement falls back to another island). Booking this
    /// worker attaches a fetch hint for that source. Fetches are best effort:
    /// the source may no longer hold the blocks when the request runs.
    pub fn source_prefix_blocks(&self) -> Option<u32> {
        self.source_prefix_blocks
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
