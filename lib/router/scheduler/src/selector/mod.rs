// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kernel side of worker selection: the [`WorkerSelector`] contract routing hosts
//! call, the request-side row builder that implements the policy's [`CandidateRowSource`],
//! and the adapter that drives a [`WorkerSelectionPolicy`] from a [`SchedulingRequest`].

use std::{cell::OnceCell, collections::HashMap};

#[cfg(any(test, feature = "testing", feature = "bench"))]
use std::cell::Cell;

mod derive;
#[cfg(test)]
mod policy_tests;
#[cfg(any(test, feature = "testing", feature = "bench"))]
mod reference;

pub use derive::{WorkerCapacityExt, WorkerInfoExt, WorkerLoadInputExt};

#[cfg(any(test, feature = "testing", feature = "bench"))]
pub use reference::DefaultWorkerSelector;

// TODO(v1.7): Remove these compatibility re-exports; use dynamo_worker_selection instead.
#[cfg(any(test, feature = "testing", feature = "bench"))]
use dynamo_worker_selection::DefaultWorkerPicker;
pub use dynamo_worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerFilter, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer,
    WorkerSelectionContext,
};
#[cfg(any(test, feature = "testing", feature = "bench"))]
use reference::{DefaultWorkerScorer, pick_default_worker};

use crate::kv_hints::{
    KvTransferScope, KvTransferSourceChoice, KvTransferSourceSummary, resolve_transfer_sources,
};
use dynamo_worker_selection::{
    CacheSnapshot, CandidateData, ResidentBlockSource, WorkerCacheData, WorkerCapacity,
    WorkerCopyInput, WorkerInputRows,
};
pub use dynamo_worker_selection::{
    CandidatePreparer, CandidateRank, CandidateRowSource, PreparedRows, RankVisitor, RowPick,
    SelectionFailure, WorkerSelectionPolicy,
};
use dynamo_worker_selection::{RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate};

use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, SchedulingResponse};
use crate::protocols::{SelectionLogFields, WorkerConfigLike, WorkerSelectionResult};
use dynamo_kv_indexer::ResidentBlockCounts;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_worker_selection::{SelectionTrace, SelectionTraceSpec};

/// Low-level selector used by routing hosts.
///
/// External policies should use [`WorkerSelectionPolicy`].
pub trait WorkerSelector<C: WorkerConfigLike> {
    /// Snapshot runtime parameters, when this selector explicitly supports updates.
    fn runtime_config(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        Err(RuntimePolicyError::Unsupported)
    }

    /// Apply a validated replacement between selections. The scheduler actor serializes calls.
    fn update_runtime_config(
        &mut self,
        _update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        Err(RuntimePolicyError::Unsupported)
    }

    /// Optional worker data required by this selector.
    fn required_worker_inputs(&self) -> WorkerInputs;

    /// Whether an eligible affinity target exclusively constrains worker selection.
    ///
    /// The default selector uses exclusive affinity. Custom policies receive affinity as
    /// advisory context and may choose another eligible worker.
    fn uses_exclusive_affinity_target(&self) -> bool {
        false
    }

    /// Whether the selector can reject workers that [`RoutingEligibility`] accepts.
    ///
    /// The host measures affinity selections against the best eligible worker only when
    /// this is `false`, so cache metrics never count a worker the policy would reject.
    fn applies_worker_filters(&self) -> bool {
        true
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;

    /// Select a worker and, when `trace` is set, also return the per-candidate costs
    /// the policy scored. The default ignores `trace`; policies that can report their
    /// candidates override it. Hosts call this instead of [`Self::select_worker`]
    /// when a selection trace is installed.
    fn select_worker_traced(
        &self,
        input: WorkerSelectionInput<'_, C>,
        _trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        self.select_worker(input).map(|result| (result, None))
    }

    /// Builds candidate rows before the request reaches the scheduler actor, for
    /// selectors whose rows depend only on the request and worker configuration
    /// until load inputs are filled.
    fn candidate_preparer(&self) -> Option<CandidatePreparer> {
        None
    }

    /// Like [`Self::select_worker_traced`], from rows that [`Self::candidate_preparer`]
    /// built against the same worker configuration when given. The default ignores them.
    fn select_worker_prepared(
        &self,
        input: WorkerSelectionInput<'_, C>,
        _prepared: Option<PreparedCandidates>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        self.select_worker_traced(input, trace)
    }
}

/// Inputs supplied by the selector's host.
#[derive(Clone, Copy)]
pub enum WorkerSelectionInput<'a, C: WorkerConfigLike> {
    Configured {
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
        /// The partition's live resident block counts, lent by the host for a policy that
        /// declared [`WorkerInputs::RESIDENT_BLOCKS`]; `None` otherwise.
        resident_blocks: Option<&'a ResidentBlockCounts>,
    },
}

pub type ConfiguredSelectionInputs<'a, C> = (
    &'a HashMap<WorkerId, C>,
    &'a SchedulingRequest,
    RoutingEligibility<'a>,
    u32,
);

impl<'a, C: WorkerConfigLike> WorkerSelectionInput<'a, C> {
    pub fn configured(
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    ) -> Self {
        Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
            resident_blocks: None,
        }
    }

    /// Lend the partition's live resident block counts to this selection. The host passes them
    /// only for a policy that declared [`WorkerInputs::RESIDENT_BLOCKS`]; every component's
    /// snapshot borrows them for this selection alone.
    pub fn with_resident_blocks(self, resident_blocks: Option<&'a ResidentBlockCounts>) -> Self {
        let Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
            ..
        } = self;
        Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
            resident_blocks,
        }
    }

    /// The resident block counts lent to this selection, if any.
    pub fn resident_blocks(&self) -> Option<&'a ResidentBlockCounts> {
        let Self::Configured {
            resident_blocks, ..
        } = self;
        *resident_blocks
    }

    pub fn into_configured(self) -> ConfiguredSelectionInputs<'a, C> {
        let Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
            ..
        } = self;
        (workers, request, eligibility, block_size)
    }
}

/// Rows a [`CandidatePreparer`] built on the caller task, with the fetch sources they
/// were built from, so the selected worker's fetch source comes from the summary that
/// scored every row.
pub struct PreparedCandidates {
    rows: PreparedRows,
    kv_transfer_sources: Option<KvTransferSourceSummary>,
}

/// Kernel-side preparation: build a policy's rows from the request and worker configuration.
pub(crate) trait CandidatePreparerExt {
    /// Returns `None` when no rank is eligible, so that selection reports why.
    fn prepare<C: WorkerConfigLike>(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Option<PreparedCandidates>;
}

impl CandidatePreparerExt for CandidatePreparer {
    fn prepare<C: WorkerConfigLike>(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Option<PreparedCandidates> {
        let (workers, request, eligibility, block_size) = input.into_configured();
        let source = SelectionRowSource::new(request, workers, eligibility, block_size);
        let rows = self.prepare_with(workers.len(), |inputs, rows| {
            eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
                let mut row = source.row_for(worker, config, inputs);
                // Prepared rows carry the worker's rank count, so the scheduler actor can tell
                // a fully full worker at selection without a config lookup.
                row.dp_size = u16::try_from(config.data_parallel_size()).unwrap_or(u16::MAX);
                rows.push(row);
            });
        })?;
        Some(PreparedCandidates {
            rows,
            kv_transfer_sources: source.into_kv_transfer_sources(),
        })
    }
}

/// The kernel's view of one selection: the request, the ranks its eligibility admits, and
/// their configuration. Yields the policy's rows as its [`CandidateRowSource`] and holds
/// what the kernel computes alongside them: the fetch-source summary, built at most once
/// and shared by every row's fetch credit and the selected worker's fetch source, so the
/// two cannot disagree.
pub(super) struct SelectionRowSource<'a, C: WorkerConfigLike> {
    pub(super) request: &'a SchedulingRequest,
    pub(super) workers: &'a HashMap<WorkerId, C>,
    pub(super) eligibility: RoutingEligibility<'a>,
    pub(super) block_size: u32,
    /// The partition's live resident block counts, when the host lent them for this selection.
    resident_blocks: Option<&'a ResidentBlockCounts>,
    /// Only tracked requests report cache reuse, so only their rows carry raw cached tokens.
    tracked: bool,
    pub(super) kv_transfer_sources: OnceCell<KvTransferSourceSummary>,
    /// The reference selector counts the rows it scores here; a policy counts its own.
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    max_raw_cached_tokens: Cell<Option<usize>>,
}

impl<'a, C: WorkerConfigLike> SelectionRowSource<'a, C> {
    pub(super) fn new(
        request: &'a SchedulingRequest,
        workers: &'a HashMap<WorkerId, C>,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    ) -> Self {
        Self {
            request,
            workers,
            eligibility,
            block_size,
            resident_blocks: None,
            tracked: request.mode.is_tracked(),
            kv_transfer_sources: OnceCell::new(),
            #[cfg(any(test, feature = "testing", feature = "bench"))]
            max_raw_cached_tokens: Cell::new(request.mode.is_tracked().then_some(0)),
        }
    }

    /// The source for a host's selection input, once its pinned worker is validated.
    fn from_input(input: WorkerSelectionInput<'a, C>) -> Result<Self, KvSchedulerError> {
        let resident_blocks = input.resident_blocks();
        let (workers, request, eligibility, block_size) = input.into_configured();
        eligibility.validate_pinned_worker_allowed()?;
        if let Some(worker) = eligibility.pinned_worker() {
            match eligibility.validate_worker_rank(workers, worker) {
                Ok(_) => {}
                Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                    return Err(KvSchedulerError::PinnedWorkerOverloaded {
                        worker_id: worker.worker_id,
                    });
                }
                Err(_) => return Err(KvSchedulerError::NoEndpoints),
            }
        }
        let mut source = Self::new(request, workers, eligibility, block_size);
        source.resident_blocks = resident_blocks;
        Ok(source)
    }

    /// The policy's view of the request, pinned to what eligibility imposes. It borrows this
    /// source for the resident block counts the host lent, so it lives no longer than the source.
    pub(super) fn context(&self) -> WorkerSelectionContext<'_> {
        selection_context(self.request, self.block_size)
            .with_pinned_worker(self.eligibility.pinned_worker())
            .with_resident_blocks(
                self.resident_blocks
                    .is_some()
                    .then_some(self as &dyn ResidentBlockSource),
            )
    }

    /// The request's fetch sources for the targets in `scope`, built on the
    /// first call.
    pub(super) fn kv_transfer_sources(
        &self,
        scope: impl FnOnce() -> KvTransferScope,
    ) -> Option<&KvTransferSourceSummary> {
        let candidates = self.request.kv_transfer_candidates.as_ref()?;
        Some(self.kv_transfer_sources.get_or_init(|| {
            let scope = scope();
            KvTransferSourceSummary::new(
                resolve_transfer_sources(candidates, self.workers, &scope),
                scope,
            )
        }))
    }

    /// The targets of every eligible rank. A row's fetch source is filled as the row is
    /// built, so the summary covers every rank a policy might keep; a wider scope
    /// admits only sources no kept rank can use, which `source_for` skips.
    fn eligible_transfer_scope(&self) -> KvTransferScope {
        let mut scope = KvTransferScope::empty();
        self.eligibility
            .for_each_eligible_worker_rank(self.workers, |worker, config| {
                if let Some(target) = config.kv_transfer_keys().target() {
                    scope.add_target(
                        target,
                        self.request.transfer_local_prefix_blocks_for(worker),
                    );
                }
            });
        scope
    }

    /// A row's fetch input; `None` when the request has no transfer candidates.
    fn kv_transfer_input(&self, worker: WorkerWithDpRank, config: &C) -> Option<WorkerCopyInput> {
        let sources = self.kv_transfer_sources(|| self.eligible_transfer_scope())?;
        let source = config.kv_transfer_keys().target().and_then(|target| {
            sources.source_for(
                worker,
                target,
                self.request.transfer_local_prefix_blocks_for(worker),
            )
        });
        Some(WorkerCopyInput {
            source_prefix_blocks: source.map(source_prefix_blocks),
        })
    }

    fn selected_kv_transfer_source(
        &self,
        worker: WorkerWithDpRank,
    ) -> Option<KvTransferSourceChoice> {
        self.request.kv_transfer_candidates.as_ref()?;
        let target = self
            .workers
            .get(&worker.worker_id)?
            .kv_transfer_keys()
            .target()?;
        let local_prefix_blocks = self.request.transfer_local_prefix_blocks_for(worker);
        let scope = || KvTransferScope::of([(target, local_prefix_blocks)]);
        self.kv_transfer_sources(scope)?
            .source_for(worker, target, local_prefix_blocks)
    }

    /// Adopt the summary preparation built over the same rows, so the selected
    /// worker's fetch source comes from the summary that scored every row.
    fn restore_kv_transfer_sources(&self, kv_transfer_sources: Option<KvTransferSourceSummary>) {
        if let Some(sources) = kv_transfer_sources {
            let _ = self.kv_transfer_sources.set(sources);
        }
    }

    fn into_kv_transfer_sources(self) -> Option<KvTransferSourceSummary> {
        self.kv_transfer_sources.into_inner()
    }

    /// Why no rank was picked: every eligible rank is overloaded, or none is eligible.
    fn no_candidate_error(&self) -> KvSchedulerError {
        if self.eligibility.has_eligible_worker_ignoring_overload(
            self.workers
                .iter()
                .map(|(&worker_id, config)| (worker_id, config)),
        ) {
            KvSchedulerError::AllEligibleWorkersOverloaded
        } else {
            KvSchedulerError::NoEndpoints
        }
    }

    /// Count a row the reference selector scored toward the best eligible cached prefix.
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    #[inline(always)]
    pub(super) fn track_kept(&self, row: &CandidateData) {
        if let Some(current_max) = self.max_raw_cached_tokens.get() {
            self.max_raw_cached_tokens
                .set(Some(current_max.max(row.raw_cached_tokens as usize)));
        }
    }

    #[cfg(any(test, feature = "testing", feature = "bench"))]
    pub(super) fn max_raw_cached_tokens(&self) -> Option<usize> {
        self.max_raw_cached_tokens.get()
    }

    /// One rank's row with the config-derived columns `inputs` asks for.
    #[inline(always)]
    fn row_for(&self, worker: WorkerWithDpRank, config: &C, inputs: WorkerInputs) -> CandidateData {
        let preferred_taint_multiplier = if inputs.contains(WorkerInputs::PREFERRED_TAINT)
            && self.eligibility.pinned_worker().is_none()
        {
            self.request
                .routing_constraints
                .preferred_taint_multiplier(config.taints())
        } else {
            None
        };
        let mut row = self.row_with_device_overlap(
            worker,
            preferred_taint_multiplier,
            inputs,
            |_, device_overlap_blocks| device_overlap_blocks,
        );
        if inputs.contains(WorkerInputs::KV_COPY) {
            row.transfer = self.kv_transfer_input(worker, config);
        }
        row
    }

    /// One rank's row from the request alone, with the preferred-taint column supplied.
    #[inline(always)]
    pub(super) fn row_with_device_overlap(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
        select_device_overlap: impl FnOnce(f64, f64) -> f64,
    ) -> CandidateData {
        let mut row = worker_input_row(
            self.request,
            worker,
            preferred_taint_multiplier,
            inputs,
            select_device_overlap,
        );
        if self.tracked {
            row.raw_cached_tokens = raw_cached_tokens_column(
                self.request.raw_cached_tokens_for(worker, self.block_size),
            );
        }
        row
    }
}

/// The row column is `u32`, which holds every prompt-bounded count.
fn raw_cached_tokens_column(raw_cached_tokens: usize) -> u32 {
    u32::try_from(raw_cached_tokens).unwrap_or(u32::MAX)
}

/// One eligible rank during the kernel's walk: the rank and its configuration in hand,
/// so a row at any columns, or the capacity, costs no lookup.
pub(super) struct SelectionRank<'r, 'a, C: WorkerConfigLike> {
    source: &'r SelectionRowSource<'a, C>,
    worker: WorkerWithDpRank,
    config: &'r C,
}

impl<C: WorkerConfigLike> CandidateRank for SelectionRank<'_, '_, C> {
    #[inline(always)]
    fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    #[inline(always)]
    fn capacity(&self) -> WorkerCapacity {
        WorkerCapacity::of(self.config)
    }

    #[inline(always)]
    fn row(&self, inputs: WorkerInputs) -> CandidateData {
        self.source.row_for(self.worker, self.config, inputs)
    }
}

impl<C: WorkerConfigLike> CandidateRowSource for SelectionRowSource<'_, C> {
    #[inline(always)]
    fn visit_ranks<V: RankVisitor>(&self, visitor: &mut V) {
        self.eligibility
            .any_eligible_worker_rank(self.workers, |worker, config| {
                visitor.visit(&SelectionRank {
                    source: self,
                    worker,
                    config,
                })
            });
    }

    fn capacity(&self, worker: WorkerWithDpRank) -> WorkerCapacity {
        self.workers
            .get(&worker.worker_id)
            .map_or_else(WorkerCapacity::default, WorkerCapacity::of)
    }

    fn load(&self, worker: WorkerWithDpRank) -> WorkerLoadInput {
        WorkerLoadInput::for_worker(&self.request.worker_loads, worker, || {
            self.request.unobserved_worker_load()
        })
    }
}

impl<C: WorkerConfigLike> ResidentBlockSource for SelectionRowSource<'_, C> {
    #[inline(always)]
    fn resident_blocks(&self, worker: WorkerWithDpRank) -> Option<u64> {
        self.resident_blocks.and_then(|counts| counts.get(worker))
    }
}

/// The fields of `request` a policy may read, as [`WorkerSelectionContext`] exposes them.
fn selection_context(request: &SchedulingRequest, block_size: u32) -> WorkerSelectionContext<'_> {
    WorkerSelectionContext::new(request.isl_tokens, block_size)
        .with_request_id(request.mode.request_id())
        .with_prefill_tracking(request.track_prefill_tokens)
        .with_pinned_worker(request.pinned_worker)
        .with_router_temperature_override(
            request
                .router_config_override
                .as_ref()
                .and_then(|config| config.router_temperature),
        )
        .with_session_context(request.session_context.as_ref())
        .with_affinity_target(request.affinity_target)
        .with_expected_output_tokens(request.expected_output_tokens)
        .with_priority(request.priority_jump, request.strict_priority)
        .with_shared_cache_hits(request.shared_cache_hits.as_ref())
        .with_tier_matches(request.has_tier_matches())
}

/// Request-wide cache facts shared by every row built from `request`.
fn cache_snapshot(request: &SchedulingRequest) -> CacheSnapshot<'_> {
    CacheSnapshot {
        shared_hits: request.shared_cache_hits.as_ref(),
        has_tier_matches: request.has_tier_matches(),
        // Classifier rows never carry counts; only worker selection reads them.
        resident_blocks: None,
    }
}

/// Build one worker's input row. Worker selection and request classification share this
/// builder, so each input group carries the same values in both.
#[inline(always)]
fn worker_input_row(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    preferred_taint_multiplier: Option<f64>,
    inputs: WorkerInputs,
    select_device_overlap: impl FnOnce(f64, f64) -> f64,
) -> CandidateData {
    let cached_tokens = if inputs.contains(WorkerInputs::CACHE) {
        request.effective_cached_tokens_for(worker)
    } else {
        0
    };
    let cache = if inputs.contains(WorkerInputs::CACHE) {
        let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
        let reported_device_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .device
            .get(&worker)
            .copied()
            .map(|blocks| blocks as f64)
            .unwrap_or(0.0);
        let device_overlap_blocks =
            select_device_overlap(effective_overlap_blocks, reported_device_overlap_blocks);
        WorkerCacheData {
            effective_overlap_blocks,
            estimated_cached_tokens: cached_tokens,
            device_overlap_blocks,
            host_overlap_blocks: request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .get(&worker)
                .copied()
                .unwrap_or(0) as f64,
            disk_overlap_blocks: request
                .overlap
                .tier_overlap_blocks
                .disk
                .get(&worker)
                .copied()
                .unwrap_or(0) as f64,
        }
    } else {
        WorkerCacheData::default()
    };
    let load = if inputs.contains(WorkerInputs::LOAD) {
        WorkerLoadInput::for_worker(&request.worker_loads, worker, || {
            request.unobserved_worker_load()
        })
    } else {
        WorkerLoadInput::default()
    };

    CandidateData {
        worker,
        inputs,
        raw_cached_tokens: 0,
        cache,
        load,
        transfer: None,
        preferred_taint_multiplier,
        dp_size: 0,
    }
}

/// Build owned input rows for every worker rank that `eligibility` admits.
pub(crate) fn worker_input_rows<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    inputs: WorkerInputs,
) -> WorkerInputRows {
    let materialize_preferred_taint =
        eligibility.pinned_worker().is_none() && inputs.contains(WorkerInputs::PREFERRED_TAINT);
    let mut rows = Vec::new();
    let mut capacities = Vec::new();
    eligibility.any_eligible_worker_rank(workers, |worker, config| {
        capacities.push(WorkerCapacity::of(config));
        let preferred_taint_multiplier = if materialize_preferred_taint {
            request
                .routing_constraints
                .preferred_taint_multiplier(config.taints())
        } else {
            None
        };
        rows.push(worker_input_row(
            request,
            worker,
            preferred_taint_multiplier,
            inputs,
            |_, device_overlap_blocks| device_overlap_blocks,
        ));
        false
    });
    WorkerInputRows::new(inputs, rows, capacities, cache_snapshot(request))
}

fn source_prefix_blocks(source: KvTransferSourceChoice) -> u32 {
    u32::try_from(source.prefix_blocks).unwrap_or(u32::MAX)
}

fn selection_result(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
    max_raw_cached_tokens: Option<usize>,
    kv_transfer_source: Option<KvTransferSourceChoice>,
) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: request.request_blocks(block_size),
        effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
        cached_tokens: request.effective_cached_tokens_for(worker),
        max_raw_cached_tokens,
        selected_raw_cached_tokens: max_raw_cached_tokens
            .map(|_| request.raw_cached_tokens_for(worker, block_size)),
        potential_decode_blocks: request
            .potential_decode_blocks_after_admission(worker, block_size),
        kv_transfer_source,
        log_fields: None,
    }
}

/// Log the scheduler's selection on the routing task once the scheduler replies, so
/// the scheduler actor does not format the line. Selections without log fields came
/// from a selector that does not log them.
pub(crate) fn log_selection(request_id: Option<&str>, response: &SchedulingResponse) {
    let Some(SelectionLogFields {
        worker_type,
        logit,
        pinned,
        session_affinity,
        host_pinned_blocks,
        disk_blocks,
        total_kv_blocks,
    }) = response.log_fields
    else {
        return;
    };
    let request_id = request_id.unwrap_or("-");
    let worker = response.best_worker;
    let effective_overlap_blocks = response.effective_overlap_blocks;

    let selection = if session_affinity {
        "session_affinity"
    } else if pinned {
        "pinned"
    } else {
        "kv"
    };

    if worker_type == "decode" {
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit,
            host_pinned_blocks,
            disk_blocks,
            selection,
            "Selected worker"
        );
    } else {
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit,
            effective_cached_blocks = effective_overlap_blocks,
            host_pinned_blocks,
            disk_blocks,
            total_kv_blocks = ?total_kv_blocks,
            selection,
            "Selected worker"
        );
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for WorkerSelectionPolicy {
    fn runtime_config(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.runtime_config()
    }

    fn update_runtime_config(
        &mut self,
        update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.update_runtime_config(update)
    }

    fn uses_exclusive_affinity_target(&self) -> bool {
        WorkerSelectionPolicy::uses_exclusive_affinity_target(self)
    }

    fn applies_worker_filters(&self) -> bool {
        WorkerSelectionPolicy::applies_worker_filters(self)
    }

    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerSelectionPolicy::required_worker_inputs(self)
    }

    #[inline(always)]
    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        self.select_worker_traced(input, None)
            .map(|(result, _)| result)
    }

    fn select_worker_traced(
        &self,
        input: WorkerSelectionInput<'_, C>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        select_with_policy(self, input, None, trace)
    }

    fn candidate_preparer(&self) -> Option<CandidatePreparer> {
        WorkerSelectionPolicy::candidate_preparer(self)
    }

    fn select_worker_prepared(
        &self,
        input: WorkerSelectionInput<'_, C>,
        prepared: Option<PreparedCandidates>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
        select_with_policy(self, input, prepared, trace)
    }
}

/// What a selection picked: the rank, its cost, the trace when one was requested, and
/// the raw cache-reuse maximum for tracked requests.
type Picked = Option<(WorkerWithDpRank, f64, Option<SelectionTrace>, Option<usize>)>;

/// Drive `policy` from the host's inputs: build the context and the row source, select,
/// and map the pick to the host's result. Eligibility outcomes and result construction stay
/// here, shared with the reference selector.
fn select_with_policy<C: WorkerConfigLike>(
    policy: &WorkerSelectionPolicy,
    input: WorkerSelectionInput<'_, C>,
    prepared: Option<PreparedCandidates>,
    trace: Option<SelectionTraceSpec>,
) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    if let Some((config, picker)) = policy.reference_parts() {
        return select_reference(policy.worker_label(), config, picker, input);
    }
    let source = SelectionRowSource::from_input(input)?;
    let context = source.context();
    let prepared = prepared.and_then(|prepared| {
        let PreparedCandidates {
            rows,
            kv_transfer_sources,
        } = prepared;
        policy.accepts_prepared(&rows).then(|| {
            source.restore_kv_transfer_sources(kv_transfer_sources);
            rows
        })
    });
    // The rows were built before the actor's projection: a worker whose every rank is now
    // past the reject thresholds leaves, the rest keep their order, and the policy recounts
    // the best eligible cached prefix from the rows it adopts.
    let prepared = prepared.map(|mut rows| {
        if source.eligibility.skips_full_workers() {
            rows.retain(|row| {
                !source
                    .eligibility
                    .is_worker_full(row.worker.worker_id, u32::from(row.dp_size))
            });
        }
        rows
    });
    let picked: Picked = match policy.select_rows(&context, &source, prepared, trace) {
        Ok(RowPick {
            worker,
            cost,
            max_raw_cached_tokens,
            trace,
        }) => Some((
            worker,
            cost,
            trace,
            source
                .request
                .mode
                .is_tracked()
                .then_some(max_raw_cached_tokens as usize),
        )),
        Err(SelectionFailure::NoCandidates) => None,
        Err(SelectionFailure::AllFiltered) => {
            return Err(KvSchedulerError::AllEligibleWorkersFiltered);
        }
        Err(SelectionFailure::Policy(error)) => return Err(error.into()),
    };
    finish_selection(&source, policy.worker_label(), picked)
}

/// Select with the reference scorer and picker, which read the request directly.
#[cfg(any(test, feature = "testing", feature = "bench"))]
fn select_reference<C: WorkerConfigLike>(
    worker_type: &'static str,
    kv_router_config: &dynamo_worker_selection::SelectionConfig,
    picker: &DefaultWorkerPicker,
    input: WorkerSelectionInput<'_, C>,
) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
    let source = SelectionRowSource::from_input(input)?;
    let context = source.context();
    let scorer = DefaultWorkerScorer {
        kv_router_config,
        worker_type,
    };
    let picked: Picked = pick_default_worker(&scorer, picker, &context, &source)
        .map(|(worker, cost)| (worker, cost, None, source.max_raw_cached_tokens()));
    finish_selection(&source, worker_type, picked)
}

fn finish_selection<C: WorkerConfigLike>(
    source: &SelectionRowSource<'_, C>,
    worker_type: &'static str,
    picked: Picked,
) -> Result<(WorkerSelectionResult, Option<SelectionTrace>), KvSchedulerError> {
    let Some((worker, cost, selection_trace, max_raw_cached_tokens)) = picked else {
        return Err(source.no_candidate_error());
    };
    let kv_transfer_source = source.selected_kv_transfer_source(worker);
    let tiers = &source.request.overlap.tier_overlap_blocks;
    let log_fields = SelectionLogFields {
        worker_type,
        logit: cost,
        pinned: source.request.pinned_worker == Some(worker),
        session_affinity: source
            .request
            .affinity_target
            .is_some_and(|target| target.matches(worker)),
        host_pinned_blocks: tiers.host_pinned.get(&worker).copied().unwrap_or(0),
        disk_blocks: tiers.disk.get(&worker).copied().unwrap_or(0),
        total_kv_blocks: source
            .workers
            .get(&worker.worker_id)
            .and_then(WorkerConfigLike::total_kv_blocks),
    };
    let result = WorkerSelectionResult {
        log_fields: Some(log_fields),
        ..selection_result(
            source.request,
            worker,
            source.block_size,
            max_raw_cached_tokens,
            kv_transfer_source,
        )
    };
    Ok((result, selection_trace))
}

#[cfg(test)]
mod test_support {
    use std::collections::HashSet;

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::{OverlapSignals, ScheduleMode};

    #[derive(Clone, Default)]
    pub(super) struct TaintedWorkerConfig {
        pub(super) taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    pub(super) fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            accounting_attribution: Default::default(),
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: Default::default(),
                effective_cached_tokens: Default::default(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    pub(super) fn worker_loads_with_active_decode(
        decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    ) -> FxHashMap<WorkerWithDpRank, dynamo_slot_tracker::WorkerLoadProjection> {
        decode_blocks
            .into_iter()
            .map(|(worker, active_decode_blocks)| {
                (
                    worker,
                    dynamo_slot_tracker::WorkerLoadProjection {
                        active_decode_blocks,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod cache_reuse_tests {
    use super::test_support::base_request;
    use super::*;
    use crate::test_utils::SimpleWorkerConfig;

    #[test]
    fn tracked_requests_report_raw_cache_reuse() {
        let mut request = base_request(128);
        request.mode = crate::ScheduleMode::Tracked {
            request_id: "test".into(),
        };
        let first = WorkerWithDpRank::from_worker_id(1);
        let second = WorkerWithDpRank::from_worker_id(2);
        request.overlap.effective_cached_tokens.insert(first, 120);
        request.overlap.effective_cached_tokens.insert(second, 16);
        request.overlap.tier_overlap_blocks.device.insert(first, 2);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(first, 1);
        request.overlap.tier_overlap_blocks.device.insert(second, 4);
        request.overlap.tier_overlap_blocks.disk.insert(second, 2);
        let workers = HashMap::from([
            (1, SimpleWorkerConfig::default()),
            (2, SimpleWorkerConfig::default()),
        ]);
        let source = SelectionRowSource::new(&request, &workers, request.eligibility(), 16);

        let mut raw_cached_tokens = HashMap::new();
        source
            .eligibility
            .for_each_eligible_worker_rank(source.workers, |worker, config| {
                let row = source.row_for(worker, config, WorkerInputs::NONE);
                source.track_kept(&row);
                raw_cached_tokens.insert(worker, row.raw_cached_tokens);
            });

        assert_eq!(raw_cached_tokens[&first], 48);
        assert_eq!(raw_cached_tokens[&second], 96);
        assert_eq!(source.max_raw_cached_tokens(), Some(96));
        let result = selection_result(&request, first, 16, source.max_raw_cached_tokens(), None);
        assert_eq!(result.max_raw_cached_tokens, Some(96));
        assert_eq!(result.selected_raw_cached_tokens, Some(48));
        assert_eq!(result.cached_tokens, 120);
    }

    #[test]
    fn query_only_requests_skip_cache_reuse_tracking() {
        let mut request = base_request(128);
        request.mode = crate::ScheduleMode::QueryOnly { request_id: None };
        let worker = WorkerWithDpRank::from_worker_id(1);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        request.overlap.tier_overlap_blocks.device.insert(worker, 6);
        let workers = HashMap::from([(1, SimpleWorkerConfig::default())]);
        let source = SelectionRowSource::new(&request, &workers, request.eligibility(), 16);

        let mut raw_cached_tokens = None;
        source
            .eligibility
            .for_each_eligible_worker_rank(source.workers, |worker, config| {
                let row = source.row_for(worker, config, WorkerInputs::NONE);
                source.track_kept(&row);
                raw_cached_tokens = Some(row.raw_cached_tokens);
            });

        // Untracked requests never read the tier blocks, so the column stays zero.
        assert_eq!(raw_cached_tokens, Some(0));
        assert_eq!(source.max_raw_cached_tokens(), None);
        let result = selection_result(&request, worker, 16, source.max_raw_cached_tokens(), None);
        assert_eq!(result.max_raw_cached_tokens, None);
        assert_eq!(result.selected_raw_cached_tokens, None);
    }

    #[test]
    fn raw_overlap_is_specific_to_the_selected_dp_rank() {
        let mut request = base_request(128);
        let selected = WorkerWithDpRank::new(1, 0);
        let other_rank = WorkerWithDpRank::new(1, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(selected, 2);
        request.overlap.tier_overlap_blocks.disk.insert(selected, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(other_rank, 7);
        request.overlap.effective_cached_tokens.insert(selected, 3);
        assert_eq!(request.raw_cached_tokens_for(selected, 16), 64);
        assert_eq!(request.raw_cached_tokens_for(other_rank, 16), 112);
        let result = selection_result(&request, selected, 16, Some(112), None);
        assert_eq!(result.selected_raw_cached_tokens, Some(64));
    }

    #[test]
    fn cache_reuse_tracking_excludes_out_of_range_dp_rank() {
        let workers = HashMap::from([(1, SimpleWorkerConfig::default())]);
        let mut request = base_request(128);
        request.mode = crate::ScheduleMode::Tracked {
            request_id: "test".into(),
        };
        let invalid_rank = WorkerWithDpRank::new(1, 1);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(invalid_rank, 6);
        let eligibility = request.eligibility();
        let source = SelectionRowSource::new(&request, &workers, eligibility, 16);

        source
            .eligibility
            .for_each_eligible_worker_rank(source.workers, |worker, config| {
                source.track_kept(&source.row_for(worker, config, WorkerInputs::NONE));
            });

        assert!(
            eligibility
                .validate_worker_rank(&workers, invalid_rank)
                .is_err()
        );
        assert_eq!(source.max_raw_cached_tokens(), Some(0));
    }
}
