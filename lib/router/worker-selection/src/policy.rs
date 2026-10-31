// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::sync::Arc;

use crossbeam_queue::ArrayQueue;

use dynamo_router_protocol::worker::WorkerWithDpRank;

#[cfg(any(test, feature = "testing", feature = "bench"))]
use crate::DefaultWorkerPicker;
use crate::{
    SelectionConfig, SelectionTrace, SelectionTraceRow, SelectionTraceSpec,
    WorkerSelectionPolicyError,
};

use crate::{
    CacheSnapshot, CandidateData, ScoredWorkerCandidate, WorkerCacheData, WorkerCacheInputs,
    WorkerCandidate, WorkerCandidates, WorkerCapacity, WorkerFilter, WorkerInfo, WorkerInputView,
    WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionContext,
};
use crate::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyState,
    RuntimePolicyUpdate,
};

/// What the kernel supplies for one selection: a walk over the eligible ranks, each
/// yielding its input columns on request.
///
/// A policy pulls rows through this rather than receiving a built table, so a policy with
/// filters asks a rank for its scorer columns only once its filters kept it, and the kernel
/// never needs to know which ranks a policy rejected. Everything is statically dispatched:
/// the walk, the rank, and the row it builds inline into the policy's scratch.
pub trait CandidateRowSource {
    /// Visit every eligible rank in turn, stopping when the visitor says so.
    fn visit_ranks<V: RankVisitor>(&self, visitor: &mut V);

    /// A rank's configured capacity, when a scorer or picker asks for its `WorkerInfo`
    /// after the walk.
    fn capacity(&self, worker: WorkerWithDpRank) -> WorkerCapacity;

    /// A rank's load after the actor's bookings, for rows prepared before them.
    fn load(&self, worker: WorkerWithDpRank) -> WorkerLoadInput;
}

/// Receives each rank of a [`CandidateRowSource`] walk. The method is generic over the
/// rank so that the walk, the rank, and the rows it builds are monomorphized together.
pub trait RankVisitor {
    /// Visit one rank; return `true` to stop the walk.
    fn visit<R: CandidateRank>(&mut self, rank: &R) -> bool;
}

/// One eligible rank during a [`CandidateRowSource`] walk.
pub trait CandidateRank {
    fn worker(&self) -> WorkerWithDpRank;

    /// The rank's configured capacity.
    fn capacity(&self) -> WorkerCapacity;

    /// The rank's row holding exactly `inputs`. The preferred-taint and fetch columns are
    /// derived by the kernel from the request; the fetch column is `Some` only when the
    /// request carries transfer candidates.
    fn row(&self, inputs: WorkerInputs) -> CandidateData;
}

/// The row a policy picked, and what it learned about the rows that reached its picker.
#[derive(Debug)]
pub struct RowPick {
    pub worker: WorkerWithDpRank,
    /// The picked row's total cost.
    pub cost: f64,
    /// Highest `raw_cached_tokens` over the rows every filter kept.
    pub max_raw_cached_tokens: u32,
    /// The retained per-candidate costs, when a trace was requested.
    pub trace: Option<SelectionTrace>,
}

/// Why a policy picked no row.
#[derive(Debug)]
pub enum SelectionFailure {
    /// The source yielded no row.
    NoCandidates,
    /// Every row the source yielded was rejected by a policy filter.
    AllFiltered,
    /// A filter, scorer, or picker failed.
    Policy(WorkerSelectionPolicyError),
}

impl From<WorkerSelectionPolicyError> for SelectionFailure {
    fn from(error: WorkerSelectionPolicyError) -> Self {
        Self::Policy(error)
    }
}

enum WorkerSelectionPolicyState {
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    Reference(Box<SelectionConfig>, DefaultWorkerPicker),
    /// Policy-local state owned and called serially by one scheduler queue actor.
    Composed(RefCell<ComposedPolicyState>),
}

struct ComposedPolicyState {
    filters: Vec<(WorkerInputs, Box<dyn WorkerFilter>)>,
    scorers: Vec<(WorkerInputs, Box<dyn WorkerScorer>)>,
    picker: Box<dyn WorkerPicker>,
    filter_inputs: WorkerInputs,
    scorer_picker_inputs: WorkerInputs,
    picker_inputs: WorkerInputs,
    unscored_candidates: Vec<CandidateData>,
    score_contributions: Vec<f64>,
    candidates: Vec<ScoredWorkerCandidate>,
    cache_inputs: Vec<WorkerCacheData>,
    load_inputs: Vec<WorkerLoadInput>,
    /// Highest `raw_cached_tokens` over the rows pushed this selection.
    max_raw_cached_tokens: u32,
    /// Row buffers selection is done with, for callers to prepare into next.
    row_buffers: Arc<ArrayQueue<Vec<CandidateData>>>,
}

/// Enough for the buffers in flight between callers and the actor to come back
/// without allocating, while bounding what an idle policy keeps.
const ROW_BUFFERS: usize = 64;

/// Native scorer/picker composition behind the kernel's `WorkerSelector`.
///
/// Routing hosts supply a policy factory. Both builtin and external policies compose
/// scorers and a picker through [`Self::new`], with optional filters through [`Self::new_with_filters`].
/// The kernel drives a policy through [`Self::select_rows`] over the rows its
/// [`CandidateRowSource`] yields.
pub struct WorkerSelectionPolicy {
    worker_label: &'static str,
    state: WorkerSelectionPolicyState,
    exclusive_affinity: bool,
    runtime: Option<Box<RuntimePolicyState>>,
}

impl WorkerSelectionPolicy {
    /// Build a policy with no filters.
    ///
    /// `worker_label` identifies the worker pool in routing logs. A typed policy factory normally
    /// passes [`dynamo_router_protocol::WorkerType::as_str`]. The config argument is retained for API compatibility;
    /// policy parameters belong to the supplied scorers and picker and are not retained here.
    pub fn new(
        kv_router_config: SelectionConfig,
        worker_label: &'static str,
        scorers: Vec<Box<dyn WorkerScorer>>,
        picker: Box<dyn WorkerPicker>,
    ) -> Self {
        Self::new_with_filters(kv_router_config, worker_label, Vec::new(), scorers, picker)
    }

    /// Build a policy from ordered filters, additive scorers, and one picker.
    ///
    /// `worker_label` identifies the worker pool in routing logs. A typed policy factory normally
    /// passes [`dynamo_router_protocol::WorkerType::as_str`]. The config argument is retained for API compatibility;
    /// policy parameters belong to the supplied scorers and picker and are not retained here.
    pub fn new_with_filters(
        _kv_router_config: SelectionConfig,
        worker_label: &'static str,
        filters: Vec<Box<dyn WorkerFilter>>,
        scorers: Vec<Box<dyn WorkerScorer>>,
        picker: Box<dyn WorkerPicker>,
    ) -> Self {
        // Fetch inputs are filled once every row is built, so only scorers receive them.
        let picker_inputs = picker
            .required_worker_inputs()
            .without(WorkerInputs::KV_TRANSFER);
        // Freeze each declaration once; callbacks must not inherit another component's access.
        let filters: Vec<_> = filters
            .into_iter()
            .map(|filter| {
                (
                    filter
                        .required_worker_inputs()
                        .without(WorkerInputs::KV_TRANSFER),
                    filter,
                )
            })
            .collect();
        let scorers: Vec<_> = scorers
            .into_iter()
            .map(|scorer| (scorer.required_worker_inputs(), scorer))
            .collect();
        let filter_inputs = filters
            .iter()
            .fold(WorkerInputs::NONE, |inputs, (required, _)| {
                inputs | *required
            });
        let scorer_picker_inputs = scorers
            .iter()
            .fold(picker_inputs, |inputs, (required, _)| inputs | *required);
        Self {
            worker_label,
            exclusive_affinity: false,
            runtime: None,
            state: WorkerSelectionPolicyState::Composed(RefCell::new(ComposedPolicyState {
                filters,
                scorers,
                picker,
                filter_inputs,
                scorer_picker_inputs,
                picker_inputs,
                unscored_candidates: Vec::new(),
                score_contributions: Vec::new(),
                candidates: Vec::new(),
                cache_inputs: Vec::new(),
                load_inputs: Vec::new(),
                max_raw_cached_tokens: 0,
                row_buffers: Arc::new(ArrayQueue::new(ROW_BUFFERS)),
            })),
        }
    }

    /// Ask the host to constrain selection to an eligible affinity target.
    /// Explicit request pins remain mandatory regardless of this option.
    pub fn with_exclusive_affinity(mut self, exclusive: bool) -> Self {
        self.exclusive_affinity = exclusive;
        self
    }

    /// Opt in to runtime parameter updates with a pure replacement builder.
    /// The initial parameters define the patchable keys and reset values. Replacements must
    /// require no new worker-input groups and preserve affinity behavior; queue and tracking state are retained.
    pub fn with_runtime_config(
        mut self,
        parameters: serde_json::Map<String, serde_json::Value>,
        builder: RuntimePolicyBuilder,
    ) -> Self {
        self.runtime = Some(Box::new(RuntimePolicyState::new(
            parameters,
            builder,
            self.required_worker_inputs(),
        )));
        self
    }

    /// Return the current complete parameters for this policy instance.
    pub fn runtime_config(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.runtime
            .as_deref()
            .map(RuntimePolicyState::config)
            .ok_or(RuntimePolicyError::Unsupported)
    }

    /// Validate and replace the policy. Hosts must serialize this with selection.
    pub fn update_runtime_config(
        &mut self,
        update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or(RuntimePolicyError::Unsupported)?;
        let (mut replacement, state) = runtime.prepare(update)?;
        // Hosts freeze these requirements when constructing their routing pipeline.
        if !runtime
            .available_inputs
            .contains(replacement.required_worker_inputs())
            || self.exclusive_affinity != replacement.exclusive_affinity
            || self.worker_label != replacement.worker_label
        {
            return Err(RuntimePolicyError::InvalidParameters(
                "replacement changes routing requirements".into(),
            ));
        }
        let response = state.config();
        replacement.runtime = Some(Box::new(state));
        *self = replacement;
        Ok(response)
    }

    /// The worker pool this policy names in routing logs.
    pub fn worker_label(&self) -> &'static str {
        self.worker_label
    }

    /// Optional worker data the policy's filters, scorers, and picker declared.
    pub fn required_worker_inputs(&self) -> WorkerInputs {
        match &self.state {
            #[cfg(any(test, feature = "testing", feature = "bench"))]
            WorkerSelectionPolicyState::Reference(..) => WorkerInputs::CACHE | WorkerInputs::LOAD,
            WorkerSelectionPolicyState::Composed(state) => {
                let state = state.borrow();
                state.filter_inputs | state.scorer_picker_inputs
            }
        }
    }

    /// Whether an eligible affinity target exclusively constrains worker selection.
    pub fn uses_exclusive_affinity_target(&self) -> bool {
        #[cfg(any(test, feature = "testing", feature = "bench"))]
        if matches!(&self.state, WorkerSelectionPolicyState::Reference(..)) {
            return true;
        }
        self.exclusive_affinity
    }

    /// A preparer for this policy's rows, when its rows depend only on the request and
    /// worker configuration until load inputs are filled. Filters are policy callbacks, so a
    /// policy with filters builds its rows at selection.
    pub fn candidate_preparer(&self) -> Option<CandidatePreparer> {
        match &self.state {
            #[cfg(any(test, feature = "testing", feature = "bench"))]
            WorkerSelectionPolicyState::Reference(..) => None,
            WorkerSelectionPolicyState::Composed(state) => {
                let state = state.borrow();
                state.filters.is_empty().then(|| CandidatePreparer {
                    inputs: state.scorer_picker_inputs,
                    row_buffers: Arc::clone(&state.row_buffers),
                })
            }
        }
    }

    /// Whether `prepared` rows are usable by this policy: built by one that would build the
    /// same rows, with the same inputs and without filters, which run as policy callbacks.
    pub fn accepts_prepared(&self, prepared: &PreparedRows) -> bool {
        match &self.state {
            #[cfg(any(test, feature = "testing", feature = "bench"))]
            WorkerSelectionPolicyState::Reference(..) => false,
            WorkerSelectionPolicyState::Composed(state) => state.borrow().accepts(prepared),
        }
    }

    /// Filter, score, and pick over the rows `source` yields.
    ///
    /// `prepared` rows, which must satisfy [`Self::accepts_prepared`], stand in for the
    /// source's rows; their load column is filled from `source` here, on the actor, so it
    /// reflects bookings since preparation. When `trace` is set, the pick carries the
    /// retained per-candidate costs.
    pub fn select_rows<S: CandidateRowSource>(
        &self,
        context: &WorkerSelectionContext<'_>,
        source: &S,
        prepared: Option<PreparedRows>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<RowPick, SelectionFailure> {
        // With the reference policy compiled out the match has one arm; clippy then asks for a
        // `let`, which would not compile with it in.
        #[cfg_attr(
            not(any(test, feature = "testing", feature = "bench")),
            allow(clippy::infallible_destructuring_match)
        )]
        let state = match &self.state {
            #[cfg(any(test, feature = "testing", feature = "bench"))]
            WorkerSelectionPolicyState::Reference(..) => {
                return Err(WorkerSelectionPolicyError::failed(
                    "the reference policy selects through the kernel",
                )
                .into());
            }
            WorkerSelectionPolicyState::Composed(state) => state,
        };
        let mut state = state.borrow_mut();
        let cache_snapshot = context.cache_snapshot();
        let has_eligible_worker = match prepared {
            Some(prepared) => {
                state.adopt_prepared(context, &cache_snapshot, source, prepared)?;
                true
            }
            None => state.collect_rows(context, &cache_snapshot, source)?,
        };
        state.pick(
            context,
            &cache_snapshot,
            source,
            self.worker_label,
            has_eligible_worker,
            trace,
        )
    }

    /// Construct the native reference implementation for parity tests and benchmarks.
    ///
    /// `worker_label` selects the built-in scoring and logging contract. Typed hosts use
    /// [`dynamo_router_protocol::WorkerType::default_selector_label`] to preserve Dynamo's historical behavior.
    /// The kernel selects for it directly; [`Self::select_rows`] reports a policy failure.
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    pub fn reference(kv_router_config: SelectionConfig, worker_label: &'static str) -> Self {
        let picker = DefaultWorkerPicker::new();
        Self {
            worker_label,
            exclusive_affinity: false,
            runtime: None,
            state: WorkerSelectionPolicyState::Reference(Box::new(kv_router_config), picker),
        }
    }

    /// The reference implementation's config and picker, for the kernel to select with.
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    pub fn reference_parts(&self) -> Option<(&SelectionConfig, &DefaultWorkerPicker)> {
        match &self.state {
            WorkerSelectionPolicyState::Reference(config, picker) => Some((config, picker)),
            WorkerSelectionPolicyState::Composed(_) => None,
        }
    }
}

#[inline(always)]
fn push_picker_candidate(
    candidate: &CandidateData,
    cost: f64,
    picker_inputs: WorkerInputs,
    candidates: &mut Vec<ScoredWorkerCandidate>,
    cache_inputs: &mut Vec<WorkerCacheData>,
    load_inputs: &mut Vec<WorkerLoadInput>,
) {
    candidates.push(ScoredWorkerCandidate {
        worker: candidate.worker,
        cost,
        preferred_taint_multiplier: if picker_inputs.contains(WorkerInputs::PREFERRED_TAINT) {
            candidate.preferred_taint_multiplier
        } else {
            None
        },
    });
    if picker_inputs.contains(WorkerInputs::CACHE) {
        cache_inputs.push(candidate.cache);
    }
    if picker_inputs.contains(WorkerInputs::LOAD) {
        load_inputs.push(candidate.load);
    }
}

impl ComposedPolicyState {
    fn accepts(&self, prepared: &PreparedRows) -> bool {
        self.filters.is_empty() && self.scorer_picker_inputs == prepared.inputs
    }

    fn clear(&mut self) {
        self.unscored_candidates.clear();
        self.candidates.clear();
        self.cache_inputs.clear();
        self.load_inputs.clear();
        self.max_raw_cached_tokens = 0;
    }

    // Keep row construction and storage together to avoid passing a full row through a call.
    #[inline(always)]
    fn push_candidate(&mut self, candidate: CandidateData) {
        // Build the picker's rows alongside the input snapshot. The scoring loop then only
        // writes costs, without growing vectors or copying optional columns across trait calls.
        push_picker_candidate(
            &candidate,
            0.0,
            self.picker_inputs,
            &mut self.candidates,
            &mut self.cache_inputs,
            &mut self.load_inputs,
        );
        // Count a rank toward the best eligible cached prefix once every filter has kept it.
        self.max_raw_cached_tokens = self.max_raw_cached_tokens.max(candidate.raw_cached_tokens);
        if !self.scorers.is_empty() {
            // Picker-only policies do not need a second copy of the worker inputs.
            self.unscored_candidates.push(candidate);
        }
    }

    /// Pull every eligible row, filter it, and score what the filters kept. Returns whether
    /// the source yielded any row, so an empty table is attributed to the filters or the source.
    fn collect_rows<S: CandidateRowSource>(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        cache_snapshot: &CacheSnapshot<'_>,
        source: &S,
    ) -> Result<bool, SelectionFailure> {
        self.clear();
        if self.filters.is_empty() {
            let inputs = self.scorer_picker_inputs;
            source.visit_ranks(&mut PushRows {
                state: self,
                inputs,
            });
            self.score_candidates(context, cache_snapshot, source)?;
            return Ok(!self.candidates.is_empty());
        }

        let filter_inputs = self.filter_inputs;
        let scorer_picker_inputs = self.scorer_picker_inputs;
        let mut visitor = FilterRows {
            state: self,
            context,
            cache_snapshot,
            filter_inputs,
            scorer_picker_inputs,
            additional_inputs: scorer_picker_inputs.without(filter_inputs),
            has_eligible_worker: false,
            error: None,
        };
        source.visit_ranks(&mut visitor);
        let FilterRows {
            has_eligible_worker,
            error,
            ..
        } = visitor;
        if let Some(error) = error {
            return Err(error.into());
        }
        self.score_candidates(context, cache_snapshot, source)?;
        Ok(has_eligible_worker)
    }

    /// Adopt prepared rows as the scorer rows, filling the load column that depends on
    /// bookings, and score them.
    fn adopt_prepared<S: CandidateRowSource>(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        cache_snapshot: &CacheSnapshot<'_>,
        source: &S,
        mut prepared: PreparedRows,
    ) -> Result<(), SelectionFailure> {
        debug_assert!(self.accepts(&prepared));
        self.clear();
        // Adopt the prepared rows rather than copying them; the outgoing buffer goes back
        // to the pool when `prepared` drops.
        std::mem::swap(&mut self.unscored_candidates, &mut prepared.rows);
        let fill_load = self.scorer_picker_inputs.contains(WorkerInputs::LOAD);
        let Self {
            scorers,
            picker_inputs,
            unscored_candidates,
            candidates,
            cache_inputs,
            load_inputs,
            max_raw_cached_tokens,
            ..
        } = self;
        for row in unscored_candidates.iter_mut() {
            row.inputs = prepared.inputs;
            if fill_load {
                row.load = source.load(row.worker);
            }
            *max_raw_cached_tokens = (*max_raw_cached_tokens).max(row.raw_cached_tokens);
            push_picker_candidate(
                row,
                0.0,
                *picker_inputs,
                candidates,
                cache_inputs,
                load_inputs,
            );
        }
        if scorers.is_empty() {
            // Picker-only policies keep no scorer rows.
            unscored_candidates.clear();
        }
        self.score_candidates(context, cache_snapshot, source)
    }

    fn score_candidates<S: CandidateRowSource>(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        cache_snapshot: &CacheSnapshot<'_>,
        source: &S,
    ) -> Result<(), SelectionFailure> {
        let Self {
            scorers,
            unscored_candidates,
            score_contributions,
            candidates,
            ..
        } = self;
        if unscored_candidates.is_empty() {
            return Ok(());
        }
        let unscored_candidates: &[CandidateData] = unscored_candidates;
        let info = |row: usize| {
            let worker = unscored_candidates[row].worker;
            WorkerInfo::new(worker, source.capacity(worker))
        };
        debug_assert_eq!(unscored_candidates.len(), candidates.len());
        score_contributions.resize(candidates.len(), f64::NAN);
        for (scorer_index, (inputs, scorer)) in scorers.iter_mut().enumerate() {
            score_contributions.fill(f64::NAN);
            scorer.score(
                context,
                WorkerCandidates::new(unscored_candidates, *inputs, cache_snapshot, &info),
                score_contributions,
            )?;
            for (row, (contribution, scored)) in score_contributions
                .iter()
                .zip(candidates.iter_mut())
                .enumerate()
            {
                let cost = scored.cost + contribution;
                if !contribution.is_finite() || !cost.is_finite() {
                    return Err(
                        WorkerSelectionPolicyError::NonFiniteCost { scorer_index, row }.into(),
                    );
                }
                scored.cost = cost;
            }
        }
        Ok(())
    }

    fn pick<S: CandidateRowSource>(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        cache_snapshot: &CacheSnapshot<'_>,
        source: &S,
        worker_label: &'static str,
        has_eligible_worker: bool,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<RowPick, SelectionFailure> {
        let Self {
            picker,
            picker_inputs,
            candidates,
            cache_inputs,
            load_inputs,
            max_raw_cached_tokens,
            ..
        } = self;
        if candidates.is_empty() {
            return Err(if has_eligible_worker {
                SelectionFailure::AllFiltered
            } else {
                SelectionFailure::NoCandidates
            });
        }
        debug_assert!(
            !picker_inputs.contains(WorkerInputs::CACHE) || cache_inputs.len() == candidates.len()
        );
        debug_assert!(
            !picker_inputs.contains(WorkerInputs::LOAD) || load_inputs.len() == candidates.len()
        );
        let candidates: &[ScoredWorkerCandidate] = candidates;
        let info = |row: usize| {
            let worker = candidates[row].worker;
            WorkerInfo::new(worker, source.capacity(worker))
        };
        let picker_input = WorkerInputView {
            candidates,
            cache: picker_inputs
                .contains(WorkerInputs::CACHE)
                .then_some(WorkerCacheInputs {
                    rows: cache_inputs,
                    snapshot: cache_snapshot,
                }),
            load: picker_inputs
                .contains(WorkerInputs::LOAD)
                .then_some(load_inputs.as_slice()),
            info: &info,
        };
        let row = picker.pick(context, picker_input)?;
        let Some(candidate) = candidates.get(row) else {
            return Err(WorkerSelectionPolicyError::InvalidPickerRow {
                row,
                candidate_count: candidates.len(),
            }
            .into());
        };
        log_scored_candidates(context.request_id(), worker_label, candidates, row);
        let trace = trace.map(|spec| {
            build_selection_trace(candidates, row, context.pinned_worker().is_some(), spec)
        });
        Ok(RowPick {
            worker: candidate.worker,
            cost: candidate.cost,
            max_raw_cached_tokens: *max_raw_cached_tokens,
            trace,
        })
    }
}

/// Pushes every rank's row at the scorer and picker inputs: the filter-less walk.
struct PushRows<'s> {
    state: &'s mut ComposedPolicyState,
    inputs: WorkerInputs,
}

impl RankVisitor for PushRows<'_> {
    #[inline(always)]
    fn visit<R: CandidateRank>(&mut self, rank: &R) -> bool {
        self.state.push_candidate(rank.row(self.inputs));
        false
    }
}

/// Runs the filters on each rank and pushes the kept ranks' rows. A rank yields only
/// what the filters asked for until the filters keep it, so a rejected rank never
/// materializes scorer inputs.
struct FilterRows<'s, 'c> {
    state: &'s mut ComposedPolicyState,
    context: &'s WorkerSelectionContext<'c>,
    cache_snapshot: &'s CacheSnapshot<'c>,
    filter_inputs: WorkerInputs,
    scorer_picker_inputs: WorkerInputs,
    additional_inputs: WorkerInputs,
    has_eligible_worker: bool,
    error: Option<WorkerSelectionPolicyError>,
}

impl RankVisitor for FilterRows<'_, '_> {
    #[inline(always)]
    fn visit<R: CandidateRank>(&mut self, rank: &R) -> bool {
        self.has_eligible_worker = true;
        let filter_candidate = rank.row(self.filter_inputs);
        let info = |_: usize| WorkerInfo::new(filter_candidate.worker, rank.capacity());
        for (inputs, filter) in &mut self.state.filters {
            match filter.keep(
                self.context,
                WorkerCandidate::new(&filter_candidate, *inputs, self.cache_snapshot, &info),
            ) {
                Ok(true) => {}
                Ok(false) => return false,
                Err(policy_error) => {
                    self.error = Some(policy_error);
                    return true;
                }
            }
        }
        let additional = rank.row(self.additional_inputs);
        let candidate = filter_candidate.with_inputs_from(&additional, self.scorer_picker_inputs);
        self.state.push_candidate(candidate);
        false
    }
}

/// Builds a [`WorkerSelectionPolicy`]'s candidate rows before the request reaches
/// the scheduler actor.
///
/// Rows hold every eligible rank's cache, preferred-taint, and fetch inputs, which
/// depend only on the request and worker configuration. Load inputs depend on the
/// actor's earlier bookings, so the actor fills them at selection.
#[derive(Clone)]
pub struct CandidatePreparer {
    inputs: WorkerInputs,
    row_buffers: Arc<ArrayQueue<Vec<CandidateData>>>,
}

/// Rows a [`CandidatePreparer`] built, in a buffer that returns to the policy's pool.
pub struct PreparedRows {
    inputs: WorkerInputs,
    rows: Vec<CandidateData>,
    row_buffers: Arc<ArrayQueue<Vec<CandidateData>>>,
}

impl Drop for PreparedRows {
    fn drop(&mut self) {
        // Back to the pool for the next preparation; a full pool drops it instead.
        if self.rows.capacity() > 0 {
            let _ = self.row_buffers.push(std::mem::take(&mut self.rows));
        }
    }
}

impl CandidatePreparer {
    /// The columns the policy reads. Rows are built without `LOAD`, which the actor fills.
    pub fn inputs(&self) -> WorkerInputs {
        self.inputs
    }

    /// Build rows into a pooled buffer. `build` receives the columns to fill now, every
    /// declared input but `LOAD`, and pushes one row per eligible rank. `capacity_hint`
    /// sizes a fresh buffer when the pool is empty. Returns `None` when no row was built,
    /// so that selection reports why.
    pub fn prepare_with(
        &self,
        capacity_hint: usize,
        build: impl FnOnce(WorkerInputs, &mut Vec<CandidateData>),
    ) -> Option<PreparedRows> {
        let mut rows = self
            .row_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(capacity_hint));
        rows.clear();
        build(self.inputs.without(WorkerInputs::LOAD), &mut rows);
        if rows.is_empty() {
            return None;
        }
        Some(PreparedRows {
            inputs: self.inputs,
            rows,
            row_buffers: Arc::clone(&self.row_buffers),
        })
    }
}

impl PreparedRows {
    /// The columns the rows will hold once the actor fills `LOAD`.
    pub fn inputs(&self) -> WorkerInputs {
        self.inputs
    }

    /// The rows, holding every declared column but `LOAD`.
    pub fn rows(&self) -> &[CandidateData] {
        &self.rows
    }
}

/// Copy the retained candidate costs out of the policy's scratch rows in one pass.
///
/// Keeps every candidate when `spec.top_k` is zero or not below the candidate count;
/// otherwise the `top_k` lowest-cost rows plus the selected row when it is not among
/// them. Rows are unordered: this runs on the scheduler actor, so sorting is left to
/// the trace emitter on the routing task.
fn build_selection_trace(
    candidates: &[ScoredWorkerCandidate],
    selected_row: usize,
    pinned: bool,
    spec: SelectionTraceSpec,
) -> SelectionTrace {
    let candidate_count = candidates.len();
    let row_of = |candidate: &ScoredWorkerCandidate| SelectionTraceRow {
        worker: candidate.worker,
        cost: candidate.cost,
    };
    let rows = if spec.top_k == 0 || spec.top_k >= candidate_count {
        candidates.iter().map(row_of).collect()
    } else {
        let mut rows: Vec<SelectionTraceRow> = Vec::with_capacity(spec.top_k + 1);
        // Index into `rows` of the costliest retained row.
        let mut costliest = 0;
        for candidate in candidates {
            if rows.len() < spec.top_k {
                if rows.is_empty() || candidate.cost > rows[costliest].cost {
                    costliest = rows.len();
                }
                rows.push(row_of(candidate));
            } else if candidate.cost < rows[costliest].cost {
                rows[costliest] = row_of(candidate);
                for (index, row) in rows.iter().enumerate() {
                    if row.cost > rows[costliest].cost {
                        costliest = index;
                    }
                }
            }
        }
        let selected = &candidates[selected_row];
        if !rows.iter().any(|row| row.worker == selected.worker) {
            rows.push(row_of(selected));
        }
        rows
    };
    SelectionTrace {
        candidate_count,
        pinned,
        rows,
    }
}

/// Log every scored candidate at TRACE, one row per worker and rank, stamped with
/// `request_id` and `worker_type` because the scheduler actor task has no request span.
fn log_scored_candidates(
    request_id: Option<&str>,
    worker_type: &'static str,
    candidates: &[ScoredWorkerCandidate],
    selected_row: usize,
) {
    if !tracing::enabled!(tracing::Level::TRACE) {
        return;
    }
    let request_id = request_id.unwrap_or("-");
    for (row, candidate) in candidates.iter().enumerate() {
        tracing::trace!(
            request_id,
            worker_type,
            worker_id = candidate.worker.worker_id,
            dp_rank = candidate.worker.dp_rank,
            cost = candidate.cost,
            selected = row == selected_row,
            "Scored worker candidate"
        );
    }
}

#[cfg(test)]
mod selection_trace_tests {
    use super::*;

    fn candidates(costs: &[f64]) -> Vec<ScoredWorkerCandidate> {
        costs
            .iter()
            .enumerate()
            .map(|(id, cost)| ScoredWorkerCandidate {
                worker: WorkerWithDpRank::from_worker_id(id as u64),
                cost: *cost,
                preferred_taint_multiplier: None,
            })
            .collect()
    }

    fn worker_ids_by_cost(trace: &SelectionTrace) -> Vec<u64> {
        let mut rows = trace.rows.clone();
        rows.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        rows.iter().map(|row| row.worker.worker_id).collect()
    }

    #[test]
    fn zero_top_k_keeps_every_candidate() {
        let rows = candidates(&[3.0, 1.0, 2.0]);
        let trace = build_selection_trace(&rows, 2, false, SelectionTraceSpec { top_k: 0 });
        assert_eq!(trace.candidate_count, 3);
        assert_eq!(worker_ids_by_cost(&trace), vec![1, 2, 0]);
        assert!(!trace.pinned);
    }

    #[test]
    fn top_k_keeps_lowest_costs_and_appends_selected_row() {
        let rows = candidates(&[5.0, 1.0, 4.0, 2.0, 3.0, 0.5]);
        // Selected row 0 costs 5.0, the worst candidate: it is appended after the three cheapest.
        let trace = build_selection_trace(&rows, 0, true, SelectionTraceSpec { top_k: 3 });
        assert_eq!(trace.candidate_count, 6);
        assert_eq!(worker_ids_by_cost(&trace), vec![5, 1, 3, 0]);
        assert!(trace.pinned);

        // Selected row already among the cheapest: nothing is appended.
        let trace = build_selection_trace(&rows, 1, false, SelectionTraceSpec { top_k: 3 });
        assert_eq!(worker_ids_by_cost(&trace), vec![5, 1, 3]);
    }

    #[test]
    fn top_k_at_or_above_candidate_count_keeps_everything() {
        let rows = candidates(&[2.0, 1.0]);
        let trace = build_selection_trace(&rows, 0, false, SelectionTraceSpec { top_k: 8 });
        assert_eq!(worker_ids_by_cost(&trace), vec![1, 0]);
    }

    #[test]
    fn top_k_pass_tracks_the_costliest_retained_row() {
        // Descending costs force a replacement on every candidate after the first k.
        let rows = candidates(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]);
        let trace = build_selection_trace(&rows, 6, false, SelectionTraceSpec { top_k: 2 });
        assert_eq!(worker_ids_by_cost(&trace), vec![6, 5]);
    }
}

/// The stage over a table that stands in for the kernel: no request, eligibility, or
/// worker configuration in sight.
#[cfg(test)]
mod row_source_tests {
    use super::*;
    use crate::{WorkerCacheData, WorkerCapacity};
    use rustc_hash::FxHashMap;
    use std::cell::Cell;

    struct Rank {
        worker: WorkerWithDpRank,
        raw_cached_tokens: u32,
        device_overlap_blocks: f64,
        capacity: WorkerCapacity,
    }

    struct Table {
        ranks: Vec<Rank>,
        loads: FxHashMap<WorkerWithDpRank, WorkerLoadInput>,
        rows_built: Cell<usize>,
    }

    impl Table {
        fn new(ranks: Vec<Rank>) -> Self {
            Self {
                ranks,
                loads: FxHashMap::default(),
                rows_built: Cell::new(0),
            }
        }

        fn row_of(&self, rank: &Rank, inputs: WorkerInputs) -> CandidateData {
            self.rows_built.set(self.rows_built.get() + 1);
            CandidateData {
                worker: rank.worker,
                inputs,
                raw_cached_tokens: rank.raw_cached_tokens,
                cache: if inputs.contains(WorkerInputs::CACHE) {
                    WorkerCacheData {
                        device_overlap_blocks: rank.device_overlap_blocks,
                        ..Default::default()
                    }
                } else {
                    WorkerCacheData::default()
                },
                load: if inputs.contains(WorkerInputs::LOAD) {
                    self.load(rank.worker)
                } else {
                    WorkerLoadInput::default()
                },
                transfer: None,
                preferred_taint_multiplier: None,
            }
        }
    }

    struct TableRank<'r> {
        table: &'r Table,
        rank: &'r Rank,
    }

    impl CandidateRank for TableRank<'_> {
        fn worker(&self) -> WorkerWithDpRank {
            self.rank.worker
        }

        fn capacity(&self) -> WorkerCapacity {
            self.rank.capacity
        }

        fn row(&self, inputs: WorkerInputs) -> CandidateData {
            self.table.row_of(self.rank, inputs)
        }
    }

    impl CandidateRowSource for Table {
        fn visit_ranks<V: RankVisitor>(&self, visitor: &mut V) {
            for rank in &self.ranks {
                if visitor.visit(&TableRank { table: self, rank }) {
                    return;
                }
            }
        }

        fn capacity(&self, worker: WorkerWithDpRank) -> WorkerCapacity {
            self.ranks
                .iter()
                .find(|rank| rank.worker == worker)
                .map_or_else(WorkerCapacity::default, |rank| rank.capacity)
        }

        fn load(&self, worker: WorkerWithDpRank) -> WorkerLoadInput {
            self.loads.get(&worker).copied().unwrap_or_default()
        }
    }

    /// Collects every rank's row at `inputs`, as the kernel's preparer does.
    struct CollectRows<'v> {
        inputs: WorkerInputs,
        rows: &'v mut Vec<CandidateData>,
    }

    impl RankVisitor for CollectRows<'_> {
        fn visit<R: CandidateRank>(&mut self, rank: &R) -> bool {
            self.rows.push(rank.row(self.inputs));
            false
        }
    }

    fn context() -> WorkerSelectionContext<'static> {
        let context = WorkerSelectionContext::new(64, 16)
            .with_request_id(Some("stage"))
            .with_prefill_tracking(true)
            .with_tier_matches(true);
        assert_eq!(context.request_blocks(), 4);
        context
    }

    fn rank(worker_id: u64, raw_cached_tokens: u32, device_overlap_blocks: f64) -> Rank {
        Rank {
            worker: WorkerWithDpRank::from_worker_id(worker_id),
            raw_cached_tokens,
            device_overlap_blocks,
            capacity: WorkerCapacity::new(Some(8192), Some(1024 + worker_id)),
        }
    }

    struct RejectWorker(WorkerWithDpRank);

    impl WorkerFilter for RejectWorker {
        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: WorkerCandidate<'_>,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            // A filter that declared nothing sees no columns.
            assert!(candidate.cache().is_none());
            assert!(candidate.load().is_none());
            Ok(candidate.worker() != self.0)
        }
    }

    struct DeviceOverlapScorer;

    impl WorkerScorer for DeviceOverlapScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidates: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            for (candidate, cost) in candidates.iter().zip(costs) {
                *cost = -candidate.cache().unwrap().device_overlap_blocks();
            }
            Ok(())
        }
    }

    struct LowestCostPicker;

    impl WorkerPicker for LowestCostPicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            input
                .candidates()
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
                .map(|(row, _)| row)
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no candidates"))
        }
    }

    #[test]
    fn empty_and_filtered_tables_are_distinct_failures() {
        let policy = WorkerSelectionPolicy::new_with_filters(
            SelectionConfig::default(),
            "test",
            vec![Box::new(RejectWorker(WorkerWithDpRank::from_worker_id(0)))],
            vec![Box::new(DeviceOverlapScorer)],
            Box::new(LowestCostPicker),
        );

        let empty = Table::new(Vec::new());
        assert!(matches!(
            policy.select_rows(&context(), &empty, None, None),
            Err(SelectionFailure::NoCandidates)
        ));

        let rejected = Table::new(vec![rank(0, 96, 6.0)]);
        assert!(matches!(
            policy.select_rows(&context(), &rejected, None, None),
            Err(SelectionFailure::AllFiltered)
        ));
        // The rejected rank was built once, at the filter's inputs, and never again.
        assert_eq!(rejected.rows_built.get(), 1);
    }

    #[test]
    fn kept_rows_alone_reach_scorers_and_the_reuse_maximum() {
        let rejected = WorkerWithDpRank::from_worker_id(0);
        let policy = WorkerSelectionPolicy::new_with_filters(
            SelectionConfig::default(),
            "test",
            vec![Box::new(RejectWorker(rejected))],
            vec![Box::new(DeviceOverlapScorer)],
            Box::new(LowestCostPicker),
        );
        let table = Table::new(vec![rank(0, 96, 6.0), rank(1, 16, 1.0), rank(2, 32, 2.0)]);

        let pick = policy
            .select_rows(
                &context(),
                &table,
                None,
                Some(SelectionTraceSpec { top_k: 0 }),
            )
            .unwrap();

        assert_eq!(pick.worker, WorkerWithDpRank::from_worker_id(2));
        assert_eq!(pick.cost, -2.0);
        // The rejected rank's 96 tokens do not count as available reuse.
        assert_eq!(pick.max_raw_cached_tokens, 32);
        let trace = pick.trace.unwrap();
        assert_eq!(trace.candidate_count, 2);
        assert!(!trace.pinned);
        // Three filter rows plus one additional row per kept rank.
        assert_eq!(table.rows_built.get(), 5);
    }

    #[test]
    fn prepared_rows_take_their_load_from_the_source_at_selection() {
        struct LeastLoaded;

        impl WorkerPicker for LeastLoaded {
            fn required_worker_inputs(&self) -> WorkerInputs {
                WorkerInputs::LOAD
            }

            fn pick(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                let loads = input.load().expect("declared load inputs");
                Ok(loads
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, load)| load.active_decode_blocks())
                    .map(|(row, _)| row)
                    .unwrap())
            }
        }

        let policy = WorkerSelectionPolicy::new(
            SelectionConfig::default(),
            "test",
            vec![Box::new(DeviceOverlapScorer)],
            Box::new(LeastLoaded),
        );
        let preparer = policy
            .candidate_preparer()
            .expect("filter-less policies prepare");
        assert_eq!(preparer.inputs(), WorkerInputs::CACHE | WorkerInputs::LOAD);
        let mut table = Table::new(vec![rank(0, 0, 1.0), rank(1, 0, 1.0)]);
        let prepared = preparer
            .prepare_with(table.ranks.len(), |inputs, rows| {
                assert_eq!(inputs, WorkerInputs::CACHE);
                table.visit_ranks(&mut CollectRows { inputs, rows });
            })
            .unwrap();
        assert!(policy.accepts_prepared(&prepared));
        assert_eq!(prepared.rows().len(), 2);
        // Bookings land after preparation; selection must read them.
        table.loads.insert(
            WorkerWithDpRank::from_worker_id(0),
            WorkerLoadInput {
                available: true,
                active_decode_blocks: 9,
                ..Default::default()
            },
        );
        table.loads.insert(
            WorkerWithDpRank::from_worker_id(1),
            WorkerLoadInput {
                available: true,
                active_decode_blocks: 2,
                ..Default::default()
            },
        );
        let rows_before = table.rows_built.get();

        let pick = policy
            .select_rows(&context(), &table, Some(prepared), None)
            .unwrap();

        assert_eq!(pick.worker, WorkerWithDpRank::from_worker_id(1));
        // Prepared rows are adopted, not rebuilt.
        assert_eq!(table.rows_built.get(), rows_before);

        let filtered = WorkerSelectionPolicy::new_with_filters(
            SelectionConfig::default(),
            "test",
            vec![Box::new(RejectWorker(WorkerWithDpRank::from_worker_id(0)))],
            Vec::new(),
            Box::new(LeastLoaded),
        );
        assert!(filtered.candidate_preparer().is_none());
    }

    #[test]
    fn picker_rows_are_bounds_checked() {
        struct OutOfRange;

        impl WorkerPicker for OutOfRange {
            fn pick(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                Ok(input.candidates().len())
            }
        }

        let policy = WorkerSelectionPolicy::new(
            SelectionConfig::default(),
            "test",
            Vec::new(),
            Box::new(OutOfRange),
        );
        let table = Table::new(vec![rank(0, 0, 0.0)]);

        assert!(matches!(
            policy.select_rows(&context(), &table, None, None),
            Err(SelectionFailure::Policy(
                WorkerSelectionPolicyError::InvalidPickerRow {
                    row: 1,
                    candidate_count: 1,
                }
            ))
        ));
    }
}
