// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Work a caller prepares for its request before the actor sees it, and the checks the actor
//! makes before it uses that work. `PreparedAdmission`'s doc lists them.

use super::*;

/// Candidate rows built on the caller task, before the request reaches the actor.
///
/// The actor selects from them only when it would build the same rows: the worker
/// configuration, availability, and overload it reads at selection match the ones
/// the rows were built from. Load inputs and the ranks past the host's reject
/// thresholds depend on the actor's earlier bookings. The caller fills the loads from
/// its own projection, `load`; at selection the actor fills again those of the ranks it
/// wrote since, or all of them, and drops the rows of workers whose every rank is full.
/// Full ranks require fresh rows only when exclusive affinity or fetch sources can
/// change preparation's eligibility.
///
/// The actor's checks, in the order it makes them:
///
/// | Check | The caller read | The actor requires | Otherwise it |
/// |---|---|---|---|
/// | `TurnPrepared::drop_judged_load` | reject limits | no await since | builds load |
/// | `TurnPrepared::take_for` | | same turn, no refreshed overlap | builds rows and load |
/// | `adopt_prepared_load` | limits, overlap, writes, table version | see there | builds load |
/// | `adopt_prepared_rows` | generation, availability, overload | see there | builds rows |
/// | `select_with_policy` | the policy's inputs | same inputs, no filters | builds rows |
///
/// "Builds load" is the actor projecting every rank itself. When preparation starts to read
/// another input, add it to the check that owns it and to this table.
pub(super) struct PreparedAdmission {
    candidates: PreparedCandidates,
    /// Configuration generation, loaded before the configuration was read.
    workers_generation: u64,
    available_worker_ids: Option<Arc<WorkerIdSet>>,
    overloaded_worker_ids: Option<Arc<HashSet<WorkerId>>>,
    load: Option<PreparedLoad>,
}

/// Projected loads the caller read for its request, with the ranks that pass found past
/// the host's reject thresholds. The actor re-reads and judges again the ranks it wrote
/// since, then selects with them instead of projecting every rank itself. A rank another
/// task changed since (a release, a prefill mark, a replica's booking) keeps the pass's
/// load, fullness and prefill decay for this selection. The commands ahead of the caller's
/// (under `ADMISSION_CHANNEL_CAPACITY + ADMISSION_BATCH`) and what they drain lie between.
pub(super) struct PreparedLoad {
    loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    /// The ranks of `loads` past `limits`; empty without limits.
    full_ranks: FullRankCounts,
    /// The reject thresholds the pass judged, when any was configured.
    limits: Option<BusyThresholds>,
    /// The actor's write count, loaded before the pass.
    actor_writes: u64,
    /// The load table's version, as the pass read it.
    load_table_version: u64,
    decay_now: Instant,
    /// Whether the pass counted each rank's overlap with the request's active prompts.
    overlap: bool,
    pool: Arc<LoadMapPool>,
}

impl Drop for PreparedLoad {
    fn drop(&mut self) {
        // Back to the pool for the next preparation; a full pool drops it instead.
        if self.loads.capacity() > 0 {
            let _ = self.pool.push((
                std::mem::take(&mut self.loads),
                std::mem::take(&mut self.full_ranks),
            ));
        }
    }
}

/// Reused projection maps with their full-rank counts, so a caller's pass refreshes one
/// in place.
pub(super) type LoadMapPool = ArrayQueue<(
    FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    FullRankCounts,
)>;

pub(super) const LOAD_MAPS: usize = 64;

/// A prepared load is re-read rank by rank only while at most one rank in this many
/// changed; past that, projecting every rank costs about as much.
const PREPARED_LOAD_MAX_STALE: usize = 16;

/// The rows a caller prepared for the request the actor received in its current turn.
///
/// A turn is one command or wakeup, including one handled during an overlap refresh. A
/// request admitted in the turn that received it may select from its caller's rows. The
/// next turn drops them: a request still queued rebuilds its rows when it is admitted, so
/// parked requests hold none. A request admitted on arrival without entering the queue
/// takes its rows straight to admission and never passes through here.
#[derive(Default)]
pub(super) struct TurnPrepared {
    turn: u64,
    prepared: Option<Box<PreparedAdmission>>,
}

impl TurnPrepared {
    /// Starts the next turn and drops the last one's rows.
    pub(super) fn begin_turn(&mut self) {
        self.turn = self.turn.wrapping_add(1);
        self.prepared = None;
    }

    /// The current turn, stamped on a request as it enters the queue.
    pub(super) fn turn(&self) -> u64 {
        self.turn
    }

    /// Holds the rows of the request that enters the queue this turn.
    pub(super) fn stash(&mut self, prepared: Option<Box<PreparedAdmission>>) {
        self.prepared = prepared;
    }

    /// After an await inside a turn: the held rows' request is not admitted yet, and the
    /// fullness its caller judged before the await is stale, so the actor judges again.
    pub(super) fn drop_judged_load(&mut self) {
        if let Some(prepared) = &mut self.prepared {
            prepared.load.take_if(|load| load.limits.is_some());
        }
    }

    /// The rows for a request leaving the queue, unless it arrived in an earlier turn or a
    /// refresh replaced the overlap they were built from.
    pub(super) fn take_for(
        &mut self,
        enqueue_turn: u64,
        refreshed: bool,
    ) -> Option<Box<PreparedAdmission>> {
        if enqueue_turn == self.turn && !refreshed {
            self.prepared.take()
        } else {
            None
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    /// Build candidate rows and initial loads before the request reaches the actor.
    pub(super) fn prepare_candidates(
        &self,
        request: &SchedulingRequest,
    ) -> Option<Box<PreparedAdmission>> {
        if request.pinned_worker.is_some() {
            return None;
        }
        let preparer = self.candidate_preparer.load();
        let preparer = preparer.as_deref()?;
        // Load before reading the configuration: a change after this load
        // advances the generation before the actor selects with these rows.
        let workers_generation = self.workers_generation.load(AtomicOrdering::Acquire);
        let (mut candidates, available_worker_ids, overloaded_worker_ids) = self.eligibility.with(
            request,
            &self.workers_with_configs,
            &FullRankCounts::default(),
            true,
            |workers, eligibility, _, available, overloaded| {
                let candidates = preparer.prepare(WorkerSelectionInput::configured(
                    workers,
                    request,
                    eligibility,
                    self.block_size,
                ))?;
                Some((candidates, available.cloned(), overloaded.cloned()))
            },
        )?;
        let load = self.prepare_load(request, preparer.inputs().contains(WorkerInputs::LOAD));
        candidates.fill_loads(request, &load.loads);
        Some(Box::new(PreparedAdmission {
            candidates,
            workers_generation,
            available_worker_ids,
            overloaded_worker_ids,
            load: Some(load),
        }))
    }

    /// Project every rank's load for `request` on the caller task and judge it against the
    /// host's reject thresholds, as the actor would with no writes of its own in between.
    fn prepare_load(&self, request: &SchedulingRequest, overlap: bool) -> PreparedLoad {
        let limits = reject_limits(self.reject_thresholds.as_ref());
        // Before any rank is read: a write this count includes is visible to the pass.
        let actor_writes = self.actor_writes.load(AtomicOrdering::Acquire);
        let decay_now = Instant::now();
        let (mut loads, mut full_ranks) = self.load_maps.pop().unwrap_or_default();
        let load_table_version = self.slots.project_worker_loads_into(
            request.token_seq.as_deref().filter(|_| overlap),
            decay_now,
            &mut loads,
            reset_full_ranks(&mut full_ranks, limits),
        );
        PreparedLoad {
            loads,
            full_ranks,
            limits,
            actor_writes,
            load_table_version,
            decay_now,
            overlap,
            pool: Arc::clone(&self.load_maps),
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// Whether a load projection counts each rank's overlap with the request's active prompts.
    /// Finding it walks the prompt trie, and only `LOAD` rows read it; without them, only the
    /// selected rank's is computed.
    pub(super) fn projects_overlap(&self) -> bool {
        self.selector
            .required_worker_inputs()
            .contains(WorkerInputs::LOAD)
    }

    /// Refresh derived load for every request, retaining the allocation through
    /// selection, booking, and response-field capture, including error paths. A load
    /// the caller prepared stands in for the projection when
    /// [`Self::adopt_prepared_load`] takes it.
    pub(super) fn with_projected_loads<R>(
        &mut self,
        mut request: SchedulingRequest,
        decay_now: Instant,
        mut prepared: Option<Box<PreparedAdmission>>,
        handle: impl FnOnce(&Self, &mut SchedulingRequest, Option<Box<PreparedAdmission>>) -> R,
    ) -> R {
        let limits = reject_limits(self.reject_thresholds.as_ref());
        let project_overlap = self.projects_overlap();
        let mut adopted = prepared.as_deref_mut().and_then(|prepared| {
            self.adopt_prepared_load(prepared, &request, project_overlap, limits)
        });
        match &mut adopted {
            Some(load) => {
                request.worker_loads = std::mem::take(&mut load.loads);
                // The caller's counts are this selection's; the last pass's go to the pool.
                std::mem::swap(&mut self.full_ranks, &mut load.full_ranks);
            }
            None => {
                request.worker_loads = std::mem::take(&mut self.projected_loads);
                // The host's reject thresholds are judged in the same pass that projects each
                // rank's load, so selection only consults the ranks found full: no second read
                // per rank.
                self.slots.project_worker_loads_into(
                    request.token_seq.as_deref().filter(|_| project_overlap),
                    decay_now,
                    &mut request.worker_loads,
                    reset_full_ranks(&mut self.full_ranks, limits),
                );
            }
        }
        let result = handle(self, &mut request, prepared);
        let loads = std::mem::take(&mut request.worker_loads);
        match adopted {
            // Back to the callers' pool as it drops.
            Some(mut load) => load.loads = loads,
            None => self.projected_loads = loads,
        }
        result
    }

    /// The load the caller prepared with `prepared`'s rows, with every rank this actor
    /// wrote since re-read at the caller's instant and judged again against `limits`, and
    /// the rows told to keep their filled loads but for those ranks. None, and the rows
    /// fill every load at selection, when it no longer stands for the pass this actor
    /// would run: other reject thresholds or overlap, a rank added or removed or reporting
    /// another capacity, or more writes than the log holds or than re-reading them beats
    /// projecting every rank.
    pub(super) fn adopt_prepared_load(
        &mut self,
        prepared: &mut PreparedAdmission,
        request: &SchedulingRequest,
        overlap: bool,
        limits: Option<BusyThresholds>,
    ) -> Option<PreparedLoad> {
        let mut load = prepared.load.take()?;
        if load.limits != limits || load.overlap != overlap {
            return None;
        }
        let ranks = self.slots.written_since(load.actor_writes)?;
        if ranks.len().saturating_mul(PREPARED_LOAD_MAX_STALE) > load.loads.len() {
            return None;
        }
        self.reread_ranks.clear();
        self.reread_ranks.extend(ranks);
        self.reread_ranks.sort_unstable();
        self.reread_ranks.dedup();
        self.slots
            .reproject_worker_loads(
                request.token_seq.as_deref().filter(|_| overlap),
                load.decay_now,
                &self.reread_ranks,
                load.load_table_version,
                &mut load.loads,
                limits.is_some().then_some(&mut load.full_ranks),
            )
            .then(|| {
                prepared.candidates.keep_filled_loads(&self.reread_ranks);
                load
            })
    }

    /// The rows the caller prepared, when the actor would build the same ones now; see
    /// [`PreparedAdmission`]. None, and the selector builds its rows, otherwise. The
    /// configuration is judged under the selection's borrow of `workers`.
    pub(super) fn adopt_prepared_rows(
        &self,
        prepared: Option<Box<PreparedAdmission>>,
        request: &SchedulingRequest,
        workers: &watch::Ref<'_, HashMap<WorkerId, C>>,
        available_worker_ids: Option<&Arc<WorkerIdSet>>,
        overloaded_worker_ids: Option<&Arc<HashSet<WorkerId>>>,
    ) -> Option<PreparedCandidates> {
        let full_ranks_keep_rows = self.full_ranks.is_empty()
            || (!(self.eligibility.exclusive && request.affinity_target.is_some())
                && request.kv_transfer_candidates.is_none());
        let prepared = prepared?;
        // Availability and overload are each one shared snapshot until they change, so
        // these usually compare pointers.
        (!workers.has_changed()
            && full_ranks_keep_rows
            && prepared.workers_generation == self.workers_generation.load(AtomicOrdering::Relaxed)
            && prepared.available_worker_ids.as_ref() == available_worker_ids
            && prepared.overloaded_worker_ids.as_ref() == overloaded_worker_ids)
            .then_some(prepared.candidates)
    }

    /// Advance the generation when the configuration changed since the actor
    /// last looked, so rows prepared against an earlier one are rebuilt.
    pub(super) fn refresh_workers_generation(&mut self) {
        if self.workers_with_configs.borrow_and_update().has_changed() {
            self.workers_generation
                .fetch_add(1, AtomicOrdering::Release);
        }
    }
}
