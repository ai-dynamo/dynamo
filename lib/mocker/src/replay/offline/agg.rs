// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use super::components::OfflineRouterSnapshot;
pub(super) use super::components::ReplayMode;
#[cfg(test)]
use super::components::TrafficStats;
use super::core::pinned::ExternalPlacement;
use super::core::round_robin::AggregatedRoundRobinPlacement;
use super::core::{
    AdmissionSource as CoreAdmissionSource, EngineEventBatch, NoEngineEvents, Placement,
    PlacementDecision, PlacementPolicy, ReadyArrival, WorkerTopology,
};
use super::events::{SimulationEvent, SimulationWorkerStage, WorkerCompletionPayload};
use super::evidence::{
    KvIngestBoundary, WorkerLifecycleTransition, WorkerLifecycleTransitionKind, WorkerPool,
    WorkerPoolState, attach_pressure_references, drain_origin, lifecycle_capture_active,
    record_lifecycle_operation, startup_origin,
};
#[cfg(test)]
use super::extensions::kv_router::AggRuntime;
use super::interactive::{
    CapturedReplayEvent, InteractiveCapture, InteractivePlacementObservation,
    InteractiveRequestIdentity, ReplayEvent, ReplayPendingPlacement, ReplayPlacementCandidate,
    ReplaySnapshot, ReplayStepStatus, ReplayWorkerSnapshot,
};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    ReadyWorkerCompletions, next_timestamp as choose_next_timestamp, pop_ready_scaling_tick,
    pop_ready_worker_completions, pop_ready_worker_ready, push_scaling_tick,
    push_worker_completions, push_worker_ready,
};
#[cfg(test)]
use super::scaling::ReplayScalingDecision;
use super::scaling::{LatestFpmBuffer, ReplayScalingPolicy, ReplayScalingSnapshot};
#[cfg(test)]
use super::state::AggRequestPhase;
#[cfg(test)]
use super::state::OfflineWorkerSnapshot;
use super::topology::{
    DEFAULT_REPLAY_POOL_ID, ResolvedPoolTopology, ResolvedPoolWorker, WorkerTarget,
};
use super::{
    components::{
        AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, NoReplayMetadata,
        ReplayAdmissionMetadata, ReplayEngineObservation, TrafficAccumulator, WorkerScaleDelta,
    },
    state::AggRequestState,
};
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot, MockEngineArgs, OutputSignal};
use crate::loadgen::{
    AgenticTrace, CascadedWorkloadTerminal, ReplayRequestPayload, WorkloadDriver,
    WorkloadTerminalStatus,
};
#[cfg(test)]
use crate::replay::ReplayRouterMode;
use crate::replay::collector::{DecodeWorkerAccountingHandle, TraceCollector};
use crate::replay::{ReplayRequestPool, ReplayTerminalStatus, ReplayWorkerLifecycleStatus};
use anyhow::bail;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
#[cfg(test)]
use std::collections::HashMap;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

fn common_origin(mut origins: impl Iterator<Item = u64>) -> Option<u64> {
    let first = origins.next()?;
    origins.all(|origin| origin == first).then_some(first)
}

#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(in crate::replay) struct AggRuntimeStats {
    dispatch_history: Vec<usize>,
    dispatch_order: Vec<Uuid>,
    assigned_worker_by_uuid: HashMap<Uuid, usize>,
    overlap_history: Vec<u32>,
    max_in_flight_seen: usize,
    prefill_marked_count: usize,
    router_freed_count: usize,
    max_router_pending_count: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct AggRuntimeSnapshot {
    now_ms: f64,
    worker_active_requests: Vec<Vec<Uuid>>,
    workers: Vec<OfflineWorkerSnapshot>,
    router_pending_request_ids: Vec<Uuid>,
    prefill_completed: Vec<Uuid>,
    router: Option<OfflineRouterSnapshot>,
}

#[cfg(not(test))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(in crate::replay) struct AggRuntimeStats;

/// One validated scheduler assignment with its authored topology identity.
/// Keeping these facts together avoids repeatedly resolving the rank and
/// cloning the pool ID between overlap, evidence, and physical dispatch.
#[derive(Debug)]
struct ResolvedPlacement {
    placement: Placement,
    logical_worker_id: usize,
    dp_rank: u32,
    authored_target: WorkerTarget,
}

pub(in crate::replay) trait AggregatedPlacement<Events, Metadata>:
    PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Events> + Sized
where
    Events: EngineEventBatch,
    Metadata: ReplayAdmissionMetadata,
{
    /// Publish a worker lifecycle transition together with the stable authored
    /// identity used by interactive/multipool replay. Native policies operate
    /// on dense engine topology and use the default adapters; external
    /// placement overrides them so sparse authored IDs remain coherent after
    /// dynamic scaling.
    fn worker_ready_authored(
        &mut self,
        worker: WorkerTopology,
        _target: &WorkerTarget,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.worker_ready(worker, now_ms)
    }

    fn worker_draining_authored(
        &mut self,
        worker: WorkerTopology,
        _target: &WorkerTarget,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.worker_draining(worker, now_ms)
    }

    fn worker_removed_authored(
        &mut self,
        worker: WorkerTopology,
        _target: &WorkerTarget,
        now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        self.worker_removed(worker, now_ms)
    }

    #[cfg(test)]
    fn is_router(&self) -> bool;

    #[cfg(test)]
    fn debug_router_snapshot(&self, now_ms: f64) -> Option<OfflineRouterSnapshot>;
}

impl<Events: EngineEventBatch> AggregatedPlacement<Events, ()>
    for AggregatedRoundRobinPlacement<Events>
{
    #[cfg(test)]
    #[inline]
    fn is_router(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn debug_router_snapshot(&self, _now_ms: f64) -> Option<OfflineRouterSnapshot> {
        None
    }
}

pub(in crate::replay) type RoundRobinAggRuntime = AggRuntimeImpl<
    AggregatedRoundRobinPlacement<()>,
    NoEngineEvents,
    NoReplayMetadata,
    AdmissionQueue<NoReplayMetadata>,
>;
pub(in crate::replay) type ExternalAggRuntime = AggRuntimeImpl<
    ExternalPlacement<()>,
    NoEngineEvents,
    NoReplayMetadata,
    AdmissionQueue<NoReplayMetadata>,
>;

/// Open-admission controls used only by the polling adapter. Keeping these
/// outside the generic [`CoreAdmissionSource`] contract lets one-shot replay
/// and [`ReplayWorkSource`](crate::replay::offline::ReplayWorkSource) supply
/// their own admission implementations while all adapters share the runtime's
/// timestamp-stepping kernel.
pub(in crate::replay::offline) trait InteractiveAdmission {
    fn append_agentic_trace(
        &mut self,
        trace: AgenticTrace,
        release_at_ms: f64,
    ) -> anyhow::Result<()>;
    fn close(&mut self) -> anyhow::Result<()>;
    fn is_open(&self) -> bool;
    fn pending_requests(&self) -> usize;
}

impl<Metadata: ReplayAdmissionMetadata> InteractiveAdmission for AdmissionQueue<Metadata> {
    fn append_agentic_trace(
        &mut self,
        trace: AgenticTrace,
        release_at_ms: f64,
    ) -> anyhow::Result<()> {
        AdmissionQueue::append_agentic_trace(self, trace, release_at_ms)
    }

    fn close(&mut self) -> anyhow::Result<()> {
        AdmissionQueue::close(self)
    }

    fn is_open(&self) -> bool {
        AdmissionQueue::is_open(self)
    }

    fn pending_requests(&self) -> usize {
        AdmissionQueue::pending_requests(self)
    }
}

pub(in crate::replay) struct AggRuntimeImpl<
    PlacementPolicyImpl,
    Observation,
    Metadata,
    Admission = AdmissionQueue<Metadata>,
> where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
    Admission: CoreAdmissionSource<
            Request = ReplayRequestPayload,
            Metadata = Metadata,
            TerminalStatus = WorkloadTerminalStatus,
            CascadedTerminal = CascadedWorkloadTerminal,
        >,
{
    now_ms: f64,
    dp_size: u32,
    next_event_seq: u64,
    next_scaling_tick_ordinal: u64,
    admission: Admission,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: EngineComponent<Observation>,
    collector: TraceCollector,
    decode_gpus_per_worker: usize,
    events: BinaryHeap<SimulationEvent<Observation::Batch>>,
    placement: PlacementPolicyImpl,
    /// Placements released by completion feedback at `now_ms`. They are held
    /// until every completion at that timestamp has published terminal/DAG
    /// feedback and newly-ready arrivals have crossed admission.
    deferred_timestamp_placements: Vec<Placement>,
    progress: ReplayProgress,
    stats: AggRuntimeStats,
    /// Latest forward pass metric per worker/rank since the previous scaling tick.
    fpm_buffer: LatestFpmBuffer,
    /// Traffic statistics accumulated between scaling ticks.
    traffic: TrafficAccumulator,
    /// Optional cap on simulated wall-clock time. When set, `run()` exits
    /// gracefully once the next scheduled timestamp exceeds this cap, leaving
    /// any in-flight requests as incomplete in the report.
    max_sim_time_ms: Option<f64>,
    /// Optional scaling component. When set, `run()` seeds recurring `ScalingTick` events.
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    /// Whether to retain the latest FPM snapshot per worker/rank. Only the planner
    /// consumes them, so the plain `run()` path leaves this `false`.
    collect_fpm: bool,
    /// Allocated only for polling-based interactive replay. Ordinary one-shot
    /// replay retains its allocation profile when this is `None`.
    interactive: Option<InteractiveCapture>,
    /// Dense engine worker index to authored pool/worker identity. Pool
    /// membership is explicit and never inferred from numeric ID ranges.
    interactive_worker_targets: Vec<WorkerTarget>,
    /// Dense, append-only engine worker index to the collector's accounting
    /// arena. Entries outlive engine tombstones and are never reused.
    decode_worker_accounting_handles: Vec<DecodeWorkerAccountingHandle>,
    interactive_workers: Vec<ResolvedPoolWorker>,
    interactive_pool_ids: Vec<String>,
    stepping_started: bool,
    #[cfg(test)]
    worker_active_requests: Vec<Vec<Uuid>>,
}

impl
    AggRuntimeImpl<
        AggregatedRoundRobinPlacement<()>,
        NoEngineEvents,
        NoReplayMetadata,
        AdmissionQueue<NoReplayMetadata>,
    >
{
    pub(in crate::replay) fn new_round_robin(
        args: &MockEngineArgs,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: ReplayMode,
    ) -> anyhow::Result<Self> {
        Self::new_composed(
            args,
            AdmissionQueue::new_requests(pending, mode),
            num_workers,
            |args, topology| {
                Ok(AggregatedRoundRobinPlacement::with_taints(
                    args.dp_size,
                    topology,
                    &args.worker_taints,
                ))
            },
        )
    }

    pub(in crate::replay) fn new_round_robin_workload(
        args: &MockEngineArgs,
        driver: WorkloadDriver,
        num_workers: usize,
        mode: ReplayMode,
    ) -> anyhow::Result<Self> {
        Self::new_composed(
            args,
            AdmissionQueue::new_workload(driver, mode),
            num_workers,
            |args, topology| {
                Ok(AggregatedRoundRobinPlacement::with_taints(
                    args.dp_size,
                    topology,
                    &args.worker_taints,
                ))
            },
        )
    }
}

impl
    AggRuntimeImpl<
        ExternalPlacement<()>,
        NoEngineEvents,
        NoReplayMetadata,
        AdmissionQueue<NoReplayMetadata>,
    >
{
    pub(in crate::replay) fn new_external_workload(
        args: &MockEngineArgs,
        driver: WorkloadDriver,
        num_workers: usize,
    ) -> anyhow::Result<Self> {
        Self::new_composed(
            args,
            AdmissionQueue::new_workload(driver, ReplayMode::Trace),
            num_workers,
            |args, topology| Ok(ExternalPlacement::new(topology, &args.worker_taints)),
        )
    }

    pub(in crate::replay::offline) fn new_external_pooled_workload(
        driver: WorkloadDriver,
        topology: ResolvedPoolTopology,
    ) -> anyhow::Result<Self> {
        let args = topology
            .workers
            .first()
            .ok_or_else(|| anyhow::anyhow!("interactive replay topology has no workers"))?
            .engine_args
            .clone();
        let pool_routers = topology.pool_routers;
        Self::new_composed_heterogeneous(
            &args,
            AdmissionQueue::new_workload(driver, ReplayMode::Trace),
            topology.workers,
            move |_args, engine_topology, workers| {
                Ok(ExternalPlacement::new_pooled(
                    engine_topology,
                    workers.to_vec(),
                    pool_routers,
                ))
            },
        )
    }

    pub(in crate::replay::offline) fn preassign_interactive(
        &mut self,
        request_id: Uuid,
        target: WorkerTarget,
        required_taints: BTreeSet<String>,
    ) -> anyhow::Result<()> {
        self.placement
            .preassign(request_id, target, required_taints)
    }

    pub(in crate::replay::offline) fn assign_interactive(
        &mut self,
        request_id: Uuid,
        target: WorkerTarget,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.placement.is_pending(request_id),
            "interactive replay request {request_id} is not awaiting placement"
        );
        self.validate_interactive_assignment_boundary(request_id)?;
        let placement = self.placement.assign(request_id, &target)?;
        let observation = self.complete_interactive_assignment_boundary(request_id)?;
        let mut placement = self.resolve_placement(placement, Some(target))?;
        self.decorate_external_overlap(&mut placement, &observation)?;
        self.dispatch_resolved_placement(placement, Some(observation))?;
        // The controller action releases the barrier. Settle until either the
        // next external placement boundary or the same-time fixed point so the
        // next observation is both fresh and immediately available.
        self.drain_current_timestamp()?;
        Ok(())
    }

    pub(in crate::replay::offline) fn assign_pool_interactive(
        &mut self,
        request_id: Uuid,
        pool_id: &str,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.placement.is_pending(request_id),
            "interactive replay request {request_id} is not awaiting placement"
        );
        self.validate_interactive_assignment_boundary(request_id)?;
        let (placement, target) = self.placement.assign_pool(request_id, pool_id)?;
        // Pool-only placement first selects the worker through the pool's
        // internal router. Only then can the runtime inspect that selected
        // scheduler's committed KV state.
        let observation = self.complete_interactive_assignment_boundary(request_id)?;
        let mut placement = self.resolve_placement(placement, Some(target))?;
        self.decorate_external_overlap(&mut placement, &observation)?;
        self.dispatch_resolved_placement(placement, Some(observation))?;
        self.drain_current_timestamp()?;
        Ok(())
    }

    pub(in crate::replay::offline) fn pending_interactive_placements(
        &mut self,
    ) -> Vec<ReplayPendingPlacement> {
        let Some(request_id) = self.placement.pending_ids().next() else {
            return Vec::new();
        };
        let mut placements = self
            .interactive
            .as_ref()
            .map(|capture| capture.pending(std::iter::once(request_id)))
            .unwrap_or_default();
        if placements.is_empty() {
            return placements;
        }
        let placement_observation = self
            .interactive
            .as_ref()
            .and_then(|capture| capture.announced_placement(request_id))
            .map(Arc::clone)
            .or_else(|| {
                self.requests
                    .get(&request_id)
                    .and_then(AggRequestState::queued_request)
                    .map(|request| self.interactive_placement_candidates(request))
            })
            .unwrap_or_else(|| InteractivePlacementObservation::from_candidates(Vec::new()));
        if let Some(capture) = self.interactive.as_mut()
            && !capture.placement_is_announced(request_id)
        {
            capture.mark_placement_observed(request_id, Arc::clone(&placement_observation));
        }
        if let Some(placement) = placements.first_mut() {
            placement.eligible_pool_ids = placement_observation.eligible_pool_ids().to_vec();
            placement.candidates = placement_observation.candidates().to_vec();
        }
        placements
    }

    pub(in crate::replay::offline) fn cancel_interactive_placement(&mut self, request_id: Uuid) {
        self.placement.cancel(request_id);
        if let Some(capture) = self.interactive.as_mut() {
            capture.cancel_placement_observation(request_id);
        }
    }
}

impl<PlacementPolicyImpl, Observation, Metadata, Admission>
    AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata, Admission>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
    Admission: CoreAdmissionSource<
            Request = ReplayRequestPayload,
            Metadata = Metadata,
            TerminalStatus = WorkloadTerminalStatus,
            CascadedTerminal = CascadedWorkloadTerminal,
        >,
{
    pub(in crate::replay::offline) fn new_composed(
        args: &MockEngineArgs,
        admission: Admission,
        num_workers: usize,
        create_placement: impl FnOnce(
            &MockEngineArgs,
            Vec<WorkerTopology>,
        ) -> anyhow::Result<PlacementPolicyImpl>,
    ) -> anyhow::Result<Self> {
        let args = args.clone().normalized()?;
        anyhow::ensure!(
            args.worker_max_num_seqs.is_empty() || args.worker_max_num_seqs.len() == num_workers,
            "worker_max_num_seqs must be empty or contain exactly one entry per replay worker: got {} for {} workers",
            args.worker_max_num_seqs.len(),
            num_workers,
        );
        anyhow::ensure!(
            args.worker_taints.is_empty() || args.worker_taints.len() == num_workers,
            "worker_taints must be empty or contain exactly one entry per replay worker: got {} for {} workers",
            args.worker_taints.len(),
            num_workers,
        );
        for (worker_id, taints) in args.worker_taints.iter().enumerate() {
            for taint in taints {
                anyhow::ensure!(
                    !taint.is_empty() && taint.trim() == taint,
                    "replay worker {worker_id} has an empty or untrimmed taint {taint:?}",
                );
            }
        }
        let workers = (0..num_workers)
            .map(|worker_id| {
                let mut worker_args = args.clone();
                if let Some(&max_num_seqs) = args.worker_max_num_seqs.get(worker_id) {
                    worker_args.max_num_seqs = Some(max_num_seqs);
                }
                ResolvedPoolWorker {
                    target: WorkerTarget::default_pool(worker_id, 0),
                    engine_args: worker_args,
                    tags: BTreeSet::new(),
                    taints: args
                        .worker_taints
                        .get(worker_id)
                        .map(|taints| taints.iter().cloned().collect())
                        .unwrap_or_default(),
                    capabilities: BTreeSet::new(),
                    active: true,
                    draining: false,
                }
            })
            .collect();
        Self::new_composed_heterogeneous(&args, admission, workers, |args, topology, _workers| {
            create_placement(args, topology)
        })
    }

    fn new_composed_heterogeneous(
        args: &MockEngineArgs,
        admission: Admission,
        workers: Vec<ResolvedPoolWorker>,
        create_placement: impl FnOnce(
            &MockEngineArgs,
            Vec<WorkerTopology>,
            &[ResolvedPoolWorker],
        ) -> anyhow::Result<PlacementPolicyImpl>,
    ) -> anyhow::Result<Self> {
        let args = args.clone().normalized()?;
        let num_workers = workers.len();
        anyhow::ensure!(
            num_workers > 0,
            "offline replay requires at least one worker"
        );
        let gpu_counts = workers
            .iter()
            .map(|worker| worker.engine_args.aic_gpus_per_worker())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            gpu_counts.len() == 1,
            "heterogeneous GPUs-per-worker accounting is unsupported; all static replay workers must use the same model parallelism, got {gpu_counts:?}"
        );
        let gpus_per_worker = *gpu_counts
            .first()
            .expect("validated topology contains at least one worker");
        let progress = ReplayProgress::new(
            CoreAdmissionSource::total_requests(&admission),
            "offline replay",
        );
        let worker_args = workers
            .iter()
            .map(|worker| worker.engine_args.clone())
            .collect();
        let mut engine = EngineComponent::<Observation>::new_ranked_heterogeneous(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            args.clone(),
            worker_args,
        );
        engine.set_scaling_args(args.clone(), Observation::CAPTURE_RAW);
        for (engine_worker_id, worker) in workers.iter().enumerate() {
            if worker.draining {
                engine.mark_for_removal(engine_worker_id);
            } else if !worker.active {
                engine.mark_static_inactive(engine_worker_id);
            }
        }
        let placement = create_placement(&args, engine.all_topology(), &workers)?;
        let interactive_worker_targets = workers
            .iter()
            .map(|worker| worker.target.clone())
            .collect::<Vec<_>>();
        let interactive_pool_ids = workers
            .iter()
            .map(|worker| worker.target.pool_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        // Aggregated replay has one decode accounting role across its static
        // pools; record the validated uniform GPUs/worker for global GPU-hours.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(0, gpus_per_worker);
        let decode_worker_accounting_handles = workers
            .iter()
            .map(|worker| {
                collector.register_decode_worker(
                    worker.target.pool_id.clone(),
                    worker.target.worker_id,
                    worker.target.dp_rank,
                    worker.lifecycle_status(),
                    gpus_per_worker,
                    0.0,
                )
            })
            .collect();

        Ok(Self {
            now_ms: 0.0,
            dp_size: args.dp_size.max(1),
            next_event_seq: 0,
            next_scaling_tick_ordinal: 0,
            admission,
            requests: FxHashMap::default(),
            engine,
            collector,
            decode_gpus_per_worker: gpus_per_worker,
            events: BinaryHeap::new(),
            placement,
            deferred_timestamp_placements: Vec::new(),
            progress,
            #[cfg(test)]
            stats: AggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: AggRuntimeStats,
            fpm_buffer: LatestFpmBuffer::default(),
            traffic: TrafficAccumulator::new(),
            max_sim_time_ms: None,
            scaling_policy: None,
            collect_fpm: false,
            interactive: None,
            interactive_worker_targets,
            decode_worker_accounting_handles,
            interactive_workers: workers,
            interactive_pool_ids,
            stepping_started: false,
            #[cfg(test)]
            worker_active_requests: vec![
                Vec::new();
                num_workers.saturating_mul(args.dp_size.max(1) as usize)
            ],
        })
    }

    /// Toggle per-request record capture on the underlying collector. When
    /// `true`, the final `TraceSimulationReport` returned from `run()` will
    /// have `per_request` populated. Default `false` (cheap).
    pub(in crate::replay) fn with_per_request_records(mut self, capture: bool) -> Self {
        self.collector.set_capture_per_request(capture);
        self
    }

    pub(in crate::replay::offline) fn enable_interactive_capture(
        &mut self,
        external_placement: bool,
        session_affinity: bool,
    ) {
        self.collector.set_capture_per_request(true);
        self.progress = ReplayProgress::disabled();
        self.interactive = Some(InteractiveCapture::new(
            external_placement,
            session_affinity,
            self.interactive_pool_ids.clone(),
        ));
    }

    pub(in crate::replay::offline) fn register_interactive_identity(
        &mut self,
        identity: InteractiveRequestIdentity,
    ) -> anyhow::Result<()> {
        self.interactive
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("offline replay interactive capture is disabled"))?
            .register(identity)
    }

    pub(in crate::replay::offline) fn append_interactive_agentic_trace(
        &mut self,
        trace: AgenticTrace,
        release_at_ms: f64,
    ) -> anyhow::Result<()>
    where
        Admission: InteractiveAdmission,
    {
        if release_at_ms < self.now_ms {
            bail!(
                "interactive replay cannot enqueue release time {release_at_ms} ms before current time {} ms",
                self.now_ms
            );
        }
        self.admission.append_agentic_trace(trace, release_at_ms)
    }

    pub(in crate::replay::offline) fn unregister_interactive_identity(&mut self, uuid: Uuid) {
        if let Some(capture) = self.interactive.as_mut() {
            capture.unregister(uuid);
        }
    }

    /// Cap the simulated wall-clock duration. After construction, call this to
    /// have `run()` stop gracefully once the simulated clock would exceed
    /// `ms`. Pass `None` to run to natural completion (the default).
    ///
    /// max_sim_time_ms is a **soft cap** on the scheduling loop, not a hard truncation
    /// of recorded work. When the next scheduled simulated timestamp would
    /// exceed the cap, the loop exits, but worker passes already in flight
    /// complete normally — even if their token timestamps land past `ms`.
    /// Requests that hadn't received their first token before the cap fired
    /// stay in the report as incomplete (`first_token_ms = None`,
    /// `e2e_latency_ms = None`). `report.duration_ms` may exceed `ms` by up
    /// to one in-flight pass's duration. Enforcing a precise cap would
    /// require plumbing a deadline into the worker / engine core; not worth
    /// it for the calibration use case this exists to serve.
    pub(in crate::replay) fn with_max_sim_time_ms(mut self, ms: Option<f64>) -> Self {
        self.max_sim_time_ms = ms;
        self
    }

    /// Attach a scaling policy and enable tick-scoped FPM collection.
    pub(in crate::replay) fn with_scaling_policy(
        mut self,
        policy: Box<dyn ReplayScalingPolicy>,
    ) -> Self {
        self.collect_fpm = true;
        for worker_id in self.engine.active_group_ids() {
            self.fpm_buffer
                .activate_worker(worker_id, self.dp_size, self.now_ms);
        }
        self.scaling_policy = Some(policy);
        self
    }

    #[cfg(test)]
    fn with_fpm_capture(mut self) -> Self {
        self.collect_fpm = true;
        self
    }

    /// Count all requests currently consuming cluster capacity, including router-queued ones.
    fn cluster_in_flight(&self) -> usize {
        // A timestamp-wide terminal phase can release router-queued work into
        // placements that intentionally remain undispatched until every
        // completion and admission release at this timestamp has settled.
        // Those requests still consume the closed-loop concurrency budget;
        // omitting them here creates a transient hole that can over-admit a
        // new request before the deferred placements are dispatched.
        self.engine.in_flight()
            + self.placement.pending_count()
            + self.deferred_timestamp_placements.len()
    }

    /// Track the peak cluster occupancy seen during the replay.
    fn record_in_flight_peak(&mut self) {
        #[cfg(test)]
        {
            self.stats.max_in_flight_seen =
                self.stats.max_in_flight_seen.max(self.cluster_in_flight());
        }
    }

    /// Track the maximum number of requests parked in the offline router.
    fn record_router_pending(&mut self) {
        #[cfg(test)]
        {
            self.stats.max_router_pending_count = self
                .stats
                .max_router_pending_count
                .max(self.placement.pending_count());
        }
    }

    /// Record which worker accepted a request and refresh in-flight stats.
    fn record_dispatch(&mut self, _uuid: Uuid, _worker_idx: usize) {
        #[cfg(test)]
        {
            self.stats.dispatch_history.push(_worker_idx);
            self.stats.dispatch_order.push(_uuid);
            self.stats
                .assigned_worker_by_uuid
                .insert(_uuid, _worker_idx);
        }
        self.record_in_flight_peak();
    }

    /// Preserve the live `(worker_id, dp_rank)` identity when forwarding a
    /// rank-local scheduler snapshot to the scaling policy.
    fn record_fpm(
        &mut self,
        rank_id: usize,
        mut snapshot: ForwardPassSnapshot,
    ) -> anyhow::Result<()> {
        let (worker_id, dp_rank) = self.engine.rank_identity(rank_id).ok_or_else(|| {
            anyhow::anyhow!("offline replay FPM references unknown rank scheduler {rank_id}")
        })?;
        snapshot.worker_id = worker_id.to_string();
        snapshot.dp_rank = dp_rank;
        self.fpm_buffer.insert(worker_id, snapshot, self.now_ms);
        Ok(())
    }

    fn resolve_placement(
        &self,
        placement: Placement,
        authored_target: Option<WorkerTarget>,
    ) -> anyhow::Result<ResolvedPlacement> {
        let (logical_worker_id, dp_rank) = self
            .engine
            .rank_identity(placement.scheduler_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "offline replay placement references unknown scheduler {}",
                    placement.scheduler_id
                )
            })?;
        let authored_target = if let Some(target) = authored_target {
            let expected = self
                .interactive_worker_targets
                .get(logical_worker_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "interactive replay scheduler {} has no authored worker target",
                        placement.scheduler_id
                    )
                })?;
            anyhow::ensure!(
                target.pool_id == expected.pool_id
                    && target.worker_id == expected.worker_id
                    && target.dp_rank == dp_rank as usize,
                "interactive replay scheduler {} resolved to pool {:?} worker {} rank {}, not validated target {:?}",
                placement.scheduler_id,
                expected.pool_id,
                expected.worker_id,
                dp_rank,
                target
            );
            target
        } else {
            let mut target = self.accounting_target(logical_worker_id);
            target.dp_rank = dp_rank as usize;
            target
        };
        Ok(ResolvedPlacement {
            placement,
            logical_worker_id,
            dp_rank,
            authored_target,
        })
    }

    /// Deliver a request to a worker and update the runtime's bookkeeping for that assignment.
    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        resolved: ResolvedPlacement,
        route_observation: Option<Arc<InteractivePlacementObservation>>,
    ) -> anyhow::Result<()> {
        let uuid = resolved.placement.request_id;
        let scheduler_id = resolved.placement.scheduler_id;
        if let Some(capture) = self.interactive.as_ref() {
            capture.identity(uuid).ok_or_else(|| {
                anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
            })?;
            anyhow::ensure!(
                route_observation.is_some(),
                "interactive replay dispatch for request {uuid} has no frozen placement observation"
            );
        }
        self.engine.dispatch(scheduler_id, request)?;
        if let Some(capture) = self.interactive.as_mut() {
            capture.set_worker(uuid, &resolved.authored_target)?;
            // `route_observation` was captured before physical engine dispatch
            // and is the observation associated with this effective route.
            // Do not reconstruct it from the post-route snapshot.
            capture.emit_routed(
                uuid,
                self.now_ms,
                route_observation.expect("interactive route observation was validated"),
            )?;
            capture.emit_queued(uuid, self.now_ms)?;
        }
        self.record_dispatch(uuid, scheduler_id);
        // Aggregated replay has one execution stage even when it has multiple
        // static pools. Treat the assignment as decode_worker_idx so records
        // carry the serving scheduler; prefill_worker_idx stays None to signal
        // that there is no separate prefill stage.
        self.collector.on_decode_assigned(uuid, scheduler_id);
        self.collector.on_static_pool_assigned(
            uuid,
            resolved.authored_target.pool_id,
            resolved.authored_target.worker_id,
            resolved.authored_target.dp_rank,
        );
        #[cfg(test)]
        self.worker_active_requests[scheduler_id].push(uuid);
        Ok(())
    }

    fn record_placement(&mut self, placement: Placement) {
        if let Some(sample) = placement.planner_cache_sample {
            self.traffic
                .on_admission(sample.overlap_blocks, sample.isl_blocks);
            #[cfg(test)]
            self.stats.overlap_history.push(sample.overlap_blocks);
        }
    }

    fn interactive_worker_snapshots(&self) -> Vec<ReplayWorkerSnapshot> {
        let mut snapshots = self.engine.interactive_snapshots();
        for snapshot in &mut snapshots {
            let engine_worker_id = snapshot.worker_id;
            if let Some(worker) = self.interactive_workers.get(engine_worker_id) {
                if snapshot.pool_id != worker.target.pool_id {
                    snapshot.pool_id.clone_from(&worker.target.pool_id);
                }
                snapshot.worker_id = worker.target.worker_id;
                snapshot.tags = worker.tags.iter().cloned().collect();
                snapshot.taints = worker.taints.iter().cloned().collect();
                snapshot.capabilities = worker.capabilities.iter().cloned().collect();
            }
        }
        snapshots
    }

    fn interactive_placement_candidates(
        &self,
        request: &ReplayRequestPayload,
    ) -> Arc<InteractivePlacementObservation> {
        let tokens = request.prompt_tokens();
        let constraints = request.routing_constraints();
        let candidates = self
            .engine
            .interactive_snapshots()
            .into_iter()
            .filter_map(|snapshot| {
                let engine_worker_id = snapshot.worker_id;
                let worker = self.interactive_workers.get(engine_worker_id)?;
                let scheduler_id = self
                    .engine
                    .scheduler_id(engine_worker_id, snapshot.dp_rank)?;
                let constraint_reason = if snapshot.draining || worker.draining {
                    Some("worker is draining".to_string())
                } else if !snapshot.active || !worker.active {
                    Some("worker is inactive".to_string())
                } else {
                    let missing = constraints
                        .required_taints
                        .iter()
                        .filter(|required| !worker.taints.contains(*required))
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    (!missing.is_empty()).then(|| format!("missing required taints {missing:?}"))
                };
                // Reuse the engine snapshot's otherwise-discarded default-pool
                // allocation for the authored pool carried by the candidate.
                let mut pool_id = snapshot.pool_id;
                if pool_id != worker.target.pool_id {
                    pool_id.clone_from(&worker.target.pool_id);
                }
                let target = WorkerTarget::new(pool_id, worker.target.worker_id, snapshot.dp_rank);
                Some(ReplayPlacementCandidate {
                    target,
                    active: snapshot.active,
                    draining: snapshot.draining,
                    eligible: constraint_reason.is_none(),
                    constraint_reason,
                    in_flight_requests: snapshot.in_flight_requests,
                    queued_requests: snapshot.queued_requests,
                    running_requests: snapshot.running_requests,
                    queued_tokens: snapshot.queued_tokens,
                    running_tokens: snapshot.running_tokens,
                    max_num_seqs: snapshot.max_num_seqs,
                    preemption_count: snapshot.preemption_count,
                    kv_prefix_overlap_tokens: self
                        .engine
                        .native_prefix_overlap_tokens(scheduler_id, &tokens),
                    kv_capacity_blocks: snapshot.kv_capacity_blocks,
                    kv_occupied_blocks: snapshot.kv_occupied_blocks,
                    kv_free_blocks: snapshot.kv_free_blocks,
                    tags: worker.tags.iter().cloned().collect(),
                    taints: worker.taints.iter().cloned().collect(),
                    capabilities: worker.capabilities.iter().cloned().collect(),
                })
            })
            .collect();
        InteractivePlacementObservation::from_candidates(candidates)
    }

    /// Attach the selected worker's scheduler-safe, committed KV overlap from
    /// the exact policy observation that authorized this external placement.
    /// The controller barrier freezes scheduler state until assignment, so a
    /// second native prefix query would be redundant and could not be fresher.
    fn decorate_external_overlap(
        &self,
        placement: &mut ResolvedPlacement,
        observation: &InteractivePlacementObservation,
    ) -> anyhow::Result<()> {
        let candidate = observation
            .candidates()
            .iter()
            .find(|candidate| candidate.target == placement.authored_target)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "interactive replay selected target {:?} was absent from the announced observation for request {}",
                    placement.authored_target,
                    placement.placement.request_id
                )
            })?;
        placement.placement.reported_overlap_tokens = candidate.kv_prefix_overlap_tokens;
        Ok(())
    }

    fn validate_interactive_assignment_boundary(&self, request_id: Uuid) -> anyhow::Result<()> {
        self.interactive
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("offline replay interactive capture is disabled"))?
            .validate_placement_assignment(request_id)
    }

    fn complete_interactive_assignment_boundary(
        &mut self,
        request_id: Uuid,
    ) -> anyhow::Result<Arc<InteractivePlacementObservation>> {
        self.interactive
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("offline replay interactive capture is disabled"))?
            .complete_placement_assignment(request_id)
    }

    /// Materialize one policy-released admission into a concrete worker
    /// dispatch. External controllers pass the exact observation that
    /// authorized the assignment; native policies construct their route view
    /// at their existing pre-dispatch boundary.
    fn dispatch_placement(
        &mut self,
        placement: Placement,
        route_observation: Option<Arc<InteractivePlacementObservation>>,
    ) -> anyhow::Result<()> {
        let resolved = self.resolve_placement(placement, None)?;
        let uuid = resolved.placement.request_id;
        let route_observation = route_observation.or_else(|| {
            self.interactive.as_ref()?;
            self.requests
                .get(&uuid)
                .and_then(AggRequestState::queued_request)
                .map(|request| self.interactive_placement_candidates(request))
        });
        self.dispatch_resolved_placement(resolved, route_observation)
    }

    fn dispatch_resolved_placement(
        &mut self,
        resolved: ResolvedPlacement,
        route_observation: Option<Arc<InteractivePlacementObservation>>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.interactive.is_none() || route_observation.is_some(),
            "interactive replay request {} has no frozen placement observation",
            resolved.placement.request_id
        );
        self.record_placement(resolved.placement);
        let uuid = resolved.placement.request_id;
        self.collector.on_route_released(
            uuid,
            ReplayRequestPool::Agg,
            self.now_ms,
            resolved.logical_worker_id,
            resolved.placement.scheduler_id,
            resolved.dp_rank,
            resolved.placement.reported_overlap_tokens,
        );
        self.collector.on_route_static_target(
            uuid,
            ReplayRequestPool::Agg,
            resolved.authored_target.pool_id.clone(),
            resolved.authored_target.worker_id,
            resolved.authored_target.dp_rank,
        );
        let request = self
            .requests
            .get_mut(&uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing queued request state for {uuid}")
            })?
            .take_queued_request(uuid)?;
        self.dispatch_to_worker(request, resolved, route_observation)
    }

    /// Materialize policy-released admissions into concrete worker dispatches.
    fn dispatch_placements(&mut self, placements: Vec<Placement>) -> anyhow::Result<()> {
        for placement in placements {
            self.dispatch_placement(placement, None)?;
        }
        Ok(())
    }

    /// Admit one external request into the collector, optional router, and worker pool.
    fn assign_request(
        &mut self,
        mut request: ReplayRequestPayload,
        arrival_time_ms: f64,
        metadata: Metadata,
        session_id: Option<String>,
    ) -> anyhow::Result<Uuid> {
        let uuid = request.metadata().uuid.unwrap_or_else(Uuid::new_v4);
        let input_length = request.input_length();
        let output_length = request.metadata().effective_max_output_tokens();
        request.metadata_mut().uuid = Some(uuid);
        // The source is authoritative for logical arrival time. This also removes
        // the runtime's dependency on the concrete AdmissionQueue replay mode.
        request.metadata_mut().arrival_timestamp_ms = Some(arrival_time_ms);

        self.collector
            .on_arrival(uuid, arrival_time_ms, input_length, output_length);
        if let Some(identity) = self
            .interactive
            .as_ref()
            .and_then(|capture| capture.identity(uuid))
        {
            self.collector.on_attempt_group_identity(
                uuid,
                identity.attempt_id.clone(),
                identity.group_id.clone(),
            );
        }
        if let Some(capture) = self.interactive.as_mut() {
            capture.mark_ready(uuid, self.now_ms)?;
        }
        self.traffic.on_arrival();

        let placement_session_id = if self
            .interactive
            .as_ref()
            .is_some_and(|capture| !capture.uses_session_affinity())
        {
            None
        } else {
            session_id
        };
        let uses_external_placement = self
            .interactive
            .as_ref()
            .is_some_and(InteractiveCapture::uses_external_placement);
        // Native policies capture their shared, scheduler-safe candidate view
        // before selection. External policy observations are constructed only
        // at the later, causally fresh controller boundary; building one here
        // would both be stale and discarded for an ordinary queued request.
        let native_policy_observation = self
            .interactive
            .as_ref()
            .filter(|_| !uses_external_placement)
            .map(|_| self.interactive_placement_candidates(&request));
        let effects =
            self.placement
                .place(&request, metadata, placement_session_id, self.now_ms)?;
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                if placement.request_id != uuid {
                    bail!(
                        "offline placement returned request {} while placing {uuid}",
                        placement.request_id
                    );
                }
                let mut resolved = self.resolve_placement(placement, None)?;
                let policy_observation = if uses_external_placement {
                    // Authored preassignment routes immediately and therefore
                    // has no announced controller boundary. Build its one
                    // pre-dispatch observation after policy resolution (which
                    // cannot mutate engine state), then reuse it for overlap
                    // and the Routed event.
                    let observation = self.interactive_placement_candidates(&request);
                    self.decorate_external_overlap(&mut resolved, &observation)?;
                    Some(observation)
                } else {
                    native_policy_observation
                };
                anyhow::ensure!(
                    self.interactive.is_none() || policy_observation.is_some(),
                    "interactive replay request {uuid} has no frozen placement observation"
                );
                self.record_placement(resolved.placement);
                self.collector.on_route_immediate(
                    uuid,
                    ReplayRequestPool::Agg,
                    self.now_ms,
                    resolved.logical_worker_id,
                    resolved.placement.scheduler_id,
                    resolved.dp_rank,
                    resolved.placement.reported_overlap_tokens,
                );
                self.collector.on_route_static_target(
                    uuid,
                    ReplayRequestPool::Agg,
                    resolved.authored_target.pool_id.clone(),
                    resolved.authored_target.worker_id,
                    resolved.authored_target.dp_rank,
                );
                self.requests.insert(
                    uuid,
                    AggRequestState::new_running(input_length, output_length),
                );
                self.dispatch_to_worker(
                    request.into_direct_request(),
                    resolved,
                    policy_observation,
                )?;
            }
            PlacementDecision::Queued => {
                self.collector
                    .on_route_queued(uuid, ReplayRequestPool::Agg, self.now_ms);
                self.requests
                    .insert(uuid, AggRequestState::new_queued(request));
            }
        }
        self.record_router_pending();
        self.dispatch_placements(effects.released)?;
        self.record_in_flight_peak();
        Ok(uuid)
    }

    /// Return true once no request work remains. Lingering `WorkerReady`/`ScalingTick`
    /// events carry no work and do not
    /// keep the run alive — otherwise a recurring tick would never let `run()` exit.
    fn is_done(&self) -> bool {
        self.only_idle_events_remain()
            && self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.engine.is_drained()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// or `ScalingTick` events remain in the queue. Lingering startup events for
    /// workers that will never receive requests should not block completion.
    fn is_workload_done(&self) -> bool {
        self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.engine.is_drained()
            && self.only_idle_events_remain()
    }

    /// True if the event heap is empty or contains only "idle" events that carry no
    /// pending request work: `WorkerReady` (a worker still starting up) or
    /// `ScalingTick` (a re-armed scaling heartbeat).
    fn only_idle_events_remain(&self) -> bool {
        use super::events::SimulationEventKind;
        self.events.iter().all(|e| {
            matches!(
                e.kind,
                SimulationEventKind::WorkerReady { .. } | SimulationEventKind::ScalingTick
            )
        })
    }

    /// Pick the next logical timestamp from either arrivals or scheduled worker completions.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let next = choose_next_timestamp(
            CoreAdmissionSource::next_internal_event_ms(&mut self.admission),
            next_event_ms,
        );
        #[cfg(feature = "kvbm-offload")]
        {
            choose_next_timestamp(next, self.engine.earliest_offload_deadline())
        }
        #[cfg(not(feature = "kvbm-offload"))]
        {
            next
        }
    }

    /// Apply router-visible KV events at the phase chosen by the scheduler core.
    fn apply_engine_observations(
        &mut self,
        events: Observation::Batch,
        boundary: KvIngestBoundary,
    ) -> anyhow::Result<()> {
        Observation::record_ingestion(&events, WorkerPool::Agg, boundary, self.now_ms)?;
        let placements = self.placement.observe(events, self.now_ms)?;
        self.dispatch_placements(placements)
    }

    #[cfg(feature = "kvbm-offload")]
    fn tick_offload_engines(&mut self) -> anyhow::Result<bool> {
        let super::components::ObservedOffloadEffects {
            engine_events,
            lifecycle_events,
            progress,
        } = self.engine.tick_offload_engines(self.now_ms);
        if !lifecycle_events.is_empty() {
            bail!(
                "aggregated replay received {} handoff lifecycle events from an offload tick",
                lifecycle_events.len()
            );
        }
        self.apply_engine_observations(engine_events, KvIngestBoundary::OffloadTick)?;
        Ok(progress.made_progress)
    }

    fn process_output_token(&mut self, signal: &OutputSignal) -> anyhow::Result<()> {
        if let Some(token_id) = signal.token_id {
            CoreAdmissionSource::on_output_token(&mut self.admission, signal.uuid, token_id)?;
            if !signal.rejected
                && let Some(capture) = self.interactive.as_mut()
            {
                capture.on_output_token(signal.uuid, self.now_ms)?;
            }
        }
        Ok(())
    }

    /// Consume one output lifecycle signal after every token carried by this
    /// timestamp's completion batch has been published in engine order.
    fn process_output_signal(&mut self, signal: &OutputSignal) -> anyhow::Result<Vec<Placement>> {
        if signal.completed {
            let workload_status = if signal.rejected {
                WorkloadTerminalStatus::Rejected
            } else {
                WorkloadTerminalStatus::Completed
            };
            let status: ReplayTerminalStatus = workload_status.into();
            if !self.requests.contains_key(&signal.uuid) {
                bail!("offline replay missing request state for {}", signal.uuid);
            }

            // Commit all fallible terminal transitions before publishing the
            // externally visible Terminal event. The event itself is emitted
            // before any placement released by this terminal, preserving the
            // decision-time ordering required by interactive controllers.
            self.placement
                .request_terminal_feedback(signal.uuid, self.now_ms)?;
            let cascaded = CoreAdmissionSource::on_terminal(
                &mut self.admission,
                signal.uuid,
                self.now_ms,
                workload_status,
            )?;
            let removed_state = self
                .requests
                .remove(&signal.uuid)
                .expect("request state was checked before terminal transitions");

            self.collector.on_terminal(signal.uuid, self.now_ms, status);
            #[cfg(test)]
            if self.placement.is_router() {
                self.stats.router_freed_count += 1;
            }
            self.record_router_pending();
            #[cfg(test)]
            self.remove_active_request(signal.uuid);
            // Rejected requests never ran: keep them out of completed-request
            // shape and latency samples. Their offered demand was already
            // recorded at arrival, matching requests_started_total.
            if !signal.rejected {
                let latencies = self.collector.request_latencies(signal.uuid);
                let actual_output_tokens = self
                    .collector
                    .actual_output_length(signal.uuid)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay missing collector state for {}",
                            signal.uuid
                        )
                    })?;
                debug_assert!(actual_output_tokens <= removed_state.output_tokens);
                self.traffic.on_completion(
                    removed_state.input_tokens,
                    actual_output_tokens,
                    latencies,
                );
            }
            self.progress.inc_completed();
            if let Some(capture) = self.interactive.as_mut() {
                capture.emit_terminal(signal.uuid, self.now_ms, status)?;
            }
            self.record_cascaded_terminals(cascaded)?;
            return Ok(Vec::new());
        }

        let already_marked = self
            .requests
            .get(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed;
        if already_marked {
            return Ok(Vec::new());
        }

        self.requests
            .get_mut(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed = true;
        let placements = self.placement.prefill_completed(signal.uuid, self.now_ms)?;
        #[cfg(test)]
        if self.placement.is_router() {
            self.stats.prefill_marked_count += 1;
        }
        self.record_router_pending();
        Ok(placements)
    }

    /// Publish dependency-cascade terminals for authored requests that never
    /// entered placement or the engine. Collision checks ensure a live request
    /// cannot be silently overwritten or terminated twice.
    fn record_cascaded_terminals(
        &mut self,
        cascaded: Vec<CascadedWorkloadTerminal>,
    ) -> anyhow::Result<()> {
        for terminal in cascaded {
            if self.requests.contains_key(&terminal.request_uuid) {
                bail!(
                    "cascaded workload terminal {} collides with an engine-visible request",
                    terminal.request_uuid
                );
            }
            if self.placement.cancel_pending(terminal.request_uuid)
                && let Some(capture) = self.interactive.as_mut()
            {
                capture.cancel_placement_observation(terminal.request_uuid);
            }
            self.collector
                .on_cascaded_terminal(&terminal, self.now_ms)?;
            if let Some(identity) = self
                .interactive
                .as_ref()
                .and_then(|capture| capture.identity(terminal.request_uuid))
            {
                self.collector.on_attempt_group_identity(
                    terminal.request_uuid,
                    identity.attempt_id.clone(),
                    identity.group_id.clone(),
                );
            }
            self.progress.inc_completed();
            if let Some(capture) = self.interactive.as_mut() {
                capture.emit_terminal(
                    terminal.request_uuid,
                    self.now_ms,
                    terminal.status.into(),
                )?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    /// Remove a request from the test-only active-request tracking for its worker.
    fn remove_active_request(&mut self, uuid: Uuid) {
        for active_requests in &mut self.worker_active_requests {
            let Some(position) = active_requests
                .iter()
                .position(|candidate| *candidate == uuid)
            else {
                continue;
            };
            active_requests.remove(position);
            return;
        }
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let Some(first) = pop_ready_worker_completions(&mut self.events, self.now_ms) else {
            return Ok(false);
        };
        let Some(second) = pop_ready_worker_completions(&mut self.events, self.now_ms) else {
            match first {
                ReadyWorkerCompletions::Single(payload) => {
                    self.settle_worker_completion(payload)?;
                }
                ReadyWorkerCompletions::Batch(payloads) => {
                    self.settle_worker_completion_batch(payloads)?;
                }
            }
            return Ok(true);
        };

        let mut payloads = SmallVec::<[WorkerCompletionPayload<Observation::Batch>; 2]>::new();
        for completions in [first, second] {
            match completions {
                ReadyWorkerCompletions::Single(payload) => payloads.push(payload),
                ReadyWorkerCompletions::Batch(batch) => payloads.extend(batch),
            }
        }
        while let Some(completions) = pop_ready_worker_completions(&mut self.events, self.now_ms) {
            match completions {
                ReadyWorkerCompletions::Single(payload) => payloads.push(payload),
                ReadyWorkerCompletions::Batch(batch) => payloads.extend(batch),
            }
        }
        self.settle_worker_completion_batch(payloads)?;
        Ok(true)
    }

    /// Settle the common singleton completion without staging it or repeatedly
    /// walking its output signals. Multi-signal payloads retain the timestamp-wide
    /// phase ordering in [`Self::settle_committed_worker_completion_batch`].
    #[inline]
    fn settle_worker_completion(
        &mut self,
        payload: WorkerCompletionPayload<Observation::Batch>,
    ) -> anyhow::Result<()> {
        let payload = self.commit_worker_completion(payload)?;
        if payload.output_signals.len() > 1 {
            let mut settled = SmallVec::new();
            settled.push(payload);
            return self.settle_committed_worker_completion_batch(settled);
        }

        self.traffic.on_accept_length_sample(
            payload.accept_length_output_tokens,
            payload.accept_length_decode_forwards,
        );
        let mut placements = if let Some(signal) = payload.output_signals.first() {
            self.process_output_token(signal)?;
            let mut placements = self.process_output_signal(signal)?;
            if signal.completed {
                placements.extend(self.placement.settle_terminal_feedback(self.now_ms)?);
            }
            placements
        } else {
            Vec::new()
        };
        Observation::record_ingestion(
            &payload.engine_events,
            WorkerPool::Agg,
            KvIngestBoundary::PassEnd,
            self.now_ms,
        )?;
        placements.extend(self.placement.observe(payload.engine_events, self.now_ms)?);
        self.deferred_timestamp_placements.extend(placements);
        Ok(())
    }

    /// Commit engine ownership for one completion before any lifecycle or
    /// observation is published.
    #[inline]
    fn commit_worker_completion(
        &mut self,
        payload: WorkerCompletionPayload<Observation::Batch>,
    ) -> anyhow::Result<WorkerCompletionPayload<Observation::Batch>> {
        debug_assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
        let mut payload = self.engine.on_scheduled_completion(payload)?;
        if self.collect_fpm
            && let Some(fpm) = payload.fpm.take()
        {
            self.record_fpm(payload.worker_idx, fpm)?;
        }
        Ok(payload)
    }

    /// Settle one timestamp-wide completion phase. Engine ownership changes
    /// for every sibling worker commit first, then terminal/DAG feedback, then
    /// router observations. No placement is dispatched from inside this phase.
    fn settle_worker_completion_batch(
        &mut self,
        payloads: impl IntoIterator<Item = WorkerCompletionPayload<Observation::Batch>>,
    ) -> anyhow::Result<()> {
        let mut settled = SmallVec::<[WorkerCompletionPayload<Observation::Batch>; 2]>::new();
        for payload in payloads {
            settled.push(self.commit_worker_completion(payload)?);
        }

        self.settle_committed_worker_completion_batch(settled)
    }

    /// Publish an already-committed timestamp-wide completion phase.
    fn settle_committed_worker_completion_batch(
        &mut self,
        settled: SmallVec<[WorkerCompletionPayload<Observation::Batch>; 2]>,
    ) -> anyhow::Result<()> {
        let mut placements = Vec::new();
        for payload in &settled {
            self.traffic.on_accept_length_sample(
                payload.accept_length_output_tokens,
                payload.accept_length_decode_forwards,
            );
        }
        // Publish every token first. A speculative/MTP pass may contain both a
        // non-terminal progress signal and the terminal signal for one UUID,
        // so apply the non-terminal state transition while its request state
        // still exists. Placements returned here remain deferred: no placement
        // event or dispatch is published until every terminal ownership/DAG
        // transition at t has completed. Order within each phase remains
        // worker/event stable.
        for signal in settled
            .iter()
            .flat_map(|payload| payload.output_signals.iter())
        {
            self.process_output_token(signal)?;
        }
        for progress in settled
            .iter()
            .flat_map(|payload| payload.output_signals.iter())
            .filter(|signal| !signal.completed)
        {
            placements.extend(self.process_output_signal(progress)?);
        }
        let mut had_terminal_feedback = false;
        for terminal in settled
            .iter()
            .flat_map(|payload| payload.output_signals.iter())
            .filter(|signal| signal.completed)
        {
            placements.extend(self.process_output_signal(terminal)?);
            had_terminal_feedback = true;
        }
        if had_terminal_feedback {
            placements.extend(self.placement.settle_terminal_feedback(self.now_ms)?);
        }
        for payload in settled {
            Observation::record_ingestion(
                &payload.engine_events,
                WorkerPool::Agg,
                KvIngestBoundary::PassEnd,
                self.now_ms,
            )?;
            placements.extend(self.placement.observe(payload.engine_events, self.now_ms)?);
        }
        self.deferred_timestamp_placements.extend(placements);
        Ok(())
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> anyhow::Result<bool> {
        let mut released_any = false;
        let cluster_in_flight = self.cluster_in_flight();
        let limit = if self
            .interactive
            .as_ref()
            .is_some_and(InteractiveCapture::uses_external_placement)
        {
            1
        } else {
            usize::MAX
        };
        for ready in CoreAdmissionSource::drain_ready_up_to(
            &mut self.admission,
            self.now_ms,
            cluster_in_flight,
            limit,
        )? {
            let ReadyArrival {
                request,
                arrival_time_ms,
                metadata,
                session_id,
                turn_index,
                logical_request_id,
                authored_turn_index,
            } = ready;
            let session_metadata = session_id.clone().zip(turn_index);
            let authored_identity = logical_request_id
                .zip(session_id.clone())
                .zip(authored_turn_index)
                .map(|((logical_request_id, session_id), authored_turn_index)| {
                    (logical_request_id, session_id, authored_turn_index)
                });
            let uuid = self.assign_request(request, arrival_time_ms, metadata, session_id)?;
            if let Some((session_id, turn_index)) = session_metadata {
                self.collector
                    .on_session_metadata(uuid, session_id, turn_index);
            }
            if let Some((logical_request_id, session_id, authored_turn_index)) = authored_identity {
                self.collector.on_authored_identity(
                    uuid,
                    logical_request_id,
                    session_id,
                    authored_turn_index,
                );
            }
            released_any = true;
        }
        Ok(released_any)
    }

    /// Start passes on every idle worker that can make progress at the current timestamp.
    fn drive_ready_workers(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        loop {
            let effects = self.engine.drive_ready(self.now_ms, &mut self.collector)?;
            attach_pressure_references(&mut self.collector);
            if effects.is_empty() {
                return Ok(changed);
            }
            changed = true;
            if self.handle_engine_effects(effects)? {
                return Ok(changed);
            }
        }
    }

    fn handle_engine_effects(
        &mut self,
        mut effects: EngineEffects<Observation::Batch>,
    ) -> anyhow::Result<bool> {
        for admission in effects.admissions.drain(..) {
            self.collector.on_pool_admission(
                admission.uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                admission.reused_input_tokens,
            );
            if let Some(capture) = self.interactive.as_mut() {
                capture.emit_admitted(
                    admission.uuid,
                    self.now_ms,
                    admission.reused_input_tokens,
                )?;
            }
        }
        self.apply_engine_observations(effects.pass_start_events, KvIngestBoundary::PassStart)?;
        let had_immediate_completions = !effects.immediate_completions.is_empty();
        if effects.immediate_completions.len() == 1 {
            self.settle_worker_completion(
                effects
                    .immediate_completions
                    .pop()
                    .expect("singleton completion was checked"),
            )?;
        } else if had_immediate_completions {
            self.settle_worker_completion_batch(effects.immediate_completions)?;
        }
        if let Some(scheduled) = effects.scheduled_completion {
            push_worker_completions(&mut self.events, &mut self.next_event_seq, scheduled);
        }
        Ok(had_immediate_completions)
    }

    /// Activate workers whose startup period has elapsed at the current timestamp.
    fn apply_worker_ready_events(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some((stage, worker_id)) = pop_ready_worker_ready(&mut self.events, self.now_ms) {
            debug_assert_eq!(stage, SimulationWorkerStage::Aggregated);
            if self.engine.mark_worker_ready(worker_id) {
                self.set_worker_accounting_status(worker_id, ReplayWorkerLifecycleStatus::Active);
                if self.collect_fpm {
                    self.fpm_buffer
                        .activate_worker(worker_id, self.dp_size, self.now_ms);
                }
                let topology = self.engine.worker_topology(worker_id).ok_or_else(|| {
                    anyhow::anyhow!("ready worker {worker_id} has no engine topology")
                })?;
                let target = self.accounting_target(worker_id);
                let placements =
                    self.placement
                        .worker_ready_authored(topology, &target, self.now_ms)?;
                let mut released = placements
                    .iter()
                    .map(|placement| placement.request_id)
                    .collect::<Vec<_>>();
                self.dispatch_placements(placements)?;
                let placements = self.placement.topology_settled(self.now_ms)?;
                released.extend(placements.iter().map(|placement| placement.request_id));
                self.dispatch_placements(placements)?;
                let origin = startup_origin(WorkerPool::Agg, worker_id);
                record_lifecycle_operation(
                    self.now_ms,
                    WorkerPool::Agg,
                    "worker_ready_event",
                    None,
                    origin,
                    vec![WorkerLifecycleTransition {
                        worker_id,
                        transition: WorkerLifecycleTransitionKind::WorkerReady,
                        prior_state: Some("starting"),
                        state: "active",
                        reason: None,
                        origin_operation_ordinal: origin,
                    }],
                    self.lifecycle_state(),
                    released,
                );
                changed = true;
            }
            // If mark_worker_ready returned false the worker was cancelled
            // during startup (scale-down) — the stale event is silently ignored.
        }
        Ok(changed)
    }

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> anyhow::Result<bool> {
        let mut made_progress = false;
        loop {
            // A policy-visible external request is a full controller barrier.
            // Do not consume arrivals, completions, lifecycle events, or start
            // scheduler work until the controller assigns that request.
            if self.external_assignment_pending() {
                break;
            }
            #[cfg_attr(not(feature = "kvbm-offload"), allow(unused_mut))]
            let mut changed = false;
            #[cfg(feature = "kvbm-offload")]
            {
                changed |= self.tick_offload_engines()?;
            }
            changed |= self.apply_worker_completions()?;
            changed |= self.apply_worker_ready_events()?;
            changed |= self.release_ready_arrivals()?;
            if self.external_assignment_pending() {
                made_progress |= changed;
                break;
            }
            if !self.deferred_timestamp_placements.is_empty() {
                let placements = std::mem::take(&mut self.deferred_timestamp_placements);
                self.dispatch_placements(placements)?;
                changed = true;
            }
            changed |= self.drive_ready_workers()?;
            let removed = self.engine.try_remove_drained();
            for worker_id in &removed {
                self.set_worker_accounting_status(*worker_id, ReplayWorkerLifecycleStatus::Removed);
                let target = self.accounting_target(*worker_id);
                self.placement.worker_removed_authored(
                    WorkerTopology {
                        worker_id: *worker_id,
                        scheduler_ids: Vec::new(),
                    },
                    &target,
                    self.now_ms,
                )?;
            }
            if !removed.is_empty() {
                let origin = common_origin(
                    removed
                        .iter()
                        .filter_map(|worker_id| drain_origin(WorkerPool::Agg, *worker_id)),
                );
                record_lifecycle_operation(
                    self.now_ms,
                    WorkerPool::Agg,
                    "drain_settlement",
                    None,
                    origin,
                    removed
                        .iter()
                        .map(|worker_id| WorkerLifecycleTransition {
                            worker_id: *worker_id,
                            transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                            prior_state: Some("draining"),
                            state: "removed",
                            reason: None,
                            origin_operation_ordinal: drain_origin(WorkerPool::Agg, *worker_id),
                        })
                        .collect(),
                    self.lifecycle_state(),
                    Vec::new(),
                );
            }
            changed |= !removed.is_empty();
            // Scaling ticks fire last so the policy observes a settled timestamp.
            if self.scaling_policy.is_some() {
                changed |= self.apply_scaling_ticks()?;
            }

            if !changed {
                break;
            }
            made_progress = true;
        }

        Ok(made_progress)
    }

    /// Seed the first `ScalingTick` from the policy's requested start time (a
    /// non-finite time means "no tick" and is skipped).
    fn seed_first_scaling_tick(&mut self) -> anyhow::Result<()> {
        let Some(mut policy) = self.scaling_policy.take() else {
            return Ok(());
        };
        let first_ms = policy.initial_tick_ms();
        self.scaling_policy = Some(policy);
        let first_ms = first_ms?;
        if first_ms.is_finite() {
            let at_ms = first_ms.max(self.now_ms);
            push_scaling_tick(&mut self.events, &mut self.next_event_seq, at_ms);
        } else {
            // No tick will ever fire to drain the FPM buffer; stop collecting it.
            self.collect_fpm = false;
        }
        Ok(())
    }

    /// Fire every `ScalingTick`: gather a settled snapshot, call the policy,
    /// apply its decision, and re-arm.
    /// Agg routes all FPM through `decode_fpm` and ignores the prefill target.
    fn apply_scaling_ticks(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while pop_ready_scaling_tick(&mut self.events, self.now_ms) {
            if self.is_workload_done() {
                continue;
            }
            let active_decode_ids = self.engine.active_group_ids();
            self.fpm_buffer
                .emit_idle_due(&active_decode_ids, self.dp_size, self.now_ms);
            let tick_ordinal = self.next_scaling_tick_ordinal;
            let snapshot = ReplayScalingSnapshot {
                tick_ordinal,
                now_ms: self.now_ms,
                prefill_fpm: Vec::new(),
                decode_fpm: self.fpm_buffer.take(),
                traffic: self.traffic.drain(self.now_ms),
                active_prefill_ids: Vec::new(),
                active_decode_ids,
                starting_prefill_ids: Vec::new(),
                starting_decode_ids: self.engine.starting_group_ids(),
                draining_prefill_ids: Vec::new(),
                draining_decode_ids: self.engine.draining_group_ids(),
            };
            self.next_scaling_tick_ordinal = self
                .next_scaling_tick_ordinal
                .checked_add(1)
                .expect("replay scaling tick ordinal overflow");
            let mut policy = self
                .scaling_policy
                .take()
                .expect("scaling tick fired without a policy");
            let decision = policy.on_tick(snapshot);
            self.scaling_policy = Some(policy);
            let decision = decision?;

            if let Some(target) = decision.target_decode {
                self.apply_scaling_with_tick(target, Some(tick_ordinal))?;
            }

            // Re-arm only into the strict, finite future and only while work
            // remains; otherwise no later tick will drain the FPM buffer, so stop
            // collecting it (prevents unbounded growth once the cadence stops).
            let next_tick = decision
                .next_tick_ms
                .filter(|next_ms| next_ms.is_finite() && *next_ms > self.now_ms);
            if let Some(next_ms) = next_tick
                && !self.is_workload_done()
            {
                push_scaling_tick(&mut self.events, &mut self.next_event_seq, next_ms);
            } else {
                self.collect_fpm = false;
            }
            changed = true;
        }
        Ok(changed)
    }

    // ------------------------------------------------------------------
    // Scaling integration used by the in-loop `ScalingTick` handler.
    // ------------------------------------------------------------------

    fn accounting_target(&self, engine_worker_id: usize) -> WorkerTarget {
        self.interactive_worker_targets
            .get(engine_worker_id)
            .cloned()
            .unwrap_or_else(|| {
                panic!("offline replay worker {engine_worker_id} has no authored target")
            })
    }

    /// Allocate and persist an authored identity for a newly scaled worker.
    /// Engine IDs are dense implementation indices and can collide with sparse
    /// static authored IDs, so choose the lowest unused ID in the dynamic
    /// default pool instead of exposing the engine index.
    fn register_dynamic_worker_target(
        &mut self,
        engine_worker_id: usize,
    ) -> anyhow::Result<WorkerTarget> {
        anyhow::ensure!(
            engine_worker_id == self.interactive_worker_targets.len(),
            "offline replay dynamic worker target sequence diverged: new engine worker {engine_worker_id}, next target slot {}",
            self.interactive_worker_targets.len()
        );
        anyhow::ensure!(
            engine_worker_id == self.interactive_workers.len(),
            "offline replay dynamic worker metadata sequence diverged: new engine worker {engine_worker_id}, next metadata slot {}",
            self.interactive_workers.len()
        );
        let used_ids = self
            .interactive_worker_targets
            .iter()
            .filter(|target| target.pool_id == DEFAULT_REPLAY_POOL_ID)
            .map(|target| target.worker_id)
            .collect::<BTreeSet<_>>();
        let mut authored_worker_id = 0usize;
        for used_id in used_ids {
            if used_id != authored_worker_id {
                break;
            }
            authored_worker_id = authored_worker_id.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!("offline replay exhausted dynamic authored worker IDs")
            })?;
        }
        let target = WorkerTarget::default_pool(authored_worker_id, 0);
        self.interactive_worker_targets.push(target.clone());
        self.interactive_workers.push(ResolvedPoolWorker {
            target: target.clone(),
            engine_args: self.engine.dynamic_worker_args(engine_worker_id),
            tags: BTreeSet::new(),
            taints: BTreeSet::new(),
            capabilities: BTreeSet::new(),
            active: true,
            draining: false,
        });
        if !self
            .interactive_pool_ids
            .iter()
            .any(|pool_id| pool_id == &target.pool_id)
        {
            self.interactive_pool_ids.push(target.pool_id.clone());
            self.interactive_pool_ids.sort();
            if let Some(capture) = self.interactive.as_mut() {
                capture.register_pool(&target.pool_id);
            }
        }
        Ok(target)
    }

    fn set_worker_accounting_status(
        &mut self,
        engine_worker_id: usize,
        lifecycle_status: ReplayWorkerLifecycleStatus,
    ) {
        let handle = self
            .decode_worker_accounting_handles
            .get(engine_worker_id)
            .copied()
            .unwrap_or_else(|| {
                panic!("offline replay worker {engine_worker_id} has no accounting handle")
            });
        self.collector
            .set_decode_worker_lifecycle_status(handle, lifecycle_status, self.now_ms);
    }

    /// Advance the sim clock to `new_now_ms`. Provisioned worker time is
    /// accounted at lifecycle boundaries and final settlement rather than on
    /// every timestamp.
    fn advance_now_ms(&mut self, new_now_ms: f64) {
        self.now_ms = new_now_ms;
    }

    /// Number of active (non-pending-removal) workers.
    #[cfg(test)]
    pub(in crate::replay) fn active_worker_count(&self) -> usize {
        self.engine.active_group_ids().len()
    }

    /// Total worker count including pending-removal.
    #[cfg(test)]
    pub(in crate::replay) fn total_worker_count(&self) -> usize {
        self.engine.worker_count()
    }

    /// Apply a scaling decision: set the target number of workers.
    ///
    /// Scale-up: if `startup_time` is configured, new workers enter a startup
    /// phase and a `WorkerReady` event is scheduled.  They become active (and
    /// are registered with the router) only when that event fires.  Without
    /// `startup_time`, workers are available immediately.
    ///
    /// Scale-down: the worker is removed from the router immediately (so no
    /// new requests land on it) and drains in-flight work in the engine.
    #[cfg(test)]
    pub(in crate::replay) fn apply_scaling(&mut self, target_workers: usize) -> anyhow::Result<()> {
        self.apply_scaling_with_tick(target_workers, None)
    }

    fn apply_scaling_with_tick(
        &mut self,
        target_workers: usize,
        planner_tick_ordinal: Option<u64>,
    ) -> anyhow::Result<()> {
        if target_workers != self.engine.non_draining_group_count() {
            self.collector.clear_static_worker_count();
        }
        let delta = self.engine.apply_target_count(target_workers);
        #[cfg(test)]
        if !delta.added.is_empty() {
            self.worker_active_requests
                .resize(self.engine.rank_id_capacity(), Vec::new());
        }
        let startup_delay_ms = self.engine.startup_time_ms();
        let mut lifecycle_releases = Vec::new();

        for &id in &delta.added {
            let target = self.register_dynamic_worker_target(id)?;
            let handle = self.collector.register_decode_worker(
                target.pool_id,
                target.worker_id,
                target.dp_rank,
                if startup_delay_ms.is_some() {
                    ReplayWorkerLifecycleStatus::Starting
                } else {
                    ReplayWorkerLifecycleStatus::Active
                },
                self.decode_gpus_per_worker,
                self.now_ms,
            );
            anyhow::ensure!(
                !self.decode_worker_accounting_handles.contains(&handle),
                "offline replay dynamic worker {id} reused an existing accounting handle {handle:?}"
            );
            anyhow::ensure!(
                id == self.decode_worker_accounting_handles.len(),
                "offline replay accounting handle sequence diverged: new engine worker {id}, next accounting slot {}",
                self.decode_worker_accounting_handles.len()
            );
            self.decode_worker_accounting_handles.push(handle);
        }
        for &id in &delta.cancelled_startups {
            self.set_worker_accounting_status(id, ReplayWorkerLifecycleStatus::Removed);
        }
        for &id in &delta.newly_draining {
            self.set_worker_accounting_status(id, ReplayWorkerLifecycleStatus::Draining);
        }
        for &id in &delta.removed {
            self.set_worker_accounting_status(id, ReplayWorkerLifecycleStatus::Removed);
        }

        for &id in &delta.added {
            match startup_delay_ms {
                Some(delay) => {
                    push_worker_ready(
                        &mut self.events,
                        &mut self.next_event_seq,
                        self.now_ms + delay,
                        SimulationWorkerStage::Aggregated,
                        id,
                    );
                }
                None => {
                    if self.collect_fpm {
                        self.fpm_buffer
                            .activate_worker(id, self.dp_size, self.now_ms);
                    }
                    let topology = self
                        .engine
                        .worker_topology(id)
                        .ok_or_else(|| anyhow::anyhow!("new worker {id} has no engine topology"))?;
                    let target = self.accounting_target(id);
                    let placements =
                        self.placement
                            .worker_ready_authored(topology, &target, self.now_ms)?;
                    lifecycle_releases
                        .extend(placements.iter().map(|placement| placement.request_id));
                    self.dispatch_placements(placements)?;
                }
            }
        }

        for &id in &delta.newly_draining {
            let topology = self.engine.worker_topology(id).unwrap_or(WorkerTopology {
                worker_id: id,
                scheduler_ids: Vec::new(),
            });
            let target = self.accounting_target(id);
            let placements =
                self.placement
                    .worker_draining_authored(topology, &target, self.now_ms)?;
            lifecycle_releases.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        for &id in &delta.removed {
            let target = self.accounting_target(id);
            let placements = self.placement.worker_removed_authored(
                WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                },
                &target,
                self.now_ms,
            )?;
            lifecycle_releases.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        let placements = self.placement.topology_settled(self.now_ms)?;
        lifecycle_releases.extend(placements.iter().map(|placement| placement.request_id));
        self.dispatch_placements(placements)?;
        self.record_scale_lifecycle(
            &delta,
            startup_delay_ms.is_some(),
            planner_tick_ordinal,
            lifecycle_releases,
        );
        self.record_router_pending();
        self.record_in_flight_peak();
        Ok(())
    }

    fn lifecycle_state(&self) -> WorkerPoolState {
        WorkerPoolState {
            active: self.engine.active_group_ids(),
            starting: self.engine.starting_group_ids(),
            draining: self.engine.draining_group_ids(),
        }
    }

    fn record_scale_lifecycle(
        &self,
        delta: &WorkerScaleDelta,
        delayed_startup: bool,
        planner_tick_ordinal: Option<u64>,
        released: Vec<Uuid>,
    ) {
        if !lifecycle_capture_active() {
            return;
        }
        let mut transitions = Vec::new();
        transitions.extend(
            delta
                .added
                .iter()
                .map(|worker_id| WorkerLifecycleTransition {
                    worker_id: *worker_id,
                    transition: if delayed_startup {
                        WorkerLifecycleTransitionKind::WorkerStarting
                    } else {
                        WorkerLifecycleTransitionKind::WorkerReady
                    },
                    prior_state: None,
                    state: if delayed_startup {
                        "starting"
                    } else {
                        "active"
                    },
                    reason: None,
                    origin_operation_ordinal: None,
                }),
        );
        transitions.extend(delta.cancelled_startups.iter().map(|worker_id| {
            WorkerLifecycleTransition {
                worker_id: *worker_id,
                transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                prior_state: Some("starting"),
                state: "removed",
                reason: Some("startup_cancelled"),
                origin_operation_ordinal: startup_origin(WorkerPool::Agg, *worker_id),
            }
        }));
        transitions.extend(delta.newly_draining.iter().map(|worker_id| {
            WorkerLifecycleTransition {
                worker_id: *worker_id,
                transition: WorkerLifecycleTransitionKind::WorkerDraining,
                prior_state: Some("active"),
                state: "draining",
                reason: None,
                origin_operation_ordinal: None,
            }
        }));
        transitions.extend(
            delta
                .removed
                .iter()
                .map(|worker_id| WorkerLifecycleTransition {
                    worker_id: *worker_id,
                    transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                    prior_state: Some("draining"),
                    state: "removed",
                    reason: None,
                    origin_operation_ordinal: drain_origin(WorkerPool::Agg, *worker_id),
                }),
        );
        let origin = common_origin(
            delta
                .cancelled_startups
                .iter()
                .filter_map(|worker_id| startup_origin(WorkerPool::Agg, *worker_id)),
        );
        record_lifecycle_operation(
            self.now_ms,
            WorkerPool::Agg,
            if planner_tick_ordinal.is_some() {
                "planner_scale"
            } else {
                "manual_scale"
            },
            planner_tick_ordinal,
            origin,
            transitions,
            self.lifecycle_state(),
            released,
        );
    }

    /// Initialize the shared stepping kernel exactly once. Legacy `run()` and
    /// polling-based sessions both enter through this boundary, which keeps
    /// timestamp settlement and scaling-tick seeding identical.
    fn ensure_stepping_started(&mut self) -> anyhow::Result<bool> {
        if self.stepping_started {
            return Ok(false);
        }
        let changed = self.drain_current_timestamp()?;
        self.seed_first_scaling_tick()?;
        self.stepping_started = true;
        Ok(changed)
    }

    fn external_assignment_pending(&self) -> bool {
        self.interactive
            .as_ref()
            .is_some_and(InteractiveCapture::uses_external_placement)
            && self.placement.pending_count() > 0
    }

    /// Advance and settle exactly the next virtual timestamp.
    fn advance_kernel_next(&mut self) -> anyhow::Result<bool> {
        self.ensure_stepping_started()?;
        if self.external_assignment_pending() {
            return Ok(false);
        }
        if self.is_done() {
            return Ok(false);
        }
        let Some(next_timestamp_ms) = self.next_timestamp() else {
            return Ok(false);
        };
        self.advance_now_ms(next_timestamp_ms);
        self.drain_current_timestamp()?;
        Ok(true)
    }

    fn interactive_status(&self, made_progress: bool) -> ReplayStepStatus {
        if self.is_done() {
            return ReplayStepStatus::Drained {
                now_ms: self.now_ms,
            };
        }
        if made_progress {
            ReplayStepStatus::Advanced {
                now_ms: self.now_ms,
            }
        } else {
            ReplayStepStatus::Quiescent {
                now_ms: self.now_ms,
            }
        }
    }

    fn fail_if_interactive_dead_end(&mut self) -> anyhow::Result<()>
    where
        Admission: InteractiveAdmission,
    {
        let can_wait_for_external_input = self.admission.is_open()
            && self.admission.pending_requests() == 0
            && self.cluster_in_flight() == 0;
        if self.is_done()
            || self.external_assignment_pending()
            || self.next_timestamp().is_some()
            || can_wait_for_external_input
        {
            return Ok(());
        }
        bail!(
            "offline replay reached a dead end with {} in-flight requests, {} pending admission requests, and {} pending placements",
            self.cluster_in_flight(),
            self.admission.pending_requests(),
            self.placement.pending_count()
        )
    }
    pub(in crate::replay::offline) fn interactive_now_ms(&self) -> f64 {
        self.now_ms
    }

    pub(in crate::replay::offline) fn interactive_next_event_time_ms(&mut self) -> Option<f64> {
        if self.external_assignment_pending() {
            None
        } else {
            self.next_timestamp()
        }
    }

    pub(in crate::replay::offline) fn interactive_advance_next(
        &mut self,
    ) -> anyhow::Result<ReplayStepStatus>
    where
        Admission: InteractiveAdmission,
    {
        let initial_progress = self.ensure_stepping_started()?;
        if initial_progress {
            return Ok(self.interactive_status(true));
        }
        if self.external_assignment_pending() {
            return Ok(self.interactive_status(false));
        }
        let advanced = self.advance_kernel_next()?;
        self.fail_if_interactive_dead_end()?;
        Ok(self.interactive_status(advanced))
    }

    pub(in crate::replay::offline) fn interactive_settle_current_time(
        &mut self,
    ) -> anyhow::Result<ReplayStepStatus>
    where
        Admission: InteractiveAdmission,
    {
        let mut changed = self.ensure_stepping_started()?;
        if self.external_assignment_pending() {
            return Ok(self.interactive_status(changed));
        }
        changed |= self.drain_current_timestamp()?;
        self.fail_if_interactive_dead_end()?;
        Ok(self.interactive_status(changed))
    }

    /// Advance to the controller boundary, stopping at an earlier Dynamo
    /// timestamp. At most one Dynamo timestamp is crossed per call.
    pub(in crate::replay::offline) fn interactive_advance_to(
        &mut self,
        target_ms: f64,
    ) -> anyhow::Result<ReplayStepStatus>
    where
        Admission: InteractiveAdmission,
    {
        if !target_ms.is_finite() || target_ms < self.now_ms {
            bail!(
                "interactive replay cannot advance from {} ms to {target_ms} ms",
                self.now_ms
            );
        }
        let initial_progress = self.ensure_stepping_started()?;
        if initial_progress {
            return Ok(self.interactive_status(true));
        }
        if target_ms == self.now_ms {
            if self.external_assignment_pending() {
                return Ok(self.interactive_status(false));
            }
            let changed = self.drain_current_timestamp()?;
            self.fail_if_interactive_dead_end()?;
            return Ok(self.interactive_status(changed));
        }
        if self.external_assignment_pending() {
            return Ok(self.interactive_status(false));
        }
        if let Some(next_timestamp_ms) = self.next_timestamp()
            && next_timestamp_ms <= target_ms
        {
            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
            return Ok(self.interactive_status(true));
        }
        self.advance_now_ms(target_ms);
        self.drain_current_timestamp()?;
        Ok(self.interactive_status(true))
    }

    pub(in crate::replay::offline) fn interactive_is_quiescent(&mut self) -> bool
    where
        Admission: InteractiveAdmission,
    {
        !self.is_done()
            && (self.external_assignment_pending()
                || (self.next_timestamp().is_none()
                    && self.admission.is_open()
                    && self.admission.pending_requests() == 0
                    && self.cluster_in_flight() == 0))
    }

    pub(in crate::replay::offline) fn interactive_is_drained(&self) -> bool {
        self.is_done()
    }

    pub(in crate::replay::offline) fn drain_interactive_events(&mut self) -> Vec<ReplayEvent> {
        self.drain_interactive_captured_events()
            .into_iter()
            .map(CapturedReplayEvent::into_owned)
            .collect()
    }

    pub(in crate::replay::offline) fn drain_interactive_captured_events(
        &mut self,
    ) -> Vec<CapturedReplayEvent> {
        let mut events = self
            .interactive
            .as_mut()
            .map(InteractiveCapture::drain_captured_events)
            .unwrap_or_default();
        let Some(request_id) = self.placement.next_pending_request_id() else {
            return events;
        };
        let should_announce = self.interactive.as_ref().is_some_and(|capture| {
            capture.uses_external_placement() && !capture.placement_is_announced(request_id)
        });
        if !should_announce {
            return events;
        }
        let Some(request) = self
            .requests
            .get(&request_id)
            .and_then(AggRequestState::queued_request)
        else {
            return events;
        };
        let placement = self.interactive_placement_candidates(request);
        if let Some(capture) = self.interactive.as_mut() {
            let event = capture
                .placement_needed_captured_event(request_id, self.now_ms, Arc::clone(&placement))
                .expect("pending external placement must retain authored identity");
            capture.mark_placement_observed(request_id, placement);
            // Existing terminal/DAG/lifecycle events retain priority. The one
            // fresh controller boundary is always appended after them.
            events.push(event);
        }
        events
    }

    pub(in crate::replay::offline) fn interactive_snapshot(&self) -> ReplaySnapshot
    where
        Admission: InteractiveAdmission,
    {
        ReplaySnapshot {
            now_ms: self.now_ms,
            admission_open: self.admission.is_open(),
            pending_request_count: self.admission.pending_requests(),
            pending_placement_count: self.placement.pending_count(),
            workers: self.interactive_worker_snapshots(),
        }
    }

    pub(in crate::replay::offline) fn interactive_uuid_for_logical_id(
        &self,
        logical_id: &str,
    ) -> Option<Uuid> {
        self.interactive
            .as_ref()
            .and_then(|capture| capture.uuid_for_logical_id(logical_id))
    }

    pub(in crate::replay::offline) fn interactive_close_admission(&mut self) -> anyhow::Result<()>
    where
        Admission: InteractiveAdmission,
    {
        self.admission.close()
    }

    pub(in crate::replay::offline) fn finish_interactive(
        mut self,
    ) -> crate::replay::TraceSimulationReport {
        self.progress.finish();
        self.collector.settle_decode_workers(self.now_ms);
        self.collector.finish()
    }

    // ------------------------------------------------------------------
    // White-box stepping helpers retain their historical surface while using
    // the same production kernel.
    // ------------------------------------------------------------------

    /// Advance the simulation up to `until_ms` simulated time, then pause.
    /// Returns `true` if the request workload is done — pending `WorkerReady`
    /// events do not block completion since there is no work for those workers.
    #[cfg(test)]
    fn advance_to(&mut self, until_ms: f64) -> anyhow::Result<bool> {
        self.ensure_stepping_started()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };

            if next_timestamp_ms > until_ms {
                if until_ms > self.now_ms {
                    self.advance_now_ms(until_ms);
                }
                break;
            }

            let advanced = self.advance_kernel_next()?;
            debug_assert!(advanced);
        }

        Ok(self.is_workload_done())
    }

    /// Current simulated time in milliseconds.
    #[cfg(test)]
    fn now_ms(&self) -> f64 {
        self.now_ms
    }

    /// Drain accumulated traffic stats since the last drain.
    #[cfg(test)]
    fn drain_traffic(&mut self) -> TrafficStats {
        self.traffic.drain(self.now_ms)
    }

    /// Finalize the replay and return the simulation report directly.
    #[cfg(test)]
    fn finalize_report(mut self) -> crate::replay::TraceSimulationReport {
        self.progress.finish();
        self.collector.settle_decode_workers(self.now_ms);
        self.collector.finish()
    }

    /// Run the aggregated offline replay until all arrivals and worker work are exhausted.
    /// If `max_sim_time_ms` is set, exits gracefully when the next scheduled
    /// timestamp would exceed that cap; in-flight requests at that point are
    /// reported as incomplete.
    pub(in crate::replay) fn run(mut self) -> anyhow::Result<(TraceCollector, AggRuntimeStats)> {
        if let Some(cap_ms) = self.max_sim_time_ms
            && (!cap_ms.is_finite() || cap_ms < 0.0)
        {
            bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
        }
        self.ensure_stepping_started()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };
            if let Some(cap_ms) = self.max_sim_time_ms
                && next_timestamp_ms > cap_ms
            {
                break;
            }
            let advanced = self.advance_kernel_next()?;
            debug_assert!(advanced);
        }

        self.progress.finish();
        self.collector.settle_decode_workers(self.now_ms);
        Ok((self.collector, self.stats))
    }

    #[cfg(test)]
    /// Test helper: advance exactly one logical timestamp worth of work.
    fn advance_one_timestamp(&mut self) -> anyhow::Result<bool> {
        if !self.stepping_started {
            self.ensure_stepping_started()?;
            return Ok(true);
        }
        self.advance_kernel_next()
    }

    #[cfg(test)]
    fn drain_fpm(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        self.fpm_buffer.take()
    }

    #[cfg(test)]
    /// Test helper: snapshot the runtime's visible request, worker, and router state.
    fn debug_snapshot(&self) -> AggRuntimeSnapshot {
        let mut router_pending_request_ids = self
            .requests
            .iter()
            .filter(|(_, state)| state.phase == AggRequestPhase::QueuedAtRouter)
            .map(|(uuid, _)| *uuid)
            .collect::<Vec<_>>();
        router_pending_request_ids.sort_unstable();
        let mut prefill_completed = self
            .requests
            .iter()
            .filter(|(_, state)| state.prefill_completed)
            .map(|(uuid, _)| *uuid)
            .collect::<Vec<_>>();
        prefill_completed.sort_unstable();

        AggRuntimeSnapshot {
            now_ms: self.now_ms,
            worker_active_requests: self.worker_active_requests.clone(),
            workers: self.engine.debug_snapshots(),
            router_pending_request_ids,
            prefill_completed,
            router: self.placement.debug_router_snapshot(self.now_ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::entrypoints::{
        run_agentic_trace_multi_collect_with_stats, run_agentic_trace_single_collect,
        run_concurrency_multi_collect_with_stats, run_concurrency_single_collect,
        run_concurrency_workload_multi_collect_with_stats, run_concurrency_workload_single_collect,
        run_trace_multi_collect_with_stats, run_trace_single_collect,
        run_trace_workload_multi_collect_with_stats, run_trace_workload_single_collect,
    };
    use super::*;
    use crate::common::protocols::{EngineType, G1Backend, SglangArgs};
    use crate::loadgen::{AgenticTrace, AgenticTurnTrace, SessionTrace, Trace, TurnTrace};
    use crate::replay::offline::core::PlacementEffects;
    use crate::replay::offline::extensions::kv_router::{ReplayKvRouterConfig, RouterQueuePolicy};
    use crate::replay::{TraceRequestStatsSnapshot, normalize_trace_requests};
    use rstest::rstest;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CompletionPhaseEvent {
        Token(Uuid, u32),
        Progress(Uuid),
        TerminalFeedback(Uuid),
        AdmissionTerminal(Uuid),
        TerminalSettlement,
        Observation,
    }

    struct CompletionAuditAdmission {
        inner: AdmissionQueue<()>,
        events: Rc<RefCell<Vec<CompletionPhaseEvent>>>,
    }

    impl CoreAdmissionSource for CompletionAuditAdmission {
        type Request = ReplayRequestPayload;
        type Metadata = ();
        type TerminalStatus = WorkloadTerminalStatus;
        type CascadedTerminal = CascadedWorkloadTerminal;

        fn next_internal_event_ms(&mut self) -> Option<f64> {
            CoreAdmissionSource::next_internal_event_ms(&mut self.inner)
        }

        fn drain_ready(
            &mut self,
            now_ms: f64,
            cluster_in_flight: usize,
        ) -> anyhow::Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
            CoreAdmissionSource::drain_ready(&mut self.inner, now_ms, cluster_in_flight)
        }

        fn drain_ready_up_to(
            &mut self,
            now_ms: f64,
            cluster_in_flight: usize,
            limit: usize,
        ) -> anyhow::Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
            CoreAdmissionSource::drain_ready_up_to(
                &mut self.inner,
                now_ms,
                cluster_in_flight,
                limit,
            )
        }

        fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> anyhow::Result<()> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::Token(request_id, token_id));
            CoreAdmissionSource::on_output_token(&mut self.inner, request_id, token_id)
        }

        fn on_terminal(
            &mut self,
            request_id: Uuid,
            now_ms: f64,
            status: WorkloadTerminalStatus,
        ) -> anyhow::Result<Vec<CascadedWorkloadTerminal>> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::AdmissionTerminal(request_id));
            CoreAdmissionSource::on_terminal(&mut self.inner, request_id, now_ms, status)
        }

        fn is_drained(&self) -> bool {
            CoreAdmissionSource::is_drained(&self.inner)
        }

        fn total_requests(&self) -> usize {
            CoreAdmissionSource::total_requests(&self.inner)
        }
    }

    struct CompletionAuditPlacement {
        events: Rc<RefCell<Vec<CompletionPhaseEvent>>>,
        next_scheduler: usize,
        scheduler_count: usize,
        progress_release: Option<Placement>,
        terminal_release: Option<Placement>,
        observation_release: Option<Placement>,
    }

    impl PlacementPolicy<ReplayRequestPayload> for CompletionAuditPlacement {
        type Metadata = ();
        type Observation = ();

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _metadata: (),
            _session_id: Option<String>,
            _now_ms: f64,
        ) -> anyhow::Result<PlacementEffects> {
            let request_id = request
                .metadata()
                .uuid
                .expect("audit request must retain its UUID");
            let scheduler_id = self.next_scheduler % self.scheduler_count;
            self.next_scheduler += 1;
            Ok(PlacementEffects {
                decision: PlacementDecision::Immediate(Placement {
                    request_id,
                    scheduler_id,
                    reported_overlap_tokens: None,
                    planner_cache_sample: None,
                }),
                released: Vec::new(),
            })
        }

        fn observe(&mut self, _observation: (), _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::Observation);
            Ok(self.observation_release.take().into_iter().collect())
        }

        fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
            false
        }

        fn request_terminal(
            &mut self,
            _request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn request_terminal_feedback(
            &mut self,
            request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<()> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::TerminalFeedback(request_id));
            Ok(())
        }

        fn settle_terminal_feedback(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::TerminalSettlement);
            Ok(self.terminal_release.take().into_iter().collect())
        }

        fn prefill_completed(
            &mut self,
            request_id: Uuid,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            self.events
                .borrow_mut()
                .push(CompletionPhaseEvent::Progress(request_id));
            Ok(self.progress_release.take().into_iter().collect())
        }

        fn pending_count(&self) -> usize {
            0
        }

        fn worker_ready(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_draining(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_removed(
            &mut self,
            _worker: WorkerTopology,
            _now_ms: f64,
        ) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }

    impl AggregatedPlacement<(), ()> for CompletionAuditPlacement {
        fn is_router(&self) -> bool {
            false
        }

        fn debug_router_snapshot(&self, _now_ms: f64) -> Option<OfflineRouterSnapshot> {
            None
        }
    }

    type CompletionAuditRuntime =
        AggRuntimeImpl<CompletionAuditPlacement, NoEngineEvents, (), CompletionAuditAdmission>;

    fn output_signal(uuid: Uuid, token_id: Option<u32>, completed: bool) -> OutputSignal {
        OutputSignal {
            uuid,
            token_id,
            completed,
            rejected: false,
            handoff_delay_ms: None,
            cached_tokens: None,
        }
    }

    fn retarget_completion_signals(
        mut payload: WorkerCompletionPayload<()>,
        expected_worker_idx: usize,
        expected_uuid: Uuid,
        expected_completed: bool,
        output_signals: Vec<OutputSignal>,
    ) -> WorkerCompletionPayload<()> {
        assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
        assert_eq!(payload.worker_idx, expected_worker_idx);
        assert_eq!(payload.output_signals.len(), 1);
        let canonical_signal = &payload.output_signals[0];
        assert_eq!(canonical_signal.uuid, expected_uuid);
        assert_eq!(canonical_signal.completed, expected_completed);
        assert!(!canonical_signal.rejected);
        assert!(canonical_signal.token_id.is_some());
        assert_eq!(
            payload.completed_requests,
            usize::from(expected_completed),
            "scheduler-issued completion payload must match core ownership"
        );
        assert!(!output_signals.is_empty());
        assert!(
            output_signals
                .iter()
                .all(|signal| signal.uuid == expected_uuid
                    && !signal.rejected
                    && signal.token_id.is_some())
        );
        assert_eq!(
            output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            usize::from(expected_completed),
            "replacement signals must preserve completion ownership"
        );
        payload.output_signals = output_signals;
        payload
    }

    fn take_ready_completion_payloads(
        events: &mut BinaryHeap<SimulationEvent<()>>,
        now_ms: f64,
        expected_payloads: usize,
    ) -> SmallVec<[WorkerCompletionPayload<()>; 2]> {
        let mut payloads = SmallVec::new();
        while let Some(completions) = pop_ready_worker_completions(events, now_ms) {
            match completions {
                ReadyWorkerCompletions::Single(payload) => payloads.push(payload),
                ReadyWorkerCompletions::Batch(batch) => payloads.extend(batch),
            }
        }
        assert_eq!(
            payloads.len(),
            expected_payloads,
            "test must consume every scheduler-issued completion at the boundary"
        );
        payloads
    }

    fn completion_test_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1.0)
            .build()
            .unwrap()
    }

    fn runtime_with_busy_requests(
        request_ids: &[Uuid],
        num_workers: usize,
        max_output_tokens: usize,
    ) -> (
        RoundRobinAggRuntime,
        SmallVec<[WorkerCompletionPayload<()>; 2]>,
    ) {
        let mut runtime = RoundRobinAggRuntime::new_round_robin(
            &completion_test_args(),
            request_ids
                .iter()
                .enumerate()
                .map(|(index, uuid)| DirectRequest {
                    tokens: vec![index as u32 + 1; 64],
                    max_output_tokens,
                    uuid: Some(*uuid),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                })
                .collect(),
            num_workers,
            ReplayMode::Trace,
        )
        .unwrap()
        .with_per_request_records(true);
        runtime.ensure_stepping_started().unwrap();
        let completion_ms = runtime
            .events
            .peek()
            .expect("scheduler-issued completion must be queued")
            .at_ms;
        runtime.advance_now_ms(completion_ms);
        let payloads =
            take_ready_completion_payloads(&mut runtime.events, runtime.now_ms, request_ids.len());
        assert!(
            runtime
                .engine
                .debug_snapshots()
                .iter()
                .all(|worker| worker.busy),
            "test payloads must settle genuinely committed worker passes"
        );
        (runtime, payloads)
    }

    fn deferred_placement(request_id: u128) -> Placement {
        Placement {
            request_id: Uuid::from_u128(request_id),
            scheduler_id: 0,
            reported_overlap_tokens: None,
            planner_cache_sample: None,
        }
    }

    fn runtime_with_audited_busy_requests(
        request_ids: &[Uuid],
        num_workers: usize,
        max_output_tokens: usize,
        progress_release: Option<Placement>,
        terminal_release: Option<Placement>,
        observation_release: Option<Placement>,
    ) -> (
        CompletionAuditRuntime,
        Rc<RefCell<Vec<CompletionPhaseEvent>>>,
        SmallVec<[WorkerCompletionPayload<()>; 2]>,
    ) {
        let pending = request_ids
            .iter()
            .enumerate()
            .map(|(index, uuid)| DirectRequest {
                tokens: vec![index as u32 + 1; 64],
                max_output_tokens,
                uuid: Some(*uuid),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            })
            .collect();
        let events = Rc::new(RefCell::new(Vec::new()));
        let admission = CompletionAuditAdmission {
            inner: AdmissionQueue::new_requests(pending, ReplayMode::Trace),
            events: Rc::clone(&events),
        };
        let placement_events = Rc::clone(&events);
        let mut runtime = CompletionAuditRuntime::new_composed(
            &completion_test_args(),
            admission,
            num_workers,
            move |_args, topology| {
                Ok(CompletionAuditPlacement {
                    events: placement_events,
                    next_scheduler: 0,
                    scheduler_count: topology
                        .iter()
                        .map(|worker| worker.scheduler_ids.len())
                        .sum(),
                    progress_release: None,
                    terminal_release: None,
                    observation_release: None,
                })
            },
        )
        .unwrap();
        runtime.ensure_stepping_started().unwrap();
        let completion_ms = runtime
            .events
            .peek()
            .expect("scheduler-issued completion must be queued")
            .at_ms;
        runtime.advance_now_ms(completion_ms);
        let payloads =
            take_ready_completion_payloads(&mut runtime.events, runtime.now_ms, request_ids.len());
        assert!(
            runtime
                .engine
                .debug_snapshots()
                .iter()
                .all(|worker| worker.busy),
            "test payloads must settle genuinely committed worker passes"
        );
        runtime.placement.progress_release = progress_release;
        runtime.placement.terminal_release = terminal_release;
        runtime.placement.observation_release = observation_release;
        events.borrow_mut().clear();
        (runtime, events, payloads)
    }

    fn request_phase(
        runtime: &RoundRobinAggRuntime,
        uuid: Uuid,
    ) -> Option<(AggRequestPhase, bool)> {
        runtime
            .requests
            .get(&uuid)
            .map(|state| (state.phase, state.prefill_completed))
    }

    fn assert_runtime_visible_state_eq(left: &RoundRobinAggRuntime, right: &RoundRobinAggRuntime) {
        let mut left_requests = left
            .requests
            .iter()
            .map(|(uuid, state)| (*uuid, state.phase, state.prefill_completed))
            .collect::<Vec<_>>();
        let mut right_requests = right
            .requests
            .iter()
            .map(|(uuid, state)| (*uuid, state.phase, state.prefill_completed))
            .collect::<Vec<_>>();
        left_requests.sort_unstable_by_key(|(uuid, _, _)| *uuid);
        right_requests.sort_unstable_by_key(|(uuid, _, _)| *uuid);
        assert_eq!(left_requests, right_requests);
        assert_eq!(
            left.engine.debug_snapshots(),
            right.engine.debug_snapshots()
        );
        assert_eq!(
            left.deferred_timestamp_placements,
            right.deferred_timestamp_placements
        );
        assert_eq!(left.stats, right.stats);
    }

    #[test]
    fn singleton_completion_fast_path_matches_atomic_batch_path() {
        for completed in [false, true] {
            let uuid = Uuid::from_u128(0x8100 + u128::from(completed));
            let max_output_tokens = if completed { 1 } else { 2 };
            let (mut fast, mut fast_payloads) =
                runtime_with_busy_requests(&[uuid], 1, max_output_tokens);
            let (mut batch, mut batch_payloads) =
                runtime_with_busy_requests(&[uuid], 1, max_output_tokens);
            let signal = output_signal(uuid, Some(71), completed);
            let fast_payload = retarget_completion_signals(
                fast_payloads
                    .pop()
                    .expect("singleton fast-path completion must exist"),
                0,
                uuid,
                completed,
                vec![signal.clone()],
            );
            assert!(fast_payloads.is_empty());
            let batch_payload = retarget_completion_signals(
                batch_payloads
                    .pop()
                    .expect("singleton batch completion must exist"),
                0,
                uuid,
                completed,
                vec![signal],
            );
            assert!(batch_payloads.is_empty());

            fast.settle_worker_completion(fast_payload).unwrap();
            batch
                .settle_worker_completion_batch([batch_payload])
                .unwrap();

            assert_runtime_visible_state_eq(&fast, &batch);
            assert_eq!(
                fast.collector.snapshot(uuid),
                batch.collector.snapshot(uuid)
            );
            assert_eq!(fast.requests.contains_key(&uuid), !completed);
            if !completed {
                assert_eq!(
                    request_phase(&fast, uuid),
                    Some((AggRequestPhase::Running, true))
                );
            }
        }
    }

    #[test]
    fn singleton_fast_path_gates_terminal_settlement_and_defers_placements() {
        for completed in [false, true] {
            let uuid = Uuid::from_u128(0x8110 + u128::from(completed));
            let progress = deferred_placement(0x8112);
            let terminal = deferred_placement(0x8113);
            let observation = deferred_placement(0x8114);
            let max_output_tokens = if completed { 1 } else { 2 };
            let (mut runtime, events, mut payloads) = runtime_with_audited_busy_requests(
                &[uuid],
                1,
                max_output_tokens,
                (!completed).then_some(progress),
                completed.then_some(terminal),
                Some(observation),
            );
            let payload = retarget_completion_signals(
                payloads
                    .pop()
                    .expect("singleton audited completion must exist"),
                0,
                uuid,
                completed,
                vec![output_signal(uuid, Some(72), completed)],
            );
            assert!(payloads.is_empty());

            runtime.settle_worker_completion(payload).unwrap();

            let expected_events = if completed {
                vec![
                    CompletionPhaseEvent::Token(uuid, 72),
                    CompletionPhaseEvent::TerminalFeedback(uuid),
                    CompletionPhaseEvent::AdmissionTerminal(uuid),
                    CompletionPhaseEvent::TerminalSettlement,
                    CompletionPhaseEvent::Observation,
                ]
            } else {
                vec![
                    CompletionPhaseEvent::Token(uuid, 72),
                    CompletionPhaseEvent::Progress(uuid),
                    CompletionPhaseEvent::Observation,
                ]
            };
            assert_eq!(*events.borrow(), expected_events);
            assert_eq!(
                runtime.deferred_timestamp_placements,
                if completed {
                    vec![terminal, observation]
                } else {
                    vec![progress, observation]
                }
            );
        }
    }

    #[test]
    fn singleton_multi_signal_completion_preserves_progress_before_terminal() {
        let uuid = Uuid::from_u128(0x8200);
        let progress = deferred_placement(0x8201);
        let terminal = deferred_placement(0x8202);
        let observation = deferred_placement(0x8203);
        let (mut runtime, events, mut payloads) = runtime_with_audited_busy_requests(
            &[uuid],
            1,
            1,
            Some(progress),
            Some(terminal),
            Some(observation),
        );
        let payload = retarget_completion_signals(
            payloads
                .pop()
                .expect("singleton multi-signal completion must exist"),
            0,
            uuid,
            true,
            vec![
                output_signal(uuid, Some(81), false),
                output_signal(uuid, Some(82), true),
            ],
        );
        assert!(payloads.is_empty());

        runtime.settle_worker_completion(payload).unwrap();

        assert!(!runtime.requests.contains_key(&uuid));
        assert_eq!(
            *events.borrow(),
            vec![
                CompletionPhaseEvent::Token(uuid, 81),
                CompletionPhaseEvent::Token(uuid, 82),
                CompletionPhaseEvent::Progress(uuid),
                CompletionPhaseEvent::TerminalFeedback(uuid),
                CompletionPhaseEvent::AdmissionTerminal(uuid),
                CompletionPhaseEvent::TerminalSettlement,
                CompletionPhaseEvent::Observation,
            ]
        );
        assert_eq!(
            runtime.deferred_timestamp_placements,
            vec![progress, terminal, observation]
        );
    }

    #[test]
    fn same_time_sibling_completion_commits_all_workers_before_terminals() {
        let first = Uuid::from_u128(0x8301);
        let second = Uuid::from_u128(0x8302);
        let terminal = deferred_placement(0x8303);
        let observation = deferred_placement(0x8304);
        let (mut runtime, events, mut payloads) = runtime_with_audited_busy_requests(
            &[first, second],
            2,
            1,
            None,
            Some(terminal),
            Some(observation),
        );
        payloads.sort_unstable_by_key(|payload| payload.worker_idx);
        assert_eq!(
            payloads
                .iter()
                .map(|payload| payload.worker_idx)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let first_payload = retarget_completion_signals(
            payloads.remove(0),
            0,
            first,
            true,
            vec![output_signal(first, Some(91), true)],
        );
        let second_payload = retarget_completion_signals(
            payloads.remove(0),
            1,
            second,
            true,
            vec![output_signal(second, Some(92), true)],
        );
        assert!(payloads.is_empty());

        runtime
            .settle_worker_completion_batch([first_payload, second_payload])
            .unwrap();

        assert!(runtime.requests.is_empty());
        assert!(
            runtime
                .engine
                .debug_snapshots()
                .iter()
                .all(|worker| !worker.busy),
            "every sibling ownership commit must precede terminal publication"
        );
        assert_eq!(
            *events.borrow(),
            vec![
                CompletionPhaseEvent::Token(first, 91),
                CompletionPhaseEvent::Token(second, 92),
                CompletionPhaseEvent::TerminalFeedback(first),
                CompletionPhaseEvent::AdmissionTerminal(first),
                CompletionPhaseEvent::TerminalFeedback(second),
                CompletionPhaseEvent::AdmissionTerminal(second),
                CompletionPhaseEvent::TerminalSettlement,
                CompletionPhaseEvent::Observation,
                CompletionPhaseEvent::Observation,
            ]
        );
        assert_eq!(
            runtime.deferred_timestamp_placements,
            vec![terminal, observation]
        );
    }

    #[derive(Debug, Clone, PartialEq)]
    enum TwoTurnSourceEvent {
        Submitted { turn_index: usize, at_ms: f64 },
        WaitingForCompletion { turn_index: usize, at_ms: f64 },
        Terminal { turn_index: usize, at_ms: f64 },
        TimerArmed { turn_index: usize, at_ms: f64 },
    }

    struct TwoTurnWorkSource {
        driver: WorkloadDriver,
        events: Rc<RefCell<Vec<TwoTurnSourceEvent>>>,
        in_flight_turns: HashMap<Uuid, usize>,
    }

    impl TwoTurnWorkSource {
        fn new(events: Rc<RefCell<Vec<TwoTurnSourceEvent>>>) -> Self {
            let trace = Trace {
                block_size: 64,
                sessions: vec![SessionTrace {
                    session_id: "two-turn-session".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 64,
                            max_output_tokens: 2,
                            hash_ids: vec![11],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 64,
                            max_output_tokens: 2,
                            hash_ids: vec![12],
                            delay_after_previous_ms: 10.0,
                            ..Default::default()
                        },
                    ],
                }],
            };
            let driver = WorkloadDriver::new_concurrency(trace, 64, 1)
                .unwrap()
                .with_deterministic_request_ids(1);
            Self {
                driver,
                events,
                in_flight_turns: HashMap::new(),
            }
        }
    }

    impl CoreAdmissionSource for TwoTurnWorkSource {
        type Request = ReplayRequestPayload;
        type Metadata = ();
        type TerminalStatus = WorkloadTerminalStatus;
        type CascadedTerminal = CascadedWorkloadTerminal;

        fn next_internal_event_ms(&mut self) -> Option<f64> {
            self.driver.next_ready_time_ms()
        }

        fn drain_ready(
            &mut self,
            now_ms: f64,
            _cluster_in_flight: usize,
        ) -> anyhow::Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
            Ok(self
                .driver
                .pop_ready_compact(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| {
                    self.in_flight_turns
                        .insert(ready.request_uuid, ready.turn_index);
                    self.events
                        .borrow_mut()
                        .push(TwoTurnSourceEvent::Submitted {
                            turn_index: ready.turn_index,
                            at_ms: now_ms,
                        });
                    if self.driver.next_ready_time_ms().is_none() {
                        self.events
                            .borrow_mut()
                            .push(TwoTurnSourceEvent::WaitingForCompletion {
                                turn_index: ready.turn_index,
                                at_ms: now_ms,
                            });
                    }
                    ReadyArrival {
                        request: ready.request,
                        arrival_time_ms: now_ms,
                        metadata: (),
                        session_id: Some(ready.session_id),
                        turn_index: Some(ready.turn_index),
                        logical_request_id: ready.logical_request_id,
                        authored_turn_index: Some(ready.authored_turn_index),
                    }
                })
                .collect())
        }

        fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> anyhow::Result<()> {
            self.driver.on_output_token(request_id, token_id)
        }

        fn on_terminal(
            &mut self,
            request_id: Uuid,
            now_ms: f64,
            status: WorkloadTerminalStatus,
        ) -> anyhow::Result<Vec<CascadedWorkloadTerminal>> {
            let turn_index = self
                .in_flight_turns
                .remove(&request_id)
                .expect("terminal request must belong to a submitted turn");
            self.events.borrow_mut().push(TwoTurnSourceEvent::Terminal {
                turn_index,
                at_ms: now_ms,
            });
            let cascaded = self.driver.on_terminal(request_id, now_ms, status)?;
            if let Some(at_ms) = self.driver.next_ready_time_ms() {
                self.events
                    .borrow_mut()
                    .push(TwoTurnSourceEvent::TimerArmed {
                        turn_index: turn_index + 1,
                        at_ms,
                    });
            }
            Ok(cascaded)
        }

        fn is_drained(&self) -> bool {
            self.driver.is_drained()
        }

        fn total_requests(&self) -> usize {
            self.driver.total_turns()
        }
    }

    #[test]
    fn generic_work_source_waits_for_terminal_before_arming_second_turn() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let source = TwoTurnWorkSource::new(Rc::clone(&events));
        let args = fast_router_args();
        let runtime = AggRuntimeImpl::<
            AggregatedRoundRobinPlacement<()>,
            NoEngineEvents,
            (),
            TwoTurnWorkSource,
        >::new_composed(&args, source, 1, |args, topology| {
            Ok(AggregatedRoundRobinPlacement::with_taints(
                args.dp_size,
                topology,
                &args.worker_taints,
            ))
        })
        .unwrap();

        let (collector, _) = runtime.run().unwrap();
        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 2);

        let events = events.borrow();
        println!("two-turn generic work-source event log:");
        for event in events.iter() {
            match event {
                TwoTurnSourceEvent::Submitted { turn_index, at_ms } => {
                    println!("  {at_ms:>9.3} ms  submit turn {turn_index}");
                }
                TwoTurnSourceEvent::WaitingForCompletion { turn_index, at_ms } => {
                    println!(
                        "  {at_ms:>9.3} ms  no source timer; wait for turn {turn_index} completion"
                    );
                }
                TwoTurnSourceEvent::Terminal { turn_index, at_ms } => {
                    println!("  {at_ms:>9.3} ms  terminal turn {turn_index}");
                }
                TwoTurnSourceEvent::TimerArmed { turn_index, at_ms } => {
                    println!("  {at_ms:>9.3} ms  arm turn {turn_index} delay timer");
                }
            }
        }

        let first_terminal_ms = events
            .iter()
            .find_map(|event| match event {
                TwoTurnSourceEvent::Terminal {
                    turn_index: 0,
                    at_ms,
                } => Some(*at_ms),
                _ => None,
            })
            .unwrap();
        let second_submit_ms = events
            .iter()
            .find_map(|event| match event {
                TwoTurnSourceEvent::Submitted {
                    turn_index: 1,
                    at_ms,
                } => Some(*at_ms),
                _ => None,
            })
            .unwrap();
        assert_eq!(second_submit_ms, first_terminal_ms + 10.0);
    }

    struct CaptureOncePolicy {
        at_ms: f64,
        captured: Rc<RefCell<Option<ReplayScalingSnapshot>>>,
    }

    impl ReplayScalingPolicy for CaptureOncePolicy {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(self.at_ms)
        }

        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            *self.captured.borrow_mut() = Some(snapshot);
            Ok(ReplayScalingDecision::default())
        }
    }

    struct ScriptedPolicy {
        initial_ms: f64,
        next_ticks: VecDeque<Option<f64>>,
        snapshots: Rc<RefCell<Vec<ReplayScalingSnapshot>>>,
    }

    impl ReplayScalingPolicy for ScriptedPolicy {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(self.initial_ms)
        }

        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            self.snapshots.borrow_mut().push(snapshot);
            Ok(ReplayScalingDecision {
                next_tick_ms: self.next_ticks.pop_front().flatten(),
                ..Default::default()
            })
        }
    }

    struct DisabledPolicy {
        calls: Rc<RefCell<usize>>,
    }

    impl ReplayScalingPolicy for DisabledPolicy {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(f64::INFINITY)
        }

        fn on_tick(
            &mut self,
            _snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            *self.calls.borrow_mut() += 1;
            Ok(ReplayScalingDecision::default())
        }
    }

    struct ScaleAtStartPolicy {
        snapshots: Rc<RefCell<Vec<ReplayScalingSnapshot>>>,
    }

    impl ReplayScalingPolicy for ScaleAtStartPolicy {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            let first = self.snapshots.borrow().is_empty();
            self.snapshots.borrow_mut().push(snapshot);
            Ok(if first {
                ReplayScalingDecision {
                    target_decode: Some(2),
                    next_tick_ms: Some(5_000.0),
                    ..Default::default()
                }
            } else {
                ReplayScalingDecision::default()
            })
        }
    }

    fn replay_args(enable_prefix_caching: bool, enable_chunked_prefill: bool) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn parity_args(engine_type: EngineType) -> MockEngineArgs {
        let mut builder = MockEngineArgs::builder()
            .engine_type(engine_type)
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0);
        if engine_type == EngineType::Sglang {
            builder = builder.sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }));
        }
        builder.build().unwrap()
    }

    fn parity_requests() -> Vec<DirectRequest> {
        vec![
            DirectRequest {
                tokens: vec![1; 4],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![2; 8],
                max_output_tokens: 4,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(101.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![3; 12],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
                ..Default::default()
            },
        ]
    }

    fn parity_workload() -> Trace {
        Trace {
            block_size: 4,
            sessions: vec![
                SessionTrace {
                    session_id: "session-a".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 4,
                            max_output_tokens: 2,
                            hash_ids: vec![11],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 12,
                            max_output_tokens: 2,
                            hash_ids: vec![21, 22, 23],
                            delay_after_previous_ms: 5.0,
                            ..Default::default()
                        },
                    ],
                },
                SessionTrace {
                    session_id: "session-b".to_string(),
                    first_arrival_timestamp_ms: Some(1.0),
                    turns: vec![TurnTrace {
                        input_length: 8,
                        max_output_tokens: 2,
                        hash_ids: vec![31, 32],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        }
    }

    fn parity_agentic_trace() -> AgenticTrace {
        AgenticTrace {
            block_size: 4,
            turns: vec![
                AgenticTurnTrace {
                    request_id: "root".to_string(),
                    session_id: "root".to_string(),
                    input_length: 4,
                    max_output_tokens: 2,
                    hash_ids: vec![1],
                    first_ready_timestamp_ms: Some(0.0),
                    delay_after_dependencies_ms: 0.0,
                    wait_for: Vec::new(),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "dependent".to_string(),
                    session_id: "dependent".to_string(),
                    input_length: 8,
                    max_output_tokens: 2,
                    hash_ids: vec![1, 2],
                    first_ready_timestamp_ms: Some(100.0),
                    delay_after_dependencies_ms: 5.0,
                    wait_for: vec!["root".to_string()],
                    prefix_reset: false,
                    ..Default::default()
                },
            ],
        }
    }

    fn sorted_snapshots(collector: &TraceCollector) -> Vec<TraceRequestStatsSnapshot> {
        let mut snapshots = collector.snapshots();
        snapshots.sort_by_key(|snapshot| snapshot.input_length);
        snapshots
    }

    fn assert_collectors_match(single: TraceCollector, multi: TraceCollector) {
        assert_eq!(sorted_snapshots(&single), sorted_snapshots(&multi));

        let single_report = single.finish();
        let multi_report = multi.finish();
        assert_eq!(
            single_report.request_counts.num_requests,
            multi_report.request_counts.num_requests
        );
        assert_eq!(
            single_report.request_counts.completed_requests,
            multi_report.request_counts.completed_requests
        );
        assert_eq!(
            single_report.request_counts.total_input_tokens,
            multi_report.request_counts.total_input_tokens
        );
        assert_eq!(
            single_report.request_counts.total_output_tokens,
            multi_report.request_counts.total_output_tokens
        );
    }

    #[test]
    fn one_shot_and_repeated_stepping_share_report_and_worker_seconds() {
        let args = parity_args(EngineType::Vllm);
        let make_runtime = || {
            AggRuntime::new(
                &args,
                None,
                None,
                normalize_trace_requests(parity_requests(), 1.0).unwrap(),
                2,
                ReplayMode::Trace,
                ReplayRouterMode::RoundRobin,
            )
            .unwrap()
        };

        let (one_shot_collector, one_shot_stats) = make_runtime().run().unwrap();
        let one_shot_report = one_shot_collector.finish();

        let mut stepped = make_runtime();
        while stepped.advance_one_timestamp().unwrap() {}
        let stepped_stats = stepped.stats.clone();
        let stepped_report = stepped.finalize_report();

        assert_eq!(one_shot_stats, stepped_stats);
        assert_eq!(
            one_shot_report.throughput.decode_worker_seconds,
            stepped_report.throughput.decode_worker_seconds
        );
        assert_eq!(
            one_shot_report.throughput.prefill_worker_seconds,
            stepped_report.throughput.prefill_worker_seconds
        );
        let one_shot_accounting = one_shot_report
            .topology_accounting
            .as_ref()
            .expect("aggregated replay registers worker accounting");
        let stepped_accounting = stepped_report
            .topology_accounting
            .as_ref()
            .expect("stepped replay registers worker accounting");
        assert_eq!(one_shot_accounting, stepped_accounting);
        assert_eq!(one_shot_accounting.workers.len(), 2);
        for worker in &one_shot_accounting.workers {
            assert_eq!(
                worker.worker_seconds,
                one_shot_report.throughput.duration_ms / 1000.0
            );
        }

        let canonical = |report: &crate::replay::TraceSimulationReport| {
            let mut value = serde_json::to_value(report).unwrap();
            let summary = value.as_object_mut().unwrap();
            for excluded in [
                "wall_time_ms",
                "processed_tokens_per_s",
                "processed_output_tokens_per_s",
            ] {
                summary.remove(excluded);
            }
            value
        };
        assert_eq!(canonical(&one_shot_report), canonical(&stepped_report));
    }

    fn fast_router_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap()
    }

    fn queueing_router_args(policy: RouterQueuePolicy) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(10.0)
            .router_queue_policy(Some(policy))
            .build()
            .unwrap()
    }

    fn queueing_router_config(policy: RouterQueuePolicy) -> ReplayKvRouterConfig {
        ReplayKvRouterConfig {
            router_queue_threshold: Some(0.5),
            router_queue_policy: policy,
            ..ReplayKvRouterConfig::default()
        }
    }

    fn run_trace_multi_queueing_collect_with_stats(
        policy: RouterQueuePolicy,
        requests: Vec<DirectRequest>,
        num_workers: usize,
    ) -> (TraceCollector, AggRuntimeStats) {
        let args = queueing_router_args(policy);
        let pending = normalize_trace_requests(requests, 1.0).unwrap();
        AggRuntime::new(
            &args,
            Some(queueing_router_config(policy)),
            None,
            pending,
            num_workers,
            ReplayMode::Trace,
            ReplayRouterMode::KvRouter,
        )
        .unwrap()
        .run()
        .unwrap()
    }

    fn run_concurrency_multi_queueing_collect_with_stats(
        policy: RouterQueuePolicy,
        requests: Vec<DirectRequest>,
        max_in_flight: usize,
        num_workers: usize,
    ) -> (TraceCollector, AggRuntimeStats) {
        let args = queueing_router_args(policy);
        AggRuntime::new(
            &args,
            Some(queueing_router_config(policy)),
            None,
            VecDeque::from(requests),
            num_workers,
            ReplayMode::Concurrency { max_in_flight },
            ReplayRouterMode::KvRouter,
        )
        .unwrap()
        .run()
        .unwrap()
    }

    fn planner_router_config() -> ReplayKvRouterConfig {
        ReplayKvRouterConfig {
            router_queue_threshold: Some(0.5),
            ..ReplayKvRouterConfig::default()
        }
    }

    fn sglang_replay_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(512)
            .speedup_ratio(1000.0)
            .sglang(Some(SglangArgs {
                page_size: Some(2),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn sglang_zero_output_request_does_not_block_following_work() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .speedup_ratio(1000.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 4],
                max_output_tokens: 0,
                uuid: Some(Uuid::from_u128(9_000)),
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![2; 4],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(9_001)),
                arrival_timestamp_ms: Some(1.0),
                ..Default::default()
            },
        ];

        let (collector, _) =
            run_trace_multi_collect_with_stats(&args, requests, 1, ReplayRouterMode::RoundRobin);
        let mut snapshots = collector.snapshots();
        snapshots.sort_by_key(|snapshot| snapshot.requested_output_length);

        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].requested_output_length, 0);
        assert_eq!(snapshots[0].output_length, 0);
        assert!(snapshots[0].first_admit_ms.is_some());
        assert_eq!(snapshots[0].first_token_ms, None);
        assert_eq!(snapshots[1].requested_output_length, 1);
        assert_eq!(snapshots[1].output_length, 1);

        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 1);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    fn planned_output_length_controls_aggregate_accounting(#[case] engine_type: EngineType) {
        let args = match engine_type {
            EngineType::Vllm => MockEngineArgs::builder()
                .block_size(4)
                .num_gpu_blocks(32)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(1))
                .speedup_ratio(1000.0)
                .build()
                .unwrap(),
            EngineType::Sglang => sglang_replay_args(),
            EngineType::Trtllm => unreachable!(),
        };
        let requests = vec![DirectRequest {
            tokens: vec![1; 4],
            max_output_tokens: 1,
            output_token_ids: Some(vec![7, 8, 9]),
            uuid: Some(Uuid::from_u128(9_002)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }];

        let (collector, _) =
            run_trace_multi_collect_with_stats(&args, requests, 1, ReplayRouterMode::RoundRobin);
        let snapshot = collector.snapshot(Uuid::from_u128(9_002)).unwrap();
        assert_eq!(snapshot.requested_output_length, 3);
        assert_eq!(snapshot.output_length, 3);
        assert_eq!(collector.finish().request_counts.total_output_tokens, 3);
    }

    #[test]
    fn sglang_completion_visible_fpm_reaches_aggregated_buffer() {
        let pending = normalize_trace_requests(
            vec![DirectRequest {
                tokens: vec![1; 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(9_001)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            }],
            1.0,
        )
        .unwrap();
        let mut runtime = AggRuntime::new(
            &sglang_replay_args(),
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_fpm_capture();

        assert!(runtime.advance_one_timestamp().unwrap());
        assert!(runtime.drain_fpm().is_empty());
        assert!(runtime.advance_one_timestamp().unwrap());
        assert!(
            !runtime.drain_fpm().is_empty(),
            "SGLang pass-end FPM must become planner-visible at completion"
        );
    }

    #[test]
    fn attention_dp_fpm_preserves_logical_worker_and_rank_identity() {
        let mut args = sglang_replay_args();
        args.dp_size = 2;
        let pending = normalize_trace_requests(
            vec![
                DirectRequest {
                    tokens: vec![1; 8],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(9_101)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 8],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(9_102)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ],
            1.0,
        )
        .unwrap();
        let mut runtime = AggRuntime::new(
            &args,
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_fpm_capture();

        while runtime.advance_one_timestamp().unwrap() {}
        let identities = runtime
            .drain_fpm()
            .into_iter()
            .map(|(worker_id, snapshot)| (worker_id, snapshot.worker_id, snapshot.dp_rank))
            .collect::<std::collections::BTreeSet<_>>();

        assert!(identities.contains(&(0, "0".to_string(), 0)));
        assert!(identities.contains(&(0, "0".to_string(), 1)));
    }

    #[test]
    fn scaling_tick_emits_idle_fpm_after_simulated_second() {
        let mut args = sglang_replay_args();
        args.dp_size = 2;
        let pending = normalize_trace_requests(
            vec![
                DirectRequest {
                    tokens: vec![1; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(9_201)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(9_202)),
                    arrival_timestamp_ms: Some(3_000.0),
                    ..Default::default()
                },
            ],
            1.0,
        )
        .unwrap();
        let captured = Rc::new(RefCell::new(None));
        let policy = CaptureOncePolicy {
            at_ms: 2_000.0,
            captured: Rc::clone(&captured),
        };

        AggRuntime::new(
            &args,
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_scaling_policy(Box::new(policy))
        .run()
        .unwrap();

        let metrics = captured
            .borrow_mut()
            .take()
            .expect("planner tick must fire");
        assert_eq!(metrics.now_ms, 2_000.0);
        assert_eq!(metrics.decode_fpm.len(), 2);
        assert!(metrics.decode_fpm.iter().all(|(worker_id, snapshot)| {
            *worker_id == 0
                && snapshot.wall_time_secs == 0.0
                && snapshot.num_prefill_requests == 0
                && snapshot.num_decode_requests == 0
                && snapshot.num_queued_prefill == 0
                && snapshot.num_queued_decode == 0
        }));
        assert_eq!(
            metrics
                .decode_fpm
                .iter()
                .map(|(_, snapshot)| snapshot.dp_rank)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn scaling_tick_clamps_recurs_settles_and_stops_on_nonfuture_time() {
        let pending =
            normalize_trace_requests(simple_requests(2, 1_000.0).into_iter().collect(), 1.0)
                .unwrap();
        let snapshots = Rc::new(RefCell::new(Vec::new()));
        let policy = ScriptedPolicy {
            initial_ms: -5.0,
            next_ticks: VecDeque::from([Some(100.0), Some(200.0), Some(200.0)]),
            snapshots: Rc::clone(&snapshots),
        };

        AggRuntime::new(
            &startup_args(0.0),
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_scaling_policy(Box::new(policy))
        .run()
        .unwrap();

        let snapshots = snapshots.borrow();
        assert_eq!(
            snapshots
                .iter()
                .map(|snapshot| snapshot.now_ms)
                .collect::<Vec<_>>(),
            vec![0.0, 100.0, 200.0]
        );
        assert_eq!(snapshots[0].active_decode_ids, vec![0]);
    }

    #[test]
    fn scaling_tick_observes_worker_ready_at_same_timestamp() {
        let pending =
            normalize_trace_requests(simple_requests(2, 10_000.0).into_iter().collect(), 1.0)
                .unwrap();
        let snapshots = Rc::new(RefCell::new(Vec::new()));

        AggRuntime::new(
            &startup_args(5.0),
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_scaling_policy(Box::new(ScaleAtStartPolicy {
            snapshots: Rc::clone(&snapshots),
        }))
        .run()
        .unwrap();

        let snapshots = snapshots.borrow();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[1].now_ms, 5_000.0);
        assert_eq!(snapshots[1].active_decode_ids, vec![0, 1]);
        assert!(snapshots[1].starting_decode_ids.is_empty());
    }

    #[test]
    fn nonfinite_initial_scaling_tick_disables_callback() {
        let pending =
            normalize_trace_requests(simple_requests(2, 1_000.0).into_iter().collect(), 1.0)
                .unwrap();
        let calls = Rc::new(RefCell::new(0));

        AggRuntime::new(
            &startup_args(0.0),
            None,
            None,
            pending,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_scaling_policy(Box::new(DisabledPolicy {
            calls: Rc::clone(&calls),
        }))
        .run()
        .unwrap();

        assert_eq!(*calls.borrow(), 0);
    }

    #[test]
    fn generic_attention_dp_counts_rank_resources_in_gpu_hours() {
        let mut args = fast_router_args();
        args.dp_size = 4;
        let runtime = AggRuntime::new(
            &args,
            None,
            None,
            simple_requests(4, 0.0),
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        let (collector, _) = runtime.run().unwrap();
        let report = collector.finish();

        assert_eq!(report.throughput.decode_gpus_per_worker, 4);
        assert_eq!(
            report.throughput.gpu_hours,
            report.throughput.decode_worker_seconds * 4.0 / 3600.0
        );
    }

    fn trtllm_reject_args() -> MockEngineArgs {
        // 4 GPU blocks * block_size 4 = 16-token to-completion budget per request.
        MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap()
    }

    fn reject_request(uuid: u128, prompt_tokens: u32, max_output: usize) -> DirectRequest {
        let base = uuid as u32 * 100_000;
        DirectRequest {
            tokens: (base..base + prompt_tokens).collect(),
            max_output_tokens: max_output,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }
    }

    /// Aggregated-runtime regression for terminal-rejection propagation. An
    /// oversized request (footprint exceeds the whole KV pool) at the FIFO head
    /// must be terminally rejected so it neither hangs the `max_in_flight = 1`
    /// slot (no terminal signal = dead-ended `in_flight`) nor is counted as a
    /// completion; the valid follower behind it runs to completion.
    #[test]
    fn trtllm_oversized_request_rejected_unblocks_follower_agg() {
        let oversized = reject_request(1, 20, 8); // 20-token prompt = 5 blocks > 4-block pool
        let valid = reject_request(2, 4, 4); // 2 blocks, fits
        let (collector, _stats) = run_concurrency_multi_collect_with_stats(
            &trtllm_reject_args(),
            vec![oversized, valid],
            1, // max_in_flight = 1: rejection must free the slot or the run hangs
            1,
            ReplayRouterMode::RoundRobin,
        );
        let report = collector.finish();
        assert_eq!(
            report.request_counts.num_requests, 2,
            "both requests arrived"
        );
        assert_eq!(
            report.request_counts.completed_requests, 1,
            "only the valid request completes; the rejected one is excluded"
        );
        assert_eq!(
            report.request_counts.total_output_tokens, 4,
            "rejected request contributes no output tokens to the report"
        );
    }

    fn multiturn_trace() -> Trace {
        Trace {
            block_size: 64,
            sessions: vec![
                SessionTrace {
                    session_id: "session-a".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 64,
                            max_output_tokens: 2,
                            hash_ids: vec![11],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 192,
                            max_output_tokens: 2,
                            hash_ids: vec![21, 22, 23],
                            delay_after_previous_ms: 10.0,
                            ..Default::default()
                        },
                    ],
                },
                SessionTrace {
                    session_id: "session-b".to_string(),
                    first_arrival_timestamp_ms: Some(5.0),
                    turns: vec![TurnTrace {
                        input_length: 128,
                        max_output_tokens: 2,
                        hash_ids: vec![31, 32],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        }
    }

    #[test]
    fn test_trace_workload_follow_up_turn_arrives_after_completion_plus_delay() {
        let args = fast_router_args();
        let (collector, stats) = run_trace_workload_multi_collect_with_stats(
            &args,
            multiturn_trace(),
            2,
            ReplayRouterMode::RoundRobin,
            false,
        );

        let first_turn_uuid = *stats
            .dispatch_order
            .iter()
            .find(|uuid| {
                collector
                    .snapshot(**uuid)
                    .is_some_and(|stats| stats.input_length == 64)
            })
            .unwrap();
        let second_turn_uuid = *stats
            .dispatch_order
            .iter()
            .find(|uuid| {
                collector
                    .snapshot(**uuid)
                    .is_some_and(|stats| stats.input_length == 192)
            })
            .unwrap();
        let session_b_uuid = *stats
            .dispatch_order
            .iter()
            .find(|uuid| {
                collector
                    .snapshot(**uuid)
                    .is_some_and(|stats| stats.input_length == 128)
            })
            .unwrap();

        let first_turn = collector.snapshot(first_turn_uuid).unwrap();
        let second_turn = collector.snapshot(second_turn_uuid).unwrap();
        let session_b = collector.snapshot(session_b_uuid).unwrap();

        assert_eq!(first_turn.arrival_time_ms, 0.0);
        assert_eq!(session_b.arrival_time_ms, 5.0);
        assert!(
            second_turn.arrival_time_ms >= first_turn.last_token_ms.unwrap() + 10.0,
            "follow-up turn should unlock after completion plus delay"
        );
    }

    #[test]
    fn test_delta_workload_reuses_generated_output_blocks() {
        let args = replay_args(true, true);
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 5,
                        hash_ids: vec![1],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 3,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        ..Default::default()
                    },
                ],
            }],
        };

        let (collector, stats) = run_trace_workload_multi_collect_with_stats(
            &args,
            trace,
            1,
            ReplayRouterMode::KvRouter,
            true,
        );
        let report = collector.finish();

        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_input_tokens, 16);
        assert_eq!(report.request_counts.total_output_tokens, 6);
        assert_eq!(
            stats.overlap_history,
            vec![0, 2],
            "second delta turn should reuse the input block and one generated-output block"
        );
    }

    #[test]
    fn test_delta_workload_tracks_clamped_and_rejected_outputs() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 20,
                        hash_ids: vec![1, 2, 3, 4],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 2,
                        hash_ids: vec![5],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![6],
                        ..Default::default()
                    },
                ],
            }],
        };

        // The 16-token pool clamps turn 0 from 20 outputs to 12. Turn 1's
        // resulting 17-token prompt is rejected, so the failed session blocks
        // turn 2 instead of treating engine rejection as successful progress.
        let args = trtllm_reject_args();
        let driver = trace
            .into_delta_accumulating_trace_driver_with_block_size(args.block_size)
            .unwrap();
        let (collector, stats) = AggRuntime::new_workload(
            &args,
            None,
            None,
            driver,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_per_request_records(true)
        .run()
        .unwrap();
        let input_lengths = stats
            .dispatch_order
            .iter()
            .map(|uuid| collector.snapshot(*uuid).unwrap().input_length)
            .collect::<Vec<_>>();
        let report = collector.finish();

        assert_eq!(input_lengths, vec![4, 17]);
        assert_eq!(report.request_counts.num_requests, 3);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request.len(), 3);
        let terminal_for_turn = |turn_index| {
            report
                .per_request
                .iter()
                .find(|record| record.turn_index == Some(turn_index))
                .map(|record| record.terminal_status)
                .unwrap_or_else(|| panic!("missing terminal record for authored turn {turn_index}"))
        };
        assert_eq!(terminal_for_turn(0), ReplayTerminalStatus::Completed);
        assert_eq!(terminal_for_turn(1), ReplayTerminalStatus::Rejected);
        assert_eq!(terminal_for_turn(2), ReplayTerminalStatus::Canceled);
        assert_eq!(
            report.request_counts.num_requests,
            report.per_request.len(),
            "every authored turn must reconcile to exactly one terminal record"
        );
    }

    #[test]
    fn test_concurrency_workload_holds_session_slot_depth_first() {
        let args = fast_router_args();
        let (collector, stats) = run_concurrency_workload_multi_collect_with_stats(
            &args,
            multiturn_trace(),
            1,
            2,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.max_in_flight_seen, 1);
        let dispatch_input_lengths = stats
            .dispatch_order
            .iter()
            .map(|uuid| collector.snapshot(*uuid).unwrap().input_length)
            .collect::<Vec<_>>();
        assert_eq!(dispatch_input_lengths, vec![64, 192, 128]);
    }

    #[test]
    fn test_concurrency_ttft_excludes_cap_wait_and_think_time() {
        // Deterministic TTFT-boundary check (no sleeps). cap=1, depth-first: session-a runs
        // t0 (input 64) → 10ms inter-turn think-time → t1 (input 192); session-b (input 128)
        // is cap-blocked the whole time. The collector defines TTFT = first_token - arrival,
        // and concurrency stamps `arrival` at DISPATCH (now_ms) — the same dispatch-time
        // stamping the online runtime uses (`live_runtime.rs`, Concurrency arm). So the cap
        // wait and the think-time (both elapse BEFORE dispatch) are excluded from TTFT, while
        // routing/prefill (AFTER dispatch) is included.
        let args = fast_router_args();
        let (collector, stats) = run_concurrency_workload_multi_collect_with_stats(
            &args,
            multiturn_trace(),
            1,
            2,
            ReplayRouterMode::RoundRobin,
        );

        let snap = |input_len: usize| {
            let uuid = stats
                .dispatch_order
                .iter()
                .find(|u| collector.snapshot(**u).unwrap().input_length == input_len)
                .expect("request with this input_length was dispatched");
            collector.snapshot(*uuid).unwrap()
        };
        let a0 = snap(64); // session-a turn-0
        let a1 = snap(192); // session-a turn-1 (behind 10ms think-time)
        let b = snap(128); // session-b (cap-blocked behind session-a)

        // TTFT (as the collector defines it: first_token - arrival) is positive for every
        // request — i.e. it is measured from dispatch and *does* include the post-dispatch
        // prefill/routing compute.
        for s in [&a0, &a1, &b] {
            assert!(
                s.first_token_ms.unwrap() - s.arrival_time_ms > 0.0,
                "prefill/routing time is included in TTFT"
            );
        }

        // Think-time excluded: a.t1 is dispatched only after a.t0 completes + 10ms think-time,
        // so that 10ms sits before a.t1's arrival and cannot be inside its TTFT.
        assert!(
            a1.arrival_time_ms >= a0.last_token_ms.unwrap() + 10.0,
            "a.t1 is admitted only after the inter-turn think-time elapses"
        );

        // Cap wait excluded: session-b is blocked for the whole time session-a runs, so it is
        // dispatched late (large arrival), yet its TTFT is only its own prefill — the long
        // pre-dispatch wait is not folded in.
        assert!(
            b.arrival_time_ms >= a1.last_token_ms.unwrap(),
            "b (cap-blocked) is admitted only after session-a fully completes"
        );
        assert!(
            b.first_token_ms.unwrap() - b.arrival_time_ms < b.arrival_time_ms,
            "the cap wait before b's dispatch is excluded from b's TTFT"
        );
    }

    #[test]
    fn test_trace_workload_kv_router_precomputed_hashes_match_request_fallback() {
        let args = fast_router_args();
        let requests = vec![
            DirectRequest {
                tokens: [vec![11; 64], vec![21; 32]].concat(),
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(111)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: [vec![11; 64], vec![22; 32]].concat(),
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(222)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
                ..Default::default()
            },
        ];
        let workload = Trace {
            block_size: 64,
            sessions: vec![
                SessionTrace {
                    session_id: "session-a".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![TurnTrace {
                        input_length: 96,
                        max_output_tokens: 2,
                        hash_ids: vec![11, 21],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
                SessionTrace {
                    session_id: "session-b".to_string(),
                    first_arrival_timestamp_ms: Some(500.0),
                    turns: vec![TurnTrace {
                        input_length: 96,
                        max_output_tokens: 2,
                        hash_ids: vec![11, 22],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        };

        let (request_collector, request_stats) =
            run_trace_multi_collect_with_stats(&args, requests, 2, ReplayRouterMode::KvRouter);
        let (workload_collector, workload_stats) = run_trace_workload_multi_collect_with_stats(
            &args,
            workload,
            2,
            ReplayRouterMode::KvRouter,
            false,
        );
        let request_report = request_collector.finish();
        let workload_report = workload_collector.finish();

        assert_eq!(request_stats.dispatch_history.len(), 2);
        assert_eq!(workload_stats.dispatch_history.len(), 2);
        assert_eq!(
            request_stats.dispatch_history[0],
            request_stats.dispatch_history[1]
        );
        assert_eq!(
            workload_stats.dispatch_history[0],
            workload_stats.dispatch_history[1]
        );
        assert_eq!(
            request_report.request_counts.completed_requests,
            workload_report.request_counts.completed_requests
        );
        assert_eq!(
            request_report.request_counts.total_input_tokens,
            workload_report.request_counts.total_input_tokens
        );
        assert_eq!(
            request_report.request_counts.total_output_tokens,
            workload_report.request_counts.total_output_tokens
        );
        assert_eq!(
            request_report.prefix_cache_reused_ratio,
            workload_report.prefix_cache_reused_ratio
        );
        assert_eq!(
            request_report.first_admission_prefix_cache_reused_ratio,
            workload_report.first_admission_prefix_cache_reused_ratio
        );
    }

    #[test]
    fn test_multi_worker_trace_kv_router_delays_cached_visibility_until_pass_completion() {
        let policy = RouterQueuePolicy::Fcfs;
        let mut args = queueing_router_args(policy);
        // Exercise the shared scheduler's KVBM event capture path.
        args.g1_backend = Some(G1Backend::Kvbm);
        let mut runtime = AggRuntime::new(
            &args,
            Some(queueing_router_config(policy)),
            None,
            normalize_trace_requests(
                vec![
                    DirectRequest {
                        tokens: vec![11; 64],
                        max_output_tokens: 8,
                        output_token_ids: None,
                        uuid: Some(Uuid::from_u128(11)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                        ..Default::default()
                    },
                    DirectRequest {
                        tokens: vec![22; 64],
                        max_output_tokens: 8,
                        output_token_ids: None,
                        uuid: Some(Uuid::from_u128(22)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                        ..Default::default()
                    },
                    DirectRequest {
                        tokens: vec![11; 64],
                        max_output_tokens: 2,
                        output_token_ids: None,
                        uuid: Some(Uuid::from_u128(33)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.1),
                        ..Default::default()
                    },
                ],
                1.0,
            )
            .unwrap(),
            2,
            ReplayMode::Trace,
            ReplayRouterMode::KvRouter,
        )
        .unwrap();

        assert!(runtime.advance_one_timestamp().unwrap());
        let initial = runtime.debug_snapshot();
        let initial_router = initial.router.as_ref().unwrap();

        assert_eq!(initial.now_ms, 0.0);
        assert!(initial.router_pending_request_ids.is_empty());
        assert!(initial_router.pending.is_empty());
        assert_eq!(
            initial
                .worker_active_requests
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(initial_router.indexer.total_cached_blocks, 0);

        assert!(runtime.advance_one_timestamp().unwrap());
        let queued = runtime.debug_snapshot();
        let queued_router = queued.router.as_ref().unwrap();

        assert_eq!(queued.now_ms, 0.1);
        assert_eq!(queued.router_pending_request_ids, vec![Uuid::from_u128(33)]);
        assert_eq!(queued_router.pending.len(), 1);
        assert_eq!(queued_router.pending[0].uuid, Uuid::from_u128(33));

        assert!(
            queued_router.pending[0]
                .overlap_blocks_by_worker
                .iter()
                .all(|(_, overlap)| *overlap == 0),
            "a mid-pass arrival must not observe KV blocks before pass completion"
        );
        while !runtime
            .stats
            .assigned_worker_by_uuid
            .contains_key(&Uuid::from_u128(33))
        {
            assert!(runtime.advance_one_timestamp().unwrap());
        }

        let dispatched = runtime.debug_snapshot();
        assert!(dispatched.router_pending_request_ids.is_empty());
        assert!(
            dispatched
                .router
                .as_ref()
                .unwrap()
                .indexer
                .total_cached_blocks
                > 0,
            "completed passes must publish their KV blocks"
        );
    }

    #[test]
    fn test_apply_scaling_drains_router_pending_immediately() {
        let args = queueing_router_args(RouterQueuePolicy::Fcfs);
        let mut runtime = AggRuntime::new(
            &args,
            Some(planner_router_config()),
            None,
            normalize_trace_requests(
                vec![
                    DirectRequest {
                        tokens: vec![11; 64],
                        max_output_tokens: 8,
                        output_token_ids: None,
                        uuid: Some(Uuid::from_u128(1)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                        ..Default::default()
                    },
                    DirectRequest {
                        tokens: vec![22; 64],
                        max_output_tokens: 8,
                        output_token_ids: None,
                        uuid: Some(Uuid::from_u128(2)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                        ..Default::default()
                    },
                ],
                1.0,
            )
            .unwrap(),
            1,
            ReplayMode::Trace,
            ReplayRouterMode::KvRouter,
        )
        .unwrap();

        let ((), evidence) = super::super::evidence::with_runtime_evidence(
            crate::replay::ReplayCaptureOptions {
                capture_planner_details: true,
                ..Default::default()
            },
            || {
                assert!(runtime.advance_one_timestamp().unwrap());
                assert_eq!(
                    runtime.debug_snapshot().router_pending_request_ids,
                    vec![Uuid::from_u128(2)]
                );

                runtime.apply_scaling(2).unwrap();

                assert!(
                    runtime
                        .debug_snapshot()
                        .router_pending_request_ids
                        .is_empty()
                );
                assert_eq!(
                    runtime.stats.assigned_worker_by_uuid[&Uuid::from_u128(2)],
                    1
                );
            },
        );
        assert_eq!(evidence.lifecycle_operations.len(), 1);
        assert_eq!(
            evidence.lifecycle_operations[0].topology_released_request_uuids,
            vec![Uuid::from_u128(2).to_string()]
        );
    }

    #[test]
    fn test_multi_worker_trace_round_robin_assigns_same_timestamp_requests_deterministically() {
        let args = replay_args(false, true);
        let (collector, _) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 4,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![7, 7, 7, 7, 8, 8, 8, 8],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                    ..Default::default()
                },
            ],
            2,
            ReplayRouterMode::RoundRobin,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let request_4 = collector.snapshot(Uuid::from_u128(44)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, 1.0);
        assert_eq!(request_4.arrival_time_ms, 1.0);

        assert!(request_3.first_admit_ms.unwrap() >= request_1.first_token_ms.unwrap());
        assert!(request_4.first_admit_ms.unwrap() >= request_2.first_token_ms.unwrap());
        assert!(request_3.first_admit_ms.unwrap() < request_4.first_admit_ms.unwrap());

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.request_counts.total_input_tokens, 40);
        assert_eq!(report.request_counts.total_output_tokens, 10);
    }

    #[test]
    fn test_multi_worker_trace_round_robin_records_dispatch_history() {
        let args = replay_args(false, true);
        let (_, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1; 8],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 8],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![3; 8],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![4; 8],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(4)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![5; 8],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(5)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ],
            4,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 1, 2, 3, 0]);
    }

    #[test]
    fn test_attention_dp_round_robin_matches_live_worker_then_rank_order() {
        let mut args = replay_args(false, true);
        args.dp_size = 2;
        let requests = (1..=5)
            .map(|id| DirectRequest {
                tokens: vec![id as u32; 8],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(id)),
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            })
            .collect();

        let (_, stats) =
            run_trace_multi_collect_with_stats(&args, requests, 2, ReplayRouterMode::RoundRobin);

        // Live routing round-robins mocker workers first, then each worker's
        // MockEngine independently round-robins its DP ranks.
        assert_eq!(stats.dispatch_history, vec![0, 2, 1, 3, 0]);
    }

    #[test]
    fn test_attention_dp_planner_counts_mocker_workers_not_ranks() {
        let mut args = replay_args(false, true);
        args.dp_size = 2;
        let mut runtime = AggRuntime::new(
            &args,
            None,
            None,
            VecDeque::new(),
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        assert_eq!(runtime.active_worker_count(), 2);
        assert_eq!(runtime.total_worker_count(), 2);
        assert_eq!(runtime.engine.active_worker_ids(), vec![0, 1, 2, 3]);

        runtime.apply_scaling(3).unwrap();
        assert_eq!(runtime.active_worker_count(), 3);
        assert_eq!(runtime.total_worker_count(), 3);
        assert_eq!(runtime.engine.active_worker_ids(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_offline_trace_replay_sglang_single_worker_completes() {
        let args = sglang_replay_args();
        let (collector, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(901)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(902)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(5.0),
                    ..Default::default()
                },
            ],
            1,
            ReplayRouterMode::RoundRobin,
        );

        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 4);
        assert_eq!(stats.dispatch_history, vec![0, 0]);
    }

    #[test]
    fn test_offline_trace_replay_sglang_kv_router_smoke() {
        let args = sglang_replay_args();
        let (collector, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![7; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(911)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![7; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(912)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(500.0),
                    ..Default::default()
                },
            ],
            2,
            ReplayRouterMode::KvRouter,
        );

        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(stats.dispatch_history.len(), 2);
        assert_eq!(
            stats.overlap_history,
            vec![0, 32],
            "second identical SGLang request should see all 32 KV blocks cached"
        );
    }

    #[test]
    fn test_multi_worker_concurrency_uses_worker_in_flight_for_cap_checks() {
        let args = replay_args(false, false);
        let (collector, _) = run_concurrency_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(900.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                    max_output_tokens: 4,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(1000.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                    ..Default::default()
                },
            ],
            2,
            2,
            ReplayRouterMode::RoundRobin,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, request_1.last_token_ms.unwrap());
        assert!(request_3.arrival_time_ms < request_2.last_token_ms.unwrap());
        assert_eq!(request_3.first_admit_ms.unwrap(), request_3.arrival_time_ms);

        assert_eq!(report.request_counts.completed_requests, 3);
        assert_eq!(report.request_counts.total_input_tokens, 24);
        assert_eq!(report.request_counts.total_output_tokens, 8);
    }

    #[test]
    fn test_multi_worker_trace_kv_router_prefers_cached_workers_after_delay() {
        let args = fast_router_args();
        let (_, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![11; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![11; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(2.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(2.0),
                    ..Default::default()
                },
            ],
            2,
            ReplayRouterMode::KvRouter,
        );

        let worker_a1 = stats.assigned_worker_by_uuid[&Uuid::from_u128(11)];
        let worker_b1 = stats.assigned_worker_by_uuid[&Uuid::from_u128(22)];
        let worker_a2 = stats.assigned_worker_by_uuid[&Uuid::from_u128(33)];
        let worker_b2 = stats.assigned_worker_by_uuid[&Uuid::from_u128(44)];

        assert_ne!(worker_a1, worker_b1);
        assert_eq!(worker_a1, worker_a2);
        assert_eq!(worker_b1, worker_b2);
    }

    #[test]
    fn test_multi_worker_trace_kv_router_marks_prefill_and_free_correctly() {
        let args = fast_router_args();
        let (_, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![9; 64],
                    max_output_tokens: 1,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(9)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![8; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(8)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ],
            2,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(stats.prefill_marked_count, 1);
        assert_eq!(stats.router_freed_count, 2);
        assert_eq!(stats.max_router_pending_count, 0);
    }

    #[test]
    fn test_multi_worker_trace_kv_router_queues_until_prefill_completion() {
        let (collector, stats) = run_trace_multi_queueing_collect_with_stats(
            RouterQueuePolicy::Fcfs,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 8,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 8,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![3; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.1),
                    ..Default::default()
                },
            ],
            2,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(1)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(2)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(3)).unwrap();
        let first_unblock_ms = request_1
            .first_token_ms
            .unwrap()
            .min(request_2.first_token_ms.unwrap());

        assert!(stats.max_router_pending_count > 0);
        assert!(request_3.first_admit_ms.unwrap() > request_3.arrival_time_ms);
        assert_eq!(request_3.first_admit_ms.unwrap(), first_unblock_ms);
        assert!(request_3.first_admit_ms.unwrap() < request_1.last_token_ms.unwrap());
        assert!(request_3.first_admit_ms.unwrap() < request_2.last_token_ms.unwrap());
    }

    #[test]
    fn test_multi_worker_trace_kv_router_fcfs_and_lcfs_dispatch_in_opposite_queue_order() {
        let requests = vec![
            DirectRequest {
                tokens: vec![10; 64],
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(10)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![20; 64],
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(20)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![30; 64],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(30)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.1),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![40; 64],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(40)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.2),
                ..Default::default()
            },
        ];

        let (_, fcfs_stats) = run_trace_multi_queueing_collect_with_stats(
            RouterQueuePolicy::Fcfs,
            requests.clone(),
            2,
        );
        let (_, lcfs_stats) =
            run_trace_multi_queueing_collect_with_stats(RouterQueuePolicy::Lcfs, requests, 2);

        assert!(fcfs_stats.max_router_pending_count > 0);
        assert!(lcfs_stats.max_router_pending_count > 0);
        assert_eq!(
            &fcfs_stats.dispatch_order[..2],
            &[Uuid::from_u128(10), Uuid::from_u128(20)]
        );
        assert_eq!(
            &lcfs_stats.dispatch_order[..2],
            &[Uuid::from_u128(10), Uuid::from_u128(20)]
        );
        assert_eq!(
            &fcfs_stats.dispatch_order[2..4],
            &[Uuid::from_u128(30), Uuid::from_u128(40)]
        );
        assert_eq!(
            &lcfs_stats.dispatch_order[2..4],
            &[Uuid::from_u128(40), Uuid::from_u128(30)]
        );
    }

    #[test]
    fn test_multi_worker_trace_kv_router_fcfs_and_lcfs_admit_queued_requests_in_opposite_timestamp_order()
     {
        let requests = vec![
            DirectRequest {
                tokens: vec![10; 64],
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(10)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![20; 128],
                max_output_tokens: 8,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(20)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![30; 64],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(30)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.1),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![40; 64],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(40)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.2),
                ..Default::default()
            },
        ];

        let (fcfs_collector, fcfs_stats) = run_trace_multi_queueing_collect_with_stats(
            RouterQueuePolicy::Fcfs,
            requests.clone(),
            2,
        );
        let (lcfs_collector, lcfs_stats) =
            run_trace_multi_queueing_collect_with_stats(RouterQueuePolicy::Lcfs, requests, 2);

        let fcfs_request_30 = fcfs_collector.snapshot(Uuid::from_u128(30)).unwrap();
        let fcfs_request_40 = fcfs_collector.snapshot(Uuid::from_u128(40)).unwrap();
        let lcfs_request_30 = lcfs_collector.snapshot(Uuid::from_u128(30)).unwrap();
        let lcfs_request_40 = lcfs_collector.snapshot(Uuid::from_u128(40)).unwrap();

        assert!(fcfs_stats.max_router_pending_count > 0);
        assert!(lcfs_stats.max_router_pending_count > 0);
        assert_eq!(
            &fcfs_stats.dispatch_order[2..4],
            &[Uuid::from_u128(30), Uuid::from_u128(40)]
        );
        assert_eq!(
            &lcfs_stats.dispatch_order[2..4],
            &[Uuid::from_u128(40), Uuid::from_u128(30)]
        );
        assert!(fcfs_request_30.first_admit_ms.unwrap() < fcfs_request_40.first_admit_ms.unwrap());
        assert!(lcfs_request_40.first_admit_ms.unwrap() < lcfs_request_30.first_admit_ms.unwrap());
    }

    #[test]
    fn test_multi_worker_concurrency_kv_router_respects_max_in_flight() {
        let (_, stats) = run_concurrency_multi_queueing_collect_with_stats(
            RouterQueuePolicy::Fcfs,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(4)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
            ],
            3,
            2,
        );

        assert_eq!(stats.max_in_flight_seen, 3);
        assert!(stats.max_router_pending_count > 0);
    }

    #[test]
    fn test_multi_worker_concurrency_kv_router_records_backfill_timing() {
        let args = queueing_router_args(RouterQueuePolicy::Fcfs);
        let (collector, stats) = run_concurrency_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 4,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![3; 64],
                    max_output_tokens: 2,
                    output_token_ids: None,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                    ..Default::default()
                },
            ],
            2,
            2,
            ReplayRouterMode::KvRouter,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, request_1.last_token_ms.unwrap());
        assert!(request_3.arrival_time_ms < request_2.last_token_ms.unwrap());
        assert_eq!(request_3.first_admit_ms.unwrap(), request_3.arrival_time_ms);
        assert_eq!(stats.max_in_flight_seen, 2);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    #[case(EngineType::Trtllm)]
    fn test_multi_worker_trace_single_worker_round_robin_matches_single_runtime(
        #[case] engine_type: EngineType,
    ) {
        let args = parity_args(engine_type);
        let requests = parity_requests();
        let single = run_trace_single_collect(args.clone(), requests.clone(), 1.0);
        let (multi, stats) =
            run_trace_multi_collect_with_stats(&args, requests, 1, ReplayRouterMode::RoundRobin);

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_collectors_match(single, multi);
    }

    #[test]
    fn test_multi_worker_trace_single_worker_kv_router_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(101.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![9, 9, 9, 9, 8, 8, 8, 8],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
                ..Default::default()
            },
        ];

        let single = run_trace_single_collect(args.clone(), requests.clone(), 1.0);
        let (multi, stats) =
            run_trace_multi_collect_with_stats(&args, requests, 1, ReplayRouterMode::KvRouter);

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_eq!(stats.max_router_pending_count, 0);
        for uuid in [11_u128, 22, 33] {
            assert_eq!(
                multi.snapshot(Uuid::from_u128(uuid)),
                single.snapshot(Uuid::from_u128(uuid))
            );
        }
        assert_eq!(multi.finish().request_counts.completed_requests, 3);
        assert_eq!(single.finish().request_counts.completed_requests, 3);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    #[case(EngineType::Trtllm)]
    fn test_multi_worker_concurrency_single_worker_round_robin_matches_single_runtime(
        #[case] engine_type: EngineType,
    ) {
        let args = parity_args(engine_type);
        let requests = parity_requests();
        let single = run_concurrency_single_collect(args.clone(), requests.clone(), 2);
        let (multi, stats) = run_concurrency_multi_collect_with_stats(
            &args,
            requests,
            2,
            1,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_collectors_match(single, multi);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    #[case(EngineType::Trtllm)]
    fn test_trace_workload_single_worker_round_robin_matches_single_runtime(
        #[case] engine_type: EngineType,
    ) {
        let args = parity_args(engine_type);
        let single = run_trace_workload_single_collect(args.clone(), parity_workload());
        let (multi, stats) = run_trace_workload_multi_collect_with_stats(
            &args,
            parity_workload(),
            1,
            ReplayRouterMode::RoundRobin,
            false,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_collectors_match(single, multi);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    #[case(EngineType::Trtllm)]
    fn test_concurrency_workload_single_worker_round_robin_matches_single_runtime(
        #[case] engine_type: EngineType,
    ) {
        let args = parity_args(engine_type);
        let single = run_concurrency_workload_single_collect(args.clone(), parity_workload(), 1);
        let (multi, stats) = run_concurrency_workload_multi_collect_with_stats(
            &args,
            parity_workload(),
            1,
            1,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_collectors_match(single, multi);
    }

    #[rstest]
    #[case(EngineType::Vllm)]
    #[case(EngineType::Sglang)]
    #[case(EngineType::Trtllm)]
    fn test_agentic_trace_single_worker_round_robin_matches_single_runtime(
        #[case] engine_type: EngineType,
    ) {
        let args = parity_args(engine_type);
        let single = run_agentic_trace_single_collect(args.clone(), parity_agentic_trace());
        let (multi, stats) = run_agentic_trace_multi_collect_with_stats(
            &args,
            parity_agentic_trace(),
            1,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0]);
        assert_collectors_match(single, multi);
    }

    #[test]
    fn test_multi_worker_concurrency_single_worker_kv_router_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(900.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                max_output_tokens: 4,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(1000.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
        ];

        let single = run_concurrency_single_collect(args.clone(), requests.clone(), 2);
        let (multi, stats) = run_concurrency_multi_collect_with_stats(
            &args,
            requests,
            2,
            1,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        assert_eq!(stats.max_router_pending_count, 0);
        for uuid in [11_u128, 22, 33] {
            assert_eq!(
                multi.snapshot(Uuid::from_u128(uuid)),
                single.snapshot(Uuid::from_u128(uuid))
            );
        }
    }

    // ---- startup delay tests ----

    fn startup_args(startup_time_s: f64) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .startup_time(Some(startup_time_s))
            .build()
            .unwrap()
    }

    fn simple_requests(n: usize, arrival_interval_ms: f64) -> VecDeque<DirectRequest> {
        (0..n)
            .map(|i| DirectRequest {
                tokens: vec![1; 64],
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(i as f64 * arrival_interval_ms),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn test_apply_scaling_with_startup_delay_defers_activation() {
        // Use enough requests spread over a long enough window that the
        // workload is still in-flight when the startup delay elapses.
        let args = startup_args(5.0); // 5-second startup delay
        let requests = simple_requests(20, 1000.0); // arrivals at 0, 1s, 2s, ... 19s
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        // Advance to t=500ms — first request dispatched to worker 0.
        rt.advance_to(500.0).unwrap();
        assert_eq!(rt.active_worker_count(), 1);
        assert_eq!(rt.total_worker_count(), 1);

        // Scale up to 2 workers. The WorkerReady event is scheduled at
        // now_ms + 5000ms.
        rt.apply_scaling(2).unwrap();
        let scale_time = rt.now_ms();
        let expected_ready_ms = scale_time + 5000.0;
        assert_eq!(rt.active_worker_count(), 1); // new worker still starting
        assert_eq!(rt.total_worker_count(), 2);

        // Advance to just before the worker is ready.
        rt.advance_to(expected_ready_ms - 1.0).unwrap();
        assert_eq!(rt.active_worker_count(), 1); // still starting

        // Advance past the startup time.
        rt.advance_to(expected_ready_ms).unwrap();
        assert_eq!(rt.active_worker_count(), 2); // now active
        assert_eq!(rt.total_worker_count(), 2);
    }

    #[test]
    fn test_worker_seconds_counts_startup_ramp() {
        // 1 worker over [0, 1s], then scale to 2 with a 5s startup delay and
        // advance to 3s. The second worker is still *starting up* over [1s, 3s]
        // but is provisioned (holds a GPU), so worker-seconds must count it:
        //   1 worker × 1s + 2 workers × 2s = 5.0 worker-seconds.
        // (If it integrated the *active* count it would wrongly be 3.0.)
        let args = startup_args(5.0);
        let requests = simple_requests(20, 1000.0);
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        rt.advance_to(1000.0).unwrap();
        rt.apply_scaling(2).unwrap();
        assert_eq!(rt.active_worker_count(), 1); // 2nd worker still starting
        assert_eq!(rt.total_worker_count(), 2); // ...but provisioned
        rt.advance_to(3000.0).unwrap();

        let report = rt.finalize_report();
        assert!(
            (report.throughput.decode_worker_seconds - 5.0).abs() < 1e-6,
            "expected 5.0 provisioned worker-seconds (startup ramp counted), got {}",
            report.throughput.decode_worker_seconds
        );
        assert_eq!(report.throughput.prefill_worker_seconds, 0.0); // agg: decode role only
    }

    #[test]
    fn accounting_handles_follow_dynamic_lifecycle_without_reuse() {
        let args = startup_args(5.0);
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            simple_requests(1, 0.0),
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        rt.advance_now_ms(1000.0);
        rt.apply_scaling(3).unwrap();
        assert_eq!(rt.engine.starting_group_ids(), vec![2]);
        rt.advance_now_ms(2000.0);

        // Cancel the startup and drain/remove the idle highest active worker.
        rt.apply_scaling(1).unwrap();
        assert_eq!(rt.engine.active_group_ids(), vec![0]);
        assert_eq!(rt.total_worker_count(), 1);
        rt.advance_now_ms(3000.0);

        // A later scale-up receives a new stable engine ID and a new collector
        // handle; neither the cancelled startup nor removed worker is reused.
        rt.apply_scaling(2).unwrap();
        assert_eq!(rt.engine.starting_group_ids(), vec![3]);
        assert_eq!(rt.decode_worker_accounting_handles.len(), 4);
        assert_ne!(
            rt.decode_worker_accounting_handles[2],
            rt.decode_worker_accounting_handles[3]
        );

        rt.advance_now_ms(6000.0);
        assert!(!rt.apply_worker_ready_events().unwrap()); // cancelled worker 2
        rt.advance_now_ms(8000.0);
        assert!(rt.apply_worker_ready_events().unwrap()); // new worker 3
        assert_eq!(rt.engine.active_group_ids(), vec![0, 3]);
        rt.apply_scaling(1).unwrap();
        assert_eq!(rt.engine.active_group_ids(), vec![0]);
        rt.advance_now_ms(9000.0);

        let report = rt.finalize_report();
        assert_eq!(report.throughput.decode_worker_seconds, 17.0);
        let accounting = report.topology_accounting.unwrap();
        assert_eq!(
            accounting
                .workers
                .iter()
                .map(|worker| (
                    worker.worker_id,
                    worker.lifecycle_status,
                    worker.worker_seconds
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, ReplayWorkerLifecycleStatus::Active, 9.0),
                (1, ReplayWorkerLifecycleStatus::Removed, 2.0),
                (2, ReplayWorkerLifecycleStatus::Removed, 1.0),
                (3, ReplayWorkerLifecycleStatus::Removed, 5.0),
            ]
        );
        assert!(accounting.reconciliation.all_reconciled());
    }

    #[test]
    fn sparse_authored_worker_ids_do_not_collide_with_dynamic_accounting() {
        let args = fast_router_args();
        let workers = [2, 7]
            .into_iter()
            .map(|worker_id| ResolvedPoolWorker {
                target: WorkerTarget::default_pool(worker_id, 0),
                engine_args: args.clone(),
                tags: BTreeSet::new(),
                taints: BTreeSet::new(),
                capabilities: BTreeSet::new(),
                active: true,
                draining: false,
            })
            .collect();
        let mut rt = ExternalAggRuntime::new_composed_heterogeneous(
            &args,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            workers,
            |_args, topology, workers| {
                Ok(ExternalPlacement::new_pooled(
                    topology,
                    workers.to_vec(),
                    vec![(
                        DEFAULT_REPLAY_POOL_ID.to_string(),
                        super::super::topology::PoolRouter::RoundRobin,
                    )],
                ))
            },
        )
        .unwrap();

        rt.advance_now_ms(1000.0);
        rt.apply_scaling(3).unwrap();
        assert_eq!(
            rt.interactive_worker_targets,
            vec![
                WorkerTarget::default_pool(2, 0),
                WorkerTarget::default_pool(7, 0),
                WorkerTarget::default_pool(0, 0),
            ]
        );
        assert_eq!(rt.decode_worker_accounting_handles.len(), 3);
        assert!(
            rt.decode_worker_accounting_handles
                .iter()
                .enumerate()
                .all(
                    |(index, handle)| !rt.decode_worker_accounting_handles[..index]
                        .contains(handle)
                )
        );
        rt.preassign_interactive(
            Uuid::from_u128(991),
            WorkerTarget::default_pool(0, 0),
            BTreeSet::new(),
        )
        .unwrap();

        rt.advance_now_ms(2000.0);
        rt.apply_scaling(2).unwrap();
        assert!(
            rt.preassign_interactive(
                Uuid::from_u128(992),
                WorkerTarget::default_pool(0, 0),
                BTreeSet::new(),
            )
            .is_err(),
            "removed dynamic authored target must leave external placement"
        );
        rt.advance_now_ms(3000.0);

        let report = rt.finalize_report();
        assert_eq!(report.throughput.decode_worker_seconds, 7.0);
        let accounting = report.topology_accounting.unwrap();
        assert_eq!(
            accounting
                .workers
                .iter()
                .map(|worker| (
                    worker.worker_id,
                    worker.lifecycle_status,
                    worker.worker_seconds,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, ReplayWorkerLifecycleStatus::Removed, 1.0),
                (2, ReplayWorkerLifecycleStatus::Active, 3.0),
                (7, ReplayWorkerLifecycleStatus::Active, 3.0),
            ]
        );
        assert!(accounting.reconciliation.all_reconciled());
    }

    #[test]
    fn test_advance_to_moves_clock_across_idle_gap() {
        let args = fast_router_args();
        let requests = VecDeque::from([DirectRequest {
            tokens: vec![1; 64],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(1000.0),
            ..Default::default()
        }]);
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        rt.advance_to(500.0).unwrap();

        assert_eq!(rt.now_ms(), 500.0);
        let stats = rt.drain_traffic();
        assert!((stats.duration_s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_drain_traffic_reports_mtp_accept_length() {
        // MTP (nextn=2, accept_rates="1,1") makes every decode forward emit
        // 3 visible tokens (1 base + 2 accepted speculative), so draining
        // traffic after the workload completes must surface
        // avg_accept_length == 3.0 alongside the requested output length
        // (osl == 12). This is the end-to-end accept-length path the planner
        // observes per tick via the drained traffic stats. (Ported from the
        // Python scaling-policy coverage that drove the
        // now-removed planner stepping API directly.)
        let args = MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(512)
            .max_num_batched_tokens(Some(2048))
            .max_num_seqs(Some(16))
            .enable_prefix_caching(false)
            .speedup_ratio(1000.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .build()
            .unwrap();
        let requests = (0..2)
            .map(|i| DirectRequest {
                tokens: vec![1; 128],
                max_output_tokens: 12,
                uuid: Some(Uuid::from_u128(i + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            })
            .collect::<VecDeque<_>>();
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        let done = rt.advance_to(1000.0).unwrap();
        assert!(done, "workload should complete within the advance window");

        let stats = rt.drain_traffic();
        assert_eq!(stats.num_req, 2);
        assert_eq!(stats.avg_osl, 12.0);
        assert!(
            (stats.avg_accept_length.unwrap() - 3.0).abs() < 1e-6,
            "expected MTP accept_length 3.0, got {:?}",
            stats.avg_accept_length
        );
    }

    #[test]
    fn test_drain_traffic_uses_context_capped_output_length() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(32)
            .max_model_len(Some(8))
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let requests = VecDeque::from([DirectRequest {
            tokens: vec![1; 7],
            max_output_tokens: 4,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        }]);
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        assert!(rt.advance_to(1000.0).unwrap());
        let stats = rt.drain_traffic();
        assert_eq!(stats.num_req, 1);
        assert_eq!(stats.avg_osl, 1.0);
    }

    #[test]
    fn test_apply_scaling_without_startup_is_immediate() {
        let args = fast_router_args(); // no startup_time
        let requests = simple_requests(4, 100.0);
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        rt.advance_to(50.0).unwrap();
        rt.apply_scaling(2).unwrap();
        // Without startup delay, new worker is immediately active.
        assert_eq!(rt.active_worker_count(), 2);
        assert_eq!(rt.total_worker_count(), 2);
    }

    #[test]
    fn scale_down_forgets_retired_round_robin_rank_state() {
        let args = fast_router_args();
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            simple_requests(2, 0.0),
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        assert!(rt.advance_one_timestamp().unwrap());
        assert_eq!(rt.placement.tracked_round_robin_workers().len(), 2);
        while rt.advance_one_timestamp().unwrap() {}

        rt.apply_scaling(1).unwrap();

        assert_eq!(rt.placement.tracked_round_robin_workers().len(), 1);
        assert!(!rt.placement.tracked_round_robin_workers().contains_key(&1));
    }

    #[test]
    fn idle_scale_down_finalizes_router_state_and_worker_seconds() {
        let args = fast_router_args();
        let requests = normalize_trace_requests(
            vec![
                DirectRequest {
                    tokens: vec![11; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(1)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(2)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ],
            1.0,
        )
        .unwrap();
        let mut rt = AggRuntime::new(
            &args,
            Some(planner_router_config()),
            None,
            requests,
            2,
            ReplayMode::Trace,
            ReplayRouterMode::KvRouter,
        )
        .unwrap();

        while !rt.is_done() {
            assert!(rt.advance_one_timestamp().unwrap());
        }
        let before = rt.debug_snapshot();
        assert!(
            before
                .router
                .as_ref()
                .unwrap()
                .indexer
                .cached_blocks_by_worker
                .iter()
                .any(|(worker_id, _)| *worker_id == 1),
            "the retiring worker should have retained cache state before finalization"
        );

        let scale_time_ms = rt.now_ms();
        rt.apply_scaling(1).unwrap();
        assert_eq!(rt.active_worker_count(), 1);
        assert_eq!(rt.total_worker_count(), 1);
        let after = rt.debug_snapshot();
        let router = after.router.as_ref().unwrap();
        assert!(
            router
                .indexer
                .cached_blocks_by_worker
                .iter()
                .all(|(worker_id, _)| *worker_id != 1)
        );
        assert!(
            router
                .active_blocks_by_worker
                .iter()
                .all(|(worker_id, _)| *worker_id != 1)
        );

        rt.advance_now_ms(scale_time_ms + 1000.0);
        let report = rt.finalize_report();
        let expected_worker_seconds = 2.0 * scale_time_ms / 1000.0 + 1.0;
        assert!(
            (report.throughput.decode_worker_seconds - expected_worker_seconds).abs() < 1e-6,
            "both initial workers plus only the remaining post-scale worker should accrue, got {} (expected {expected_worker_seconds})",
            report.throughput.decode_worker_seconds
        );
    }

    #[test]
    fn busy_scale_down_retires_after_final_completion() {
        let args = fast_router_args();
        let requests = normalize_trace_requests(
            vec![
                DirectRequest {
                    tokens: vec![11; 64],
                    max_output_tokens: 32,
                    uuid: Some(Uuid::from_u128(1)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 32,
                    uuid: Some(Uuid::from_u128(2)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ],
            1.0,
        )
        .unwrap();
        let mut rt = AggRuntime::new(
            &args,
            Some(planner_router_config()),
            None,
            requests,
            2,
            ReplayMode::Trace,
            ReplayRouterMode::KvRouter,
        )
        .unwrap();

        assert!(rt.advance_one_timestamp().unwrap());
        assert_eq!(
            rt.debug_snapshot()
                .worker_active_requests
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );

        rt.apply_scaling(1).unwrap();
        assert_eq!(rt.active_worker_count(), 1);
        assert_eq!(
            rt.total_worker_count(),
            2,
            "busy retiring worker must remain provisioned while draining"
        );
        assert!(
            rt.debug_snapshot()
                .router
                .as_ref()
                .unwrap()
                .active_tokens_by_worker
                .iter()
                .any(|(worker_id, _)| *worker_id == 1),
            "router ownership must remain until the worker's final completion"
        );

        while rt.total_worker_count() == 2 {
            assert!(rt.advance_one_timestamp().unwrap());
        }
        assert_eq!(rt.total_worker_count(), 1);
        let router = rt.debug_snapshot().router.unwrap();
        assert!(
            router
                .active_tokens_by_worker
                .iter()
                .all(|(worker_id, _)| *worker_id != 1)
        );
        assert!(
            router
                .indexer
                .cached_blocks_by_worker
                .iter()
                .all(|(worker_id, _)| *worker_id != 1)
        );
    }

    #[test]
    fn test_startup_cancel_ignores_stale_event() {
        let args = startup_args(5.0);
        let requests = simple_requests(20, 1000.0); // long enough to span startup
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        let ((), evidence) = super::super::evidence::with_runtime_evidence(
            crate::replay::ReplayCaptureOptions {
                capture_planner_details: true,
                ..Default::default()
            },
            || {
                // Scale up to 4 (2 new workers starting).
                rt.apply_scaling(4).unwrap();
                assert_eq!(rt.active_worker_count(), 2);
                assert_eq!(rt.total_worker_count(), 4);

                // Immediately scale back to 2 — should cancel both startup workers.
                rt.apply_scaling(2).unwrap();
                assert_eq!(rt.active_worker_count(), 2);
                assert_eq!(rt.total_worker_count(), 2);

                // Advance past the original startup time. No crash, counts unchanged.
                rt.advance_to(6000.0).unwrap();
                assert_eq!(rt.active_worker_count(), 2);
                assert_eq!(rt.total_worker_count(), 2);
            },
        );

        assert_eq!(evidence.lifecycle_operations.len(), 2);
        let startup = &evidence.lifecycle_operations[0];
        assert_eq!(startup.cause, "manual_scale");
        assert_eq!(startup.origin_operation_ordinal, None);
        assert_eq!(startup.state_after_batch.active, vec![0, 1]);
        assert_eq!(startup.state_after_batch.starting, vec![2, 3]);
        assert!(startup.state_after_batch.draining.is_empty());
        assert_eq!(
            startup
                .transitions
                .iter()
                .map(|transition| (
                    transition.worker_id,
                    transition.transition,
                    transition.reason
                ))
                .collect::<Vec<_>>(),
            vec![
                (2, WorkerLifecycleTransitionKind::WorkerStarting, None),
                (3, WorkerLifecycleTransitionKind::WorkerStarting, None),
            ]
        );
        let cancellation = &evidence.lifecycle_operations[1];
        assert_eq!(cancellation.cause, "manual_scale");
        assert_eq!(cancellation.origin_operation_ordinal, Some(0));
        assert_eq!(cancellation.state_after_batch.active, vec![0, 1]);
        assert!(cancellation.state_after_batch.starting.is_empty());
        assert!(cancellation.state_after_batch.draining.is_empty());
        assert_eq!(
            cancellation
                .transitions
                .iter()
                .map(|transition| (
                    transition.worker_id,
                    transition.transition,
                    transition.reason
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    3,
                    WorkerLifecycleTransitionKind::WorkerRemoved,
                    Some("startup_cancelled")
                ),
                (
                    2,
                    WorkerLifecycleTransitionKind::WorkerRemoved,
                    Some("startup_cancelled")
                ),
            ]
        );
    }

    #[test]
    fn test_advance_to_reports_done_when_workload_finishes_before_startup() {
        // Short trace (4 requests at 0-300ms) with a long startup delay.
        // The workload finishes well before the startup delay elapses.
        let args = startup_args(30.0); // 30s startup
        let requests = simple_requests(4, 100.0); // all done by ~400ms
        let mut rt = AggRuntime::new(
            &args,
            None,
            None,
            requests,
            1,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        // Scale up before requests arrive.
        rt.apply_scaling(2).unwrap();
        assert_eq!(rt.active_worker_count(), 1);

        // Advance well past all request completions but before startup.
        let done = rt.advance_to(10_000.0).unwrap();
        // Workload is done even though the WorkerReady event is at ~30000ms.
        assert!(
            done,
            "advance_to should report done when workload is complete"
        );
    }

    fn cap_request(uuid: u128, arrival_ms: f64) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; 64],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(arrival_ms),
            ..Default::default()
        }
    }

    /// Verifies that the cap operates on **simulated** time: with arrivals
    /// at 0/1/2/3/4 seconds of sim time and a 2.5s cap, the resulting
    /// simulated duration stays at or below the cap. Real wall-clock
    /// runtime is microseconds (speedup_ratio=1000).
    #[test]
    fn test_agg_multi_max_sim_time_truncates_run() {
        let args = fast_router_args();
        let submitted = 5;
        let cap_ms = 2500.0;
        let pending = VecDeque::from([
            cap_request(1, 0.0),
            cap_request(2, 1000.0),
            cap_request(3, 2000.0),
            cap_request(4, 3000.0),
            cap_request(5, 4000.0),
        ]);
        let (collector, _) = AggRuntime::new(
            &args,
            None,
            None,
            pending,
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_max_sim_time_ms(Some(cap_ms))
        .run()
        .unwrap();
        let report = collector.finish();
        assert!(
            report.request_counts.num_requests < submitted,
            "cap should admit fewer than {} requests; got num_requests={}",
            submitted,
            report.request_counts.num_requests
        );
        assert!(
            report.throughput.duration_ms <= cap_ms,
            "simulated duration must respect cap; got duration_ms={} cap_ms={}",
            report.throughput.duration_ms,
            cap_ms
        );
    }

    /// Sanity: uncapped, the same setup admits all requests and the
    /// simulated duration extends past the last arrival.
    #[test]
    fn test_agg_multi_no_cap_completes_everything() {
        let args = fast_router_args();
        let pending = VecDeque::from([
            cap_request(1, 0.0),
            cap_request(2, 1000.0),
            cap_request(3, 2000.0),
            cap_request(4, 3000.0),
            cap_request(5, 4000.0),
        ]);
        let (collector, _) = AggRuntime::new(
            &args,
            None,
            None,
            pending,
            2,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .run()
        .unwrap();
        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 5);
        assert_eq!(report.request_counts.num_requests, 5);
        assert!(
            report.throughput.duration_ms >= 4000.0,
            "uncapped sim duration should extend past last arrival; got {}",
            report.throughput.duration_ms
        );
    }
}
