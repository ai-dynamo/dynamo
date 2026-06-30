// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{Result, anyhow, bail};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{KvCacheEventData, LocalBlockHash, RouterEvent};
use uuid::Uuid;

pub(super) use super::components::ReplayMode;
#[cfg(test)]
use super::components::TrafficStats;
use super::components::{
    AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, OfflineReplayRouter,
    ReadyArrival, ScheduledWorkerCompletion, TrafficAccumulator, WorkerAdmission,
};
use super::events::{SimulationEvent, SimulationWorkerStage};
use super::planner_hook::{PlannerHook, PlannerTickMetrics};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    next_timestamp as choose_next_timestamp, pop_ready_planner_tick, pop_ready_transfer_complete,
    pop_ready_worker_completion, pop_ready_worker_ready, push_planner_tick, push_transfer_complete,
    push_worker_completion, push_worker_ready,
};
#[cfg(test)]
use super::state::DisaggRequestSnapshot;
use super::state::{DisaggPhase, DisaggRequestState};
use crate::common::handoff::{
    HandoffAction, HandoffActionOutcome, HandoffCompletion, HandoffFact, HandoffId, HandoffOrder,
    IssuedHandoffAction, NormalizedHandoffConformance, NormalizedHandoffEvent,
    NormalizedStoredTiming,
};
use crate::common::protocols::{
    DirectRequest, EngineType, ForwardPassSnapshot, MockEngineArgs, OutputSignal,
};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayTerminalStatus,
    SlaThresholds, TraceCollector,
};
use crate::scheduler::{
    AdmissionEvent, SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent,
};

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisaggTransition {
    PrefillMarkCompleted { uuid: Uuid },
    PrefillFree { uuid: Uuid },
    SourceHeld { uuid: Uuid },
    DestinationAccepted { uuid: Uuid },
    DestinationReserved { uuid: Uuid },
    TransferQueued { uuid: Uuid },
    DestinationActivated { uuid: Uuid },
    SourceReleased { uuid: Uuid },
    HandoffCompleted { uuid: Uuid },
    DecodeAdmitted { uuid: Uuid },
    DecodeFree { uuid: Uuid },
    RequestMarkedDone { uuid: Uuid },
    WorkloadCompleted { uuid: Uuid },
}

#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq)]
pub(in crate::replay) struct DisaggRuntimeStats {
    request_snapshots: HashMap<Uuid, DisaggRequestSnapshot>,
    prefill_assignments: HashMap<Uuid, usize>,
    decode_assignments: HashMap<Uuid, usize>,
    handoff_ms: HashMap<Uuid, f64>,
    prefill_marked_count: usize,
    prefill_router_freed_count: usize,
    decode_router_freed_count: usize,
    max_prefill_router_pending_count: usize,
    max_decode_router_pending_count: usize,
    transition_log: Vec<DisaggTransition>,
}

#[cfg(not(test))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(in crate::replay) struct DisaggRuntimeStats;

#[derive(Default)]
struct HandoffConformanceCapture {
    lifecycle: Vec<NormalizedHandoffEvent>,
    source_output_tokens: usize,
    stored_before_activation: usize,
    stored_on_activation: usize,
    activation_stored_hashes: HashSet<LocalBlockHash>,
    repeated_activation_hashes_after_activation: usize,
}

impl HandoffConformanceCapture {
    fn record_before_activation(&mut self, events: &[RouterEvent]) {
        self.stored_before_activation += stored_hashes(events).count();
    }

    fn record_activation(&mut self, events: &[RouterEvent]) {
        for hash in stored_hashes(events) {
            self.stored_on_activation += 1;
            self.activation_stored_hashes.insert(hash);
        }
    }

    fn record_after_activation(&mut self, events: &[RouterEvent]) {
        self.repeated_activation_hashes_after_activation += stored_hashes(events)
            .filter(|hash| self.activation_stored_hashes.contains(hash))
            .count();
    }
}

fn stored_hashes(events: &[RouterEvent]) -> impl Iterator<Item = LocalBlockHash> + '_ {
    events
        .iter()
        .flat_map(|event| match &event.event.data {
            KvCacheEventData::Stored(store) => store.blocks.as_slice(),
            KvCacheEventData::Removed(_) | KvCacheEventData::Cleared => &[],
        })
        .map(|block| block.tokens_hash)
}

enum ActionExecution {
    Applied,
    WaitingForWorker {
        action: IssuedHandoffAction,
        stage: SimulationWorkerStage,
    },
    Deferred {
        action: IssuedHandoffAction,
        stage: SimulationWorkerStage,
        worker_idx: usize,
    },
}

type QueuedHandoffAction = (Uuid, IssuedHandoffAction);

#[derive(Default)]
struct DisaggActionQueues {
    pending: VecDeque<QueuedHandoffAction>,
    waiting_prefill: VecDeque<QueuedHandoffAction>,
    waiting_decode: VecDeque<QueuedHandoffAction>,
    deferred_prefill: HashMap<usize, VecDeque<QueuedHandoffAction>>,
    deferred_decode: HashMap<usize, VecDeque<QueuedHandoffAction>>,
    queued_by_uuid: HashMap<Uuid, usize>,
}

impl DisaggActionQueues {
    fn enqueue_all(&mut self, uuid: Uuid, actions: impl IntoIterator<Item = IssuedHandoffAction>) {
        for action in actions {
            self.pending.push_back((uuid, action));
            self.increment(uuid);
        }
    }

    fn pop_pending(&mut self) -> Option<QueuedHandoffAction> {
        let action = self.pending.pop_front()?;
        self.decrement(action.0);
        Some(action)
    }

    fn wait_for_worker(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        stage: SimulationWorkerStage,
    ) {
        self.waiting_mut(stage).push_back((uuid, action));
        self.increment(uuid);
    }

    fn defer(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        stage: SimulationWorkerStage,
        worker_idx: usize,
    ) {
        self.deferred_mut(stage)
            .entry(worker_idx)
            .or_default()
            .push_back((uuid, action));
        self.increment(uuid);
    }

    fn wake_worker_waiters(&mut self, stage: SimulationWorkerStage) {
        let mut waiting = std::mem::take(self.waiting_mut(stage));
        self.pending.append(&mut waiting);
    }

    fn wake_deferred(&mut self, stage: SimulationWorkerStage, worker_idx: usize) {
        if let Some(actions) = self.deferred_mut(stage).remove(&worker_idx) {
            self.pending.extend(actions);
        }
    }

    fn remove(&mut self, uuid: Uuid) {
        self.pending.retain(|(action_uuid, _)| *action_uuid != uuid);
        self.waiting_prefill
            .retain(|(action_uuid, _)| *action_uuid != uuid);
        self.waiting_decode
            .retain(|(action_uuid, _)| *action_uuid != uuid);
        Self::remove_deferred(&mut self.deferred_prefill, uuid);
        Self::remove_deferred(&mut self.deferred_decode, uuid);
        self.queued_by_uuid.remove(&uuid);
    }

    fn contains(&self, uuid: Uuid) -> bool {
        self.queued_by_uuid.contains_key(&uuid)
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
            && self.waiting_prefill.is_empty()
            && self.waiting_decode.is_empty()
            && self.deferred_prefill.is_empty()
            && self.deferred_decode.is_empty()
    }

    fn waiting_mut(&mut self, stage: SimulationWorkerStage) -> &mut VecDeque<QueuedHandoffAction> {
        match stage {
            SimulationWorkerStage::Prefill => &mut self.waiting_prefill,
            SimulationWorkerStage::Decode => &mut self.waiting_decode,
            SimulationWorkerStage::Aggregated => {
                unreachable!("disagg action cannot target an aggregated worker")
            }
        }
    }

    fn deferred_mut(
        &mut self,
        stage: SimulationWorkerStage,
    ) -> &mut HashMap<usize, VecDeque<QueuedHandoffAction>> {
        match stage {
            SimulationWorkerStage::Prefill => &mut self.deferred_prefill,
            SimulationWorkerStage::Decode => &mut self.deferred_decode,
            SimulationWorkerStage::Aggregated => {
                unreachable!("disagg action cannot target an aggregated worker")
            }
        }
    }

    fn remove_deferred(deferred: &mut HashMap<usize, VecDeque<QueuedHandoffAction>>, uuid: Uuid) {
        for actions in deferred.values_mut() {
            actions.retain(|(action_uuid, _)| *action_uuid != uuid);
        }
        deferred.retain(|_, actions| !actions.is_empty());
    }

    fn increment(&mut self, uuid: Uuid) {
        let count = self.queued_by_uuid.entry(uuid).or_default();
        *count = count.checked_add(1).expect("queued action count overflow");
    }

    fn decrement(&mut self, uuid: Uuid) {
        let count = self
            .queued_by_uuid
            .get_mut(&uuid)
            .expect("pending action missing queued-action accounting");
        *count = count.checked_sub(1).expect("queued action count underflow");
        if *count == 0 {
            self.queued_by_uuid.remove(&uuid);
        }
    }
}

pub(in crate::replay) struct DisaggRuntime {
    now_ms: f64,
    next_prefill_worker_idx: usize,
    next_decode_worker_idx: usize,
    next_event_seq: u64,
    admission: AdmissionQueue,
    prefill_engine: EngineComponent,
    decode_engine: EngineComponent,
    prefill_router: Option<OfflineReplayRouter>,
    decode_router: Option<OfflineReplayRouter>,
    requests: HashMap<Uuid, DisaggRequestState>,
    requests_by_handoff: HashMap<HandoffId, Uuid>,
    handoff_order: HandoffOrder,
    prefill_block_size: usize,
    decode_block_size: usize,
    action_queues: DisaggActionQueues,
    logical_in_flight: usize,
    stale_transfer_events: usize,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    progress: ReplayProgress,
    stats: DisaggRuntimeStats,
    conformance_capture: Option<HandoffConformanceCapture>,
    /// Forward pass metrics accumulated between planner ticks, keyed by (stage, worker_idx).
    prefill_fpm_buffer: Vec<(usize, ForwardPassSnapshot)>,
    decode_fpm_buffer: Vec<(usize, ForwardPassSnapshot)>,
    /// Traffic statistics accumulated between planner ticks.
    traffic: TrafficAccumulator,
    /// Optional cap on simulated wall-clock time. When set, `run()` exits
    /// gracefully once the next scheduled timestamp exceeds this cap, leaving
    /// any in-flight requests as incomplete in the report.
    max_sim_time_ms: Option<f64>,
    /// Planner hook. When set, `run()` seeds a recurring `PlannerTick` event and
    /// calls back into the planner at each tick (this is the unified replacement
    /// for the old Python-driven `advance_to` stepping loop).
    planner_hook: Option<Box<dyn PlannerHook>>,
    /// Whether to retain per-pass FPM snapshots in the buffers above. Only the
    /// planner consumes them, so the plain `run()` path leaves this `false` —
    /// otherwise the buffers grow unbounded (one entry per worker pass) for the
    /// whole run with no reader (the memory leak this gating fixes).
    collect_fpm: bool,
}

impl DisaggRuntime {
    /// Create a disaggregated offline runtime seeded from an explicit request queue.
    pub(in crate::replay) fn new(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        pending: VecDeque<DirectRequest>,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        Self::new_with_source(
            config,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::new_requests(pending, mode),
            router_mode,
            false,
        )
    }

    /// Create a disaggregated offline runtime whose admissions come from a workload driver.
    pub(in crate::replay) fn new_workload(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        driver: WorkloadDriver,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        Self::new_with_source(
            config,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::new_workload(driver, mode),
            router_mode,
            false,
        )
    }

    /// Construct the deterministic one-request runtime used by cross-surface
    /// handoff conformance tests.
    pub(super) fn new_handoff_conformance(
        config: &OfflineDisaggReplayConfig,
        pending: VecDeque<DirectRequest>,
    ) -> Result<Self> {
        Self::new_with_source(
            config,
            None,
            None,
            AdmissionQueue::new_requests(pending, ReplayMode::Trace),
            ReplayRouterMode::RoundRobin,
            true,
        )
    }

    /// Shared constructor for both raw-request and workload-driven admissions.
    fn new_with_source(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        admission: AdmissionQueue,
        router_mode: ReplayRouterMode,
        capture_conformance: bool,
    ) -> Result<Self> {
        let handoff_order = match (
            config.prefill_args.engine_type,
            config.decode_args.engine_type,
        ) {
            (EngineType::Vllm, EngineType::Vllm) => HandoffOrder::SourceFirst,
            (EngineType::Sglang, EngineType::Sglang) => HandoffOrder::DestinationFirst,
            (EngineType::Trtllm, _) | (_, EngineType::Trtllm) => {
                bail!("offline disaggregated replay does not support TRT-LLM")
            }
            _ => bail!("offline disaggregated replay requires matching backend engine types"),
        };
        let progress = ReplayProgress::new(admission.total_requests(), "offline disagg replay");
        let (prefill_router, decode_router) = match router_mode {
            ReplayRouterMode::RoundRobin => (None, None),
            ReplayRouterMode::KvRouter => {
                let prefill_router_config =
                    derive_prefill_router_config(&config.prefill_args, router_config.clone());
                let decode_router_config =
                    derive_decode_router_config(&config.decode_args, router_config);
                (
                    Some(OfflineReplayRouter::new(
                        &config.prefill_args,
                        Some(prefill_router_config),
                        prefill_load_estimator,
                        config.num_prefill_workers,
                    )?),
                    Some(OfflineReplayRouter::new(
                        &config.decode_args,
                        Some(decode_router_config),
                        None,
                        config.num_decode_workers,
                    )?),
                )
            }
        };

        let prefill_capture_kv = prefill_router.is_some();
        let mut prefill_engine = EngineComponent::new(
            SimulationWorkerStage::Prefill,
            EnginePassMode::Hidden,
            (0..config.num_prefill_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.prefill_args.clone(),
                        prefill_capture_kv,
                    )
                })
                .collect(),
        );
        prefill_engine.set_scaling_args(config.prefill_args.clone(), prefill_capture_kv);
        let mut decode_engine = EngineComponent::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            (0..config.num_decode_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.decode_args.clone(),
                        capture_conformance,
                    )
                })
                .collect(),
        );
        decode_engine.set_scaling_args(config.decode_args.clone(), capture_conformance);

        // Record each pool's GPUs/worker from its engine parallelism so the
        // report can express GPU-hours from the mocker's own config.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(
            config.prefill_args.aic_gpus_per_worker(),
            config.decode_args.aic_gpus_per_worker(),
        );

        Ok(Self {
            now_ms: 0.0,
            next_prefill_worker_idx: 0,
            next_decode_worker_idx: 0,
            next_event_seq: 0,
            admission,
            prefill_engine,
            decode_engine,
            prefill_router,
            decode_router,
            requests: HashMap::new(),
            requests_by_handoff: HashMap::new(),
            handoff_order,
            prefill_block_size: config.prefill_args.block_size,
            decode_block_size: config.decode_args.block_size,
            action_queues: DisaggActionQueues::default(),
            logical_in_flight: 0,
            stale_transfer_events: 0,
            collector,
            events: BinaryHeap::new(),
            progress,
            #[cfg(test)]
            stats: DisaggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: DisaggRuntimeStats,
            conformance_capture: capture_conformance.then(HandoffConformanceCapture::default),
            prefill_fpm_buffer: Vec::new(),
            decode_fpm_buffer: Vec::new(),
            traffic: TrafficAccumulator::new(),
            max_sim_time_ms: None,
            planner_hook: None,
            collect_fpm: false,
        })
    }

    /// Toggle per-request record capture on the underlying collector. When
    /// `true`, the final `TraceSimulationReport` returned from `run()` will
    /// have `per_request` populated. Default `false` (cheap).
    pub(in crate::replay) fn with_per_request_records(mut self, capture: bool) -> Self {
        self.collector.set_capture_per_request(capture);
        self
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

    /// Set the SLA thresholds used to classify goodput in the final report.
    pub(in crate::replay) fn with_sla_thresholds(mut self, sla: SlaThresholds) -> Self {
        self.collector.set_sla_thresholds(sla);
        self
    }

    /// Attach a planner hook. Enables FPM collection and makes `run()` drive the
    /// planner via recurring `PlannerTick` events (one `on_tick` callback per tick).
    pub(in crate::replay) fn with_planner_hook(mut self, hook: Box<dyn PlannerHook>) -> Self {
        self.collect_fpm = true;
        self.planner_hook = Some(hook);
        self
    }

    #[cfg(test)]
    fn with_fpm_capture(mut self) -> Self {
        self.collect_fpm = true;
        self
    }

    /// Count all requests consuming cluster capacity across prefill, decode, and router queues.
    fn cluster_in_flight(&self) -> usize {
        self.logical_in_flight
    }

    /// Pick the next active prefill worker in round-robin order.
    fn next_prefill_worker(&mut self) -> usize {
        let active = self.prefill_engine.active_worker_ids();
        debug_assert!(
            !active.is_empty(),
            "no active prefill workers for round-robin"
        );
        let idx = self.next_prefill_worker_idx % active.len();
        self.next_prefill_worker_idx = idx + 1;
        active[idx]
    }

    /// Pick the next active decode worker in round-robin order.
    fn next_decode_worker(&mut self) -> usize {
        let active = self.decode_engine.active_worker_ids();
        debug_assert!(
            !active.is_empty(),
            "no active decode workers for round-robin"
        );
        let idx = self.next_decode_worker_idx % active.len();
        self.next_decode_worker_idx = idx + 1;
        active[idx]
    }

    /// Track the peak number of requests parked in each stage router.
    fn record_router_pending(&mut self) {
        #[cfg(test)]
        {
            self.stats.max_prefill_router_pending_count =
                self.stats.max_prefill_router_pending_count.max(
                    self.prefill_router
                        .as_ref()
                        .map_or(0, OfflineReplayRouter::pending_count),
                );
            self.stats.max_decode_router_pending_count =
                self.stats.max_decode_router_pending_count.max(
                    self.decode_router
                        .as_ref()
                        .map_or(0, OfflineReplayRouter::pending_count),
                );
        }
    }

    /// Borrow immutable request state with a structured missing-request error.
    fn state(&self, uuid: Uuid) -> Result<&DisaggRequestState> {
        self.requests
            .get(&uuid)
            .ok_or_else(|| anyhow!("offline disagg replay missing request state for {uuid}"))
    }

    /// Borrow mutable request state with a structured missing-request error.
    fn state_mut(&mut self, uuid: Uuid) -> Result<&mut DisaggRequestState> {
        self.requests
            .get_mut(&uuid)
            .ok_or_else(|| anyhow!("offline disagg replay missing request state for {uuid}"))
    }

    fn acknowledge_action(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        outcome: HandoffActionOutcome,
    ) -> Result<()> {
        if matches!(outcome, HandoffActionOutcome::Failed(_)) {
            self.collector
                .on_terminal(uuid, ReplayTerminalStatus::Failed);
        }
        let actions = self
            .state_mut(uuid)?
            .coordinator
            .on_action_outcome(action.id, outcome)?;
        self.action_queues.enqueue_all(uuid, actions);
        Ok(())
    }

    fn apply_handoff_fact(&mut self, uuid: Uuid, fact: HandoffFact) -> Result<()> {
        let terminal_status = match fact {
            HandoffFact::Failed { .. } | HandoffFact::TimedOut { .. } => {
                Some(ReplayTerminalStatus::Failed)
            }
            HandoffFact::Canceled { .. } => Some(ReplayTerminalStatus::Canceled),
            _ => None,
        };
        if let Some(status) = terminal_status {
            self.collector.on_terminal(uuid, status);
        }
        let actions = self.state_mut(uuid)?.coordinator.on_fact(fact)?;
        self.action_queues.enqueue_all(uuid, actions);
        Ok(())
    }

    /// Submit a coordinator-owned prefill onto a selected worker.
    fn dispatch_prefill(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
    ) -> Result<()> {
        let request = self.state(uuid)?.build_prefill_request()?;
        let handoff_id = self.state(uuid)?.handoff_id;
        let effects = match self.prefill_engine.apply_command(
            worker_idx,
            SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request,
            },
        ) {
            Ok(effects) => effects,
            Err(error) => {
                self.acknowledge_action(
                    uuid,
                    action,
                    HandoffActionOutcome::Failed(error.to_string()),
                )?;
                return Ok(());
            }
        };
        if !matches!(effects.result, SchedulerCommandResult::Submitted(id) if id == uuid) {
            bail!("offline disagg replay prefill submission returned an unexpected result");
        }
        self.state_mut(uuid)?.start_prefill(worker_idx);
        self.collector.on_prefill_assigned(uuid, worker_idx);
        #[cfg(test)]
        {
            self.stats.prefill_assignments.insert(uuid, worker_idx);
        }
        self.acknowledge_action(uuid, action, HandoffActionOutcome::Submitted)?;
        self.process_lifecycle_events(effects.lifecycle_events)?;
        Ok(())
    }

    /// Accept destination ownership on a selected decode worker.
    fn reserve_destination(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
    ) -> Result<()> {
        let request = self.state(uuid)?.original_request()?.clone();
        let handoff_id = self.state(uuid)?.handoff_id;
        let effects = match self.decode_engine.apply_command(
            worker_idx,
            SchedulerCommand::ReserveDestination {
                handoff_id,
                request,
            },
        ) {
            Ok(effects) => effects,
            Err(error) => {
                self.acknowledge_action(
                    uuid,
                    action,
                    HandoffActionOutcome::Failed(error.to_string()),
                )?;
                return Ok(());
            }
        };
        if !matches!(
            effects.result,
            SchedulerCommandResult::DestinationAccepted { request_id } if request_id == uuid
        ) {
            bail!("offline disagg replay destination acceptance returned an unexpected result");
        }
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture.record_before_activation(&effects.kv_events);
            capture
                .lifecycle
                .push(NormalizedHandoffEvent::DestinationAccepted);
        }
        self.state_mut(uuid)?.assign_decode(worker_idx);
        self.collector.on_decode_assigned(uuid, worker_idx);
        #[cfg(test)]
        {
            self.stats.decode_assignments.insert(uuid, worker_idx);
            self.stats
                .transition_log
                .push(DisaggTransition::DestinationAccepted { uuid });
        }
        self.acknowledge_action(uuid, action, HandoffActionOutcome::Accepted)?;
        self.process_lifecycle_events(effects.lifecycle_events)?;
        Ok(())
    }

    /// Turn prefill router admissions into concrete worker dispatches.
    fn dispatch_prefill_admissions(&mut self, admissions: Vec<WorkerAdmission>) -> Result<()> {
        for WorkerAdmission {
            uuid,
            worker_idx,
            overlap_blocks,
            isl_blocks,
        } in admissions
        {
            self.traffic.on_admission(overlap_blocks, isl_blocks);
            let input_tokens = self.state(uuid)?.original_request()?.tokens.len();
            let overlap_tokens =
                (overlap_blocks as usize * self.prefill_block_size).min(input_tokens);
            self.collector
                .on_prefill_route_overlap(uuid, overlap_tokens);
            if self.state(uuid)?.phase != DisaggPhase::QueuedPrefill {
                bail!("offline disagg replay expected queued prefill request for {uuid}");
            }
            let action = self
                .state_mut(uuid)?
                .pending_prefill_action
                .take()
                .ok_or_else(|| anyhow!("missing coordinator prefill action for {uuid}"))?;
            self.dispatch_prefill(uuid, worker_idx, action)?;
        }
        Ok(())
    }

    /// Turn decode-router admissions into destination ownership commands.
    fn dispatch_decode_admissions(&mut self, admissions: Vec<WorkerAdmission>) -> Result<()> {
        for WorkerAdmission {
            uuid,
            worker_idx,
            overlap_blocks,
            ..
        } in admissions
        {
            let input_tokens = self.state(uuid)?.original_request()?.tokens.len();
            let overlap_tokens =
                (overlap_blocks as usize * self.decode_block_size).min(input_tokens);
            self.collector.on_decode_route_overlap(uuid, overlap_tokens);
            if self.state(uuid)?.phase != DisaggPhase::AwaitingDestination {
                bail!("offline disagg replay expected destination-waiting request for {uuid}");
            }
            let action = self
                .state_mut(uuid)?
                .pending_destination_action
                .take()
                .ok_or_else(|| anyhow!("missing coordinator destination action for {uuid}"))?;
            self.reserve_destination(uuid, worker_idx, action)?;
        }
        Ok(())
    }

    fn route_prefill(&mut self, uuid: Uuid, action: IssuedHandoffAction) -> Result<()> {
        self.state_mut(uuid)?.phase = DisaggPhase::QueuedPrefill;
        if self.prefill_router.is_none() {
            self.collector.on_prefill_route_overlap(uuid, 0);
            let worker_idx = self.next_prefill_worker();
            return self.dispatch_prefill(uuid, worker_idx, action);
        }
        {
            let state = self.state_mut(uuid)?;
            state.pending_prefill_action = Some(action);
            state.prefill_routed = true;
        }
        let request = self.state(uuid)?.build_prefill_request()?;
        let replay_hashes = self.state_mut(uuid)?.take_replay_hashes();
        let admissions = self
            .prefill_router
            .as_mut()
            .expect("prefill router presence checked above")
            .on_request_arrival(&request, replay_hashes, self.now_ms)?
            .admissions;
        self.record_router_pending();
        self.dispatch_prefill_admissions(admissions)
    }

    fn route_destination(&mut self, uuid: Uuid, action: IssuedHandoffAction) -> Result<()> {
        self.state_mut(uuid)?.await_destination();
        if self.decode_router.is_none() {
            self.collector.on_decode_route_overlap(uuid, 0);
            let worker_idx = self.next_decode_worker();
            return self.reserve_destination(uuid, worker_idx, action);
        }
        {
            let state = self.state_mut(uuid)?;
            state.pending_destination_action = Some(action);
            state.destination_routed = true;
        }
        let request = self.state(uuid)?.original_request()?.clone();
        let admissions = self
            .decode_router
            .as_mut()
            .expect("decode router presence checked above")
            .on_request_arrival(&request, None, self.now_ms)?
            .admissions;
        self.record_router_pending();
        self.dispatch_decode_admissions(admissions)?;
        Ok(())
    }

    fn drive_pending_actions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some((uuid, action)) = self.action_queues.pop_pending() {
            match self.execute_action(uuid, action)? {
                ActionExecution::Applied => {
                    changed = true;
                }
                ActionExecution::WaitingForWorker { action, stage } => {
                    self.action_queues.wait_for_worker(uuid, action, stage);
                }
                ActionExecution::Deferred {
                    action,
                    stage,
                    worker_idx,
                } => {
                    self.action_queues.defer(uuid, action, stage, worker_idx);
                }
            }
        }
        Ok(changed)
    }

    fn wake_worker_waiters(&mut self, stage: SimulationWorkerStage) {
        self.action_queues.wake_worker_waiters(stage);
    }

    fn wake_deferred_actions(&mut self, stage: SimulationWorkerStage, worker_idx: usize) {
        self.action_queues.wake_deferred(stage, worker_idx);
    }

    fn execute_action(
        &mut self,
        uuid: Uuid,
        issued: IssuedHandoffAction,
    ) -> Result<ActionExecution> {
        match issued.action {
            HandoffAction::SubmitPrefill { .. } => {
                if !self.prefill_engine.has_active_workers() {
                    return Ok(ActionExecution::WaitingForWorker {
                        action: issued,
                        stage: SimulationWorkerStage::Prefill,
                    });
                }
                self.route_prefill(uuid, issued)?;
            }
            HandoffAction::ReserveDestination { .. } => {
                if !self.decode_engine.has_active_workers() {
                    return Ok(ActionExecution::WaitingForWorker {
                        action: issued,
                        stage: SimulationWorkerStage::Decode,
                    });
                }
                self.route_destination(uuid, issued)?;
            }
            HandoffAction::StartTransfer { delay_ms, .. } => {
                self.acknowledge_action(uuid, issued, HandoffActionOutcome::Scheduled)?;
                self.state_mut(uuid)?.transfer_pending();
                if delay_ms > 0.0 {
                    let handoff_id = self.state(uuid)?.handoff_id;
                    push_transfer_complete(
                        &mut self.events,
                        &mut self.next_event_seq,
                        self.now_ms + delay_ms,
                        handoff_id,
                    );
                    #[cfg(test)]
                    self.stats
                        .transition_log
                        .push(DisaggTransition::TransferQueued { uuid });
                } else {
                    let handoff_id = self.state(uuid)?.handoff_id;
                    self.apply_handoff_fact(uuid, HandoffFact::TransferCompleted { handoff_id })?;
                }
            }
            HandoffAction::ActivateDestination { handoff_id } => {
                let worker_idx = self
                    .state(uuid)?
                    .decode_worker_idx()
                    .ok_or_else(|| anyhow!("destination activation has no worker for {uuid}"))?;
                if self.decode_engine.worker_is_busy(worker_idx)? {
                    return Ok(ActionExecution::Deferred {
                        action: issued,
                        stage: SimulationWorkerStage::Decode,
                        worker_idx,
                    });
                }
                let effects = self.decode_engine.apply_command(
                    worker_idx,
                    SchedulerCommand::ActivateDestination { handoff_id },
                )?;
                if effects.result != SchedulerCommandResult::Applied {
                    self.acknowledge_action(
                        uuid,
                        issued,
                        HandoffActionOutcome::Failed(
                            "destination activation was not applied".to_string(),
                        ),
                    )?;
                    return Ok(ActionExecution::Applied);
                }
                if let Some(capture) = self.conformance_capture.as_mut() {
                    capture.record_activation(&effects.kv_events);
                    capture
                        .lifecycle
                        .push(NormalizedHandoffEvent::DestinationActivated);
                }
                self.state_mut(uuid)?.ready_decode();
                self.collector.on_destination_activated(uuid, self.now_ms);
                #[cfg(test)]
                self.stats
                    .transition_log
                    .push(DisaggTransition::DestinationActivated { uuid });
                self.acknowledge_action(uuid, issued, HandoffActionOutcome::Applied)?;
                self.process_lifecycle_events(effects.lifecycle_events)?;
            }
            HandoffAction::ReleaseSource { handoff_id } => {
                let worker_idx = self
                    .state(uuid)?
                    .prefill_worker_idx()
                    .ok_or_else(|| anyhow!("source release has no worker for {uuid}"))?;
                if self.prefill_engine.worker_is_busy(worker_idx)? {
                    return Ok(ActionExecution::Deferred {
                        action: issued,
                        stage: SimulationWorkerStage::Prefill,
                        worker_idx,
                    });
                }
                let effects = self
                    .prefill_engine
                    .apply_command(worker_idx, SchedulerCommand::ReleaseSource { handoff_id })?;
                let outcome = match effects.result {
                    SchedulerCommandResult::Applied => HandoffActionOutcome::Applied,
                    SchedulerCommandResult::Noop => HandoffActionOutcome::Noop,
                    _ => bail!("source release returned an unexpected result"),
                };
                #[cfg(test)]
                self.stats
                    .transition_log
                    .push(DisaggTransition::SourceReleased { uuid });
                if let Some(capture) = self.conformance_capture.as_mut() {
                    capture
                        .lifecycle
                        .push(NormalizedHandoffEvent::SourceReleased);
                }
                self.apply_prefill_router_events(effects.kv_events)?;
                self.collector.on_source_released(uuid, self.now_ms);
                self.acknowledge_action(uuid, issued, outcome)?;
                self.process_lifecycle_events(effects.lifecycle_events)?;
            }
            HandoffAction::CancelSource { handoff_id } => {
                let Some(worker_idx) = self.state(uuid)?.prefill_worker_idx() else {
                    self.cancel_prefill_route(uuid)?;
                    self.acknowledge_action(uuid, issued, HandoffActionOutcome::Noop)?;
                    return Ok(ActionExecution::Applied);
                };
                if self.prefill_engine.worker_is_busy(worker_idx)? {
                    return Ok(ActionExecution::Deferred {
                        action: issued,
                        stage: SimulationWorkerStage::Prefill,
                        worker_idx,
                    });
                }
                let effects = self
                    .prefill_engine
                    .apply_command(worker_idx, SchedulerCommand::CancelSource { handoff_id })?;
                let outcome = command_cleanup_outcome(effects.result)?;
                self.apply_prefill_router_events(effects.kv_events)?;
                self.acknowledge_action(uuid, issued, outcome)?;
                self.process_lifecycle_events(effects.lifecycle_events)?;
            }
            HandoffAction::CancelDestination { handoff_id } => {
                let Some(worker_idx) = self.state(uuid)?.decode_worker_idx() else {
                    self.cancel_decode_route(uuid)?;
                    self.acknowledge_action(uuid, issued, HandoffActionOutcome::Noop)?;
                    return Ok(ActionExecution::Applied);
                };
                if self.decode_engine.worker_is_busy(worker_idx)? {
                    return Ok(ActionExecution::Deferred {
                        action: issued,
                        stage: SimulationWorkerStage::Decode,
                        worker_idx,
                    });
                }
                let effects = self.decode_engine.apply_command(
                    worker_idx,
                    SchedulerCommand::CancelDestination { handoff_id },
                )?;
                let outcome = command_cleanup_outcome(effects.result)?;
                self.acknowledge_action(uuid, issued, outcome)?;
                self.process_lifecycle_events(effects.lifecycle_events)?;
            }
            HandoffAction::Complete { .. } => self.complete_handoff(uuid)?,
        }
        Ok(ActionExecution::Applied)
    }

    fn process_lifecycle_events(&mut self, events: Vec<SchedulerLifecycleEvent>) -> Result<()> {
        for event in events {
            match event {
                SchedulerLifecycleEvent::SourceHeld {
                    handoff_id,
                    request_id,
                    transfer_timing,
                } => {
                    let uuid = self.uuid_for_handoff(handoff_id)?;
                    if uuid != request_id {
                        bail!("source lifecycle request ID does not match its handoff");
                    }
                    #[cfg(test)]
                    self.stats
                        .transition_log
                        .push(DisaggTransition::SourceHeld { uuid });
                    if let Some(capture) = self.conformance_capture.as_mut() {
                        capture.lifecycle.push(NormalizedHandoffEvent::SourceHeld);
                    }
                    self.collector.on_source_held(uuid, self.now_ms);
                    self.apply_handoff_fact(
                        uuid,
                        HandoffFact::SourceHeld {
                            handoff_id,
                            transfer_timing,
                        },
                    )?;
                }
                SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id,
                    request_id,
                    transferable_prompt_tokens,
                } => {
                    let uuid = self.uuid_for_handoff(handoff_id)?;
                    if uuid != request_id {
                        bail!("destination lifecycle request ID does not match its handoff");
                    }
                    #[cfg(test)]
                    self.stats
                        .transition_log
                        .push(DisaggTransition::DestinationReserved { uuid });
                    if let Some(capture) = self.conformance_capture.as_mut() {
                        capture
                            .lifecycle
                            .push(NormalizedHandoffEvent::DestinationReserved);
                    }
                    self.collector.on_destination_reserved(uuid, self.now_ms);
                    self.apply_handoff_fact(
                        uuid,
                        HandoffFact::DestinationReserved {
                            handoff_id,
                            transferable_prompt_tokens,
                        },
                    )?;
                }
            }
        }
        Ok(())
    }

    fn uuid_for_handoff(&self, handoff_id: HandoffId) -> Result<Uuid> {
        self.requests_by_handoff
            .get(&handoff_id)
            .copied()
            .ok_or_else(|| anyhow!("offline disagg replay missing handoff {handoff_id:?}"))
    }

    fn cancel_prefill_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.prefill_routed {
            return Ok(());
        }
        self.state_mut(uuid)?.pending_prefill_action = None;
        let router = self
            .prefill_router
            .as_mut()
            .expect("prefill route flag requires a router");
        let admissions = if router.cancel_pending(uuid) {
            Vec::new()
        } else {
            router.on_request_completed(uuid, self.now_ms)?.admissions
        };
        self.state_mut(uuid)?.prefill_routed = false;
        self.record_router_pending();
        self.dispatch_prefill_admissions(admissions)
    }

    fn cancel_decode_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.destination_routed {
            return Ok(());
        }
        self.state_mut(uuid)?.pending_destination_action = None;
        let router = self
            .decode_router
            .as_mut()
            .expect("destination route flag requires a router");
        let admissions = if router.cancel_pending(uuid) {
            Vec::new()
        } else {
            router.on_request_completed(uuid, self.now_ms)?.admissions
        };
        self.state_mut(uuid)?.destination_routed = false;
        self.record_router_pending();
        self.dispatch_decode_admissions(admissions)
    }

    fn complete_prefill_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.prefill_routed {
            return Ok(());
        }
        let admissions = self
            .prefill_router
            .as_mut()
            .expect("prefill route flag requires a router")
            .on_request_completed(uuid, self.now_ms)?
            .admissions;
        self.state_mut(uuid)?.prefill_routed = false;
        #[cfg(test)]
        {
            self.stats.prefill_router_freed_count += 1;
            self.stats
                .transition_log
                .push(DisaggTransition::PrefillFree { uuid });
        }
        self.record_router_pending();
        self.dispatch_prefill_admissions(admissions)
    }

    fn complete_handoff(&mut self, uuid: Uuid) -> Result<()> {
        match self.state(uuid)?.coordinator.completion() {
            Some(HandoffCompletion::Success) => {
                self.complete_prefill_route(uuid)?;
                #[cfg(test)]
                {
                    self.stats.handoff_ms.insert(uuid, self.now_ms);
                    self.stats
                        .transition_log
                        .push(DisaggTransition::HandoffCompleted { uuid });
                }
                if let Some(capture) = self.conformance_capture.as_mut() {
                    capture.lifecycle.push(NormalizedHandoffEvent::Completed);
                }
                self.retire_completed_request(uuid)?;
            }
            Some(HandoffCompletion::Canceled) => {
                self.collector
                    .on_terminal(uuid, ReplayTerminalStatus::Canceled);
                self.cancel_prefill_route(uuid)?;
                self.cancel_decode_route(uuid)?;
                self.finish_logical_request(uuid, true)?;
            }
            None => bail!("handoff completed without a terminal coordinator outcome"),
        }
        Ok(())
    }

    fn finish_logical_request(&mut self, uuid: Uuid, remove_actions: bool) -> Result<()> {
        let transfer_was_pending = {
            let state = self.state_mut(uuid)?;
            if !state.counted_in_flight || state.phase == DisaggPhase::Done {
                bail!("offline disagg replay finalized request {uuid} more than once");
            }
            let transfer_was_pending = state.phase == DisaggPhase::TransferPending;
            state.counted_in_flight = false;
            state.complete_decode();
            transfer_was_pending
        };
        self.logical_in_flight = self
            .logical_in_flight
            .checked_sub(1)
            .expect("logical in-flight request count underflow");
        if transfer_was_pending {
            self.stale_transfer_events = self
                .stale_transfer_events
                .checked_add(1)
                .expect("stale transfer event count overflow");
        }
        if remove_actions {
            self.action_queues.remove(uuid);
        }
        self.admission.on_request_completed(uuid, self.now_ms)?;
        self.progress.inc_completed();
        #[cfg(test)]
        {
            self.stats
                .transition_log
                .push(DisaggTransition::RequestMarkedDone { uuid });
            if self.admission.is_workload() {
                self.stats
                    .transition_log
                    .push(DisaggTransition::WorkloadCompleted { uuid });
            }
        }
        self.retire_completed_request(uuid)?;
        Ok(())
    }

    fn retire_completed_request(&mut self, uuid: Uuid) -> Result<()> {
        let ready = {
            let state = self.state(uuid)?;
            !state.counted_in_flight && state.coordinator.is_complete()
        };
        if !ready {
            return Ok(());
        }
        if self.action_queues.contains(uuid) {
            bail!("offline disagg replay completed handoff still has queued actions for {uuid}");
        }

        let handoff_id = self.state(uuid)?.handoff_id;
        self.state_mut(uuid)?.mark_done();
        let removed = self.requests_by_handoff.remove(&handoff_id);
        if removed != Some(uuid) {
            bail!("offline disagg replay handoff index is inconsistent for {uuid}");
        }
        Ok(())
    }

    /// Admit one external request into prefill-side state, collector state, and optional router.
    fn on_external_arrival(
        &mut self,
        mut request: DirectRequest,
        arrival_time_ms: f64,
        replay_hashes: Option<ReplayRequestHashes>,
    ) -> Result<Uuid> {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        request.uuid = Some(uuid);
        request.arrival_timestamp_ms = Some(arrival_time_ms);

        self.collector.on_arrival(
            uuid,
            arrival_time_ms,
            request.tokens.len(),
            request.max_output_tokens,
        );
        if self.requests.contains_key(&uuid) {
            bail!("offline disagg replay request {uuid} is already active");
        }
        let handoff_id = HandoffId::new();
        let mut state = DisaggRequestState::new(
            request,
            arrival_time_ms,
            handoff_id,
            self.handoff_order,
            replay_hashes,
        );
        let actions = state.coordinator.start()?;
        self.requests.insert(uuid, state);
        self.requests_by_handoff.insert(handoff_id, uuid);
        self.logical_in_flight = self
            .logical_in_flight
            .checked_add(1)
            .expect("logical in-flight request count overflow");
        self.action_queues.enqueue_all(uuid, actions);
        Ok(uuid)
    }

    /// Return true once both stages, both routers, and all admissions are fully
    /// drained. Lingering `WorkerReady`/`PlannerTick` events (worker startup, a
    /// re-armed planner heartbeat) do not represent request work, so they do not
    /// keep the run alive — otherwise a recurring tick would never let `run()` exit.
    fn is_done(&self) -> bool {
        self.only_idle_events_remain()
            && self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.prefill_engine.is_drained()
            && self.decode_engine.is_drained()
            && self.action_queues.is_empty()
            && self.requests_by_handoff.is_empty()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// or `PlannerTick` events remain in the queue.
    fn is_workload_done(&self) -> bool {
        self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.prefill_engine.is_drained()
            && self.decode_engine.is_drained()
            && self.action_queues.is_empty()
            && self.requests_by_handoff.is_empty()
            && self.only_idle_events_remain()
    }

    /// True if the event heap is empty or contains only "idle" events that carry no
    /// pending request work: `WorkerReady` (a worker still starting up) or
    /// `PlannerTick` (a re-armed planner heartbeat).
    fn only_idle_events_remain(&self) -> bool {
        use super::events::SimulationEventKind;
        self.events.iter().all(|e| {
            matches!(
                e.kind,
                SimulationEventKind::WorkerReady { .. } | SimulationEventKind::PlannerTick
            )
        })
    }

    /// Pick the next logical timestamp from arrivals, worker completions, or decode handoffs.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let next = choose_next_timestamp(self.admission.next_ready_time_ms(), next_event_ms);
        #[cfg(feature = "kvbm-offload")]
        {
            let next_offload = choose_next_timestamp(
                self.prefill_engine.earliest_offload_deadline(),
                self.decode_engine.earliest_offload_deadline(),
            );
            return choose_next_timestamp(next, next_offload);
        }
        #[cfg(not(feature = "kvbm-offload"))]
        next
    }

    /// Apply prefill-side KV router events at the scheduler-selected visibility phase.
    fn apply_prefill_router_events(&mut self, events: Vec<RouterEvent>) -> Result<()> {
        let Some(prefill_router) = self.prefill_router.as_mut() else {
            return Ok(());
        };
        let effects = prefill_router.on_kv_events(events)?;
        if !effects.admissions.is_empty() {
            bail!("offline disagg replay prefill KV events must not admit requests");
        }
        Ok(())
    }

    #[cfg(feature = "kvbm-offload")]
    fn tick_offload_engines(&mut self) -> Result<bool> {
        let prefill = self.prefill_engine.tick_offload_engines(self.now_ms);
        let decode = self.decode_engine.tick_offload_engines(self.now_ms);
        let changed = !prefill.kv_events.is_empty()
            || !decode.kv_events.is_empty()
            || !prefill.lifecycle_events.is_empty()
            || !decode.lifecycle_events.is_empty();
        self.apply_prefill_router_events(prefill.kv_events)?;
        if !decode.kv_events.is_empty() {
            tracing::debug!(
                events = decode.kv_events.len(),
                "offline disagg replay dropping decode-side offload router events"
            );
        }
        self.process_lifecycle_events(prefill.lifecycle_events)?;
        self.process_lifecycle_events(decode.lifecycle_events)?;
        Ok(changed)
    }

    /// Process one prefill output signal, including router updates and decode handoff scheduling.
    fn process_prefill_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.rejected
            && let Some(capture) = self.conformance_capture.as_mut()
        {
            capture.source_output_tokens += 1;
        }
        if !signal.completed {
            return Ok(());
        }

        if signal.rejected {
            let handoff_id = self.state(signal.uuid)?.handoff_id;
            self.collector
                .on_terminal(signal.uuid, ReplayTerminalStatus::Rejected);
            return self.apply_handoff_fact(signal.uuid, HandoffFact::Failed { handoff_id });
        }

        if self.prefill_router.is_some() {
            let prefill_complete_admissions = {
                let prefill_router = self.prefill_router.as_mut().expect("router checked above");
                prefill_router
                    .on_prefill_completed(signal.uuid, self.now_ms)?
                    .admissions
            };
            #[cfg(test)]
            {
                self.stats.prefill_marked_count += 1;
                self.stats
                    .transition_log
                    .push(DisaggTransition::PrefillMarkCompleted { uuid: signal.uuid });
            }
            self.record_router_pending();
            self.dispatch_prefill_admissions(prefill_complete_admissions)?;
        }
        Ok(())
    }

    /// Process one decode output signal, including decode router frees and request completion.
    fn process_decode_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.completed {
            return Ok(());
        }

        let admissions = if let Some(decode_router) = self.decode_router.as_mut() {
            let admissions = decode_router
                .on_request_completed(signal.uuid, self.now_ms)?
                .admissions;
            self.state_mut(signal.uuid)?.destination_routed = false;
            #[cfg(test)]
            {
                self.stats.decode_router_freed_count += 1;
                self.stats
                    .transition_log
                    .push(DisaggTransition::DecodeFree { uuid: signal.uuid });
            }
            admissions
        } else {
            Vec::new()
        };
        self.record_router_pending();
        // A request rejected at decode never ran, so it produced no tokens or
        // latency — keep it out of the planner-facing traffic deltas (mirror the
        // aggregated path). It still frees its slot, advances, and is marked done.
        if !signal.rejected {
            let state = self.state(signal.uuid)?;
            let original = state.original_request()?;
            let input_tokens = original.tokens.len();
            let output_tokens = original.max_output_tokens;
            let latencies = self.collector.request_latencies(signal.uuid);
            self.traffic
                .on_request(input_tokens, output_tokens, latencies);
        }
        let terminal_status = if signal.rejected {
            ReplayTerminalStatus::Rejected
        } else {
            ReplayTerminalStatus::Completed
        };
        self.collector.on_terminal(signal.uuid, terminal_status);
        self.finish_logical_request(signal.uuid, false)?;
        self.dispatch_decode_admissions(admissions)?;
        Ok(())
    }

    /// Apply the side effects of a finished prefill pass.
    fn process_prefill_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        kv_events: Vec<RouterEvent>,
    ) -> Result<()> {
        self.apply_prefill_router_events(kv_events)?;
        for signal in output_signals {
            self.process_prefill_signal(signal)?;
        }
        self.process_lifecycle_events(lifecycle_events)?;
        Ok(())
    }

    /// Apply the side effects of a finished decode pass.
    fn process_decode_pass(
        &mut self,
        output_signals: Vec<OutputSignal>,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        kv_events: Vec<RouterEvent>,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    ) -> Result<()> {
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture.record_after_activation(&kv_events);
        }
        self.traffic
            .on_accept_length_sample(accept_length_output_tokens, accept_length_decode_forwards);
        for signal in output_signals {
            self.process_decode_signal(signal)?;
        }
        self.process_lifecycle_events(lifecycle_events)?;
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some(payload) = pop_ready_worker_completion(&mut self.events, self.now_ms) {
            match payload.stage {
                SimulationWorkerStage::Prefill => {
                    let payload = self.prefill_engine.on_scheduled_completion(payload)?;
                    self.wake_deferred_actions(SimulationWorkerStage::Prefill, payload.worker_idx);
                    if self.collect_fpm
                        && let Some(fpm) = payload.fpm
                    {
                        self.prefill_fpm_buffer.push((payload.worker_idx, fpm));
                    }
                    self.process_prefill_pass(
                        payload.worker_idx,
                        payload.completed_requests,
                        payload.output_signals,
                        payload.lifecycle_events,
                        payload.kv_events,
                    )?;
                }
                SimulationWorkerStage::Decode => {
                    let payload = self.decode_engine.on_scheduled_completion(payload)?;
                    self.wake_deferred_actions(SimulationWorkerStage::Decode, payload.worker_idx);
                    if self.collect_fpm
                        && let Some(fpm) = payload.fpm
                    {
                        self.decode_fpm_buffer.push((payload.worker_idx, fpm));
                    }
                    self.process_decode_pass(
                        payload.output_signals,
                        payload.lifecycle_events,
                        payload.kv_events,
                        payload.accept_length_output_tokens,
                        payload.accept_length_decode_forwards,
                    )?;
                }
                SimulationWorkerStage::Aggregated => {
                    bail!("offline disagg replay received an aggregated completion event")
                }
            }
            changed = true;
        }
        Ok(changed)
    }

    /// Drain transfer completions scheduled for the current logical timestamp.
    fn apply_transfer_completions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some(handoff_id) = pop_ready_transfer_complete(&mut self.events, self.now_ms) {
            let Some(uuid) = self.requests_by_handoff.get(&handoff_id).copied() else {
                continue;
            };
            self.apply_handoff_fact(uuid, HandoffFact::TransferCompleted { handoff_id })?;
            changed = true;
        }
        Ok(changed)
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> Result<bool> {
        let mut released_any = false;
        for ready in self
            .admission
            .drain_ready(self.now_ms, self.cluster_in_flight())?
        {
            let ReadyArrival {
                request,
                arrival_time_ms,
                replay_hashes,
                session_id,
                turn_index,
            } = ready;
            let session_metadata = session_id.zip(turn_index);
            let uuid = self.on_external_arrival(request, arrival_time_ms, replay_hashes)?;
            if let Some((session_id, turn_index)) = session_metadata {
                self.collector
                    .on_session_metadata(uuid, session_id, turn_index);
            }
            released_any = true;
        }
        Ok(released_any)
    }

    /// Start passes on every idle prefill worker that can make progress at the current timestamp.
    fn drive_prefill_workers(&mut self) -> Result<bool> {
        let mut changed = false;
        loop {
            let effects = self.prefill_engine.drive_ready(self.now_ms, None)?;
            if effects.is_empty() {
                return Ok(changed);
            }
            changed = true;
            self.handle_prefill_engine_effects(effects)?;
        }
    }

    /// Start passes on every idle decode worker that can make progress at the current timestamp.
    fn drive_decode_workers(&mut self) -> Result<bool> {
        let mut changed = false;
        loop {
            let effects = self
                .decode_engine
                .drive_ready(self.now_ms, Some(&mut self.collector))?;
            if effects.is_empty() {
                return Ok(changed);
            }
            changed = true;
            self.handle_decode_engine_effects(effects)?;
        }
    }

    fn handle_prefill_engine_effects(&mut self, effects: EngineEffects) -> Result<()> {
        if self.collect_fpm {
            self.prefill_fpm_buffer.extend(effects.fpm_snapshots);
        }
        self.record_prefill_admissions(effects.admissions);
        self.apply_prefill_router_events(effects.pass_start_kv_events)?;
        for payload in effects.immediate_completions {
            let payload = self.prefill_engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.prefill_fpm_buffer.push((payload.worker_idx, fpm));
            }
            self.process_prefill_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
                payload.lifecycle_events,
                payload.kv_events,
            )?;
        }
        for ScheduledWorkerCompletion { at_ms, payload } in effects.scheduled_completions {
            push_worker_completion(&mut self.events, &mut self.next_event_seq, at_ms, payload);
        }
        Ok(())
    }

    fn record_prefill_admissions(&mut self, admissions: Vec<AdmissionEvent>) {
        for admission in admissions {
            self.collector.on_prefill_admit(
                admission.uuid,
                self.now_ms,
                admission.reused_input_tokens,
            );
        }
    }

    fn record_decode_admissions(&mut self, admissions: Vec<AdmissionEvent>) -> Result<()> {
        for admission in admissions {
            self.collector.on_decode_admit(
                admission.uuid,
                self.now_ms,
                admission.reused_input_tokens,
            );
            match self.state(admission.uuid)?.phase {
                DisaggPhase::ReadyDecode => {
                    self.state_mut(admission.uuid)?.start_decode();
                    #[cfg(test)]
                    self.stats
                        .transition_log
                        .push(DisaggTransition::DecodeAdmitted {
                            uuid: admission.uuid,
                        });
                }
                DisaggPhase::RunningDecode => {}
                phase => bail!(
                    "offline disagg replay decode admission for {} in phase {phase:?}",
                    admission.uuid
                ),
            }
        }
        Ok(())
    }

    fn handle_decode_engine_effects(&mut self, effects: EngineEffects) -> Result<()> {
        if self.collect_fpm {
            self.decode_fpm_buffer.extend(effects.fpm_snapshots);
        }
        self.record_decode_admissions(effects.admissions)?;
        for payload in effects.immediate_completions {
            let payload = self.decode_engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.decode_fpm_buffer.push((payload.worker_idx, fpm));
            }
            self.process_decode_pass(
                payload.output_signals,
                payload.lifecycle_events,
                payload.kv_events,
                payload.accept_length_output_tokens,
                payload.accept_length_decode_forwards,
            )?;
        }
        for ScheduledWorkerCompletion { at_ms, payload } in effects.scheduled_completions {
            push_worker_completion(&mut self.events, &mut self.next_event_seq, at_ms, payload);
        }
        Ok(())
    }

    /// Activate workers whose startup period has elapsed at the current timestamp.
    fn apply_worker_ready_events(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some((stage, worker_id)) = pop_ready_worker_ready(&mut self.events, self.now_ms) {
            match stage {
                SimulationWorkerStage::Prefill => {
                    if self.prefill_engine.mark_worker_ready(worker_id) {
                        if let Some(router) = self.prefill_router.as_mut() {
                            router.add_worker(worker_id)?;
                            let effects = router.try_drain_pending(self.now_ms)?;
                            self.dispatch_prefill_admissions(effects.admissions)?;
                        }
                        self.wake_worker_waiters(SimulationWorkerStage::Prefill);
                        changed = true;
                    }
                }
                SimulationWorkerStage::Decode => {
                    if self.decode_engine.mark_worker_ready(worker_id) {
                        if let Some(router) = self.decode_router.as_mut() {
                            router.add_worker(worker_id)?;
                            let effects = router.try_drain_pending(self.now_ms)?;
                            self.dispatch_decode_admissions(effects.admissions)?;
                        }
                        self.wake_worker_waiters(SimulationWorkerStage::Decode);
                        changed = true;
                    }
                }
                SimulationWorkerStage::Aggregated => {
                    unreachable!("disagg replay should not receive aggregated worker ready events")
                }
            }
        }
        Ok(changed)
    }

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> Result<()> {
        loop {
            #[cfg_attr(not(feature = "kvbm-offload"), allow(unused_mut))]
            let mut changed = self.prune_stale_transfer_events();
            #[cfg(feature = "kvbm-offload")]
            {
                changed |= self.tick_offload_engines()?;
            }
            changed |= self.apply_worker_completions()?;
            changed |= self.apply_worker_ready_events()?;
            changed |= self.apply_transfer_completions()?;
            changed |= self.release_ready_arrivals()?;
            changed |= self.drive_pending_actions()?;
            changed |= self.drive_prefill_workers()?;
            changed |= self.drive_decode_workers()?;
            let removed_prefill = self.prefill_engine.try_remove_drained();
            if let Some(router) = self.prefill_router.as_mut() {
                for worker_id in &removed_prefill {
                    router.finalize_worker_removal(*worker_id)?;
                }
            }
            changed |= !removed_prefill.is_empty();
            let removed_decode = self.decode_engine.try_remove_drained();
            if let Some(router) = self.decode_router.as_mut() {
                for worker_id in &removed_decode {
                    router.finalize_worker_removal(*worker_id)?;
                }
            }
            changed |= !removed_decode.is_empty();
            // Planner ticks fire LAST so the planner observes a fully settled
            // timestamp (matching the old advance-then-tick ordering). Any scaling
            // it applies is picked up by the next loop iteration.
            if self.planner_hook.is_some() {
                changed |= self.apply_planner_ticks()?;
            }

            if !changed {
                break;
            }
        }
        Ok(())
    }

    fn prune_stale_transfer_events(&mut self) -> bool {
        let mut removed = false;
        while self.events.peek().is_some_and(|event| {
            matches!(
                &event.kind,
                super::events::SimulationEventKind::TransferComplete { handoff_id }
                    if !self.requests_by_handoff.contains_key(handoff_id)
            )
        }) {
            self.events.pop();
            self.stale_transfer_events = self
                .stale_transfer_events
                .checked_sub(1)
                .expect("stale transfer event count underflow");
            removed = true;
        }
        if self.stale_transfer_events > 32
            && self.stale_transfer_events.saturating_mul(2) > self.events.len()
        {
            let requests_by_handoff = &self.requests_by_handoff;
            self.events.retain(|event| {
                !matches!(
                    &event.kind,
                    super::events::SimulationEventKind::TransferComplete { handoff_id }
                        if !requests_by_handoff.contains_key(handoff_id)
                )
            });
            self.stale_transfer_events = 0;
            removed = true;
        }
        removed
    }

    /// Seed the first `PlannerTick` event from the hook's requested start time.
    /// A non-finite time means "no tick" (e.g. `NoopPlannerHook`) and is skipped.
    fn seed_first_planner_tick(&mut self) -> Result<()> {
        let Some(mut hook) = self.planner_hook.take() else {
            return Ok(());
        };
        let first_ms = hook.initial_tick_ms();
        self.planner_hook = Some(hook);
        let first_ms = first_ms?;
        if first_ms.is_finite() {
            let at_ms = first_ms.max(self.now_ms);
            push_planner_tick(&mut self.events, &mut self.next_event_seq, at_ms);
        } else {
            // No tick will ever fire to drain the FPM buffers; stop collecting them.
            self.collect_fpm = false;
        }
        Ok(())
    }

    /// Fire every `PlannerTick` scheduled for the current timestamp: gather the
    /// drained metrics, call the planner, apply its scaling decision, and re-arm
    /// the next tick. Called only when a hook is attached.
    fn apply_planner_ticks(&mut self) -> Result<bool> {
        let mut changed = false;
        while pop_ready_planner_tick(&mut self.events, self.now_ms) {
            // Once the workload is finished, drop the tick without bothering the
            // planner and without re-arming (mirrors the Python loop's pre-tick
            // `if is_done: break`), so the heap drains and `run()` exits.
            if self.is_workload_done() {
                continue;
            }
            let metrics = PlannerTickMetrics {
                now_ms: self.now_ms,
                prefill_fpm: std::mem::take(&mut self.prefill_fpm_buffer),
                decode_fpm: std::mem::take(&mut self.decode_fpm_buffer),
                traffic: self.traffic.drain(self.now_ms),
                active_prefill: self.active_prefill_count(),
                active_decode: self.active_decode_count(),
                total_prefill: self.total_prefill_count(),
                total_decode: self.total_decode_count(),
            };
            // Borrow the hook out so the runtime stays mutably available for
            // apply_scaling; restore it before propagating any error.
            let mut hook = self
                .planner_hook
                .take()
                .expect("planner tick fired without a hook");
            let decision = hook.on_tick(metrics);
            self.planner_hook = Some(hook);
            let decision = decision?;

            if decision.target_prefill.is_some() || decision.target_decode.is_some() {
                let target_prefill = decision
                    .target_prefill
                    .unwrap_or_else(|| self.total_prefill_count());
                let target_decode = decision
                    .target_decode
                    .unwrap_or_else(|| self.total_decode_count());
                self.apply_scaling(target_prefill, target_decode)?;
            }

            // Re-arm only into the strict, finite future and only while work
            // remains; the `at_ms == now_ms` pop guard plus this check prevent a
            // same-pass re-fire or an infinite spin from a degenerate
            // `next_tick_ms <= now_ms`. When no future tick will fire, stop FPM
            // collection so neither buffer grows unbounded after the cadence ends.
            let next_tick = decision
                .next_tick_ms
                .filter(|next_ms| next_ms.is_finite() && *next_ms > self.now_ms);
            if let Some(next_ms) = next_tick
                && !self.is_workload_done()
            {
                push_planner_tick(&mut self.events, &mut self.next_event_seq, next_ms);
            } else {
                self.collect_fpm = false;
            }
            changed = true;
        }
        Ok(changed)
    }

    /// Finalize test-only request snapshots before returning.
    fn finish_test_stats(&mut self) {
        #[cfg(test)]
        {
            let counted = self
                .requests
                .values()
                .filter(|state| state.counted_in_flight)
                .count();
            assert_eq!(self.logical_in_flight, counted);
            for state in self.requests.values() {
                assert_eq!(
                    state.counted_in_flight,
                    !matches!(state.phase, DisaggPhase::CleanupPending | DisaggPhase::Done)
                );
            }
            self.stats.request_snapshots = self
                .requests
                .iter()
                .map(|(uuid, state)| (*uuid, state.debug_snapshot()))
                .collect();
        }
    }

    // ------------------------------------------------------------------
    // Planner integration: scaling + worker-count accessors used by the
    // in-loop `PlannerTick` handler (apply_planner_ticks).
    // ------------------------------------------------------------------

    /// Advance the sim clock to `new_now_ms`, integrating provisioned
    /// worker-seconds for both pools over the interval just elapsed.
    /// `worker_count()` counts active + starting-up + draining workers, so this
    /// captures the startup ramp and the scale-down drain tail.
    fn advance_now_ms(&mut self, new_now_ms: f64) {
        let dt_ms = (new_now_ms - self.now_ms).max(0.0);
        if dt_ms > 0.0 {
            let prefill_worker_seconds = self.prefill_engine.worker_count() as f64 * dt_ms / 1000.0;
            let decode_worker_seconds = self.decode_engine.worker_count() as f64 * dt_ms / 1000.0;
            self.collector
                .add_worker_seconds(prefill_worker_seconds, decode_worker_seconds);
        }
        self.now_ms = new_now_ms;
    }

    pub(in crate::replay) fn active_prefill_count(&self) -> usize {
        self.prefill_engine.active_worker_ids().len()
    }

    pub(in crate::replay) fn active_decode_count(&self) -> usize {
        self.decode_engine.active_worker_ids().len()
    }

    pub(in crate::replay) fn total_prefill_count(&self) -> usize {
        self.prefill_engine.worker_count()
    }

    pub(in crate::replay) fn total_decode_count(&self) -> usize {
        self.decode_engine.worker_count()
    }

    /// Apply a scaling decision with separate prefill and decode targets.
    ///
    /// Scale-up: if `startup_time` is configured on the respective engine args,
    /// new workers enter a startup phase and a `WorkerReady` event is scheduled.
    /// They become active (and are registered with the router) only when that
    /// event fires.  Without `startup_time`, workers are available immediately.
    ///
    /// Scale-down: the worker is removed from the router immediately so no
    /// new requests land on it while it drains in-flight work.
    pub(in crate::replay) fn apply_scaling(
        &mut self,
        target_prefill: usize,
        target_decode: usize,
    ) -> Result<()> {
        // -- prefill --
        let (added, newly_marked, removed) = self.prefill_engine.apply_target_count(target_prefill);
        let prefill_delay = self.prefill_engine.startup_time_ms();
        for &id in &added {
            match prefill_delay {
                Some(delay) => {
                    push_worker_ready(
                        &mut self.events,
                        &mut self.next_event_seq,
                        self.now_ms + delay,
                        SimulationWorkerStage::Prefill,
                        id,
                    );
                }
                None => {
                    if let Some(router) = self.prefill_router.as_mut() {
                        router.add_worker(id)?;
                    }
                }
            }
        }
        let prefill_admissions = if let Some(router) = self.prefill_router.as_mut() {
            for id in newly_marked {
                router.remove_worker(id)?;
            }
            for id in removed {
                router.finalize_worker_removal(id)?;
            }
            router.on_topology_changed(self.now_ms)?.admissions
        } else {
            Vec::new()
        };
        if !added.is_empty() && prefill_delay.is_none() {
            self.wake_worker_waiters(SimulationWorkerStage::Prefill);
        }

        // -- decode --
        let (added, newly_marked, removed) = self.decode_engine.apply_target_count(target_decode);
        let decode_delay = self.decode_engine.startup_time_ms();
        for &id in &added {
            match decode_delay {
                Some(delay) => {
                    push_worker_ready(
                        &mut self.events,
                        &mut self.next_event_seq,
                        self.now_ms + delay,
                        SimulationWorkerStage::Decode,
                        id,
                    );
                }
                None => {
                    if let Some(router) = self.decode_router.as_mut() {
                        router.add_worker(id)?;
                    }
                }
            }
        }
        let decode_admissions = if let Some(router) = self.decode_router.as_mut() {
            for id in newly_marked {
                router.remove_worker(id)?;
            }
            for id in removed {
                router.finalize_worker_removal(id)?;
            }
            router.on_topology_changed(self.now_ms)?.admissions
        } else {
            Vec::new()
        };
        if !added.is_empty() && decode_delay.is_none() {
            self.wake_worker_waiters(SimulationWorkerStage::Decode);
        }
        self.record_router_pending();
        self.dispatch_prefill_admissions(prefill_admissions)?;
        self.dispatch_decode_admissions(decode_admissions)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Test-only stepping helpers. White-box unit tests advance the sim to a
    // chosen simulated time, inspect mid-flight state, apply a manual scaling
    // decision, and resume — a granularity `run()` (which goes straight to
    // completion) cannot offer. Not part of the production drive path.
    // ------------------------------------------------------------------

    /// Advance the simulation up to `until_ms` simulated time, then pause.
    /// Returns `true` if the request workload is done — pending `WorkerReady`
    /// events do not block completion since there is no work for those workers.
    #[cfg(test)]
    fn advance_to(&mut self, until_ms: f64) -> Result<bool> {
        self.drain_current_timestamp()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                if until_ms > self.now_ms {
                    self.advance_now_ms(until_ms);
                }
                break;
            };

            if next_timestamp_ms > until_ms {
                if until_ms > self.now_ms {
                    self.advance_now_ms(until_ms);
                }
                break;
            }

            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
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

    #[cfg(test)]
    fn drain_prefill_fpm(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        std::mem::take(&mut self.prefill_fpm_buffer)
    }

    #[cfg(test)]
    fn drain_decode_fpm(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        std::mem::take(&mut self.decode_fpm_buffer)
    }

    fn run_to_completion(&mut self) -> Result<()> {
        if let Some(cap_ms) = self.max_sim_time_ms
            && (!cap_ms.is_finite() || cap_ms < 0.0)
        {
            bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
        }
        self.drain_current_timestamp()?;
        // With a planner attached, seed the recurring heartbeat; ticks then fire as
        // events inside drain_current_timestamp (see apply_planner_ticks).
        self.seed_first_planner_tick()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline disagg replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };
            if let Some(cap_ms) = self.max_sim_time_ms
                && next_timestamp_ms > cap_ms
            {
                break;
            }
            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
        }

        Ok(())
    }

    /// Run the staged offline replay until both prefill and decode pipelines are drained.
    /// If `max_sim_time_ms` is set, exits gracefully when the next scheduled
    /// timestamp would exceed that cap; in-flight requests at that point are
    /// reported as incomplete.
    pub(in crate::replay) fn run(mut self) -> Result<(TraceCollector, DisaggRuntimeStats)> {
        self.run_to_completion()?;

        self.progress.finish();
        self.finish_test_stats();
        Ok((self.collector, self.stats))
    }

    pub(super) fn run_handoff_conformance(
        mut self,
        engine_type: EngineType,
    ) -> Result<NormalizedHandoffConformance> {
        self.run_to_completion()?;

        let source_drained = self.prefill_engine.is_drained();
        let destination_drained = self.decode_engine.is_drained();
        let driver_drained = self.is_done()
            && self.action_queues.is_empty()
            && self
                .prefill_router
                .as_ref()
                .is_none_or(|router| router.pending_count() == 0)
            && self
                .decode_router
                .as_ref()
                .is_none_or(|router| router.pending_count() == 0)
            && self.requests.values().all(|state| {
                !state.counted_in_flight && !state.prefill_routed && !state.destination_routed
            });
        let capture = self
            .conformance_capture
            .take()
            .ok_or_else(|| anyhow!("offline handoff conformance capture was not enabled"))?;

        self.progress.finish();
        let report = self.collector.finish();
        let conformance = NormalizedHandoffConformance {
            engine_type,
            order: self.handoff_order,
            lifecycle: capture.lifecycle,
            source_output_tokens: capture.source_output_tokens,
            destination_output_tokens: report.request_counts.total_output_tokens,
            completed_requests: report.request_counts.completed_requests,
            destination_stored: NormalizedStoredTiming {
                before_activation: capture.stored_before_activation,
                on_activation: capture.stored_on_activation,
                repeated_activation_hashes_after_activation: capture
                    .repeated_activation_hashes_after_activation,
            },
            source_drained,
            destination_drained,
            driver_drained,
        };
        conformance.validate()?;
        Ok(conformance)
    }
}

fn command_cleanup_outcome(result: SchedulerCommandResult) -> Result<HandoffActionOutcome> {
    match result {
        SchedulerCommandResult::Applied => Ok(HandoffActionOutcome::Applied),
        SchedulerCommandResult::Noop => Ok(HandoffActionOutcome::Noop),
        _ => bail!("handoff cleanup returned an unexpected scheduler result"),
    }
}

fn base_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
}

fn derive_prefill_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.router_track_active_blocks = false;
    config
}

fn derive_decode_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.overlap_score_credit = 0.0;
    config.router_selection_policy = dynamo_kv_router::config::RouterSelectionPolicy::Linear;
    config.router_assume_kv_reuse = false;
    config.router_track_prefill_tokens = false;
    config.router_prefill_load_model = dynamo_kv_router::config::RouterPrefillLoadModel::None;
    config
}

#[cfg(test)]
#[path = "disagg_tests.rs"]
mod tests;
