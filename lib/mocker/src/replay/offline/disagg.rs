// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{Result, anyhow, bail};
use uuid::Uuid;

pub(super) use super::components::ReplayMode;
#[cfg(test)]
use super::components::TrafficStats;
use super::components::{
    AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, NoReplayMetadata,
    ReplayAdmissionMetadata, ReplayEngineObservation, ScheduledWorkerCompletion,
    TrafficAccumulator,
};
use super::core::round_robin::PoolRoundRobinPlacement;
use super::core::{
    AdmissionSource as CoreAdmissionSource, EngineEventBatch, NoEngineEvents, Placement,
    PlacementDecision, PlacementPolicy, ReadyArrival, WorkerTopology,
};
use super::events::{SimulationEvent, SimulationWorkerStage};
#[cfg(test)]
use super::extensions::kv_router::{
    DisaggRuntime, ReplayKvRouterConfig, derive_decode_router_config, derive_prefill_router_config,
};
use super::planner_hook::{LatestFpmBuffer, PlannerHook, PlannerTickMetrics};
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
#[cfg(test)]
use crate::common::protocols::ForwardPassSnapshot;
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, ReplayRequestPayload, WorkloadDriver};
#[cfg(test)]
use crate::replay::ReplayRouterMode;
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayTerminalStatus, SlaThresholds, TraceCollector,
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
    activation_stored_hashes: HashSet<u64>,
    repeated_activation_hashes_after_activation: usize,
}

impl HandoffConformanceCapture {
    fn record_before_activation(&mut self, stored_hashes: &[u64]) {
        self.stored_before_activation += stored_hashes.len();
    }

    fn record_activation(&mut self, stored_hashes: &[u64]) {
        for &hash in stored_hashes {
            self.stored_on_activation += 1;
            self.activation_stored_hashes.insert(hash);
        }
    }

    fn record_after_activation(&mut self, stored_hashes: &[u64]) {
        self.repeated_activation_hashes_after_activation += stored_hashes
            .iter()
            .filter(|hash| self.activation_stored_hashes.contains(hash))
            .count();
    }
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

struct DisaggFlowState {
    requests: HashMap<Uuid, DisaggRequestState>,
    requests_by_handoff: HashMap<HandoffId, Uuid>,
    handoff_order: HandoffOrder,
    action_queues: DisaggActionQueues,
    logical_in_flight: usize,
    stale_transfer_events: usize,
    conformance_capture: Option<HandoffConformanceCapture>,
}

enum PrefillSignalDisposition {
    Pending,
    Rejected,
    Completed,
}

struct ScheduledTransfer {
    at_ms: f64,
    handoff_id: HandoffId,
}

impl DisaggFlowState {
    fn new(handoff_order: HandoffOrder, capture_conformance: bool) -> Self {
        Self {
            requests: HashMap::new(),
            requests_by_handoff: HashMap::new(),
            handoff_order,
            action_queues: DisaggActionQueues::default(),
            logical_in_flight: 0,
            stale_transfer_events: 0,
            conformance_capture: capture_conformance.then(HandoffConformanceCapture::default),
        }
    }

    #[inline(never)]
    fn state(&self, uuid: Uuid) -> Result<&DisaggRequestState> {
        self.requests
            .get(&uuid)
            .ok_or_else(|| anyhow!("offline disagg replay missing request state for {uuid}"))
    }

    #[inline(never)]
    fn state_mut(&mut self, uuid: Uuid) -> Result<&mut DisaggRequestState> {
        self.requests
            .get_mut(&uuid)
            .ok_or_else(|| anyhow!("offline disagg replay missing request state for {uuid}"))
    }

    #[inline(never)]
    fn acknowledge_action(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        outcome: HandoffActionOutcome,
        now_ms: f64,
        collector: &mut TraceCollector,
    ) -> Result<()> {
        if matches!(outcome, HandoffActionOutcome::Failed(_)) {
            collector.on_terminal(uuid, now_ms, ReplayTerminalStatus::Failed);
        }
        let actions = self
            .state_mut(uuid)?
            .coordinator
            .on_action_outcome(action.id, outcome)?;
        self.action_queues.enqueue_all(uuid, actions);
        Ok(())
    }

    #[inline(never)]
    fn apply_handoff_fact(
        &mut self,
        uuid: Uuid,
        fact: HandoffFact,
        now_ms: f64,
        collector: &mut TraceCollector,
    ) -> Result<()> {
        let terminal_status = match fact {
            HandoffFact::Failed { .. } | HandoffFact::TimedOut { .. } => {
                Some(ReplayTerminalStatus::Failed)
            }
            HandoffFact::Canceled { .. } => Some(ReplayTerminalStatus::Canceled),
            _ => None,
        };
        if let Some(status) = terminal_status {
            collector.on_terminal(uuid, now_ms, status);
        }
        let actions = self.state_mut(uuid)?.coordinator.on_fact(fact)?;
        self.action_queues.enqueue_all(uuid, actions);
        Ok(())
    }

    #[inline(never)]
    fn record_prefill_placement(
        &self,
        placement: Placement,
        traffic: &mut TrafficAccumulator,
        collector: &mut TraceCollector,
    ) -> Result<()> {
        if let Some(sample) = placement.planner_cache_sample {
            traffic.on_admission(sample.overlap_blocks, sample.isl_blocks);
        }
        let input_tokens = self.state(placement.request_id)?.input_length()?;
        collector.on_prefill_route_overlap(
            placement.request_id,
            placement.reported_overlap_tokens.min(input_tokens),
        );
        Ok(())
    }

    #[inline(never)]
    fn record_decode_placement(
        &self,
        placement: Placement,
        collector: &mut TraceCollector,
    ) -> Result<()> {
        let input_tokens = self.state(placement.request_id)?.input_length()?;
        collector.on_decode_route_overlap(
            placement.request_id,
            placement.reported_overlap_tokens.min(input_tokens),
        );
        Ok(())
    }

    #[inline(never)]
    fn prepare_prefill_submission(&mut self, uuid: Uuid) -> Result<(DirectRequest, HandoffId)> {
        let handoff_id = self.state(uuid)?.handoff_id;
        Ok((self.state_mut(uuid)?.build_prefill_request()?, handoff_id))
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn finish_prefill_submission(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        now_ms: f64,
        collector: &mut TraceCollector,
        stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
        self.state_mut(uuid)?.start_prefill(worker_idx);
        collector.on_prefill_assigned(uuid, worker_idx);
        #[cfg(test)]
        {
            stats.prefill_assignments.insert(uuid, worker_idx);
        }
        self.acknowledge_action(
            uuid,
            action,
            HandoffActionOutcome::Submitted,
            now_ms,
            collector,
        )?;
        self.process_lifecycle_events(lifecycle_events, now_ms, collector, stats)
    }

    #[inline(never)]
    fn prepare_destination_reservation(
        &mut self,
        uuid: Uuid,
    ) -> Result<(DirectRequest, HandoffId)> {
        let handoff_id = self.state(uuid)?.handoff_id;
        Ok((
            self.state_mut(uuid)?
                .materialize_original_request()?
                .clone(),
            handoff_id,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn finish_destination_reservation(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
        stored_hashes: &[u64],
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        now_ms: f64,
        collector: &mut TraceCollector,
        stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture.record_before_activation(stored_hashes);
            capture
                .lifecycle
                .push(NormalizedHandoffEvent::DestinationAccepted);
        }
        self.state_mut(uuid)?.assign_decode(worker_idx);
        collector.on_decode_assigned(uuid, worker_idx);
        #[cfg(test)]
        {
            stats.decode_assignments.insert(uuid, worker_idx);
            stats
                .transition_log
                .push(DisaggTransition::DestinationAccepted { uuid });
        }
        self.acknowledge_action(
            uuid,
            action,
            HandoffActionOutcome::Accepted,
            now_ms,
            collector,
        )?;
        self.process_lifecycle_events(lifecycle_events, now_ms, collector, stats)
    }

    #[inline(never)]
    fn take_prefill_placement(
        &mut self,
        placement: Placement,
        traffic: &mut TrafficAccumulator,
        collector: &mut TraceCollector,
    ) -> Result<(Uuid, usize, IssuedHandoffAction)> {
        let uuid = placement.request_id;
        let worker_idx = placement.scheduler_id;
        self.record_prefill_placement(placement, traffic, collector)?;
        if self.state(uuid)?.phase != DisaggPhase::QueuedPrefill {
            bail!("offline disagg replay expected queued prefill request for {uuid}");
        }
        let action = self
            .state_mut(uuid)?
            .pending_prefill_action
            .take()
            .ok_or_else(|| anyhow!("missing coordinator prefill action for {uuid}"))?;
        Ok((uuid, worker_idx, action))
    }

    #[inline(never)]
    fn take_decode_placement(
        &mut self,
        placement: Placement,
        collector: &mut TraceCollector,
    ) -> Result<(Uuid, usize, IssuedHandoffAction)> {
        let uuid = placement.request_id;
        let worker_idx = placement.scheduler_id;
        self.record_decode_placement(placement, collector)?;
        if self.state(uuid)?.phase != DisaggPhase::AwaitingDestination {
            bail!("offline disagg replay expected destination-waiting request for {uuid}");
        }
        let action = self
            .state_mut(uuid)?
            .pending_destination_action
            .take()
            .ok_or_else(|| anyhow!("missing coordinator destination action for {uuid}"))?;
        Ok((uuid, worker_idx, action))
    }

    fn uuid_for_handoff(&self, handoff_id: HandoffId) -> Result<Uuid> {
        self.requests_by_handoff
            .get(&handoff_id)
            .copied()
            .ok_or_else(|| anyhow!("offline disagg replay missing handoff {handoff_id:?}"))
    }

    #[inline(never)]
    fn process_lifecycle_events(
        &mut self,
        events: Vec<SchedulerLifecycleEvent>,
        now_ms: f64,
        collector: &mut TraceCollector,
        _stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
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
                    _stats
                        .transition_log
                        .push(DisaggTransition::SourceHeld { uuid });
                    if let Some(capture) = self.conformance_capture.as_mut() {
                        capture.lifecycle.push(NormalizedHandoffEvent::SourceHeld);
                    }
                    collector.on_source_held(uuid, now_ms);
                    self.apply_handoff_fact(
                        uuid,
                        HandoffFact::SourceHeld {
                            handoff_id,
                            transfer_timing,
                        },
                        now_ms,
                        collector,
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
                    _stats
                        .transition_log
                        .push(DisaggTransition::DestinationReserved { uuid });
                    if let Some(capture) = self.conformance_capture.as_mut() {
                        capture
                            .lifecycle
                            .push(NormalizedHandoffEvent::DestinationReserved);
                    }
                    collector.on_destination_reserved(uuid, now_ms);
                    self.apply_handoff_fact(
                        uuid,
                        HandoffFact::DestinationReserved {
                            handoff_id,
                            transferable_prompt_tokens,
                        },
                        now_ms,
                        collector,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn on_external_arrival(
        &mut self,
        mut request: ReplayRequestPayload,
        arrival_time_ms: f64,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        collector: &mut TraceCollector,
    ) -> Result<Uuid> {
        let uuid = request.metadata().uuid.unwrap_or_else(Uuid::new_v4);
        let input_length = request.input_length();
        let output_length = request.metadata().max_output_tokens;
        request.metadata_mut().uuid = Some(uuid);
        request.metadata_mut().arrival_timestamp_ms = Some(arrival_time_ms);

        collector.on_arrival(uuid, arrival_time_ms, input_length, output_length);
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
            session_id,
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

    #[inline(never)]
    fn inspect_prefill_signal(
        &mut self,
        signal: &OutputSignal,
        now_ms: f64,
        collector: &mut TraceCollector,
    ) -> Result<PrefillSignalDisposition> {
        if !signal.rejected
            && signal.token_id.is_some()
            && let Some(capture) = self.conformance_capture.as_mut()
        {
            capture.source_output_tokens += 1;
        }
        if !signal.completed {
            return Ok(PrefillSignalDisposition::Pending);
        }
        if !signal.rejected {
            return Ok(PrefillSignalDisposition::Completed);
        }

        let handoff_id = self.state(signal.uuid)?.handoff_id;
        collector.on_terminal(signal.uuid, now_ms, ReplayTerminalStatus::Rejected);
        self.apply_handoff_fact(
            signal.uuid,
            HandoffFact::Failed { handoff_id },
            now_ms,
            collector,
        )?;
        Ok(PrefillSignalDisposition::Rejected)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn start_transfer(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        delay_ms: f64,
        now_ms: f64,
        collector: &mut TraceCollector,
    ) -> Result<Option<ScheduledTransfer>> {
        self.acknowledge_action(
            uuid,
            action,
            HandoffActionOutcome::Scheduled,
            now_ms,
            collector,
        )?;
        self.state_mut(uuid)?.transfer_pending();
        let handoff_id = self.state(uuid)?.handoff_id;
        if delay_ms > 0.0 {
            return Ok(Some(ScheduledTransfer {
                at_ms: now_ms + delay_ms,
                handoff_id,
            }));
        }
        self.apply_handoff_fact(
            uuid,
            HandoffFact::TransferCompleted { handoff_id },
            now_ms,
            collector,
        )?;
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn finish_destination_activation(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        stored_hashes: &[u64],
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        now_ms: f64,
        collector: &mut TraceCollector,
        stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture.record_activation(stored_hashes);
            capture
                .lifecycle
                .push(NormalizedHandoffEvent::DestinationActivated);
        }
        self.state_mut(uuid)?.ready_decode();
        collector.on_destination_activated(uuid, now_ms);
        #[cfg(test)]
        stats
            .transition_log
            .push(DisaggTransition::DestinationActivated { uuid });
        self.acknowledge_action(
            uuid,
            action,
            HandoffActionOutcome::Applied,
            now_ms,
            collector,
        )?;
        self.process_lifecycle_events(lifecycle_events, now_ms, collector, stats)
    }

    #[inline(never)]
    fn record_source_release(&mut self, _uuid: Uuid, _stats: &mut DisaggRuntimeStats) {
        #[cfg(test)]
        _stats
            .transition_log
            .push(DisaggTransition::SourceReleased { uuid: _uuid });
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture
                .lifecycle
                .push(NormalizedHandoffEvent::SourceReleased);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn finish_source_release(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        outcome: HandoffActionOutcome,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        now_ms: f64,
        collector: &mut TraceCollector,
        stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
        collector.on_source_released(uuid, now_ms);
        self.acknowledge_action(uuid, action, outcome, now_ms, collector)?;
        self.process_lifecycle_events(lifecycle_events, now_ms, collector, stats)
    }

    #[inline(never)]
    fn complete_successful_handoff(
        &mut self,
        uuid: Uuid,
        _now_ms: f64,
        _stats: &mut DisaggRuntimeStats,
    ) -> Result<()> {
        #[cfg(test)]
        {
            _stats.handoff_ms.insert(uuid, _now_ms);
            _stats
                .transition_log
                .push(DisaggTransition::HandoffCompleted { uuid });
        }
        if let Some(capture) = self.conformance_capture.as_mut() {
            capture.lifecycle.push(NormalizedHandoffEvent::Completed);
        }
        self.retire_completed_request(uuid)
    }

    #[inline(never)]
    fn record_decode_terminal(
        &self,
        signal: &OutputSignal,
        now_ms: f64,
        collector: &mut TraceCollector,
        traffic: &mut TrafficAccumulator,
    ) -> Result<()> {
        if !signal.rejected {
            let (input_tokens, requested_output_tokens) = {
                let state = self.state(signal.uuid)?;
                let original = state.original_request()?;
                (original.tokens.len(), original.max_output_tokens)
            };
            let actual_output_tokens =
                collector.actual_output_length(signal.uuid).ok_or_else(|| {
                    anyhow!("offline replay missing collector state for {}", signal.uuid)
                })?;
            debug_assert!(actual_output_tokens <= requested_output_tokens);
            let latencies = collector.request_latencies(signal.uuid);
            traffic.on_request(input_tokens, actual_output_tokens, latencies);
        }
        let terminal_status = if signal.rejected {
            ReplayTerminalStatus::Rejected
        } else {
            ReplayTerminalStatus::Completed
        };
        collector.on_terminal(signal.uuid, now_ms, terminal_status);
        Ok(())
    }

    #[inline(never)]
    fn prepare_logical_finish(&mut self, uuid: Uuid, remove_actions: bool) -> Result<()> {
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
        Ok(())
    }

    #[inline(never)]
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
}

pub(in crate::replay) trait PoolPlacement<Events, Metadata>:
    PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Events> + Sized
where
    Events: EngineEventBatch,
    Metadata: ReplayAdmissionMetadata,
{
    fn is_router(&self) -> bool;
}

impl<Events: EngineEventBatch> PoolPlacement<Events, ()> for PoolRoundRobinPlacement<Events> {
    #[inline]
    fn is_router(&self) -> bool {
        false
    }
}

pub(in crate::replay) type RoundRobinDisaggRuntime =
    DisaggRuntimeImpl<PoolRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

pub(in crate::replay) struct DisaggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PoolPlacement<Observation::Batch, Metadata>,
{
    now_ms: f64,
    next_event_seq: u64,
    admission: AdmissionQueue<Metadata>,
    prefill_engine: EngineComponent<Observation>,
    decode_engine: EngineComponent<Observation>,
    prefill_placement: PlacementPolicyImpl,
    decode_placement: PlacementPolicyImpl,
    flow: DisaggFlowState,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent<Observation::Batch>>,
    progress: ReplayProgress,
    stats: DisaggRuntimeStats,
    /// Latest forward pass metric per worker/rank since the previous planner tick.
    prefill_fpm_buffer: LatestFpmBuffer,
    decode_fpm_buffer: LatestFpmBuffer,
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
    /// Whether to retain the latest FPM snapshot per worker/rank in the buffers
    /// above. Only the planner consumes them, so the plain `run()` path leaves this
    /// `false`.
    collect_fpm: bool,
}

impl DisaggRuntimeImpl<PoolRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata> {
    pub(in crate::replay) fn new_round_robin(
        config: &OfflineDisaggReplayConfig,
        pending: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Result<Self> {
        Self::new_composed(
            config,
            AdmissionQueue::new_requests(pending, mode),
            false,
            false,
            false,
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
        )
    }

    pub(in crate::replay) fn new_round_robin_workload(
        config: &OfflineDisaggReplayConfig,
        driver: WorkloadDriver,
        mode: ReplayMode,
    ) -> Result<Self> {
        Self::new_composed(
            config,
            AdmissionQueue::new_workload(driver, mode),
            false,
            false,
            false,
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
        )
    }
}

impl<PlacementPolicyImpl, Observation, Metadata>
    DisaggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PoolPlacement<Observation::Batch, Metadata>,
{
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replay::offline) fn new_composed(
        config: &OfflineDisaggReplayConfig,
        admission: AdmissionQueue<Metadata>,
        prefill_capture_raw: bool,
        decode_capture_raw: bool,
        capture_conformance: bool,
        create_prefill_placement: impl FnOnce(
            &MockEngineArgs,
            Vec<WorkerTopology>,
        ) -> Result<PlacementPolicyImpl>,
        create_decode_placement: impl FnOnce(
            &MockEngineArgs,
            Vec<WorkerTopology>,
        ) -> Result<PlacementPolicyImpl>,
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
        let progress = ReplayProgress::new(
            CoreAdmissionSource::total_requests(&admission),
            "offline disagg replay",
        );
        let mut prefill_engine = EngineComponent::<Observation>::new(
            SimulationWorkerStage::Prefill,
            EnginePassMode::Hidden,
            (0..config.num_prefill_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.prefill_args.clone(),
                        prefill_capture_raw,
                    )
                })
                .collect(),
        );
        prefill_engine.set_scaling_args(config.prefill_args.clone(), prefill_capture_raw);
        let mut decode_engine = EngineComponent::<Observation>::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            (0..config.num_decode_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.decode_args.clone(),
                        decode_capture_raw,
                    )
                })
                .collect(),
        );
        decode_engine.set_scaling_args(config.decode_args.clone(), decode_capture_raw);
        let prefill_placement =
            create_prefill_placement(&config.prefill_args, prefill_engine.active_topology())?;
        let decode_placement =
            create_decode_placement(&config.decode_args, decode_engine.active_topology())?;

        // Record each pool's GPUs/worker from its engine parallelism so the
        // report can express GPU-hours from the mocker's own config.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(
            config.prefill_args.aic_gpus_per_worker(),
            config.decode_args.aic_gpus_per_worker(),
        );

        Ok(Self {
            now_ms: 0.0,
            next_event_seq: 0,
            admission,
            prefill_engine,
            decode_engine,
            prefill_placement,
            decode_placement,
            flow: DisaggFlowState::new(handoff_order, capture_conformance),
            collector,
            events: BinaryHeap::new(),
            progress,
            #[cfg(test)]
            stats: DisaggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: DisaggRuntimeStats,
            prefill_fpm_buffer: LatestFpmBuffer::default(),
            decode_fpm_buffer: LatestFpmBuffer::default(),
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
        let prefill_dp_size = self.prefill_engine.dp_size();
        for worker_id in self.prefill_engine.active_group_ids() {
            self.prefill_fpm_buffer
                .activate_worker(worker_id, prefill_dp_size, self.now_ms);
        }
        let decode_dp_size = self.decode_engine.dp_size();
        for worker_id in self.decode_engine.active_group_ids() {
            self.decode_fpm_buffer
                .activate_worker(worker_id, decode_dp_size, self.now_ms);
        }
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
        self.flow.logical_in_flight
    }

    /// Track the peak number of requests parked in each stage router.
    fn record_router_pending(&mut self) {
        #[cfg(test)]
        {
            self.stats.max_prefill_router_pending_count = self
                .stats
                .max_prefill_router_pending_count
                .max(self.prefill_placement.pending_count());
            self.stats.max_decode_router_pending_count = self
                .stats
                .max_decode_router_pending_count
                .max(self.decode_placement.pending_count());
        }
    }

    /// Borrow immutable request state with a structured missing-request error.
    fn state(&self, uuid: Uuid) -> Result<&DisaggRequestState> {
        self.flow.state(uuid)
    }

    /// Borrow mutable request state with a structured missing-request error.
    fn state_mut(&mut self, uuid: Uuid) -> Result<&mut DisaggRequestState> {
        self.flow.state_mut(uuid)
    }

    fn acknowledge_action(
        &mut self,
        uuid: Uuid,
        action: IssuedHandoffAction,
        outcome: HandoffActionOutcome,
    ) -> Result<()> {
        self.flow
            .acknowledge_action(uuid, action, outcome, self.now_ms, &mut self.collector)
    }

    fn apply_handoff_fact(&mut self, uuid: Uuid, fact: HandoffFact) -> Result<()> {
        self.flow
            .apply_handoff_fact(uuid, fact, self.now_ms, &mut self.collector)
    }

    /// Submit a coordinator-owned prefill onto a selected worker.
    fn dispatch_prefill(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
    ) -> Result<()> {
        let (request, handoff_id) = self.flow.prepare_prefill_submission(uuid)?;
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
        self.flow.finish_prefill_submission(
            uuid,
            worker_idx,
            action,
            effects.lifecycle_events,
            self.now_ms,
            &mut self.collector,
            &mut self.stats,
        )
    }

    /// Accept destination ownership on a selected decode worker.
    fn reserve_destination(
        &mut self,
        uuid: Uuid,
        worker_idx: usize,
        action: IssuedHandoffAction,
    ) -> Result<()> {
        let (request, handoff_id) = self.flow.prepare_destination_reservation(uuid)?;
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
        let stored_hashes = self
            .flow
            .conformance_capture
            .as_ref()
            .map(|_| Observation::stored_hashes(&effects.engine_events))
            .unwrap_or_default();
        self.flow.finish_destination_reservation(
            uuid,
            worker_idx,
            action,
            &stored_hashes,
            effects.lifecycle_events,
            self.now_ms,
            &mut self.collector,
            &mut self.stats,
        )
    }

    fn record_prefill_placement(&mut self, placement: Placement) -> Result<()> {
        self.flow
            .record_prefill_placement(placement, &mut self.traffic, &mut self.collector)
    }

    fn dispatch_prefill_placements(&mut self, placements: Vec<Placement>) -> Result<()> {
        for placement in placements {
            let (uuid, worker_idx, action) = self.flow.take_prefill_placement(
                placement,
                &mut self.traffic,
                &mut self.collector,
            )?;
            self.dispatch_prefill(uuid, worker_idx, action)?;
        }
        Ok(())
    }

    fn record_decode_placement(&mut self, placement: Placement) -> Result<()> {
        self.flow
            .record_decode_placement(placement, &mut self.collector)
    }

    fn dispatch_decode_placements(&mut self, placements: Vec<Placement>) -> Result<()> {
        for placement in placements {
            let (uuid, worker_idx, action) = self
                .flow
                .take_decode_placement(placement, &mut self.collector)?;
            self.reserve_destination(uuid, worker_idx, action)?;
        }
        Ok(())
    }

    fn route_prefill(&mut self, uuid: Uuid, action: IssuedHandoffAction) -> Result<()> {
        self.state_mut(uuid)?.phase = DisaggPhase::QueuedPrefill;
        let metadata =
            Metadata::from_hashes(self.state_mut(uuid)?.take_replay_hashes()).for_prefill();
        let session_id = self.state(uuid)?.session_id().map(str::to_owned);
        let request = self.flow.state(uuid)?.request_payload()?;
        let effects = self
            .prefill_placement
            .place(request, metadata, session_id, self.now_ms)?;
        self.dispatch_prefill_placements(effects.released)?;
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                let routed = self.prefill_placement.is_router();
                self.state_mut(uuid)?.prefill_routed = routed;
                self.record_prefill_placement(placement)?;
                self.dispatch_prefill(uuid, placement.scheduler_id, action)?;
            }
            PlacementDecision::Queued => {
                let state = self.state_mut(uuid)?;
                state.pending_prefill_action = Some(action);
                state.prefill_routed = true;
            }
        }
        self.record_router_pending();
        Ok(())
    }

    fn route_destination(&mut self, uuid: Uuid, action: IssuedHandoffAction) -> Result<()> {
        self.state_mut(uuid)?.await_destination();
        // TODO: Keep the destination side compact through decode routing and
        // reservation once decode-block hashes can be derived without prompt
        // expansion and the scheduler accepts compact metadata. Destination-
        // first SGLang currently materializes here before prefill; source-first
        // vLLM has already materialized at prefill worker submission.
        self.state_mut(uuid)?.materialize_original_request()?;
        let session_id = self.state(uuid)?.session_id().map(str::to_owned);
        let request = self.flow.state(uuid)?.request_payload()?;
        let effects = self.decode_placement.place(
            request,
            Metadata::from_hashes(None),
            session_id,
            self.now_ms,
        )?;
        self.dispatch_decode_placements(effects.released)?;
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                let routed = self.decode_placement.is_router();
                self.state_mut(uuid)?.destination_routed = routed;
                self.record_decode_placement(placement)?;
                self.reserve_destination(uuid, placement.scheduler_id, action)?;
            }
            PlacementDecision::Queued => {
                let state = self.state_mut(uuid)?;
                state.pending_destination_action = Some(action);
                state.destination_routed = true;
            }
        }
        self.record_router_pending();
        Ok(())
    }

    fn drive_pending_actions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some((uuid, action)) = self.flow.action_queues.pop_pending() {
            match self.execute_action(uuid, action)? {
                ActionExecution::Applied => {
                    changed = true;
                }
                ActionExecution::WaitingForWorker { action, stage } => {
                    self.flow.action_queues.wait_for_worker(uuid, action, stage);
                }
                ActionExecution::Deferred {
                    action,
                    stage,
                    worker_idx,
                } => {
                    self.flow
                        .action_queues
                        .defer(uuid, action, stage, worker_idx);
                }
            }
        }
        Ok(changed)
    }

    fn wake_worker_waiters(&mut self, stage: SimulationWorkerStage) {
        self.flow.action_queues.wake_worker_waiters(stage);
    }

    fn wake_deferred_actions(&mut self, stage: SimulationWorkerStage, worker_idx: usize) {
        self.flow.action_queues.wake_deferred(stage, worker_idx);
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
                if let Some(transfer) = self.flow.start_transfer(
                    uuid,
                    issued,
                    delay_ms,
                    self.now_ms,
                    &mut self.collector,
                )? {
                    push_transfer_complete(
                        &mut self.events,
                        &mut self.next_event_seq,
                        transfer.at_ms,
                        transfer.handoff_id,
                    );
                    #[cfg(test)]
                    self.stats
                        .transition_log
                        .push(DisaggTransition::TransferQueued { uuid });
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
                let stored_hashes = self
                    .flow
                    .conformance_capture
                    .as_ref()
                    .map(|_| Observation::stored_hashes(&effects.engine_events))
                    .unwrap_or_default();
                self.flow.finish_destination_activation(
                    uuid,
                    issued,
                    &stored_hashes,
                    effects.lifecycle_events,
                    self.now_ms,
                    &mut self.collector,
                    &mut self.stats,
                )?;
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
                self.flow.record_source_release(uuid, &mut self.stats);
                self.apply_prefill_observations(effects.engine_events)?;
                self.flow.finish_source_release(
                    uuid,
                    issued,
                    outcome,
                    effects.lifecycle_events,
                    self.now_ms,
                    &mut self.collector,
                    &mut self.stats,
                )?;
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
                self.apply_prefill_observations(effects.engine_events)?;
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
        self.flow.process_lifecycle_events(
            events,
            self.now_ms,
            &mut self.collector,
            &mut self.stats,
        )
    }

    fn cancel_prefill_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.prefill_routed {
            return Ok(());
        }
        self.state_mut(uuid)?.pending_prefill_action = None;
        let placements = if self.prefill_placement.cancel_pending(uuid) {
            Vec::new()
        } else {
            self.prefill_placement.request_terminal(uuid, self.now_ms)?
        };
        self.state_mut(uuid)?.prefill_routed = false;
        self.record_router_pending();
        self.dispatch_prefill_placements(placements)
    }

    fn cancel_decode_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.destination_routed {
            return Ok(());
        }
        self.state_mut(uuid)?.pending_destination_action = None;
        let placements = if self.decode_placement.cancel_pending(uuid) {
            Vec::new()
        } else {
            self.decode_placement.request_terminal(uuid, self.now_ms)?
        };
        self.state_mut(uuid)?.destination_routed = false;
        self.record_router_pending();
        self.dispatch_decode_placements(placements)
    }

    fn complete_prefill_route(&mut self, uuid: Uuid) -> Result<()> {
        if !self.state(uuid)?.prefill_routed {
            return Ok(());
        }
        let placements = self.prefill_placement.request_terminal(uuid, self.now_ms)?;
        self.state_mut(uuid)?.prefill_routed = false;
        #[cfg(test)]
        {
            self.stats.prefill_router_freed_count += 1;
            self.stats
                .transition_log
                .push(DisaggTransition::PrefillFree { uuid });
        }
        self.record_router_pending();
        self.dispatch_prefill_placements(placements)
    }

    fn complete_handoff(&mut self, uuid: Uuid) -> Result<()> {
        match self.state(uuid)?.coordinator.completion() {
            Some(HandoffCompletion::Success) => {
                self.complete_prefill_route(uuid)?;
                self.flow
                    .complete_successful_handoff(uuid, self.now_ms, &mut self.stats)?;
            }
            Some(HandoffCompletion::Canceled) => {
                self.collector
                    .on_terminal(uuid, self.now_ms, ReplayTerminalStatus::Canceled);
                self.cancel_prefill_route(uuid)?;
                self.cancel_decode_route(uuid)?;
                self.finish_logical_request(uuid, true)?;
            }
            None => bail!("handoff completed without a terminal coordinator outcome"),
        }
        Ok(())
    }

    fn finish_logical_request(&mut self, uuid: Uuid, remove_actions: bool) -> Result<()> {
        self.flow.prepare_logical_finish(uuid, remove_actions)?;
        CoreAdmissionSource::on_terminal(&mut self.admission, uuid, self.now_ms, false)?;
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
        self.flow.retire_completed_request(uuid)
    }

    /// Admit one external request into prefill-side state, collector state, and optional router.
    fn on_external_arrival(
        &mut self,
        request: ReplayRequestPayload,
        arrival_time_ms: f64,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Result<Uuid> {
        self.flow.on_external_arrival(
            request,
            arrival_time_ms,
            replay_hashes,
            session_id,
            &mut self.collector,
        )
    }

    /// Return true once both stages, both routers, and all admissions are fully
    /// drained. Lingering `WorkerReady`/`PlannerTick` events (worker startup, a
    /// re-armed planner heartbeat) do not represent request work, so they do not
    /// keep the run alive — otherwise a recurring tick would never let `run()` exit.
    fn is_done(&self) -> bool {
        self.only_idle_events_remain()
            && self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.prefill_engine.is_drained()
            && self.decode_engine.is_drained()
            && self.flow.action_queues.is_empty()
            && self.flow.requests_by_handoff.is_empty()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// or `PlannerTick` events remain in the queue.
    fn is_workload_done(&self) -> bool {
        self.cluster_in_flight() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.prefill_engine.is_drained()
            && self.decode_engine.is_drained()
            && self.flow.action_queues.is_empty()
            && self.flow.requests_by_handoff.is_empty()
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
        let next = choose_next_timestamp(
            CoreAdmissionSource::next_ready_time_ms(&mut self.admission),
            next_event_ms,
        );
        #[cfg(feature = "kvbm-offload")]
        {
            let next_offload = choose_next_timestamp(
                self.prefill_engine.earliest_offload_deadline(),
                self.decode_engine.earliest_offload_deadline(),
            );
            choose_next_timestamp(next, next_offload)
        }
        #[cfg(not(feature = "kvbm-offload"))]
        {
            next
        }
    }

    fn apply_prefill_observations(&mut self, events: Observation::Batch) -> Result<()> {
        let placements = self.prefill_placement.observe(events, self.now_ms)?;
        self.dispatch_prefill_placements(placements)
    }

    #[cfg(feature = "kvbm-offload")]
    fn tick_offload_engines(&mut self) -> Result<bool> {
        let prefill = self.prefill_engine.tick_offload_engines(self.now_ms);
        let decode = self.decode_engine.tick_offload_engines(self.now_ms);
        let changed = prefill.progress.made_progress
            || decode.progress.made_progress
            || !prefill.lifecycle_events.is_empty()
            || !decode.lifecycle_events.is_empty();
        self.apply_prefill_observations(prefill.engine_events)?;
        if !decode.engine_events.is_empty() {
            tracing::debug!("offline disagg replay dropping decode-side offload router events");
        }
        self.process_lifecycle_events(prefill.lifecycle_events)?;
        self.process_lifecycle_events(decode.lifecycle_events)?;
        Ok(changed)
    }

    /// Process one prefill output signal, including router updates and decode handoff scheduling.
    fn process_prefill_signal(&mut self, signal: OutputSignal) -> Result<()> {
        match self
            .flow
            .inspect_prefill_signal(&signal, self.now_ms, &mut self.collector)?
        {
            PrefillSignalDisposition::Pending | PrefillSignalDisposition::Rejected => {
                return Ok(());
            }
            PrefillSignalDisposition::Completed => {}
        }

        if self.prefill_placement.is_router() {
            let prefill_complete_placements = self
                .prefill_placement
                .prefill_completed(signal.uuid, self.now_ms)?;
            #[cfg(test)]
            {
                self.stats.prefill_marked_count += 1;
                self.stats
                    .transition_log
                    .push(DisaggTransition::PrefillMarkCompleted { uuid: signal.uuid });
            }
            self.record_router_pending();
            self.dispatch_prefill_placements(prefill_complete_placements)?;
        }
        Ok(())
    }

    /// Process one decode output signal, including decode router frees and request completion.
    fn process_decode_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.completed {
            return Ok(());
        }

        let placements = if self.decode_placement.is_router() {
            let placements = self
                .decode_placement
                .request_terminal(signal.uuid, self.now_ms)?;
            self.state_mut(signal.uuid)?.destination_routed = false;
            #[cfg(test)]
            {
                self.stats.decode_router_freed_count += 1;
                self.stats
                    .transition_log
                    .push(DisaggTransition::DecodeFree { uuid: signal.uuid });
            }
            placements
        } else {
            Vec::new()
        };
        self.record_router_pending();
        self.flow.record_decode_terminal(
            &signal,
            self.now_ms,
            &mut self.collector,
            &mut self.traffic,
        )?;
        self.finish_logical_request(signal.uuid, false)?;
        self.dispatch_decode_placements(placements)?;
        Ok(())
    }

    /// Apply the side effects of a finished prefill pass.
    fn process_prefill_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        engine_events: Observation::Batch,
    ) -> Result<()> {
        self.apply_prefill_observations(engine_events)?;
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
        engine_events: Observation::Batch,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    ) -> Result<()> {
        if let Some(capture) = self.flow.conformance_capture.as_mut() {
            capture.record_after_activation(&Observation::stored_hashes(&engine_events));
        }
        let placements = self.decode_placement.observe(engine_events, self.now_ms)?;
        self.dispatch_decode_placements(placements)?;
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
                        self.prefill_fpm_buffer
                            .insert(payload.worker_idx, fpm, self.now_ms);
                    }
                    self.process_prefill_pass(
                        payload.worker_idx,
                        payload.completed_requests,
                        payload.output_signals,
                        payload.lifecycle_events,
                        payload.engine_events,
                    )?;
                }
                SimulationWorkerStage::Decode => {
                    let payload = self.decode_engine.on_scheduled_completion(payload)?;
                    self.wake_deferred_actions(SimulationWorkerStage::Decode, payload.worker_idx);
                    if self.collect_fpm
                        && let Some(fpm) = payload.fpm
                    {
                        self.decode_fpm_buffer
                            .insert(payload.worker_idx, fpm, self.now_ms);
                    }
                    self.process_decode_pass(
                        payload.output_signals,
                        payload.lifecycle_events,
                        payload.engine_events,
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
            let Some(uuid) = self.flow.requests_by_handoff.get(&handoff_id).copied() else {
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
        let cluster_in_flight = self.cluster_in_flight();
        for ready in
            CoreAdmissionSource::drain_ready(&mut self.admission, self.now_ms, cluster_in_flight)?
        {
            let ReadyArrival {
                request,
                arrival_time_ms,
                metadata,
                session_id,
                turn_index,
            } = ready;
            let session_metadata = session_id.clone().zip(turn_index);
            let uuid = self.on_external_arrival(
                request,
                arrival_time_ms,
                metadata.into_hashes(),
                session_id,
            )?;
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

    fn handle_prefill_engine_effects(
        &mut self,
        effects: EngineEffects<Observation::Batch>,
    ) -> Result<()> {
        self.record_prefill_admissions(effects.admissions);
        self.apply_prefill_observations(effects.pass_start_events)?;
        for payload in effects.immediate_completions {
            let payload = self.prefill_engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.prefill_fpm_buffer
                    .insert(payload.worker_idx, fpm, self.now_ms);
            }
            self.process_prefill_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
                payload.lifecycle_events,
                payload.engine_events,
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

    fn handle_decode_engine_effects(
        &mut self,
        effects: EngineEffects<Observation::Batch>,
    ) -> Result<()> {
        self.record_decode_admissions(effects.admissions)?;
        for payload in effects.immediate_completions {
            let payload = self.decode_engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.decode_fpm_buffer
                    .insert(payload.worker_idx, fpm, self.now_ms);
            }
            self.process_decode_pass(
                payload.output_signals,
                payload.lifecycle_events,
                payload.engine_events,
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
                        if self.collect_fpm {
                            self.prefill_fpm_buffer.activate_worker(
                                worker_id,
                                self.prefill_engine.dp_size(),
                                self.now_ms,
                            );
                        }
                        let topology =
                            self.prefill_engine
                                .worker_topology(worker_id)
                                .ok_or_else(|| {
                                    anyhow!(
                                        "ready prefill worker {worker_id} has no engine topology"
                                    )
                                })?;
                        let placements =
                            self.prefill_placement.worker_ready(topology, self.now_ms)?;
                        self.dispatch_prefill_placements(placements)?;
                        let placements = self.prefill_placement.topology_settled(self.now_ms)?;
                        self.dispatch_prefill_placements(placements)?;
                        self.wake_worker_waiters(SimulationWorkerStage::Prefill);
                        changed = true;
                    }
                }
                SimulationWorkerStage::Decode => {
                    if self.decode_engine.mark_worker_ready(worker_id) {
                        if self.collect_fpm {
                            self.decode_fpm_buffer.activate_worker(
                                worker_id,
                                self.decode_engine.dp_size(),
                                self.now_ms,
                            );
                        }
                        let topology =
                            self.decode_engine
                                .worker_topology(worker_id)
                                .ok_or_else(|| {
                                    anyhow!(
                                        "ready decode worker {worker_id} has no engine topology"
                                    )
                                })?;
                        let placements =
                            self.decode_placement.worker_ready(topology, self.now_ms)?;
                        self.dispatch_decode_placements(placements)?;
                        let placements = self.decode_placement.topology_settled(self.now_ms)?;
                        self.dispatch_decode_placements(placements)?;
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
            for worker_id in &removed_prefill {
                let placements = self.prefill_placement.worker_removed(
                    WorkerTopology {
                        worker_id: *worker_id,
                        scheduler_ids: Vec::new(),
                    },
                    self.now_ms,
                )?;
                self.dispatch_prefill_placements(placements)?;
            }
            changed |= !removed_prefill.is_empty();
            let removed_decode = self.decode_engine.try_remove_drained();
            for worker_id in &removed_decode {
                let placements = self.decode_placement.worker_removed(
                    WorkerTopology {
                        worker_id: *worker_id,
                        scheduler_ids: Vec::new(),
                    },
                    self.now_ms,
                )?;
                self.dispatch_decode_placements(placements)?;
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
                    if !self.flow.requests_by_handoff.contains_key(handoff_id)
            )
        }) {
            self.events.pop();
            self.flow.stale_transfer_events = self
                .flow
                .stale_transfer_events
                .checked_sub(1)
                .expect("stale transfer event count underflow");
            removed = true;
        }
        if self.flow.stale_transfer_events > 32
            && self.flow.stale_transfer_events.saturating_mul(2) > self.events.len()
        {
            let requests_by_handoff = &self.flow.requests_by_handoff;
            self.events.retain(|event| {
                !matches!(
                    &event.kind,
                    super::events::SimulationEventKind::TransferComplete { handoff_id }
                        if !requests_by_handoff.contains_key(handoff_id)
                )
            });
            self.flow.stale_transfer_events = 0;
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
            let active_prefill_ids = self.prefill_engine.active_group_ids();
            let active_decode_ids = self.decode_engine.active_group_ids();
            self.prefill_fpm_buffer.emit_idle_due(
                &active_prefill_ids,
                self.prefill_engine.dp_size(),
                self.now_ms,
            );
            self.decode_fpm_buffer.emit_idle_due(
                &active_decode_ids,
                self.decode_engine.dp_size(),
                self.now_ms,
            );
            let metrics = PlannerTickMetrics {
                now_ms: self.now_ms,
                prefill_fpm: self.prefill_fpm_buffer.take(),
                decode_fpm: self.decode_fpm_buffer.take(),
                traffic: self.traffic.drain(self.now_ms),
                active_prefill_ids,
                active_decode_ids,
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
                .flow
                .requests
                .values()
                .filter(|state| state.counted_in_flight)
                .count();
            assert_eq!(self.flow.logical_in_flight, counted);
            for state in self.flow.requests.values() {
                assert_eq!(
                    state.counted_in_flight,
                    !matches!(state.phase, DisaggPhase::CleanupPending | DisaggPhase::Done)
                );
            }
            self.stats.request_snapshots = self
                .flow
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

    #[cfg(test)]
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
                    if self.collect_fpm {
                        self.prefill_fpm_buffer.activate_worker(
                            id,
                            self.prefill_engine.dp_size(),
                            self.now_ms,
                        );
                    }
                    let topology = self
                        .prefill_engine
                        .worker_topology(id)
                        .ok_or_else(|| anyhow!("new prefill worker {id} has no engine topology"))?;
                    let placements = self.prefill_placement.worker_ready(topology, self.now_ms)?;
                    self.dispatch_prefill_placements(placements)?;
                }
            }
        }
        for id in newly_marked {
            let topology = self
                .prefill_engine
                .worker_topology(id)
                .unwrap_or(WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                });
            let placements = self
                .prefill_placement
                .worker_draining(topology, self.now_ms)?;
            self.dispatch_prefill_placements(placements)?;
        }
        for id in removed {
            let placements = self.prefill_placement.worker_removed(
                WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                },
                self.now_ms,
            )?;
            self.dispatch_prefill_placements(placements)?;
        }
        let placements = self.prefill_placement.topology_settled(self.now_ms)?;
        self.dispatch_prefill_placements(placements)?;
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
                    if self.collect_fpm {
                        self.decode_fpm_buffer.activate_worker(
                            id,
                            self.decode_engine.dp_size(),
                            self.now_ms,
                        );
                    }
                    let topology = self
                        .decode_engine
                        .worker_topology(id)
                        .ok_or_else(|| anyhow!("new decode worker {id} has no engine topology"))?;
                    let placements = self.decode_placement.worker_ready(topology, self.now_ms)?;
                    self.dispatch_decode_placements(placements)?;
                }
            }
        }
        for id in newly_marked {
            let topology = self
                .decode_engine
                .worker_topology(id)
                .unwrap_or(WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                });
            let placements = self
                .decode_placement
                .worker_draining(topology, self.now_ms)?;
            self.dispatch_decode_placements(placements)?;
        }
        for id in removed {
            let placements = self.decode_placement.worker_removed(
                WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                },
                self.now_ms,
            )?;
            self.dispatch_decode_placements(placements)?;
        }
        let placements = self.decode_placement.topology_settled(self.now_ms)?;
        self.dispatch_decode_placements(placements)?;
        if !added.is_empty() && decode_delay.is_none() {
            self.wake_worker_waiters(SimulationWorkerStage::Decode);
        }
        self.record_router_pending();
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
        self.prefill_fpm_buffer.take()
    }

    #[cfg(test)]
    fn drain_decode_fpm(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        self.decode_fpm_buffer.take()
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
            && self.flow.action_queues.is_empty()
            && self.prefill_placement.pending_count() == 0
            && self.decode_placement.pending_count() == 0
            && self.flow.requests.values().all(|state| {
                !state.counted_in_flight && !state.prefill_routed && !state.destination_routed
            });
        let capture = self
            .flow
            .conformance_capture
            .take()
            .ok_or_else(|| anyhow!("offline handoff conformance capture was not enabled"))?;

        self.progress.finish();
        let report = self.collector.finish();
        let conformance = NormalizedHandoffConformance {
            engine_type,
            order: self.flow.handoff_order,
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

#[cfg(test)]
#[path = "disagg_tests.rs"]
mod tests;
