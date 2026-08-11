// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::artifact::{ReplayArtifactRequest, ReplayArtifactSink};
pub(super) use super::components::ReplayMode;
use super::core::{
    AdmissionSource as CoreAdmissionSource, Placement, PlacementDecision, PlacementPolicy,
    WorkerTopology,
};
use super::events::{SimulationEvent, SimulationWorkerStage, WorkerCompletionPayload};
use super::evidence::{
    KvIngestBoundary, ReplayEvidenceCollector, WorkerLifecycleTransition,
    WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState, common_origin,
};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    next_timestamp as choose_next_timestamp, pop_ready_scaling_tick, pop_ready_worker_completions,
    pop_ready_worker_ready, push_scaling_tick, push_worker_completions, push_worker_ready,
};
use super::scaling::{LatestFpmBuffer, ReplayScalingPolicy, ReplayScalingSnapshot};
use super::{
    components::{
        AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, ReplayAdmissionMetadata,
        ReplayEngineObservation, ReplayReadyArrival, TrafficAccumulator,
    },
    state::AggRequestState,
};
use crate::replay::engine::ReplayRoleFactory;
use crate::replay::loadgen::ReplayRequestPayload;
use crate::replay::protocol::{DirectRequest, ForwardPassSnapshot, OutputSignal};
use crate::replay::{ReplayCaptureOptions, ReplayRequestPool};
use crate::replay::{ReplayTerminalStatus, TraceCollector};
use anyhow::{Context, bail};
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use uuid::Uuid;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct AggRuntimeStats;

pub(crate) struct AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    now_ms: f64,
    dp_size: u32,
    next_event_seq: u64,
    next_scaling_tick_ordinal: u64,
    admission: AdmissionQueue<Metadata>,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: EngineComponent<Observation>,
    collector: TraceCollector,
    artifact_sink: Option<ReplayArtifactSink>,
    evidence: ReplayEvidenceCollector,
    events: BinaryHeap<SimulationEvent<Observation::Batch>>,
    placement: PlacementPolicyImpl,
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
}

impl<PlacementPolicyImpl, Observation, Metadata>
    AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PlacementPolicy<ReplayRequestPayload, Metadata = Metadata, Observation = Observation::Batch>,
{
    pub(crate) fn new_composed(
        factory: ReplayRoleFactory,
        admission: AdmissionQueue<Metadata>,
        num_workers: usize,
        startup_time_ms: Option<f64>,
        create_placement: impl FnOnce(u32, Vec<WorkerTopology>) -> anyhow::Result<PlacementPolicyImpl>,
    ) -> anyhow::Result<Self> {
        let progress = ReplayProgress::new(
            CoreAdmissionSource::total_requests(&admission),
            "offline replay",
        );
        let dp_size = factory.dp_size();
        let gpus_per_worker = factory.gpus_per_worker()?;
        let engine = EngineComponent::<Observation>::new_with_factory(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            factory,
            num_workers,
            startup_time_ms,
        )?;
        let placement = create_placement(dp_size, engine.active_topology())?;

        // Aggregated replay has a single (decode) pool; record its GPUs/worker
        // so the report can express GPU-hours from the mocker's own parallelism.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(0, gpus_per_worker);

        Ok(Self {
            now_ms: 0.0,
            dp_size,
            next_event_seq: 0,
            next_scaling_tick_ordinal: 0,
            admission,
            requests: FxHashMap::default(),
            engine,
            collector,
            artifact_sink: None,
            evidence: ReplayEvidenceCollector::default(),
            events: BinaryHeap::new(),
            placement,
            progress,
            stats: AggRuntimeStats,
            fpm_buffer: LatestFpmBuffer::default(),
            traffic: TrafficAccumulator::new(),
            max_sim_time_ms: None,
            scaling_policy: None,
            collect_fpm: false,
        })
    }

    /// Toggle per-request record capture on the underlying collector. When
    /// `true`, the final `ReplayReport` returned from `run()` will
    /// have `per_request` populated. Default `false` (cheap).
    pub(crate) fn with_per_request_records(mut self, capture: bool) -> Self {
        self.collector.set_capture_per_request(capture);
        self
    }

    pub(crate) fn with_capture_options(mut self, options: ReplayCaptureOptions) -> Self {
        self.evidence = ReplayEvidenceCollector::new(options);
        self
    }

    pub(crate) fn with_artifact_sink(mut self, sink: ReplayArtifactSink) -> Self {
        self.engine.set_artifact_kv_capture(true);
        self.artifact_sink = Some(sink);
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
    pub(crate) fn with_max_sim_time_ms(mut self, ms: Option<f64>) -> Self {
        self.max_sim_time_ms = ms;
        self
    }

    /// Attach a scaling policy and enable tick-scoped FPM collection.
    pub(crate) fn with_scaling_policy(mut self, policy: Box<dyn ReplayScalingPolicy>) -> Self {
        self.collect_fpm = true;
        for worker_id in self.engine.active_group_ids() {
            self.fpm_buffer
                .activate_worker(worker_id, self.dp_size, self.now_ms);
        }
        self.scaling_policy = Some(policy);
        self
    }

    /// Count all requests currently consuming cluster capacity, including router-queued ones.
    fn cluster_in_flight(&self) -> usize {
        self.engine.in_flight() + self.placement.pending_count()
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

    /// Deliver a request to a worker and update the runtime's bookkeeping for that assignment.
    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        worker_idx: usize,
    ) -> anyhow::Result<()> {
        self.engine.dispatch(worker_idx, request, self.now_ms)?;
        // Aggregated replay uses a single pool. Treat the assignment as the
        // decode_worker_idx so per-request records consistently carry the
        // worker that served the request; prefill_worker_idx stays None,
        // signaling "no separate prefill pool".
        self.collector.on_decode_assigned(uuid, worker_idx);
        Ok(())
    }

    fn record_placement(&mut self, placement: Placement) {
        if let Some(sample) = placement.cache_sample {
            self.traffic
                .on_admission(sample.overlap_blocks, sample.isl_blocks);
        }
    }

    /// Materialize policy-released admissions into concrete worker dispatches.
    fn dispatch_placements(&mut self, placements: Vec<Placement>) -> anyhow::Result<()> {
        for placement in placements {
            self.record_placement(placement);
            let uuid = placement.request_id;
            let (logical_worker_id, dp_rank) = self
                .engine
                .rank_identity(placement.scheduler_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "offline replay placement references unknown scheduler {}",
                        placement.scheduler_id
                    )
                })?;
            self.collector.on_route_released(
                uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                logical_worker_id,
                placement.scheduler_id,
                dp_rank,
                placement.reported_overlap_tokens,
            );
            let request = self
                .requests
                .get_mut(&uuid)
                .ok_or_else(|| {
                    anyhow::anyhow!("offline replay missing queued request state for {uuid}")
                })?
                .take_queued_request(uuid)?;
            self.dispatch_to_worker(request, uuid, placement.scheduler_id)?;
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
        if matches!(self.admission.mode(), ReplayMode::Concurrency { .. }) {
            request.metadata_mut().arrival_timestamp_ms = Some(arrival_time_ms);
        }

        self.collector
            .on_arrival(uuid, arrival_time_ms, input_length, output_length);
        if let Some(context) = request.metadata().replay_context.as_ref() {
            self.collector.on_request_context(uuid, context);
        }
        self.traffic.on_arrival();

        let effects = self
            .placement
            .place(&request, metadata, session_id, self.now_ms)?;
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                if placement.request_id != uuid {
                    bail!(
                        "offline placement returned request {} while placing {uuid}",
                        placement.request_id
                    );
                }
                self.record_placement(placement);
                let (logical_worker_id, dp_rank) = self
                    .engine
                    .rank_identity(placement.scheduler_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay placement references unknown scheduler {}",
                            placement.scheduler_id
                        )
                    })?;
                self.collector.on_route_immediate(
                    uuid,
                    ReplayRequestPool::Agg,
                    logical_worker_id,
                    placement.scheduler_id,
                    dp_rank,
                    placement.reported_overlap_tokens,
                );
                self.requests.insert(
                    uuid,
                    AggRequestState::new_running(input_length, output_length),
                );
                self.dispatch_to_worker(
                    request.into_direct_request(),
                    uuid,
                    placement.scheduler_id,
                )?;
            }
            PlacementDecision::Queued => {
                self.collector
                    .on_route_queued(uuid, ReplayRequestPool::Agg, self.now_ms);
                self.requests
                    .insert(uuid, AggRequestState::new_queued(request));
            }
        }
        self.dispatch_placements(effects.released)?;
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
        choose_next_timestamp(
            CoreAdmissionSource::next_ready_time_ms(&mut self.admission),
            next_event_ms,
        )
    }

    /// Apply router-visible KV events at the phase chosen by the scheduler core.
    fn apply_engine_observations(
        &mut self,
        events: Observation::Batch,
        boundary: KvIngestBoundary,
    ) -> anyhow::Result<()> {
        if let Some(event_count) = Observation::kv_ingest_event_count(&events) {
            self.evidence.record_kv_ingest(
                WorkerPool::Agg,
                boundary,
                self.now_ms,
                event_count,
                |encoder| Observation::encode_kv_ingest(&events, encoder),
            )?;
        }
        let placements = self.placement.observe(events, self.now_ms)?;
        self.dispatch_placements(placements)
    }

    /// Consume one output signal, updating router state, collector state, and completion counts.
    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        if let Some(token_id) = signal.token_id {
            CoreAdmissionSource::on_output_token(&mut self.admission, signal.uuid, token_id)?;
            self.collector.on_token(signal.uuid, self.now_ms);
        }
        if signal.completed {
            let status = if signal.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            self.collector.on_terminal(signal.uuid, self.now_ms, status);
            let placements = self.placement.request_terminal(signal.uuid, self.now_ms)?;
            let removed_state = self.requests.remove(&signal.uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?;
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
            CoreAdmissionSource::on_terminal(
                &mut self.admission,
                signal.uuid,
                self.now_ms,
                signal.rejected,
            )?;
            self.progress.inc_completed();
            self.dispatch_placements(placements)?;
            return Ok(());
        }

        let already_marked = self
            .requests
            .get(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed;
        if already_marked {
            return Ok(());
        }

        self.requests
            .get_mut(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed = true;
        let placements = self.placement.prefill_completed(signal.uuid, self.now_ms)?;
        self.dispatch_placements(placements)?;

        Ok(())
    }

    /// Apply one completed pass: free request slots, publish KV events, and handle outputs.
    fn process_completed_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        engine_events: Observation::Batch,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    ) -> anyhow::Result<()> {
        self.apply_engine_observations(engine_events, KvIngestBoundary::PassEnd)?;
        self.traffic
            .on_accept_length_sample(accept_length_output_tokens, accept_length_decode_forwards);
        for signal in output_signals {
            self.process_output_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        // TODO: Same-time DP-rank completions settle in deterministic event order, and
        // each pass may drain router-pending work before its sibling ranks are applied.
        // Preserve this lower-rank-first tie-break for now. Atomic settlement requires
        // splitting router state mutation from pending-admission draining.
        while let Some(completion) = pop_ready_worker_completions(&mut self.events, self.now_ms) {
            for payload in self
                .engine
                .on_scheduled_completion(completion, self.now_ms)?
            {
                self.process_worker_completion_payload(payload)?;
            }
            changed = true;
        }

        Ok(changed)
    }

    fn process_worker_completion_payload(
        &mut self,
        payload: WorkerCompletionPayload<Observation::Batch>,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
        if let Some(sink) = &self.artifact_sink {
            sink.record_pass_completion_kv_events(
                payload.pass_started_at_ms,
                self.now_ms,
                payload
                    .artifact_pass_end_kv_events
                    .as_deref()
                    .unwrap_or_default(),
            )?;
            sink.record_outputs(self.now_ms, &payload.output_signals)?;
        }
        if self.collect_fpm
            && let Some(fpm) = payload.fpm
        {
            self.record_fpm(payload.worker_idx, fpm)?;
        }
        self.process_completed_pass(
            payload.worker_idx,
            payload.completed_requests,
            payload.output_signals,
            payload.engine_events,
            payload.accept_length_output_tokens,
            payload.accept_length_decode_forwards,
        )
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> anyhow::Result<bool> {
        let mut released_any = false;
        let cluster_in_flight = self.cluster_in_flight();
        for ready in self.admission.drain_ready_compact(
            self.now_ms,
            cluster_in_flight,
            self.artifact_sink.is_some(),
        )? {
            let ReplayReadyArrival {
                request,
                arrival_time_ms,
                scheduled_ready_at_ms,
                metadata,
                replay_hashes,
                session_id,
                turn_index,
            } = ready;
            let input_length = request.input_length();
            let output_length = request.metadata().effective_max_output_tokens();
            let session_metadata = session_id.clone().zip(turn_index);
            let uuid = self.assign_request(request, arrival_time_ms, metadata, session_id)?;
            if let Some(sink) = &self.artifact_sink {
                sink.record_request(ReplayArtifactRequest {
                    request_id: uuid,
                    observed_at_ms: self.now_ms,
                    scheduled_ready_at_ms,
                    input_length,
                    output_length,
                    replay_hashes,
                })?;
            }
            if let Some((session_id, turn_index)) = session_metadata {
                self.collector
                    .on_session_metadata(uuid, session_id, turn_index);
            }
            released_any = true;
        }
        Ok(released_any)
    }

    /// Start passes on every idle worker that can make progress at the current timestamp.
    fn drive_ready_workers(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        loop {
            let effects = self
                .engine
                .drive_ready(self.now_ms, Some(&mut self.collector))?;
            if effects.is_empty() {
                return Ok(changed);
            }
            changed = true;
            self.handle_engine_effects(effects)?;
        }
    }

    fn handle_engine_effects(
        &mut self,
        effects: EngineEffects<Observation::Batch>,
    ) -> anyhow::Result<()> {
        // Native scheduling is engine-owned now, so admissions cross the
        // generalized-engine boundary as pass-start effects. Preserve the old
        // scheduler contract by making them visible to Replay's collector at
        // the same virtual timestamp before any completion is processed.
        for admission in effects.admissions {
            self.collector
                .on_admit(admission.uuid, self.now_ms, admission.reused_input_tokens);
            self.collector.on_pool_admission(
                admission.uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                admission.reused_input_tokens,
            );
            self.evidence
                .record_pressure_readmission(admission.uuid, WorkerPool::Agg, self.now_ms);
        }
        for pressure in effects.pressure_events {
            self.evidence.record_native_pressure(
                &mut self.collector,
                WorkerPool::Agg,
                pressure.worker_id,
                pressure.dp_rank,
                pressure.event,
            );
        }
        if let (Some(sink), Some(pass_start)) = (&self.artifact_sink, effects.artifact_pass_start) {
            sink.record_pass_start_kv_events(pass_start.at_ms, &pass_start.kv_events)?;
        }
        self.apply_engine_observations(effects.pass_start_events, KvIngestBoundary::PassStart)?;
        for payload in effects.immediate_completions {
            self.process_worker_completion_payload(payload)?;
        }
        if let Some(scheduled) = effects.scheduled_completion {
            push_worker_completions(&mut self.events, &mut self.next_event_seq, scheduled);
        }
        Ok(())
    }

    /// Activate workers whose startup period has elapsed at the current timestamp.
    fn apply_worker_ready_events(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some((stage, worker_id)) = pop_ready_worker_ready(&mut self.events, self.now_ms) {
            debug_assert_eq!(stage, SimulationWorkerStage::Aggregated);
            if self.engine.mark_worker_ready(worker_id) {
                if self.collect_fpm {
                    self.fpm_buffer
                        .activate_worker(worker_id, self.dp_size, self.now_ms);
                }
                let topology = self.engine.worker_topology(worker_id).ok_or_else(|| {
                    anyhow::anyhow!("ready worker {worker_id} has no engine topology")
                })?;
                let placements = self.placement.worker_ready(topology, self.now_ms)?;
                let mut released = placements
                    .iter()
                    .map(|placement| placement.request_id)
                    .collect::<Vec<_>>();
                self.dispatch_placements(placements)?;
                let placements = self.placement.topology_settled(self.now_ms)?;
                released.extend(placements.iter().map(|placement| placement.request_id));
                self.dispatch_placements(placements)?;
                let origin = self.evidence.startup_origin(WorkerPool::Agg, worker_id);
                let state = self.lifecycle_state();
                self.evidence.record_lifecycle_operation(
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
                    state,
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
    fn drain_current_timestamp(&mut self) -> anyhow::Result<()> {
        loop {
            let mut changed = false;
            changed |= self.apply_worker_completions()?;
            changed |= self.apply_worker_ready_events()?;
            changed |= self.release_ready_arrivals()?;
            changed |= self.drive_ready_workers()?;
            let removed = self
                .engine
                .try_remove_drained()
                .context("failed to remove drained aggregated workers")?;
            let mut released = Vec::new();
            for worker_id in &removed {
                let placements = self.placement.worker_removed(
                    WorkerTopology {
                        worker_id: *worker_id,
                        scheduler_ids: Vec::new(),
                    },
                    self.now_ms,
                )?;
                released.extend(placements.iter().map(|placement| placement.request_id));
                self.dispatch_placements(placements)?;
            }
            if !removed.is_empty() {
                let origin = common_origin(removed.iter().filter_map(|worker_id| {
                    self.evidence.drain_origin(WorkerPool::Agg, *worker_id)
                }));
                let transitions = removed
                    .iter()
                    .map(|worker_id| WorkerLifecycleTransition {
                        worker_id: *worker_id,
                        transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                        prior_state: Some("draining"),
                        state: "removed",
                        reason: None,
                        origin_operation_ordinal: self
                            .evidence
                            .drain_origin(WorkerPool::Agg, *worker_id),
                    })
                    .collect();
                let state = self.lifecycle_state();
                self.evidence.record_lifecycle_operation(
                    self.now_ms,
                    WorkerPool::Agg,
                    "drain_settlement",
                    None,
                    origin,
                    transitions,
                    state,
                    released,
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
        }

        Ok(())
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
            let Some(mut policy) = self.scaling_policy.take() else {
                bail!("scaling tick fired without a policy");
            };
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

    /// Advance the sim clock to `new_now_ms`, integrating provisioned
    /// worker-seconds over the interval just elapsed. `worker_count()` counts
    /// active + starting-up + draining workers, so this captures the startup
    /// ramp and the scale-down drain tail. Aggregated replay has no separate
    /// prefill pool, so it reports through the decode role (prefill = 0).
    fn advance_now_ms(&mut self, new_now_ms: f64) {
        let dt_ms = (new_now_ms - self.now_ms).max(0.0);
        if dt_ms > 0.0 {
            let decode_worker_seconds = self.engine.worker_count() as f64 * dt_ms / 1000.0;
            self.collector
                .add_worker_seconds(0.0, decode_worker_seconds);
        }
        self.now_ms = new_now_ms;
    }

    fn apply_scaling_with_tick(
        &mut self,
        target_workers: usize,
        planner_tick_ordinal: Option<u64>,
    ) -> anyhow::Result<()> {
        if target_workers != self.engine.non_draining_group_count() {
            self.collector.clear_static_worker_count();
        }
        let starting_before = self.engine.starting_group_ids();
        let (added, newly_marked, removed) = self
            .engine
            .apply_target_count(target_workers)
            .with_context(|| {
                format!("failed to apply aggregated worker target {target_workers}")
            })?;
        let startup_delay_ms = self.engine.startup_time_ms();
        let mut released = Vec::new();

        for &id in &added {
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
                    let placements = self.placement.worker_ready(topology, self.now_ms)?;
                    released.extend(placements.iter().map(|placement| placement.request_id));
                    self.dispatch_placements(placements)?;
                }
            }
        }

        for &id in &newly_marked {
            let topology = self.engine.worker_topology(id).unwrap_or(WorkerTopology {
                worker_id: id,
                scheduler_ids: Vec::new(),
            });
            let placements = self.placement.worker_draining(topology, self.now_ms)?;
            released.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        for &id in &removed {
            let placements = self.placement.worker_removed(
                WorkerTopology {
                    worker_id: id,
                    scheduler_ids: Vec::new(),
                },
                self.now_ms,
            )?;
            released.extend(placements.iter().map(|placement| placement.request_id));
            self.dispatch_placements(placements)?;
        }
        let placements = self.placement.topology_settled(self.now_ms)?;
        released.extend(placements.iter().map(|placement| placement.request_id));
        self.dispatch_placements(placements)?;
        self.record_scale_lifecycle(
            &added,
            &newly_marked,
            &removed,
            &starting_before,
            startup_delay_ms.is_some(),
            planner_tick_ordinal,
            released,
        );
        Ok(())
    }

    fn lifecycle_state(&self) -> WorkerPoolState {
        WorkerPoolState {
            active: self.engine.active_group_ids(),
            starting: self.engine.starting_group_ids(),
            draining: self.engine.draining_group_ids(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_scale_lifecycle(
        &mut self,
        added: &[usize],
        newly_draining: &[usize],
        removed: &[usize],
        starting_before: &[usize],
        delayed_startup: bool,
        planner_tick_ordinal: Option<u64>,
        released: Vec<Uuid>,
    ) {
        if !self.evidence.options().capture_lifecycle_evidence {
            return;
        }
        let mut transitions = added
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
            })
            .collect::<Vec<_>>();
        transitions.extend(
            newly_draining
                .iter()
                .map(|worker_id| WorkerLifecycleTransition {
                    worker_id: *worker_id,
                    transition: WorkerLifecycleTransitionKind::WorkerDraining,
                    prior_state: Some("active"),
                    state: "draining",
                    reason: None,
                    origin_operation_ordinal: None,
                }),
        );
        transitions.extend(removed.iter().map(|worker_id| {
            let cancelled = starting_before.binary_search(worker_id).is_ok();
            WorkerLifecycleTransition {
                worker_id: *worker_id,
                transition: WorkerLifecycleTransitionKind::WorkerRemoved,
                prior_state: Some(if cancelled { "starting" } else { "draining" }),
                state: "removed",
                reason: cancelled.then_some("startup_cancelled"),
                origin_operation_ordinal: if cancelled {
                    self.evidence.startup_origin(WorkerPool::Agg, *worker_id)
                } else {
                    self.evidence.drain_origin(WorkerPool::Agg, *worker_id)
                },
            }
        }));
        let origin = common_origin(
            removed
                .iter()
                .filter(|worker_id| starting_before.binary_search(worker_id).is_ok())
                .filter_map(|worker_id| self.evidence.startup_origin(WorkerPool::Agg, *worker_id)),
        );
        let state = self.lifecycle_state();
        self.evidence.record_lifecycle_operation(
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
            state,
            released,
        );
    }

    /// Run the aggregated offline replay until all arrivals and worker work are exhausted.
    /// If `max_sim_time_ms` is set, exits gracefully when the next scheduled
    /// timestamp would exceed that cap; in-flight requests at that point are
    /// reported as incomplete.
    pub(crate) fn run(mut self) -> anyhow::Result<(TraceCollector, AggRuntimeStats)> {
        if let Some(cap_ms) = self.max_sim_time_ms
            && (!cap_ms.is_finite() || cap_ms < 0.0)
        {
            bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
        }
        self.drain_current_timestamp()?;
        // With a planner attached, seed the recurring heartbeat; ticks then fire as
        // events inside drain_current_timestamp.
        self.seed_first_scaling_tick()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                // Aggregated workers have no external handoff dependency. If
                // the event queue is empty while an engine still owns a
                // request, the preceding zero-duration pass lost its terminal
                // effect (or can otherwise never wake) and is a liveness
                // invariant failure, not ordinary quiescence.
                if self.engine.has_runnable_worker() || self.engine.in_flight() > 0 {
                    bail!(
                        "offline replay detected an effect-free zero-duration pass with {} in-flight requests remaining",
                        self.cluster_in_flight()
                    );
                }
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
            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
        }

        self.progress.finish();
        self.collector.set_runtime_evidence(self.evidence.finish());
        Ok((self.collector, self.stats))
    }
}
