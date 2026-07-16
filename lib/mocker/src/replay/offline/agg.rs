// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use super::components::OfflineRouterSnapshot;
pub(super) use super::components::ReplayMode;
#[cfg(test)]
use super::components::TrafficStats;
use super::events::{SimulationEvent, SimulationWorkerStage};
use super::planner_hook::{LatestFpmBuffer, PlannerHook, PlannerTickMetrics};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    next_timestamp as choose_next_timestamp, pop_ready_planner_tick, pop_ready_worker_completion,
    pop_ready_worker_ready, push_planner_tick, push_worker_completion, push_worker_ready,
};
#[cfg(test)]
use super::state::AggRequestPhase;
#[cfg(test)]
use super::state::OfflineWorkerSnapshot;
use super::{
    components::{
        AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, OfflineReplayRouter,
        ReadyArrival, ScheduledWorkerCompletion, TrafficAccumulator, WorkerAdmission,
    },
    state::AggRequestState,
};
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::{
    ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayTerminalStatus, SlaThresholds,
    TraceCollector, session_affinity::ReplaySessionSimulationOptions,
};
use anyhow::bail;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RouterEvent;
use rustc_hash::FxHashMap;
#[cfg(test)]
use std::collections::HashMap;
use std::collections::{BinaryHeap, VecDeque};
use uuid::Uuid;

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

pub(in crate::replay) struct AggRuntime {
    now_ms: f64,
    next_worker_idx: usize,
    next_dp_rank_by_worker: FxHashMap<usize, u32>,
    dp_size: u32,
    next_event_seq: u64,
    admission: AdmissionQueue,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: EngineComponent,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    router: Option<OfflineReplayRouter>,
    progress: ReplayProgress,
    stats: AggRuntimeStats,
    /// Latest forward pass metric per worker/rank since the previous planner tick.
    fpm_buffer: LatestFpmBuffer,
    /// Traffic statistics accumulated between planner ticks.
    traffic: TrafficAccumulator,
    /// Optional cap on simulated wall-clock time. When set, `run()` exits
    /// gracefully once the next scheduled timestamp exceeds this cap, leaving
    /// any in-flight requests as incomplete in the report.
    max_sim_time_ms: Option<f64>,
    /// Planner hook. When set, `run()` seeds a recurring `PlannerTick` event and
    /// calls back into the planner at each tick (the unified replacement for the
    /// old Python-driven `advance_to` stepping loop).
    planner_hook: Option<Box<dyn PlannerHook>>,
    /// Whether to retain the latest FPM snapshot per worker/rank. Only the planner
    /// consumes them, so the plain `run()` path leaves this `false`.
    collect_fpm: bool,
    #[cfg(test)]
    worker_active_requests: Vec<Vec<Uuid>>,
    #[cfg(test)]
    stepped: bool,
}

impl AggRuntime {
    /// Create an aggregated offline runtime seeded from an explicit request queue.
    pub(in crate::replay) fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(
            args,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::new_requests(pending, mode),
            num_workers,
            router_mode,
            ReplaySessionSimulationOptions::default(),
        )
    }

    /// Create an aggregated offline runtime whose admissions come from a workload driver.
    pub(in crate::replay) fn new_workload(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        driver: WorkloadDriver,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(
            args,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::new_workload(driver, mode),
            num_workers,
            router_mode,
            ReplaySessionSimulationOptions::default(),
        )
    }

    pub(in crate::replay) fn new_workload_with_session_affinity(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        driver: WorkloadDriver,
        num_workers: usize,
        mode: ReplayMode,
        session_options: ReplaySessionSimulationOptions,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(
            args,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::new_workload(driver, mode),
            num_workers,
            ReplayRouterMode::KvRouter,
            session_options,
        )
    }

    /// Shared constructor for both raw-request and workload-driven admissions.
    fn new_with_source(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        admission: AdmissionQueue,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        session_options: ReplaySessionSimulationOptions,
    ) -> anyhow::Result<Self> {
        let args = args.clone().normalized()?;
        let progress = ReplayProgress::new(admission.total_requests(), "offline replay");
        let router = match router_mode {
            ReplayRouterMode::RoundRobin => None,
            ReplayRouterMode::KvRouter => Some(
                OfflineReplayRouter::new_with_session_affinity_and_observation(
                    &args,
                    router_config,
                    prefill_load_estimator,
                    num_workers,
                    session_options.affinity_mode,
                    session_options.kv_observation_mode,
                )?,
            ),
        };
        let capture_kv_events = router.is_some();
        let mut engine = EngineComponent::new_ranked(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            args.clone(),
            num_workers,
            capture_kv_events,
        );
        engine.set_scaling_args(args.clone(), capture_kv_events);

        // Aggregated replay has a single (decode) pool; record its GPUs/worker
        // so the report can express GPU-hours from the mocker's own parallelism.
        let mut collector = TraceCollector::default();
        collector.set_gpus_per_worker(0, args.aic_gpus_per_worker());

        Ok(Self {
            now_ms: 0.0,
            next_worker_idx: 0,
            next_dp_rank_by_worker: FxHashMap::default(),
            dp_size: args.dp_size.max(1),
            next_event_seq: 0,
            admission,
            requests: FxHashMap::default(),
            engine,
            collector,
            events: BinaryHeap::new(),
            router,
            progress,
            #[cfg(test)]
            stats: AggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: AggRuntimeStats,
            fpm_buffer: LatestFpmBuffer::default(),
            traffic: TrafficAccumulator::new(),
            max_sim_time_ms: None,
            planner_hook: None,
            collect_fpm: false,
            #[cfg(test)]
            worker_active_requests: vec![
                Vec::new();
                num_workers.saturating_mul(args.dp_size.max(1) as usize)
            ],
            #[cfg(test)]
            stepped: false,
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
        for worker_id in self.engine.active_group_ids() {
            self.fpm_buffer
                .activate_worker(worker_id, self.dp_size, self.now_ms);
        }
        self.planner_hook = Some(hook);
        self
    }

    #[cfg(test)]
    fn with_fpm_capture(mut self) -> Self {
        self.collect_fpm = true;
        self
    }

    /// Count all requests currently consuming cluster capacity, including router-queued ones.
    fn cluster_in_flight(&self) -> usize {
        self.engine.in_flight()
            + self
                .router
                .as_ref()
                .map_or(0, OfflineReplayRouter::pending_count)
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
        let Some(router) = self.router.as_ref() else {
            return;
        };
        #[cfg(test)]
        {
            self.stats.max_router_pending_count = self
                .stats
                .max_router_pending_count
                .max(router.pending_count());
        }
    }

    /// Pick the next active worker in round-robin order.
    fn next_worker(&mut self) -> usize {
        let active = self.engine.active_group_ids();
        debug_assert!(!active.is_empty(), "no active workers for round-robin");
        let idx = self.next_worker_idx % active.len();
        self.next_worker_idx = idx + 1;
        let worker_id = active[idx];
        let dp_rank = self.next_dp_rank_by_worker.entry(worker_id).or_default();
        let rank = *dp_rank % self.dp_size;
        *dp_rank = rank + 1;
        self.engine
            .rank_id(worker_id, rank)
            .expect("active worker must contain every configured DP rank")
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
    /// rank-local scheduler snapshot to the planner bridge.
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
        self.engine.dispatch(worker_idx, request)?;
        self.record_dispatch(uuid, worker_idx);
        // Aggregated replay uses a single pool. Treat the assignment as the
        // decode_worker_idx so per-request records consistently carry the
        // worker that served the request; prefill_worker_idx stays None,
        // signaling "no separate prefill pool".
        self.collector.on_decode_assigned(uuid, worker_idx);
        #[cfg(test)]
        self.worker_active_requests[worker_idx].push(uuid);
        Ok(())
    }

    /// Materialize router admissions into concrete worker dispatches.
    fn dispatch_router_admissions(
        &mut self,
        admissions: Vec<WorkerAdmission>,
    ) -> anyhow::Result<()> {
        for WorkerAdmission {
            uuid,
            worker_idx,
            overlap_blocks,
            isl_blocks,
        } in admissions
        {
            self.traffic.on_admission(overlap_blocks, isl_blocks);
            #[cfg(test)]
            self.stats.overlap_history.push(overlap_blocks);
            let request = self
                .requests
                .get_mut(&uuid)
                .ok_or_else(|| {
                    anyhow::anyhow!("offline replay missing queued request state for {uuid}")
                })?
                .take_queued_request(uuid)?;
            self.dispatch_to_worker(request, uuid, worker_idx)?;
        }
        Ok(())
    }

    /// Admit one external request into the collector, optional router, and worker pool.
    fn assign_request(
        &mut self,
        mut request: DirectRequest,
        arrival_time_ms: f64,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> anyhow::Result<Uuid> {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        request.uuid = Some(uuid);
        if matches!(self.admission.mode(), ReplayMode::Concurrency { .. }) {
            request.arrival_timestamp_ms = Some(arrival_time_ms);
        }

        self.collector.on_arrival(
            uuid,
            arrival_time_ms,
            request.tokens.len(),
            request.max_output_tokens,
        );

        if self.router.is_none() {
            self.requests.insert(
                uuid,
                AggRequestState::new_running(request.tokens.len(), request.max_output_tokens),
            );
            let worker_idx = self.next_worker();
            self.dispatch_to_worker(request, uuid, worker_idx)?;
            return Ok(uuid);
        }
        let admissions = {
            let router = self.router.as_mut().expect("router presence checked above");
            router
                .on_request_arrival_for_session(&request, replay_hashes, session_id, self.now_ms)?
                .admissions
        };
        self.requests
            .insert(uuid, AggRequestState::new_queued(request));
        self.record_router_pending();
        self.dispatch_router_admissions(admissions)?;
        self.record_in_flight_peak();
        Ok(uuid)
    }

    /// Return true once no request work remains. Lingering `WorkerReady`/`PlannerTick`
    /// events (worker startup, a re-armed planner heartbeat) carry no work and do not
    /// keep the run alive — otherwise a recurring tick would never let `run()` exit.
    fn is_done(&self) -> bool {
        self.only_idle_events_remain()
            && self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.engine.is_drained()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// or `PlannerTick` events remain in the queue. Lingering startup events for
    /// workers that will never receive requests should not block completion.
    fn is_workload_done(&self) -> bool {
        self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.engine.is_drained()
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

    /// Pick the next logical timestamp from either arrivals or scheduled worker completions.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let next = choose_next_timestamp(self.admission.next_ready_time_ms(), next_event_ms);
        #[cfg(feature = "kvbm-offload")]
        {
            return choose_next_timestamp(next, self.engine.earliest_offload_deadline());
        }
        #[cfg(not(feature = "kvbm-offload"))]
        next
    }

    /// Apply router-visible KV events at the phase chosen by the scheduler core.
    fn apply_router_events(&mut self, events: Vec<RouterEvent>) -> anyhow::Result<()> {
        let Some(router) = self.router.as_mut() else {
            return Ok(());
        };
        let effects = router.on_kv_events(events)?;
        if !effects.admissions.is_empty() {
            bail!("offline replay router KV event application must not admit requests");
        }
        Ok(())
    }

    #[cfg(feature = "kvbm-offload")]
    fn tick_offload_engines(&mut self) -> anyhow::Result<bool> {
        let crate::scheduler::OffloadTickEffects {
            kv_events,
            lifecycle_events,
        } = self.engine.tick_offload_engines(self.now_ms);
        if !lifecycle_events.is_empty() {
            bail!(
                "aggregated replay received {} handoff lifecycle events from an offload tick",
                lifecycle_events.len()
            );
        }
        let changed = !kv_events.is_empty();
        self.apply_router_events(kv_events)?;
        Ok(changed)
    }

    /// Consume one output signal, updating router state, collector state, and completion counts.
    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        let mut admissions = Vec::new();
        if let Some(token_id) = signal.token_id {
            self.admission.on_output_token(signal.uuid, token_id)?;
        }
        if signal.completed {
            let status = if signal.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            self.collector.on_terminal(signal.uuid, status);
            #[cfg(test)]
            self.remove_active_request(signal.uuid);
            if let Some(router) = self.router.as_mut() {
                admissions = router
                    .on_request_completed(signal.uuid, self.now_ms)?
                    .admissions;
                #[cfg(test)]
                {
                    self.stats.router_freed_count += 1;
                }
                self.record_router_pending();
            }
            let removed_state = self.requests.remove(&signal.uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?;
            // Rejected requests never ran: keep them out of the planner-facing
            // traffic deltas (they still free their slot and advance below).
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
                self.traffic.on_request(
                    removed_state.input_tokens,
                    actual_output_tokens,
                    latencies,
                );
            }
            self.admission
                .on_request_terminal(signal.uuid, self.now_ms, signal.rejected)?;
            self.progress.inc_completed();
            self.dispatch_router_admissions(admissions)?;
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
        if let Some(router) = self.router.as_mut() {
            admissions = router
                .on_prefill_completed(signal.uuid, self.now_ms)?
                .admissions;
            #[cfg(test)]
            {
                self.stats.prefill_marked_count += 1;
            }
            self.record_router_pending();
        }
        self.dispatch_router_admissions(admissions)?;

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

    /// Apply one completed pass: free request slots, publish KV events, and handle outputs.
    fn process_completed_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<RouterEvent>,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    ) -> anyhow::Result<()> {
        self.apply_router_events(kv_events)?;
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
        while let Some(payload) = pop_ready_worker_completion(&mut self.events, self.now_ms) {
            debug_assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
            let payload = self.engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.record_fpm(payload.worker_idx, fpm)?;
            }
            self.process_completed_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
                payload.kv_events,
                payload.accept_length_output_tokens,
                payload.accept_length_decode_forwards,
            )?;
            changed = true;
        }

        Ok(changed)
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> anyhow::Result<bool> {
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
            let session_metadata = session_id.clone().zip(turn_index);
            let uuid = self.assign_request(request, arrival_time_ms, replay_hashes, session_id)?;
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

    fn handle_engine_effects(&mut self, effects: EngineEffects) -> anyhow::Result<()> {
        self.apply_router_events(effects.pass_start_kv_events)?;
        for payload in effects.immediate_completions {
            let payload = self.engine.on_scheduled_completion(payload)?;
            if self.collect_fpm
                && let Some(fpm) = payload.fpm
            {
                self.record_fpm(payload.worker_idx, fpm)?;
            }
            self.process_completed_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
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
    fn apply_worker_ready_events(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some((stage, worker_id)) = pop_ready_worker_ready(&mut self.events, self.now_ms) {
            debug_assert_eq!(stage, SimulationWorkerStage::Aggregated);
            if self.engine.mark_worker_ready(worker_id) {
                if self.collect_fpm {
                    self.fpm_buffer
                        .activate_worker(worker_id, self.dp_size, self.now_ms);
                }
                if let Some(router) = self.router.as_mut() {
                    router.add_worker(worker_id)?;
                    // Drain any requests that were queued while all workers
                    // were busy — the new worker may have capacity for them.
                    let effects = router.try_drain_pending(self.now_ms)?;
                    self.dispatch_router_admissions(effects.admissions)?;
                }
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
            #[cfg_attr(not(feature = "kvbm-offload"), allow(unused_mut))]
            let mut changed = false;
            #[cfg(feature = "kvbm-offload")]
            {
                changed |= self.tick_offload_engines()?;
            }
            changed |= self.apply_worker_completions()?;
            changed |= self.apply_worker_ready_events()?;
            changed |= self.release_ready_arrivals()?;
            changed |= self.drive_ready_workers()?;
            let removed = self.engine.try_remove_drained();
            for worker_id in &removed {
                self.next_dp_rank_by_worker.remove(worker_id);
            }
            if let Some(router) = self.router.as_mut() {
                for worker_id in &removed {
                    router.finalize_worker_removal(*worker_id)?;
                }
            }
            changed |= !removed.is_empty();
            // Planner ticks fire LAST so the planner observes a fully settled
            // timestamp; any scaling it applies is picked up by the next iteration.
            if self.planner_hook.is_some() {
                changed |= self.apply_planner_ticks()?;
            }

            if !changed {
                break;
            }
        }

        Ok(())
    }

    /// Seed the first `PlannerTick` from the hook's requested start time (a
    /// non-finite time means "no tick" and is skipped).
    fn seed_first_planner_tick(&mut self) -> anyhow::Result<()> {
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
            // No tick will ever fire to drain the FPM buffer; stop collecting it.
            self.collect_fpm = false;
        }
        Ok(())
    }

    /// Fire every `PlannerTick` scheduled for the current timestamp: gather the
    /// drained metrics, call the planner, apply its scaling decision, and re-arm.
    /// Agg routes all FPM through `decode_fpm` and ignores the prefill target.
    fn apply_planner_ticks(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while pop_ready_planner_tick(&mut self.events, self.now_ms) {
            if self.is_workload_done() {
                continue;
            }
            let active_decode_ids = self.engine.active_group_ids();
            self.fpm_buffer
                .emit_idle_due(&active_decode_ids, self.dp_size, self.now_ms);
            let metrics = PlannerTickMetrics {
                now_ms: self.now_ms,
                prefill_fpm: Vec::new(),
                decode_fpm: self.fpm_buffer.take(),
                traffic: self.traffic.drain(self.now_ms),
                active_prefill_ids: Vec::new(),
                active_decode_ids,
                total_prefill: 0,
                total_decode: self.total_worker_count(),
            };
            let mut hook = self
                .planner_hook
                .take()
                .expect("planner tick fired without a hook");
            let decision = hook.on_tick(metrics);
            self.planner_hook = Some(hook);
            let decision = decision?;

            if decision.target_decode.is_some() {
                let target = decision
                    .target_decode
                    .unwrap_or_else(|| self.total_worker_count());
                self.apply_scaling(target)?;
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
                push_planner_tick(&mut self.events, &mut self.next_event_seq, next_ms);
            } else {
                self.collect_fpm = false;
            }
            changed = true;
        }
        Ok(changed)
    }

    // ------------------------------------------------------------------
    // Planner integration: scaling + worker-count accessors used by the
    // in-loop `PlannerTick` handler (apply_planner_ticks).
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

    /// Number of active (non-pending-removal) workers.
    #[cfg(test)]
    pub(in crate::replay) fn active_worker_count(&self) -> usize {
        self.engine.active_group_ids().len()
    }

    /// Total worker count including pending-removal.
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
    pub(in crate::replay) fn apply_scaling(&mut self, target_workers: usize) -> anyhow::Result<()> {
        let (added, newly_marked, removed) = self.engine.apply_target_count(target_workers);
        let engine = &self.engine;
        self.next_dp_rank_by_worker
            .retain(|worker_id, _| engine.rank_id(*worker_id, 0).is_some());
        #[cfg(test)]
        if !added.is_empty() {
            self.worker_active_requests
                .resize(self.engine.rank_id_capacity(), Vec::new());
        }
        let startup_delay_ms = self.engine.startup_time_ms();

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
                    if let Some(router) = self.router.as_mut() {
                        router.add_worker(id)?;
                    }
                }
            }
        }

        let admissions = if let Some(router) = self.router.as_mut() {
            for id in newly_marked {
                router.remove_worker(id)?;
            }
            for id in removed {
                router.finalize_worker_removal(id)?;
            }
            let admissions = router.on_topology_changed(self.now_ms)?.admissions;
            self.record_router_pending();
            admissions
        } else {
            Vec::new()
        };
        self.dispatch_router_admissions(admissions)?;
        self.record_in_flight_peak();
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
    fn advance_to(&mut self, until_ms: f64) -> anyhow::Result<bool> {
        self.drain_current_timestamp()?;

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

    /// Finalize the replay and return the simulation report directly.
    #[cfg(test)]
    fn finalize_report(self) -> crate::replay::TraceSimulationReport {
        self.progress.finish();
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
        self.drain_current_timestamp()?;
        // With a planner attached, seed the recurring heartbeat; ticks then fire as
        // events inside drain_current_timestamp (see apply_planner_ticks).
        self.seed_first_planner_tick()?;

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
            self.advance_now_ms(next_timestamp_ms);
            self.drain_current_timestamp()?;
        }

        self.progress.finish();
        Ok((self.collector, self.stats))
    }

    #[cfg(test)]
    /// Test helper: advance exactly one logical timestamp worth of work.
    fn advance_one_timestamp(&mut self) -> anyhow::Result<bool> {
        if self.is_done() {
            return Ok(false);
        }

        if !self.stepped {
            self.stepped = true;
            self.drain_current_timestamp()?;
            return Ok(true);
        }

        let Some(next_timestamp_ms) = self.next_timestamp() else {
            bail!(
                "offline replay reached a dead end with {} in-flight requests remaining",
                self.cluster_in_flight()
            );
        };

        self.now_ms = next_timestamp_ms;
        self.drain_current_timestamp()?;
        Ok(true)
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
            router: self
                .router
                .as_ref()
                .map(|router| router.debug_snapshot(self.now_ms)),
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
    use crate::common::protocols::{EngineType, SglangArgs};
    use crate::loadgen::{AgenticTrace, AgenticTurnTrace, SessionTrace, Trace, TurnTrace};
    use crate::replay::{TraceRequestStatsSnapshot, normalize_trace_requests};
    use dynamo_kv_router::config::{KvRouterConfig, RouterQueuePolicy};
    use rstest::rstest;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct CaptureOnceHook {
        at_ms: f64,
        captured: Rc<RefCell<Option<PlannerTickMetrics>>>,
    }

    impl PlannerHook for CaptureOnceHook {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(self.at_ms)
        }

        fn on_tick(
            &mut self,
            metrics: PlannerTickMetrics,
        ) -> anyhow::Result<super::super::planner_hook::PlannerTickDecision> {
            *self.captured.borrow_mut() = Some(metrics);
            Ok(super::super::planner_hook::PlannerTickDecision::default())
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
                    prefix_reset: true,
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
                    prefix_reset: true,
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

    fn queueing_router_config(policy: RouterQueuePolicy) -> KvRouterConfig {
        KvRouterConfig {
            router_queue_threshold: Some(0.5),
            router_queue_policy: policy,
            ..KvRouterConfig::default()
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

    fn planner_router_config() -> KvRouterConfig {
        KvRouterConfig {
            router_queue_threshold: Some(0.5),
            ..KvRouterConfig::default()
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
    fn planner_tick_emits_idle_fpm_after_simulated_second() {
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
        let hook = CaptureOnceHook {
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
        .with_planner_hook(Box::new(hook))
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
        // resulting 17-token prompt is rejected, so it contributes no output
        // before turn 2 adds its one-token input delta.
        let (collector, stats) = run_trace_workload_multi_collect_with_stats(
            &trtllm_reject_args(),
            trace,
            1,
            ReplayRouterMode::RoundRobin,
            true,
        );
        let input_lengths = stats
            .dispatch_order
            .iter()
            .map(|uuid| collector.snapshot(*uuid).unwrap().input_length)
            .collect::<Vec<_>>();
        let report = collector.finish();

        assert_eq!(input_lengths, vec![4, 17, 18]);
        assert_eq!(report.request_counts.num_requests, 3);
        assert_eq!(report.request_counts.completed_requests, 1);
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
    fn test_multi_worker_trace_kv_router_debug_snapshot_tracks_queue_and_cached_dispatch() {
        let policy = RouterQueuePolicy::Fcfs;
        let args = queueing_router_args(policy);
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
        assert!(initial_router.indexer.total_cached_blocks > 0);

        assert!(runtime.advance_one_timestamp().unwrap());
        let queued = runtime.debug_snapshot();
        let queued_router = queued.router.as_ref().unwrap();

        assert_eq!(queued.now_ms, 0.1);
        assert_eq!(queued.router_pending_request_ids, vec![Uuid::from_u128(33)]);
        assert_eq!(queued_router.pending.len(), 1);
        assert_eq!(queued_router.pending[0].uuid, Uuid::from_u128(33));

        let cached_workers = queued_router.pending[0]
            .overlap_blocks_by_worker
            .iter()
            .filter(|(_, overlap)| *overlap > 0)
            .map(|(worker_idx, _)| *worker_idx)
            .collect::<Vec<_>>();
        assert_eq!(cached_workers.len(), 1);
        let cached_worker = cached_workers[0];

        while !runtime
            .stats
            .assigned_worker_by_uuid
            .contains_key(&Uuid::from_u128(33))
        {
            assert!(runtime.advance_one_timestamp().unwrap());
        }

        let dispatched = runtime.debug_snapshot();
        assert!(dispatched.router_pending_request_ids.is_empty());
        assert_eq!(
            runtime.stats.assigned_worker_by_uuid[&Uuid::from_u128(33)],
            cached_worker
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
        // Python `test_planner_bridge_drains_mtp_accept_length` that drove the
        // now-removed bridge stepping API directly.)
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
        assert_eq!(rt.next_dp_rank_by_worker.len(), 2);
        while rt.advance_one_timestamp().unwrap() {}

        rt.apply_scaling(1).unwrap();

        assert_eq!(rt.next_dp_rank_by_worker.len(), 1);
        assert!(!rt.next_dp_rank_by_worker.contains_key(&1));
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
        assert!(
            (report.throughput.decode_worker_seconds - 1.0).abs() < 1e-6,
            "only the remaining worker should accrue during the post-scale interval, got {}",
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
