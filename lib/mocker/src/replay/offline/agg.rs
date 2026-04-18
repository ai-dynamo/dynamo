// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use super::components::OfflineRouterSnapshot;
pub(super) use super::components::ReplayMode;
use super::events::{SimulationEvent, SimulationWorkerStage};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    next_timestamp as choose_next_timestamp, pop_ready_worker_completion, pop_ready_worker_ready,
    push_worker_completion, push_worker_ready,
};
#[cfg(test)]
use super::state::AggRequestPhase;
#[cfg(test)]
use super::state::OfflineWorkerSnapshot;
use super::{
    components::{
        AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, OfflineReplayRouter,
        ScheduledWorkerCompletion, TrafficAccumulator, TrafficStats, WorkerAdmission,
    },
    state::AggRequestState,
};
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode, TraceCollector};
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
pub(super) struct AggRuntimeStats {
    dispatch_history: Vec<usize>,
    dispatch_order: Vec<Uuid>,
    assigned_worker_by_uuid: HashMap<Uuid, usize>,
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
pub(super) struct AggRuntimeStats;

pub(in crate::replay) struct AggRuntime {
    now_ms: f64,
    next_worker_idx: usize,
    next_event_seq: u64,
    admission: AdmissionQueue,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: EngineComponent,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    router: Option<OfflineReplayRouter>,
    progress: ReplayProgress,
    stats: AggRuntimeStats,
    /// Forward pass metrics accumulated between planner ticks.
    fpm_buffer: Vec<(usize, ForwardPassSnapshot)>,
    /// Traffic statistics accumulated between planner ticks.
    traffic: TrafficAccumulator,
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
    ) -> anyhow::Result<Self> {
        let args = args.clone().normalized()?;
        let progress = ReplayProgress::new(admission.total_requests(), "offline replay");
        let router = match router_mode {
            ReplayRouterMode::RoundRobin => None,
            ReplayRouterMode::KvRouter => Some(OfflineReplayRouter::new(
                &args,
                router_config,
                prefill_load_estimator,
                num_workers,
            )?),
        };
        let capture_kv_events = router.is_some();
        let mut engine = EngineComponent::new(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            (0..num_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        args.clone(),
                        capture_kv_events,
                    )
                })
                .collect(),
        );
        engine.set_scaling_args(args, capture_kv_events);

        Ok(Self {
            now_ms: 0.0,
            next_worker_idx: 0,
            next_event_seq: 0,
            admission,
            requests: FxHashMap::default(),
            engine,
            collector: TraceCollector::default(),
            events: BinaryHeap::new(),
            router,
            progress,
            #[cfg(test)]
            stats: AggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: AggRuntimeStats,
            fpm_buffer: Vec::new(),
            traffic: TrafficAccumulator::new(),
            #[cfg(test)]
            worker_active_requests: vec![Vec::new(); num_workers],
            #[cfg(test)]
            stepped: false,
        })
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
        let active = self.engine.active_worker_ids();
        debug_assert!(!active.is_empty(), "no active workers for round-robin");
        let idx = self.next_worker_idx % active.len();
        self.next_worker_idx = idx + 1;
        active[idx]
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

    /// Deliver a request to a worker and update the runtime's bookkeeping for that assignment.
    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        worker_idx: usize,
    ) -> anyhow::Result<()> {
        self.engine.dispatch(worker_idx, request)?;
        self.record_dispatch(uuid, worker_idx);
        #[cfg(test)]
        self.worker_active_requests[worker_idx].push(uuid);
        Ok(())
    }

    /// Materialize router admissions into concrete worker dispatches.
    fn dispatch_router_admissions(
        &mut self,
        admissions: Vec<WorkerAdmission>,
    ) -> anyhow::Result<()> {
        for WorkerAdmission { uuid, worker_idx } in admissions {
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
        let queued_request = request.clone();
        self.requests
            .insert(uuid, AggRequestState::new_queued(request));
        let admissions = {
            let router = self.router.as_mut().expect("router presence checked above");
            router
                .on_request_arrival(&queued_request, replay_hashes, self.now_ms)?
                .admissions
        };
        self.record_router_pending();
        self.dispatch_router_admissions(admissions)?;
        self.record_in_flight_peak();
        Ok(uuid)
    }

    /// Return true once no events, workers, router queues, or admissions remain.
    fn is_done(&self) -> bool {
        self.events.is_empty()
            && self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.engine.is_drained()
    }

    /// Return true once the request workload is complete, even if `WorkerReady`
    /// events remain in the queue. Used by `advance_to` so the planner adapter
    /// can terminate when there is no more work — lingering startup events for
    /// workers that will never receive requests should not block completion.
    fn is_workload_done(&self) -> bool {
        self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.engine.is_drained()
            && self.only_worker_ready_events_remain()
    }

    /// True if the event heap is empty or contains only `WorkerReady` events.
    fn only_worker_ready_events_remain(&self) -> bool {
        use super::events::SimulationEventKind;
        self.events
            .iter()
            .all(|e| matches!(e.kind, SimulationEventKind::WorkerReady { .. }))
    }

    /// Pick the next logical timestamp from either arrivals or scheduled worker completions.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        choose_next_timestamp(
            self.admission.next_ready_time_ms(self.cluster_in_flight()),
            next_event_ms,
        )
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

    /// Consume one output signal, updating router state, collector state, and completion counts.
    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        let mut admissions = Vec::new();
        if signal.completed {
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
            let latencies = self.collector.request_latencies(signal.uuid);
            self.traffic.on_request(
                removed_state.input_tokens,
                removed_state.output_tokens,
                latencies,
            );
            self.admission
                .on_request_completed(signal.uuid, self.now_ms)?;
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
    ) -> anyhow::Result<()> {
        self.apply_router_events(kv_events)?;
        for signal in output_signals {
            self.process_output_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some(payload) = pop_ready_worker_completion(&mut self.events, self.now_ms) {
            debug_assert_eq!(payload.stage, SimulationWorkerStage::Aggregated);
            let payload = self.engine.on_scheduled_completion(payload)?;
            self.process_completed_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
                payload.kv_events,
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
            self.assign_request(ready.request, ready.arrival_time_ms, ready.replay_hashes)?;
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
        self.fpm_buffer.extend(effects.fpm_snapshots);
        self.apply_router_events(effects.pass_start_kv_events)?;
        for payload in effects.immediate_completions {
            let payload = self.engine.on_scheduled_completion(payload)?;
            self.process_completed_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
                payload.kv_events,
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
            let mut changed = self.apply_worker_completions()?;
            changed |= self.apply_worker_ready_events()?;
            changed |= self.release_ready_arrivals()?;
            changed |= self.drive_ready_workers()?;

            if !changed {
                break;
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Planner integration: step-based execution
    // ------------------------------------------------------------------

    /// Advance the simulation up to `until_ms` simulated time, then pause.
    /// Returns `true` if the request workload is done — pending `WorkerReady`
    /// events do not block completion since there is no work for those workers.
    pub(in crate::replay) fn advance_to(&mut self, until_ms: f64) -> anyhow::Result<bool> {
        self.drain_current_timestamp()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };

            if next_timestamp_ms > until_ms {
                break;
            }

            self.now_ms = next_timestamp_ms;
            self.drain_current_timestamp()?;
        }

        Ok(self.is_workload_done())
    }

    /// Current simulated time in milliseconds.
    pub(in crate::replay) fn now_ms(&self) -> f64 {
        self.now_ms
    }

    /// Number of active (non-pending-removal) workers.
    pub(in crate::replay) fn active_worker_count(&self) -> usize {
        self.engine.active_worker_ids().len()
    }

    /// Total worker count including pending-removal.
    pub(in crate::replay) fn total_worker_count(&self) -> usize {
        self.engine.worker_count()
    }

    /// Drain accumulated FPM snapshots since the last drain.
    pub(in crate::replay) fn drain_fpm(&mut self) -> Vec<(usize, ForwardPassSnapshot)> {
        std::mem::take(&mut self.fpm_buffer)
    }

    /// Drain accumulated traffic stats since the last drain.
    pub(in crate::replay) fn drain_traffic(&mut self) -> TrafficStats {
        self.traffic.drain(self.now_ms)
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
        let (added, newly_marked) = self.engine.apply_target_count(target_workers);
        #[cfg(test)]
        if let Some(new_len) = added.iter().max().map(|id| id + 1) {
            self.worker_active_requests.resize(new_len, Vec::new());
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

    /// Finalize the replay: finish progress bar, return collector and stats.
    pub(in crate::replay::offline) fn finalize(self) -> (TraceCollector, AggRuntimeStats) {
        self.progress.finish();
        (self.collector, self.stats)
    }

    /// Finalize the replay and return the simulation report directly.
    pub(in crate::replay) fn finalize_report(self) -> crate::replay::TraceSimulationReport {
        let (collector, _stats) = self.finalize();
        collector.finish()
    }

    /// Run the aggregated offline replay until all arrivals and worker work are exhausted.
    pub(in crate::replay::offline) fn run(
        mut self,
    ) -> anyhow::Result<(TraceCollector, AggRuntimeStats)> {
        self.drain_current_timestamp()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };

            self.now_ms = next_timestamp_ms;
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
        run_concurrency_multi_collect_with_stats, run_concurrency_single_collect,
        run_concurrency_workload_multi_collect_with_stats, run_trace_multi_collect_with_stats,
        run_trace_single_collect, run_trace_workload_multi_collect_with_stats,
    };
    use super::*;
    use crate::common::protocols::{EngineType, SglangArgs};
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};
    use crate::replay::normalize_trace_requests;
    use dynamo_kv_router::config::{KvRouterConfig, RouterQueuePolicy};

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
                        },
                        TurnTrace {
                            input_length: 192,
                            max_output_tokens: 2,
                            hash_ids: vec![21, 22, 23],
                            delay_after_previous_ms: 10.0,
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
    fn test_concurrency_workload_delayed_follow_up_does_not_bypass_other_ready_sessions() {
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
        assert_eq!(dispatch_input_lengths, vec![64, 128, 192]);
    }

    #[test]
    fn test_trace_workload_kv_router_precomputed_hashes_match_request_fallback() {
        let args = fast_router_args();
        let requests = vec![
            DirectRequest {
                tokens: [vec![11; 64], vec![21; 32]].concat(),
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(111)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
            },
            DirectRequest {
                tokens: [vec![11; 64], vec![22; 32]].concat(),
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(222)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
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
    }

    #[test]
    fn test_multi_worker_trace_kv_router_debug_snapshot_tracks_queue_and_cached_dispatch() {
        let args = queueing_router_args(RouterQueuePolicy::Fcfs);
        let mut runtime = AggRuntime::new(
            &args,
            None,
            None,
            normalize_trace_requests(
                vec![
                    DirectRequest {
                        tokens: vec![11; 64],
                        max_output_tokens: 8,
                        uuid: Some(Uuid::from_u128(11)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                    },
                    DirectRequest {
                        tokens: vec![22; 64],
                        max_output_tokens: 8,
                        uuid: Some(Uuid::from_u128(22)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                    },
                    DirectRequest {
                        tokens: vec![11; 64],
                        max_output_tokens: 2,
                        uuid: Some(Uuid::from_u128(33)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.1),
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
                        uuid: Some(Uuid::from_u128(1)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
                    },
                    DirectRequest {
                        tokens: vec![22; 64],
                        max_output_tokens: 8,
                        uuid: Some(Uuid::from_u128(2)),
                        dp_rank: 0,
                        arrival_timestamp_ms: Some(0.0),
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
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                },
                DirectRequest {
                    tokens: vec![7, 7, 7, 7, 8, 8, 8, 8],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
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
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![2; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![3; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![4; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(4)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![5; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(5)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
            ],
            4,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 1, 2, 3, 0]);
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
                    uuid: Some(Uuid::from_u128(901)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(902)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(5.0),
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
                    uuid: Some(Uuid::from_u128(911)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![7; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(912)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(500.0),
                },
            ],
            2,
            ReplayRouterMode::KvRouter,
        );

        let report = collector.finish();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(stats.dispatch_history.len(), 2);
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
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(900.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(1000.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
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
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![11; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(2.0),
                },
                DirectRequest {
                    tokens: vec![22; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(2.0),
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
                    uuid: Some(Uuid::from_u128(9)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![8; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(8)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
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
        let args = queueing_router_args(RouterQueuePolicy::Fcfs);
        let (collector, stats) = run_trace_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 8,
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 8,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.0),
                },
                DirectRequest {
                    tokens: vec![3; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(0.1),
                },
            ],
            2,
            ReplayRouterMode::KvRouter,
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
                uuid: Some(Uuid::from_u128(10)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
            },
            DirectRequest {
                tokens: vec![20; 64],
                max_output_tokens: 8,
                uuid: Some(Uuid::from_u128(20)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
            },
            DirectRequest {
                tokens: vec![30; 64],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(30)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.1),
            },
            DirectRequest {
                tokens: vec![40; 64],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(40)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.2),
            },
        ];

        let (_, fcfs_stats) = run_trace_multi_collect_with_stats(
            &queueing_router_args(RouterQueuePolicy::Fcfs),
            requests.clone(),
            2,
            ReplayRouterMode::KvRouter,
        );
        let (_, lcfs_stats) = run_trace_multi_collect_with_stats(
            &queueing_router_args(RouterQueuePolicy::Lcfs),
            requests,
            2,
            ReplayRouterMode::KvRouter,
        );

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
                uuid: Some(Uuid::from_u128(10)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
            },
            DirectRequest {
                tokens: vec![20; 128],
                max_output_tokens: 8,
                uuid: Some(Uuid::from_u128(20)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
            },
            DirectRequest {
                tokens: vec![30; 64],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(30)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.1),
            },
            DirectRequest {
                tokens: vec![40; 64],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(40)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.2),
            },
        ];

        let (fcfs_collector, fcfs_stats) = run_trace_multi_collect_with_stats(
            &queueing_router_args(RouterQueuePolicy::Fcfs),
            requests.clone(),
            2,
            ReplayRouterMode::KvRouter,
        );
        let (lcfs_collector, lcfs_stats) = run_trace_multi_collect_with_stats(
            &queueing_router_args(RouterQueuePolicy::Lcfs),
            requests,
            2,
            ReplayRouterMode::KvRouter,
        );

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
        let args = queueing_router_args(RouterQueuePolicy::Fcfs);
        let (_, stats) = run_concurrency_multi_collect_with_stats(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(1)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(2)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
                DirectRequest {
                    tokens: vec![1; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(3)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(4)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
            ],
            3,
            2,
            ReplayRouterMode::KvRouter,
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
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
                DirectRequest {
                    tokens: vec![2; 64],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
                },
                DirectRequest {
                    tokens: vec![3; 64],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: None,
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

    #[test]
    fn test_multi_worker_trace_single_worker_round_robin_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(101.0),
            },
            DirectRequest {
                tokens: vec![9, 9, 9, 9, 8, 8, 8, 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
            },
        ];

        let single = run_trace_single_collect(args.clone(), requests.clone(), 1.0);
        let (multi, stats) =
            run_trace_multi_collect_with_stats(&args, requests, 1, ReplayRouterMode::RoundRobin);

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        for uuid in [11_u128, 22, 33] {
            assert_eq!(
                multi.snapshot(Uuid::from_u128(uuid)),
                single.snapshot(Uuid::from_u128(uuid))
            );
        }
        assert_eq!(multi.finish().request_counts.completed_requests, 3);
        assert_eq!(single.finish().request_counts.completed_requests, 3);
    }

    #[test]
    fn test_multi_worker_trace_single_worker_kv_router_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(101.0),
            },
            DirectRequest {
                tokens: vec![9, 9, 9, 9, 8, 8, 8, 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
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

    #[test]
    fn test_multi_worker_concurrency_single_worker_round_robin_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(900.0),
            },
            DirectRequest {
                tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                max_output_tokens: 4,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(1000.0),
            },
            DirectRequest {
                tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
        ];

        let single = run_concurrency_single_collect(args.clone(), requests.clone(), 2);
        let (multi, stats) = run_concurrency_multi_collect_with_stats(
            &args,
            requests,
            2,
            1,
            ReplayRouterMode::RoundRobin,
        );

        assert_eq!(stats.dispatch_history, vec![0, 0, 0]);
        for uuid in [11_u128, 22, 33] {
            assert_eq!(
                multi.snapshot(Uuid::from_u128(uuid)),
                single.snapshot(Uuid::from_u128(uuid))
            );
        }
    }

    #[test]
    fn test_multi_worker_concurrency_single_worker_kv_router_matches_single_runtime() {
        let args = replay_args(true, true);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(900.0),
            },
            DirectRequest {
                tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                max_output_tokens: 4,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(1000.0),
            },
            DirectRequest {
                tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
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
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(i as f64 * arrival_interval_ms),
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
}
